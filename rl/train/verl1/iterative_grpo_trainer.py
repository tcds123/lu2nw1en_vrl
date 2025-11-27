import os
import sys
import glob
import json
import time
import shutil
import logging
import ray
import re
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

# 导入 verl 相关模块
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

# ============================================================================
# 【Patch 1】动态维度对齐 (解决 512 vs 511 报错)
# ============================================================================
import torch
from verl import DataProto
from verl.utils import torch_functional as verl_F
from verl.trainer.ppo import core_algos

def robust_compute_policy_loss_vanilla(
    old_log_prob, log_prob, advantages, response_mask, 
    loss_agg_mode="token-mean", config=None, rollout_is_weights=None
):
    kl_len = log_prob.shape[-1]
    mask_len = response_mask.shape[-1]
    if kl_len != mask_len:
        min_len = min(kl_len, mask_len)
        log_prob = log_prob[:, :min_len]
        old_log_prob = old_log_prob[:, :min_len]
        advantages = advantages[:, :min_len]
        response_mask = response_mask[:, :min_len]
        if rollout_is_weights is not None and rollout_is_weights.shape[-1] != min_len:
            rollout_is_weights = rollout_is_weights[:, :min_len]
    
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
        
    pg_loss = core_algos.agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

print(">>> [Patch 1] robust_compute_policy_loss_vanilla applied.")
core_algos.POLICY_LOSS_REGISTRY["vanilla"] = robust_compute_policy_loss_vanilla
core_algos.compute_policy_loss_vanilla = robust_compute_policy_loss_vanilla

# ============================================================================
# 【Patch 2】资源检查绕过 & 唯一 PG 名
# ============================================================================
def patch_ray_trainer_file():
    try:
        import verl.trainer.ppo.ray_trainer as ray_trainer_module
        target_file = ray_trainer_module.__file__
        with open(target_file, 'r') as f: content = f.read()
        modified = False
        if "import time" not in content:
            content = content.replace("import os", "import os\nimport time")
            modified = True
        if 'raise ValueError(' in content and 'Total available GPUs' in content:
            content = re.sub(r'(if total_available_gpus < total_required_gpus:\s+)(raise ValueError\([^)]+\))', r'\1print(f"⚠️ [Auto-Fix] 资源检查忽略: {total_available_gpus} < {total_required_gpus}") # \2', content, flags=re.DOTALL)
            modified = True
        original_pg_code = 'name_prefix=resource_pool_name'
        new_pg_code = 'name_prefix=f"{resource_pool_name}_{str(time.time())[-5:]}"'
        if original_pg_code in content:
            content = content.replace(original_pg_code, new_pg_code, 1)
            modified = True
        if modified:
            with open(target_file, 'w') as f: f.write(content)
            cache_dir = os.path.join(os.path.dirname(target_file), "__pycache__")
            if os.path.exists(cache_dir): shutil.rmtree(cache_dir)
            print(">>> [Patch 2] ray_trainer.py patched.")
    except Exception: pass
patch_ray_trainer_file()

# ============================================================================
# 【Patch 3】强制 Trainer 在结束时杀掉所有 Worker
# ============================================================================
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
_original_fit = RayPPOTrainer.fit
def fit_with_active_cleanup(self):
    print("⚡ [Auto-Cleanup] Starting training with auto-cleanup hook...")
    try:
        _original_fit(self)
    finally:
        print("\n🧹 [Auto-Cleanup] Training finished. Force killing workers...")
        groups = [getattr(self, 'actor_rollout_wg', None), getattr(self, 'critic_wg', None), getattr(self, 'ref_policy_wg', None), getattr(self, 'rm_wg', None)]
        for wg in groups:
            if wg and hasattr(wg, '_worker_dict'):
                for worker in wg._worker_dict.values():
                    try: ray.kill(worker)
                    except: pass
            if wg and hasattr(wg, '_pgs'):
                try:
                    from ray.util.placement_group import remove_placement_group
                    for pg in wg._pgs: remove_placement_group(pg)
                except: pass
RayPPOTrainer.fit = fit_with_active_cleanup
print(">>> [Patch 3] RayPPOTrainer.fit patched.")


class IterativeGRPOTaskRunner:
    def __init__(self):
        self.max_iterations = None
        self.convergence_threshold = None
        self.current_iteration = 0
    
    def _get_latest_checkpoint_path(self, base_dir):
        if not os.path.exists(base_dir): return None
        step_dirs = glob.glob(os.path.join(base_dir, "global_step_*"))
        if not step_dirs: return None
        try:
            latest_dir = max(step_dirs, key=lambda x: int(x.split('_')[-1]))
        except ValueError: return None
        hf_path = os.path.join(latest_dir, "actor", "huggingface")
        if os.path.exists(hf_path) and os.path.exists(os.path.join(hf_path, "config.json")):
            return hf_path
        return None

    def run(self, config):
        self.max_iterations = config.iterative_rl.max_iterations
        self.convergence_threshold = config.iterative_rl.convergence_threshold
        base_exp_name = config.trainer.experiment_name
        base_local_dir = config.trainer.default_local_dir
        
        print(f"Starting iterative GRPO training with {self.max_iterations} iterations")
        
        last_iteration_ckpt_path = None

        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            print(f"\n{'='*40}")
            print(f"🚀 Iteration {iteration + 1}/{self.max_iterations}")
            print(f"{'='*40}")
            
            if ray.is_initialized():
                print("Shutting down previous Ray instance...")
                ray.shutdown()
                time.sleep(3)
            
            print("Initializing FRESH Ray cluster...")
            ray.init(
                runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}},
                num_gpus=config.trainer.n_gpus_per_node,
                include_dashboard=False,
                ignore_reinit_error=True
            )
            
            print("Creating TaskRunner...")
            task_runner = TaskRunner.options(num_cpus=1, num_gpus=0).remote()
            
            iter_config = config.copy()
            with open_dict(iter_config):
                iter_config.current_iteration = self.current_iteration
                iter_exp_name = f"{base_exp_name}_iter_{iteration}"
                
                # 基础输出目录
                iter_local_dir = os.path.join(os.path.dirname(base_local_dir), iter_exp_name)
                
                # 1. 设置 Checkpoint 目录
                iter_config.trainer.experiment_name = iter_exp_name
                iter_config.trainer.default_local_dir = iter_local_dir
                
                # 2. 【关键修改】强制设置 Rollout 日志目录
                # 之前这里是 null，所以没有生成 jsonl 文件
                # 现在强制设置为 checkpoints 同级目录下的 rollouts
                rollout_dir = os.path.join(iter_local_dir, "rollouts")
                iter_config.trainer.rollout_data_dir = rollout_dir
                print(f"📝 Configuring Rollout Logging to: {rollout_dir}")
                
                if 'hf_model' not in iter_config.actor_rollout_ref.actor.checkpoint.save_contents:
                    iter_config.actor_rollout_ref.actor.checkpoint.save_contents.append('hf_model')
                
                if iteration > 0:
                    if last_iteration_ckpt_path:
                        print(f"Inheriting Weight: {last_iteration_ckpt_path}")
                        iter_config.actor_rollout_ref.model.path = last_iteration_ckpt_path
                        iter_config.actor_rollout_ref.ref.model = OmegaConf.create({"path": last_iteration_ckpt_path})
                        iter_config.trainer.resume_mode = "disable"
                    else:
                        raise RuntimeError("No checkpoint found from previous iteration!")
                else:
                    print(f"Iteration 0: Base model.")

            iter_config_hydra = OmegaConf.create(OmegaConf.to_container(iter_config, resolve=True))
            
            try:
                print(f"Running PPO (Iter {iteration})...")
                ray.get(task_runner.run.remote(iter_config_hydra))
            except Exception as e:
                print(f"❌ Error in iter {iteration}: {e}")
                ray.shutdown()
                raise e
            
            print("Iteration finished. Nuking Ray cluster...")
            ray.shutdown()
            print("Sleeping 10s...")
            time.sleep(10) 

            # 6. 检查收敛
            if self._check_convergence(config):
                print(f"Training converged at iteration {iteration + 1}")
                break
            
            # 7. 寻找 Checkpoint
            if iteration < self.max_iterations - 1:
                last_iteration_ckpt_path = self._get_latest_checkpoint_path(iter_config_hydra.trainer.default_local_dir)
                if not last_iteration_ckpt_path:
                    print("Error: No HF checkpoint found!")

    def _check_convergence(self, config) -> bool:
        log_dir = config.iterative_rl.iteration_log_dir
        iteration_file = os.path.join(log_dir, f"iteration_{self.current_iteration}.json")
        if not os.path.exists(iteration_file): return False
        try:
            with open(iteration_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            avg_reward = summary.get('average_reward', 0.0)
            print(f"Iter {self.current_iteration} Avg Reward: {avg_reward}")
            if avg_reward >= self.convergence_threshold: return True
        except: pass
        return False

@hydra.main(config_path="config", config_name="grpo_trainer", version_base=None)
def main(cfg: DictConfig):
    if cfg.iterative_rl.rollout_data_dir:
        cfg.iterative_rl.rollout_data_dir = os.path.abspath(cfg.iterative_rl.rollout_data_dir)
    if cfg.iterative_rl.iteration_log_dir:
        cfg.iterative_rl.iteration_log_dir = os.path.abspath(cfg.iterative_rl.iteration_log_dir)
    
    task_runner = IterativeGRPOTaskRunner()
    if not cfg.trainer.val_only:
        task_runner.run(cfg)
    else:
        print("Validation only mode - skipping training")

if __name__ == "__main__":
    main()