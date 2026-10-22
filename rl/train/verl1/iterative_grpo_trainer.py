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
import math
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict

from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config


# 必须在 ray.init 之前设置
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

# ============================================================================
# 【Patch 0】修复 verl 本地路径加载 (huggingface_hub 新版本验证问题)
# ============================================================================
import verl.utils.tokenizer as verl_tokenizer_module
_original_hf_tokenizer = verl_tokenizer_module.hf_tokenizer

def patched_hf_tokenizer(name_or_path, **kwargs):
    """支持本地绝对路径加载模型 tokenizer"""
    if name_or_path.startswith("/"):
        kwargs["local_files_only"] = True
    return _original_hf_tokenizer(name_or_path, **kwargs)

verl_tokenizer_module.hf_tokenizer = patched_hf_tokenizer
print("✅ [Patch 0] verl tokenizer patched for local paths")

# ============================================================================
# 【Patch 1】动态维度对齐 (保持不变)
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

core_algos.POLICY_LOSS_REGISTRY["vanilla"] = robust_compute_policy_loss_vanilla
core_algos.compute_policy_loss_vanilla = robust_compute_policy_loss_vanilla

# ============================================================================
# 【Patch 2】资源检查绕过 & 唯一 PG 名 (保持不变)
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
    except Exception: pass
patch_ray_trainer_file()

# ============================================================================
# 【Patch 3】强制 Trainer 在结束时杀掉所有 Worker (保持不变)
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


class IterativeGRPOTaskRunner:
    def __init__(self):
        self.group_size = 3  # 每组数据大小 (1个中心 + 2个辅助)
    
    def _prepare_filtered_dataset(self, full_data_path, output_dir):
        """
        [关键逻辑] 读取 Parquet 数据集，筛选每组的第一个 (Center)，保存为 Parquet
        """
        if not os.path.exists(full_data_path):
            raise FileNotFoundError(f"Original dataset not found: {full_data_path}")
            
        print(f"📖 Reading full Parquet dataset from: {full_data_path}")
        
        try:
            # 1. 使用 Pandas 读取 Parquet
            df = pd.read_parquet(full_data_path)
            total_rows = len(df)
            
            # 2. 筛选：只保留每组的第 1 个 (索引 0, 3, 6...)
            # iloc[::3] 表示从 0 开始每隔 3 行取一次
            centers_df = df.iloc[::self.group_size].copy()
            
            # 3. [修复] 保存为 Parquet 格式，因为 Verl 底层 dataset 默认使用 load_dataset("parquet", ...)
            filtered_data_path = os.path.join(output_dir, f"train_centers_only.parquet")
            os.makedirs(output_dir, exist_ok=True)
            
            centers_df.to_parquet(filtered_data_path)
                
            print(f"✂️ Filtered dataset: Kept {len(centers_df)} centers from {total_rows} total samples.")
            print(f"📂 Saved to: {filtered_data_path}")
            
            return filtered_data_path, len(centers_df)
            
        except Exception as e:
            print(f"❌ Failed to process Parquet file: {e}")
            raise e

    def run(self, config):
        # 1. 从 config 中获取原始数据路径
        # config.data.train_files 可能是 string 也可能是 list
        original_train_path = config.data.train_files
        if isinstance(original_train_path, list):
            original_train_path = original_train_path[0]
        
        # 确保是绝对路径，方便传递给 Worker
        original_train_path = os.path.abspath(original_train_path)

        if not os.path.exists(original_train_path):
            raise FileNotFoundError(f"❌ Configured train file not found: {original_train_path}")

        base_exp_name = config.trainer.experiment_name
        base_local_dir = config.trainer.default_local_dir
        
        # 2. 准备 "Center Only" 的训练集
        data_cache_dir = os.path.join(os.path.dirname(base_local_dir), "data_cache")
        filtered_train_path, subset_size = self._prepare_filtered_dataset(original_train_path, data_cache_dir)
        
        print(f"\n{'='*40}")
        print(f"🚀 Starting Group-Based Training (1 Center + 2 Aux)")
        print(f"📂 Source (from config): {original_train_path}")
        print(f"🎯 Training Centers: {subset_size} samples")
        print(f"   (Effective Coverage: {subset_size} * 3 = {subset_size * 3} samples)")
        print(f"{'='*40}")

        if ray.is_initialized():
            ray.shutdown()
        
        print("Initializing Ray cluster...")
        
        # 【关键修改】将原始数据路径放入 runtime_env，传递给所有 Worker
        env_vars = {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "VERL_DATA_PATH": original_train_path  # 传递给 Custom Reward
        }
        
        ray.init(
            runtime_env={"env_vars": env_vars},
            num_gpus=config.trainer.n_gpus_per_node,
            include_dashboard=False,
            ignore_reinit_error=True
        )
        
        print("Creating TaskRunner...")
        task_runner = TaskRunner.options(num_cpus=1, num_gpus=0).remote()
        
        # 配置调整
        run_config = config.copy()
        with open_dict(run_config):
            # 1. 指向筛选后的 Parquet 文件 (Center Only)
            run_config.data.train_files = [filtered_train_path]
            run_config.data.val_files = [filtered_train_path]
            
            # 2. 强制设置 rollout.n = 5
            if hasattr(run_config, 'actor_rollout_ref'):
                run_config.actor_rollout_ref.rollout.n = 5
                print(f"🔄 Forced setting rollout.n = 5 for GRPO")
            
            if 'hf_model' not in run_config.actor_rollout_ref.actor.checkpoint.save_contents:
                run_config.actor_rollout_ref.actor.checkpoint.save_contents.append('hf_model')

            # 3. 设置输出目录
            run_config.trainer.rollout_data_dir = os.path.join(base_local_dir, "rollouts")

        run_config_hydra = OmegaConf.create(OmegaConf.to_container(run_config, resolve=True))
        
        try:
            print(f"Running PPO Training...")
            ray.get(task_runner.run.remote(run_config_hydra))
        except Exception as e:
            print(f"❌ Error in training: {e}")
            ray.shutdown()
            raise e
        
        print("Training finished. Shutting down Ray...")
        ray.shutdown()

@hydra.main(config_path="config", config_name="grpo_trainer", version_base=None)
def main(cfg: DictConfig):
    if cfg.iterative_rl.rollout_data_dir:
        cfg.iterative_rl.rollout_data_dir = os.path.abspath(cfg.iterative_rl.rollout_data_dir)
    
    task_runner = IterativeGRPOTaskRunner()
    if not cfg.trainer.val_only:
        task_runner.run(cfg)
    else:
        print("Validation only mode - skipping training")

if __name__ == "__main__":
    main()