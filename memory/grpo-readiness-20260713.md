---
name: grpo-readiness-20260713
description: GRPO 训练跑通验证 + 8×L40 优化配置 (2026-07-13) - 代码就绪, 可排卡
metadata:
  type: project
---

# GRPO 跑通验证 + 8×L40 配置 (2026-07-13)

## 跑通验证 (全部通过 ✅)

在 1×L40 开发机上对整条训练链路做了端到端冒烟测试:

| 检查项 | 结果 |
|--------|------|
| verl 0.6.0 / torch 2.8 / CUDA | ✅ |
| 模型 Qwen3-4B-Instruct-2507 (3 safetensors) | ✅ |
| 训练数据 train_testexample_bestmax_exp.parquet (58MB) | ✅ |
| `compute_custom_reward` 端到端 (B+C API, 返回 reward_tensor) | ✅ |
| Step 1 全量注入 (50 flat skills 进 trace) | ✅ |
| Step 2 record_outcome 更新 online_success_rate (50 skill EMA=-0.27) | ✅ |
| Step 4 debug_trace 字段写入 case_info | ✅ |
| `validate_config` (8 GPU, DP=8, TP=1, n=5, bs=40) | ✅ passed |

## 修复的 2 个阻塞 bug (详见 [[phase2_issues]])

1. **PYTHONPATH 缺父目录**: `custom_reward.py` 用 `from verl1.X import`, verl 按 `spec_from_file_location` 加载不会注入父目录, 旧脚本 PYTHONPATH 只有 `.../rl/train/verl1` -> 重启即 `ImportError: No module named 'verl1'`。修: 脚本 PYTHONPATH 加 `/data/zhuldz/lunwen/rl/train`。
2. **skills_used 嵌套**: `generate_code_with_skills_used` 返回 `List[List[str]]`, `_generate_codes` 未展平 -> `record_outcome([[G217,...]])` -> `_skills.get(list)` TypeError 被 try/except 吞掉 -> Step 2 静默失效。修: `_generate_codes` 展平+去重为 `List[str]`。

## 8×L40 优化配置 (`rl/train/verl1/train_8l40.sh`)

关键 override (hydra, 已通过 validate_config):
```
trainer.n_gpus_per_node=8
actor_rollout_ref.rollout.data_parallel_size=8      # 8 倍 rollout 吞吐
actor_rollout_ref.rollout.tensor_model_parallel_size=1  # 4B 单卡放得下, 避免 TP 通信
data.train_batch_size=40                            # 40 = 8(DP)×5(n), 最小可行
actor_rollout_ref.actor.ppo_mini_batch_size=40
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
actor_rollout_ref.actor.fsdp_config.param_offload=false   # 8×46GB 充足, 关 offload 提速
actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
actor_rollout_ref.model.enable_gradient_checkpointing=True  # 首跑保稳; 确认不 OOM 后可关掉加速
```
注: `rollout.n` 被 `iterative_grpo_trainer.py:237` 强制设为 5 (覆盖 config 的 4)。
注: ref 的 param_offload override 没生效 (verl 内部处理, ref 无 optimizer 故 optimizer_offload 无关; ref offload=true 安全, 仅 KL forward 稍慢)。

## ⚠️ 真实瓶颈: API, 不是 GPU

B(DeepSeek-V3.2) + C(DeepSeek-V4-Flash) 是网络调用, 受 `MAX_CONCURRENCY=30` 限制。每 step:
- train_batch_size=40, n=5 -> 200 条轨迹 -> 200 B + 200 C API 调用
- 30 并发 -> ~7 串行批次 × ~5s -> **每 step API 等待 ~1-2 min, GPU 此时空闲**
- 8×L40 主要加速 A 模型 FSDP 训练 + vllm rollout (DP=8), 但 reward 阶段 GPU 闲置

提速选项 (按收益排序):
1. 调大 `custom_reward.py:MAX_CONCURRENCY` (现 30) - **前提: 确认 DeepSeek API 配额允许更高并发** (可能触发限流)
2. API 余量足时 `train_batch_size` 升到 80 (40×2), 但每 step API 等待翻倍
3. 关 `enable_gradient_checkpointing` (确认不 OOM 后) - 提速 A 模型训练阶段
4. 关 ref param_offload (若 verl 允许) - 提速 KL forward

## 用法

```bash
cd /data/zhuldz/lunwen/rl/train/verl1
bash train_8l40.sh
# 日志: outputs/log/8l40_<ts>/out.txt, gpu_monitor.csv, progress.txt
```

## 待办 (用户决策)

- ⚠️ **Step 1 效果未达预期** (见 [[step1-validation-result-20260713]]): 全量注入在 BigCodeBench p=0.349 不显著。重启 GRPO 前建议先在 MBPP/Humaneval 验证, 或消融 Meta-Skill。
- 重启 GRPO 后: trace 的 `skills_used` 将是实际注入子集 (50), `online_success_rate` 开始累积; 首轮后从新 trace 蒸馏候选跑 `experience/updater.py`。

## 相关文件
- [[five-step-status-20260713]] - 5 项改造代码状态
- [[step1-validation-result-20260713]] - Step 1 负结果 (重启前必读)
- [[phase2_issues]] - 2 个阻塞 bug 详记

## 注入路径切换开关 (2026-07-13 新增)

`b_agent_interface.py` 的 `_build_full_system` 现支持 3 种注入模式, 由环境变量 `BAGENT_INJECTION_MODE` 控制 (默认 `full`), **无需改代码即可在 GRPO 重启时切换**:

| 模式 | 注入 | Meta-Skill | 用途 |
|------|------|-----------|------|
| `full` (默认) | 全量 50 条 compact | ✅ | Step 1 新路径 |
| `full_no_meta` | 全量 50 条 compact | ❌ | 若诊断显示 Meta-Skill 过度抑制 |
| `retrieve` | retrieve top_k=5 | ❌ | 旧路径 (曾 p=0.0118 显著) |

- `train_8l40.sh` 已留注释行 `# export BAGENT_INJECTION_MODE=...`, 诊断结果出来后取消注释即可。
- 已验证 3 模式注入正确 (full=6672字符/50skill+Meta, full_no_meta=6070/50, retrieve=1602/5)。
- GRPO 路径 `custom_reward.py` -> `load_b_model()` -> `BAgentInterface()` 在 init 时读 env, 全链路生效。
