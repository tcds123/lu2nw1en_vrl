---
name: gpu-count-decision-20260713
description: 为什么 GRPO 用 4 卡 (而非 8 卡) - 真实 timing 数据支撑的瓶颈分析与卡数决策
metadata:
  type: project
---

# GRPO 卡数决策: 4 卡 (非 8 卡) - 2026-07-13

## 决策

**申请 4 张 L40 (而非 8 张)**。队列紧张时可用 2 张。脚本: `rl/train/verl1/train_fewer_gpu.sh [2|4]` (参数化, 默认 4)。

## 依据: 真实 timing 数据

从 `training_history.csv` (821 steps, 8×L40 实跑) 提取每阶段耗时:

| 阶段 | mean | 占 step | 类型 |
|------|------|--------|------|
| `timing_s/reward` (B+C API) | 300s | **79%** | API, GPU 空闲 |
| `timing_s/update_actor` (train) | 51s | 13% | GPU |
| `timing_s/gen` (rollout) | 21s | 6% | GPU |
| `timing_s/step` | 379s | 100% | - |

- **step 是串行累加** (非 max): `step ≈ gen + reward + update` (21+300+51≈379 ✓)。reward 阶段阻塞整步, 期间 GPU 完全空闲。
- **GPU 实际忙碌仅 20%**, MFU=2.6% (8 卡时每卡只分到 25 条序列, vLLM 吃不饱)。
- reward 是 API 调用 (MAX_CONCURRENCY=30), **与 GPU 数量无关**。减卡只影响 gen+update (72s @ 8卡), 按 ~1/DP 缩放。

## 盈亏平衡分析

`gen+update = reward` 时 GPU 刚好不成为瓶颈: `72 × (8/n) ≈ 300` -> **n ≈ 2**。

| 卡数 | gen+update | vs reward(300s) | step 估算 | 显存/卡 | 评价 |
|------|-----------|----------------|----------|---------|------|
| 8 | 72s | 远低于 | 379s | ~16GB | 浪费 (GPU 闲 80%) |
| **4** | ~115-144s | 低于 | ~415-444s (慢 ~10-17%) | ~24GB | **甜点** ✅ |
| 2 | ~200-288s | ≈ 平衡 | ~500-588s (慢 ~55%) | ~32GB | 可用 (队列紧张时) |
| 1 | ~576s | 超过 (GPU 瓶颈) | ~876s | ~64GB ❌ | 放不下 (FSDP 未分片) |

- FSDP 优化器状态跨 rank 分片: DP=4 ~24GB/卡, DP=2 ~32GB/卡, 均 < L40 48GB; DP=1 ~64GB 放不下。
- **额外收益**: 卡少 -> 每卡 rollout batch 更大 -> vLLM 利用率上升 (8 卡 MFU 2.6% 是因 batch 太小)。
- **divisibility 约束**: `train_batch_size×n % n_gpus == 0` -> `40×5=200 % {2,4,8} == 0` ✓ (verl `validate_config` 实测逻辑)。

## 两个互补 lever

1. **减卡** (本决策): 适配 API 瓶颈, 省卡 + 更快排队。4 卡仅慢 ~15%。
2. **调大 MAX_CONCURRENCY** (`custom_reward.py:79`): 若 DeepSeek API 配额允许, 把 30 提到 60 可让 reward 时间减半 -> 盈亏点上移到 ~4 卡。需先确认配额, 否则触发 429。

## 配置 (train_fewer_gpu.sh)

保持 `train_batch_size=40` 不变 (reward 调用量不变), 只缩 DP:
- `trainer.n_gpus_per_node=${NUM_GPUS}`
- `actor_rollout_ref.rollout.data_parallel_size=${NUM_GPUS}`
- `tensor_model_parallel_size=1` (4B 单卡放得下)
- `ppo_micro_batch_size_per_gpu=4` (不变, 显存按 micro-batch 算, 与 DP 无关)
- `param_offload=false` / `optimizer_offload=false` (FSDP 分片后 DP>=2 显存够)
- `enable_gradient_checkpointing=True`

## 相关

- [[grpo-readiness-20260713]] - GRPO 跑通验证 + 注入路径开关 (BAGENT_INJECTION_MODE)
- [[five-step-status-20260713]] - 5 项改造状态
- [[step1-validation-result-20260713]] - Step 1 诊断进行中 (决定 BAGENT_INJECTION_MODE 取值)
