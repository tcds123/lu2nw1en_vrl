---
name: gpu-strategy-1card-smoke-8card-prod-20260722
description: GPU资源策略 - 本机现可用1卡L40(Qwen3-0.6B冒烟), 正式训练申请托管8卡(Qwen3-4B); S0预计算与A模型大小无关
metadata:
  type: project
---

# GPU 资源策略: 1卡冒烟 + 8卡正式 (2026-07-22 用户确认)

## 关键事实 (用户 2026-07-22 明确指示)
- **本机现已可用 1 卡 GPU**: `nvidia-smi` 显示 1× NVIDIA L40 (46068 MiB)。更新 [[gpu-training-platform-20260721]] "本机无GPU" 的旧记录——本机现在有 1 卡 L40 了。
- **8 卡 = 托管训练平台**: 需要正式跑论文结果时，再向托管平台申请 8×L40。
- **策略**: 本机 1 卡跑 **Qwen3-0.6B** 做端到端冒烟 (打通链路), 托管 8 卡跑 **Qwen3-4B-Instruct-2507** 出正式结果。

## 分工
| 资源 | 模型 | 用途 |
|------|------|------|
| 本机 1×L40 | Qwen3-0.6B (`models/Qwen3-0.6B/v1/`) | 阶段0端到端冒烟: parquet加载→reward调B API+LCB单测+delta→GRPO更新不报错 |
| 托管 8×L40 | Qwen3-4B-Instruct-2507 (`models/Qwen3-4B-Instruct-2507/`) | 阶段1正式训练, 出论文主结果 |

## 重要: S_0 预计算与 A 模型大小无关
- S_0 baseline 预计算 (`precompute_s0_baseline.py`) 用的是 **B 模型 (DeepSeek-V3.2)** 跑 S_0, 不涉及 A 模型。
- → 0.6B 冒烟和 4B 正式训练 **共用同一份 `s0_baseline_lcb.json`**, 不用为 0.6B 单独重算。
- 这一份预计算结果对两种规模通用。

## 冒烟脚本配置 (1卡 0.6B)
`train_lcb_skillopt.sh` 现写的是 8 卡配置 (n_gpus_per_node=8, dp=8, batch=8)。冒烟需改 1 卡版:
- model.path → `models/Qwen3-0.6B/v1`
- trainer.n_gpus_per_node=1
- actor_rollout_ref.rollout.data_parallel_size=1
- data.train_batch_size 调小 (如 4), n 可保持 5 (轨迹够算 advantage)
- max_iterations=1 (只验 1 step 不报错)

## 当前阻塞 (2026-07-22 已定位并修复)
- `precompute_s0_baseline.py` 旧 bug: 只设 `TARGET_BACKEND` 未设 `QWEN_CHAT_*` backend env → 61题全报 "Azure OpenAI endpoint is not configured", pass_rate=-3.0。
- 已修: 补全 `QWEN_CHAT_API_KEY/BASE_URL/MODEL/...` setdefault (与 train_lcb_skillopt.sh:35-44 一致)。
- 1题冒烟已通过 (agent_ok=True, backend 通)。
- 全量 61 题重跑中。

Related: [[our-grpo-skill-method-design-20260721]], [[gpu-training-platform-20260721]], [[grpo-readiness-20260713]]
