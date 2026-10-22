---
name: gpu-training-platform-20260721
description: GPU资源信息 - 本机无GPU但可申请最多8卡L40在托管训练平台训练, 不要再询问GPU
metadata:
  type: project
---

# GPU / 训练资源 (2026-07-21 确认)

## 关键事实 (用户明确指示, 后续不要再询问 GPU)
- **⚠️ 2026-07-22 更新: 本机现已可用 1 卡 GPU** (nvidia-smi 显示 1× NVIDIA L40 46068MiB)。推翻下方旧记录"本机无GPU"。本机 1 卡用于端到端冒烟 (Qwen3-0.6B), 不出论文结果。详见 [[gpu-strategy-1card-smoke-8card-prod-20260722]]。
- ~~(旧, 2026-07-21) 本机 (aimp-notebook-zhuldz-0) 无 GPU~~ — 已过时, 见上。
- **正式训练在托管训练平台**: 可申请最多 **8 卡 L40** 在托管训练平台跑训练 (GRPO 等)。本机 1 卡仅冒烟用。
- **不要再问用户 GPU 资源**: 已确认可用 (本机 1×L40 冒烟 / 托管 8×L40 正式)。需要正式训练时按 8×L40 规划并提交到托管平台。

## 影响
- GRPO 训练 A 模型 (我们的创新方法: GRPO 优化 skill) **可执行**, 在托管训练平台 8×L40 上跑。
- [[status-review-20260713]] "GRPO 卡数决策 4 卡" 可上调到 8 卡 (L40 48GB, Qwen3-4B 够)。或维持 4 卡省资源, 视 reward 占比 (曾 79%) 而定。
- 本机负责: 数据准备、eval (skillopt/DeepSeek-V3.2 API)、survey 核实、方法设计; 托管平台负责: GRPO 训练。

## A 模型权重 (本地就绪)
- /data/zhuldz/lunwen/models/Qwen3-4B-Instruct-2507 (7.6G) — A 模型, GRPO 训练对象
- /data/zhuldz/lunwen/models/Qwen3-0.6B — 冒烟/小规模验证用

## 训练入口
- rl/train/verl1/train_iterative_grpo.sh — GRPO 启动脚本 (PYTHONPATH 已修, 见 phase2_issues)
- rl/train/verl1/config/grpo_trainer.yaml — 配置
- rl/train/verl1/custom_reward.py — reward 函数 (B+C API + 单测)
- 需提交托管平台时, 把 rl/train/verl1/ + models/ + data/ 同步过去

Related: [[status-review-20260713]], [[gpu-count-decision-20260713]], [[grpo-readiness-20260713]], [[code-skill-method-fix-task-20260721]]
