#!/bin/bash
# ============================================================
# LCB skillopt GRPO 冒烟脚本 (1卡 0.6B, [[gpu-strategy-1card-smoke-8card-prod-20260722]])
#
# 目的: 本机 1×L40 + Qwen3-0.6B 端到端打通链路, 不追求效果:
#   - parquet 加载 ✓
#   - custom_reward_lcb 调通 B API (DeepSeek-V3.2) + LCB stdin/stdout 单测 + delta 计算 ✓
#   - qid 反查对齐 (TASK_POOL / PROMPT_TO_QID) — 阶段0.2 关键验证
#   - GRPO 1 step 更新不报错 ✓
# 验证通过后, 同样配置换 model.path + n_gpus=8 上托管 8×L40 跑正式 4B 训练。
#
# 用法: cd /data/zhuldz/lunwen/rl/train/verl1 && bash train_lcb_skillopt_1card_smoke.sh
# ============================================================
set -e
set -x

echo "Starting LCB skillopt GRPO 1-card SMOKE (Qwen3-0.6B)..."

export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:/data/zhuldz/lunwen/generation:/data/zhuldz/lunwen/judge:/data/zhuldz/lunwen/rl/train/verl:/data/zhuldz/lunwen/rl/train:/data/zhuldz/lunwen/rl/train/verl1:/data/zhuldz/skillopt"
unset http_proxy https_proxy

# skillopt B 模型 API (DeepSeek-V3.2 via aimpapi) — 与 train_lcb_skillopt.sh 一致
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:?请先设置环境变量 QWEN_CHAT_API_KEY}"
export QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1}"
export QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-/model/DeepSeek-V3.2}"
export QWEN_CHAT_AIGC_USER="${QWEN_CHAT_AIGC_USER:-zhuldz}"
export QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-4096}"
export QWEN_CHAT_TEMPERATURE="${QWEN_CHAT_TEMPERATURE:-0.0}"
export QWEN_CHAT_MAX_RPM="${QWEN_CHAT_MAX_RPM:-28}"
export QWEN_CHAT_BURST="${QWEN_CHAT_BURST:-2}"
export TARGET_BACKEND="${TARGET_BACKEND:-qwen_chat}"
export OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-qwen_chat}"

# 数据 + S_0 缓存路径 (reward 读取) — 与 8 卡正式版共用同一份 parquet + s0 cache
export VERL_DATA_PATH=/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_lcb_skillopt.parquet
export SKILLOPT_S0_PATH=/data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md
export SKILLOPT_S0_CACHE=/data/zhuldz/lunwen/rl/train/verl1/outputs/s0_baseline_lcb.json

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }
LOG_DIR="/data/zhuldz/lunwen/rl/train/verl1/outputs/log/lcb_skillopt_smoke_$(timestamp)"
mkdir -p "$LOG_DIR"
echo "Logging to: $LOG_DIR"
echo $$ > "$LOG_DIR/training.pid"

# ============================================================
# GRPO 参数 (1卡 0.6B 冒烟)
# - model: Qwen3-0.6B (models/Qwen3-0.6B/v1)
# - 1 GPU: n_gpus_per_node=1, dp=1
# - batch=4, n=5 → 20 轨迹/step (够算 advantage, 够小不爆显存)
# - max_iterations=1 (只验 1 step 链路通)
# ============================================================
python -m iterative_grpo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files="$VERL_DATA_PATH" \
    data.val_files=/data/zhuldz/lunwen/data/CodeEval-Pro/dataset_1/humaneval_pro.parquet \
    actor_rollout_ref.model.path=/data/zhuldz/lunwen/models/Qwen3-0.6B/v1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.train_batch_size=4 \
    data.micro_batch_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=false \
    iterative_rl.max_iterations=1 \
    iterative_rl.convergence_threshold=0.01 \
    custom_reward_function.path=/data/zhuldz/lunwen/rl/train/verl1/legacy/custom_reward_lcb.py \
    custom_reward_function.name=compute_custom_reward \
    > "$LOG_DIR/out.txt" 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

PROGRESS_FILE="$LOG_DIR/progress.txt"
echo "Training started at $(date)" > "$PROGRESS_FILE"

while kill -0 $TRAIN_PID 2>/dev/null; do
  echo "=== $(date) ===" >> "$PROGRESS_FILE"
  echo "out.txt size: $(wc -c < "$LOG_DIR/out.txt" 2>/dev/null || echo 0) bytes" >> "$PROGRESS_FILE"
  tail -1 "$LOG_DIR/out.txt" 2>/dev/null | grep -v "^$" | grep -v "^[[:space:]]*$" >> "$PROGRESS_FILE" 2>/dev/null
  sleep 60
done

wait $TRAIN_PID
EXIT_CODE=$?
echo "Training finished with exit code: $EXIT_CODE at $(date)" >> "$PROGRESS_FILE"
echo "Training completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
