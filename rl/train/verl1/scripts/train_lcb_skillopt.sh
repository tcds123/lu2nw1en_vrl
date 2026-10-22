#!/bin/bash
# ============================================================
# LCB skillopt GRPO 训练 (改造1+2, [[our-grpo-skill-method-design-20260721]])
# A 模型 GRPO 优化初始 skill S_0, reward = delta pass-rate (skill_pass - s0_pass)
#
# 与 train_8l40.sh 的差异:
#   - 数据: train_lcb_skillopt.parquet (61题, LCB 同域)
#   - reward: custom_reward_lcb.py (stdin/stdout 执行器 + delta pass-rate)
#   - B 模型: skillopt chat_target (DeepSeek-V3.2), 不经 BAgent skill curation
#   - S_0 baseline: 离线预计算 (precompute_s0_baseline.py), 训练时读缓存
#
# 前置 (本机或训练机执行一次):
#   1. 生成 parquet: python data/OpenCodeInstruct/json2parquet_skillopt.py
#   2. 预计算 S_0 baseline: cd /data/zhuldz/skillopt && \
#      python /data/zhuldz/lunwen/rl/train/verl1/skill_evolution/precompute_s0_baseline.py
#      (产出 outputs/s0_baseline_lcb.json)
#   3. 设 API env (与 run_code.sh 一致):
#      export QWEN_CHAT_API_KEY="msk-bf2324..."
#      export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/v1"
#      export QWEN_CHAT_MAX_RPM=28
#
# 用法: cd /data/zhuldz/lunwen/rl/train/verl1 && bash train_lcb_skillopt.sh
# ============================================================
set -e
set -x

echo "Starting LCB skillopt GRPO Training..."

export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:/data/zhuldz/lunwen/generation:/data/zhuldz/lunwen/judge:/data/zhuldz/lunwen/rl/train/verl:/data/zhuldz/lunwen/rl/train:/data/zhuldz/lunwen/rl/train/verl1:/data/zhuldz/skillopt"
unset http_proxy https_proxy

# skillopt B 模型 API (DeepSeek-V3.2 via aimpapi) — 与 run_code.sh 一致, 必须在训练机设
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:?请先设置环境变量 QWEN_CHAT_API_KEY}"
export QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1}"
export QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-/model/DeepSeek-V3.2}"
export QWEN_CHAT_AIGC_USER="${QWEN_CHAT_AIGC_USER:-zhuldz}"
export QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-4096}"
export QWEN_CHAT_TEMPERATURE="${QWEN_CHAT_TEMPERATURE:-0.0}"
export QWEN_CHAT_MAX_RPM="${QWEN_CHAT_MAX_RPM:-28}"
export QWEN_CHAT_BURST="${QWEN_CHAT_BURST:-2}"
# skillopt backend: 必须在 import skillopt.model 前设 (默认 openai_chat 走 Azure 报错)
export TARGET_BACKEND="${TARGET_BACKEND:-qwen_chat}"
export OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-qwen_chat}"

# 数据 + S_0 缓存路径 (reward 读取)
export VERL_DATA_PATH=/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_lcb_skillopt.parquet
export SKILLOPT_S0_PATH=/data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md
export SKILLOPT_S0_CACHE=/data/zhuldz/lunwen/rl/train/verl1/outputs/s0_baseline_lcb.json

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }
LOG_DIR="/data/zhuldz/lunwen/rl/train/verl1/outputs/log/lcb_skillopt_$(timestamp)"
mkdir -p "$LOG_DIR"
echo "Logging to: $LOG_DIR"
echo $$ > "$LOG_DIR/training.pid"

PROGRESS_FILE="$LOG_DIR/progress.txt"
echo "Training started at $(date)" > "$PROGRESS_FILE"

# ============================================================
# GRPO 参数 (LCB 61题, batch 调小)
# LCB train=41 (easy8/medium13/hard20), 全量天然分布起步 (待决策项2)
# batch_size=8, n=5 -> 40 轨迹/step (与 41 题量级匹配, 每题约 1 采样)
# ============================================================
python -m iterative_grpo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files="$VERL_DATA_PATH" \
    data.val_files=/data/zhuldz/lunwen/data/CodeEval-Pro/dataset_1/humaneval_pro.parquet \
    actor_rollout_ref.model.path=/data/zhuldz/lunwen/models/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.train_batch_size=8 \
    data.micro_batch_size=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.data_parallel_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=false \
    iterative_rl.max_iterations=5 \
    iterative_rl.convergence_threshold=0.01 \
    custom_reward_function.path=/data/zhuldz/lunwen/rl/train/verl1/legacy/custom_reward_lcb.py \
    custom_reward_function.name=compute_custom_reward \
    > "$LOG_DIR/out.txt" 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID" >> "$PROGRESS_FILE"

# GPU 监控 (每30秒, 写 csv)
{
  echo "timestamp,gpu_util,mem_used,mem_total" > "$LOG_DIR/gpu_monitor.csv"
  while true; do
    TS=$(date +"%H:%M:%S")
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
      awk -v ts="$TS" '{print ts","$0}' >> "$LOG_DIR/gpu_monitor.csv"
    sleep 30
  done
} &
GPU_MONITOR_PID=$!

while kill -0 $TRAIN_PID 2>/dev/null; do
  echo "=== $(date) ===" >> "$PROGRESS_FILE"
  echo "out.txt size: $(wc -c < "$LOG_DIR/out.txt" 2>/dev/null || echo 0) bytes" >> "$PROGRESS_FILE"
  # 抓 reward 统计 (🏁) + step 完成 (✅ Step) + 错误, 便于训练中快速判断健康度
  grep -E "🏁|✅ \[System\] Step|❌ Error|AttributeError|Traceback" "$LOG_DIR/out.txt" 2>/dev/null | tail -3 >> "$PROGRESS_FILE"
  echo "reward traces: $(ls /data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout_lcb/readable_trace_*.log 2>/dev/null | wc -l) files" >> "$PROGRESS_FILE"
  sleep 60
done

kill $GPU_MONITOR_PID 2>/dev/null || true

wait $TRAIN_PID
EXIT_CODE=$?
echo "Training finished with exit code: $EXIT_CODE at $(date)" >> "$PROGRESS_FILE"
echo "Training completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
