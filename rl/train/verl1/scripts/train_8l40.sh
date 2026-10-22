#!/bin/bash
# ============================================================
# 8×L40 优化版 GRPO 训练启动脚本 (Qwen3-4B-Instruct-2507)
# 基于 train_iterative_grpo.sh, 增加 8-GPU 并行 + 内存优化 override。
# 用法: cd /data/zhuldz/lunwen/rl/train/verl1 && bash train_8l40.sh
# ============================================================
set -e
set -x

echo "Starting 8×L40 GRPO Training (verl framework)..."

# 禁用 Ray OOM 自动查杀
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
# 关键: 包含 rl/train 父目录, 否则 custom_reward.py 的 `from verl1.X import` 会 ImportError
export PYTHONPATH="$PYTHONPATH:/data/zhuldz/lunwen/generation:/data/zhuldz/lunwen/judge:/data/zhuldz/lunwen/rl/train/verl:/data/zhuldz/lunwen/rl/train:/data/zhuldz/lunwen/rl/train/verl1"
unset http_proxy https_proxy

# Step 1 注入路径切换 (诊断结果决定, 无需改代码):
#   full         - 全量 compact skills + Meta-Skill (Step 1 默认, 不设置即此项)
#   full_no_meta - 全量 compact skills, 无 Meta-Skill (若诊断显示 Meta-Skill 过度抑制)
#   retrieve     - retrieve(top_k=5) 旧路径 (若诊断显示旧路径更优)
# export BAGENT_INJECTION_MODE=full

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }
LOG_DIR="/data/zhuldz/lunwen/rl/train/verl1/outputs/log/8l40_$(timestamp)"
mkdir -p "$LOG_DIR"
echo "Logging to: $LOG_DIR"
echo $$ > "$LOG_DIR/training.pid"

# GPU 监控 (每30秒)
{
  echo "timestamp,gpu_id,gpu_util,mem_used,mem_total,temp" > "$LOG_DIR/gpu_monitor.csv"
  while true; do
    TS=$(date +"%H:%M:%S")
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | \
      while read line; do echo "$TS,$line"; done >> "$LOG_DIR/gpu_monitor.csv"
    sleep 30
  done
} &
GPU_MONITOR_PID=$!

PROGRESS_FILE="$LOG_DIR/progress.txt"
echo "Training started at $(date)" > "$PROGRESS_FILE"

# ============================================================
# 8×L40 训练参数 (hydra override)
# ============================================================
# 模型: Qwen3-4B (bf16 ~8GB), 8×L40 (46GB each) = 368GB 总显存, 远超模型需求。
# 策略: TP=1 (4B 单卡放得下, 避免 TP 通信开销) + DP=8 (8 倍 rollout 吞吐)。
#
# ⚠️ 关键瓶颈提示: B(DeepSeek API) + C(DeepSeek API) 是网络调用, 非 GPU 计算。
#    train_batch_size=40, rollout.n=5 (trainer 强制) -> 每 step 200 条轨迹
#    -> 200 B + 200 C API 调用, 受 MAX_CONCURRENCY=30 限制 -> GPU 在 reward 阶段空闲。
#    想再提速: (1) 调大 custom_reward.py 的 MAX_CONCURRENCY (确认 API 配额允许);
#    (2) 若 API 余量足, train_batch_size 可升到 80 (40×2), 但每 step API 等待翻倍。
# ============================================================
python -m iterative_grpo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax_exp.parquet \
    data.val_files=/data/zhuldz/lunwen/data/CodeEval-Pro/dataset_1/humaneval_pro.parquet \
    actor_rollout_ref.model.path=/data/zhuldz/lunwen/models/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.train_batch_size=40 \
    data.micro_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=40 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
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
    > "$LOG_DIR/out.txt" 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID" >> "$PROGRESS_FILE"

# 进度监控 (每60秒快照)
while kill -0 $TRAIN_PID 2>/dev/null; do
  echo "=== $(date) ===" >> "$PROGRESS_FILE"
  echo "out.txt size: $(wc -c < "$LOG_DIR/out.txt" 2>/dev/null || echo 0) bytes" >> "$PROGRESS_FILE"
  echo "rollout traces today: $(ls /data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout/readable_trace_$(date +%Y%m%d)*.log 2>/dev/null | wc -l) files" >> "$PROGRESS_FILE"
  tail -1 "$LOG_DIR/out.txt" 2>/dev/null | grep -v "^$" | grep -v "^[[:space:]]*$" >> "$PROGRESS_FILE" 2>/dev/null
  sleep 60
done

kill $GPU_MONITOR_PID 2>/dev/null || true
wait $TRAIN_PID
EXIT_CODE=$?
echo "Training finished with exit code: $EXIT_CODE at $(date)" >> "$PROGRESS_FILE"
echo "Training completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
