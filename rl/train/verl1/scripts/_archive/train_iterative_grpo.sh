#!/bin/bash
set -e
set -x

echo "Starting Iterative GRPO Training with verl framework..."

# 禁用 Ray 的 Worker 自动查杀（针对 OOM）
export RAY_memory_monitor_refresh_ms=0

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:/data/zhuldz/lunwen/generation:/data/zhuldz/lunwen/judge:/data/zhuldz/lunwen/rl/train/verl:/data/zhuldz/lunwen/rl/train:/data/zhuldz/lunwen/rl/train/verl1"
# ↑ 增加 /data/zhuldz/lunwen/rl/train: 让 custom_reward.py 里 `from verl1.b_agent_interface import ...`
#   能解析 (verl1 作 namespace package)。verl 用 spec_from_file_location 按文件路径加载
#   reward fn, 不会自动加父目录到 sys.path, 不加这行重启会 ImportError: No module named 'verl1'。
unset http_proxy
unset https_proxy

# 定义时间戳函数
timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}

# 固定日志路径（共享存储，notebook 可见）
LOG_DIR="/data/zhuldz/lunwen/rl/train/verl1/outputs/log/$(timestamp)"
mkdir -p "$LOG_DIR"
echo "Logging to: $LOG_DIR"

# ============================================================
# 监控探针 1: 记录进程 PID
# ============================================================
echo $$ > "$LOG_DIR/training.pid"
echo "PID: $$"

# ============================================================
# 监控探针 2: 后台 GPU 监控 (每30秒采样)
# ============================================================
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
echo "GPU monitor PID: $GPU_MONITOR_PID"

# ============================================================
# 监控探针 3: 训练进度追踪
# ============================================================
PROGRESS_FILE="$LOG_DIR/progress.txt"
echo "Training started at $(date)" > "$PROGRESS_FILE"

python -m iterative_grpo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax_exp.parquet \
    data.val_files=/data/zhuldz/lunwen/data/CodeEval-Pro/dataset_1/humaneval_pro.parquet \
    actor_rollout_ref.model.path=/data/zhuldz/lunwen/models/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.nnodes=1 \
    iterative_rl.max_iterations=5 \
    iterative_rl.convergence_threshold=0.01 \
    > "$LOG_DIR/out.txt" 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID" >> "$PROGRESS_FILE"

# ============================================================
# 监控探针 4: 等待训练完成，期间每60秒写入进度快照
# ============================================================
while kill -0 $TRAIN_PID 2>/dev/null; do
  # 记录当前 GPU 状态
  echo "=== $(date) ===" >> "$PROGRESS_FILE"
  echo "out.txt size: $(wc -c < "$LOG_DIR/out.txt" 2>/dev/null || echo 0) bytes" >> "$PROGRESS_FILE"
  echo "rollout traces today: $(ls /data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout/readable_trace_$(date +%Y%m%d)*.log 2>/dev/null | wc -l) files" >> "$PROGRESS_FILE"
  # 抓取最近一行非 tqdm 的实质性日志
  tail -1 "$LOG_DIR/out.txt" 2>/dev/null | grep -v "^$" | grep -v "^[[:space:]]*$" >> "$PROGRESS_FILE" 2>/dev/null
  sleep 60
done

# ============================================================
# 清理
# ============================================================
kill $GPU_MONITOR_PID 2>/dev/null || true
wait $TRAIN_PID
EXIT_CODE=$?

echo "Training finished with exit code: $EXIT_CODE at $(date)" >> "$PROGRESS_FILE"
echo "Training completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
