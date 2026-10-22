#!/bin/bash
# ============================================================
# 少卡版 GRPO 训练启动脚本 (Qwen3-4B-Instruct-2507)
# 用法: cd /data/zhuldz/lunwen/rl/train/verl1 && bash train_fewer_gpu.sh [2|4]
# 默认 4 卡。基于真实 timing 数据 (training_history.csv, 821 steps) 优化。
# ============================================================
set -e
set -x

NUM_GPUS="${1:-4}"
if [[ "$NUM_GPUS" != "2" && "$NUM_GPUS" != "4" ]]; then
  echo "❌ NUM_GPUS 必须是 2 或 4 (当前: $NUM_GPUS)。DP=1 放不下优化器状态, DP=8 见 train_8l40.sh。"
  exit 1
fi

echo "Starting ${NUM_GPUS}×L40 GRPO Training (verl framework)..."

# ============================================================
# 为什么少卡更优 (真实数据, training_history.csv 821 steps, 8×L40):
#
#   每 step = gen + reward + update (串行累加, 非 max):
#     gen    (rollout, GPU)  :  21s   ( 6%)
#     reward (B+C API)       : 300s   (79%)  <- 瓶颈, GPU 空闲
#     update (train,  GPU)   :  51s   (13%)
#     step 总计              : 379s
#   GPU 实际忙碌 (gen+update)/step = 20%, MFU=2.6%
#
#   reward 是 API 调用 (MAX_CONCURRENCY=30), 与 GPU 数量无关。
#   减少 GPU 只影响 gen+update (72s @ 8卡), 按 ~1/DP 缩放:
#     4卡: gen+update ≈ 115-144s  < reward 300s  -> 仍 reward-bound, 仅慢 ~10-17%
#     2卡: gen+update ≈ 200-288s  ≈ reward 300s  -> 盈亏平衡点 (breakeven)
#     1卡: 放不下 (FSDP 优化器状态未分片 ~64GB > 48GB), 且 GPU 成瓶颈
#
#   FSDP 分片: 优化器状态跨 rank 分片。
#     DP=4: ~24GB/卡   DP=2: ~32GB/卡   (L40 48GB 均放得下)
#
#   额外收益: 卡少 -> 每卡 rollout batch 更大 -> vLLM 利用率更高
#   (8卡 MFU 仅 2.6%, 因每卡只分到 25 条序列, vLLM 吃不饱)。
#
# 结论: 4卡是甜点 (50% 卡省下, 仅慢 ~15%); 2卡是底线 (75% 卡省下, 慢 ~55%, 队列紧张时用)。
# 想再提速可调大 custom_reward.py:MAX_CONCURRENCY (前提: DeepSeek API 配额允许)。
# ============================================================

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
LOG_DIR="/data/zhuldz/lunwen/rl/train/verl1/outputs/log/${NUM_GPUS}l40_$(timestamp)"
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
echo "Training started at $(date) with ${NUM_GPUS} GPUs" > "$PROGRESS_FILE"

# ============================================================
# ${NUM_GPUS}×L40 训练参数 (hydra override)
# 保持 train_batch_size=40 不变 -> reward 阶段 API 调用量不变 (200 任务),
# 只让 gen+update 按 DP 缩放。offload=false (FSDP 分片后 DP>=2 显存够)。
# ============================================================
python -m iterative_grpo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax_exp.parquet \
    data.val_files=/data/zhuldz/lunwen/data/CodeEval-Pro/dataset_1/humaneval_pro.parquet \
    actor_rollout_ref.model.path=/data/zhuldz/lunwen/models/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    data.train_batch_size=40 \
    data.micro_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=40 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.data_parallel_size=${NUM_GPUS} \
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
