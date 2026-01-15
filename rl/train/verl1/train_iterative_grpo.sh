#!/bin/bash
set -e
set -x

echo "Starting Iterative GRPO Training with verl framework..."

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:/data/zhuldz/lunwen/generation:/data/zhuldz/lunwen/judge:/data/zhuldz/lunwen/rl/train/verl"
unset http_proxy 
unset https_proxy

# 定义时间戳函数
timestamp() {
  date +"%Y-%m-%d_%H-%M-%S-%N"
}

# 设置日志路径
LOG_PATH="/data/zhuldz/lunwen/rl/train/verl1/outputs/log/$(timestamp)-$RANDOM"
mkdir -p "$LOG_PATH"

echo "Logging to: $LOG_PATH"


python -m iterative_grpo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax.parquet \
    data.val_files=/data/zhuldz/lunwen/data/CodeEval-Pro/dataset_1/humaneval_pro.parquet \
    data.train_batch_size=8 \
    data.micro_batch_size=1 \
    actor_rollout_ref.model.path=/data/zhuldz/lunwen/models/Qwen3-4B \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.nnodes=1 \
    iterative_rl.max_iterations=5 \
    iterative_rl.convergence_threshold=0.01 \
    > "$LOG_PATH/out.txt" 2>&1

echo "Training completed successfully!"