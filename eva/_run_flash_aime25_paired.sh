#!/usr/bin/env bash
# 阶段① 1b: AIME25 paired 评测 (evolver_best + skillopt_best vs S₀) on flash — held-out.
# cap 隔离 16384 (#26/#29), 复用 flash AIME25 S₀ cache (已转嵌套格式).
# ⚠️ 串行: 等 AIME24 paired (PID 进程) 跑完再起, 共享 flash 20RPM 配额.
set -euo pipefail
cd /data/zhuldz/lunwen
source lunwen/bin/activate

LOG="eva/flash_aime25_paired_20261104.log"
nohup python eva/crossmodel_flash_baseline.py \
  --provider deepseek-v4-flash \
  --data /data/zhuldz/lunwen/data/aime25.jsonl \
  --s0_cache rl/train/verl1/outputs/crossmodel_baseline/flash_aime25_s0_20261104_100708/s0_cache_aime25.json \
  --cand_skill \
    rl/train/verl1/skill_evolution/outputs/skill_evolver_aime_v2/full_off_20260930/run_20260930_161552/best_skill_evolved.md \
    rl/train/verl1/outputs/tf_skill_aime_v2/skillopt/aime-skillopt-20261008/train/best_skill.md \
  --eval_k 8 --max_tokens 16384 \
  > "$LOG" 2>&1 &

echo "launched PID=$!"
echo "log: $LOG"
