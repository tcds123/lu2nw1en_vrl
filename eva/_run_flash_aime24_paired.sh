#!/usr/bin/env bash
# 阶段① 1a: AIME24 paired 评测 (evolver_best + skillopt_best vs S₀) on flash
# cap 隔离 16384 (#26/#29), 复用 flash AIME24 S₀ cache, 不重跑 S₀.
set -euo pipefail
cd /data/zhuldz/lunwen
source lunwen/bin/activate

LOG="eva/flash_aime24_paired_20261104.log"
nohup python eva/crossmodel_flash_baseline.py \
  --provider deepseek-v4-flash \
  --data /data/zhuldz/lunwen/data/aime24.jsonl \
  --s0_cache rl/train/verl1/outputs/crossmodel_baseline/run_20261102_170450/s0_cache_aime.json \
  --cand_skill \
    rl/train/verl1/skill_evolution/outputs/skill_evolver_aime_v2/full_off_20260930/run_20260930_161552/best_skill_evolved.md \
    rl/train/verl1/outputs/tf_skill_aime_v2/skillopt/aime-skillopt-20261008/train/best_skill.md \
  --eval_k 8 --max_tokens 16384 \
  > "$LOG" 2>&1 &

echo "launched PID=$!"
echo "log: $LOG"
