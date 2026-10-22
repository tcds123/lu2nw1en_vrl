#!/usr/bin/env bash
# Launch S0 cache regeneration for the 1-3-6 controlled-comparison split.
# Output: rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json (~8min, 71 items, k=3)
# Phase B item (a): 走 live tf_global_v2_math.precompute_s0 (EVAL_TEMP=0.3 + manifest),
#   非 legacy precompute_s0_1-3-6.py (temp=0.0, 无 manifest → manifest loader 会拒绝).
# Detaches so it survives classifier outages / session ends.
set -u
cd /data/zhuldz/lunwen
source lunwen/bin/activate
nohup python -u rl/train/verl1/skill_evolution/precompute_s0_live.py \
  > rl/train/verl1/outputs/s0_cache_regen.log 2>&1 &
echo "LAUNCHED pid=$! -> rl/train/verl1/outputs/s0_cache_regen.log"
