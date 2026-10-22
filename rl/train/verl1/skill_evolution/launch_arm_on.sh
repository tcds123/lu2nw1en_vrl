#!/usr/bin/env bash
# ON arm of pool_filter controlled comparison (--pool_filter ENABLED).
# Prereq: rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json must exist (~71 items).
# ~8h, 8 iters K=4 eval_k=3 facet_tree+expand. Detaches to survive classifier outages.
set -u
cd /data/zhuldz/lunwen
source lunwen/bin/activate

CACHE=rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json
if [ ! -f "$CACHE" ]; then
  echo "ABORT: s0_cache missing ($CACHE). Run launch_s0_regen.sh first." >&2
  exit 1
fi

nohup python -u rl/train/verl1/skill_evolution/skill_evolver_math.py \
  --s0 /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --init_skill /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --split_dir /data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42 \
  --s0_cache "$CACHE" \
  --iters 8 --K 4 --eval_k 3 --n_diag 8 --train_n 35 \
  --facet_tree --expand \
  --pool_filter \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_poolfilter \
  > rl/train/verl1/outputs/arm_on_poolfilter.log 2>&1 &
echo "ON arm LAUNCHED pid=$! -> rl/train/verl1/outputs/arm_on_poolfilter.log"
