#!/usr/bin/env bash
# Launch LiveMath newmech smoke run on V3.2. Source _env.sh for backend.
# Usage:  bash _launch_smoke_v32.sh
# Needs QWEN_CHAT_API_KEY set in env first (export it before running).
set -euo pipefail
cd /data/zhuldz/lunwen
source lunwen/bin/activate
: "${QWEN_CHAT_API_KEY:?请先 export QWEN_CHAT_API_KEY}"
source rl/train/verl1/skill_evolution/_env.sh
mkdir -p rl/train/verl1/outputs
nohup python -u rl/train/verl1/skill_evolution/skill_evolver_math.py \
  --init_skill /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --iters 2 --K 4 --eval_k 8 --train_n 18 \
  --facet_tree --expand \
  --paired_rescore \
  --rescue_on_reject --veto_upper_bound --veto_skip_n1 \
  --facet_group_admit --subset_admit \
  --out_dir rl/train/verl1/outputs/livemath_newmech_smoke_v32_r2 \
  > rl/train/verl1/outputs/livemath_newmech_smoke_v32_r2.log 2>&1 &
echo "SMOKE_PID=$!"
echo "log: rl/train/verl1/outputs/livemath_newmech_smoke_v32_r2.log"
