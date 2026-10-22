#!/usr/bin/env bash
set -u
cd /data/zhuldz/lunwen
source lunwen/bin/activate
export PYTHONPATH="/data/zhuldz/skillopt:${PYTHONPATH:-}"
nohup python -u rl/train/verl1/skill_evolution/precompute_s0_chunked.py \
  > rl/train/verl1/outputs/s0_cache_regen.log 2>&1 &
echo "LAUNCHED pid=$!"
