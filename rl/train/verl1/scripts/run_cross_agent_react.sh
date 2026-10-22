#!/bin/bash
# Task 3 跨 agent 全量 ReAct 评测 (单次启动)
# single-call baseline 用 8-04 fresh (multiseed_test_ablation/, 避免 7-30 cached drift)
# ReAct 侧 fresh, 同 skill(12633201)/模型/28RPM, 仅差 REACT_SUFFIX+execute_code 工具循环
set -e
cd /data/zhuldz/lunwen
source lunwen/bin/activate
exec python -u rl/train/verl1/eval/eval_bagent_react_multiseed.py \
  --skills evolver --seeds 4 --eval_k 3 --max_steps 6 --max_workers 40 \
  --single_dir rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test_ablation \
  2>&1 | tee rl/train/verl1/outputs/cross_agent_react_run.log
