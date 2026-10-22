#!/usr/bin/env bash
# Launch Exp 2: 五位置轮换敏感性检查 (CF F01).
# 4 模型臂 (s0/on/off/skillopt) × 5 位置 × 106 题 × k=3 = 6360 调用 ~3.8h.
# rule_meta_guess 臂已零 API 跑完 (5 个 per_rule_meta_guess_rule_pos*_rot.json 已 cache).
# 必须串行在 Exp1 + s0_regen 之后启动 (process-local 28 RPM TokenBucket 不跨进程,
# 并发=2×28=56 RPM 超 DeepSeek 30 RPM cap → 429 风暴).
set -u
cd /data/zhuldz/lunwen
source lunwen/bin/activate
nohup python -u rl/train/verl1/skill_evolution/eval_test_5pos_rotation.py \
  --skills s0,on,off,skillopt,rule_meta_guess \
  > rl/train/verl1/outputs/skill_evolver_math_poolfilter/phaseA_shuffle_20260917/exp2_rotation_run.log 2>&1 &
echo "LAUNCHED pid=$! -> phaseA_shuffle_20260917/exp2_rotation_run.log"
