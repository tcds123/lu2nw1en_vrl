#!/usr/bin/env bash
# Launch LiveMath newmech FULL RUN on V3.2 (冻结方案 Option A, 2026-11-26).
# 严格使用冻结参数, 不改实验配置. 启动前 source _env.sh (V3.2 后端, RPM=28).
#
# 冻结配置 (见 memory/livemath-fullrun-plan-20261126.md):
#   eval_k=3, iters=6, K=4, train_n=18, pool_filter=ON (核实B确认不与rescue冲突)
#   新机制4项ON: paired_rescore/rescue/veto_upper_bound+veto_skip_n1/facet_group_admit
#   门控全部不动 (eps=0.02, CI>0, subset/group/veto 不变)
#   预算硬停: --max_calls 60000 --wall_seconds 129600 (60k/36h, iter4 必留checkpoint)
#   per_item_delta 落盘 (gate_trace.jsonl, 补 smoke 缺口)
#
# 三层评估 (run 后单独跑, 不在本脚本): L1 select=71 in-sample / L2 test=106 独立 / L3 分层
#
# Usage:  bash launch_livemath_fullrun_v32.sh
# Needs QWEN_CHAT_API_KEY set in env first (export it before running).
# ⚠️ 付费 API 执行仍需单独授权 — 本脚本仅备好, 不自动启动.
set -euo pipefail
cd /data/zhuldz/lunwen
source lunwen/bin/activate
: "${QWEN_CHAT_API_KEY:?请先 export QWEN_CHAT_API_KEY}"
source rl/train/verl1/skill_evolution/_env.sh
mkdir -p rl/train/verl1/outputs

OUT=rl/train/verl1/outputs/livemath_fullrun_v32_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

nohup python -u rl/train/verl1/skill_evolution/skill_evolver_math.py \
  --init_skill /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --iters 6 --K 4 --eval_k 3 --train_n 18 \
  --facet_tree --expand \
  --pool_filter \
  --paired_rescore \
  --rescue_on_reject --veto_upper_bound --veto_skip_n1 \
  --facet_group_admit --subset_admit \
  --max_calls 60000 --wall_seconds 129600 \
  --out_dir "$OUT" \
  > "${OUT}.log" 2>&1 &

echo "FULLRUN_PID=$!"
echo "out_dir: $OUT"
echo "log: ${OUT}.log"
echo "预算硬停: 60000 calls / 129600s (36h), iter4 checkpoint, per_item_delta 落盘 gate_trace.jsonl"
echo "⚠️ 付费 API 已启动. 监控: tail -f ${OUT}.log"
