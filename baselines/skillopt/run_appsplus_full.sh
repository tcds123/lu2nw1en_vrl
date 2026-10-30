#!/usr/bin/env bash
# APPS_Plus interview 全量托管实验 (2026-07-17, 周六周日 ~48h 预算)
#
# 论证目标: 在 appsplus interview 上拿到 SkillOpt 优化收敛后的 best_skill,
#          与 noskill / seed 做全量 test=2765 三组对位, 把 +6pp 泛化做到统计显著.
#
# Pipeline (顺序执行, 任一失败不阻塞后续 eval):
#   1) optimize 4 epochs  (timeout 40h; best_skill.md 持续更新, 即使被切也有可用 best)
#   2) eval noskill       (test=2765, 全量)
#   3) eval seed skill    (test=2765, 全量, 手写 initial.md)
#   4) eval best skill    (test=2765, 全量, optimize 产物 best_skill.md)
#
# 用法 (托管环境, 断连不死):
#   export QWEN_CHAT_API_KEY="msk-..."
#   nohup bash run_appsplus_full.sh > appsplus_full.log 2>&1 &
#   tail -f appsplus_full.log
#
# 续跑: 已完成的 eval 组会被跳过 (检测 eval_summary.json 存在). optimize 不可续跑,
#       若 optimize 被切, 直接用当前 best_skill.md 进入 eval.
set -uo pipefail
cd "$(dirname "$0")"

PY=/data/zhuldz/venv-skillopt/bin/python
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:?need QWEN_CHAT_API_KEY}"
export QWEN_CHAT_BASE_URL=https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1
export QWEN_CHAT_MODEL=/model/DeepSeek-V3.2
export QWEN_CHAT_AIGC_USER=zhuldz
export QWEN_CHAT_MAX_TOKENS=4096
export QWEN_CHAT_TEMPERATURE=0.0
export TARGET_BACKEND=qwen_chat OPTIMIZER_BACKEND=qwen_chat
export QWEN_CHAT_MAX_RPM=28          # DeepSeek-V3.2 硬上限 30 req/min
export QWEN_CHAT_BURST=2
export BENCH=appsplus
RATIO=2:1:7
DATA_OPTS="model.backend=qwen_chat model.target_backend=qwen_chat model.optimizer_backend=qwen_chat env.data_path=/data/zhuldz/lunwen/data/APPS_Plus/data/v1 env.split_mode=ratio env.split_ratio=$RATIO env.split_dir= env.dataset_glob=data_with_id.json env.difficulty_filter=interview"

CFG=configs/appsplus/default.yaml
DATA=/data/zhuldz/lunwen/data/APPS_Plus/data/v1
OPT_OUT=outputs/appsplus_full_optimize
BEST_SKILL="$OPT_OUT/best_skill.md"
SEED_SKILL=skillopt/envs/appsplus/skills/initial.md

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === APPS_Plus full pipeline START ==="
echo "[$(ts)] RPM=28 RATIO=$RATIO optimize_timeout=${OPT_TIMEOUT_S:-32400}s"

# ---------- 1) optimize (default 9h=32400s; ~18 steps, leaves 5h for eval) ----------
echo "[$(ts)] [1/4] optimize 4 epochs (timeout 40h) -> $OPT_OUT"
mkdir -p "$OPT_OUT"
timeout "${OPT_TIMEOUT_S:-32400}" "$PY" -u scripts/train.py --config "$CFG" \
  --optimizer_backend qwen_chat --target_backend qwen_chat \
  --optimizer_model /model/DeepSeek-V3.2 --target_model /model/DeepSeek-V3.2 \
  --out_root "$OPT_OUT" \
  --cfg-options $DATA_OPTS train.num_epochs=4 \
  > "$OPT_OUT/optimize.log" 2>&1
OPT_RC=$?
if [ "$OPT_RC" -eq 124 ]; then
  echo "[$(ts)] [1/4] optimize hit 40h timeout (124) — using current best_skill.md for eval"
elif [ "$OPT_RC" -ne 0 ]; then
  echo "[$(ts)] [1/4] optimize exited rc=$OPT_RC — continuing to eval with whatever best_skill exists"
else
  echo "[$(ts)] [1/4] optimize completed cleanly"
fi

if [ -f "$BEST_SKILL" ]; then
  echo "[$(ts)] [1/4] best_skill.md ready ($(wc -c <"$BEST_SKILL") chars)"
else
  echo "[$(ts)] [1/4] !! no best_skill.md produced — falling back to seed skill for 'best' eval"
  BEST_SKILL="$SEED_SKILL"
fi

# ---------- eval helper (skip if done) ----------
run_eval(){
  local name="$1" skill="$2" tag="$3"
  local out="outputs/${tag}"
  if [ -f "$out/eval_summary.json" ]; then
    echo "[$(ts)] [${name}] SKIP (eval_summary.json exists in $out)"
    return
  fi
  echo "[$(ts)] [${name}] eval skill=$skill -> $out"
  : > empty_skill.md
  "$PY" -u scripts/eval_only.py --config "$CFG" --skill "$skill" --split test \
    --target_backend qwen_chat --target_model /model/DeepSeek-V3.2 \
    --data_path "$DATA" --split_mode ratio --split_ratio "$RATIO" \
    --max_turns 1 --workers 8 \
    --out_root "$out" --cfg-options $DATA_OPTS \
    > "$out.eval.log" 2>&1
  echo "[$(ts)] [${name}] done rc=$? see $out/eval_summary.json"
}

# ---------- 2) noskill ----------
echo "[$(ts)] [2/4] noskill full test"
: > empty_skill.md
run_eval "noskill" "$(pwd)/empty_skill.md" "appsplus_full_noskill"

# ---------- 3) seed ----------
echo "[$(ts)] [3/4] seed skill full test"
run_eval "seed" "$SEED_SKILL" "appsplus_full_seed"

# ---------- 4) best ----------
echo "[$(ts)] [4/4] best skill full test"
run_eval "best" "$BEST_SKILL" "appsplus_full_best"

echo "[$(ts)] === ALL DONE ==="
echo "[$(ts)] summaries:"
for t in appsplus_full_noskill appsplus_full_seed appsplus_full_best; do
  echo "--- $t ---"; cat "outputs/$t/eval_summary.json" 2>/dev/null || echo "(missing)"
done
