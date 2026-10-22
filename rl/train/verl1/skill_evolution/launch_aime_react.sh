#!/usr/bin/env bash
# Launch AIME ReAct eval (agent capability = ReAct + execute_code self-debug).
#
# FAIRNESS CONTRACT — identical to the single-call AIME25 baseline (eval_aime25_multiseed.py):
#   • backend  = DeepSeek-V3.2 (same model, same key)
#   • data     = AIME25 (30 held-out problems)
#   • eval_k=3, seeds=3, EVAL_TEMP=0.3  (matches aime25_test baseline exactly)
#   • skills   = s0 + evolver (same skill files whose hashes = aime25_test cache)
#   • baseline = aime25_test single-call cache (per_{name}_{hash}_seed{N}.json), NOT re-run
#   • ONLY difference vs single-call = REACT_SUFFIX + execute_code tool loop
#
# API: shares the 28 RPM global quota with skillopt training. When skillopt is still
# running, launch this with a LOWER RPM (QWEN_CHAT_MAX_RPM) so neither starves; otherwise
# default 28. After skillopt finishes, re-run with full RPM if desired.
#
# Usage:
#   bash launch_aime_react.sh                 # default: AIME25, s0+evolver, k=3 seeds=3
#   RPM=14 bash launch_aime_react.sh          # polite share while skillopt runs
#   SKILLS="s0,evolver,skillopt" bash launch_aime_react.sh   # after skillopt done
set -euo pipefail

cd /data/zhuldz/lunwen/rl/train/verl1/skill_evolution

# ── DeepSeek-V3.2 backend (authorized: DeepSeek + Qwen3-235B only) ──────────
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:?请先设置环境变量 QWEN_CHAT_API_KEY}"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1"
export QWEN_CHAT_MODEL="/model/DeepSeek-V3.2"
export QWEN_CHAT_AIGC_USER="zhuldz"
export QWEN_CHAT_MAX_TOKENS="4096"
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_ENABLE_THINKING="false"
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="${RPM:-28}"
export QWEN_CHAT_BURST="${BURST:-2}"

export PYTHONPATH="/data/zhuldz/skillopt:/data/zhuldz/lunwen/rl/train/verl1/skill_evolution:${PYTHONPATH:-}"

DATA="${DATA:-/data/zhuldz/lunwen/data/aime25.jsonl}"
SKILLS="${SKILLS:-s0,evolver}"
EVAL_K="${EVAL_K:-3}"
SEEDS="${SEEDS:-3}"
MAX_STEPS="${MAX_STEPS:-6}"
WORKERS="${WORKERS:-14}"
OUT_DIR="${OUT_DIR:-/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/aime25_react}"
# single-call baseline cache dir (matches current skill hashes c11d7dc0 / f00fa794)
SINGLE_DIR="${SINGLE_DIR:-/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/aime25_test}"

TS="$(date +%Y%m%d-%H%M%S)"
echo "=== [aime-react] data=$DATA skills=$SKILLS k=$EVAL_K seeds=$SEEDS max_steps=$MAX_STEPS rpm=$QWEN_CHAT_MAX_RPM ==="
echo "=== [aime-react] out=$OUT_DIR single_dir=$SINGLE_DIR ==="

python -u eval_aime_react_multiseed.py \
  --data "$DATA" \
  --skills "$SKILLS" \
  --eval_k "$EVAL_K" --seeds "$SEEDS" --max_steps "$MAX_STEPS" \
  --max_workers "$WORKERS" \
  --out_dir "$OUT_DIR" \
  --single_dir "$SINGLE_DIR" \
  2>&1 | tee "$OUT_DIR/react_run_${TS}.log"

echo "=== done -> $OUT_DIR/react_eval.json ==="
