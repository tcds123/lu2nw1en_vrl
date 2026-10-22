#!/usr/bin/env bash
# Launch AIME CodeAgent eval (Paradigm 2: Program-of-Thoughts, smolagents CodeAgent).
#
# FAIRNESS CONTRACT — identical to the single-call AIME25 baseline (eval_aime25_multiseed.py):
#   • backend  = DeepSeek-V3.2 (same model, same key)
#   • data     = AIME25 (30 held-out problems)
#   • eval_k=3, seeds=3, EVAL_TEMP=0.3  (matches aime25_test baseline exactly)
#   • skills   = s0 + evolver (same skill files whose hashes = aime25_test cache)
#   • baseline = aime25_test single-call cache (per_{name}_{hash}_seed{N}.json), NOT re-run
#   • ONLY difference vs single-call = CodeAgent PoT loop (write Python, run in
#     persistent interpreter, final_answer(N) terminates). Opposite paradigm to ReAct.
#
# Rate-limit fairness: CodeAgent calls go through the OpenAI SDK, NOT skillopt's urllib
# path. We import skillopt's _TokenBucket via _get_rate_limiter() and acquire() before
# each SDK call, so CodeAgent traffic serializes under the SAME 28-RPM quota as single-
# call would use. Run this eval ALONE (no concurrent skillopt/react) to avoid two
# consumers on one bucket.
#
# Usage:
#   bash launch_aime_codeagent.sh                 # default: AIME25, s0+evolver, k=3 seeds=3
#   RPM=14 bash launch_aime_codeagent.sh          # polite share if skillopt still runs
#   SKILLS="s0,evolver,v2" bash launch_aime_codeagent.sh   # add v2 skill as 3rd data point
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
# CodeAgent builds its own model per worker; the rate limiter is process-global so
# parallel agents still serialize on the 28-RPM bucket. 8 workers is enough to keep
# the bucket saturated (each worker blocks on acquire() when the bucket is empty).
WORKERS="${WORKERS:-8}"
OUT_DIR="${OUT_DIR:-/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/aime25_codeagent}"
# single-call baseline cache dir (matches current skill hashes c11d7dc0 / f00fa794 / f4837ca8)
SINGLE_DIR="${SINGLE_DIR:-/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/aime25_test}"

TS="$(date +%Y%m%d-%H%M%S)"
echo "=== [aime-codeagent PoT] data=$DATA skills=$SKILLS k=$EVAL_K seeds=$SEEDS max_steps=$MAX_STEPS rpm=$QWEN_CHAT_MAX_RPM ==="
echo "=== [aime-codeagent PoT] out=$OUT_DIR single_dir=$SINGLE_DIR ==="

mkdir -p "$OUT_DIR"
python -u eval_aime_codeagent_multiseed.py \
  --data "$DATA" \
  --skills "$SKILLS" \
  --eval_k "$EVAL_K" --seeds "$SEEDS" --max_steps "$MAX_STEPS" \
  --max_workers "$WORKERS" \
  --out_dir "$OUT_DIR" \
  --single_dir "$SINGLE_DIR" \
  2>&1 | tee "$OUT_DIR/codeagent_run_${TS}.log"

echo "=== done -> $OUT_DIR/codeagent_eval.json ==="
