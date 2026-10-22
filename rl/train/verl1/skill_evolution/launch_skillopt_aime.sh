#!/usr/bin/env bash
# Launch shim: run skillopt's ReflACTTrainer on the AIME env adapter.
#
# FAIRNESS CONTRACT — matches the lunwen project's own algorithm exactly:
#   • backend  = DeepSeek-V3.2 (qwen_chat) — same as lunwen B/C model
#   • train    = AIME24 (30 problems) via ratio split (18 train / 6 val / 6 gate-test)
#   • held-out = AIME25 (30 problems) via eval_only.py — generalization test
#   • optimizer/target temperature = 0.0 (skillopt default); the lunwen evolver
#     uses EVAL_TEMP=0.3 k-voting — a deliberate, documented protocol difference.
#
# The QWEN_CHAT_* env vars configure the qwen_chat backend (see
# skillopt/model/qwen_backend.py: configure_qwen_chat reads these when the yaml
# omits them). API key is NOT committed — it lives only in this local shim.
set -euo pipefail

# ── DeepSeek-V3.2 backend (authorized: DeepSeek + Qwen3-235B only) ──────────
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:?请先设置环境变量 QWEN_CHAT_API_KEY}"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1"
export QWEN_CHAT_MODEL="/model/DeepSeek-V3.2"
export QWEN_CHAT_AIGC_USER="zhuldz"
export QWEN_CHAT_MAX_TOKENS="4096"
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_ENABLE_THINKING="false"
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="${QWEN_CHAT_MAX_RPM:-28}"
export QWEN_CHAT_BURST="${QWEN_CHAT_BURST:-2}"

SKILLOPT="/data/zhuldz/skillopt"
cd "$SKILLOPT"

export PYTHONPATH="$SKILLOPT:${PYTHONPATH:-}"

RUN_TAG="${RUN_TAG:-aime-skillopt-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/skillopt/${RUN_TAG}}"
mkdir -p "$OUT_ROOT"

MODE="${MODE:-train}"   # train | eval | both

if [[ "$MODE" == "train" || "$MODE" == "both" ]]; then
  echo "=== [skillopt-AIME] TRAIN on AIME24 (ratio split 6:2:2) → $OUT_ROOT ==="
  # Shared S0: both methods start from lunwen's 4-facet AIME S0 (公平对比控制变量).
  # NOTE: train.py uses strict parse_args() + flat legacy flags (mapped to dotted
  # paths via _LEGACY_TO_STRUCTURED). Dotted --env.X/--train.X flags crash. Use flat.
  python scripts/train.py \
    --config configs/aime/default.yaml \
    --out_root "$OUT_ROOT/train" \
    --skill_init /data/zhuldz/lunwen/data/skill_aime_initial_v2.md \
    --seed 42 \
    2>&1 | tee "$OUT_ROOT/train.log"
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
  # Held-out generalization: AIME25. The final evolved skill is auto-written by
  # the trainer to $OUT_ROOT/train/<run>/skill_final.md — point eval_only.py at it.
  FINAL_SKILL="${FINAL_SKILL:-$(find "$OUT_ROOT/train" -name 'skill_final.md' -o -name 'final_skill.md' 2>/dev/null | head -1)}"
  if [[ -z "${FINAL_SKILL:-}" || ! -f "$FINAL_SKILL" ]]; then
    echo "WARN: no final skill found under $OUT_ROOT/train; falling back to S0."
    FINAL_SKILL="$SKILLOPT/skillopt/envs/aime/skills/initial.md"
  fi
  echo "=== [skillopt-AIME] EVAL on held-out AIME25, skill=$FINAL_SKILL ==="
  python scripts/eval_only.py \
    --config configs/aime/default.yaml \
    --env.name aime \
    --env.data_path /data/zhuldz/lunwen/data/aime25.jsonl \
    --env.split_mode split_dir \
    --env.split_dir /data/zhuldz/lunwen/data/aime25_eval \
    --env.skill_init "$FINAL_SKILL" \
    --env.out_root "$OUT_ROOT/eval_aime25" \
    2>&1 | tee "$OUT_ROOT/eval_aime25.log"
fi

echo "=== done. outputs under $OUT_ROOT ==="
