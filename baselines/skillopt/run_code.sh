#!/usr/bin/env bash
# SkillOpt run wrapper for code benchmarks via the Midea DeepSeek-V3.2 API.
# Discriminative experiment: does SkillOpt's skill-optimization method have
# headroom on CODE data, using our assert-based unit-test verifier?
#
# Switchable via env vars (all have defaults):
#   DATASET_GLOB  : file prefix in $DATA dir  (default: mbpp_pro*)  -> humaneval_pro* etc.
#   RATIO         : train:val:test split      (default: 2:1:7)      -> 1:2:7 to enlarge val
#   TAG           : label for output dir      (default: derived from DATASET_GLOB+RATIO)
#
# Usage:
#   bash run_code.sh smoke        # no-skill, 5 test items (pipeline check)
#   bash run_code.sh noskill [N]  # no-skill baseline, N test items (default: all)
#   bash run_code.sh evalskill <skill.md> [N]   # eval a specific skill
#   bash run_code.sh optimize [epochs]          # SkillOpt optimization loop
# Examples:
#   DATASET_GLOB=humaneval_pro* bash run_code.sh noskill
#   RATIO=1:2:7 bash run_code.sh optimize 4
set -euo pipefail
cd "$(dirname "$0")"

PY=/data/zhuldz/venv-skillopt/bin/python
BENCH=${BENCH:-mbpppro}   # mbpppro | appsplus | lcb
case "$BENCH" in
  appsplus)
    CFG=configs/appsplus/default.yaml
    DATA=/data/zhuldz/lunwen/data/APPS_Plus/data/v1
    DATASET_GLOB=${DATASET_GLOB:-data_with_id.json}
    ;;
  lcb)
    CFG=configs/lcb/default.yaml
    DATA=/data/zhuldz/lunwen/data/LCB
    DATASET_GLOB=${DATASET_GLOB:-test*.jsonl}
    ;;
  mbpppro|*)
    BENCH=mbpppro
    CFG=configs/mbpppro/default.yaml
    DATA=/data/zhuldz/lunwen/data/CodeEval-Pro/dataset
    DATASET_GLOB=${DATASET_GLOB:-mbpp_pro*}
    ;;
esac
RATIO=${RATIO:-2:1:7}
TAG=${TAG:-${BENCH}_$(echo "$RATIO" | tr ':' '-')}

# qwen backend via env (Midea DeepSeek-V3.2). API key read from $QWEN_CHAT_API_KEY.
# Export here so run_code.sh is self-contained (run_midea.sh already exports it;
# without this, runs launched from a fresh shell get HTTP 403 "鉴权出错-无法获取code").
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:?QWEN_CHAT_API_KEY 未设置, 请从 .env.example 复制并 export}"
export QWEN_CHAT_BASE_URL=https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1
export QWEN_CHAT_MODEL=/model/DeepSeek-V3.2
export QWEN_CHAT_AIGC_USER=zhuldz
export QWEN_CHAT_MAX_TOKENS=4096
export QWEN_CHAT_TEMPERATURE=0.0
export TARGET_BACKEND=qwen_chat OPTIMIZER_BACKEND=qwen_chat
# DeepSeek-V3.2 API hard cap = 30 req/min. Process-wide token bucket keeps
# concurrent workers strictly under it (default 28 RPM) + 429 backoff in
# qwen_backend.py retries. Disable with QWEN_CHAT_MAX_RPM=0.
export QWEN_CHAT_MAX_RPM=${QWEN_CHAT_MAX_RPM:-28}
export QWEN_CHAT_BURST=${QWEN_CHAT_BURST:-2}

DATA_OPTS="model.backend=qwen_chat model.target_backend=qwen_chat model.optimizer_backend=qwen_chat env.data_path=$DATA env.split_mode=ratio env.split_ratio=$RATIO env.split_dir= env.dataset_glob=$DATASET_GLOB"
[ "$BENCH" = "appsplus" ] && DATA_OPTS="$DATA_OPTS env.difficulty_filter=interview"
# LCB: pass contamination filter (contest_date_after) through env.
if [ "$BENCH" = "lcb" ]; then
  DATA_OPTS="$DATA_OPTS env.contest_date_after=${CONTEST_DATE_AFTER:-2025-01-01} env.difficulty_filter=${DIFFICULTY_FILTER:-}"
  [ -n "${CONTEST_DATE_BEFORE:-}" ] && DATA_OPTS="$DATA_OPTS env.contest_date_before=${CONTEST_DATE_BEFORE}"
fi

MODE=${1:-smoke}
case "$MODE" in
  smoke)
    : > empty_skill.md
    "$PY" -u scripts/eval_only.py --config "$CFG" --skill empty_skill.md --split test \
      --target_backend qwen_chat --target_model /model/DeepSeek-V3.2 \
      --data_path "$DATA" --split_mode ratio --split_ratio "$RATIO" \
      --test_env_num 5 --max_turns 1 --workers 4 \
      --out_root "outputs/${TAG}_smoke" --cfg-options $DATA_OPTS
    ;;
  noskill)
    N=${2:-0}
    : > empty_skill.md
    LIMIT=""; [ "${N:-0}" -gt 0 ] 2>/dev/null && LIMIT="--test_env_num $N"
    "$PY" -u scripts/eval_only.py --config "$CFG" --skill empty_skill.md --split test \
      --target_backend qwen_chat --target_model /model/DeepSeek-V3.2 \
      --data_path "$DATA" --split_mode ratio --split_ratio "$RATIO" \
      $LIMIT --max_turns 1 --workers 8 \
      --out_root "outputs/${TAG}_noskill_${N:-all}" --cfg-options $DATA_OPTS
    ;;
  evalskill)
    SKILL=${2:?need skill.md path}; N=${3:-0}
    LIMIT=""; [ "${N:-0}" -gt 0 ] 2>/dev/null && LIMIT="--test_env_num $N"
    "$PY" -u scripts/eval_only.py --config "$CFG" --skill "$SKILL" --split test \
      --target_backend qwen_chat --target_model /model/DeepSeek-V3.2 \
      --data_path "$DATA" --split_mode ratio --split_ratio "$RATIO" \
      $LIMIT --max_turns 1 --workers 8 \
      --out_root "outputs/${TAG}_evalskill_$(basename "$SKILL" .md)" --cfg-options $DATA_OPTS
    ;;
  optimize)
    EP=${2:-4}
    "$PY" -u scripts/train.py --config "$CFG" \
      --optimizer_backend qwen_chat --target_backend qwen_chat \
      --optimizer_model /model/DeepSeek-V3.2 --target_model /model/DeepSeek-V3.2 \
      --out_root "outputs/${TAG}_optimize_ep${EP}" \
      --cfg-options $DATA_OPTS train.num_epochs=$EP
    ;;
  *)
    echo "Unknown mode: $MODE"; echo "Usage: bash run_code.sh {smoke|noskill|evalskill|optimize}"; exit 1
    ;;
esac
