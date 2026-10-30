#!/usr/bin/env bash
# SkillOpt run wrapper for the APPS-Plus (interview) code benchmark.
#
# APPS-Plus is stdin/stdout competitive programming — harder than mbpp/humaneval
# single-function tasks, giving skill optimization real headroom. Uses the
# appsplus env (subprocess stdin/stdout verifier, no Docker).
#
# Env vars (all have defaults):
#   DIFFICULTY   : APPS_Plus difficulty subset   (default: interview)  -> introductory/competition/all
#   RATIO        : train:val:test split          (default: 2:1:7)
#   TAG          : label for output dir          (default: appsplus_<difficulty>_<ratio>)
#
# Usage:
#   bash run_appsplus.sh smoke                 # no-skill, 5 test items (pipeline check)
#   bash run_appsplus.sh noskill [N]           # no-skill baseline, N test items (default: all test split)
#   bash run_appsplus.sh evalskill <skill.md> [N]
#   bash run_appsplus.sh optimize [epochs]
# Examples:
#   bash run_appsplus.sh noskill 50            # 50-item baseline to check headroom
#   RATIO=1:2:7 bash run_appsplus.sh optimize 4
set -euo pipefail
cd "$(dirname "$0")"

PY=/data/zhuldz/venv-skillopt/bin/python
CFG=configs/appsplus/default.yaml
DATA=/data/zhuldz/lunwen/data/APPS_Plus/data/v1
DATASET_GLOB=data_with_id.json
DIFFICULTY=${DIFFICULTY:-interview}
RATIO=${RATIO:-2:1:7}
TAG=${TAG:-appsplus_${DIFFICULTY}_$(echo "$RATIO" | tr ':' '-')}

# qwen backend via env (Midea DeepSeek-V3.2). API key read from $QWEN_CHAT_API_KEY.
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

DATA_OPTS="model.backend=qwen_chat model.target_backend=qwen_chat model.optimizer_backend=qwen_chat env.data_path=$DATA env.split_mode=ratio env.split_ratio=$RATIO env.split_dir= env.dataset_glob=$DATASET_GLOB env.difficulty_filter=$DIFFICULTY"

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
    echo "Unknown mode: $MODE"; echo "Usage: bash run_appsplus.sh {smoke|noskill|evalskill|optimize}"; exit 1
    ;;
esac
