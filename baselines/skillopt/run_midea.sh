#!/usr/bin/env bash
# SkillOpt run wrapper for the Midea DeepSeek-V3.2 API (qwen_chat backend, AIGC-USER patched).
# LiveMathematicianBench (text MCQ, max_turns=1, deterministic verifier) - no Docker needed.
#
# Usage:
#   bash run_midea.sh smoke        # no-skill, 5 test items (pipeline check)
#   bash run_midea.sh noskill [N]  # no-skill baseline, N test items (default: all ~124)
#   bash run_midea.sh optimize [epochs]   # SkillOpt optimization (Step 0 reproduce)
#   bash run_midea.sh evalskill <skill.md> [N]  # eval a specific skill
set -euo pipefail
cd "$(dirname "$0")"

PY=/data/zhuldz/venv-skillopt/bin/python
CFG=configs/livemathematicianbench/default.yaml

# --- qwen backend via env (key 从环境变量读取, 不写入仓库; 见 .env.example) ---
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:?QWEN_CHAT_API_KEY 未设置, 请从 .env.example 复制并 export}"
export QWEN_CHAT_BASE_URL=https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1
export QWEN_CHAT_MODEL=/model/DeepSeek-V3.2
export QWEN_CHAT_AIGC_USER=zhuldz
export QWEN_CHAT_MAX_TOKENS=4096
export QWEN_CHAT_TEMPERATURE=0.0
export TARGET_BACKEND=qwen_chat OPTIMIZER_BACKEND=qwen_chat

# Common cfg-options for LiveMathematicianBench with local data + qwen backend
DATA_OPTS="model.backend=qwen_chat model.target_backend=qwen_chat model.optimizer_backend=qwen_chat env.data_path=data/livemath_raw env.split_mode=ratio env.split_ratio=2:1:7 env.split_dir="

MODE=${1:-smoke}
case "$MODE" in
  smoke)
    : > empty_skill.md
    "$PY" -u scripts/eval_only.py --config "$CFG" --skill empty_skill.md --split test \
      --target_backend qwen_chat --target_model /model/DeepSeek-V3.2 \
      --data_path data/livemath_raw --split_mode ratio --split_ratio 2:1:7 \
      --test_env_num 5 --max_turns 1 --workers 4 \
      --out_root outputs/smoke_noskill --cfg-options $DATA_OPTS
    ;;
  noskill)
    N=${2:-0}
    : > empty_skill.md
    LIMIT=""; [ "${N:-0}" -gt 0 ] 2>/dev/null && LIMIT="--test_env_num $N"
    "$PY" -u scripts/eval_only.py --config "$CFG" --skill empty_skill.md --split test \
      --target_backend qwen_chat --target_model /model/DeepSeek-V3.2 \
      --data_path data/livemath_raw --split_mode ratio --split_ratio 2:1:7 \
      $LIMIT --max_turns 1 --workers 8 \
      --out_root "outputs/noskill_${N:-all}" --cfg-options $DATA_OPTS
    ;;
  optimize)
    EP=${2:-4}
    "$PY" -u scripts/train.py --config "$CFG" \
      --optimizer_backend qwen_chat --target_backend qwen_chat \
      --optimizer_model /model/DeepSeek-V3.2 --target_model /model/DeepSeek-V3.2 \
      --out_root "outputs/optimize_ep${EP}" \
      --cfg-options $DATA_OPTS train.num_epochs=$EP
    ;;
  evalskill)
    SKILL=${2:?need skill.md path}; N=${3:-0}
    LIMIT=""; [ "${N:-0}" -gt 0 ] 2>/dev/null && LIMIT="--test_env_num $N"
    "$PY" -u scripts/eval_only.py --config "$CFG" --skill "$SKILL" --split test \
      --target_backend qwen_chat --target_model /model/DeepSeek-V3.2 \
      --data_path data/livemath_raw --split_mode ratio --split_ratio 2:1:7 \
      $LIMIT --max_turns 1 --workers 8 \
      --out_root "outputs/evalskill_$(basename "$SKILL" .md)" --cfg-options $DATA_OPTS
    ;;
  *) echo "unknown mode: $MODE"; exit 1 ;;
esac
