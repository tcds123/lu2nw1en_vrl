#!/usr/bin/env bash
# run2 (issue #76): 纯 S₀ 起点 + budget 轴进 focus —— 检验"算法能否自己写出收尾纪律".
#
# 与 run1 (run_evolve_trunc_v1.sh) 的差别只有两处, 其余参数逐项相同:
#   1. 起点从 trunc_v1 (v2 S₀ + 我手写的 Budget & Commit Discipline) 换回纯 S₀
#      data/skill_aime_initial_v2.md (md5 c11d7dc0c8, 无任何 commit/budget 词).
#      这样收尾纪律只能由进化过程自己产生, 其收益才归因于算法而非作者.
#   2. 加 --budget_focus: Budget-Exhausted 轴参与 focus 选择 (改前它是"永远未覆盖"的哨兵,
#      4 轮 focus 全落在答错轴上, evolver 从没被要求降低截断).
# 其余: iters=4 K=3 eval_k=3 n_parallel=6 cap=16384 temp=0.3 train_n=18, B=deepseek-v4-flash.
# 产物: outputs/skill_evolver_trunc_20261116/<run>/ ; 日志 logs/.
# 用法: bash scripts/run_evolve_budget_v2.sh [RUN_NAME]
#       DRY=1 bash ...        # 只打印配置不跑
#       BUDGET_FLAG= bash ... # 对照臂: 同起点但关掉 budget 轴 (隔离"focus 轴"这一唯一变量)
set -euo pipefail
PKG="${PKG_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LUNWEN_YAML="${LUNWEN_YAML:-$PKG/../configs/models_aimp.local.yaml}"
RUN_NAME="${1:-run2}"
INIT_SKILL="${INIT_SKILL:-$PKG/data/skill_aime_initial_v2.md}"
OUT_DIR="${OUT_DIR:-$PKG/outputs/skill_evolver_budget_20261116/$RUN_NAME}"
ITERS="${ITERS:-4}"; K="${K:-3}"; EVAL_K="${EVAL_K:-3}"; PAR="${PAR:-6}"
CAP="${CAP:-16384}"
BUDGET_FLAG="${BUDGET_FLAG---budget_focus}"

[[ -f "$INIT_SKILL" ]] || { echo "ERR: 起点 skill 不存在: $INIT_SKILL" >&2; exit 1; }
# 起点必须是真的 S₀ (无手写收尾纪律), 否则本轮的归因作废
if grep -qiE "commit checkpoint|token budget|running total" "$INIT_SKILL"; then
  echo "ERR: 起点已含收尾纪律词, 会污染归因: $INIT_SKILL" >&2; exit 1
fi

FLASH_KEY=$(python3 -c "
import yaml
print(yaml.safe_load(open('$LUNWEN_YAML'))['providers']['deepseek-v4-flash']['api_key'])
") || { echo "ERR: 读 flash key 失败" >&2; exit 1; }
[[ -n "$FLASH_KEY" ]] || { echo "ERR: flash key 为空" >&2; exit 1; }

export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER || true
export TARGET_BACKEND=qwen_chat OPTIMIZER_BACKEND=qwen_chat
export TARGET_DEPLOYMENT=deepseek-v4-flash OPTIMIZER_DEPLOYMENT=deepseek-v4-flash
export QWEN_CHAT_MAX_TOKENS="$CAP" B_MAX_COMPLETION_TOKENS="$CAP" B_HARD_MAX_TOKENS="$CAP" B_MAX_RETRY_ON_TRUNC=0
export QWEN_CHAT_TRUNC_RETRY=0
export QWEN_CHAT_STREAM=0 QWEN_CHAT_TIMEOUT_SECONDS=900 QWEN_CHAT_MAX_RPM=20 QWEN_CHAT_BURST=2
export EVAL_TEMP=0.3
unset FLASH_KEY

echo "▶ ARGS: init=$INIT_SKILL iters=$ITERS K=$K eval_k=$EVAL_K n_parallel=$PAR cap=$CAP"
echo "▶ FLAGS: ${BUDGET_FLAG:-(none)}  out=$OUT_DIR"
if [[ -n "${DRY:-}" ]]; then
  cd "$PKG/code" && python -c "import _env; _env.require_key(); _env.print_config()"
  exit 0
fi

mkdir -p "$OUT_DIR" "$PKG/logs"
LOG="$PKG/logs/evolve_budget_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "▶ 日志: $LOG"
cd "$PKG/code"
exec python -u skill_evolver_aime_v2.py \
  --init_skill "$INIT_SKILL" \
  --iters "$ITERS" --K "$K" --eval_k "$EVAL_K" --n_parallel "$PAR" \
  --train_n 18 --shuffle_seed 0 \
  --data_driven_focus ${BUDGET_FLAG} \
  --out_dir "$OUT_DIR" 2>&1 | tee "$LOG"
