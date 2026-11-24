#!/usr/bin/env bash
# 旁路进化 (issue #81 建议2): 用跨 run 蒸馏出的「防截断 base rule」当起点, 开防截断门.
#
# 与 run_evolve_trunc_v1.sh 的差别只有两处 —— 其余参数逐一相同 (iters/K/eval_k/par/cap/RPM/
# train_n/shuffle_seed), 这样主路径与旁路分支的差异是单一变量:
#   1. --init_skill = trunc_bank 产出的注入版 skill (S₀ + "## Commit Discipline (anti-truncation)")
#   2. --trunc_gate 打开 (建议1 的逐轮防截断检验), 让"变坏"的候选不进 best
#
# 前置 (一次性, 已跑过则跳过):
#   cd code && python trunc_bank.py collect --runs <run_dir...>
#   cd code && python trunc_bank.py distill --runs <run_dir...>     # 1 次 API
#   cd code && python trunc_bank.py apply --skill <S0.md> --out <bank>/side_channel_init.md
#
# 用法: bash scripts/run_evolve_sidechannel_trunc.sh [RUN_NAME]
#       DRY=1 bash ...        # 只打印配置不跑
#       INIT_SKILL=/path/x.md bash ...   # 换注入文件 (例如对另一版 S₀ 注入)
set -euo pipefail
PKG="${PKG_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LUNWEN_YAML="${LUNWEN_YAML:-$PKG/../configs/models_aimp.local.yaml}"
RUN_NAME="${1:-side1}"
BANK="${BANK:-$PKG/outputs/trunc_bank}"
INIT_SKILL="${INIT_SKILL:-$BANK/side_channel_init.md}"
OUT_DIR="${OUT_DIR:-$PKG/outputs/skill_evolver_sidechannel_20261117/$RUN_NAME}"
ITERS="${ITERS:-4}"; K="${K:-3}"; EVAL_K="${EVAL_K:-3}"; PAR="${PAR:-6}"
CAP="${CAP:-16384}"

[[ -f "$INIT_SKILL" ]] || { echo "ERR: 注入版 skill 不存在: $INIT_SKILL (先跑 trunc_bank.py apply)" >&2; exit 1; }
grep -q "Commit Discipline (anti-truncation)" "$INIT_SKILL" || {
  echo "ERR: $INIT_SKILL 里没有 base rule 小节, 旁路分支失去单一变量" >&2; exit 1; }

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

echo "▶ SIDE-CHANNEL: init=$INIT_SKILL iters=$ITERS K=$K eval_k=$EVAL_K n_parallel=$PAR cap=$CAP out=$OUT_DIR"
echo "▶ 门: --trunc_gate (整体率容差 +1%+1样本, 配对 ≤25% 题上升)"
if [[ -n "${DRY:-}" ]]; then
  cd "$PKG/code" && python -c "import _env; _env.require_key(); _env.print_config()"
  exit 0
fi

mkdir -p "$OUT_DIR" "$PKG/logs"
LOG="$PKG/logs/evolve_sidechannel_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "▶ 日志: $LOG"
cd "$PKG/code"
exec python -u skill_evolver_aime_v2.py \
  --init_skill "$INIT_SKILL" \
  --iters "$ITERS" --K "$K" --eval_k "$EVAL_K" --n_parallel "$PAR" \
  --train_n 18 --shuffle_seed 0 \
  --trunc_gate \
  --out_dir "$OUT_DIR" 2>&1 | tee "$LOG"
