#!/usr/bin/env bash
# 首版"防截断 skill"自主进化 (2026-11-16).
# 差别于 run_evolve.sh:
#   1. 起点 = v2 S₀ + "Budget & Commit Discipline" 段 (data/skills_v2/trunc_v1.md);
#      起点里已埋两条预算规则, 用来检验自主进化能否在"已有防截断诉求"之上继续压长度/减循环,
#      而不是从零发现"要交卷". 纯 v2 S₀ 起点的 A/B 见 paired 两臂 eval.
#   2. B = deepseek-v4-flash (与 5 臂表同模型, 20 RPM); 旧 run_evolve.sh 走 _env 默认 DeepSeek-V3.2.
#   3. 规模压到首版: iters=4, K=3, eval_k=3, n_parallel=6 (20RPM 下 6 并发 ≈ 满速, 长样本留余量).
# 产物: outputs/skill_evolver_trunc_20261116/<run>/ ; 日志 logs/.
# 用法: bash scripts/run_evolve_trunc_v1.sh [RUN_NAME]
#       DRY=1 bash ...   # 只打印配置不跑
set -euo pipefail
PKG="${PKG_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LUNWEN_YAML="${LUNWEN_YAML:-$PKG/../configs/models_aimp.local.yaml}"
RUN_NAME="${1:-run1}"
INIT_SKILL="${INIT_SKILL:-$PKG/data/skills_v2/trunc_v1.md}"
OUT_DIR="${OUT_DIR:-$PKG/outputs/skill_evolver_trunc_20261116/$RUN_NAME}"
ITERS="${ITERS:-4}"; K="${K:-3}"; EVAL_K="${EVAL_K:-3}"; PAR="${PAR:-6}"
CAP="${CAP:-16384}"

[[ -f "$INIT_SKILL" ]] || { echo "ERR: 起点 skill 不存在: $INIT_SKILL" >&2; exit 1; }

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

echo "▶ ARGS: init=$INIT_SKILL iters=$ITERS K=$K eval_k=$EVAL_K n_parallel=$PAR cap=$CAP out=$OUT_DIR"
if [[ -n "${DRY:-}" ]]; then
  cd "$PKG/code" && python -c "import _env; _env.require_key(); _env.print_config()"
  exit 0
fi

mkdir -p "$OUT_DIR" "$PKG/logs"
LOG="$PKG/logs/evolve_trunc_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "▶ 日志: $LOG"
cd "$PKG/code"
exec python -u skill_evolver_aime_v2.py \
  --init_skill "$INIT_SKILL" \
  --iters "$ITERS" --K "$K" --eval_k "$EVAL_K" --n_parallel "$PAR" \
  --train_n 18 --shuffle_seed 0 \
  --out_dir "$OUT_DIR" 2>&1 | tee "$LOG"
