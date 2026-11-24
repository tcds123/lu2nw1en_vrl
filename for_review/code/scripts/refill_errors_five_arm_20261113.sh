#!/usr/bin/env bash
# 补跑 5 臂配对 run 的 error 样本 (2026-11-13). 与 run_paired_five_arm_20261113.sh 同环境变量。
# 用法: RUN_DIR=outputs/paired_five_arm_20261113/run1 bash scripts/refill_errors_five_arm_20261113.sh
set -euo pipefail
PKG="${PKG_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LUNWEN_YAML="${LUNWEN_YAML:-$PKG/../configs/models_aimp.local.yaml}"
if [[ $# -gt 0 ]]; then echo "ERR: 只认环境变量 (收到: $*)"; exit 2; fi
DIR="${RUN_DIR:-$PKG/outputs/paired_five_arm_20261113/run1}"
[[ -d "$DIR" ]] || { echo "ERR: 目录不存在 $DIR" >&2; exit 1; }
if pgrep -f refill_errors_five_arm.py >/dev/null; then echo "ERR: 已有补跑在跑"; exit 3; fi

FLASH_KEY=$(python3 -c "
import yaml
print(yaml.safe_load(open('$LUNWEN_YAML'))['providers']['deepseek-v4-flash']['api_key'])
") || { echo "ERR: 读 flash key 失败" >&2; exit 1; }
[[ -n "$FLASH_KEY" ]] || { echo "ERR: flash key 为空" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER || true
export TARGET_BACKEND=qwen_chat OPTIMIZER_BACKEND=qwen_chat
export TARGET_DEPLOYMENT=deepseek-v4-flash OPTIMIZER_DEPLOYMENT=deepseek-v4-flash
export QWEN_CHAT_MAX_TOKENS=16384 B_MAX_COMPLETION_TOKENS=16384 B_HARD_MAX_TOKENS=16384 B_MAX_RETRY_ON_TRUNC=0
export QWEN_CHAT_STREAM=1 QWEN_CHAT_TIMEOUT_SECONDS=600 QWEN_CHAT_MAX_RPM="${RPM:-10}" QWEN_CHAT_BURST=2
export QWEN_CHAT_TRUNC_RETRY=0  # issue #58
export EVAL_TEMP=0.3
unset FLASH_KEY

cd "$PKG/code"
python3 refill_errors_five_arm.py --dir "$DIR" --attempts "${ATTEMPTS:-8}"
