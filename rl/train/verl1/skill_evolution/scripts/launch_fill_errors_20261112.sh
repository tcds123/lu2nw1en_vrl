#!/usr/bin/env bash
# 补跑 5 臂 AIME24 (cap 65536) 中 3 条 HTTP 500 缺失样本 (2026-11-12)。
# 与 launch_flash_aime24_dskills.sh 同款接入; 失败用同一请求顺序退避重试 (最多 ATTEMPTS 次)。
# 用法: bash scripts/launch_fill_errors_20261112.sh   (只认环境变量, 不接命令行参数)
set -euo pipefail
if [[ $# -gt 0 ]]; then echo "ERR: 只认环境变量 (收到: $*)"; exit 2; fi
if pgrep -f fill_errors_20261112.py >/dev/null; then echo "ERR: 已有 fill_errors 在跑, 拒绝重复启动 (会重复耗用 API)"; exit 3; fi

PKG="/data/zhuldz/lunwen/for_review_aime_standalone"
LUNWEN_YAML="/data/zhuldz/lunwen/configs/models_aimp.local.yaml"
PREV="$PKG/outputs/flash_aime24_dskills/20261110_144232/results.json"

FLASH_KEY=$(python3 -c "
import yaml
print(yaml.safe_load(open('$LUNWEN_YAML'))['providers']['deepseek-v4-flash']['api_key'])
") || { echo "ERR: 读 flash key 失败" >&2; exit 1; }
[[ -n "$FLASH_KEY" ]] || { echo "ERR: flash key 为空" >&2; exit 1; }

export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER
export QWEN_CHAT_MAX_TOKENS=65536
export QWEN_CHAT_TEMPERATURE=0.0
export QWEN_CHAT_MAX_RPM="${RPM:-20}"
export QWEN_CHAT_BURST=2
export QWEN_CHAT_TIMEOUT_SECONDS=900
export QWEN_CHAT_STREAM=1
export TARGET_BACKEND=qwen_chat OPTIMIZER_BACKEND=qwen_chat
unset FLASH_KEY

cd "$PKG"
python scripts/fill_errors_20261112.py --prev "$PREV" --max-tokens 65536 \
  --attempts "${ATTEMPTS:-8}" --concurrency "${CONCURRENCY:-1}"
