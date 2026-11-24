#!/usr/bin/env bash
# 配对评测: 预算纪律 patch 是否降低截断率 (issue #53 追加9/10).
# 产物落 <standalone>/outputs/paired_budget_s0/ (gitignore, 机器重启不丢); /tmp 只放临时脚本.
# key 只在运行期从 gitignored 的 configs/models_aimp.local.yaml 读取, 从不打印/落盘.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$HERE"
export PYTHONUNBUFFERED=1

FLASH_KEY=$(python3 -c "import yaml;print(yaml.safe_load(open('$ROOT/../configs/models_aimp.local.yaml'))['providers']['deepseek-v4-flash']['api_key'])")
export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER || true
export TARGET_BACKEND="qwen_chat" OPTIMIZER_BACKEND="qwen_chat"
export TARGET_DEPLOYMENT="deepseek-v4-flash" OPTIMIZER_DEPLOYMENT="deepseek-v4-flash"
# 预算恒定 = 截断率可比 (不加倍重试)
export QWEN_CHAT_MAX_TOKENS=16384 B_MAX_COMPLETION_TOKENS=16384 B_HARD_MAX_TOKENS=16384 B_MAX_RETRY_ON_TRUNC=0
export QWEN_CHAT_STREAM=1 QWEN_CHAT_TIMEOUT_SECONDS=300 EVAL_TEMP=0.3
unset FLASH_KEY

OUT="$ROOT/outputs/paired_budget_s0"
python3 paired_budget_s0_v1.py \
  --data "$ROOT/data/aime24.jsonl" \
  --s0 "$ROOT/data/skill_aime_initial_v2.md" \
  --s1 "$OUT/candidates/demo5_budget_focus.md" --s1_label demo5_budget_focus \
  --s2 "$OUT/candidates/demo6_nofinal_focus.md" --s2_label demo6_nofinal_focus \
  --k 20 --seeds 2 --n_parallel 24 --out_dir "$OUT/run1" "$@"
