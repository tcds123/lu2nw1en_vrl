#!/usr/bin/env bash
# Q4 正式 protocol: 5 臂配对评测 (none/s0/d1/d2/d3) on deepseek-v4-flash (2026-11-13).
# protocol: cap=16384, retry_on_trunc=0, temp=0.3, k=20, seeds=2, n_parallel=24, stream=1, timeout=300, RPM=20
# 产物: outputs/paired_five_arm_20261113/<run>/ ; key 只在运行期从 gitignored yaml 读, 不打印/不落盘.
# 用法: bash scripts/run_paired_five_arm_20261113.sh            # 固定 run1 (可续跑, CACHED)
#       DRY_RUN=1 bash ...                                      # 只打印 PLAN
#       ARMS=none,s0 DATA=... OUT=... bash ...                  # 换子集/换数据源 (Q2 用)
#       CAP=4096 ARMS=none,s0,s1,s2 CAPS=s1=...,s2=... bash ... # Q1: 降 cap 扫描 (每 cap 单独 RUN)
set -euo pipefail
PKG="${PKG_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LUNWEN_YAML="${LUNWEN_YAML:-$PKG/../configs/models_aimp.local.yaml}"

ARMS="${ARMS:-none,s0,d1,d2,d3}"
DATA="${DATA:-$PKG/data/aime24.jsonl}"
OUT="${OUT:-$PKG/outputs/paired_five_arm_20261113}"
RUN="${RUN:-run1}"
CAP="${CAP:-16384}"
CAPS="${CAPS:-,}"   # 额外 skill 文件: s1=/path/a.md,s2=/path/b.md (Q1 cap 扫描用)
K="${K:-20}"
SEEDS="${SEEDS:-2}"
PAR="${PAR:-24}"
REF="${REF:-none}"

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
# 恒定预算 = 截断率可比; 关掉截断自动扩预算重试
export QWEN_CHAT_MAX_TOKENS="$CAP" B_MAX_COMPLETION_TOKENS="$CAP" B_HARD_MAX_TOKENS="$CAP" B_MAX_RETRY_ON_TRUNC=0
# STREAM=0 实测: stream 下 20 并发吞吐仅 ~0.7 调用/秒 (平均延迟 ~45s, 长样本 180s+) → 6000 次需 2.4h。
# 非 stream 不改变生成分布, 只改变读超时语义; 为压掉 2026-11-12 的 #53 (非 stream 长输出 500), 读超时提到 900s。
export QWEN_CHAT_STREAM="${STREAM:-0}" QWEN_CHAT_TIMEOUT_SECONDS="${TIMEOUT:-900}" QWEN_CHAT_MAX_RPM="${RPM:-20}" QWEN_CHAT_BURST=2
export QWEN_CHAT_TRUNC_RETRY=0  # issue #58: backend 层截断加宽重试必须同时关, 否则 16384 截断被藏成 stop@32768
export EVAL_TEMP=0.3
unset FLASH_KEY

mkdir -p "$OUT/$RUN"
cd "$PKG/code"
python3 paired_five_arm_v1.py --data "$DATA" --arms "$ARMS" --k "$K" --seeds "$SEEDS" \
  --n_parallel "$PAR" --ref "$REF" --caps "$CAPS" --out_dir "$OUT/$RUN" ${DRY_RUN:+--dry}
