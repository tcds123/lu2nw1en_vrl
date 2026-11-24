#!/usr/bin/env bash
# Launch shim: AIME24 5-arm skill 评测 (none/s0/d1/d2/d3) on deepseek-v4-flash.
#
# 目的 (用户 2026-11-09): 复盘 §6 "harness 与模型贡献混淆" 的本地补跑 ——
#   学生从 V3.2 (单次 API) 换成 flash, 同一 5 臂、同一 cap, 看 D1/D2/D3 相对 none/s0
#   的排序是否在换学生后保持. 只是 skill 文本注入, 不分叉代码路径.
#
# 复用: for_review_aime_standalone/scripts/eval_d_skills_on_aime24.py (原样, 不改).
#   该脚本本身已是: system = SYSTEM_TMPL.format(skill=<skill 全文>) → chat_target(...),
#   即 "skill 直接当文本注入", 端点/模型全由 QWEN_CHAT_* 决定.
#
# ── flash 接入 (与 launch_flash_lcb_s0.sh 同款) ───────────────────────────────
#   key 运行时从 lunwen 本地 yaml 读 (不落盘明文), base_url/model/RPM 显式覆盖.
#   AIGC_USER 必须为空: flash OpenAI-compat 端点不带 AIGC-USER 路由头.
#   cap: --max-tokens 显式传 → token_limit=min(该值, QWEN_CHAT_MAX_TOKENS).
#        故 QWEN_CHAT_MAX_TOKENS 必须 >= 目标 cap, 否则被静默钳回.
#
# 用法:
#   bash launch_flash_aime24_dskills.sh                 # 默认 5 臂, cap 32768, 截断清零主跑
#   CAP=65536 ONLY_TRUNCATED=<上一轮 results.json> bash launch_flash_aime24_dskills.sh
#   ARMS=none,d2 CONCURRENCY=1 bash launch_flash_aime24_dskills.sh   # 小样试跑
set -euo pipefail

PKG="${PKG_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LUNWEN_YAML="${LUNWEN_YAML:-$PKG/../configs/models_aimp.local.yaml}"
if [[ ! -f "$LUNWEN_YAML" ]]; then echo "ERR: $LUNWEN_YAML 不存在 (flash key 来源)" >&2; exit 1; fi

FLASH_KEY=$(python3 -c "
import yaml
c=yaml.safe_load(open('$LUNWEN_YAML'))
print(c['providers']['deepseek-v4-flash']['api_key'])
" 2>/dev/null) || { echo "ERR: 从 yaml 读 flash key 失败" >&2; exit 1; }
[[ -n "$FLASH_KEY" ]] || { echo "ERR: flash key 为空" >&2; exit 1; }

CAP="${CAP:-32768}"
export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER
export QWEN_CHAT_MAX_TOKENS="$CAP"     # 与 --max-tokens 一致, 避免 min() 钳制
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_MAX_RPM="${RPM:-20}"  # flash 配额 20 RPM
export QWEN_CHAT_BURST="2"
# 2026-11-11 issue #53: 失败全落在超时线 (600s) 附近的慢请求; 提到 1800s 覆盖长尾 (实测成功最慢 178.9s).
export QWEN_CHAT_TIMEOUT_SECONDS="${TIMEOUT:-900}"  # 2026-11-11: 网关 ~300s 硬失败, 超时再大也救不回; 900s 只防客户端卡死 (issue #53)
# 2026-11-12 issue #53 追加4: 流式请求默认开. 非流式请求受网关 ~300s 硬预算约束 (超时即 HTTP 500,
# 与模型是否在跑无关), 是此前 "失败全落在超时线" 的根因; 流式 60/60 探针全部成功且长推理完整返回.
export QWEN_CHAT_STREAM="${QCHAT_STREAM:-1}"
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

ARMS="${ARMS:-none,s0,d1,d2,d3}"
CONCURRENCY="${CONCURRENCY:-1}"  # 2026-11-11: 并发非主因, 但串行便于定位超时; 需要提速可显式设 2-4
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-/data/zhuldz/lunwen/rl/train/verl1/outputs/flash_aime24_dskills/${ONLY_TRUNCATED:+rerun_}${TS}}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

cd "$PKG"
echo "=== [flash-AIME24-5arm] → $OUT_DIR ===" | tee "$LOG"
echo "    model=deepseek-v4-flash cap=$CAP arms=$ARMS concurrency=$CONCURRENCY" | tee -a "$LOG"
if [[ -n "${ONLY_TRUNCATED:-}" ]]; then
  echo "    only-truncated=$ONLY_TRUNCATED" | tee -a "$LOG"
fi

if [[ $# -gt 0 ]]; then
  echo "ERR: 本 launcher 只认环境变量, 不接收命令行参数 (收到: $*)" >&2
  echo "     例: CAP=65536 ARMS=none,d2 bash $0   (issue #54: 旧版会静默丢弃参数并真跑)" >&2
  exit 2
fi
ARGS=(--arms "$ARMS" --max-tokens "$CAP" --concurrency "$CONCURRENCY" --repeats 1 --out "$OUT_DIR")
if [[ -n "${ONLY_TRUNCATED:-}" ]]; then
  ARGS+=(--only-truncated "$ONLY_TRUNCATED" --only-status "${ONLY_STATUS:-truncated}")
fi
if [[ -n "${SEED_FILTER:-}" ]];     then ARGS+=(--seed-filter "$SEED_FILTER"); fi
if [[ -n "${DRY_RUN:-}" ]];         then ARGS+=(--dry-run); fi

if [[ -n "${DRY_RUN:-}" ]]; then
  python scripts/eval_d_skills_on_aime24.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
  exit 0
fi
python scripts/eval_d_skills_on_aime24.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== analyze (截断/配对 McNemar) ===" | tee -a "$LOG"
if [[ -n "${ONLY_TRUNCATED:-}" ]]; then
  python scripts/analyze_truncation_free.py "$ONLY_TRUNCATED" "$OUT_DIR/results.json" 2>&1 | tee -a "$LOG"
else
  python scripts/analyze_truncation_free.py "$OUT_DIR/results.json" 2>&1 | tee -a "$LOG"
fi
