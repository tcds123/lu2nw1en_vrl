#!/usr/bin/env bash
# Launch shim: D2 修复冒烟验收 (task #50 / codex 2026-11-14).
#
# 最小闭合验证 rollout_group rewriter 是否真正执行 + 温度是否真正生效。
# env/key 读取模式与 launch_tf_appsplus_routing.sh 完全一致 (同源 flash backend):
#   - flash key 从 configs/models_aimp.local.yaml 运行时读取 (不落盘明文)
#   - unset QWEN_CHAT_AIGC_USER (flash OpenAI-compat 端点禁 AIGC-USER)
#   - RPM=20 (flash 配额), cap=16384, temp=0, no thinking
#   - TF_D2_DEBUG=1 开启临界区温度轨迹 (供冒烟解析 applied/saved/restored)
#
# 用法:
#   bash launch_smoke_d2.sh
set -euo pipefail

# ── flash key 从 lunwen 本地 yaml 运行时读取 (不落盘明文) ──────────────────────
LUNWEN_YAML="/data/zhuldz/lunwen/configs/models_aimp.local.yaml"
if [[ ! -f "$LUNWEN_YAML" ]]; then
  echo "ERR: $LUNWEN_YAML 不存在 (flash key 来源)" >&2; exit 1
fi
FLASH_KEY=$(python3 -c "
import yaml
c=yaml.safe_load(open('$LUNWEN_YAML'))
print(c['providers']['deepseek-v4-flash']['api_key'])
" 2>/dev/null) || { echo "ERR: 从 yaml 读 flash key 失败" >&2; exit 1; }
[[ -n "$FLASH_KEY" ]] || { echo "ERR: flash key 为空" >&2; exit 1; }

# ── deepseek-v4-flash backend (OpenAI-compat 端点) ─────────────────────────────
export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER          # flash OpenAI-compat 端点不需要
export QWEN_CHAT_MAX_TOKENS="16384"   # cap (必须, F7 教训)
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_ENABLE_THINKING="false"
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="20"
export QWEN_CHAT_BURST="2"

# ── backend 冻结: import 时读 TARGET/OPTIMIZER_BACKEND, shell export 早于 python import.
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

# ── D2-debug: 临界区温度轨迹 (env-gated, 生产不触发) ──────────────────────────
export TF_D2_DEBUG="1"

# ── 路径 ──────────────────────────────────────────────────────────────────────
SKILLOPT="/data/zhuldz/skillopt"
export PYTHONPATH="$SKILLOPT:${PYTHONPATH:-}"

SCRIPT_DIR="/data/zhuldz/lunwen/rl/train/verl1/skill_evolution"
cd "$SCRIPT_DIR"

# ── venv ──────────────────────────────────────────────────────────────────────
source /data/zhuldz/lunwen/lunwen/bin/activate

echo "=== [smoke-D2] rollout_group rewriter 修复验收 (deepseek-v4-flash, g=2, 1 题) ==="
echo "    backend=deepseek-v4-flash  cap=16384  RPM=20  TF_D2_DEBUG=1"
echo "    验收 5 点: (1) ≥1 真实 rewriter API (2) 温度匹配 (3) validity (4) 下游 eval (5) 临界区还原"
echo

python smoke_d2_rewriter_fix.py 2>&1
RC=$?

echo
echo "=== smoke exit=$RC (0=all-pass, 1=has-fail, 2=abort-0-real-candidate) ==="
echo "    证据: $SCRIPT_DIR/_smoke_d2_out/smoke_d2_request.log"
echo "          $SCRIPT_DIR/_smoke_d2_out/smoke_d2_candidates.json"
exit $RC
