#!/usr/bin/env bash
# Launch shim: Phase A 条件路由上界实测验证 (APPS_Plus flash).
#
# 目的 (task #31/#34): 独立脚本实测路由上界假设 — turn1 复用盘上 S₀ baseline (不重调),
# turn2 live-call best_skill.md 只对 S₀-fail 题 → keep-better. 预期 routed hard ≈ 225/300
# (+5.7pp), rescue=17, regression=0, McNemar vs base p<0.001. 不改 SkillOpt 源码.
#
# 复用 launch_skillopt_flash_appsplus.sh 的 env/key 读取模式 (同源 backend, 同参数):
#   - flash key 从 configs/models_aimp.local.yaml 读 (python3 -c, 不打印明文)
#   - unset QWEN_CHAT_AIGC_USER (flash OpenAI-compat 端点禁 AIGC-USER)
#   - RPM=20 (flash 配额), cap=16384, temp=0, no thinking
#
# 输入 (盘上 2026-11-11-v2 run):
#   --baseline_results  test_eval_baseline/results.json  (S₀ 逐题, turn1)
#   --best_skill        best_skill.md                    (候选 skill, turn2)
#   --best_results      test_eval/results.json           (best 逐题, 重建匹配率)
#
# 用法:
#   bash launch_validate_routing.sh
#   OUT_ROOT=/somewhere RUN_TAG=xxx bash launch_validate_routing.sh
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
export QWEN_CHAT_MAX_TOKENS="16384"   # cap (必须与 best run 一致, turn2 才 ≈ best 逐题)
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_ENABLE_THINKING="false"
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="${QWEN_CHAT_MAX_RPM:-20}"   # flash 配额 20 RPM
export QWEN_CHAT_BURST="${QWEN_CHAT_BURST:-2}"

# ── backend 冻结: import 时读 TARGET/OPTIMIZER_BACKEND. shell export 早于任何 python import.
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

# ── 路径 ──────────────────────────────────────────────────────────────────────
SKILLOPT="/data/zhuldz/skillopt"
export PYTHONPATH="$SKILLOPT:${PYTHONPATH:-}"

SCRIPT_DIR="/data/zhuldz/lunwen/rl/train/verl1/skill_evolution"
RUN_DIR="/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/appsplus-skillopt-flash-20261111-v2/train"

BASELINE_RESULTS="$RUN_DIR/test_eval_baseline/results.json"
BEST_SKILL="$RUN_DIR/best_skill.md"
BEST_RESULTS="$RUN_DIR/test_eval/results.json"
CONFIG="$SKILLOPT/configs/appsplus/default_flash.yaml"

for f in "$BASELINE_RESULTS" "$BEST_SKILL" "$BEST_RESULTS" "$CONFIG"; do
  [[ -f "$f" ]] || { echo "ERR: 缺输入 $f" >&2; exit 1; }
done

RUN_TAG="${RUN_TAG:-routing-validate-flash-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/${RUN_TAG}}"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/validate.log"

echo "=== [Phase A] 条件路由上界实测验证 (APPS_Plus flash) → $OUT_ROOT ==="
echo "    backend=deepseek-v4-flash  cap=16384  RPM=20  single-shot  no-feedback turn2"
echo "    config=$CONFIG  split=valid_unseen(test)  test_env_num=300  seed=42"
echo "    turn1 = 复用盘上 S₀ baseline (不重调): $BASELINE_RESULTS"
echo "    turn2 = live-call candidate only on S₀-fail (~92 calls): $BEST_SKILL"
echo "    best  = 盘上 step4 test_eval (重建匹配率对照): $BEST_RESULTS"
echo "    预期 routed hard ≈ 225/300 = 0.7500 (+5.7pp), rescue=17, regression=0, p<0.001"

cd "$SCRIPT_DIR"
python validate_routing_appsplus.py \
  --config "$CONFIG" \
  --baseline_results "$BASELINE_RESULTS" \
  --best_skill "$BEST_SKILL" \
  --best_results "$BEST_RESULTS" \
  --out_root "$OUT_ROOT" \
  2>&1 | tee "$LOG"

echo "=== done. outputs under $OUT_ROOT ==="
echo "    summary:    $OUT_ROOT/routing_validation_result.json"
echo "    routed:     $OUT_ROOT/routing_routed_results.json"
echo "    turn2 dir:  $OUT_ROOT/routing_turn2/"
echo "    log:        $LOG"
