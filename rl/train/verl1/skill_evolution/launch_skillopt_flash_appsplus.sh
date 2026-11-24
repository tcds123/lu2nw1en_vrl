#!/usr/bin/env bash
# Launch shim: SkillOpt ReflACTTrainer on APPS_Plus env — deepseek-v4-flash.
#
# 目的 (task #29, 用户 "换大数据集"): 验证 flash + gated slow_update 能否在
# APPS_Plus (val/test 远超 ClassEval val=10 饱和点) 让 skill-text 优化 Δ 跳出噪声.
#
# vs V3.2 判死 run (appsplus_full_optimize, 2026-07, 30h 21步 未跑完) 三处差异:
#   1. model V3.2→flash           (B1 已证强模型翻转 LCB 负结论)
#   2. cap 4096→16384             (同 B1 口径)
#   3. slow_update_gate_with_selection false→true  ⭐ THE FIX (防 force-accept 膨胀退化)
# + 采样三重 cap (train80/val100/test300) 把 80步30h 压到 ~8步2h 且跳小样本陷阱.
#
# ── 采样说明 ──────────────────────────────────────────────────────────────────
# APPS_Plus interview=3950, ratio 2:1:7 seed42 → train=790/val=395/test=2765.
# sel_env_num=100 / test_env_num=300 在 ratio 模式取**前 N** (确定性, base.py:504-505).
# 即取 train[:80] / val[:100] / test[:300]. seed42 确定性 → 可复现, test_env_num=0 可扩全量复测.
# 大数据集意义在 eval 信号分辨率 (val/test 大), 非 train 算力; 8 步够优化 (B1 证).
#
# ── AIGC_USER ─────────────────────────────────────────────────────────────────
# flash OpenAI-compat 端点 (llm-openai-api) 禁止 AIGC-USER header.
# qwen_backend call-time 读 os.environ.get(QWEN_CHAT_AIGC_USER), shell unset 即可.
# ENABLE_THINKING=false → 后端不发 chat_template_kwargs (无 thinking 干扰).
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
export QWEN_CHAT_MAX_TOKENS="16384"   # cap (与 yaml 两处 16384 配合)
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_ENABLE_THINKING="false"
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="${QWEN_CHAT_MAX_RPM:-20}"   # flash 配额 20 RPM
export QWEN_CHAT_BURST="${QWEN_CHAT_BURST:-2}"

# ── backend 冻结: import 时读 TARGET/OPTIMIZER_BACKEND. shell export 早于任何 python import.
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

SKILLOPT="/data/zhuldz/skillopt"
cd "$SKILLOPT"
export PYTHONPATH="$SKILLOPT:${PYTHONPATH:-}"

RUN_TAG="${RUN_TAG:-appsplus-skillopt-flash-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/${RUN_TAG}}"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/train.log"

echo "=== [APPS_Plus] flash SkillOpt → $OUT_ROOT ==="
echo "    backend=deepseek-v4-flash  cap=16384  RPM=20  single-shot"
echo "    config=configs/appsplus/default_flash.yaml  epochs=4 seed=42"
echo "    split=ratio 2:1:7 seed42 (train=790/val=395/test=2765)"
echo "    cap: train_size=80 (8 steps) | sel_env_num=100 | test_env_num=300"
echo "    slow_update_gate_with_selection=true (gated, 非 V3.2 force-accept)"
echo "    对照 V3.2 判死: full-test noskill 0.6564 > best 0.6341 (-2.2pp, force-accept 膨胀)"
echo "    对照 B1 LCB: flash 翻转 V3.2 负结论 +4.9pp (强模型 + gated 拿正信号)"

python scripts/train.py \
  --config configs/appsplus/default_flash.yaml \
  --out_root "$OUT_ROOT/train" \
  --skill_init /data/zhuldz/skillopt/skillopt/envs/appsplus/skills/initial.md \
  --seed 42 \
  2>&1 | tee "$LOG"

echo "=== done. outputs under $OUT_ROOT ==="
echo "    best_skill: $OUT_ROOT/train/best_skill.md"
echo "    summary:    $OUT_ROOT/train/summary.json"
echo "    test_eval:  $OUT_ROOT/train/test_eval*/"
