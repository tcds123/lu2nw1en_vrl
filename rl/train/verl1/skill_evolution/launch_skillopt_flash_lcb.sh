#!/usr/bin/env bash
# Launch shim: SkillOpt ReflACTTrainer on LCB env — deepseek-v4-flash (B1).
#
# B1 = V3.2 判死 run (lcb_NEW_optimize_ep4) 的 flash 精确复刻.
# V3.2 判死: test hard 0.193→0.193→0.181 零收益, val+15pp test-1.4pp 过拟合.
# B1 唯二变量 (vs V3.2 config.json): model→flash, cap 4096→16384. 其余全对齐.
# 回答: 同方法/同数据/只换强模型, skill 优化是否从负转正?
#
# ── split 对位 (最关键) ─────────────────────────────────────────────────────────
# S₀ baseline (flash-lcb-s0-result-20261106, test hard=0.6923) 用预生成 split:
#   lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42 (train=41/val=20/test=143)
# SkillOpt 必须 split_mode=split_dir 复用此 split, 否则 test≠143 → 与 S₀ 对比作废.
# split_dir 模式读 <dir>/{train,val,test}/items.json (已含 difficulty/contest_date 字段).
#
# ── cap (与 evolver/crossmodel 的差异) ─────────────────────────────────────────
# SkillOpt 不复用 tf_global run_one (无 4096 字面量). cap 在 yaml 两处:
#   model.qwen_chat_max_tokens + env.max_completion_tokens, 都已 16384 (default_flash.yaml).
# qwen_backend L282 token_limit=min(16384,16384)=16384. 无需字节码补丁.
#
# ── AIGC_USER ─────────────────────────────────────────────────────────────────
# flash OpenAI-compat 端点 (llm-openai-api) 禁止 AIGC-USER header.
# qwen_backend L245 call-time 读 os.environ.get(QWEN_CHAT_AIGC_USER), shell unset 即可
# (SkillOpt 无 precompute 的 setdefault 加回 gotcha). ENABLE_THINKING=false → 后端不发
# extra_body (qwen_backend L294-295 仅 enable_thinking=True 发 chat_template_kwargs).
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

# ── #31 修复 (backend 冻结): backend_config import 时读 TARGET/OPTIMIZER_BACKEND.
# 未设 → 默认 openai_chat(Azure) → ValueError. shell export 早于任何 python import.
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

SKILLOPT="/data/zhuldz/skillopt"
cd "$SKILLOPT"
export PYTHONPATH="$SKILLOPT:${PYTHONPATH:-}"

# ── split 对位: 复用 S₀ 的预生成 split (test=143, 同 flash-lcb-s0) ───────────────
SPLIT_DIR="/data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42"
if [[ ! -f "$SPLIT_DIR/split_manifest.json" ]]; then
  echo "ERR: 预生成 split 不存在: $SPLIT_DIR" >&2; exit 1
fi

RUN_TAG="${RUN_TAG:-lcb-skillopt-flash-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/${RUN_TAG}}"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/train.log"

echo "=== [B1] flash SkillOpt on LCB → $OUT_ROOT ==="
echo "    backend=deepseek-v4-flash  cap=16384 (yaml 两处, 无字节码补丁)  RPM=20"
echo "    config=configs/lcb/default_flash.yaml  epochs=4 seed=42 (对齐 V3.2 判死 run)"
echo "    split=split_dir:$SPLIT_DIR (复用 S₀ 的 test=143, train=41, val=20)"
echo "    对照 S₀ baseline: test hard=0.6923, hard-strat=0.5542(n=83), easy=0.97, med=0.77"
echo "    对照 V3.2 判死: test hard 0.193→0.193→0.181 (val+15pp test-1.4pp 过拟合)"

python scripts/train.py \
  --config configs/lcb/default_flash.yaml \
  --out_root "$OUT_ROOT/train" \
  --skill_init /data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md \
  --seed 42 \
  --cfg-options \
    env.split_mode=split_dir \
    env.split_dir="$SPLIT_DIR" \
  2>&1 | tee "$LOG"

echo "=== done. outputs under $OUT_ROOT ==="
echo "    best_skill: $OUT_ROOT/train/best_skill.md"
echo "    summary:    $OUT_ROOT/train/summary.json"
echo "    test_eval:  $OUT_ROOT/train/test_eval*/"
