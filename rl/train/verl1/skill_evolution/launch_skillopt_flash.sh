#!/usr/bin/env bash
# Launch shim: run skillopt's ReflACTTrainer on AIME env adapter — flash 跨模型阶段②.
#
# 与 launch_skillopt_aime.sh (V3.2) 的差异:
#   • backend  = deepseek-v4-flash (OpenAI-compat 端点, 无需 AIGC_USER header)
#   • cap      = 16384 (两处: yaml model.qwen_chat_max_tokens + env.max_completion_tokens)
#                 #26/#29: flash 把 CoT 写进 content, 硬题需 ~13k completion tok;
#                 cap 8192 → finish=length → <answer> 未达 → pass=0. 8192 不够, 提到 16384.
#   • config   = configs/aime/default_flash.yaml (V3.2 仍用 default.yaml, 互不污染)
#   • thinking=disabled 非有效兜底 (#29 实测: 关了输出仍长, 仍截断). 故只靠提 cap.
#     qwen_backend.py:294-295 仅 enable_thinking=True 时发 chat_template_kwargs, 无 extra_body —
#     flash OpenAI-compat 端点默认行为无关紧要 (关思考也救不了, 必须给 cap).
#
# 公平对比控制变量: skill_init 仍用 lunwen 的 4-facet AIME S₀ (与 evolver 同起点).
# 温度差异: SkillOpt optimizer/target=0.0 (skillopt default); lunwen evolver EVAL_TEMP=0.3 —
# 这是文档化的协议差异, 不在本 shim 修正.
set -euo pipefail

# ── flash key 从 lunwen 本地 yaml 运行时读取 (不落盘明文) ──────────────────────
LUNWEN_YAML="/data/zhuldz/lunwen/configs/models_aimp.local.yaml"
if [[ ! -f "$LUNWEN_YAML" ]]; then
  echo "ERR: $LUNWEN_YAML 不存在 (flash key 来源)" >&2; exit 1
fi
FLASH_KEY=$(python3 -c "
import yaml,sys
c=yaml.safe_load(open('$LUNWEN_YAML'))
p=c['providers']['deepseek-v4-flash']
print(p['api_key'])
" 2>/dev/null) || { echo "ERR: 从 yaml 读 flash key 失败" >&2; exit 1; }
[[ -n "$FLASH_KEY" ]] || { echo "ERR: flash key 为空" >&2; exit 1; }

# ── deepseek-v4-flash backend (OpenAI-compat 端点) ─────────────────────────────
export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER          # flash OpenAI-compat 端点不需要
export QWEN_CHAT_MAX_TOKENS="16384"   # #26/#29: 8192 在 flash 硬题截断
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_ENABLE_THINKING="false"
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="${QWEN_CHAT_MAX_RPM:-20}"   # flash 配额 20 RPM
export QWEN_CHAT_BURST="${QWEN_CHAT_BURST:-2}"

SKILLOPT="/data/zhuldz/skillopt"
cd "$SKILLOPT"

export PYTHONPATH="$SKILLOPT:${PYTHONPATH:-}"

RUN_TAG="${RUN_TAG:-aime-skillopt-flash-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/skillopt/${RUN_TAG}}"
mkdir -p "$OUT_ROOT"

MODE="${MODE:-train}"   # train | eval | both

if [[ "$MODE" == "train" || "$MODE" == "both" ]]; then
  echo "=== [skillopt-flash] TRAIN on AIME24 (ratio split 6:2:2, cap 16384) → $OUT_ROOT ==="
  echo "    backend=deepseek-v4-flash  cap=16384 (#26/#29)  config=configs/aime/default_flash.yaml"
  python scripts/train.py \
    --config configs/aime/default_flash.yaml \
    --out_root "$OUT_ROOT/train" \
    --skill_init /data/zhuldz/lunwen/data/skill_aime_initial_v2.md \
    --seed 42 \
    2>&1 | tee "$OUT_ROOT/train.log"
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
  FINAL_SKILL="${FINAL_SKILL:-$(find "$OUT_ROOT/train" -name 'skill_final.md' -o -name 'final_skill.md' 2>/dev/null | head -1)}"
  if [[ -z "${FINAL_SKILL:-}" || ! -f "$FINAL_SKILL" ]]; then
    echo "WARN: no final skill found under $OUT_ROOT/train; falling back to S0."
    FINAL_SKILL="$SKILLOPT/skillopt/envs/aime/skills/initial.md"
  fi
  echo "=== [skillopt-flash] EVAL on held-out AIME25, skill=$FINAL_SKILL ==="
  python scripts/eval_only.py \
    --config configs/aime/default_flash.yaml \
    --env.name aime \
    --env.data_path /data/zhuldz/lunwen/data/aime25.jsonl \
    --env.split_mode split_dir \
    --env.split_dir /data/zhuldz/lunwen/data/aime25_eval \
    --env.skill_init "$FINAL_SKILL" \
    --env.out_root "$OUT_ROOT/eval_aime25" \
    2>&1 | tee "$OUT_ROOT/eval_aime25.log"
fi

echo "=== done. outputs under $OUT_ROOT ==="
