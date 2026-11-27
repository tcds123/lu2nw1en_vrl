#!/usr/bin/env bash
# RQ3 #19: 启动 GLM-5.2 独立复现线 (S₀ train+val), 与 deepseek-v4-pro 并行 (不共享任何产物)。
#
# 调度变更 (2026-11-27, 用户裁决): 原冻结文档规定「先跑通一个模型再启动第二个」, 现改为并行。
# 研究设计不变: 各自独立 S₀ 评测 / 失败池 / Evolver 训练 / Arm1 / 2765 test / 配对统计。
# 启动要求: API 配额 (本脚本 workers=4, rpm=20, 不与 pro 进程共用 limiter 对象但共用网关) /
#           产物隔离 (只用 outputs/rq3/glm-5.2/) / 预算隔离 (RQ3_MODEL=glm-5.2 单独计数) /
#           协议一致 (不因 pro 中途结果调整 GLM 任何参数)。
#
# key 从 configs/models_aimp.local.yaml 读取, 不打印。
set -euo pipefail
cd /data/zhuldz/lunwen

MODEL=glm-5.2
OUT=outputs/rq3/$MODEL
BASE_URL="https://aimpapi.midea.com/t-aigc/aimp-glm/v1"

# ── 防重复启动 (#127): 同名进程已在跑则直接退出 ──
for pat in 'rq3_a_s0_train[.]py --model glm-5[.]2' 'rq3_b_eval_val[.]py --model glm-5[.]2'; do
  if [ -n "$(pgrep -f "$pat" || true)" ]; then
    echo "[glm] 已在运行: $pat → 跳过" >&2
    exit 0
  fi
done

# 产物隔离断言: 不允许把 GLM 结果写进 pro 目录
case "$OUT" in
  outputs/rq3/glm-5.2) ;;
  *) echo "[glm] BLOCKED: out_dir=$OUT 非隔离目录" >&2; exit 2 ;;
esac
mkdir -p "$OUT"

export QWEN_CHAT_API_KEY
QWEN_CHAT_API_KEY=$(python3 - <<'PY'
import yaml
p = yaml.safe_load(open("configs/models_aimp.local.yaml", encoding="utf-8"))["providers"]
print(p["glm-5.2"]["api_key"])
PY
)
[ -n "$QWEN_CHAT_API_KEY" ] || { echo "[glm] BLOCKED: yaml 无 glm-5.2 key" >&2; exit 2; }

COMMON=(--model "$MODEL" --out-dir "$OUT" --base-url "$BASE_URL"
        --workers 2 --rpm 12 --burst 2 --max-tokens 16384)

echo "[glm] $(date -Is) stage=train S₀ (80 题)" | tee -a "$OUT/launch.log"
python3 -u scripts/rq3_a_s0_train.py "${COMMON[@]}" 2>&1 | tee -a "$OUT/launch.log"

echo "[glm] $(date -Is) stage=val S₀ (100 题)" | tee -a "$OUT/launch.log"
python3 -u scripts/rq3_b_eval_val.py "${COMMON[@]}" 2>&1 | tee -a "$OUT/launch.log"

echo "[glm] $(date -Is) stage=train+val done" | tee -a "$OUT/launch.log"
