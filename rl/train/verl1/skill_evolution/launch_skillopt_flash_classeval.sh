#!/usr/bin/env bash
# Launch shim: SkillOpt ReflACTTrainer on ClassEval env — deepseek-v4-flash.
#
# ClassEval (FudanSELab/ClassEval, ACL 2024): 100 class / 410 methods / avg 33
# test cases per class. Process-internal unittest (importlib +
# unittest.TextTestRunner in child process) → zero Docker → reward = 单测通过率.
# 这是 "最方便的数据集" 先看 skill 效果的 run.
#
# 对照: B1 (flash-lcb, 单文件单函数, test hard S0=0.6923 → Δ+4.9pp). 预期
# class 级多方法协作任务上 skill 杠杆更大 (hard 段提升更显著).
#
# ── cap ──────────────────────────────────────────────────────────────────────
# SkillOpt 不复用 tf_global run_one. cap 在 yaml 两处: model.qwen_chat_max_tokens
# + env.max_completion_tokens, 都已 16384. qwen_backend token_limit=min(16384,16384).
#
# ── AIGC_USER ─────────────────────────────────────────────────────────────────
# flash OpenAI-compat 端点 (llm-openai-api) 禁止 AIGC-USER header. shell unset 即可.
# ENABLE_THINKING=false → 后端不发 extra_body.
set -euo pipefail

# ── CRITICAL: 用 lunwen venv python, 不能用系统 python3 ─────────────────────────
# ClassEval 测试代码依赖第三方库 (bs4/docx/gensim/nltk/PyPDF2/reportlab), 只装在
# lunwen venv 里. evaluator 用 multiprocessing.Process 跑子进程, 子进程继承父解释器
# (sys.executable). 若 shim 调系统 python3 → 5 题 (34/38/44/69/93) import_error.
# venv python 下 ground-truth baseline = 96/100 (vs 系统 python3 的 91/100).
PYTHON="/data/zhuldz/lunwen/lunwen/bin/python"
[[ -x "$PYTHON" ]] || { echo "ERR: venv python 不存在: $PYTHON" >&2; exit 1; }

# nltk_data 持久缓存目录: ClassEval_44 (HtmlUtil) 经 gensim 间接触发 nltk lazy-load.
# 网络通时下载一次缓存于此; 子进程 (multiprocessing) 经 NLTK_DATA 找到, 不再每次 fetch.
NLTK_DATA_DIR="/data/zhuldz/lunwen/data/nltk_data"
mkdir -p "$NLTK_DATA_DIR"
export NLTK_DATA="$NLTK_DATA_DIR"

# ── flash key 从 lunwen 本地 yaml 运行时读取 (不落盘明文) ──────────────────────
LUNWEN_YAML="/data/zhuldz/lunwen/configs/models_aimp.local.yaml"
if [[ ! -f "$LUNWEN_YAML" ]]; then
  echo "ERR: $LUNWEN_YAML 不存在 (flash key 来源)" >&2; exit 1
fi
FLASH_KEY=$("$PYTHON" -c "
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

# ── backend 冻结: import 时读 TARGET/OPTIMIZER_BACKEND, 未设→默认 openai_chat→报错.
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

SKILLOPT="/data/zhuldz/skillopt"
cd "$SKILLOPT"
export PYTHONPATH="$SKILLOPT:${PYTHONPATH:-}"

RUN_TAG="${RUN_TAG:-classeval-skillopt-flash-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/${RUN_TAG}}"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/train.log"

echo "=== flash SkillOpt on ClassEval → $OUT_ROOT ==="
echo "    backend=deepseek-v4-flash  cap=16384  RPM=20"
echo "    config=configs/classeval/default_flash.yaml  epochs=4 seed=42"
echo "    data=ClassEval_data.json (100 class, 2:1:7 ratio split)"
echo "    exec=unittest importlib (child proc, 10s timeout, no Docker)"
echo "    对照 B1: flash-lcb test S0=0.6923 Δ+4.9pp (单函数); 预期 class 级 skill 杠杆更大"

# ── split_mode: 默认 ratio (100 题 2:1:7 train/val/test).
# 若已预生成 split (复用 S0 baseline), 用 SPLIT_DIR 环境变量指向该目录 + 改 split_mode.
SPLIT_DIR="${SPLIT_DIR:-}"
# EXTRA_CFG: 追加任意 --cfg-options override (空格分隔的 key=value), 例如
#   EXTRA_CFG="evaluation.gate_metric=soft" RUN_TAG=... bash launch...sh
# 这些 override 与 split override 合并进同一个 --cfg-options.
read -ra _EXTRA <<< "${EXTRA_CFG:-}"
CFG_OPTS=()
if [[ -n "$SPLIT_DIR" ]]; then
  if [[ ! -f "$SPLIT_DIR/split_manifest.json" ]]; then
    echo "ERR: 预生成 split 不存在: $SPLIT_DIR" >&2; exit 1
  fi
  CFG_OPTS+=(--cfg-options env.split_mode=split_dir env.split_dir="$SPLIT_DIR")
  echo "    split=split_dir:$SPLIT_DIR (复用预生成 split)"
else
  echo "    split=ratio (2:1:7, seed=42) — 用 yaml 默认"
fi
if [[ ${#_EXTRA[@]} -gt 0 ]]; then
  # 首次启动 --cfg-options flag (split 模式已含则只追加 value)
  if [[ ${#CFG_OPTS[@]} -eq 0 ]]; then
    CFG_OPTS+=(--cfg-options)
  fi
  CFG_OPTS+=("${_EXTRA[@]}")
  echo "    extra_cfg: ${_EXTRA[*]}"
fi

# ratio 模式: yaml 已含 split_mode=ratio/split_ratio=2:1:7/split_seed=42, 无需 --cfg-options.
# 仅当有 split_dir 或 EXTRA_CFG 时才追加 --cfg-options (空 array 展开 bare flag 会报 argparse 错).
"$PYTHON" scripts/train.py \
  --config configs/classeval/default_flash.yaml \
  --out_root "$OUT_ROOT/train" \
  --skill_init "${SKILL_INIT:-/data/zhuldz/skillopt/skillopt/envs/classeval/skills/initial.md}" \
  --seed 42 \
  "${CFG_OPTS[@]}" \
  2>&1 | tee "$LOG"

echo "=== done. outputs under $OUT_ROOT ==="
echo "    best_skill: $OUT_ROOT/train/best_skill.md"
echo "    summary:    $OUT_ROOT/train/summary.json"
echo "    test_eval:  $OUT_ROOT/train/test_eval*/"
