#!/usr/bin/env bash
# Launch shim: Phase B — 我们 evolver (tf_skill_optimizer) 接 APPS_Plus + 路由.
#
# 目的 (task #31/#35/#36): 把 tf_skill_optimizer.py (LCB) 移植到 APPS_Plus 并给 gate/final
# eval 加条件路由, 让 evolver 在纯 rescue 信号下进化, 看能否找到 rescue>step4 的 skill,
# 最终用路由评测超过 SkillOpt 211/300。不改 SkillOpt 源码 (只读库)。
#
# 复用 launch_validate_routing.sh 的 env/key 读取模式 (同源 flash backend, 同参数):
#   - flash key 从 configs/models_aimp.local.yaml 读 (python3 -c, 不打印明文)
#   - unset QWEN_CHAT_AIGC_USER (flash OpenAI-compat 端点禁 AIGC-USER)
#   - RPM=20 (flash 配额), cap=16384, temp=0, no thinking
#
# 输入 (盘上 2026-11-11-v2 run, 复用 val/test S₀-cache):
#   --s0_cache_val   selection_eval_baseline/results.json  (val S₀, 100 题 hard=58, turn1)
#   --s0_cache_test  test_eval_baseline/results.json       (test S₀, 300 题 hard=208, turn1 + McNemar base)
#   --best_results   test_eval/results.json                (SkillOpt best test 211, McNemar 对照)
#   --s0_cache_train <out_dir>/train_s0_cache.json        (train S₀, 不存在则预计算 80 题, 幂等复用)
#
# 用法:
#   bash launch_tf_appsplus_routing.sh                      # 默认 3 iters, train_n=80, g=4
#   ITERS=1 TRAIN_N=10 G=2 bash launch_tf_appsplus_routing.sh   # 冒烟
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
export QWEN_CHAT_MAX_TOKENS="16384"   # cap (必须, F7 教训: 4096 截断长 CoT → pass=0)
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

S0="$SKILLOPT/skillopt/envs/appsplus/skills/initial.md"
SPLIT_DIR="$SKILLOPT/data/appsplus_capped_seed42"
S0_CACHE_VAL="$RUN_DIR/selection_eval_baseline/results.json"
# test S₀ 参考基线: 2026-11-19 定版为重判版 (hard=215, 见 docs/48_analysis_plan_amendment_20261119.md §2)。
# 原 results.json (208) 只作历史值保留, 不再作为当前评测器下的基线。
S0_CACHE_TEST="$RUN_DIR/test_eval_baseline/results.rejudged_d5_v2.json"
BEST_RESULTS="$RUN_DIR/test_eval/results.json"

for f in "$S0" "$SPLIT_DIR/train/items.json" "$S0_CACHE_VAL" "$S0_CACHE_TEST" "$BEST_RESULTS"; do
  [[ -f "$f" ]] || { echo "ERR: 缺输入 $f" >&2; exit 1; }
done

# ── 参数 (env 覆盖) ────────────────────────────────────────────────────────────
ITERS="${ITERS:-3}"
TRAIN_N="${TRAIN_N:-80}"
G="${G:-4}"
NUM_EXP="${NUM_EXP:-3}"
MAX_POOL="${MAX_POOL:-12}"
ABLATION_ARM="${ABLATION_ARM:-full}"   # #48 训练时消融: full|drop_interface|drop_scaffold|suppress_exp

RUN_TAG="${RUN_TAG:-tf-appsplus-routing-flash-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/${RUN_TAG}}"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"
TRAIN_CACHE="${TRAIN_CACHE:-$OUT_ROOT/train_s0_cache.json}"   # 幂等: #48 各 arm 须共享同一 train S₀-cache (TRAIN_CACHE env 指向共享路径 → 同 fail 集, 控制变量)

echo "=== [Phase B] tf_skill_optimizer APPS_Plus + 路由 (deepseek-v4-flash) → $OUT_ROOT ==="
echo "    iters=$ITERS  train_n=$TRAIN_N  g=$G  cap=16384  RPM=20  ablation_arm=$ABLATION_ARM"
echo "    S₀=$S0 ($(wc -c <"$S0") chars)"
echo "    val cache (复用 v2): $S0_CACHE_VAL (100 题 hard=58)"
echo "    test cache (复用 v2): $S0_CACHE_TEST (300 题 hard=208)"
echo "    train cache: $TRAIN_CACHE (${TRAIN_CACHE:+shared, }不存在则预计算 80 题, ~5min)"
echo "    best 对照: $BEST_RESULTS (SkillOpt 211)"

cd "$SCRIPT_DIR"
python tf_skill_optimizer_appsplus.py \
  --s0 "$S0" \
  --split_dir "$SPLIT_DIR" \
  --s0_cache_val "$S0_CACHE_VAL" \
  --s0_cache_test "$S0_CACHE_TEST" \
  --s0_cache_train "$TRAIN_CACHE" \
  --best_results "$BEST_RESULTS" \
  --iters "$ITERS" \
  --train_n "$TRAIN_N" \
  --g "$G" \
  --num_exp "$NUM_EXP" \
  --max_pool "$MAX_POOL" \
  --ablation_arm "$ABLATION_ARM" \
  --out_dir "$OUT_ROOT" \
  2>&1 | tee "$LOG"

echo "=== done. outputs under $OUT_ROOT ==="
echo "    skill:    $OUT_ROOT/tf_global_skill.md"
echo "    result:   $OUT_ROOT/tf_appsplus_routing_result.json"
echo "    routed:   $OUT_ROOT/test_routed_results.json"
echo "    indep:    $OUT_ROOT/test_independent_results.json"
echo "    log:      $LOG"
