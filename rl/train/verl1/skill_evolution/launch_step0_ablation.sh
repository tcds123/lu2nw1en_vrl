#!/usr/bin/env bash
# Launch shim: Step 0 四组配对消融 (task #47).
#
# 跑 C/D 两组各 300 题 (600 次 flash 调用), A/B 复用盘上结果 (0 调用).
# 两主终点 Preservation(208 S0-pass) + Acquisition(92 S0-fail), A=S₀ 参照, Holm-Bonferroni 校正.
#
# 复用 launch_tf_appsplus_routing.sh 的 flash env/key 读取模式 (同源, 同参数):
#   - flash key 从 configs/models_aimp.local.yaml 读 (python3 -c, 不打印明文, 连部分前缀也不)
#   - unset QWEN_CHAT_AIGC_USER (flash OpenAI-compat 端点禁 AIGC-USER)
#   - RPM=20 (flash 配额), cap=16384, temp=0, no thinking
#
# 对等性: C/D 用 ThreadPoolExecutor(8) 同 B 组 (B 的 retention/routed 就是这么跑的)。
#   qwen_backend._TokenBucket 进程级线程安全 → 并发线程数不改变 RPM 上限 (永远 ≤20/min),
#   不引入并发度新变量 (B 组也是 8 线程)。600 次 ~30-45min (用户估算)。
#
# 用法:
#   bash launch_step0_ablation.sh                # 默认全 300 题, 跑 C+D
#   LIMIT=5 ABL_GROUPS=C bash launch_step0_ablation.sh   # 冒烟 (5 题, 只 C)
# ⚠️ 不能用 GROUPS 作环境变量名 — 它是 bash 特殊内置数组 (用户组 GID),
#    GROUPS=C 会被忽略, ${GROUPS} 取 GROUPS[0]=0 → 传给 runner 的是 "0". 用 ABL_GROUPS.
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

# ── deepseek-v4-flash backend (OpenAI-compat 端点) ────────────────────────────
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
SO_RUN="/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/appsplus-skillopt-flash-20261111-v2/train"

S0_CACHE_TEST="$SO_RUN/test_eval_baseline/results.json"   # A组 (300, hard=208)
SPLIT_DIR="$SKILLOPT/data/appsplus_capped_seed42"

for f in "$S0_CACHE_TEST" "$SPLIT_DIR/test/items.json" \
         "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/step0_ablation_20261116/skills/skill_C_no_step0.md" \
         "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/step0_ablation_20261116/skills/skill_D_evidence.md"; do
  [[ -f "$f" ]] || { echo "ERR: 缺输入 $f" >&2; exit 1; }
done

# ── 参数 (env 覆盖) ────────────────────────────────────────────────────────────
LIMIT="${LIMIT:-}"                 # 空 = 全 300; 冒烟设 N
ABL_GROUPS="${ABL_GROUPS:-CD}"      # 默认跑 C+D (⚠️ 避开 bash 内置 GROUPS)
RUN_TAG="${RUN_TAG:-step0-ablation-flash-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/step0_ablation_20261116"
mkdir -p "$OUT_ROOT/results"
LOG="$OUT_ROOT/results/run_${RUN_TAG}.log"

echo "=== [task #47] Step0 四组配对消融 (deepseek-v4-flash) → $OUT_ROOT ==="
echo "    groups=new($ABL_GROUPS)  limit=${LIMIT:-全300}  cap=16384  RPM=20  ThreadPool=8"
echo "    A=S0 cache (复用): $S0_CACHE_TEST (300 题 hard=208)"
echo "    B=retention+routed-fail (复用): 0 调用"
echo "    C=skill_C_no_step0.md  D=skill_D_evidence.md (各 300 新调用)"
echo "    A/B 复用前提已核验 (decompose milestone: B 209=185+24)."

cd "$SCRIPT_DIR"
LIMIT_ARG=""
[[ -n "$LIMIT" ]] && LIMIT_ARG="--limit $LIMIT"
python step0_ablation_runner.py \
  --s0_cache_test "$S0_CACHE_TEST" \
  --split_dir "$SPLIT_DIR" \
  --groups "$ABL_GROUPS" \
  $LIMIT_ARG \
  2>&1 | tee "$LOG"

echo "=== done. outputs under $OUT_ROOT/results ==="
echo "    log: $LOG"
