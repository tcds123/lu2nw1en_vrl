#!/usr/bin/env bash
# AIME 从头全链 (fresh 版): S0(AIME) -> v2(AIME) -> evolver(AIME) -> AIME25 held-out test.
# 与 run_aime_chain.sh 唯一的区别: 用**全新时间戳 EVAL_OUT**, 避免与 08-05 旧 AIME 阴性结果
#   (skill hash 6f0e7669) 的 eval 缓存 (per_aime25_*_seed*.json) 冲突, 确保"完整重跑"的清洁性.
# ⚠️ 28 RPM 全局共享: 必须在 Run F seed 扩样 (已完成, 10:24) 结束后启动, 否则 429.
#
# 用法: bash rl/train/verl1/run_aime_chain_fresh.sh
set -u
cd /data/zhuldz/lunwen
source lunwen/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
V2_OUT="/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2"
EVO_OUT="/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime"
EVAL_OUT="/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/aime25_test_fresh_${TS}"
mkdir -p "$V2_OUT" "$EVO_OUT" "$EVAL_OUT"

ITERS_V2=2
ITERS_EVO=4
EVAL_K=3
SEEDS=4
TRAIN_N=18
K=4

echo "=========================================================="
echo "[AIME CHAIN FRESH] start $(date)"
echo "  v2 iters=$ITERS_V2  evolver iters=$ITERS_EVO  eval_k=$EVAL_K  seeds=$SEEDS  train_n=$TRAIN_N  K=$K"
echo "  EVAL_OUT(fresh)=$EVAL_OUT"
echo "=========================================================="

echo "[$(date +%H:%M)] === Step 1/3: v2_aime ==="
python -u rl/train/verl1/skill_evolution/tf_global_v2_aime.py \
  --iters $ITERS_V2 --g 4 --eval_k $EVAL_K --train_n $TRAIN_N \
  --out_dir "$V2_OUT" 2>&1 | tee "$V2_OUT/v2_aime_fresh_${TS}.log"
V2_RUN=$(ls -dt "$V2_OUT"/run_* | head -1)
V2_SKILL="$V2_RUN/tf_global_skill_v2_aime.md"
echo "[$(date +%H:%M)] v2 done -> $V2_SKILL"
if [ ! -f "$V2_SKILL" ]; then echo "❌ v2 skill not found, abort"; exit 1; fi

echo "[$(date +%H:%M)] === Step 2/3: evolver_aime (FULL) ==="
python -u rl/train/verl1/skill_evolution/skill_evolver_aime.py \
  --init_skill "$V2_SKILL" \
  --iters $ITERS_EVO --K $K --eval_k $EVAL_K --train_n $TRAIN_N \
  --out_dir "$EVO_OUT" 2>&1 | tee "$EVO_OUT/evolver_aime_fresh_${TS}.log"
EVO_RUN=$(ls -dt "$EVO_OUT"/run_* | head -1)
EVO_SKILL="$EVO_RUN/best_skill_evolved.md"
echo "[$(date +%H:%M)] evolver done -> $EVO_SKILL"
if [ ! -f "$EVO_SKILL" ]; then echo "❌ evolver skill not found, abort"; exit 1; fi

echo "[$(date +%H:%M)] === Step 3/3: AIME25 test eval (fresh outdir) ==="
python -u rl/train/verl1/skill_evolution/eval_aime25_multiseed.py \
  --v2_best "$V2_SKILL" --evolver_best "$EVO_SKILL" \
  --eval_k $EVAL_K --seeds $SEEDS --skills s0,v2,evolver \
  --out_dir "$EVAL_OUT" 2>&1 | tee "$EVAL_OUT/aime25_eval.log"

echo "=========================================================="
echo "[AIME CHAIN FRESH] done $(date)"
echo "  结果: $EVAL_OUT/aime25_eval.json"
echo "=========================================================="