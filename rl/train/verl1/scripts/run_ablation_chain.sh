#!/bin/bash
# 机制消融: 4 个 leave-one-out ablation, 串行跑 (28RPM 全局硬限, 不能并行).
# 配置完全对齐主结果 (run_20260730_111306, +16.1pp, hash 12633201):
#   iters=2 eval_k=3 K=4 train_n=5 init=v2(b98a9abd10) s0_cache=k5(1-3-6)
# FULL control = 主结果 skill 12633201 (已 4-seed eval = 0.4623), 无需重跑.
# 顺序: A2(改动最大) -> A1 -> A3 -> A4. 无 set -e (一个崩了其他仍跑, 共享 bug 会快速暴露).
cd /data/zhuldz/lunwen
source lunwen/bin/activate
OUT=rl/train/verl1/outputs/skill_evolver_math_ablation
mkdir -p "$OUT"
COMMON="--iters 2 --eval_k 3 --K 4 --train_n 5 --out_dir $OUT"

echo "=== [$(date)] A2 ablate_failure START (最易出错, 先跑) ==="
python -u rl/train/verl1/skill_evolution/skill_evolver_math.py $COMMON --ablate_failure > "$OUT/run_ablate_failure.log" 2>&1
echo "=== [$(date)] A2 ablate_failure DONE ==="

echo "=== [$(date)] A1 ablate_gepa START ==="
python -u rl/train/verl1/skill_evolution/skill_evolver_math.py $COMMON --ablate_gepa > "$OUT/run_ablate_gepa.log" 2>&1
echo "=== [$(date)] A1 ablate_gepa DONE ==="

echo "=== [$(date)] A3 ablate_consolidate START ==="
python -u rl/train/verl1/skill_evolution/skill_evolver_math.py $COMMON --ablate_consolidate > "$OUT/run_ablate_consolidate.log" 2>&1
echo "=== [$(date)] A3 ablate_consolidate DONE ==="

echo "=== [$(date)] A4 ablate_antideg START ==="
python -u rl/train/verl1/skill_evolution/skill_evolver_math.py $COMMON --ablate_antideg > "$OUT/run_ablate_antideg.log" 2>&1
echo "=== [$(date)] A4 ablate_antideg DONE ==="
echo "=== [$(date)] ALL 4 ABLATIONS DONE ==="
