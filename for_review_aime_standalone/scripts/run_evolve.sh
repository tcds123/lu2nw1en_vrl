#!/usr/bin/env bash
# 正式进化 (复现 run#3 配置: EFL 涌现 facet + facet-tree + expand + group-admit 门控).
# 规模: AIME24 train 18 / val 12, 10 轮 x 4 候选, eval_k=3, 28 rpm. 约 6-8 小时, ~4870 次 API 调用.
# 与 results/run3_group_admit_20261014 的 args 逐项一致 (n_diag=8 / patience=3 / consolidate_every=3 /
# train_n=18 / shuffle_seed=0 均为默认值, 故未显式写出). 改这些默认值前先看 results/*/evolver_log.json 的 args.
# 运行前会打印本脚本全部参数 → 落进 run_manifest.json 的 args/argv, 事后可核对.
#
# 用法: bash scripts/run_evolve.sh [RUN_NAME]
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
cd "$PKG/code"

python - <<'EOF'
import sys; sys.path.insert(0, ".")
import _env; _env.require_key(); _env.print_config()
EOF

echo "▶ ARGS: --emergent_facets --facet_tree --expand --facet_group_admit --subset_admit iters=${ITERS:-10} eval_k=${EVAL_K:-3} K=${K:-4} train_n=18 shuffle_seed=0"
mkdir -p "$PKG/logs"
LOG="$PKG/logs/${RUN_NAME}.log"
echo "▶ 日志: $LOG"
exec python -u skill_evolver_aime_v2.py \
  --emergent_facets --facet_tree --expand \
  --facet_group_admit --subset_admit \
  --iters "${ITERS:-10}" --eval_k "${EVAL_K:-3}" --K "${K:-4}" --train_n 18 \
  --shuffle_seed 0 \
  --out_dir "$PKG/outputs/skill_evolver_aime_v2/$RUN_NAME" 2>&1 | tee "$LOG"
