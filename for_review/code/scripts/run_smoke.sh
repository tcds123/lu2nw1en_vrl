#!/usr/bin/env bash
# 冒烟测试: 极小配置走通完整进化流程 (约 60-100 次 API 调用, 10-25 分钟).
#   - 1 轮进化, 每轮 2 候选, eval_k=1, 只在 val 上评测
#   - 产物写到 outputs/smoke/, 不污染正式 run
# 目的: 交付对象确认 key/网络/依赖全通, 再跑正式实验.
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PKG/code"

python - <<'EOF'
import sys; sys.path.insert(0, ".")
import _env; _env.require_key(); _env.print_config()
EOF

exec python skill_evolver_aime_v2.py \
  --iters 1 --K 2 --eval_k 1 --n_diag 4 --patience 1 \
  --emergent_facets \
  --out_dir "$PKG/outputs/smoke"
