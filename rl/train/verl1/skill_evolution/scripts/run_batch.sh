#!/usr/bin/env bash
# 分批调度: 跑 N 套完整评测 (进化 run_evolve + 2 skill 测评 run_eval), 支持 5h 滑动窗口限流与断点续跑.
#
# 用于按「请求次数」计费的 API 套餐 (如 Coding Plan): 额度只按次, 不按 token,
# 因此**保持完整思考链、不要压输出**是划算的 (见 README §12 / docs/phase2_issues.md #25).
#
# 用量:   单套 ≈ 5,950 次请求 (进化 ≈ 4,870 + 2 skill × 540; 30 题, k=3, 3 seed, K=4, 10 轮)
# 限流:   --window 5h 内最多 --window_cap 次请求 (默认 5500, 取官方 6 套上限的 90% 留余量)
#
# 用法:
#   bash scripts/run_batch.sh 3                        # 跑 3 套, 默认 5h/5500 次窗口
#   bash scripts/run_batch.sh 3 --window_cap 5800
#   MAX_WALL_H=10 bash scripts/run_batch.sh 10 --window 5h --window_cap 5500
#
# 断点: 每套完成写 outputs/batch/<BATCH_ID>/done_<i>.txt; 重跑同一 BATCH_ID 自动跳过已完成套数.
#       run_eval.sh 本身按 per_*.json 缓存, 中断后重跑不重复计费. run_evolve 目前不支持中途续跑,
#       中断会丢弃该套 (若要节省, 先用 run_evolve.sh 单独跑大的, eval 可随时续).
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PKG"

N_SETS=1; WINDOW_H=5; WINDOW_CAP=5500; MAX_WALL_H="${MAX_WALL_H:-99}"
eval_k=3; seeds=3; iters=10; K=4; stride=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --window)     WINDOW_H="$2"; shift 2;;
    --window_cap) WINDOW_CAP="$2"; shift 2;;
    --eval_k)     eval_k="$2"; shift 2;;
    --seeds)      seeds="$2"; shift 2;;
    --iters)      iters="$2"; shift 2;;
    --K)          K="$2"; shift 2;;
    --stride_h)   stride="$2"; shift 2;;       # 每套开始前错开的小时数 (与别人共享账号时用)
    -h|--help)    sed -n '2,20p' "$0"; exit 0;;
    *)            N_SETS="$1"; shift;;
  esac
done

python - <<'EOF'
import sys; sys.path.insert(0, "code")
import _env; _env.require_key()
EOF

BATCH_ID="${BATCH_ID:-batch_$(date +%Y%m%d_%H%M%S)}"
BD="$PKG/outputs/batch/$BATCH_ID"; mkdir -p "$BD" "$PKG/logs"
NOW_EP=$(date +%s); START_EP=$NOW_EP; WEOF=$((START_EP + MAX_WALL_H * 3600))

if (( $(date +%s) >= WEOF )); then echo "⏰ MAX_WALL_H=$MAX_WALL_H 太小, 不启动"; exit 0; fi
# 每套请求次数 (与 plan.py 口径一致: 30 题, k, K, iters)
per_set=$(python -c "k=$eval_k;s=$seeds;K=$K;it=$iters;print(30*k+it*(12*k+12*k*2+30*k*K+4*K)+20+2*30*k*s)")

echo "BATCH_ID=$BATCH_ID  sets=$N_SETS  per_set≈$per_set req  total≈$((per_set*N_SETS))"
echo "window=${WINDOW_H}h / $WINDOW_CAP req   max_wall=${MAX_WALL_H}h   eval_k=$eval_k seeds=$seeds iters=$iters K=$K"

# sent 记录: 每行 "<epoch> <reqs>"
SENT="$BD/sent.log"; : >> "$SENT"
WINDOW_S=$(python -c "print(int($WINDOW_H*3600))")
sent_in_window() { awk -v now="$(date +%s)" -v w="$WINDOW_S" '$1 > now-w {s+=$2} END {print s+0}' "$SENT"; }

wait_for_window() {
  local need=$1
  while :; do
    local used; used=$(sent_in_window)
    if (( used + need <= WINDOW_CAP )); then
      if (( used > 0 )); then echo "   [window] 已用 $used + 本套 $need <= $WINDOW_CAP"; fi
      return 0
    fi
    # 找到窗口内最早一条过期时间
    local until_ep sleep_s
    # 窗口内最早一条过期于点 1 的记录 (awk 取 min, 不依赖管道)
    until_ep=$(awk -v now="$(date +%s)" -v w="$(python -c "print($WINDOW_H*3600)")" \
      '$1 > now-w { if (m==0 || $1 < m) m=$1 } END {print m+0}' "$SENT")
    sleep_s=$(( until_ep + WINDOW_S - $(date +%s) + 30 ))
    if (( sleep_s < 60 )); then sleep_s=60; fi
    if (( $(date +%s) + sleep_s > WEOF )); then echo "⏳ 窗口等待会超出 MAX_WALL_H 预算 ($used 已用, 需 $need), 停止"; exit 3; fi
    echo "⏳ 5h 窗口已用 $used + 本套 $need > $WINDOW_CAP, 等待 $((sleep_s/60)) min"
    sleep "$sleep_s"
  done
}

for (( i=1; i<=N_SETS; i++ )); do
  if [[ -f "$BD/done_$i.txt" ]]; then echo "↷ set $i/$N_SETS 已完成, 跳过"; continue; fi
  if (( $(date +%s) > WEOF )); then echo "⏰ MAX_WALL_H 到, 停止于 set $i"; exit 0; fi
  wait_for_window "$per_set"
  RUN="s_${BATCH_ID#batch_}_$i"
  echo "▶ set $i/$N_SETS  ($RUN)  $(date '+%F %T')"
  # 评测产物: outputs/eval/aime24_<RUN>/... 与 skill_evolver_aime_v2/<RUN>/best_skill_evolved.md
  bash "$PKG/scripts/run_evolve.sh" "$RUN" > "$PKG/logs/${RUN}_evolve.log" 2>&1
  BEST="outputs/skill_evolver_aime_v2/$RUN/best_skill_evolved.md"
  EVAL_K=$eval_k SEEDS=$seeds RUN_TAG="$RUN" \
    bash "$PKG/scripts/run_eval.sh" "$BEST" > "$PKG/logs/${RUN}_eval.log" 2>&1 || {
      echo "⚠ set $i eval 失败, 见 logs/${RUN}_eval.log (per_*.json 缓存可在重跑时续)"; exit 2; }
  echo "$(date +%s) $per_set" >> "$SENT"
  echo "✓ set $i/$N_SETS 完成 $(date '+%F %T')" > "$BD/done_$i.txt"
  # 零 API 离线汇总 (不影响主流程): outputs/analysis_summary/set_$i/REPORT.md
  python "$PKG/code/scripts/analyze_run.py" "outputs/skill_evolver_aime_v2/$RUN" \
    "outputs/eval/aime24_$RUN" "outputs/eval/aime25_$RUN" \
    --out_dir "outputs/analysis_summary/set_$i" --no_bundle > "$PKG/logs/${RUN}_analysis.log" 2>&1 \
    || echo "   ⚠ 离线分析失败, 见 logs/${RUN}_analysis.log (不影响主流程)"
  echo "   window 已用 $(sent_in_window)/$WINDOW_CAP"
  if (( stride > 0 && i < N_SETS )); then echo "⏸ stride ${stride}h"; sleep $((stride*3600)); fi
done
echo "✅ 全部完成: $N_SETS 套, 见 $BD"
