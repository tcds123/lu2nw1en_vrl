#!/usr/bin/env bash
# §10.1 多 run 方差实验 — on/off 各 N=2 独立演化 + 2-seed test eval.
# (2026-09-21 #3 降级 N=3→N=2: 环境重启≠会话断开, nohup 也防不住环境重建;
#  iters=4 不动保 on 臂信号, 砍 N 主要砍 OFF 死 run. on1/off1 已 iters=4 完美兼容, 只补 on2/off2.)
# 串行 (rate limiter process-local, 并发超 30 RPM cap). 交替 on/off 顺序 (中途断点两臂均衡).
# resume-capable: evolver out_dir 已有 evolver_log.json 则跳过该 run.
# 每 run 完成后立即跑 2-seed shuffle test eval, 产物独立落盘.
set -u
cd /data/zhuldz/lunwen
source lunwen/bin/activate

S0=/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md
SPLIT=/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42
CACHE=rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json
EVOLVER=rl/train/verl1/skill_evolution/skill_evolver_math.py
EVAL=rl/train/verl1/skill_evolution/eval_test_multiseed_shuffle.py
OUTBASE=rl/train/verl1/outputs/skill_evolver_math_poolfilter
TS=20260920_multirun

LOG=rl/train/verl1/outputs/multirun_5101.log
: > "$LOG"
echo "[$(date '+%F %T')] §10.1 multi-run variance START (on/off ×2, iters=4)" | tee -a "$LOG"

run_one () {
  local arm=$1 rid=$2
  local POOLFLAG=""
  [ "$arm" = "on" ] && POOLFLAG="--pool_filter"
  local OD="${OUTBASE}_${arm}_multirun${rid}_${TS}"

  # ⚠️ evolver 把产物写在 ${OD}/run_<ts>/ 子目录 (L3677 无条件套时间戳, 无 flag 关).
  #    编排脚本判据 + eval 调用都用 ${OD}/ 根路径 → 须 evolver 完成后把 run_<ts>/* 提到 ${OD}/.
  # resume 分两态 (run 级粒度, 不用 evolver --resume):
  #  (a) 已完成 (${OD}/evolver_log.json + best_skill_evolved.md 都在, 已从 run_<ts> 提升) → 整 run 跳过
  #  (b) 未完成 (含: 未开始 / 中途 kill 只留 run_<ts>/evolver_log.json) → 清半成品重跑 fresh
  # 不用 evolver --resume: facet_tree 路径下 --resume 失效 (循环无视 resume_from 从 iter1 重跑 +
  # iters 记录重复, 见 docs/phase2_issues.md 2026-09-20 条). 中途 kill 的 run 整 run 重跑,
  # 丢失该 run 已完成 iter (最多 ~4.7h) 但保证 run 间可比 + 演化轨迹正确.
  local BEST_MD="${OD}/best_skill_evolved.md"
  if [ -f "${OD}/evolver_log.json" ] && [ -f "$BEST_MD" ]; then
    echo "[$(date '+%F %T')] [${arm}${rid}] evolver already done, SKIP -> ${OD}" | tee -a "$LOG"
  else
    # 清半成品: 残留的 run_<ts>/ 子目录 (未提升) + 已提升的产物, 避免 fresh run 落盘混淆
    if [ -d "$OD" ]; then
      echo "[$(date '+%F %T')] [${arm}${rid}] incomplete run, 清半成品重跑 fresh -> ${OD}" | tee -a "$LOG"
      rm -rf "${OD}"/run_* 2>/dev/null
      rm -f "${OD}"/evolver_log.json "${OD}"/best_skill_evolved.md \
            "${OD}"/skill_iter*.md "${OD}"/best_skill_iter*.md \
            "${OD}"/archive_iter*.json "${OD}"/tree_iter*.json \
            "${OD}"/archive_final.json 2>/dev/null
    fi
    echo "[$(date '+%F %T')] [${arm}${rid}] evolver START (fresh) -> ${OD} (pool_filter=${arm})" | tee -a "$LOG"
    python -u "$EVOLVER" \
      --s0 "$S0" --init_skill "$S0" \
      --split_dir "$SPLIT" --s0_cache "$CACHE" \
      --iters 4 --K 4 --eval_k 3 --n_diag 8 --train_n 35 \
      --facet_tree --expand $POOLFLAG \
      --out_dir "$OD" >> "${OD}.evolver.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] [${arm}${rid}] evolver DONE rc=$rc -> ${OD}" | tee -a "$LOG"
    # evolver 无条件写 ${OD}/run_<ts>/ (L3677), 提升到 ${OD}/ 让判据 + eval 对得上
    local sub
    sub=$(ls -d "${OD}"/run_*/ 2>/dev/null | head -1)
    if [ -n "$sub" ]; then
      echo "[$(date '+%F %T')] [${arm}${rid}] 提升 ${sub}/* -> ${OD}/" | tee -a "$LOG"
      mv "${sub}"* "${OD}/" 2>/dev/null
      rmdir "$sub" 2>/dev/null
    fi
    if [ $rc -ne 0 ] || [ ! -f "$BEST_MD" ]; then
      echo "[$(date '+%F %T')] [${arm}${rid}] ⚠️ evolver FAILED (rc=$rc, no best_skill). CONTINUE next (下次整 run 重跑)." | tee -a "$LOG"
      return $rc
    fi
  fi

  # 2-seed shuffle test eval (独立 out_dir, resume: 跳过已有 summary)
  local SKILL="${OD}/best_skill_evolved.md"
  local EVAL_OD="${OUTBASE}_${arm}_multirun${rid}_${TS}_eval"
  local EVAL_FLAG=""
  if [ "$arm" = "on" ]; then EVAL_FLAG="--on $SKILL"; else EVAL_FLAG="--off $SKILL"; fi
  if [ -f "${EVAL_OD}/summary.json" ] || ls "${EVAL_OD}"/per_${arm}_*_seed*_shuf.json >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] [${arm}${rid}] eval already done, SKIP -> ${EVAL_OD}" | tee -a "$LOG"
  else
    echo "[$(date '+%F %T')] [${arm}${rid}] eval START (2-seed shuffle) -> ${EVAL_OD}" | tee -a "$LOG"
    python -u "$EVAL" \
      --skills "s0,${arm}" $EVAL_FLAG \
      --eval_k 3 --seeds 2 --out_dir "$EVAL_OD" >> "${OD}.eval.log" 2>&1
    local rc2=$?
    echo "[$(date '+%F %T')] [${arm}${rid}] eval DONE rc=$rc2 -> ${EVAL_OD}" | tee -a "$LOG"
  fi
  return 0
}

# 交替顺序: on1 off1 on2 off2 (中途断点两臂均衡)
for rid in 1 2; do
  run_one on  $rid
  run_one off $rid
done

echo "[$(date '+%F %T')] §10.1 multi-run variance ALL DONE" | tee -a "$LOG"
echo "=== 产物清单 ===" | tee -a "$LOG"
for arm in on off; do
  for rid in 1 2; do
    OD="${OUTBASE}_${arm}_multirun${rid}_${TS}"
    ED="${OUTBASE}_${arm}_multirun${rid}_${TS}_eval"
    s=$([ -f "${OD}/best_skill_evolved.md" ] && echo "✅" || echo "❌")
    e=$(ls "${ED}"/per_${arm}_*_seed*_shuf.json 2>/dev/null | wc -l)
    echo "  ${arm}${rid}: skill=$s eval_files=${e}/2" | tee -a "$LOG"
  done
done
