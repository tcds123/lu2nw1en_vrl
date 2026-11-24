#!/usr/bin/env bash
# Q1 cap 扫描 (2026-11-13): 固定 skill, 降低 cap, 检验预算纪律 skill 在"写不完"的预算下的优势.
# 顺序跑 2 个 cap (8192/4096) (避免并发争用网关 RPM); 每个 cap 一套独立 RUN, 可续跑 (CACHED).
# 2026-11-13 用户决定: 2048/1024 太小, 砍掉 (信息来源不足).
#   s0 = skill_aime_initial_v2.md        s1 = demo5 预算纪律补丁 (截断率 2.08%→0.25%)
#   s2 = demo6 宁缺毋滥补丁              none = 无 skill (参照)
# 用法: bash scripts/run_cap_sweep_20261113.sh [cap1 cap2 ...]   (默认 8192 4096)
set -euo pipefail
PKG="${PKG_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
L="$PKG/outputs/paired_budget_s0/candidates"
CAPS="${CAPS_SKILL:-s1=$L/demo5_budget_focus.md,s2=$L/demo6_nofinal_focus.md}"
ARMS="${ARMS:-none,s0,s1,s2}"
OUT="${OUT:-$PKG/outputs/paired_cap_sweep_20261113}"
[[ $# -gt 0 ]] && LIST=("$@") || LIST=(8192 4096)
for cap in "${LIST[@]}"; do
  echo "===== CAP=$cap  $(date '+%F %T') ====="
  # 失败 (如参数错) 不静默跳过: 记 FAILED 并中止整个扫描
  if ! CAP="$cap" RUN="cap$cap" ARMS="$ARMS" CAPS="$CAPS" OUT="$OUT" \
      bash "$PKG/scripts/run_paired_five_arm_20261113.sh" >> "$OUT.cap$cap.log" 2>&1; then
    echo "===== CAP=$cap FAILED $(date '+%F %T') (见 $OUT.cap$cap.log) ====="
    grep -E "Error|error|Traceback|SystemExit|未知" "$OUT.cap$cap.log" | tail -5
    exit 1
  fi
  echo "===== CAP=$cap DONE $(date '+%F %T') ====="
  # 每 cap 补跑 error 样本 (protocol: n_errors=0 才进表); 补跑脚本固定 16384, 对低 cap 只用于清稀少的网关错误
  python3 - "$OUT/cap$cap" "$cap" <<'PY'
import json,glob,sys
d,cap=sys.argv[1],sys.argv[2]
n=0
for f in glob.glob(d+"/per_*.json"):
    for ss in json.load(open(f)).values(): n+=sum(1 for s in ss if s.get("error"))
print(f"  n_errors(cap={cap})={n}")
PY
  python3 - "$OUT/cap$cap/paired_report.json" <<'PY' || echo "WARN: $cap report 缺失"
import json,sys
r=json.load(open(sys.argv[1]))
for arm,m in r.get("skills",{}).items():
    x=m["mean"]; print(f"  {arm}: em={x['sample_em']:.4f} trunc={x['sample_trunc']:.4f} kvote={x['k_vote_pass']:.4f} nerr={x['n_errors']:.1f}")
for k,v in r.get("pairwise",{}).items(): print(f"  {k}: wins/losses={v['pooled']['wins']}/{v['pooled']['losses']} p={v['pooled']['p']}")
for k,v in r.get("truncation_pairwise",{}).items(): print(f"  trunc {k}: fewer/more={v['pooled']['units_with_fewer_trunc']}/{v['pooled']['units_with_more_trunc']} p={v['pooled']['p']}")
PY
done
