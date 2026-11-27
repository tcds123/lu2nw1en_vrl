#!/usr/bin/env python
# 纯离线诊断 (零 API): 候选轨迹 + 逐题贡献 + 预算对账 (2026-11-27, 用户建议三项)
import json, os, sys
from collections import defaultdict

RUN_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/livemath_fullrun_v32_20261126_161211/run_20261126_161212"

# ============ 诊断1: 候选轨迹 ============
print("=" * 70)
print("诊断1: 候选轨迹 (6轮各候选 + 晋升 + 是否有未晋升但更大收益)")
print("=" * 70)
gt = [json.loads(l) for l in open(os.path.join(RUN_DIR, "gate_trace.jsonl")) if l.strip()]
cands = [r for r in gt if r.get("operator")]  # 只取候选行, 跳过 budget_snapshot
print(f"候选行数: {len(cands)} (跨 {len(set(r['iter'] for r in cands))} 轮)")
print()
print(f"{'iter':>4} {'operator':<12} {'facet':<26} {'delta':>8} {'CI':>8} {'hash':<10} {'>best?':>6}")
print("-" * 80)
# evolver_log 的最终 best=0.0704 (iter6). 看各候选是否超此
FINAL_BEST = 0.0704
for r in cands:
    gt_str = "✓PROMOTE?" if r["val_delta"] >= FINAL_BEST - 0.001 else ""
    print(f"{r['iter']:>4} {r['operator']:<12} {r['facet']:<26} {r['val_delta']:>+8.4f} {r['val_ci']:>+8.4f} {r['cand_hash'][:8]:<10} {gt_str:>6}")
print()
# 最高 delta 候选 (不管 CI)
top_delta = sorted(cands, key=lambda r: -r["val_delta"])[:5]
print("Top-5 delta 候选 (不管 CI, 警惕同搜索数据选择偏差):")
for r in top_delta:
    print(f"  iter{r['iter']} {r['facet']:<26} delta={r['val_delta']:+.4f} CI={r['val_ci']:+.4f} hash={r['cand_hash'][:8]}")
# CI>0 候选 (过门)
ci_pos = [r for r in cands if r["val_ci"] > 0]
print(f"\nCI>0 过门候选: {len(ci_pos)}")
for r in ci_pos:
    print(f"  iter{r['iter']} {r['facet']:<26} delta={r['val_delta']:+.4f} CI={r['val_ci']:+.4f}")

# ============ 诊断2: 逐题贡献 ============
print()
print("=" * 70)
print("诊断2: 逐题贡献 (L3-meta 分布 + L2/L3 bootstrap 核实)")
print("=" * 70)
res = json.load(open(os.path.join(RUN_DIR, "L2_L3_test_results.json")))
pid = res["per_item_delta"]
print(f"L2 per_item_delta 题数: {len(pid)}")

# 需 meta 标签: 从 test split 读
sys.path.insert(0, "/data/zhuldz/lunwen")
sys.path.insert(0, "/data/zhuldz/skillopt")
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
from tf_global_v2_math import load_split, is_meta_item
import skill_evolver_math as se
test_items = load_split(se.SPLIT_DEFAULT, "test", shuffle_seed=0)
meta_qids = {str(it["id"]) for it in test_items if is_meta_item(it)}
nm_qids = {str(it["id"]) for it in test_items if not is_meta_item(it)}
print(f"test=106: meta={len(meta_qids)} not_meta={len(nm_qids)}")

# L3-meta 逐题 delta 分布
meta_deltas = {q: pid[q][0] for q in meta_qids if q in pid}
nm_deltas = {q: pid[q][0] for q in nm_qids if q in pid}
meta_vals = sorted(meta_deltas.values(), reverse=True)
print(f"\nL3-meta (n={len(meta_vals)}) delta 分布:")
print(f"  max={max(meta_vals):+.3f} min={min(meta_vals):+.3f} mean={sum(meta_vals)/len(meta_vals):+.4f}")
# 多题小幅 vs 少数题主导?
pos = [v for v in meta_vals if v > 0]
neg = [v for v in meta_vals if v < 0]
zero = [v for v in meta_vals if v == 0]
print(f"  正向题数={len(pos)} (sum={sum(pos):+.3f}) 负向={len(neg)} (sum={sum(neg):+.3f}) 零={len(zero)}")
# 贡献集中度: top-k 题贡献占比
total_pos = sum(pos)
sorted_pos = sorted(pos, reverse=True)
for k in [3, 5, 10]:
    if len(sorted_pos) >= k:
        print(f"  top-{k} 正向题贡献: {sum(sorted_pos[:k]):+.3f} ({100*sum(sorted_pos[:k])/total_pos:.0f}% of 正向总量)")
print(f"\nL3-meta 贡献集中度: {'少数题主导' if (len(sorted_pos)>=5 and sum(sorted_pos[:3])/total_pos>0.5) else '多题小幅改善'}")
print(f"\nL3-meta top-10 正向题:")
for q, v in sorted(meta_deltas.items(), key=lambda x: -x[1])[:10]:
    print(f"  {q:>12}: {v:+.3f}")

# bootstrap 核实 (重算 L2/L3 CI)
from tf_global_v2_math import bootstrap_ci_lower
l2_ci = bootstrap_ci_lower(pid)
meta_pid = {q: pid[q] for q in meta_qids if q in pid}
nm_pid = {q: pid[q] for q in nm_qids if q in pid}
meta_ci = bootstrap_ci_lower(meta_pid)
nm_ci = bootstrap_ci_lower(nm_pid)
print(f"\nbootstrap 核实 (重算):")
print(f"  L2 (n=106): CI_lower={l2_ci:+.4f} (落盘={res['L2']['ci_lower']:+.4f})")
print(f"  L3-meta (n={len(meta_pid)}): CI_lower={meta_ci:+.4f} (落盘={res['L3']['meta']['ci_lower']:+.4f})")
print(f"  L3-not_meta (n={len(nm_pid)}): CI_lower={nm_ci:+.4f} (落盘={res['L3']['not_meta']['ci_lower']:+.4f})")
print(f"  复算与落盘一致: L2={abs(l2_ci-res['L2']['ci_lower'])<1e-4} meta={abs(meta_ci-res['L3']['meta']['ci_lower'])<1e-4}")

# 子集多重检验注意
print(f"\n⚠️ 子集检验: 报告了 meta + not_meta 两个子集 CI, 名义 95% 各自. 若严苛需 Bonferroni 校正 (α/2=0.025 → CI 用 98.75% 分位). 当前用名义 95%, 论文应说明.")

# ============ 诊断3: 预算对账 ============
print()
print("=" * 70)
print("诊断3: 预算对账 (15640 calls vs 估计40k, 来源?)")
print("=" * 70)
ledger = json.load(open(os.path.join(RUN_DIR, "budget_ledger.json")))
print(f"实际总 calls: {ledger['n_requests']}")
print(f"wall: {ledger['wall_elapsed']:.0f}s = {ledger['wall_elapsed']/3600:.1f}h")
# evolver_log operator_counts
elog = json.load(open(os.path.join(RUN_DIR, "evolver_log.json")))
print(f"\nevolver_log iters: {len(elog['iters'])}")
op_total = defaultdict(lambda: defaultdict(int))
for it in elog["iters"]:
    oc = it.get("operator_counts", {})
    for op, d in oc.items():
        for k, v in d.items():
            op_total[op][k] += v
print("算子触发累计 (attempted/evaluated/admitted/promoted):")
for op, d in op_total.items():
    print(f"  {op}: attempted={d.get('attempted',0)} evaluated={d.get('evaluated',0)} admitted={d.get('admitted',0)} promoted={d.get('promoted',0)}")
# 估计 calls: evaluated 候选 × 71题 × k=3 × 2(paired) + S0(213) + collect_failures
n_eval = sum(d.get("evaluated", 0) for d in op_total.values())
print(f"\n候选评测总数 (evaluated): {n_eval}")
print(f"  估计评测 calls = {n_eval} × 71题 × k=3 × 2(paired) = {n_eval*71*3*2}")
print(f"  S0 重算 = 71 × 3 = 213")
print(f"  collect_failures 每轮 ~71×3=213 × 6轮 = {213*6}")
print(f"  估计总计 ≈ {n_eval*71*3*2 + 213 + 213*6}")
print(f"  实际 {ledger['n_requests']} vs 估计 {n_eval*71*3*2 + 213 + 213*6}")
# 候选数低于预期?
print(f"\n候选数分析: evaluated={n_eval}, 冻结方案预期每轮 3facet×4K=12 + overlay/expand/crossover ≈ 15/轮 × 6 = 90")
print(f"  实际 evaluated={n_eval} {'<< 预期 (候选数远低于估计, 主因)' if n_eval < 60 else '接近预期'}")
