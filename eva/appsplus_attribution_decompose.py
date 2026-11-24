#!/usr/bin/env python3
"""核对 26:5 vs 12:5 的一致性 + 候选 vs 候选纯能力对比 (都不路由)。

回应外部审阅 Part I 的逻辑追问:
  如果两系统都路由, 全量分歧应 = fail 子集分歧. 但 12:5 ≠ 26:5.
  原因假设: SkillOpt 的 211 是候选独立 (不路由, regress=14≠0),
            而 232 是我们的路由系统 (regress=0).
            → 26:5 是"路由系统 vs 非路由候选系统", 不是"两个路由系统".

本脚本验证:
  (1) 拆 26:5 为 pass/fail 子集贡献, 确认多出的 14 来自 pass 子集 (我们路由保护 vs SO候选退化).
  (2) 候选 vs 候选 全300 (都不路由): 我们候选(retention∪routed-fail) vs SO候选.
      → 预期 ours 209 vs SO 211, discordant 20 vs 22, 不显著.
  (3) 若都路由 (架构对等): 归约到 fail 子集 candidate-vs-candidate = 12 vs 5.

纯离线, 零 flash.
"""
import json, math

ROOT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline"
OURS = f"{ROOT}/tf-appsplus-routing-retention-20261113/run_20261113_152603"
SO = f"{ROOT}/appsplus-skillopt-flash-20261111-v2/train"


def load(path):
    d = json.load(open(path))
    if isinstance(d, dict):
        return {str(k): v for k, v in d.items()}
    return {str(x["id"]): x for x in d}


def hard1(r):
    return int(r.get("hard", 0)) == 1


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    p1 = sum(math.comb(n, i) * (0.5 ** n) for i in range(lo + 1))
    return min(1.0, 2 * p1)


def table(a_pass, b_pass):
    """返回 (both_pass, a_only=a更好, b_only=b更好, both_fail)."""
    common = set(a_pass) & set(b_pass)
    a = b = c = d = 0
    for k in common:
        av, bv = a_pass[k], b_pass[k]
        if av and bv: a += 1
        elif av and not bv: b += 1
        elif not av and bv: c += 1
        else: d += 1
    return a, b, c, d, len(common)


ours_routed = load(f"{OURS}/test_routed_results.json")      # 路由系统 (pass=S0, fail=候选)
ours_ret = load(f"{OURS}/test_retention_results.json")      # 我们候选 独立 on 208 pass
so_best = load(f"{SO}/test_eval/results.json")              # SO候选 独立 on 全300
so_s0 = load(f"{SO}/test_eval_baseline/results.json")        # S0 baseline (=我们的S0)

pass_ids = [k for k, v in so_s0.items() if int(v["hard"]) == 1]   # 208
fail_ids = [k for k, v in so_s0.items() if int(v["hard"]) == 0]  # 92

print("=" * 74)
print("(1) 拆解 26:5 — 路由系统(ours_routed) vs SO候选(so_best), 全300")
ours_all = {k: hard1(ours_routed[k]) for k in so_s0 if k in ours_routed}
so_all = {k: hard1(so_best[k]) for k in so_s0}
a, b, c, d, n = table(ours_all, so_all)
print(f"  全300: ours_pass={sum(ours_all.values())} SO_pass={sum(so_all.values())}")
print(f"  both_pass={a} ours_only(更好)={b} SO_only(更好)={c} both_fail={d}")
print(f"  discordant {b} vs {c}, p={mcnemar(b,c):.4e}")
print()

# pass 子集贡献
ours_pass_sub = {k: hard1(ours_routed[k]) for k in pass_ids if k in ours_routed}
so_pass_sub = {k: hard1(so_best[k]) for k in pass_ids if k in so_best}
ap, bp, cp, dp, np_ = table(ours_pass_sub, so_pass_sub)
print(f"  pass子集(208): ours_routed(=S0全pass) vs SO候选(194pass/14退化)")
print(f"    both_pass={ap} ours_only={bp} SO_only={cp} both_fail={dp}")
print(f"    → 路由保护贡献 ours_only+={bp} (SO候选退化而我们S0保住), SO_only+={cp}")
# fail 子集贡献
ours_fail_sub = {k: hard1(ours_routed[k]) for k in fail_ids if k in ours_routed}
so_fail_sub = {k: hard1(so_best[k]) for k in fail_ids if k in so_best}
af, bf, cf, df, nf = table(ours_fail_sub, so_fail_sub)
print(f"  fail子集(92): ours候选 vs SO候选 (candidate-vs-candidate)")
print(f"    both_rescue={af} ours_only={bf} SO_only={cf} neither={df}")
print(f"    → 候选rescue贡献 ours_only+={bf}, SO_only+={cf}")
print(f"  核对: ours_only {bp}+{bf}={bp+bf} (应=26={b}); SO_only {cp}+{cf}={cp+cf} (应=5={c})")
print(f"  结论: 多出的 {b-26+12} ... 直接看: pass贡献{bp}vs{cp}, fail贡献{bf}vs{cf}")
print()

print("=" * 74)
print("(2) 候选 vs 候选 纯能力 (都不路由, 全300) — 关键公平对比")
# 我们候选独立全300 = retention(208 pass) + routed-fail(92 fail候选)
ours_cand = {}
for k in pass_ids:
    if k in ours_ret: ours_cand[k] = hard1(ours_ret[k])
for k in fail_ids:
    if k in ours_routed: ours_cand[k] = hard1(ours_routed[k])  # routed在fail上=候选
so_cand = {k: hard1(so_best[k]) for k in so_s0 if k in so_best}
a2, b2, c2, d2, n2 = table(ours_cand, so_cand)
print(f"  全300: ours候选pass={sum(ours_cand.values())} SO候选pass={sum(so_cand.values())}")
print(f"  both_pass={a2} ours_only(我们更好)={b2} SO_only(SO更好)={c2} both_fail={d2}")
print(f"  discordant {b2} vs {c2}, p={mcnemar(b2,c2):.4e}")
sig = "✅显著" if mcnemar(b2,c2)<0.05 else "❌不显著"
print(f"  → 候选纯能力: {sig}, 方向={'我们更好' if b2>c2 else 'SO更好' if c2>b2 else '持平'} (净{b2-c2:+d})")
print()

print("=" * 74)
print("(3) 若都路由 (架构对等) — 归约到 fail 子集 candidate-vs-candidate")
print(f"  两系统都路由 → pass子集都=S0全pass (0分歧) → 全部分歧来自fail子集")
print(f"  fail子集 candidate-vs-candidate: ours_only={bf} SO_only={cf}, p={mcnemar(bf,cf):.4e}")
sig3 = "✅显著" if mcnemar(bf,cf)<0.05 else "❌不显著"
print(f"  → 架构对等下: {sig3} (与Rescue配对检验一致)")
print()

print("=" * 74)
print("汇总")
print(f"  路由系统 vs SO候选系统 (232 vs 211): {b}vs{c} p={mcnemar(b,c):.2e} ✅ 显著 — 但架构不对等(我路由/SO不路由)")
print(f"  候选 vs 候选 纯能力 (209 vs 211):   {b2}vs{c2} p={mcnemar(b2,c2):.2e} ❌ 不显著 — 候选持平(SO略优)")
print(f"  都路由 架构对等 (归约fail子集):      {bf}vs{cf} p={mcnemar(bf,cf):.2e} ❌ 不显著 — =Rescue配对")
print(f"  26:5 拆解: pass子集{bp}vs{cp}(路由保护) + fail子集{bf}vs{cf}(候选rescue)")
print(f"  → 232显著性来源: 路由架构({bp}vs{cp}) + 候选rescue({bf}vs{cf}); 候选纯能力本身持平")
