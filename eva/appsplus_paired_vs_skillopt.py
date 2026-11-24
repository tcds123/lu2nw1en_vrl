#!/usr/bin/env python3
"""APPS_Plus 候选 skill vs SkillOpt 候选 skill 的两个配对 McNemar 检验。

纯离线分析 (零 flash 调用), 回答两个"不能下结论"的统计缺口:

  检验 1 (Rescue 配对): 92 道 S0-fail 题里, 我们的候选 vs SkillOpt 候选,
          各自救回多少? 24 vs 17 的差异是否显著?
  检验 2 (Retention 配对): 208 道 S0-pass 题里, 我们的候选 vs SkillOpt 候选,
          各自退化多少? 23 vs 14 (4.33pp) 的差异是否显著?

关键事实: 我们的 s0_cache_test == SkillOpt 的 test_eval_baseline/results.json
→ 两个系统共用同一份 S0 baseline, 逐题结果完全相同 → 比较公平。

数据源:
  我们 routed (S0-fail 题上=候选独立执行, S0-pass 题上=S0 cache):
    crossmodel_baseline/tf-appsplus-routing-retention-20261113/run_20261113_152603/test_routed_results.json
  我们 retention (208 S0-pass 题上候选独立重跑):
    同目录/test_retention_results.json
  SkillOpt best skill (全 300 题候选独立执行):
    crossmodel_baseline/appsplus-skillopt-flash-20261111-v2/train/test_eval/results.json
  SkillOpt S0 baseline (= 我们的 S0):
    同目录/test_eval_baseline/results.json
"""
import json, os, math
from collections import Counter

ROOT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline"
OURS_RUN = f"{ROOT}/tf-appsplus-routing-retention-20261113/run_20261113_152603"
SO_DIR = f"{ROOT}/appsplus-skillopt-flash-20261111-v2/train"


def load(path):
    d = json.load(open(path))
    if isinstance(d, dict):
        return {str(k): v for k, v in d.items()}
    return {str(x["id"]): x for x in d}


def hard1(r):
    """pass 判定: hard==1 (整型/浮点/字符串都兼容)"""
    return int(r.get("hard", 0)) == 1


def mcnemar_exact(b, c):
    """精确双侧 McNemar (二项检验, n=b+c, p=0.5)。

    返回 (b, c, p_two_sided, or_stat)。
    """
    n = b + c
    if n == 0:
        return b, c, 1.0, 0.0
    # 双侧 p = 2 * P(X <= min(b,c)) ， X~Binom(n, 0.5)
    lo = min(b, c)
    p_one = sum(math.comb(n, i) * (0.5 ** n) for i in range(lo + 1))
    p_two = min(1.0, 2 * p_one)
    # 方向统计: 比例差
    return b, c, p_two, (b - c) / n


def table2x2(a_pass, b_pass, label_a, label_b):
    """a_pass/b_pass: dict id->bool (是否通过)。
    返回 (a, b, c, d):
      a = 两者都 pass
      b = a pass & b fail   (a 更好方向)
      c = a fail & b pass   (b 更好方向)
      d = 两者都 fail
    """
    common = set(a_pass) & set(b_pass)
    a = b = c = d = 0
    for k in common:
        av, bv = a_pass[k], b_pass[k]
        if av and bv:
            a += 1
        elif av and not bv:
            b += 1
        elif not av and bv:
            c += 1
        else:
            d += 1
    return a, b, c, d, len(common)


def main():
    # ── 加载 ──
    ours_routed = load(f"{OURS_RUN}/test_routed_results.json")
    ours_ret = load(f"{OURS_RUN}/test_retention_results.json")
    so_best = load(f"{SO_DIR}/test_eval/results.json")
    so_s0 = load(f"{SO_DIR}/test_eval_baseline/results.json")

    # ── 共用 S0 baseline: 确认我们的 S0 == SkillOpt 的 S0 ──
    # 我们 result.json 的 args.s0_cache_test 指向 so_s0 同文件
    s0_hard = {k: hard1(v) for k, v in so_s0.items()}
    n_s0_pass = sum(s0_hard.values())
    n_s0_fail = len(s0_hard) - n_s0_pass
    print("=" * 72)
    print("共用 S0 baseline (SkillOpt test_eval_baseline = 我们的 s0_cache_test)")
    print(f"  全 300 题: S0 pass={n_s0_pass}, S0 fail={n_s0_fail}")
    print(f"  → retention 子集 = {n_s0_pass} 道 S0-pass; rescue 子集 = {n_s0_fail} 道 S0-fail")
    print()

    # ────────────────────────────────────────────────────────────
    # 检验 1: Rescue 配对 (92 道 S0-fail)
    # 我们 routed 在这些题上 = 候选独立执行 (路由 turn2 调候选)
    # SkillOpt best (so_best) 在这些题上 = 候选独立执行
    # 两者都是"候选 skill 在 S0-fail 题上的独立结果" → 公平比较
    # ────────────────────────────────────────────────────────────
    print("=" * 72)
    print("检验 1 — Rescue 配对 (我们的候选 vs SkillOpt 候选, 在 92 道 S0-fail 题)")
    print("  问题: 24 vs 17 的救回差异是否统计显著?")
    print()
    fail_ids = [k for k, v in s0_hard.items() if not v]
    ours_rescue = {k: hard1(ours_routed[k]) for k in fail_ids if k in ours_routed}
    so_rescue = {k: hard1(so_best[k]) for k in fail_ids if k in so_best}
    a, b, c, d, n = table2x2(ours_rescue, so_rescue, "我们", "SkillOpt")
    ours_r = sum(ours_rescue.values())
    so_r = sum(so_rescue.values())
    print(f"  子集 n={n} (S0-fail)")
    print(f"  我们救回 (routed 候选)    = {ours_r}")
    print(f"  SkillOpt 救回 (best 候选) = {so_r}")
    print()
    print(f"  2×2 配对表 (行=我们, 列=SkillOpt):")
    print(f"            SkillOpt救回   SkillOpt没救回")
    print(f"  我们救回      {a:>5}          {b:>5}   ← 我们更好方向 (b)")
    print(f"  我们没救回    {c:>5}          {d:>5}   ← SkillOpt更好方向 (c)")
    print()
    bb, cc, p, orstat = mcnemar_exact(b, c)
    sig = "✅ 显著" if p < 0.05 else ("⚠️ 边际" if p < 0.1 else "❌ 不显著")
    print(f"  McNemar (discordant b+c={b+c}): b={b}(我们更好) c={c}(SkillOpt更好)")
    print(f"  精确双侧 p = {p:.4e}   [{sig}]")
    better = "我们" if b > c else ("SkillOpt" if c > b else "持平")
    print(f"  方向: {better} 救回更多 (净 discordant 差 {abs(b-c)})")
    print()

    # ────────────────────────────────────────────────────────────
    # 检验 2: Retention 配对 (208 道 S0-pass)
    # 我们 retention = 候选独立重跑
    # SkillOpt best (so_best) 在这些题上 = 候选独立执行
    # 两者都是"候选 skill 在 S0-pass 题上的独立结果" → 公平比较
    # ────────────────────────────────────────────────────────────
    print("=" * 72)
    print("检验 2 — Retention 配对 (我们的候选 vs SkillOpt 候选, 在 208 道 S0-pass 题)")
    print("  问题: 退化 23 vs 14 (4.33pp) 的差异是否统计显著?")
    print()
    pass_ids = [k for k, v in s0_hard.items() if v]
    ours_keep = {k: hard1(ours_ret[k]) for k in pass_ids if k in ours_ret}
    so_keep = {k: hard1(so_best[k]) for k in pass_ids if k in so_best}
    a2, b2, c2, d2, n2 = table2x2(ours_keep, so_keep, "我们", "SkillOpt")
    ours_reg = n2 - sum(ours_keep.values())  # 我们退化 = 没保持
    so_reg = n2 - sum(so_keep.values())      # SkillOpt 退化
    print(f"  子集 n={n2} (S0-pass)")
    print(f"  我们保持    (retention 候选) = {sum(ours_keep.values())} / 退化 {ours_reg}")
    print(f"  SkillOpt 保持(best 候选)    = {sum(so_keep.values())} / 退化 {so_reg}")
    print()
    print(f"  2×2 配对表 (行=我们, 列=SkillOpt, 按保持计数):")
    print(f"            SkillOpt保持   SkillOpt退化")
    print(f"  我们保持      {a2:>5}          {b2:>5}   ← 我们更好 (SkillOpt退化而我们保持)")
    print(f"  我们退化      {c2:>5}          {d2:>5}   ← SkillOpt更好 (我们退化而SkillOpt保持)")
    print()
    bb2, cc2, p2, orstat2 = mcnemar_exact(b2, c2)
    sig2 = "✅ 显著" if p2 < 0.05 else ("⚠️ 边际" if p2 < 0.1 else "❌ 不显著")
    print(f"  McNemar (discordant b+c={b2+c2}): b={b2}(我们更好) c={c2}(SkillOpt更好)")
    print(f"  精确双侧 p = {p2:.4e}   [{sig2}]")
    better2 = "我们" if b2 > c2 else ("SkillOpt" if c2 > b2 else "持平")
    print(f"  方向: {better2} 保持更好 (净 discordant 差 {abs(b2-c2)})")
    print()

    # ────────────────────────────────────────────────────────────
    # 交叉确认: routed 232 vs SkillOpt best 211 的逐题配对 (全 300 题)
    # ────────────────────────────────────────────────────────────
    print("=" * 72)
    print("交叉确认 — Routed(232) vs SkillOpt best(211) 全 300 题配对")
    ours_all = {k: hard1(ours_routed[k]) for k in so_s0 if k in ours_routed}
    so_all = {k: hard1(so_best[k]) for k in so_s0 if k in so_best}
    a3, b3, c3, d3, n3 = table2x2(ours_all, so_all, "我们", "SkillOpt")
    print(f"  我们通过 (routed) = {sum(ours_all.values())} ; SkillOpt 通过 = {sum(so_all.values())}")
    print(f"  2×2: 都通过={a3} 我们更好(我们通过&SO没)={b3} SO更好={c3} 都没={d3}")
    bb3, cc3, p3, _ = mcnemar_exact(b3, c3)
    print(f"  McNemar p = {p3:.4e}  (文档记录 p=1.9e-4, 核对一致性)")
    print()

    # ── 汇总 ──
    print("=" * 72)
    print("汇总 — 三个配对检验")
    print(f"  Rescue   (92 fail):  我们{ours_r} vs SO{so_r}  | discordant {b}vs{c} p={p:.2e} [{sig}]")
    print(f"  Retention(208 pass): 退化{ours_reg} vs {so_reg} | discordant {c2}vs{b2} p={p2:.2e} [{sig2}]")
    print(f"  Routed   (300):     232 vs 211      | discordant {b3}vs{c3} p={p3:.2e}")
    print()

    # 落盘
    out = {
        "note": "候选 skill vs SkillOpt 候选 skill 配对 McNemar; 共用 S0 baseline",
        "s0_baseline": "SkillOpt test_eval_baseline = 我们的 s0_cache (同一文件)",
        "n_s0_pass": n_s0_pass, "n_s0_fail": n_s0_fail,
        "rescue_paired": {
            "subset_n": n, "ours_rescue": ours_r, "skillopt_rescue": so_r,
            "both_rescue": a, "ours_only": b, "skillopt_only": c, "neither": d,
            "mcnemar_b_ours_better": b, "mcnemar_c_so_better": c,
            "p_exact_two_sided": p, "verdict": sig, "direction": better,
        },
        "retention_paired": {
            "subset_n": n2, "ours_regress": ours_reg, "skillopt_regress": so_reg,
            "both_preserved": a2, "ours_preserved_so_regressed": b2,
            "ours_regressed_so_preserved": c2, "both_regressed": d2,
            "mcnemar_b_ours_better": b2, "mcnemar_c_so_better": c2,
            "p_exact_two_sided": p2, "verdict": sig2, "direction": better2,
        },
        "routed_vs_skillopt_best_300": {
            "ours_pass": sum(ours_all.values()), "skillopt_pass": sum(so_all.values()),
            "both_pass": a3, "ours_only": b3, "skillopt_only": c3, "neither": d3,
            "p_exact_two_sided": p3,
        },
    }
    outp = f"{OURS_RUN}/paired_vs_skillopt_analysis.json"
    json.dump(out, open(outp, "w"), ensure_ascii=False, indent=2)
    print(f"已落盘: {outp}")


if __name__ == "__main__":
    main()
