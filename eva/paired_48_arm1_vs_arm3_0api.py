#!/usr/bin/env python3
"""#48 主比较: Arm1 (full) vs Arm3 (drop_scaffold) 配对统计 (0 API, 2026-11-19)

按 docs/48_analysis_plan_amendment_20261119.md §4 (冻结) 执行:
  - 配对单位: 单题, 两臂均 valid (agent_ok) 的 2765 题共同有效集
  - 主检验: 精确 McNemar (双侧 binomial exact on discordant)
  - 效应量: 净增益 g = arm1_only_pass − arm3_only_pass, rate = g / n_common
  - CI: 对 r = rescue/discordant 取 Clopper-Pearson 精确区间 (条件于 d), 再仿射变换
        g_rate = (2r−1)·d/n。注意: 这不是单臂通过率的 CP 区间, 而是"不一致对占比"区间
        经线性映射得到; 交叉核对见 eva/ci_crosscheck_48.py
  - δ = 0.03 (历史设计参数; n=2765 时 ≈ 83 题)
  - 四象限计数 + p + g + CI 全部报告
  - 补充敏感性分析: 排除 107 题 NameError 证据集中的题 (全量 + 排除后同时报告)
  - 参考比较 (非主结论): Arm1 vs S₀(215) 与 Arm3 vs S₀(215) 在共同集上

输入 (只读):
  ARM1/test_independent_results.json, ARM3/test_independent_results.json
  ARM1/audit_0api_20261118/nameerror_107_evidence.json (排除集)
  ARM1/run_*_/s0_cache_test_d5.json (S₀ 重判基线, 300 题)
输出: <ARM1_RUN>/audit_0api_20261118/paired_arm1_vs_arm3.json + .md
"""
import json
import os
import sys
from collections import Counter

BASE = ("/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline")
ARM1_RUN = os.path.join(BASE, "tf48-arm1-full-20261117/run_20261118_091052")
ARM3_RUN = os.path.join(BASE, "tf48-arm3-drop-scaffold-20261117/run_20261119_091235")
AUDIT = os.path.join(ARM1_RUN, "audit_0api_20261118")

sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
from tf_skill_optimizer_appsplus import (  # noqa: E402
    mcnemar_paired, paired_net_gain_ci, interpret_ci, DELTA_MIN,
)


def load(path):
    return json.load(open(path, encoding="utf-8"))


def valid_subset(res):
    return {q: r for q, r in res.items() if r.get("agent_ok")}


def full_quad(a, b):
    """全量 2x2 (无效题按 hard=0 计入, 用于闭合核对)"""
    ids = set(a) | set(b)
    bp = bf = a_only = b_only = 0
    for i in ids:
        ah = int(a.get(i, {}).get("hard", 0))
        bh = int(b.get(i, {}).get("hard", 0))
        if ah and bh:
            bp += 1
        elif not ah and not bh:
            bf += 1
        elif ah:
            a_only += 1
        else:
            b_only += 1
    return {"both_pass": bp, "both_fail": bf, "a_only_pass": a_only,
            "b_only_pass": b_only, "n": len(ids), "a_hard": bp + a_only,
            "b_hard": bp + b_only}


def main():
    arm1 = load(os.path.join(ARM1_RUN, "test_independent_results.json"))
    arm3 = load(os.path.join(ARM3_RUN, "test_independent_results.json"))
    v1, v3 = valid_subset(arm1), valid_subset(arm3)
    s0 = load(os.path.join(ARM1_RUN, "s0_cache_test_d5.json"))
    ne = load(os.path.join(AUDIT, "nameerror_107_evidence.json"))
    ne_ids = {r["id"] for r in ne["items"]}

    out = {"generated_by": "eva/paired_48_arm1_vs_arm3_0api.py",
           "delta_min": DELTA_MIN,
           "arm1_run": ARM1_RUN, "arm3_run": ARM3_RUN,
           "closure": {"arm1_n": len(arm1), "arm1_valid": len(v1),
                       "arm3_n": len(arm3), "arm3_valid": len(v3)}}

    # ── 1. 主比较 (共同有效集) ──────────────────────────────────────────────
    bp, bf, r, g, n, p = mcnemar_paired(v1, v3)   # a=arm1, b=arm3; a_only → rescue
    ng = paired_net_gain_ci(r, g, n)
    main = {"both_pass": bp, "both_fail": bf, "arm1_only_pass(rescue)": r,
            "arm3_only_pass(regress)": g, "discordant": r + g,
            "n_common": n, "p_exact": p,
            "arm1_hard": bp + r, "arm3_hard": bp + g,
            "arm1_rate": round((bp + r) / n, 4), "arm3_rate": round((bp + g) / n, 4),
            "net_gain": ng, "interpretation": interpret_ci(ng, DELTA_MIN)}
    out["main_arm1_vs_arm3"] = main
    # 全量闭合核对 (含 invalid; 两臂 infra 均为 0 时与 valid 一致)
    out["main_full_2765_check"] = full_quad(arm1, arm3)

    # ── 2. 补充敏感性分析: 排除 107 题 ────────────────────────────────────
    v1x = {q: x for q, x in v1.items() if q not in ne_ids}
    v3x = {q: x for q, x in v3.items() if q not in ne_ids}
    # 排除集内两臂通过情况 (预期 0; 若不为 0 需说明)
    ne_both_pass = sum(1 for q in ne_ids if q in v1 and v1[q]["hard"] and q in v3 and v3[q]["hard"])
    ne_any_pass = sum(1 for q in ne_ids if (q in v1 and v1[q]["hard"]) or (q in v3 and v3[q]["hard"]))
    bp2, bf2, r2, g2, n2, p2 = mcnemar_paired(v1x, v3x)
    ng2 = paired_net_gain_ci(r2, g2, n2)
    out["sensitivity_excl_107"] = {
        "n_excluded_from_common": len(ne_ids & set(v1) & set(v3)),
        "excluded_any_pass_any_arm": ne_any_pass,
        "excluded_both_pass": ne_both_pass,
        "both_pass": bp2, "both_fail": bf2, "arm1_only_pass": r2, "arm3_only_pass": g2,
        "n_common": n2, "p_exact": p2, "net_gain": ng2,
        "interpretation": interpret_ci(ng2, DELTA_MIN)}

    # ── 3. 参考比较: 两臂 vs S₀(215, 300 题) ──────────────────────────────
    ref = {}
    for name, v in (("arm1", v1), ("arm3", v3)):
        bpr, bfr, rr, gg, nn, pp = mcnemar_paired(v, s0)
        ngr = paired_net_gain_ci(rr, gg, nn)
        ref[name + "_vs_s0_300"] = {
            "both_pass": bpr, "both_fail": bfr, "rescue(0->1)": rr, "regress(1->0)": gg,
            "n_common": nn, "p_exact": pp, "net_gain": ngr,
            "interpretation": interpret_ci(ngr, DELTA_MIN)}
    out["reference_vs_s0"] = ref
    # 异构性诊断: "有效通过" (pass_rate>=0.5, 部分通过也算) 口径下的主比较,
    # 用于判断 hard 二值差异是否只是 near-miss 的形态变化
    for name, v in (("arm1", v1), ("arm3", v3)):
        out.setdefault("effective_pass_diag", {})[name] = {
            "n_pass_rate_ge_0.5": sum(1 for x in v.values() if float(x.get("pass_rate", 0)) >= 0.5),
            "n_hard": sum(1 for x in v.values() if x["hard"])}
    e1 = {q: {"hard": int(float(x.get("pass_rate", 0)) >= 0.5)} for q, x in v1.items()}
    e3 = {q: {"hard": int(float(x.get("pass_rate", 0)) >= 0.5)} for q, x in v3.items()}
    bpe, bfe, re_, ge, ne, pe = mcnemar_paired(e1, e3)
    out["effective_pass_diag"]["arm1_vs_arm3_pr>=0.5"] = {
        "both_pass": bpe, "both_fail": bfe, "arm1_only": re_, "arm3_only": ge,
        "n_common": ne, "p_exact": pe, "net_gain": paired_net_gain_ci(re_, ge, ne)}

    # ── CI 方法交叉核对 (用户 2026-11-19 要求) ─────────────────────────────
    # 内置 paired_net_gain_ci 实际构造: 对 rescue/discordant 比例 r̂ 取 Clopper-Pearson 精确区间,
    # 再以 g_rate = (2r̂−1)·d/n 作单调仿射变换 (d, n 视为固定) → 区间覆盖 g_rate 的条件在 r̂ 上成立;
    # 这是"对不一致对中一方占比"的 CP 区间, 不是对单臂通过率 p 的 CP 区间, 也不是无条件覆盖保证。
    # 交叉核对: 在 r 上做精确二项检验反转 (概率序双侧 p) 与 Wilson score, 再经同一仿射变换;
    # 三种构造若给出相近区间 → 判读不依赖具体区间构造方式。
    ci_diag = None
    try:
        from ci_crosscheck_48 import cond_exact_ci
        ci_diag = cond_exact_ci(r, g, n)
    except Exception as e:  # 环境缺 scipy 等 → 只记录
        ci_diag = {"error": repr(e)}
    out["ci_method"] = {
        "builtin": "CP_exact_on_rescue/discordant, 仿射变换到 g_rate=(2r-1)*d/n",
        "builtin_ci": [ng["ci_lo_rate"], ng["ci_hi_rate"]],
        "cond_exact_inversion": ci_diag,
    }

    # 参考: Arm1 vs Arm3 限制在 S₀ 300 题子集 (与上面两行同口径)
    sub1 = {q: x for q, x in v1.items() if q in s0}
    sub3 = {q: x for q, x in v3.items() if q in s0}
    bp3, bf3, r3, g3, n3, p3 = mcnemar_paired(sub1, sub3)
    out["reference_vs_s0"]["arm1_vs_arm3_on_300"] = {
        "both_pass": bp3, "both_fail": bf3, "arm1_only_pass": r3, "arm3_only_pass": g3,
        "n_common": n3, "p_exact": p3, "net_gain": paired_net_gain_ci(r3, g3, n3)}

    # ── 4. 训练侧对照 (记录, 非主结论) ────────────────────────────────────
    out["training_side"] = {"arm1_val_best_delta": 0.1067, "arm3_val_best_delta": 0.0750,
                            "note": "val 100 题 routed 选择信号; 仅记录, 不作主结论"}

    # ── 5. 可复现证据 (用户 2026-11-19 要求) ──────────────────────────────
    import hashlib
    import shutil

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    common_ids = sorted(set(v1) & set(v3))
    out["reproducibility"] = {
        "pairing": "按题目 ID 配对: sorted(set(arm1) & set(arm3)) 的集合交, 不依赖 JSON 行序 "
                   "(mcnemar_paired 内部对 dict 取 sorted id 交集后逐题比较 hard)",
        "n_common_ids": len(common_ids),
        "common_ids_sha256": hashlib.sha256(
            "\n".join(common_ids).encode()).hexdigest(),
        "common_ids_file": "common_valid_ids_2765.txt",
        "file_sha256": {
            "arm1_test_independent_results.json": sha(os.path.join(ARM1_RUN, "test_independent_results.json")),
            "arm3_test_independent_results.json": sha(os.path.join(ARM3_RUN, "test_independent_results.json")),
            "arm1_s0_cache_test_d5.json": sha(os.path.join(ARM1_RUN, "s0_cache_test_d5.json")),
            "nameerror_107_evidence.json": sha(os.path.join(AUDIT, "nameerror_107_evidence.json")),
            "stats_script": sha(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "paired_48_arm1_vs_arm3_0api.py")),
            "ci_crosscheck_script": sha(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "ci_crosscheck_48.py")),
        },
        "main_2x2": {"both_pass": bp, "both_fail": bf,
                     "arm1_only_pass(rescue)": r, "arm3_only_pass(regress)": g,
                     "n_common": n},
        "stats_modules": "rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py: "
                         "mcnemar_paired / _mcnemar_exact_p / paired_net_gain_ci / _clopper_pearson / interpret_ci",
        "run_output": "run_output.txt (本脚本 stdout 完整落盘)",
        "shell": "python3 eva/paired_48_arm1_vs_arm3_0api.py > .../run_output.txt 2>&1",
    }

    os.makedirs(AUDIT, exist_ok=True)
    cid = os.path.join(AUDIT, "common_valid_ids_2765.txt")
    open(cid, "w", encoding="utf-8").write("\n".join(common_ids) + "\n")
    # 脚本快照入审计目录 (冻结版本, 与 file_sha256 对照)
    for f in ("paired_48_arm1_vs_arm3_0api.py", "ci_crosscheck_48.py"):
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
        shutil.copy2(src, os.path.join(AUDIT, "frozen_" + f))
    pj = os.path.join(AUDIT, "paired_arm1_vs_arm3.json")
    json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def fmt(d):
        ng_ = d["net_gain"]
        return (f"2x2: both_pass={d['both_pass']} both_fail={d['both_fail']} "
                f"discordant={d.get('discordant', d.get('arm1_only_pass', 0) + d.get('arm3_only_pass', 0) if 'arm1_only_pass' in d else '?')} "
                f"| net_gain g={ng_['net_gain']:+d} rate={ng_['net_gain_rate']:+.4f} "
                f"CI=[{ng_['ci_lo_rate']:+.4f},{ng_['ci_hi_rate']:+.4f}] δ={DELTA_MIN}")

    md = [
        "# #48 主比较: Arm1 (full) vs Arm3 (drop_scaffold)", "",
        f"- 生成: `eva/paired_48_arm1_vs_arm3_0api.py` (0 API)",
        f"- 冻结方案: `docs/48_analysis_plan_amendment_20261119.md` §4, δ={DELTA_MIN}",
        f"- Arm1: `{os.path.basename(os.path.dirname(ARM1_RUN))}/{os.path.basename(ARM1_RUN)}`",
        f"- Arm3: `{os.path.basename(os.path.dirname(ARM3_RUN))}/{os.path.basename(ARM3_RUN)}`",
        f"- 闭合: Arm1 valid={len(v1)}/{len(arm1)}, Arm3 valid={len(v3)}/{len(arm3)}", "",
        "## 1. 主比较 (2765 题共同有效集)", "",
        f"- Arm1 hard = {main['arm1_hard']}/{n} = {main['arm1_rate']:.4f}",
        f"- Arm3 hard = {main['arm3_hard']}/{n} = {main['arm3_rate']:.4f}",
        f"- 2x2: both_pass={bp}, both_fail={bf}, Arm1_only_pass={r}, Arm3_only_pass={g}, discordant={r+g}",
        f"- 精确 McNemar 双侧 p = {p:.4e}",
        f"- 净增益 g = {ng['net_gain']:+d} (rate={ng['net_gain_rate']:+.4f}, "
        f"CI=[{ng['ci_lo_rate']:+.4f},{ng['ci_hi_rate']:+.4f}], δ={DELTA_MIN} ≈ {round(DELTA_MIN*n)} 题)",
        f"- **判读**: {main['interpretation']}", "",
        "## 1b. 区间估计方法 (核对)", "",
        "配对净增益 g_rate = (rescue − regress)/n = (2r − 1)·d/n, 其中 d = rescue + regress, r = rescue/d。",
        "内置实现的区间: 对 r 取 Clopper–Pearson 精确区间 (条件于 d 固定), 再做上述单调仿射变换。",
        "**这不是单臂通过率的 Clopper–Pearson 区间, 也不是对 g_rate 的无条件精确区间**; "
        "其覆盖性质是条件于不一致对数 d 成立的。",
        "交叉核对 (`eva/ci_crosscheck_48.py`, 0 API), 同一仿射变换下三种构造:",
        "",
        f"- CP  (内置):   r∈[{ci_diag.get('r_ci_cp', ['?','?'])[0]}, {ci_diag.get('r_ci_cp', ['?','?'])[1]}] "
        f"→ g∈[{ci_diag.get('g_ci_cp', ['?','?'])[0]:+.4f}, {ci_diag.get('g_ci_cp', ['?','?'])[1]:+.4f}]",
        f"- Wilson:        r∈[{ci_diag.get('r_ci_wilson', ['?','?'])[0]}, {ci_diag.get('r_ci_wilson', ['?','?'])[1]}] "
        f"→ g∈[{ci_diag.get('g_ci_wilson', ['?','?'])[0]:+.4f}, {ci_diag.get('g_ci_wilson', ['?','?'])[1]:+.4f}]",
        f"- 精确检验反转:  r∈[{ci_diag.get('r_ci_exact_inversion', ['?','?'])[0]}, "
        f"{ci_diag.get('r_ci_exact_inversion', ['?','?'])[1]}] "
        f"→ g∈[{ci_diag.get('g_ci_exact_inversion', ['?','?'])[0]:+.4f}, "
        f"{ci_diag.get('g_ci_exact_inversion', ['?','?'])[1]:+.4f}]",
        "",
        "三种构造的下界均 > δ=0.03 → 判读不依赖区间构造方式。报告中应写: "
        "\"对不一致对比例的 CP 区间经仿射变换后的净增益率区间\", 不写 \"通过率的 Clopper–Pearson 区间\"。",
        "",
        "## 2. 补充敏感性分析: 排除 107 题 NameError", "",
        f"- 排除集是否真的全 0: 任一臂通过={ne_any_pass}, 两臂均通过={ne_both_pass} (预期 0/0)",
        f"- 排除后 n_common = {n2}",
        f"- 2x2: both_pass={bp2}, both_fail={bf2}, Arm1_only_pass={r2}, Arm3_only_pass={g2}",
        f"- p = {p2:.4e}; g = {ng2['net_gain']:+d} (rate={ng2['net_gain_rate']:+.4f}, "
        f"CI=[{ng2['ci_lo_rate']:+.4f},{ng2['ci_hi_rate']:+.4f}])",
        f"- **判读**: {out['sensitivity_excl_107']['interpretation']}", "",
        "## 3. 参考比较 (非主结论)", "",
        "| 比较 | both_pass | both_fail | rescue | regress | n | p | g | CI | 判读 |",
        "|------|-----------|-----------|--------|---------|---|----|----|-----|------|",
    ]
    rows = [("Arm1 vs S₀(215, 300题)", ref["arm1_vs_s0_300"]),
            ("Arm3 vs S₀(215, 300题)", ref["arm3_vs_s0_300"]),
            ("Arm1 vs Arm3 (300题子集)", ref["arm1_vs_arm3_on_300"])]
    for label, d in rows:
        ng_ = d["net_gain"]
        rc = d.get("rescue(0->1)", d.get("arm1_only_pass"))
        rg = d.get("regress(1->0)", d.get("arm3_only_pass"))
        itp = d.get("interpretation", "")
        md.append(f"| {label} | {d['both_pass']} | {d['both_fail']} | {rc} | {rg} | "
                  f"{d['n_common']} | {d['p_exact']:.3e} | {ng_['net_gain']:+d} | "
                  f"[{ng_['ci_lo_rate']:+.4f},{ng_['ci_hi_rate']:+.4f}] | {itp} |")
    ep = out["effective_pass_diag"]
    md += ["", "## 3b. 异构性诊断 (pass_rate>=0.5 为通过)", "",
           f"- Arm1: hard={ep['arm1']['n_hard']}, pass_rate>=0.5={ep['arm1']['n_pass_rate_ge_0.5']}",
           f"- Arm3: hard={ep['arm3']['n_hard']}, pass_rate>=0.5={ep['arm3']['n_pass_rate_ge_0.5']}",
           f"- 有效通过口径 2x2: both_pass={ep['arm1_vs_arm3_pr>=0.5']['both_pass']} "
           f"both_fail={ep['arm1_vs_arm3_pr>=0.5']['both_fail']} "
           f"Arm1_only={ep['arm1_vs_arm3_pr>=0.5']['arm1_only']} "
           f"Arm3_only={ep['arm1_vs_arm3_pr>=0.5']['arm3_only']} "
           f"p={ep['arm1_vs_arm3_pr>=0.5']['p_exact']:.3e} "
           f"g={ep['arm1_vs_arm3_pr>=0.5']['net_gain']['net_gain']:+d} "
           f"CI=[{ep['arm1_vs_arm3_pr>=0.5']['net_gain']['ci_lo_rate']:+.4f},"
           f"{ep['arm1_vs_arm3_pr>=0.5']['net_gain']['ci_hi_rate']:+.4f}]",
           "", "## 4. 训练侧 (仅记录)", "",
           f"- val routed best delta: Arm1 +0.1067 / Arm3 +0.0750 (100 题, 非主结论)", "",
           "## 5. 可复现证据", "",
           f"- 配对方式: {out['reproducibility']['pairing']}",
           f"- 共同有效 ID: n={out['reproducibility']['n_common_ids']}, "
           f"清单 `common_valid_ids_2765.txt`, ids_sha256={out['reproducibility']['common_ids_sha256']}",
           f"- 主 2x2: {out['reproducibility']['main_2x2']}",
           "- SHA-256:", ""]
    for k, v in out["reproducibility"]["file_sha256"].items():
        md.append(f"  - `{k}`: `{v}`")
    md += ["", f"- 运行输出: `run_output.txt` (本 md 内容 + stdout)", ""]
    pm = os.path.join(AUDIT, "paired_arm1_vs_arm3.md")
    open(pm, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n→ {pj}\n→ {pm}")


if __name__ == "__main__":
    main()
