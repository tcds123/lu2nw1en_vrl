#!/usr/bin/env python3
"""机制消融矩阵分析: FULL evolver vs 4 个 leave-one-out ablation.

读取 eval_test_multiseed 产出的 multiseed_test_eval.json (含每 skill 每 seed 的 per_qid),
计算每个机制的贡献 = FULL - ablation_i (正值=机制有用), 配对 t 检验 (per-qid, 跨 seed 池化).

机制映射:
  M1 (GEPA 候选种群+选择)   = FULL - A1(ablate_gepa)
  M2 (失败驱动 patch)       = FULL - A2(ablate_failure)
  M3 (慢更新 consolidate)   = FULL - A3(ablate_consolidate)
  M4 (防退化+多样性)        = FULL - A4(ablate_antideg)

输出: 控制台表格 + ablation_matrix.json
"""
import argparse
import json
import math
import os
from collections import defaultdict


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def paired_t_test(x_qid, y_qid):
    """配对 t 检验: x vs y, 按 qid 配对. 返回 (mean_diff, t, df, p_two_sided_approx).
    x_qid/y_qid: {qid: [pass values across seeds]}. 用每 qid 的均值做配对差."""
    qids = sorted(set(x_qid) & set(y_qid))
    if len(qids) < 2:
        return None
    diffs = [_mean(x_qid[q]) - _mean(y_qid[q]) for q in qids]
    n = len(diffs)
    m = _mean(diffs)
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else 0.0
    # 正态近似 p (n>=30 合理; 这里 n=test 题 ~106, 可接受)
    # 用 t 分布近似: 自由度 n-1, 双侧
    df = n - 1
    p = _t_sf_two_sided(t, df)
    return {"mean_diff": m, "t": t, "df": df, "p": p, "n": n}


def _t_sf_two_sided(t, df):
    """双侧 t 检验 p 值近似 (无 scipy). 用正态近似 + 小样本修正."""
    if df <= 0:
        return 1.0
    # 标准 normal CDF 近似
    def norm_cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    # t 分布尾部比正态宽, df 小时 p 偏大. 用 Peizer-Pratt 近似太繁, 这里用正态近似 + 修正系数
    z = abs(t)
    p_norm = 2 * (1 - norm_cdf(z))
    # 小样本放宽: df<30 时 p 乘一个放大因子 (粗略, 仅作参考; 精确 p 需 scipy)
    if df < 30:
        p_norm = min(1.0, p_norm * (1 + 0.5 * (30 - df) / 30))
    return p_norm


def collect_per_qid(runs):
    """runs: list of {seed, mean_pass, per_qid}. 返回 {qid: [pass per seed]}."""
    qid = defaultdict(list)
    for r in runs:
        for q, pr in r.get("per_qid", {}).items():
            qid[q].append(pr)
    return qid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", default="/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test_ablation/multiseed_test_eval.json")
    args = ap.parse_args()

    d = json.load(open(args.eval_json, encoding="utf-8"))
    runs = d.get("runs", {})
    summary = d.get("summary", {})

    # skill 名 -> 角色
    FULL = "evolver"  # FULL control = evolver 主结果 (12633201)
    ABL = {"abl_gepa": "M1(GEPA种群+选择)", "abl_failure": "M2(失败驱动patch)",
           "abl_consolidate": "M3(慢更新)", "abl_antideg": "M4(防退化+多样性)"}

    print(f"\n{'='*70}\n机制消融矩阵 (test={d.get('n_test')}, eval_k={d['args'].get('eval_k')}, seeds={d['args'].get('seeds')})\n{'='*70}")
    print(f"{'skill':<18} {'mean_pass':>10} {'ΔvsS0':>10} {'seeds':>8}")
    print("-" * 50)
    for name in ["s0", FULL] + list(ABL.keys()):
        if name not in summary:
            print(f"  {name:<16} {'(missing)':>10}")
            continue
        s = summary[name]
        dvs = s.get("delta_vs_s0", {}).get("mean", 0.0) if name != "s0" else 0.0
        print(f"  {name:<16} {s['mean_pass']:>10.4f} {dvs:>+10.4f} {s['n_seeds']:>8}")

    # 机制贡献 = FULL - ablation
    print(f"\n{'='*70}\n机制贡献 (FULL - ablation, 正值=机制有用)\n{'='*70}")
    print(f"{'机制':<28} {'FULL':>8} {'ablation':>10} {'Δ贡献':>10} {'t':>8} {'p':>8} {'n':>6}")
    print("-" * 80)
    full_qid = collect_per_qid(runs.get(FULL, []))
    full_pass = summary.get(FULL, {}).get("mean_pass", 0.0)
    matrix = {}
    for abl_name, mech in ABL.items():
        if abl_name not in runs or not runs[abl_name] or FULL not in runs:
            print(f"  {mech:<26} (数据缺失)")
            continue
        abl_qid = collect_per_qid(runs[abl_name])
        abl_pass = summary[abl_name]["mean_pass"]
        delta = full_pass - abl_pass
        tt = paired_t_test(full_qid, abl_qid)
        if tt:
            print(f"  {mech:<26} {full_pass:>8.4f} {abl_pass:>10.4f} {delta:>+10.4f} {tt['t']:>8.2f} {tt['p']:>8.3f} {tt['n']:>6}")
            matrix[abl_name] = {"mechanism": mech, "full_pass": full_pass, "ablation_pass": abl_pass,
                                "contribution": delta, "t": tt["t"], "p": tt["p"], "df": tt["df"], "n": tt["n"]}
        else:
            print(f"  {mech:<26} {full_pass:>8.4f} {abl_pass:>10.4f} {delta:>+10.4f} (n<2, 无 t 检验)")
            matrix[abl_name] = {"mechanism": mech, "full_pass": full_pass, "ablation_pass": abl_pass, "contribution": delta}

    out = {"args": d.get("args"), "n_test": d.get("n_test"),
           "skills": {k: {"mean_pass": v.get("mean_pass"), "delta_vs_s0": v.get("delta_vs_s0", {}).get("mean")}
                      for k, v in summary.items()},
           "mechanism_contribution": matrix}
    out_path = os.path.join(os.path.dirname(args.eval_json), "ablation_matrix.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n✅ 矩阵 -> {out_path}")

    # 结论速览
    print(f"\n结论速览:")
    for abl_name, m in matrix.items():
        contrib = m.get("contribution", 0)
        p = m.get("p", 1.0)
        sig = "显著" if p < 0.05 else ("临界" if p < 0.1 else "不显著")
        direction = "有用" if contrib > 0 else ("有害" if contrib < 0 else "中性")
        print(f"  {m['mechanism']}: Δ={contrib:+.4f} ({direction}, p={p:.3f} {sig})")


if __name__ == "__main__":
    main()
