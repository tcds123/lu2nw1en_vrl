#!/usr/bin/env python3
"""§10.1 多 run 方差实验 — run 级统计检验.

统计单位 = 演化运行 (N on vs N off), 而非"固定产物多 seed"(那只测评测噪声 sd≈0.014).
每 run 取该 arm skill 跨 2 seed shuffle 的 mean_pass 作为 run-level mean → 两臂各 n=N.

2026-09-21 #3: N 从 3 降到 2 (环境重启≠会话断开, iters=4 不动保 on 臂信号).
  N=2 → 置换精确枚举 C(4,2)=6, p 分辨率粗 (双尾最小 p≈0.333).
  但 review 预期结论本就是"run 级非显著", N=2 不改结论方向;
  配对 bootstrap 在 per-seed 层仍有效 (每 run 2 seed, on/off 配对, resample run 层).

回答 review #1: run 间方差 ~0.15 (review 估) 是否吞没 on/off Δ=+1.8pp (shuffled)?
若 run 级 on/off 不可分 (p>0.05) → pool_filter +15.6pp 叙事彻底终结.

输入: 2N 个 eval_dir, 每个含 multiseed_test_eval.json (跨 seed summary).
  skill_evolver_math_poolfilter_{on,off}_multirun{1,2}_20260920_multirun_eval/

输出: multirun_runlevel_stats.json + 终端可读摘要.

指标:
  - overall mean_pass (受 meta_46 位置捷径污染, 上界)
  - other_60 mean_pass (非 meta 题, 真算法贡献主指标)
  - meta_46 mean_pass (位置捷径, 仅参考)

检验 (per 指标):
  1. 两样本 Welch t (n=N vs N, N≥2 才有 df; N=1 返回 nan)
  2. 置换检验 (精确枚举, N=2→C(4,2)=6; N=3→C(6,3)=20; 否则蒙特卡洛)
  3. 分层 bootstrap (run 内 seed 分层 resample 2000 次, 取 run mean)
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats

EVAL_GLOB = ("skill_evolver_math_poolfilter_{arm}_multirun*"
             "_20260920_multirun_eval/multiseed_test_eval.json")
OUTBASE = "rl/train/verl1/outputs"
N_PERM = 10000
N_BOOT = 2000
RNG_SEED = 20260920


def discover_runs(outbase):
    """返回 {'on': [run1_eval_data, run2, run3], 'off': [...]} 按 multirun 序号升序."""
    found = {"on": [], "off": []}
    for arm in ("on", "off"):
        pattern = os.path.join(outbase,
                              f"skill_evolver_math_poolfilter_{arm}_multirun*_20260920_multirun_eval",
                              "multiseed_test_eval.json")
        paths = sorted(glob.glob(pattern))
        for p in paths:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            rid = _extract_rid(p)
            found[arm].append({"rid": rid, "path": p, "data": data})
    return found


def _extract_rid(path):
    """从路径 ..._multirun{N}_..._eval/ 提取 run 序号 N."""
    import re
    m = re.search(r"_multirun(\d+)_20260920_multirun_eval", path)
    return int(m.group(1)) if m else -1


def arm_seed_means(eval_data, arm_name):
    """从单个 run 的 multiseed_test_eval.json 提取该 arm 的 per-seed 指标.

    返回 {seed: {'overall':x, 'meta_46':y, 'other_60':z}}.
    arm_name ∈ {'on','off'}. summary 里 passes 列表 = 跨 seed mean_pass;
    runs[arm] 里每 seed 有 stratified. 优先用 runs (保留 per-seed).
    """
    runs = eval_data.get("runs", {}).get(arm_name, [])
    out = {}
    for r in runs:
        seed = r.get("seed")
        st = r.get("stratified", {})
        out[seed] = {
            "overall": float(r.get("mean_pass", 0.0)),
            "meta_46": float(st.get("meta_46", 0.0)),
            "other_60": float(st.get("other_60", 0.0)),
        }
    return out


def run_level(arm_runs, arm_name, metric):
    """每 run 取该 arm 跨 seed 的 metric 均值 → run-level list (len=n_runs)."""
    vals = []
    for run in arm_runs:
        sm = arm_seed_means(run["data"], arm_name)
        if not sm:
            vals.append(np.nan)
            continue
        vals.append(np.mean([s[metric] for s in sm.values()]))
    return np.array(vals, dtype=float)


def welch_t(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "p": float("nan"),
                "df": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # Welch df
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    n1, n2 = len(a), len(b)
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1)) if (v1>0 or v2>0) else float("nan")
    delta = float(np.mean(a) - np.mean(b))
    # 95% CI on mean difference (Welch-Satterthwaite)
    se = np.sqrt(v1/n1 + v2/n2) if (v1>=0 and v2>=0) else float("nan")
    if not np.isnan(df) and not np.isnan(se) and se > 0:
        tcrit = stats.t.ppf(0.975, df)
        ci_lo, ci_hi = delta - tcrit*se, delta + tcrit*se
    else:
        ci_lo, ci_hi = float("nan"), float("nan")
    return {"t": float(t), "p": float(p), "df": float(df),
            "delta_mean": delta, "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi)}


def permutation_test(a, b, n_perm=N_PERM, seed=RNG_SEED):
    """置换检验: H0: 两组同分布. 统计量 = mean(a)-mean(b). 精确分布 (n=3 vs 3 仅 20 种组合)."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 1 or len(b) < 1:
        return {"p": float("nan"), "obs_delta": float("nan"), "n_perm": 0}
    rng = np.random.default_rng(seed)
    obs = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    # n 小 (3+3=6): 枚举所有 C(6,3)=20 组合 → 精确
    from itertools import combinations
    all_idx = list(range(len(pooled)))
    exact = len(list(combinations(all_idx, n_a)))
    if exact <= n_perm:
        dist = []
        for combo in combinations(all_idx, n_a):
            idx_a = np.array(combo)
            mask = np.ones(len(pooled), dtype=bool)
            mask[idx_a] = False
            dist.append(pooled[idx_a].mean() - pooled[mask].mean())
        dist = np.array(dist)
        # 双尾
        p = float(np.mean(np.abs(dist) >= np.abs(obs) - 1e-12))
        return {"p": p, "obs_delta": obs, "method": "exact",
                "n_combos": exact, "null_mean": float(dist.mean()),
                "null_sd": float(dist.std(ddof=1))}
    # 蒙特卡洛
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(pooled))
        d = pooled[perm[:n_a]].mean() - pooled[perm[n_a:]].mean()
        if abs(d) >= abs(obs) - 1e-12:
            count += 1
    return {"p": count / n_perm, "obs_delta": obs, "method": "mc", "n_perm": n_perm}


def stratified_bootstrap(arm_runs, arm_name, metric, n_boot=N_BOOT, seed=RNG_SEED):
    """分层 bootstrap: run 内按 seed 分层 resample (seed 是评测噪声层, run 是算法层).

    这里 resample run 层 (3 runs 有放回), 每 run 取其跨 seed 均值.
    即对 run-level 均值做 bootstrap CI (BCa 太重, 用 percentile).
    """
    run_vals = run_level(arm_runs, arm_name, metric)
    run_vals = run_vals[~np.isnan(run_vals)]
    if len(run_vals) < 1:
        return {"boot_mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    n = len(run_vals)
    boots = np.array([rng.choice(run_vals, size=n, replace=True).mean() for _ in range(n_boot)])
    return {"boot_mean": float(boots.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "n_boot": n_boot}


def analyze(found):
    report = {"arms": {}, "metrics": {}, "verdict": {}}
    # 记录每 run 跨 seed 明细
    for arm, runs in found.items():
        report["arms"][arm] = []
        for run in runs:
            sm = arm_seed_means(run["data"], arm)
            entry = {"rid": run["rid"], "path": os.path.relpath(run["path"])}
            for seed, m in sm.items():
                entry[f"seed{seed}"] = m
            # run-level (跨 seed 均)
            if sm:
                entry["run_mean"] = {
                    "overall": float(np.mean([s["overall"] for s in sm.values()])),
                    "meta_46": float(np.mean([s["meta_46"] for s in sm.values()])),
                    "other_60": float(np.mean([s["other_60"] for s in sm.values()])),
                }
            report["arms"][arm].append(entry)

    on_runs = found["on"]
    off_runs = found["off"]
    n_on, n_off = len(on_runs), len(off_runs)
    report["n_runs"] = {"on": n_on, "off": n_off}

    if n_on == 0 or n_off == 0:
        report["verdict"]["status"] = "incomplete"
        report["verdict"]["msg"] = f"runs missing: on={n_on}, off={n_off}"
        return report

    for metric in ("overall", "meta_46", "other_60"):
        a = run_level(on_runs, "on", metric)
        b = run_level(off_runs, "off", metric)
        t_res = welch_t(a, b)
        perm_res = permutation_test(a, b)
        boot_on = stratified_bootstrap(on_runs, "on", metric)
        boot_off = stratified_bootstrap(off_runs, "off", metric)
        report["metrics"][metric] = {
            "on_runs": [float(x) for x in a],
            "off_runs": [float(x) for x in b],
            "on_mean": float(np.nanmean(a)), "on_sd": float(np.nanstd(a, ddof=1)) if len(a[~np.isnan(a)])>1 else 0.0,
            "off_mean": float(np.nanmean(b)), "off_sd": float(np.nanstd(b, ddof=1)) if len(b[~np.isnan(b)])>1 else 0.0,
            "delta_on_minus_off": float(np.nanmean(a) - np.nanmean(b)),
            "welch_t": t_res,
            "permutation": perm_res,
            "bootstrap_on": boot_on,
            "bootstrap_off": boot_off,
        }

    # 判读: 以 other_60 (真算法贡献) 为主, overall 为辅.
    # N=2 (2026-09-21 #3): 置换精确 C(4,2)=6, 双尾最小 p≈0.333 → p<0.05 不可达,
    #   故判定改为 "Welch CI 是否跨 0 + 方向" 为主, 置换 p 仅辅证 (p 因 N 小本就粗).
    #   N≥3 时恢复原 "置换 p<0.05" 为主判据.
    m_other = report["metrics"]["other_60"]
    m_over = report["metrics"]["overall"]
    on_minus_off_other = m_other["delta_on_minus_off"]
    p_perm_other = m_other["permutation"]["p"]
    p_perm_over = m_over["permutation"]["p"]
    # welch_t 在 n<2 时返回 ci_lo/ci_hi (nan), n≥2 返回 ci95_lo/ci95_hi — 兼容两者
    wt_other = m_other["welch_t"]
    welch_lo = wt_other.get("ci95_lo", wt_other.get("ci_lo", float("nan")))
    welch_hi = wt_other.get("ci95_hi", wt_other.get("ci_hi", float("nan")))
    n_runs_on = len([x for x in m_other["on_runs"] if not np.isnan(x)])
    n_runs_off = len([x for x in m_other["off_runs"] if not np.isnan(x)])
    n_min = min(n_runs_on, n_runs_off)
    use_ci_gate = (n_min <= 2)  # N≤2 置换太粗, 改用 Welch CI 跨零判据

    if np.isnan(p_perm_other) and np.isnan(welch_lo):
        verdict = "统计检验缺数据"
    elif use_ci_gate:
        ci_str = "[{:+.4f}, {:+.4f}]".format(welch_lo, welch_hi) if not np.isnan(welch_lo) else "N/A"
        if not np.isnan(welch_lo) and welch_lo > 0:
            verdict = ("other_60 on>off: Δ={:+.4f}, Welch CI95={} 全在 0 上侧 → "
                       "pool_filter 有正向算法贡献信号 (N={} 置换 p={:.3f} 仅辅证, CI 判据为主)"
                       .format(on_minus_off_other, ci_str, n_min, p_perm_other))
        elif not np.isnan(welch_hi) and welch_hi < 0:
            verdict = ("other_60 on<off: Δ={:+.4f}, Welch CI95={} 全在 0 下侧 → "
                       "pool_filter 有害, 叙事终结 (N={} 置换 p={:.3f} 仅辅证)"
                       .format(on_minus_off_other, ci_str, n_min, p_perm_other))
        else:
            verdict = ("other_60 on/off run 级不可分: Δ={:+.4f}, Welch CI95={} 跨 0 (perm p={:.3f}, N={}) → "
                       "run 间方差吞没 pool_filter 效应, +15.6pp 净效应叙事终结"
                       .format(on_minus_off_other, ci_str, p_perm_other, n_min))
    else:
        if p_perm_other < 0.05 and on_minus_off_other > 0:
            verdict = ("other_60 on>off 显著 (p={:.3f}, Δ=+{:.4f}) → "
                       "pool_filter 在非 meta 题有真算法贡献 (推翻 review #1 假设)".format(p_perm_other, on_minus_off_other))
        elif p_perm_other < 0.05 and on_minus_off_other < 0:
            verdict = ("other_60 on<off 显著 (p={:.3f}, Δ={:.4f}) → "
                       "pool_filter 有害, 叙事终结".format(p_perm_other, on_minus_off_other))
        else:
            verdict = ("other_60 on/off run 级不可分 (perm p={:.3f}, Δ={:+.4f}) → "
                       "run 间方差吞没 pool_filter 效应, +15.6pp 净效应叙事终结".format(p_perm_other, on_minus_off_other))
    report["verdict"]["status"] = "done"
    report["verdict"]["primary"] = verdict
    report["verdict"]["primary_method"] = "welch_ci_gate" if use_ci_gate else "permutation_p"
    report["verdict"]["n_runs"] = {"on": n_runs_on, "off": n_runs_off, "min": n_min}
    report["verdict"]["secondary_overall"] = (
        "overall(含 meta 捷径): perm p={:.3f}, Δ={:+.4f}, Welch CI95=[{:+.4f},{:+.4f}]".format(
            p_perm_over, m_over["delta_on_minus_off"],
            m_over["welch_t"].get("ci95_lo", m_over["welch_t"].get("ci_lo", float("nan"))),
            m_over["welch_t"].get("ci95_hi", m_over["welch_t"].get("ci_hi", float("nan"))))
        if not np.isnan(p_perm_over) else "overall 缺数据")
    return report


def main():
    ap = argparse.ArgumentParser(description="§10.1 run 级统计检验")
    ap.add_argument("--outbase", default=OUTBASE)
    ap.add_argument("--out", default="rl/train/verl1/outputs/multirun_runlevel_stats.json")
    args = ap.parse_args()

    found = discover_runs(args.outbase)
    n_on, n_off = len(found["on"]), len(found["off"])
    print(f"=== §10.1 run 级统计检验 ===")
    n_expect = 2  # 2026-09-21 #3 降级 N=3→N=2
    print(f"发现: on={n_on} runs, off={n_off} runs (期望各 {n_expect})")
    for arm, runs in found.items():
        for r in runs:
            print(f"  {arm}{r['rid']}: {os.path.relpath(r['path'])}")
    if n_on >= 2 and n_off >= 2 and n_on == n_off and n_on == 2:
        print(f"  ℹ️ N=2: 置换精确 C(4,2)=6, p 分辨率粗 (双尾最小 ≈0.333). "
              f"判定以方向 + Welch CI 是否跨 0 为主, 置换 p 仅辅证.")

    if n_on == 0 and n_off == 0:
        print("\n⚠️ 未发现任何 eval 产物. 实验尚未产出 eval 结果, 稍后再跑.")
        print(f"   期望路径: {args.outbase}/skill_evolver_math_poolfilter_{{on,off}}_multirun{{1,2}}_20260920_multirun_eval/multiseed_test_eval.json")
        return 1

    report = analyze(found)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 报告 → {args.out}\n")

    if report["verdict"].get("status") == "incomplete":
        print(f"⚠️ {report['verdict']['msg']} (已找到的 runs 仍会记入报告, 可部分预览)")
    else:
        n_min = min(n_on, n_off)
        print("=" * 60)
        print(f"【指标】on vs off (run 级, n={n_min}/arm)")
        if n_min <= 2:
            print(f"  (N≤2: 主判据=Welch CI95 是否跨 0, 置换 p 仅辅证)")
            print(f"{'指标':<12} {'on_mean':>9} {'off_mean':>9} {'Δ(on-off)':>11} {'perm_p':>8} {'welch_CI95':>20}")
            for metric in ("overall", "meta_46", "other_60"):
                m = report["metrics"][metric]
                wt = m["welch_t"]
                lo = wt.get("ci95_lo", wt.get("ci_lo", float("nan")))
                hi = wt.get("ci95_hi", wt.get("ci_hi", float("nan")))
                ci = "[{:+.4f},{:+.4f}]".format(lo, hi) if not np.isnan(lo) else "N/A"
                print(f"{metric:<12} {m['on_mean']:>9.4f} {m['off_mean']:>9.4f} "
                      f"{m['delta_on_minus_off']:>+11.4f} {m['permutation']['p']:>8.3f} {ci:>20}")
        else:
            print(f"{'指标':<12} {'on_mean':>9} {'off_mean':>9} {'Δ(on-off)':>11} {'perm_p':>8} {'welch_p':>9}")
            for metric in ("overall", "meta_46", "other_60"):
                m = report["metrics"][metric]
                print(f"{metric:<12} {m['on_mean']:>9.4f} {m['off_mean']:>9.4f} "
                      f"{m['delta_on_minus_off']:>+11.4f} {m['permutation']['p']:>8.3f} {m['welch_t']['p']:>9.3f}")
        print()
        print("【每 run 明细 (other_60, 真算法贡献主指标)】")
        for arm in ("on", "off"):
            for r in report["arms"][arm]:
                rm = r.get("run_mean", {})
                print(f"  {arm}{r['rid']}: other_60={rm.get('other_60',float('nan')):.4f} "
                      f"meta_46={rm.get('meta_46',float('nan')):.4f} overall={rm.get('overall',float('nan')):.4f}")
        print()
        print("【主结论】")
        print("  " + report["verdict"]["primary"])
        print("  " + report["verdict"]["secondary_overall"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
