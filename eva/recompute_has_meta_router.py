#!/usr/bin/env python3
"""
Recompute the has_meta routed signal (议 plan 763 的唯一 pre-computed 正向信号) 落盘重算.

背景: plan 方向 B/router 引用的 "routed(8238b5e8‖expand by has_meta)=0.5487, CI=[+0.0008,+0.0283]"
在仓库中查无实据 (仅存在于转录)。本脚本用项目自身已定型的 meta-option 分类启发式
(_META_Q_SIG 于 skill_evolver_math.py:125-127) 对 test=106 每 qid 标 has_meta,
再跑 routed paired bootstrap, 产出一个可审计的 {qid:has_meta} + routed 统计落盘文件。

零新 API: has_meta 标签用确定性关键词分类 (与计划 "meta-claim题'stronger result can be proven'" 同义),
不走 C 模型标注 (成本更低且更贴合项目已使用的 meta 定义)。
"""
import json
import io
import sys
import numpy as np
from scipy import stats

# --- has_meta 标签 (复用 skill_evolver_math 的 meta-option 签名, 零 API) ---
_META_Q_SIG = ("strongest", "can be proved", "can be proven",
               "a stronger result", "a stronger statement")
_META_OPT_SIG = ("one of the remaining",)


def has_meta(qtext: str, theorem: str, choices) -> bool:
    q = (str(qtext) + " " + str(theorem)).lower()
    opt_text = " ".join(str(c.get("text", "")) for c in choices).lower() if choices else ""
    return any(sig in q for sig in _META_Q_SIG) or any(sig in opt_text for sig in _META_OPT_SIG)


def paired_bootstrap(base_mp, routed_mp, n_boot=20000, seed=42):
    """per-qid paired bootstrap CI on mean(diff). base_mp/routed_mp: list of floats (per qid)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray([r - b for r, b in zip(routed_mp, base_mp)])
    n = len(arr)
    samples = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def load_eval(path):
    d = json.load(open(path))
    return {r["seed"]: r["per_qid"] for r in d["runs"]["evolver"]}


def avg_over_seeds(per_qid_list, qids):
    return {q: float(np.mean([r.get(q, np.nan) for r in per_qid_list])) for q in qids}


def main():
    out = {}
    base_path = "rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/multiseed_test_eval_8238b5e8.json"
    expand_path = "rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/multiseed_test_eval.json"
    items_path = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42/test/items.json"

    base_runs = load_eval(base_path)   # 8238b5e8 (overlay 基准)
    ex_runs = load_eval(expand_path)   # 1a63e45710 (expand 版)
    items = {i["id"]: i for i in json.load(open(items_path))}

    qids = sorted(base_runs[list(base_runs)[0]].keys())
    # 两个 eval 的 per_qid 都必须有全部 qid
    bq = set(qids) & set(ex_runs[list(ex_runs)[0]].keys())
    qids = sorted(bq)
    print(f"[has_meta router repro] qids overlaid = {len(qids)} / 106")

    # has_meta 标签 (确定性)
    meta = {q: has_meta(items[q]["question"], items[q].get("theorem", ""), items[q].get("choices")) for q in qids}
    n_meta = sum(meta.values())
    out["meta_label"] = meta
    out["meta_counts"] = {"has_meta": n_meta, "not_meta": len(qids) - n_meta}
    print(f"  has_meta={n_meta}  not_meta={len(qids)-n_meta}")

    # per-qid 4-seed 均值 (base 与 expand)
    base_av = avg_over_seeds(list(base_runs.values()), qids)
    ex_av = avg_over_seeds(list(ex_runs.values()), qids)
    out["per_qid_base_avg"] = base_av
    out["per_qid_expand_avg"] = ex_av

    # 路由: has_meta=True -> 用 base (8238b5e8); has_meta=False -> 用 expand
    routed = {q: (base_av[q] if meta[q] else ex_av[q]) for q in qids}
    out["per_qid_routed_avg"] = routed

    base_v = [base_av[q] for q in qids]
    routed_v = [routed[q] for q in qids]

    # 汇总 base vs routed
    mb, mr = float(np.mean(base_v)), float(np.mean(routed_v))
    out["summary"] = {
        "base_8238b5e8_mean": mb,
        "routed_mean": mr,
        "routed_delta_vs_base": mr - mb,
    }
    print(f"  base mean={mb:.4f}  routed mean={mr:.4f}  routed delta={mr-mb:.4f}")

    # routed vs base paired bootstrap (主判据 = 计划 CI=[+0.0008,+0.0283] 复算)
    dmean, dlo, dhi = paired_bootstrap(base_v, routed_v, n_boot=20000, seed=42)
    out["routed_vs_base_bootstrap"] = {
        "n_boot": 20000, "seed": 42, "mean": dmean,
        "95CI": [dlo, dhi], "ci_lower_gt_0": dlo > 0, "ci_upper_lt_0": dhi < 0,
    }
    print(f"  routed-vs-base bootstrap: mean={dmean:+.4f} CI=[{dlo:+.4f},{dhi:+.4f}] ci_lower_gt_0={dlo>0}")

    # 对照: 直接 base OR expand max (ensemble 上界) 而非 has_meta 路由
    ens = {q: max(base_av[q], ex_av[q]) for q in qids}
    ens_m = float(np.mean(list(ens.values())))
    out["ensemble_max_vs_base_bootstrap"] = {
        "ensemble_max_mean": ens_m,
        "delta_vs_base": ens_m - mb,
    }

    # --- 关键分层检验: not-meta 子集 上 expand vs base, 在 正确层级(per-seed) 复核 ---
    # 记忆: 全部 gain 来自 not-meta 17 题子集 (该子集 expand 稳定胜 base)。
    # 这里把该子集单独拎出来, 在 per-seed 层级(而非 per-qid bootstrap)做配对 t 检验,
    # 避免 n=17 的 per-qid 伪精细 CI 掩盖 per-seed 不显著。
    notmeta_qids = sorted(q for q in qids if not meta[q])
    per_seed_delta = []          # expand - base, 每个 seed 一个均值
    notmeta_per_qid_delta = {}   # {qid: [d_seed1..d_seed4]}
    for q in notmeta_qids:
        notmeta_per_qid_delta[q] = [float(ex_runs[sd][q] - base_runs[sd][q]) for sd in sorted(base_runs)]
    for sd in sorted(base_runs):  # seed=1..4
        per_seed_delta.append(float(np.mean([ex_runs[sd][q] - base_runs[sd][q] for q in notmeta_qids])))
    t_stat, p_val = stats.ttest_1samp(per_seed_delta, 0.0)
    out["not_meta_subset_per_seed"] = {
        "subsets": {"qids": notmeta_qids, "n": len(notmeta_qids)},
        "per_seed_delta_expand_minus_base": per_seed_delta,  # [+0.157,+0.098,+0.020,+0.059]
        "per_seed_mean_delta": float(np.mean(per_seed_delta)),
        "per_seed_paired_t": {"t": float(t_stat), "df": len(per_seed_delta) - 1,
                              "p": float(p_val), "p_lt_0_05": bool(p_val < 0.05),
                              "all_seeds_positive": bool(all(d > 0 for d in per_seed_delta))},
        "per_qid_delta_by_seed": notmeta_per_qid_delta,
    }
    print(f"[not-meta 子集 per-seed] n={len(notmeta_qids)} deltas={[f'{d:+.4f}' for d in per_seed_delta]} "
          f"mean={np.mean(per_seed_delta):+.4f} t({len(per_seed_delta)-1})={t_stat:.3f} p={p_val:.4f} "
          f"p_lt_0_05={p_val<0.05} all_seeds_pos={all(d>0 for d in per_seed_delta)}")

    # 落盘
    outpath = "rl/train/verl1/outputs/tf_skill_math_v2/has_meta_router_recompute.json"
    json.dump(out, open(outpath, "w"), indent=2, ensure_ascii=False)
    print(f"\n[落盘] {outpath}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()