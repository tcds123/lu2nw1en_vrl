#!/usr/bin/env python3
"""
expand-v2 (2f3c0c1f, 5段含Quantifier) vs 8238b5e8 (4段, 0.5354) 4-seed 复测统计.

判据 (见 memory/expand-v2-4seed-launch-20260902.md):
  ① 整体不再 -0.07: v1 (1a63e45710) 曾 CI 上界<0 显著退步 -0.0700.
     v2 目标 = per-qid paired bootstrap CI(n=20000) 跨 0 或 >0, per-seed paired t 不显著退步.
  ② not-meta 子集 (17题) 仍全正: per-seed delta 全>0, t检验达 p<0.05 (§六曾 p=0.0653 不显著).

零 API: 全部读 per_qid 缓存 (dict {qid: pass_rate}), has_meta 标签复用 has_meta_router_recompute.json.
口径自洽: 8238b5e8 / expand-v2 同 test=106, eval_k=3, seeds=1-4 完全对齐 → 可 paired.
"""
import json
import os
import numpy as np
from scipy import stats

BASE_DIR = "rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/"
HASH_EXPAND_V2 = "2f3c0c1f"
HASH_BASE = "8238b5e8"
META_CACHE = "rl/train/verl1/outputs/tf_skill_math_v2/has_meta_router_recompute.json"
N_BOOT = 20000


def load_per_qid(seed, hash8):
    f = os.path.join(BASE_DIR, f"per_evolver_{hash8}_seed{seed}.json")
    return json.load(open(f))


def paired_bootstrap(base_mp, routed_mp, n_boot=N_BOOT, seed=42):
    """per-qid paired bootstrap CI on mean(diff). 同 recompute_has_meta_router.py:31."""
    rng = np.random.default_rng(seed)
    arr = np.asarray([r - b for r, b in zip(routed_mp, base_mp)])
    n = len(arr)
    samples = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def per_seed_deltas(expand_runs, base_runs, qids):
    """每个 seed 的 mean(expand-base) over qids. 用于 per-seed paired t (ttest_1samp vs 0)."""
    return [float(np.mean([expand_runs[s][q] - base_runs[s][q] for q in qids]))
            for s in sorted(base_runs)]


def report_block(name, expand_runs, base_runs, qids):
    """整体/子集通用统计块: per-seed paired t + per-qid bootstrap + 汇总."""
    n_q = len(qids)
    # per-seed mean (整体), 配对 t
    expand_seeds = [float(np.mean([expand_runs[s][q] for q in qids])) for s in sorted(expand_runs)]
    base_seeds = [float(np.mean([base_runs[s][q] for q in qids])) for s in sorted(base_runs)]
    deltas = [e - b for e, b in zip(expand_seeds, base_seeds)]
    t_stat, p_val = stats.ttest_1samp(deltas, 0.0)

    # per-qid bootstrap (4-seed 平均后)
    ex_av = {q: float(np.mean([expand_runs[s][q] for s in sorted(expand_runs)])) for q in qids}
    ba_av = {q: float(np.mean([base_runs[s][q] for s in sorted(base_runs)])) for q in qids}
    ba_v = [ba_av[q] for q in qids]
    ex_v = [ex_av[q] for q in qids]
    dmean, dlo, dhi = paired_bootstrap(ba_v, ex_v)

    print(f"\n=== {name} (n_qid={n_q}, n_seed={len(deltas)}) ===")
    print(f"  base(8238b5e8) per-seed: {[f'{v:.4f}' for v in base_seeds]} mean={np.mean(base_seeds):.4f}")
    print(f"  expand-v2 per-seed:      {[f'{v:.4f}' for v in expand_seeds]} mean={np.mean(expand_seeds):.4f}")
    print(f"  per-seed delta (v2-base): {[f'{d:+.4f}' for d in deltas]} mean={np.mean(deltas):+.4f}")
    print(f"  per-seed paired t(1samp): t={t_stat:.3f} df={len(deltas)-1} p={p_val:.4f} "
          f"p_lt_0.05={p_val<0.05} all_pos={all(d>0 for d in deltas)}")
    print(f"  per-qid bootstrap(n={N_BOOT}): delta_mean={dmean:+.4f} "
          f"95%CI=[{dlo:+.4f},{dhi:+.4f}] ci_lower_gt_0={dlo>0} ci_upper_lt_0={dhi<0}")
    sig = "显著增益" if dlo > 0 else ("显著退步" if dhi < 0 else "不显著(跨0)")
    print(f"  方向判定: {sig}")
    return dict(base_seeds=base_seeds, expand_seeds=expand_seeds, deltas=deltas,
                t=float(t_stat), p=float(p_val), all_pos=bool(all(d > 0 for d in deltas)),
                bootstrap=dict(mean=dmean, ci=[dlo, dhi], sig=sig))


def main():
    expand_runs = {s: load_per_qid(s, HASH_EXPAND_V2) for s in range(1, 5)}
    base_runs = {s: load_per_qid(s, HASH_BASE) for s in range(1, 5)}
    qids = sorted(base_runs[1].keys())
    assert set(qids) == set(expand_runs[1].keys()), "题集不一致!"
    print(f"配对题数: {len(qids)} | seeds: 1-4")

    meta = json.load(open(META_CACHE))["meta_label"]
    notmeta_qids = sorted(q for q in qids if not meta[q])
    print(f"has_meta 标签复用: has_meta={sum(meta.values())} not_meta={len(notmeta_qids)}")

    out = {"expand_v2_hash": HASH_EXPAND_V2, "base_hash": HASH_BASE, "n_qid": len(qids),
           "not_meta_qids": notmeta_qids, "n_boot": N_BOOT}

    # 判据①: 整体 106 题
    out["overall"] = report_block("整体 (106题, 判据①)", expand_runs, base_runs, qids)
    # 判据②: not-meta 子集 17 题
    out["not_meta_subset"] = report_block("not-meta 子集 (17题, 判据②)", expand_runs, base_runs, notmeta_qids)

    # v1 对照回填 (memory 已知值, 供横向对比)
    out["v1_reference"] = {"v1_hash": "1a63e45710", "v1_delta": -0.0700,
                           "v1_verdict": "CI上界<0 显著退步"}

    outpath = "rl/train/verl1/outputs/tf_skill_math_v2/expand_v2_vs_8238_stat.json"
    json.dump(out, open(outpath, "w"), indent=2, ensure_ascii=False)
    print(f"\n[落盘] {outpath}")


if __name__ == "__main__":
    main()
