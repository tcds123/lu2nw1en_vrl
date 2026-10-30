"""B4 — cross_q_consistency 假设检验 (零 API 离线脚本)。

读 evolver 的 paired_eval_samples.jsonl, 对每个 (stage, iter, 候选 skill_version) 算题级方向一致率。

预登记见 memory/b4-preregistration-20261029.md, 判定结局 (a)/(b)/(c) 写报告。

两个口径 (2026-10-29 修正, 首版只报了全题口径):
  - **all**: 全部题 (含 base=cand=0 的"双方都不会"题)。这些题 delta 恒为 0, 不进 pos/neg 分母,
    但其存在本身由题目难度决定, 与候选无关 → 全题口径的一致性会随 base 难度漂移。
  - **informative** (主指标): 只取 base 通过率 > 0 的题 (即候选把某题从"能过"变成"不能过了"
    或反过来才可观测)。这才是 cross_q_consistency 想度量的事。

用法:
    python -m rl.train.verl1.skill_evolution.b4_cross_q_consistency --samples <path> [--json out]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict

# held-out 段的 stage 名与训练段不对应, 手工映射 (来自 evolver_log 的 gate 记录)
HELDOUT_PARENT = {
    "heldout_facet_edit": ("evolver_facet_paired", 1),
    "heldout_overlay": ("evolver_overlay_paired", 3),
    "heldout_expand": ("evolver_expand_paired", 4),
}


def load_candidates(samples_path: str):
    """返回 list[dict]: 每个 evolver 候选一条, 含 per-item {qid: (cand_rate, base_rate)}。

    跳过 `*_base` 单行哨兵 (那 1 行是 eval_paired 的 stage 探针, score=0, 不是一条评测)。
    """
    cand = defaultdict(lambda: defaultdict(list))
    base = defaultdict(lambda: defaultdict(list))
    dropped = defaultdict(int)
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            stage, role = r["stage"], r["skill_role"]
            if stage.endswith("_base"):
                continue  # 单行哨兵
            qid, it = r["item_id"], r["iter"]
            if r.get("parse_status") == "abandoned" or r.get("score") is None:
                dropped[(stage, it)] += 1
                continue
            if role == "cand":
                cand[(stage, it, r["skill_version"])][qid].append(float(r["score"]))
            else:
                base[(stage, it)][qid].append(float(r["score"]))

    mean = lambda v: sum(v) / len(v)
    out = []
    for (stage, it, sver), cdict in cand.items():
        bkey = (stage, it)
        if bkey not in base:
            continue
        bm = {q: mean(v) for q, v in base[bkey].items()}
        pairs = {q: (mean(v), bm[q]) for q, v in cdict.items() if q in bm}
        if not pairs:
            continue
        out.append({
            "stage": stage, "iter": it, "cand": sver,
            "kind": "heldout" if stage.startswith("heldout_") else "train",
            "pairs": pairs, "n_dropped": dropped[bkey],
        })
    return out


def _stats(deltas):
    eps = 1e-9
    pos = sum(1 for d in deltas if d > eps)
    neg = sum(1 for d in deltas if d < -eps)
    n = len(deltas)
    denom = pos + neg
    return {
        "n": n, "pos": pos, "neg": neg, "zero": n - pos - neg,
        "consistency": (pos / denom) if denom else None,
        "mean_delta": (sum(deltas) / n) if n else 0.0,
        "sign_p": _sign_test(pos, neg),
    }


def cross_q_consistency(pairs: dict, informative_only: bool = True):
    """pairs: {qid: (cand_rate, base_rate)} → stats dict。"""
    ds = [c - b for c, b in pairs.values() if (b > 0 or not informative_only)]
    return _stats(ds)


def _sign_test(pos: int, neg: int) -> float:
    """精确二项双尾 p (H0: consistency=0.5)。"""
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _identity_null(n_items: int, informative_frac: float, n_draws: int = 2000, seed: int = 0):
    """identity 候选的理论零分布: 真效应=0, 两臂各自 k=8 二项采样。
    条件在 base 的经验难度上 (从 rep5 的 base 每题 pass rate 重采样)。
    返回 (median, p95, p99) of |consistency - 0.5|。
    """
    rng = random.Random(seed)
    devs = []
    for _ in range(n_draws):
        ds = []
        n_info = max(1, int(round(n_items * informative_frac)))
        for _ in range(n_info):
            p = rng.uniform(0.1, 0.9)      # 信息题: base 在 (0,1) 之间
            c = sum(rng.random() < p for _ in range(8)) / 8.0
            b = sum(rng.random() < p for _ in range(8)) / 8.0
            ds.append(c - b)
        st = _stats(ds)
        if st["consistency"] is not None:
            devs.append(abs(st["consistency"] - 0.5))
    devs.sort()
    q = lambda f: devs[min(len(devs) - 1, int(f * len(devs)))]
    return q(0.5), q(0.95), q(0.99)


def classify(train_rows):
    cs = [(r["stage"], r["iter"], r["cand"], r["inf"]["consistency"])
          for r in train_rows if r["inf"]["consistency"] is not None]
    if not cs:
        return "(b) 无分离", "consistency 全 None"
    vals = [c for *_, c in cs]
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    in_band = sum(1 for c in vals if 0.40 <= c <= 0.60) / len(vals)
    extreme = sum(1 for c in vals if c > 0.8 or c < 0.2) / len(vals)
    if extreme >= 0.5:
        return "(c) 过度分离", f"极值占比 {extreme:.0%}"
    if rng >= 0.20 and in_band < 0.9:
        return "(a) 有分离", f"极差 {rng:.3f} ∈ [{lo:.3f}, {hi:.3f}]"
    return "(b) 无分离", f"{in_band:.0%} 落在 [0.40,0.60], 极差 {rng:.3f}"


def main():
    ap = argparse.ArgumentParser(description="B4 cross_q_consistency 假设检验")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = load_candidates(args.samples)
    for r in rows:
        r["inf"] = cross_q_consistency(r["pairs"], informative_only=True)
        r["all"] = cross_q_consistency(r["pairs"], informative_only=False)

    train = sorted([r for r in rows if r["kind"] == "train"], key=lambda r: (r["stage"], r["iter"], r["cand"]))
    held = sorted([r for r in rows if r["kind"] == "heldout"], key=lambda r: (r["stage"], r["iter"], r["cand"]))
    idents = {r["cand"] for r in train if r["stage"] == "evolver_overlay_paired"}

    print(f"\n数据: {args.samples}")
    print(f"候选: train={len(train)}  heldout={len(held)}   "
          f"(identity 候选 {sum(1 for r in train if r['cand'] in idents)} 个)\n")
    print("=== 训练集候选   [主口径 informative: 只算 base>0 的题] ===")
    print(f"{'stage':<24}{'it':>3} {'cand':<11}{'nI':>4}{'pos':>4}{'neg':>4}{'cons':>7}"
          f"{'meanΔ_I':>9}{'p':>7} |{'n_all':>6}{'cons_all':>9}{'meanΔ_all':>10}")
    for r in train:
        i, a = r["inf"], r["all"]
        cop = lambda s: f"{s['consistency']:.3f}" if s['consistency'] is not None else "  -  "
        print(f"{r['stage']:<24}{r['iter']:>3} {r['cand']:<11}{i['n']:>4}{i['pos']:>4}{i['neg']:>4}"
              f"{cop(i):>7}{i['mean_delta']:+9.4f}{i['sign_p']:>7.3f} |{a['n']:>6}{cop(a):>9}"
              f"{a['mean_delta']:+10.4f}{'  ★' if r['cand'] in idents else ''}")

    verdict, reason = classify(train)
    print(f"\n=== 预登记判据 → 结局 {verdict} ===\n理由: {reason}")

    # identity 零分布
    ident_rows = [r for r in train if r["cand"] in idents]
    if ident_rows:
        print(f"\n=== identity 候选 (真效应 = 0, n={len(ident_rows)}) ===")
        devs = [abs(r["inf"]["consistency"] - 0.5) for r in ident_rows if r["inf"]["consistency"] is not None]
        md = [r["inf"]["mean_delta"] for r in ident_rows]
        print(f"  |cons-0.5|: {[round(d,3) for d in devs]}")
        print(f"  meanΔ_I:    {[round(x,4) for x in md]}   均值 {sum(md)/len(md):+.4f}")
        med, p95, p99 = _identity_null(30, informative_frac=len(devs) and
                                       sum(1 for _ in ident_rows) and
                                       (sum(r["inf"]["n"] for r in ident_rows) / len(ident_rows)) / 30)
        n_beyond = sum(1 for d in devs if d > p95)
        print(f"  MC 零分布 (n_info≈{sum(r['inf']['n'] for r in ident_rows)/len(ident_rows):.0f},"
              f" 2000 draws): |cons-0.5| median={med:.3f} p95={p95:.3f} p99={p99:.3f}")
        print(f"  → 超出 p95 的 identity 候选: {n_beyond}/{len(devs)}"
              f"  {'⚠ 指标本身有偏 (超过纯采样噪声)' if n_beyond >= 2 else '✅ 与纯采样噪声一致'}")

    print("\n=== held-out 候选 (与训练集同 iter 父候选对照) ===")
    trmap = {(r["stage"], r["iter"], r["cand"]): r for r in train}
    for r in held:
        parent = HELDOUT_PARENT.get(r["stage"])
        tr = trmap.get((*parent, r["cand"])) if parent else None
        i = r["inf"]
        cs = f"{i['consistency']:.3f}" if i["consistency"] is not None else "  -  "
        tc = (f"{tr['inf']['consistency']:.3f}" if tr and tr["inf"]["consistency"] is not None else "-")
        print(f"{r['stage']:<20}{r['iter']:>3} {r['cand']:<11} nI={i['n']:>3} "
              f"cons={cs}  train_cons={tc}  meanΔ={i['mean_delta']:+.4f}")

    if args.json:
        payload = {
            "samples": args.samples, "verdict": verdict, "reason": reason,
            "train": [{k: r[k] for k in ("stage", "iter", "cand", "inf", "all", "n_dropped")} for r in train],
            "heldout": [{k: r[k] for k in ("stage", "iter", "cand", "inf", "all", "n_dropped")} for r in held],
            "identity_null": dict(zip(("median", "p95", "p99"), _identity_null(30, 0.85))),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已写: {args.json}")


if __name__ == "__main__":
    main()
