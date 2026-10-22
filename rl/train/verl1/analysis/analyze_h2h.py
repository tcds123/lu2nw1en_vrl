#!/usr/bin/env python3
"""h2h 8-seed evolver vs skillopt 配对分析 (同窗口同seed组).

守 CI-band 纪律: 只判方向 (CI_lower>0 / 跨0 / CI_upper<0), 不判量级.
两个判据互相印证:
  判据1 seed-level paired t  — 同seed组, 本次有效 (今早4-seed的无效因seed组不对齐)
  判据2 per-qid paired bootstrap (主) — 每题N-seed均值之差, bootstrap over 106题
用法: python rl/train/verl1/analyze_h2h.py [--n_seed 8] [--out_dir ...]
未跑满时自动用已有 seed 数算中间态 (会标注"中间态非结论").
"""
import argparse
import json
import math
import os
import random
import statistics

OUT_DIR = "outputs/tf_skill_math_v2/multiseed_test_h2h_0827"
EVO_H = "8238b5e8"
SK_H = "cfae156d"

# t 分布 0.975 分位 (双尾 95%)
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131}


def load_per(out_dir, name, hsh, nseed):
    pers = {}
    for s in range(1, nseed + 1):
        p = os.path.join(out_dir, f"per_{name}_{hsh}_seed{s}.json")
        if not os.path.exists(p):
            break
        pers[s] = json.load(open(p, encoding="utf-8"))
    return pers


def verdict(lo, hi):
    if lo > 0:
        return "CI下界>0 → evolver 显著优于 skillopt ✅"
    if hi < 0:
        return "CI上界<0 → evolver 显著落后于 skillopt 🔴"
    return "CI跨0 → 统计不可区分 🟡"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--n_seed", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evo_h", default=EVO_H, help="evolver skill hash (strip-md5 前8位); 默认 8238b5e8")
    ap.add_argument("--sk_h", default=SK_H, help="skillopt skill hash; 默认 cfae156d")
    args = ap.parse_args()

    evo = load_per(args.out_dir, "evolver", args.evo_h, args.n_seed)
    sk = load_per(args.out_dir, "skillopt", args.sk_h, args.n_seed)
    ne, nk = len(evo), len(sk)
    n = min(ne, nk)
    final = (ne >= args.n_seed and nk >= args.n_seed)
    print(f"{'='*60}")
    print(f"evolver {ne}/{args.n_seed} seed, skillopt {nk}/{args.n_seed} seed"
          + ("  [最终]" if final else "  [中间态, 非结论]"))
    print(f"{'='*60}")

    evo_sm = [sum(evo[s].values()) / len(evo[s]) for s in sorted(evo)]
    sk_sm = [sum(sk[s].values()) / len(sk[s]) for s in sorted(sk)]
    evo_m, sk_m = sum(evo_sm) / len(evo_sm), sum(sk_sm) / len(sk_sm)
    print(f"evolver  mean={evo_m:.4f}  per-seed={[round(m,4) for m in evo_sm]}")
    print(f"skillopt mean={sk_m:.4f}  per-seed={[round(m,4) for m in sk_sm]}")
    print(f"Δ(evo−sk) = {evo_m - sk_m:+.4f}  (点估计, 非CI)")

    # ── 判据1: seed-level paired t (同seed组, 本次有效) ──
    diffs = [evo_sm[i] - sk_sm[i] for i in range(n)]
    md = sum(diffs) / n
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    tc = T_CRIT.get(n - 1, 1.96)
    lo, hi = md - tc * se, md + tc * se
    print(f"\n── 判据1 seed-level paired t (n={n}, 同seed组) ──")
    print(f"  per-seed diff = {[round(d,4) for d in diffs]}")
    print(f"  mean={md:+.4f} sd={sd:.4f} SE={se:.4f}  t-CI(95%)=[{lo:+.4f},{hi:+.4f}]")
    print(f"  正/负 = {sum(1 for d in diffs if d>0)}/{sum(1 for d in diffs if d<0)}")
    print(f"  {verdict(lo, hi)}")

    # ── 判据2: per-qid paired bootstrap (主) ──
    common = sorted(set(evo[1]) & set(sk[1]))
    qd = []
    pos = neg = zero = 0
    for q in common:
        e = sum(evo[s][q] for s in evo) / len(evo)
        k = sum(sk[s][q] for s in sk) / len(sk)
        d = e - k
        qd.append(d)
        if d > 1e-9:
            pos += 1
        elif d < -1e-9:
            neg += 1
        else:
            zero += 1
    rng = random.Random(args.seed)
    boots = sorted(sum(rng.choices(qd, k=len(qd))) / len(qd) for _ in range(args.n_boot))
    blo = boots[int(0.025 * len(boots))]
    bhi = boots[int(0.975 * len(boots))]
    qm = sum(qd) / len(qd)
    print(f"\n── 判据2 per-qid paired bootstrap (主, n_boot={args.n_boot}) ──")
    print(f"  n_q={len(common)} 正/负/零={pos}/{neg}/{zero}")
    print(f"  qmean={qm:+.4f}  95%CI=[{blo:+.4f},{bhi:+.4f}]")
    print(f"  {verdict(blo, bhi)}")

    print(f"\n{'─'*60}")
    if not final:
        print("⚠️ 中间态: 上述为趋势参考, 非8-seed最终结论")
    print("守 CI-band 纪律: 只判方向, 真值量级分不出")


if __name__ == "__main__":
    main()
