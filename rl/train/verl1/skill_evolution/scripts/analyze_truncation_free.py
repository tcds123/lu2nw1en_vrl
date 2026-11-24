#!/usr/bin/env python3
"""截断归零后的配对判定 (复盘 §4.5 配套).

输入: 同一实验的 1–2 轮 results.json (第二轮为 --only-truncated 补跑)
  python scripts/analyze_truncation_free.py <round1.json> [round2.json]
处理:
  - 同一 (arm, id) 取"最后一轮非截断样本"为准 (同一 cap、temp=0, 重跑只是换预算)
  - 报每组: 有效 n / 截断残留 / 答对 / 准确率
  - 每对 vs `none`: 配对 McNemar 精确检验 (b, c) + 准确率差 95% CI (Wilson 区间近似配对)
  - 天花板判定: none 的准确率 ≥ 0.95 或本轮任何组 100% 且至少 2 组 100% → 标 CEILING
输出: stdout 表格 + 同目录 analysis.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from itertools import combinations


def load(paths):
    """合并多轮: (arm, id) -> 记录; 截断样本被同键的非截断样本覆盖."""
    merged: dict[tuple[str, str], dict] = {}
    for p in paths:
        data = json.load(open(p, encoding="utf-8"))
        for rec in data["results"]:
            key = (rec["arm"], str(rec["id"]))
            for s in rec.get("samples", []):
                if "error" in s:
                    continue
                prev = merged.get(key)
                if prev is None or (prev["truncated"] and not s["truncated"]):
                    merged[key] = s
    return merged


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 McNemar / binomial 检验 (b, c 为不一致对数)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _z(p: float) -> float:  # 95%
    return 1.959963984540054


def diff_ci(b: int, c: int, n: int) -> tuple[float, float]:
    """配对准确率差的 Wald 区间: (b - c)/n ± z*sqrt((b + c) - (b - c)^2/n)/n."""
    if n == 0:
        return (0.0, 0.0)
    d = (b - c) / n
    var = max(0.0, (b + c) - n * d * d) / (n * n)
    h = _z(0.95) * math.sqrt(var)
    return (round(d - h, 4), round(d + h, 4))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    paths = sys.argv[1:]
    merged = load(paths)
    arms = sorted({a for a, _ in merged})
    ids = sorted({i for _, i in merged}, key=lambda x: (len(x), x))

    table = {}
    for a in arms:
        cells = [merged[(a, i)] for i in ids if (a, i) in merged]
        n = len(cells)
        trunc = sum(1 for s in cells if s["truncated"])
        ok = sum(1 for s in cells if s["correct"])
        table[a] = {"n": n, "truncated": trunc, "correct": ok,
                    "acc": round(ok / n, 4) if n else None}

    print(f"rounds={paths}")
    print(f"{'arm':6s} {'n':>3s} {'trunc':>5s} {'correct':>7s} {'acc':>7s}   (n = 有效样本, 截断残留单列)")
    for a, t in table.items():
        print(f"{a:6s} {t['n']:3d} {t['truncated']:5d} {t['correct']:7d} {str(t['acc']):>7s}")

    pairs = {}
    base = "none" if "none" in table else arms[0]
    for a in arms:
        if a == base:
            continue
        both = [i for i in ids if (base, i) in merged and (a, i) in merged]
        b = sum(1 for i in both if merged[(a, i)]["correct"] and not merged[(base, i)]["correct"])
        c = sum(1 for i in both if merged[(base, i)]["correct"] and not merged[(a, i)]["correct"])
        p = mcnemar_exact(b, c)
        lo, hi = diff_ci(b, c, len(both))
        pairs[f"{a}_vs_{base}"] = {"n_paired": len(both), "loss": c, "gain": b, "diff": round((b - c) / len(both), 4) if both else None,
                                   "ci95": [lo, hi], "p_mcnemar": round(p, 4), "sig": p < 0.05}

    print(f"\n--- 配对 McNemar vs {base} (gain = 该组对/基线错, loss = 基线对/该组错) ---")
    for k, v in pairs.items():
        print(f"{k:16s} n={v['n_paired']:2d} gain={v['gain']} loss={v['loss']} diff={v['diff']:+.3f} "
              f"CI95={v['ci95']} p={v['p_mcnemar']:.3f}{'  *' if v['sig'] else ''}")

    # 天花板判定: 用户假说 4.5 — 不截断后各组都满分 → 数据集无区分度
    accs = [t["acc"] for t in table.values() if t["acc"] is not None]
    n_full = sum(1 for x in accs if x == 1.0)
    ceiling = (table.get(base, {}).get("acc") or 0) >= 0.95 or n_full >= max(2, len(accs) - 1)
    residual_trunc = sum(t["truncated"] for t in table.values())
    verdict = ("CEILING: 截断=0 后仍无可区分差异 → 换更难分布 (HMMT/Olympiad), 本集上不比较 skill"
               if ceiling else "OK: 有区分度, 可在此集比较 skill")
    if residual_trunc:
        verdict = f"INCOMPLETE: 仍有 {residual_trunc} 条截断, 先用 --only-truncated + 更大 cap 补跑; {verdict}"
    print(f"\n[verdict] {verdict}")

    out = {"rounds": paths, "table": table,
           "verdict": verdict, "base": base, "pairs": pairs,
           "residual_truncation": residual_trunc}
    op = os.path.join(os.path.dirname(os.path.abspath(paths[0])), "analysis.json")
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {op}")


if __name__ == "__main__":
    main()
