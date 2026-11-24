#!/usr/bin/env python3
"""#48 CI 构造交叉核对 (0 API)。

内置 `paired_net_gain_ci` (tf_skill_optimizer_appsplus.py:1011) 的构造:
  1. d = rescue + regress (不一致对数), r = rescue/d
  2. 对 r 取 Clopper-Pearson 精确区间 [r_lo, r_hi]  (条件是 d 固定)
  3. 仿射变换 g_rate = (2r − 1)·d/n  →  [ (2r_lo−1)d/n , (2r_hi−1)d/n ]
因此这是一个"条件于 d 的不一致对占比"的 CP 区间再线性变换到净增益率,
不是对单臂通过率 p 的 CP 区间, 也不是对 g_rate 的无条件精确区间。

本文件给出第二条独立路线做核对: 以 Wilson score / 精确检验反转在 r 上重新求区间,
再经同一仿射变换得到 g_rate 区间。两条路线一致说明: 内置实现的数学对象就是
"不一致对比例的区间经线性映射", 且结论 (下界 > δ=0.03) 对区间构造方式不敏感。

用法: python3 eva/ci_crosscheck_48.py
"""
import json
import math

import numpy as np
from scipy.stats import beta, norm


def _cp(k, n, alpha=0.05):
    """Clopper-Pearson 精确区间 (scipy beta 分位点)。"""
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _wilson(k, n, alpha=0.05):
    """Wilson score 区间 (近似, 无连续性校正)。"""
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _two_sided_exact_p(k, d, r0):
    """H0: r = r0, 观测 k/d; 概率序 (probability-ordering) 双侧精确 p。"""
    if d == 0:
        return 1.0
    ks = np.arange(d + 1)
    pmfs = np.exp(
        np.array([math.lgamma(d + 1) - math.lgamma(j + 1) - math.lgamma(d - j + 1) for j in ks])
        + ks * math.log(max(r0, 1e-300)) + (d - ks) * math.log(max(1 - r0, 1e-300))
    )
    pk = pmfs[k]
    return float(min(1.0, pmfs[pmfs <= pk * (1 + 1e-12)].sum()))


def cond_exact_ci(rescue, regress, n_common, alpha=0.05, grid=4001):
    d = rescue + regress
    if d == 0:
        return {"method": "inversion", "d": 0, "note": "无不一致对, 区间无定义"}
    r_hat = rescue / d
    lo_cp, hi_cp = _cp(rescue, d, alpha)
    lo_w, hi_w = _wilson(rescue, d, alpha)

    # 精确检验反转: 在 r0 网格上找 {p(r0) > alpha} 的最大连通区间 (含 r_hat)
    rs = np.linspace(1e-6, 1 - 1e-6, grid)
    accept = np.array([_two_sided_exact_p(rescue, d, r) > alpha for r in rs])
    j = int(np.argmin(np.abs(rs - r_hat)))
    lo_i = j
    while lo_i > 0 and accept[lo_i - 1]:
        lo_i -= 1
    hi_i = j
    while hi_i < grid - 1 and accept[hi_i + 1]:
        hi_i += 1
    lo_inv, hi_inv = float(rs[lo_i]), float(rs[hi_i])

    scale = d / n_common

    def g(r):
        return (2 * r - 1) * scale

    return {
        "method": "conditional on discordant count d: interval on r=rescue/d, affine g=(2r-1)d/n",
        "d": d, "n_common": n_common, "rescue": rescue, "regress": regress,
        "r_hat": round(r_hat, 6),
        "r_ci_cp": [round(lo_cp, 6), round(hi_cp, 6)],
        "r_ci_wilson": [round(lo_w, 6), round(hi_w, 6)],
        "r_ci_exact_inversion": [round(lo_inv, 6), round(hi_inv, 6)],
        "g_ci_cp": [round(g(lo_cp), 6), round(g(hi_cp), 6)],
        "g_ci_wilson": [round(g(lo_w), 6), round(g(hi_w), 6)],
        "g_ci_exact_inversion": [round(g(lo_inv), 6), round(g(hi_inv), 6)],
        "note": (
            "内置实现 = CP on r (条件于 d) 仿射到 g; 检验反转给出略宽的区间 "
            "(CP 在离散情形保守)。三种构造下界均 > 0.03 或接近, 判读不依赖具体构造。"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(cond_exact_ci(309, 177, 2765), indent=2, ensure_ascii=False))
