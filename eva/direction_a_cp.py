"""Direction A v3 区间函数 (0 API, 纯计算)。

关键约束 (v3 §2.3 / §11, issue #112):
  - `P_r` (单轮无条件通过率) -> `uncond_cp(k, n)`: 单样本 Clopper-Pearson。
  - `Δ_r` / R1b 两两配对 (配对净变化) -> harness `paired_net_gain_ci` (条件精确区间)。
  - 两者**不得共用同一个区间函数**。

自测: python eva/direction_a_cp.py   (三条断言须在 E0 前通过)
"""
import sys

from scipy.stats import beta

sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")


def uncond_cp(k, n, alpha=0.05):
    """单样本 Clopper-Pearson 双侧精确区间, 返回 (lo, hi)。"""
    if not 0 <= k <= n:
        raise ValueError(f"require 0 <= k <= n, got k={k}, n={n}")
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def main():
    # (i) 复现 Phase 4-A 参照值
    lo, hi = uncond_cp(132, 260)
    assert abs(lo - 0.4452) < 1e-4 and abs(hi - 0.5700) < 1e-4, (lo, hi)

    # (ii) 端点退化情形
    assert uncond_cp(0, 260)[0] == 0.0
    assert uncond_cp(260, 260)[1] == 1.0

    # (iii) 与直接 beta.ppf 手算一致
    k, n, a = 97, 199, 0.05
    assert abs(uncond_cp(k, n, a)[0] - beta.ppf(a / 2, k, n - k + 1)) < 1e-9
    assert abs(uncond_cp(k, n, a)[1] - beta.ppf(1 - a / 2, k + 1, n - k)) < 1e-9

    # (iv) 与 harness 的 CP 实现一致 (独立实现交叉核对)
    from tf_skill_optimizer_appsplus import _clopper_pearson
    lo2, hi2 = _clopper_pearson(132, 260, 0.05)
    assert abs(lo - lo2) < 1e-9 and abs(hi - hi2) < 1e-9

    # (v) 负对照: 条件区间函数在 g=0 时退化, 不得用于 P_r
    from tf_skill_optimizer_appsplus import paired_net_gain_ci
    deg = paired_net_gain_ci(132, 0, 260)
    assert deg["ci_hi_rate"] - deg["ci_lo_rate"] < 0.05, deg
    assert hi - lo > 0.1, (lo, hi)

    a1, a2 = uncond_cp(97, 199)
    print(f"ok: uncond_cp(132,260)=[{lo:.4f},{hi:.4f}]  uncond_cp(97,199)=[{a1:.4f},{a2:.4f}]  "
          f"paired(132,0,260)=[{deg['ci_lo_rate']:.4f},{deg['ci_hi_rate']:.4f}] (negative control)")


if __name__ == "__main__":
    main()
