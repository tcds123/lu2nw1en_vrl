"""Direction A v3 §5.4 dry-run + §2.3 单样本 CP 核对 (0 API, 只读归档)。

用法: python eva/direction_a_dryrun.py

用途:
  1. 在 Phase 4-A 归档上复算 260 道 class 题、Arm1 pass 132、R_O 候选集合与 sha256;
  2. 用合成场景跑通 §5.4 三种口径 (g_main / g_cons / g_eval) 与全部闭合断言;
  3. 核对 §2.3 的 P_r 单样本 CP 与"不可用"的 paired_net_gain_ci 的差异;
  4. 复核 §3.3 的 R_O 候选定义 (A/B/C) 题数与 Arm1 pass 数。

只读: 不写任何文件, 不发起任何 API 调用。
"""
import glob
import hashlib
import json
import random
import sys

sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
from scipy.stats import beta  # noqa: E402

from tf_skill_optimizer_appsplus import (  # noqa: E402
    _clopper_pearson,
    load_split,
    paired_net_gain_ci,
)

BASE = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline"
ARM1 = BASE + "/tf48-arm1-full-20261117/run_20261118_091052/test_independent_results.json"
PHASE3B = BASE + "/phase3b-static-baseline-2765-20261120/run_20261120_030416_full/chunks/*.json"
SPLIT = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
N_TOTAL = 260


def uncond_cp(k, n, alpha=0.05):
    """单样本 Clopper-Pearson 双侧区间 (§2.3 P_r 专用; 不得用 paired_net_gain_ci 代替)。"""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def _p(s, needle):
    return needle in (s or "")


def load():
    items = load_split(SPLIT, "test")
    q = {x["id"]: x["question"] for x in items}
    cls = [x["id"] for x in items
           if "class Solution" in x["question"] and "-----Input-----" not in x["question"]]
    arm = json.load(open(ARM1))
    s0, so = {}, {}
    for f in sorted(glob.glob(PHASE3B)):
        d = json.load(open(f))
        s0.update(d["s0"])
        so.update(d["skillopt"])
    return cls, q, arm, s0, so


def main():
    cls, q, arm, s0, so = load()
    pas = lambda m, i: int(m[i]["hard"]) == 1
    nm = lambda m, i: _p(m[i].get("fail_reason"), "NameError") and _p(m[i].get("fail_reason"), "Solution")

    n_pass = sum(pas(arm, i) for i in cls)
    print(f"[facts] class={len(cls)} arm1_pass={n_pass} P_A={n_pass/len(cls):.4f}")
    print(f"[facts] class_sha={hashlib.sha256(chr(10).join(cls).encode()).hexdigest()[:16]}")

    # §3.3 R_O 候选 (issue #112)
    cands = {
        "A_both_fail": [i for i in cls if not pas(s0, i) and not pas(so, i)],
        "B_both_nameerror": [i for i in cls if nm(s0, i) and nm(so, i)],
        "C_either_nameerror": [i for i in cls if nm(s0, i) or nm(so, i)],
        "D_arm1_only": [i for i in cls if not pas(s0, i) and not pas(so, i) and pas(arm, i)],
    }
    for k, s in cands.items():
        print(f"[R_O] {k:20s} n={len(s):3d} arm1_pass={sum(pas(arm, i) for i in s):3d} "
              f"sha={hashlib.sha256(chr(10).join(s).encode()).hexdigest()[:16]}")

    # §2.3 单样本 CP vs 条件区间
    print(f"[cp] uncond_cp(132,260) = [{uncond_cp(132,260)[0]:.4f}, {uncond_cp(132,260)[1]:.4f}]")
    print(f"[cp] harness _clopper_pearson(132,260,0.05) = {_clopper_pearson(132, 260, 0.05)}")
    deg = paired_net_gain_ci(132, 0, 260)
    print(f"[cp] paired_net_gain_ci(132,0,260) ci=[{deg['ci_lo_rate']:.4f}, {deg['ci_hi_rate']:.4f}] (退化, 不可用于 P_r)")
    lo, hi = uncond_cp(132, 260)
    assert abs(lo - 0.4452) < 1e-4 and abs(hi - 0.5700) < 1e-4, (lo, hi)
    assert uncond_cp(0, 260)[0] == 0.0 and uncond_cp(260, 260)[1] == 1.0

    # §5.4 dry-run
    ref = {i: ("pass" if pas(arm, i) else "fail") for i in cls}
    reff = {i for i in cls if ref[i] == "pass"}

    def tabulate(new):
        t = dict(both_pass=0, both_fail=0, rescue=0, regress=0,
                 missing=0, both_missing=0, ref_unrecorded=0)
        for i in cls:
            a, b = new.get(i, "infra_missing"), ref.get(i, "infra_missing")
            if b == "infra_missing":
                t["both_missing" if a == "infra_missing" else "ref_unrecorded"] += 1
            elif a == "infra_missing":
                t["missing"] += 1
            elif a == "pass" and b == "pass":
                t["both_pass"] += 1
            elif a == "fail" and b == "fail":
                t["both_fail"] += 1
            elif a == "pass":
                t["rescue"] += 1
            else:
                t["regress"] += 1
        return t

    def run(name, new):
        t = tabulate(new)
        nc = t["rescue"] + t["regress"] + t["both_pass"] + t["both_fail"]
        assert sum(t.values()) == N_TOTAL, (name, t)
        assert nc == N_TOTAL - t["missing"] - t["both_missing"] - t["ref_unrecorded"], (name, t)
        g_main = t["rescue"] - t["regress"]
        g_cons = g_main - sum(1 for i in cls if new.get(i, "infra_missing") == "infra_missing" and ref[i] == "pass")
        g_eval = g_main
        ci = paired_net_gain_ci(t["rescue"], t["regress"], nc)
        print(f"[§5.4] {name:22s} nc={nc:3d} g_main={g_main:+3d} ({g_main/N_TOTAL:+.4f}) "
              f"CI=[{ci['ci_lo_rate']:+.4f},{ci['ci_hi_rate']:+.4f}] g_cons={g_cons:+3d} "
              f"({g_cons/N_TOTAL:+.4f}) g_eval={g_eval:+3d} ({g_eval/max(nc,1):+.4f}) "
              f"missing={t['missing']} nc/260={nc/N_TOTAL:.3f}")

    rng = random.Random(0)
    run("A identity", dict(ref))
    run("B 10% flip", {i: ("fail" if rng.random() < 0.05 else "pass" if rng.random() < 0.05 else ref[i])
                       for i in cls})
    run("C 12 missing", {**ref, **{i: "infra_missing" for i in cls[:12]}})
    run("D 12 worst-missing", {**ref, **{i: "infra_missing" for i in rng.sample(sorted(reff), 12)}})
    run("E 30% missing", {**ref, **{i: "infra_missing" for i in cls[:78]}})
    print("[§5.4] all closure assertions passed")


if __name__ == "__main__":
    main()
