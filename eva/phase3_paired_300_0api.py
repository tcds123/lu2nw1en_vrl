#!/usr/bin/env python3
"""Phase 3: Evolver (Arm1/Arm3) vs 静态基线 (S₀/SkillOpt) 的 300 题配对审计 (0 API)。

口径:
  - 配对集合 = 300 个 S₀ 定版题 ∩ 两臂 agent_ok=True；
  - 臂侧用 **独立非路由** 结果 (test_independent_results.json)，基线用 D5 重判后的存储 code；
  - McNemar 精确 p；净增益率为对不一致对比例 r=rescue/d 的 CP 区间仿射
    g_rate = (2r−1)·d/n (与 tf_skill_optimizer_appsplus.paired_net_gain_ci 同法)；
  - δ = 0.03。

用法: python3 eva/phase3_paired_300_0api.py
输出: <arm1 run>/audit_0api_20261120/phase3_paired_300.json
"""
import hashlib
import json
import os

import numpy as np
from scipy.stats import beta, binomtest

BASE = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline"
V2 = os.path.join(BASE, "appsplus-skillopt-flash-20261111-v2/train")
ARMS = {
    "arm1_full": os.path.join(BASE, "tf48-arm1-full-20261117/run_20261118_091052"),
    "arm3_drop_scaffold": os.path.join(BASE, "tf48-arm3-drop-scaffold-20261117/run_20261119_091235"),
}
BASELINES = {
    "S0": os.path.join(V2, "test_eval_baseline/results.rejudged_d5_v2.json"),
    "SkillOpt": os.path.join(V2, "test_eval/results.rejudged_d5_v2.json"),
}
DELTA_MIN = 0.03
OUT = os.path.join(ARMS["arm1_full"], "audit_0api_20261120")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def main():
    os.makedirs(OUT, exist_ok=True)
    s0 = {str(r["id"]): r for r in json.load(open(BASELINES["S0"], encoding="utf-8"))}
    base = {k: {str(r["id"]): r for r in json.load(open(p, encoding="utf-8"))}
            for k, p in BASELINES.items()}
    arm = {t: {str(k): v for k, v in
               json.load(open(os.path.join(d, "test_independent_results.json"), encoding="utf-8")).items()}
           for t, d in ARMS.items()}

    # 配对集合: 300 题中两臂都 agent_ok 的 ID (基线 300/300 agent_ok 已核对)
    ids = [i for i in s0 if all(i in a and a[i]["agent_ok"] for a in arm.values())]
    ids.sort(key=int)
    n = len(ids)

    table = {}
    for t in ARMS:
        h = sum(int(arm[t][i]["hard"]) for i in ids)
        table[t] = {"n": n, "hard": h, "rate": round(h / n, 6)}
        for b in BASELINES:
            a_ = [int(arm[t][i]["hard"]) for i in ids]
            b_ = [int(base[b][i]["hard"]) for i in ids]
            r = sum(1 for x, y in zip(a_, b_) if x == 1 and y == 0)
            g = sum(1 for x, y in zip(a_, b_) if x == 0 and y == 1)
            d = r + g
            lo, hi = cp(r, d) if d else (0.0, 1.0)
            p = float(binomtest(r, d, 0.5).pvalue) if d else 1.0
            ci = [round((2 * x - 1) * d / n, 6) for x in (lo, hi)]
            table[t][f"vs_{b}"] = {
                "baseline_hard": sum(b_), "gain": sum(a_) - sum(b_),
                "rescue": r, "regress": g, "discordant": d,
                "net_gain_rate": round((r - g) / n, 6), "ci": ci,
                "mcnemar_exact_p": round(p, 6),
                "meets_delta": bool(ci[0] > DELTA_MIN),
                "ci_excludes_0": bool(ci[0] > 0),
            }

    ids_file = os.path.join(OUT, "phase3_common_ids_300.txt")
    with open(ids_file, "w") as f:
        f.write("\n".join(ids) + "\n")

    record = {
        "generated_by": "eva/phase3_paired_300_0api.py",
        "protocol": {
            "pairing": "300 S₀ ids ∩ both arms agent_ok; arms use independent (non-routed) results",
            "ci": "CP on r=rescue/d (conditional on d), affine g_rate=(2r-1)d/n",
            "delta_min": DELTA_MIN,
        },
        "inputs": {**{f"baseline_{k}": {"path": p, "sha256": sha(p)} for k, p in BASELINES.items()},
                   **{f"{t}_independent": {"path": os.path.join(d, "test_independent_results.json"),
                                           "sha256": sha(os.path.join(d, "test_independent_results.json"))}
                      for t, d in ARMS.items()}},
        "n_common": n, "ids_file": ids_file,
        "ids_sha256": hashlib.sha256(open(ids_file, "rb").read()).hexdigest(),
        "table": table,
        "verdict": ("四条比较 CI 均跨 0 且覆盖 δ=0.03 → Evolver (Arm1) 相对静态 S₀/SkillOpt 的增益 "
                    "未达统计显著, 也未达实际有用阈值; 不能宣称 Evolver 有效。"
                    "300 为文件顺序前缀 (非预注册子集), 不能外推 2765。"),
        "caveat": ("p>0.05 ≠ 无效; 300 题 MDD≈11.2pp, 点估计 +3.0pp 接近 δ。"
                   "routed 结果 (243/236) 复用 S₀ 缓存, 仅作诊断, 不入本表。"),
    }
    p_out = os.path.join(OUT, "phase3_paired_300.json")
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(json.dumps({"n": n, **{t: {k: v for k, v in table[t].items() if k.startswith(("hard", "vs_"))}
                                 for t in table}}, ensure_ascii=False, indent=1))
    print("→", p_out)


if __name__ == "__main__":
    main()
