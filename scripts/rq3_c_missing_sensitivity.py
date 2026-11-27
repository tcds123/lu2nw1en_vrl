#!/usr/bin/env python3
"""RQ3-C #18 补充: 缺失值敏感性分析 (0 API)。

背景: S₀ 侧 71 题、Arm1 侧若干题因网关 HTTP 500 无有效结果。缺失**未必随机** —— 网关异常与
「内容 × 长提示」相关, 可能与题目长度/复杂度相关。故除完整配对分析外, 需给出:
  1. 缺失题 vs 可配对题 在**静态难度特征**上的分布差异 (题面长度、测试用例数、参考答案长度、是否需输入)
  2. 缺失题在**另一实验臂**上的表现 (S₀ 缺失题看 Arm1 结果; Arm1 缺失题看 S₀ 结果)
  3. 最有利 / 最不利情景下净增益 g 的边界 (把缺失题按极端结果补齐后重算 McNemar + CI)

边界定义 (预注册, 先写死再算):
  - 最有利 (Arm1-favorable): 仅 S₀ 有有效结果、Arm1 缺失的题 → 记 Arm1 pass(1)、S₀ fail(0);
    仅 Arm1 有、S₀ 缺失的题 → 记 Arm1 pass(1)、S₀ fail(0); 两侧都缺失的题 → 记 Arm1 pass、S₀ fail。
  - 最不利 (Arm1-unfavorable): 上述三类题一律记 Arm1 fail(0)、S₀ pass(1)。
  - 对照: 缺失题在两侧都缺失时, 边界只由 pair 集决定; 参考点 = 有效配对 (exclude) 结果。

统计复用生产函数 (mcnemar_paired / paired_net_gain_ci / interpret_ci), 与 rq3_c_paired_eval 同口径。

用法: python3 scripts/rq3_c_missing_sensitivity.py --model deepseek-v4-pro \
        --arm1 outputs/rq3/deepseek-v4-pro/arm1_test_combined.json --s0 outputs/rq3/deepseek-v4-pro/test_s0_cache.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys

ROOT = "/data/zhuldz/lunwen"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rq3_c_paired_eval import SPLIT_DIR, DELTA_MIN, _by_id, _load_evolver  # noqa: E402


def _feats(item):
    q = item.get("question") or ""
    tc = item.get("test_code") or ""
    try:
        cases = json.loads(tc).get("stdin", []) if tc.strip().startswith("{") else []
    except Exception:  # noqa: BLE001
        cases = []
    # 题目签名里的参数个数 ~ 接口复杂度
    m = re.search(r"def\s+\w+\(([^)]*)\)", q)
    params = len([p for p in m.group(1).split(",") if p.strip()]) if m else None
    return {"q_chars": len(q), "q_lines": q.count("\n") + 1,
            "raw_solution_chars": len(item.get("raw_solution") or ""),
            "n_test_cases": len(cases), "has_stdin": int(bool(cases)),
            "n_params": params}


def _summary(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None
    vals_s = sorted(vals)
    return {"n": len(vals), "mean": round(st.mean(vals), 2), "median": st.median(vals),
            "p25": vals_s[len(vals_s) // 4], "p75": vals_s[(3 * len(vals_s)) // 4]}


def _mwu_p(x, y):
    """Mann-Whitney U 正态近似双侧 p (无 scipy 依赖时兜底)。"""
    import math
    nx, ny = len(x), len(y)
    if nx < 3 or ny < 3:
        return None
    allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rx = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 0)
    u = rx - nx * (nx + 1) / 2.0
    mu = nx * ny / 2.0
    sd = math.sqrt(nx * ny * (nx + ny + 1) / 12.0)
    if sd == 0:
        return None
    z = (u - mu) / sd
    return round(math.erfc(abs(z) / math.sqrt(2)), 4)


def _gap(pair_rate, sens_rows):
    if not sens_rows or not pair_rate:
        return None
    sens_rate = sum(r for r in sens_rows) / len(sens_rows)
    ratio = sens_rate / pair_rate if pair_rate > 0 else None
    return {"other_arm_pass_rate": round(sens_rate, 4),
            "paired_pass_rate": round(pair_rate, 4),
            "ratio": round(ratio, 3) if ratio is not None else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--arm1", default="")
    ap.add_argument("--s0", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = os.path.join(ROOT, "outputs/rq3", a.model)
    a.arm1 = a.arm1 or os.path.join(d, "arm1_test_combined.json")
    a.s0 = a.s0 or os.path.join(d, "test_s0_cache.json")
    a.out = a.out or os.path.join(d, "rq3c_missing_sensitivity.json")

    ev = _load_evolver()
    items = {str(it["id"]): it for it in ev.load_split(SPLIT_DIR, "test", truncate=False)}
    arm, _ = _by_id(a.arm1)
    s0, _ = _by_id(a.s0)
    split = set(items)
    unknown = (set(arm) | set(s0)) - split
    if unknown:
        print(f"[sens] BLOCKED: {len(unknown)} 个 id 不在 split", file=sys.stderr)
        return 2
    arm_ok = {k: v for k, v in arm.items() if v.get("agent_ok")}
    s0_ok = {k: v for k, v in s0.items() if v.get("agent_ok")}
    valid = set(arm_ok) & set(s0_ok)
    arm_only = set(s0_ok) - set(arm_ok)   # S₀ 有结果, Arm1 缺
    s0_only = set(arm_ok) - set(s0_ok)    # Arm1 有结果, S₀ 缺
    both_missing = split - set(arm_ok) - set(s0_ok)

    # ── 1. 静态特征: 缺失 vs 可配对 ──
    feats = {k: _feats(v) for k, v in items.items()}
    keys = ["q_chars", "q_lines", "raw_solution_chars", "n_test_cases"]
    groups = {"paired": sorted(valid, key=int),
              "missing_arm1": sorted(arm_only, key=int),
              "missing_s0": sorted(s0_only, key=int),
              "missing_both": sorted(both_missing, key=int)}
    feat_block, tests = {}, []
    for g, ids in groups.items():
        feat_block[g] = {"n": len(ids), **{k: _summary([feats[i] for i in ids], k) for k in keys},
                         "has_stdin_rate": round(sum(feats[i]["has_stdin"] for i in ids) / max(1, len(ids)), 4)}
    for g in ("missing_arm1", "missing_s0", "missing_both"):
        for k in keys:
            x = [feats[i][k] for i in groups[g] if feats[i][k] is not None]
            y = [feats[i][k] for i in groups["paired"] if feats[i][k] is not None]
            tests.append({"group": g, "feature": k, "mwu_p": _mwu_p(x, y),
                          "median_missing": st.median(x) if x else None,
                          "median_paired": st.median(y) if y else None})

    # ── 2. 缺失题在另一臂的表现 ──
    pair_s0_rate = sum(int(s0_ok[i]["hard"]) for i in valid) / max(1, len(valid))
    pair_arm_rate = sum(int(arm_ok[i]["hard"]) for i in valid) / max(1, len(valid))
    other_arm = {
        "arm1_missing_examined_via_s0": _gap(pair_s0_rate, [int(s0_ok[i]["hard"]) for i in arm_only]),
        "s0_missing_examined_via_arm1": _gap(pair_arm_rate, [int(arm_ok[i]["hard"]) for i in s0_only]),
        "both_missing_examined_via_none": {"n": len(both_missing)},
    }

    # ── 3. 净增益边界 (预注册: 见 docstring) ──
    def _pair_dict(extra_arm, extra_s0):
        a_by = {k: v for k, v in arm_ok.items() if k in valid}
        b_by = {k: v for k, v in s0_ok.items() if k in valid}
        for k, v in extra_arm.items():
            a_by[k] = {"hard": v}
        for k, v in extra_s0.items():
            b_by[k] = {"hard": v}
        common = set(a_by) & set(b_by)
        return {k: a_by[k] for k in common}, {k: b_by[k] for k in common}

    a_ex, b_ex = _pair_dict({}, {})
    bp, bf, r_ex, g_ex, n_ex, p_ex = ev.mcnemar_paired(a_ex, b_ex)
    ci_ex = ev.paired_net_gain_ci(r_ex, g_ex, n_ex)

    unresolved = arm_only | s0_only | both_missing
    # 最有利: Arm1 全 pass, S₀ 全 fail (对每个未配对题)
    a_fav = dict(a_ex); b_fav = dict(b_ex)
    for k in unresolved:
        a_fav[k] = {"hard": 1}
        b_fav[k] = {"hard": 0}
    bp_f, bf_f, r_f, g_f, n_f, p_f = ev.mcnemar_paired(a_fav, b_fav)
    ci_f = ev.paired_net_gain_ci(r_f, g_f, n_f)
    # 最不利: Arm1 全 fail, S₀ 全 pass
    a_unf = dict(a_ex); b_unf = dict(b_ex)
    for k in unresolved:
        a_unf[k] = {"hard": 0}
        b_unf[k] = {"hard": 1}
    bp_u, bf_u, r_u, g_u, n_u, p_u = ev.mcnemar_paired(a_unf, b_unf)
    ci_u = ev.paired_net_gain_ci(r_u, g_u, n_u)
    rep = {
        "model": a.model, "inputs": {"arm1": os.path.abspath(a.arm1), "s0": os.path.abspath(a.s0)},
        "n_split": len(split), "n_paired": len(valid),
        "missing": {"arm1_missing_out_of_s0_valid": len(arm_only), "s0_missing_out_of_arm1_valid": len(s0_only),
                    "both_missing": len(both_missing),
                    "ids_arm1_missing": sorted(arm_only, key=int),
                    "ids_s0_missing": sorted(s0_only, key=int),
                    "ids_both_missing": sorted(both_missing, key=int)},
        "features_note": "静态特征 (测试集自带题面/参考答案/用例), 与哪一臂缺失无关; 缺失题在某一臂缺 = "
                         "该臂网关 HTTP 500, 特征分布用于检验缺失是否与题目长度/复杂度相关 (MWU 双侧 p)",
        "features": feat_block, "feature_tests_mwu": tests,
        "other_arm_gap": other_arm,
        "sensitivity": {
            "definition": "未配对题 (arm_only ∪ s0_only ∪ both_missing) 按极端情景补齐后重算 McNemar 精确 p 与 Clopper-Pearson CI; "
                          "fav = Arm1 pass/S₀ fail, unfav = Arm1 fail/S₀ pass",
            "paired_only_reference": {"mcnemar": {"rescue": r_ex, "regress": g_ex, "n_common": n_ex, "p_one_sided": p_ex},
                                      "net_gain": ci_ex, "interpret": ev.interpret_ci(ci_ex, DELTA_MIN)},
            "best_case_arm1_favorable": {"mcnemar": {"rescue": r_f, "regress": g_f, "n_common": n_f, "p_one_sided": p_f},
                                         "net_gain": ci_f, "interpret": ev.interpret_ci(ci_f, DELTA_MIN)},
            "worst_case_arm1_unfavorable": {"mcnemar": {"rescue": r_u, "regress": g_u, "n_common": n_u, "p_one_sided": p_u},
                                            "net_gain": ci_u, "interpret": ev.interpret_ci(ci_u, DELTA_MIN)},
            "n_edge_trials": len(unresolved),
        },
        "conclusion_template": ("边界区间 [worst.net_gain_rate, best.net_gain_rate] 覆盖 0 ⇔ 缺失不能推翻结论; "
                                "若边界同号则结论对缺失情景稳健; 结合 other_arm_gap 判断缺失是否与可解性相关."),
    }
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[sens] → {a.out}")
    print(json.dumps({"n_paired": len(valid), "missing": {k: rep["missing"][k] for k in
                                                          ("arm1_missing_out_of_s0_valid", "s0_missing_out_of_arm1_valid",
                                                           "both_missing")},
                      "other_arm_gap": other_arm,
                      "paired_net_gain": {k: ci_ex[k] for k in ("rescue", "regress", "n_common", "net_gain_rate",
                                                                "ci_lo_rate", "ci_hi_rate")},
                      "best": {k: ci_f[k] for k in ("rescue", "regress", "n_common", "net_gain_rate",
                                                    "ci_lo_rate", "ci_hi_rate")},
                      "worst": {k: ci_u[k] for k in ("rescue", "regress", "n_common", "net_gain_rate",
                                                     "ci_lo_rate", "ci_hi_rate")},
                      "p_paired": p_ex, "p_best": p_f, "p_worst": p_u}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
