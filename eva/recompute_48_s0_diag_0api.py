#!/usr/bin/env python3
"""#48 P0-b: S₀ 300 题 cache 可复核性诊断 (2026-11-18, 0 API)

只读: 不调 API、不重写 cache、不改任何产物。
对 s0_cache_test_d5.json 的 300 条做三层动作:
  (1) 用当前 _evaluate_d5 在存储 code 上重评 → 与 cache hard 逐题比对
      (硬一致 = 同一 evaluator 同一 code 给出同一 hard; 不一致 = cache 陈旧/坏)
  (2) 对不一致题跑 STDIN 路径强制对照 (appsplus_evaluate) —— 判定差异来源
  (3) 统计不一致题对 paired (Arm1 vs S₀) 的影响: 若把 cache hard 改成当前 D5 结果,
      rescue/regress/net 如何变化 (SENSITIVITY ONLY, 不是"采纳新数字")

输出: <RUN>/audit_0api_20261118/s0_cache_diagnostics.json
"""
import json
import os
import sys
from collections import Counter

RUN = ("/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/"
       "tf48-arm1-full-20261117/run_20261118_091052")
TEST_ITEMS = "/data/zhuldz/skillopt/data/appsplus_capped_seed42/test/items.json"
OUT = os.path.join(RUN, "audit_0api_20261118")

sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
from tf_skill_optimizer_appsplus import (  # noqa: E402
    _evaluate_d5, _d5_is_fn_call_item, mcnemar_paired, paired_net_gain_ci,
)

appsplus_evaluate = None
try:
    from skillopt.envs.appsplus.evaluator import evaluate as appsplus_evaluate  # noqa: E402
except Exception as e:  # pragma: no cover
    print(f"[warn] appsplus_evaluate import failed: {e!r}")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(OUT, exist_ok=True)
    items = {str(it["id"]): it for it in load(TEST_ITEMS)}
    s0 = load(os.path.join(RUN, "s0_cache_test_d5.json"))
    arm = load(os.path.join(RUN, "test_independent_results.json"))

    rows = []
    for qid, r in sorted(s0.items(), key=lambda kv: kv[0]):
        it = items.get(qid)
        if it is None:
            rows.append({"id": qid, "status": "id_not_in_test_items"})
            continue
        code = r.get("code") or ""
        cache_hard = int(r.get("hard", 0))
        if not code:
            rows.append({"id": qid, "cache_hard": cache_hard, "status": "no_stored_code",
                         "cache_fail_reason": (r.get("fail_reason") or "")[:120]})
            continue
        ev = _evaluate_d5(code, str(it.get("test_code", "")), str(it.get("question", "")))
        cur_hard = int(ev.get("hard", 0))
        rec = {
            "id": qid,
            "cache_hard": cache_hard,
            "cache_error_type": r.get("error_type") or "",
            "cache_fail_reason": (r.get("fail_reason") or "")[:160],
            "current_d5_hard": cur_hard,
            "current_d5_error_type": ev.get("error_type") or "",
            "current_d5_fail_reason": (ev.get("fail_reason") or "")[:160],
            "fn_call_detector": bool(_d5_is_fn_call_item(it)),
            "arm1_hard": int(arm.get(qid, {}).get("hard", 0)),
            "status": "consistent" if cur_hard == cache_hard else "MISMATCH",
        }
        if cur_hard != cache_hard and appsplus_evaluate is not None:
            try:
                ev2 = appsplus_evaluate(code, str(it.get("test_code", "")))
                rec["stdin_forced_hard"] = int(ev2.get("hard", 0))
                rec["stdin_forced_fail_reason"] = (ev2.get("fail_reason") or "")[:160]
            except Exception as e:
                rec["stdin_forced_error"] = repr(e)[:200]
        rows.append(rec)

    mism = [r for r in rows if r.get("status") == "MISMATCH"]
    no_code = [r for r in rows if r.get("status") == "no_stored_code"]

    summary = {
        "n_cache": len(s0),
        "n_rechecked": len(rows) - len(no_code),
        "n_no_stored_code": len(no_code),
        "consistent": len(rows) - len(mism) - len(no_code),
        "mismatch": len(mism),
        "mismatch_cache_hard_sum": sum(r["cache_hard"] for r in mism),
        "mismatch_current_hard_sum": sum(r["current_d5_hard"] for r in mism),
        "mismatch_by_cache_error_type": dict(Counter(r["cache_error_type"] for r in mism)),
        "mismatch_by_pair": dict(Counter(
            f"cache={r['cache_hard']}->cur={r['current_d5_hard']}" for r in mism)),
        "fn_call_detector_among_mismatch": sum(1 for r in mism if r["fn_call_detector"]),
        "arm1_hard_among_mismatch": sum(r["arm1_hard"] for r in mism),
    }

    # 敏感性: 仅在内存里把 cache hard 替换成当前 D5 结果, 重算 paired (不写回 cache)
    s0_corrected = {}
    for qid, r in s0.items():
        rr = dict(r)
        if r.get("code"):
            it = items.get(qid)
            if it is not None:
                rr["hard"] = int(_evaluate_d5(r["code"], str(it.get("test_code", "")),
                                              str(it.get("question", ""))).get("hard", 0))
        s0_corrected[qid] = rr

    def paired(baseline):
        bp, bf, a_only, b_only, n, p = mcnemar_paired(arm, baseline)
        ng = paired_net_gain_ci(a_only, b_only, n)
        return {
            "s0_hard": sum(int(r.get("hard", 0)) for r in baseline.values()),
            "s0_hard_rate": round(sum(int(r.get("hard", 0)) for r in baseline.values()) / n, 4),
            "n_common": n, "both_pass": bp, "both_fail": bf,
            "rescue(arm1_only)": a_only, "regress(s0_only)": b_only,
            "discordant": a_only + b_only, "mcnemar_p": p,
            "net_gain": ng["net_gain"], "net_gain_rate": ng["net_gain_rate"],
            "ci95_rate": [ng["ci_lo_rate"], ng["ci_hi_rate"]],
        }

    sensitivity = {
        "as_stored": paired(s0),
        "re_evaluated_current_d5": paired(s0_corrected),
        "note": ("re_evaluated_current_d5 = 同一批 S₀ 代码, 用当前 _evaluate_d5 重判 hard。"
                 "仅表示 cache 陈旧的方向与量级; 不是新的官方基线数字, 也不构成改口径。"),
    }

    out = {"run_dir": RUN, "summary": summary, "sensitivity": sensitivity,
           "mismatch_items": mism,
           "mismatch_fail_reason_pairs": sorted({
               (r["cache_error_type"], r["current_d5_error_type"]) for r in mism})}
    p = os.path.join(OUT, "s0_cache_diagnostics.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"→ {p}")
    print(json.dumps({k: summary[k] for k in summary if k != "mismatch_items"},
                     ensure_ascii=False, indent=2))
    print("\nsensitivity:", json.dumps(sensitivity, ensure_ascii=False, indent=2))
    print("\nmismatch ids:", [(r["id"], r["cache_hard"], r["current_d5_hard"],
                               r["arm1_hard"], r["cache_error_type"]) for r in mism])


if __name__ == "__main__":
    main()
