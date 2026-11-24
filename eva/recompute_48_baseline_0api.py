#!/usr/bin/env python3
"""#48 0-API 基线重算 (2026-11-18)

不调 API、不重跑模型。从原始 JSON 重算 S₀ 全量基线 / Arm1 配对 / 109 题核查。
复用 tf_skill_optimizer_appsplus 的统计函数 (mcnemar_paired / paired_net_gain_ci / _mcnemar_exact_p)。

注意: eval_independent 在 Arm1 里跑的是全部 2765 题 (optimizer 无截断参数),
       而 s0_cache_test (300) 是 SkillOpt v2 run 的截断 cache。两者不能直接配对。
S₀ 全量只能由 Arm1 同一次 run 的 s0_cache_best_d5.json 提供 (它保存了 S₀ 生成后
       用 Arm1 skill 重评的结果?? → 见下方 cache 语义核查结论)。
"""
import hashlib
import json
import os
import sys

RUN = ("/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/"
       "tf48-arm1-full-20261117/run_20261118_091052")
TEST_ITEMS = "/data/zhuldz/skillopt/data/appsplus_capped_seed42/test/items.json"
TRAIN_ITEMS = "/data/zhuldz/skillopt/data/appsplus_capped_seed42/train/items.json"
V2_TEST_BASE = ("/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/"
                "appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.json")
OUT = os.path.join(RUN, "audit_0api_20261118")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def rate(k, n):
    return round(k / n, 4) if n else 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    test_items = load(TEST_ITEMS)
    train_items = load(TRAIN_ITEMS)
    arm = load(os.path.join(RUN, "test_independent_results.json"))
    s0_300 = load(os.path.join(RUN, "s0_cache_test_d5.json"))
    best_300 = load(os.path.join(RUN, "s0_cache_best_d5.json"))
    v2_base = load(V2_TEST_BASE)
    v2_by_id = {str(r["id"]): r for r in v2_base}

    T = [str(it["id"]) for it in test_items]
    Tset = set(T)
    item_by_id = {str(it["id"]): it for it in test_items}

    report = {"run_dir": RUN, "generated_by": "eva/recompute_48_baseline_0api.py"}

    # ---------- 1. ID 一致性 ----------
    report["dataset"] = {
        "test_items_n": len(T),
        "test_unique_ids": len(Tset),
        "train_items_n": len(train_items),
        "arm1_result_n": len(arm),
        "s0_cache_test_d5_n": len(s0_300),
        "s0_cache_best_d5_n": len(best_300),
        "v2_test_eval_baseline_n": len(v2_base),
        "test_minus_arm1": sorted(Tset - set(arm))[:20],
        "arm1_minus_test": sorted(set(arm) - Tset)[:20],
        "s0_300_minus_test": sorted(set(s0_300) - Tset)[:20],
        "s0_300_is_first300": set(s0_300) == set(T[:300]),
        "best_300_is_first300": set(best_300) == set(T[:300]),
        "v2_is_first300": set(v2_by_id) == set(T[:300]),
        "first300": T[:300],
    }

    # ---------- 2. cache 语义核查 ----------
    # s0_cache_test_d5 == D5-重建的 v2 test_eval_baseline (S₀ 生成)? 逐题比对 response/code。
    sem = {}
    for name, cache in (("s0_cache_test_d5", s0_300), ("s0_cache_best_d5", best_300)):
        same_resp = same_code = 0
        for qid, r in cache.items():
            v = v2_by_id.get(qid)
            if not v:
                continue
            if (r.get("response") or "") == (v.get("response") or ""):
                same_resp += 1
            if (r.get("code") or "") == (v.get("code") or ""):
                same_code += 1
        sem[name] = {"n_compared": len(cache), "same_response_as_v2_baseline": same_resp,
                     "same_code_as_v2_baseline": same_code}
    report["cache_semantics"] = sem

    # ---------- 3. S₀ 覆盖核查 (全量 2765) ----------
    # optimizer 的 base_hard 只来自 s0_cache_test (300 题); 无 2765 全量 S₀ 评测。
    s0_full_available = (set(s0_300) == Tset)
    report["s0_coverage"] = {
        "test ∩ s0_cache_test_d5": len(Tset & set(s0_300)),
        "s0_full_2765_available": s0_full_available,
        "note": ("S₀ 只有在 s0_cache_best_d5 中才有 2765 覆盖, 但那不是 S₀ 的独立输出 —— "
                 "它是 Arm1 best skill 的评测结果 (D5 重建)，不可当 S₀ 基线"),
    }

    # ---------- 4. Arm1 全量 + 300 子集/前300切片 ----------
    arm_valid = {q: r for q, r in arm.items() if r.get("agent_ok")}
    arm_hard = sum(int(r["hard"]) for r in arm_valid.values())
    report["arm1"] = {
        "n_total": len(arm),
        "n_valid": len(arm_valid),
        "n_agent_fail": sum(1 for r in arm.values() if not r.get("agent_ok")),
        "hard_pass": arm_hard,
        "hard_rate": rate(arm_hard, len(arm_valid)),
        "first300_hard": sum(int(arm[q]["hard"]) for q in T[:300] if q in arm),
        "first300_rate": rate(sum(int(arm[q]["hard"]) for q in T[:300] if q in arm), 300),
    }

    # ---------- 5. 配对: Arm1 vs S₀ (仅 300 交集, 两口径) ----------
    sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
    from tf_skill_optimizer_appsplus import mcnemar_paired, paired_net_gain_ci  # noqa: E402

    paired = {}
    for name, cache, note in (
        ("s0_cache_test_d5", s0_300, "SkillOpt v2 test_eval_baseline, D5 重建 (S₀ hard=215)"),
        ("s0_cache_best_d5", best_300, "SkillOpt v2 best skill (hard=218), 非 S₀"),
    ):
        bp, bf, a_only, b_only, n, p = mcnemar_paired(arm, cache)  # a=Arm1, b=baseline
        ng = paired_net_gain_ci(a_only, b_only, n)
        paired[name] = {
            "note": note, "n_common": n,
            "both_pass": bp, "both_fail": bf,
            "arm1_only_pass(rescue)": a_only, "baseline_only_pass(regress)": b_only,
            "discordant": a_only + b_only, "mcnemar_p": p,
            "net_gain": ng["net_gain"], "net_gain_rate": ng["net_gain_rate"],
            "ci95_rate": [ng["ci_lo_rate"], ng["ci_hi_rate"]],
        }
    report["paired"] = paired

    # ---------- 6. 109 题结构性不可解题核查 ----------
    ne_ids = [q for q, r in arm.items() if "NameError" in (r.get("fail_reason") or "")
              and "'Solution'" in (r.get("fail_reason") or "")]
    # 客观判据: 题面无 `def <name>(` 可提取签名? test_code 是 stdin 格式?
    # 用与 optimizer 完全相同的检测器
    from tf_skill_optimizer_appsplus import _d5_is_fn_call_item  # noqa: E402
    audit = []
    for q in ne_ids:
        it = item_by_id.get(q, {})
        tc = str(it.get("test_code", ""))
        try:
            tco = json.loads(tc)
        except Exception:
            tco = None
        qtext = str(it.get("question", ""))
        has_class_solution = "class Solution" in qtext
        has_def_in_q = bool(__import__("re").search(r"def\s+\w+\s*\(", qtext))
        audit.append({
            "id": q,
            "hard_arm1": int(arm[q]["hard"]),
            "agent_ok": arm[q].get("agent_ok"),
            "in_first300": q in set(T[:300]),
            "hard_s0_300": int(s0_300[q]["hard"]) if q in s0_300 else None,
            "hard_best_300": int(best_300[q]["hard"]) if q in best_300 else None,
            "fn_call_detector": _d5_is_fn_call_item(it),
            "test_code_keys": sorted(tco.keys()) if isinstance(tco, dict) else "not-json",
            "question_has_class_Solution": has_class_solution,
            "question_has_def": has_def_in_q,
            "fail_reason_head": (arm[q].get("fail_reason") or "")[:120],
        })
    report["nameerror_solution_audit"] = {
        "n": len(audit),
        "in_first300": sum(1 for a in audit if a["in_first300"]),
        "hard_s0_300_sum": sum(a["hard_s0_300"] or 0 for a in audit),
        "hard_best_300_sum": sum(a["hard_best_300"] or 0 for a in audit),
        "fn_call_detector_true": sum(1 for a in audit if a["fn_call_detector"]),
        "question_has_class_Solution": sum(1 for a in audit if a["question_has_class_Solution"]),
        "items": audit,
    }

    # 全体 test 集里 "question 含 class Solution 但 test_code 是 stdin 格式" 的题数
    mixed = []
    for it in test_items:
        q = str(it["id"])
        qtext = str(it.get("question", ""))
        if "class Solution" not in qtext:
            continue
        try:
            tco = json.loads(str(it.get("test_code", "")))
        except Exception:
            continue
        if isinstance(tco, dict) and "stdin" in tco:
            mixed.append(q)
    report["mixed_format_items"] = {
        "n": len(mixed),
        "of_which_arm1_hard0": sum(1 for q in mixed if not int(arm.get(q, {}).get("hard", 0))),
        "of_which_s0_300_hard1": sum(1 for q in mixed if int(s0_300.get(q, {}).get("hard", 0))),
        "ids": mixed,
    }

    # ---------- 7. 文件哈希 (可复核锚点) ----------
    report["hashes"] = {p.split("/")[-1]: sha8(p) for p in (
        os.path.join(RUN, "test_independent_results.json"),
        os.path.join(RUN, "s0_cache_test_d5.json"),
        os.path.join(RUN, "s0_cache_best_d5.json"),
        os.path.join(RUN, "tf_global_skill.md"),
        TEST_ITEMS, V2_TEST_BASE,
    )}

    out_path = os.path.join(OUT, "baseline_recompute.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"→ {out_path}")
    return report


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("dataset", "cache_semantics", "s0_coverage",
                                        "arm1", "paired", "hashes")},
                     ensure_ascii=False, indent=2)[:4000])
    na = r["nameerror_solution_audit"]
    print("\n109题核查:", {k: na[k] for k in na if k != "items"})
    print("混合格式题:", {k: v for k, v in r["mixed_format_items"].items() if k != "ids"})
