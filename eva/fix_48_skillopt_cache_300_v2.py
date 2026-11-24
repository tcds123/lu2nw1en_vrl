#!/usr/bin/env python3
"""#48 / Phase 3: 用当前运行时评测器重判 SkillOpt 300 题 cache 并生成审计记录 (0 API)

背景: Phase 3 需要把 SkillOpt 基线放到与 S₀、"Arm1/3 运行时" 同一评测口径。
      SkillOpt 原 results.json 的 hard=211 来自旧 serde 路径; optimizer 内存里
      rebuild_fn_call_cache 得到 218, 但磁盘无版本化产物 → 不可复现。

做法 (与 eva/fix_48_s0_cache_300_v2.py 完全同构):
  - 输入 = 已存储的 code, 不重新生成, 不调 API;
  - 用运行时 _evaluate_d5 + load_split(test, truncate=True) 的题面形态重判;
  - 写版本化新文件, 不覆盖 results.json;
  - 保存原始/新文件 hash 与逐题 diff。

输出 (同目录, 不覆盖任何原文件):
  <SKILLOPT_DIR>/test_eval/results.rejudged_d5_v2.json       # 300 条重判结果
  <V2_DIR>/audit_0api_20261120/skillopt_cache_300_rejudge.json  # 审计记录
"""
import hashlib
import json
import os
import sys

BASE = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/appsplus-skillopt-flash-20261111-v2/train"
V2_DIR = BASE
SKILLOPT_DIR = os.path.join(BASE, "test_eval")
SRC = os.path.join(SKILLOPT_DIR, "results.json")
DST = os.path.join(SKILLOPT_DIR, "results.rejudged_d5_v2.json")
S0_REJUDGED = os.path.join(BASE, "test_eval_baseline", "results.rejudged_d5_v2.json")
AUDIT = os.path.join(V2_DIR, "audit_0api_20261120")
TEST_ITEMS = "/data/zhuldz/skillopt/data/appsplus_capped_seed42/test/items.json"

sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
from tf_skill_optimizer_appsplus import _evaluate_d5, load_split  # noqa: E402

EVAL_VERSION = "runtime_evaluate_d5_rejudge_v1"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(AUDIT, exist_ok=True)
    src = json.load(open(SRC, encoding="utf-8"))
    existing = json.load(open(DST, encoding="utf-8")) if os.path.exists(DST) else None
    items = {str(it["id"]): it for it in load_split(os.path.dirname(os.path.dirname(TEST_ITEMS)), "test")}
    s0 = {str(r["id"]): r for r in json.load(open(S0_REJUDGED, encoding="utf-8"))}

    if existing is not None:
        # 已重判过: 校验磁盘文件与原 cache 的 code 完全一致, 直接复用, 只补审计
        ex = {str(r["id"]): r for r in existing}
        srm = {str(r["id"]): r for r in src}
        assert set(ex) == set(srm), "id 集不一致"
        assert all(ex[i]["code"] == srm[i]["code"] for i in ex), "code 与原 cache 不一致"
        out = existing
        diffs = [{"id": i, "old_hard": int(ex[i]["hard_legacy_stored"]), "new_hard": int(ex[i]["hard"])}
                 for i in ex if int(ex[i]["hard_legacy_stored"]) != int(ex[i]["hard"])]
    else:
        out, diffs = [], []
        for r in src:
            qid = str(r["id"])
            it = items.get(qid)
            e = dict(r)
            ev = _evaluate_d5(r["code"], str(it["test_code"]), str(it["question"]))
            e.update({
                "hard_legacy_stored": int(r.get("hard", 0)),
                "soft_legacy_stored": r.get("soft"),
                "pass_rate_legacy_stored": r.get("pass_rate"),
                "error_type_legacy_stored": r.get("error_type") or "",
                "fail_reason_legacy_stored": (r.get("fail_reason") or "")[:500],
                "hard": int(ev["hard"]), "soft": ev["soft"], "pass_rate": ev["pass_rate"],
                "error_type": ev.get("error_type") or "",
                "fail_reason": (ev.get("fail_reason") or "")[:500],
                "evaluator_version": EVAL_VERSION, "rejudge_status": "ok",
            })
            if int(ev["hard"]) != int(r.get("hard", 0)):
                diffs.append({"id": qid, "old_hard": int(r.get("hard", 0)), "new_hard": int(ev["hard"]),
                              "response": (r.get("response") or "")[:2000],
                              "s0_rejudged_hard": int(s0.get(qid, {}).get("hard", 0))})
            out.append(e)
        with open(DST, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    s0_flips = [i for i, r in s0.items() if int(r["hard"]) != int(r["hard_legacy_stored"])]
    record = {
        "generated_by": "eva/fix_48_skillopt_cache_300_v2.py",
        "evaluator": EVAL_VERSION,
        "item_shape": "load_split(test, truncate=True) — 与 run_one 运行时同一形态",
        "inputs": {"source": SRC, "sha256": sha(SRC),
                   "test_items": TEST_ITEMS, "test_items_sha256": sha(TEST_ITEMS),
                   "s0_rejudged": S0_REJUDGED, "s0_rejudged_sha256": sha(S0_REJUDGED)},
        "output": {"path": DST, "sha256": sha(DST), "n": len(out)},
        "hard_legacy_stored": sum(int(r.get("hard", 0)) for r in src),
        "hard_rejudged": sum(int(r["hard"]) for r in out),
        "n_changed": len(diffs),
        "changed": sorted(diffs, key=lambda d: d["id"]),
        "s0_changed_ids": sorted(s0_flips, key=lambda x: int(x)),
        "note": ("SkillOpt 存储 hard 211 → 重判 218 (7 题 fn-call 旧 harness 误判为失败, 全部 0→1, 无 1→0)。"
                 "S₀ 重判同为 7 题 0→1 (215), 两基线翻转数一致 → 协议修正对称, 非单臂挑选。"
                 "在 optimizer 内该数字已由 rebuild_fn_call_cache 得到 (best_hard=218), 本文件使其落盘可复现。"),
    }
    p = os.path.join(AUDIT, "skillopt_cache_300_rejudge.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: record[k] for k in ("n_changed", "hard_legacy_stored", "hard_rejudged")},
                     ensure_ascii=False))
    print("changed:", [(d["id"], d["old_hard"], d["new_hard"]) for d in record["changed"]])
    print("src sha256:", record["inputs"]["sha256"][:16], "| dst sha256:", record["output"]["sha256"][:16])
    print("→", p)


if __name__ == "__main__":
    main()
