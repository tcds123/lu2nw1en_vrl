#!/usr/bin/env python3
"""#48 Step1: 用当前运行时评测器重判 S₀ 300 题 cache, 写版本化新文件 (2026-11-19, 0 API)

背景: v2 test_eval_baseline (results.json) 的 hard 由旧 serde 路径产生, 其中 7 条为陈旧
      hard=1 (存储 code 用当前 _evaluate_d5 实为 0)。用户 2026-11-19 批准:
        - 用当前运行时评测器重算 300 题 (输入 = 存储的 code, 不重生成);
        - 写入新版本化文件, 不覆盖原始 cache;
        - 保存原始/新 cache 的 hash;
        - 记录 7 题的旧值、新值和对应 response;
        - 新版本 = 本轮参考基线; 215 只作为历史存储值。

输出 (同目录, 不覆盖任何原文件):
  <V2_DIR>/test_eval_baseline/results.rejudged_d5_v2.json   # 300 条, list 形态, 同原 schema + 新字段
  audit_0api_20261118/s0_cache_300_rejudge.json             # 审计记录 (hash / 逐条 diff / 7 题明细)
"""
import hashlib
import json
import os
import sys

RUN = ("/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/"
       "tf48-arm1-full-20261117/run_20261118_091052")
AUDIT = os.path.join(RUN, "audit_0api_20261118")
V2_DIR = ("/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/"
          "appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline")
SRC = os.path.join(V2_DIR, "results.json")
DST = os.path.join(V2_DIR, "results.rejudged_d5_v2.json")
TEST_ITEMS = "/data/zhuldz/skillopt/data/appsplus_capped_seed42/test/items.json"

sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
from tf_skill_optimizer_appsplus import _evaluate_d5, load_split  # noqa: E402

EVAL_VERSION = "runtime_evaluate_d5_rejudge_v1"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(AUDIT, exist_ok=True)
    src = json.load(open(SRC, encoding="utf-8"))
    # 用与运行时完全相同的截断形态 (题面 = load_split(truncate=True))
    items = {str(it["id"]): it for it in load_split(os.path.dirname(os.path.dirname(TEST_ITEMS)), "test")}
    arm = json.load(open(os.path.join(RUN, "test_independent_results.json"), encoding="utf-8"))

    out, diffs = [], []
    for r in src:
        qid = str(r["id"])
        it = items.get(qid)
        e = dict(r)
        if it is None or not (r.get("code") or ""):
            e["rejudge_status"] = "skipped_missing_item_or_code"
            out.append(e)
            continue
        ev = _evaluate_d5(r["code"], str(it.get("test_code", "")), str(it.get("question", "")))
        old = (int(r.get("hard", 0)), r.get("soft"), r.get("pass_rate"), r.get("error_type") or "")
        new = (int(ev["hard"]), ev["soft"], ev["pass_rate"], ev.get("error_type") or "")
        e.update({
            "hard_legacy_stored": int(r.get("hard", 0)),
            "soft_legacy_stored": r.get("soft"),
            "pass_rate_legacy_stored": r.get("pass_rate"),
            "error_type_legacy_stored": r.get("error_type") or "",
            "fail_reason_legacy_stored": (r.get("fail_reason") or "")[:500],
            "hard": new[0], "soft": new[1], "pass_rate": new[2],
            "error_type": new[3], "fail_reason": (ev.get("fail_reason") or "")[:500],
            "evaluator_version": EVAL_VERSION,
            "rejudge_status": "ok",
        })
        if old != new and old[0] != new[0]:  # 只记 hard 变化; soft/pass_rate 漂移无关协议
            diffs.append({
                "id": qid, "old": {"hard": old[0], "soft": old[1], "pass_rate": old[2], "error_type": old[3]},
                "new": {"hard": new[0], "soft": new[1], "pass_rate": new[2], "error_type": new[3]},
                "response": (r.get("response") or "")[:2000],
                "arm1_hard": int(arm.get(qid, {}).get("hard", 0)),
            })
        out.append(e)

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    hard_old = sum(int(r.get("hard", 0)) for r in src)
    hard_new = sum(int(e["hard"]) for e in out if e.get("rejudge_status") == "ok")
    record = {
        "generated_by": "eva/fix_48_s0_cache_300_v2.py",
        "evaluator": EVAL_VERSION,
        "item_shape": "load_split(test, truncate=True) — 与 run_one 运行时同一形态",
        "inputs": {
            "source": SRC, "sha256": sha(SRC),
            "test_items": TEST_ITEMS, "test_items_sha256": sha(TEST_ITEMS),
        },
        "output": {"path": DST, "sha256": sha(DST)},
        "n": len(out),
        "hard_legacy_stored": hard_old,
        "hard_rejudged": hard_new,
        "n_changed": len(diffs),
        "changed_ids": [d["id"] for d in diffs],
        "changed": diffs,
        "note": ("重判后 hard=215 (= Arm1 run 内 Path-B 重建出来的同一数字), 原 results.json 存 208。"
                 "差异 7 题 (2798/3812/5090/5149/6508/863/933) 旧 hard=0 (PartialFailure) 新 hard=1, "
                 "即旧 cache 把 7 道 fn-call 题烙成失败, 与先前 issue #90 的方向相反 —— 见 issue #92。"
                 "本轮参考基线 = hard_rejudged = 215; 208 为历史存储值。"),
    }
    p = os.path.join(AUDIT, "s0_cache_300_rejudge.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"→ {DST}\n→ {p}")
    print(json.dumps({k: record[k] for k in ("n", "hard_legacy_stored", "hard_rejudged",
                                             "n_changed", "changed_ids")}, ensure_ascii=False, indent=2))
    print("src sha256:", record["inputs"]["sha256"][:16], "| dst sha256:", record["output"]["sha256"][:16])


if __name__ == "__main__":
    main()
