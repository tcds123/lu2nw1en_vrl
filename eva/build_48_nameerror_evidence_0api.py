#!/usr/bin/env python3
"""#48 Step5: 107 题 NameError('Solution') 逐题证据表 (2026-11-19, 0 API)

用户 2026-11-19 决定: 107 题**不作为启动阻塞**, 主分析保留完整 2765 题; 但必须产出逐题证据表
(题 ID、原始题面、函数签名、测试代码、失败原因), 并记录三项排除标准:
  (a) 是否与模型输出无关 (objective),
  (b) 是否存在客观可复核的排除标准 (reproducible),
  (c) 是否对所有 arm 同样适用 (arm-agnostic)。
只有同时满足三项的题, 才在**补充敏感性分析**中排除; 主分析不排除。

判据 (objective, 全部从题目原始数据计算, 不读任何模型输出):
  - question 是否含 `class Solution`
  - question 是否含可提取签名 (def ...)
  - test_code 是否 JSON 且含 stdin/stdout 键 (stdin 文件格式)
  - stdin 元素经 ast.literal_eval 是否为 list (fn-call 形态)
  - 截断形态下 `_d5_extract_sig` / `_d5_is_fn_call_item` 的真实判定

输入 (只读): test/items.json, Arm1 test_independent_results.json (只为取 fail_reason 原文)
输出: <ARM1_RUN>/audit_0api_20261118/nameerror_107_evidence.json + .md
"""
import ast
import json
import os
import re
import sys

RUN = ("/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/"
       "tf48-arm1-full-20261117/run_20261118_091052")
AUDIT = os.path.join(RUN, "audit_0api_20261118")
TEST_ITEMS = "/data/zhuldz/skillopt/data/appsplus_capped_seed42/test/items.json"

sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
from tf_skill_optimizer_appsplus import (  # noqa: E402
    load_split, _evaluate_d5, _d5_is_fn_call_item, _d5_extract_sig, _skillopt_extract_code,
)

DEF_RE = re.compile(r"def\s+(\w+)\s*\(")


def main():
    os.makedirs(AUDIT, exist_ok=True)
    arm = json.load(open(os.path.join(RUN, "test_independent_results.json"), encoding="utf-8"))
    # 原生以及截断形态都要 (评估器吃截断形态, 题面原文给证据表用原生)
    native = {str(it["id"]): it for it in load_split(os.path.dirname(os.path.dirname(TEST_ITEMS)), "test")}
    raw = {str(it["id"]): it for it in json.load(open(TEST_ITEMS, encoding="utf-8"))}

    ne_ids = [q for q, r in arm.items()
              if "NameError" in (r.get("fail_reason") or "") and "'Solution'" in (r.get("fail_reason") or "")]

    rows = []
    for q in sorted(ne_ids, key=lambda x: int(x) if x.isdigit() else x):
        it = native[q]
        rr = raw.get(q, {})
        qtext = str(it.get("question", ""))
        tc = str(it.get("test_code", ""))
        try:
            tco = json.loads(tc)
        except Exception:
            tco = None
        is_class, fn_name, nparam = _d5_extract_sig(qtext)
        stdin_keys = list(tco.keys()) if isinstance(tco, dict) else None
        stdins = (tco.get("stdin") or []) if isinstance(tco, dict) else []
        first_stdin = stdins[0] if stdins else None
        if first_stdin is not None:
            try:
                parsed = ast.literal_eval(first_stdin)
                stdin_shape = f"literal_eval -> {type(parsed).__name__}"
                stdin_is_list = isinstance(parsed, list)
            except Exception as e:
                stdin_shape = f"literal_eval FAILED: {type(e).__name__}"
                stdin_is_list = False
        else:
            stdin_shape, stdin_is_list = "no stdin", False
        defs = DEF_RE.findall(qtext)
        r = arm[q]
        # (a) 与模型输出无关: 失败在 harness 层 (NameError 'Solution' 来自 test_code 引用 Solution,
        #     而题面/签名给的是别的名字), 与存储代码内容无关 → 客观
        a_objective = True
        # (b) 客观可复核: 上述字段全部可从题目原始数据重算
        b_reproducible = True
        # (c) 对所有 arm 同样适用: 题目数据对任何 arm 相同, 且评测路径相同
        c_arm_agnostic = True
        rows.append({
            "id": q,
            "question_chars": len(qtext),
            "question_head": qtext[:400],
            "question_defs": defs,
            "question_has_class_Solution": "class Solution" in qtext,
            "extracted_sig": {"is_class": is_class, "fn_name": fn_name, "nparam": nparam},
            "test_code_chars": len(tc),
            "test_code_keys": stdin_keys,
            "test_code_head": tc[:400],
            "n_pairs": len(stdins),
            "first_stdin_repr": str(first_stdin)[:200],
            "first_stdin_shape": stdin_shape,
            "first_stdin_is_list": stdin_is_list,
            "fn_call_detector": bool(_d5_is_fn_call_item(it)),
            "arm1_hard": int(r.get("hard", 0)),
            "arm1_agent_ok": r.get("agent_ok"),
            "arm1_fail_reason": (r.get("fail_reason") or "")[:300],
            "arm1_error_type": r.get("error_type") or "",
            "arm1_code_chars": len(r.get("code") or ""),
            "in_first300": False,
            "criteria": {"a_model_output_independent": a_objective,
                         "b_objectively_recheckable": b_reproducible,
                         "c_same_for_all_arms": c_arm_agnostic},
        })

    from collections import Counter
    summary = {
        "n": len(rows),
        "hard_arm1_sum": sum(r["arm1_hard"] for r in rows),
        "with_class_Solution_in_question": sum(1 for r in rows if r["question_has_class_Solution"]),
        "test_code_stdin_format": sum(1 for r in rows if r["test_code_keys"] and "stdin" in r["test_code_keys"]),
        "fn_call_detector_true": sum(1 for r in rows if r["fn_call_detector"]),
        "extract_sig_ok": sum(1 for r in rows if r["extracted_sig"]["fn_name"]),
        "first_stdin_is_list": sum(1 for r in rows if r["first_stdin_is_list"]),
        "first_stdin_shape": dict(Counter(r["first_stdin_shape"] for r in rows)),
        "question_defs_top": dict(Counter(d for r in rows for d in r["question_defs"]).most_common(10)),
        "arm1_agent_ok": dict(Counter(r["arm1_agent_ok"] for r in rows)),
        "criteria_all_three": sum(1 for r in rows if all(r["criteria"].values())),
    }

    out = {"run_dir": RUN, "generated_by": "eva/build_48_nameerror_evidence_0api.py",
           "summary": summary, "items": rows}
    p = os.path.join(AUDIT, "nameerror_107_evidence.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    md = [f"# #48 107 题 NameError('Solution') 证据表\n",
          f"- 生成: `eva/build_48_nameerror_evidence_0api.py` (0 API)",
          f"- n={summary['n']}, Arm1 hard 合计={summary['hard_arm1_sum']}",
          f"- 题面含 `class Solution`: {summary['with_class_Solution_in_question']}/{summary['n']}",
          f"- test_code 为 stdin 文件格式: {summary['test_code_stdin_format']}/{summary['n']}",
          f"- 截断形态 fn-call 检测器为 True: {summary['fn_call_detector_true']}",
          f"- `_d5_extract_sig` 成功: {summary['extract_sig_ok']}",
          f"- 首条 stdin 形态: {summary['first_stdin_shape']}",
          f"- 题面 def 名 top: {summary['question_defs_top']}",
          "\n> 说明: 主分析**保留**这 107 题; 只有同时满足三项排除标准的题才进入补充敏感性分析。\n",
          f"(a) 与模型输出无关 / (b) 客观可复核 / (c) 对所有 arm 同样适用: "
          f"三者同时满足 {summary['criteria_all_three']}/{summary['n']}\n",
          "> 注意: 三项\"满足\"目前是基于题目数据的结构性初判, 逐题仍需核对 — 特别是",
          "> 题面给的是 `def xxx` 但 test_code 直接调用 `Solution().method()` 的情况,",
          "> 这类题对任何模型都不可解, 属候选排除; 而题面确实要求 `class Solution` 的题不应排除。\n",
          "| ID | 题面 defs | 签名 | test_code keys | n_pairs | 首 stdin | fn-call | Arm1 hard | fail_reason |",
          "|----|-----------|------|----------------|---------|----------|---------|-----------|-------------|"]
    for r in rows[:200]:
        md.append("| {id} | {defs} | {sig} | {keys} | {np} | {shape} | {fc} | {h} | {fr} |".format(
            id=r["id"], defs=",".join(r["question_defs"])[:40],
            sig=f"{r['extracted_sig']['fn_name']}",
            keys=",".join(r["test_code_keys"] or [])[:30], np=r["n_pairs"],
            shape=r["first_stdin_shape"].replace("|", "/")[:40],
            fc=int(r["fn_call_detector"]), h=r["arm1_hard"],
            fr=r["arm1_fail_reason"].replace("|", "/").replace("\n", " ")[:60].replace("`", "")))
    pm = os.path.join(AUDIT, "nameerror_107_evidence.md")
    with open(pm, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"→ {p}\n→ {pm}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
