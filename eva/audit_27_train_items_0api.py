#!/usr/bin/env python3
"""#13 数据基础审计 — 27 道实际训练题逐题产物 (0 API, 只读).

产出 (输出目录默认 `rl/train/verl1/outputs/crossmodel_baseline/phase4a-audit-20261120/`):
  audit_27_train_items.json   机器产物
  audit_27_train_items.md     人读表 + 判读

数据来源 (全部已有文件, 不发任何请求):
  for_review/data/appsplus_capped_seed42/{train,test}/items.json     题面 + test_code
  .../tf-appsplus-routing-formal-20261113/train_s0_cache.json       80 题 S₀ hard/soft
  .../phase4a-audit-20261120/near_dup_ids.json                      21 道近重复 test 题
  .../tf48-arm1-full-20261117/run_*/test_independent_results.json   Arm1 逐题结果

口径:
  - fn_call: 与 evolver 内部 `_d5_is_fn_call_item` **同一检测器** (runtime 形态):
    test_code.stdin[0] 能被 ast.literal_eval 成 list。这是"题目属于 fn-call 形态"的定义,
    不等于"Interface 条款实际干预了该题" (Issue #93, 后者需要看 skill 文本无法逐题归因)。
  - class Solution: question 里出现 "class Solution"。
  - has_example: question 里出现 "Example" 块。
"""

import ast
import json
import os
import re

ROOT = "/data/zhuldz/lunwen"
CAPPED = f"{ROOT}/for_review/data/appsplus_capped_seed42"
ROUTING = f"{ROOT}/rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-formal-20261113"
AUDIT = f"{ROOT}/rl/train/verl1/outputs/crossmodel_baseline/phase4a-audit-20261120"
ARM1 = f"{ROOT}/rl/train/verl1/outputs/crossmodel_baseline/tf48-arm1-full-20261117"
OUT_DIR = AUDIT


def is_fn_call(item):
    """与 tf_skill_optimizer_appsplus._d5_is_fn_call_item 逐字一致。"""
    try:
        obj = json.loads(item.get("test_code", ""))
    except Exception:
        return False
    si = obj.get("stdin") or [None]
    if si[0] is None:
        return False
    try:
        return isinstance(ast.literal_eval(si[0]), list)
    except Exception:
        return False


def truncate_test_code(test_code, max_pairs=8, max_case_bytes=4096):
    """与 json2parquet_skillopt.truncate_test_code 同语义 (复刻以避免 import 依赖)。

    关键: 每对用例被 `str(...)` 化 → raw 的 list 输入变成 `"[...]"` 字符串,
    这正是 fn-call 在 raw 形态下检测为 0、在 runtime 截断形态下检测为 N 的机制 (Issue #93)。
    """
    try:
        obj = json.loads(test_code)
    except (json.JSONDecodeError, TypeError):
        return test_code
    stdins = obj.get("stdin") or []
    stdouts = obj.get("stdout") or []
    n = min(len(stdins), len(stdouts))
    if n == 0:
        return test_code
    pairs = [(str(stdins[i]), str(stdouts[i])) for i in range(n)]
    small = [p for p in pairs if len(p[0]) <= max_case_bytes and len(p[1]) <= max_case_bytes]
    if small:
        small.sort(key=lambda p: len(p[0]) + len(p[1]))
        chosen = small[:max_pairs]
    else:
        pairs.sort(key=lambda p: len(p[0]) + len(p[1]))
        s, o = pairs[0]
        chosen = [(s[:max_case_bytes], o[:max_case_bytes])]
    return json.dumps({"stdin": [p[0] for p in chosen], "stdout": [p[1] for p in chosen]}, ensure_ascii=False)


def find_arm1_results():
    for d in sorted(os.listdir(ARM1), reverse=True):
        p = os.path.join(ARM1, d, "test_independent_results.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Arm1 test_independent_results.json not found")


def main():
    train_items = json.load(open(f"{CAPPED}/train/items.json", encoding="utf-8"))
    test_items = json.load(open(f"{CAPPED}/test/items.json", encoding="utf-8"))
    s0 = json.load(open(f"{ROUTING}/train_s0_cache.json", encoding="utf-8"))
    near = [str(x) for x in json.load(open(f"{AUDIT}/near_dup_ids.json", encoding="utf-8"))]
    arm1_path = find_arm1_results()
    _arm1_raw = json.load(open(arm1_path, encoding="utf-8"))
    arm1 = (_arm1_raw if isinstance(_arm1_raw, dict)
            else {str(r["id"]): r for r in _arm1_raw})
    arm1 = {str(k): v for k, v in arm1.items()}

    by_id = {str(it["id"]): it for it in train_items}
    assert len(by_id) == 80, len(by_id)

    fail_ids = sorted([k for k, v in s0.items() if int(v["hard"]) == 0], key=int)
    rows = []
    for i, tid in enumerate(fail_ids, 1):
        it = by_id[tid]
        q = it["question"]
        rows.append({
            "rank": i,                  # 在 80 题中的文件顺序位 (1-based)
            "id": tid,
            "difficulty": it.get("difficulty"),
            "fn_call_raw": is_fn_call(it),            # 直接读盘 (preassess 口径)
            "fn_call_d5": is_fn_call({**it, "test_code": truncate_test_code(it["test_code"])}),  # runtime 口径
            "stdin0_raw": (json.loads(it["test_code"]).get("stdin") or [None])[0],
            "class_solution": "class Solution" in q,
            "has_example": "Example" in q,
            "has_input_call": "input(" in q,
            "def_line": next((ln.strip() for ln in q.splitlines() if ln.startswith("def ")), None),
            "s0_hard": int(s0[tid]["hard"]),
            "s0_soft": s0[tid].get("soft"),
            "s0_pass_rate": s0[tid].get("pass_rate"),
            "s0_fail_reason_head": (s0[tid].get("fail_reason") or "")[:80],
            "line_is_train_3953_dup_source": tid == "3953",
        })

    # class 判定按**整题面**, 而 def_line 只抓文件首行 -> 单独从 question 全文判
    for r in rows:
        r["class_solution"] = "class Solution" in by_id[r["id"]]["question"]
    fn_ids = [r["id"] for r in rows if r["fn_call_d5"]]
    fn_raw_ids = [r["id"] for r in rows if r["fn_call_raw"]]
    fn_train_all = [str(it["id"]) for it in train_items if
                    is_fn_call({**it, "test_code": truncate_test_code(it["test_code"])})]
    cls_ids = [r["id"] for r in rows if r["class_solution"]]
    ex_ids = [r["id"] for r in rows if r["has_example"]]

    # 21 道近重复 test 题 + Arm1 逐题结果
    tset = {str(it["id"]) for it in test_items}
    near_rows = []
    for nid in near:
        r = arm1.get(nid, {})
        near_rows.append({
            "id": nid, "in_test_split": nid in tset,
            "arm1_hard": int(r["hard"]) if r else None,
            "arm1_fail_reason_head": (r.get("fail_reason") or "")[:80] if r else None,
        })
    near_pass = sum(1 for x in near_rows if x["arm1_hard"] == 1)

    payload = {
        "meta": {
            "source": "0 API read-only", "arm1_results": arm1_path,
            "detector": "_d5_is_fn_call_item (test_code.stdin[0] literal_eval → list)",
            "note": "fn_call_d5 = 题目属于 fn-call 形态, 不等于 Interface 条款实际干预该题 (Issue #93)",
        },
        "counts": {
            "train_capped": 80, "s0_pass_in_80": len(s0) - len(fail_ids),
            "fail_pool": len(fail_ids),
            "fn_call": len(fn_ids), "fn_call_raw": len(fn_raw_ids),
            "fn_call_train_all80": len(fn_train_all),
            "fn_call_ids": sorted(fn_ids, key=int),
            "class_solution": len(cls_ids), "has_example": len(ex_ids),
            "fn_and_class": len(set(fn_ids) & set(cls_ids)),
            "fn_not_class": sorted(set(fn_ids) - set(cls_ids), key=int),
            "class_not_fn": sorted(set(cls_ids) - set(fn_ids), key=int),
            "no_example": sorted(set(fail_ids) - set(ex_ids), key=int),
            "near_dup_test": len(near_rows), "near_dup_arm1_pass": near_pass,
            "near_dup_all_in_test": all(x["in_test_split"] for x in near_rows),
        },
        "train_fail_items": rows,
        "near_dup_test_items": near_rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    jp = f"{OUT_DIR}/audit_27_train_items.json"
    json.dump(payload, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    c = payload["counts"]
    L = []
    L.append("# 27 道实际训练题逐题产物 (#13, 0 API 只读)\n")
    L.append(f"产物: `{jp}` (脚本 `eva/audit_27_train_items_0api.py`)\n")
    L.append("## 口径与筛选链\n")
    L.append("790 train ⇒ 文件顺序前 80 (`trainer.py:432` train_size=80 硬匹配) ⇒ S₀ 失败 "
             f"**{c['fail_pool']}** 题 (80 中 S₀ 做出 {c['s0_pass_in_80']} 题)。\n")
    L.append(f"- fn-call (D5 **runtime 截断形态**, 与 evolver `coverage.fn_call_in_fail_pool={c['fn_call']}` 对齐): "
             f"**{c['fn_call']}** 题 — {', '.join(c['fn_call_ids'])}")
    L.append(f"- fn-call (raw 直接读盘形态, preassess 口径): **{c['fn_call_raw']}** 题 — "
             "这正是 Issue #93 `0 vs 9` 的完整来源 (机制见下)")
    L.append(f"- 80 题训练集内 runtime 形态 fn-call: **{c['fn_call_train_all80']}** 题, 全部 S₀ 失败 → "
             "与 log 的 `fn_call_in_train_used=9` 一致")
    L.append("\n### #93 机制 (2026-11-24 已定位, 0 API)\n")
    L.append("`truncate_test_code` (`json2parquet_skillopt.py:105`) 把每对用例做 `str(...)` 化: raw 的 list 输入 "
             "`[\"0110111\"]` 变成字符串 `'[\"0110111\"]'`。检测器 `ast.literal_eval(stdin[0]) is list`:")
    L.append("- raw 盘上: `literal_eval(['\"0110111\"'])` 对 **list 入参**抛 ValueError (示例题 1430 实测) → 返回 False, "
             "27 题 fail 池计 **0** (preassess 口径);")
    L.append("- runtime 截断后: `literal_eval('[\"0110111\"]')` → list → True。9 道题在 80 题内 **全部 S₀ 失败**, "
             "进入 27 题训练池。")
    L.append("⇒ **计数差异是形态差异, 不是错误**: 9 = `fn_call_in_fail_pool` = `fn_call_in_train_used` 皆成立。"
             "仍需记住: 这是**题目形态**, 不是 Interface 条款实际干预了这 9 题 (Issue #93 四问仍未闭合)。")
    L.append(f"\n- class Solution: **{c['class_solution']}** 题; 带 Example 块: **{c['has_example']}** 题; "
             f"无 Example: {len(c['no_example'])} 题 {c['no_example']}")
    L.append(f"- runtime fn-call 且 class: **{c['fn_and_class']}** 题; runtime fn-call 但题面是 stdin 伪装: "
             f"{', '.join(c['fn_not_class']) or '无'}; 题面 class 但 test_code 是 stdin 形态: "
             f"{', '.join(c['class_not_fn']) or '无'} (结构性不可解, 见 #104)")
    L.append("\n⚠️ `fn_call_d5=True` 只说明**题目属于 fn-call 形态**, 不等同于 Interface 条款实际干预了该题 "
             "(Issue #93 四问仍未解决)。\n")
    L.append("## 27 题逐题表\n")
    L.append("| # | id | runtime 形态 | raw stdin[0] 类型 | class | Example | input() | S₀ pass | S₀ 失败摘要 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        form = ("**fn-call**" if r["fn_call_d5"]
                else ("class·stdin" if r["class_solution"] else "stdin"))
        s0v = r["stdin0_raw"]
        t = ("str" if isinstance(s0v, str) else
             "list" if isinstance(s0v, list) else
             "int" if isinstance(s0v, int) else type(s0v).__name__)
        L.append(f"| {r['rank']} | {r['id']}{' ⭐' if r['line_is_train_3953_dup_source'] else ''} | {form} | {t} | "
                 f"{'✓' if r['class_solution'] else ''} | {'✓' if r['has_example'] else ''} | "
                 f"{'✓' if r['has_input_call'] else ''} | {r['s0_pass_rate']} | "
                 f"{(r['s0_fail_reason_head'].splitlines() or [''])[0][:40]} |")
    L.append("\n⭐ = 21 道近重复 test 题共同的来源题 (#3953)。\n")
    L.append("## 21 道近重复 test 题 + Arm1 逐题结果\n")
    L.append(f"全部在 2765 test 集内: **{c['near_dup_all_in_test']}**; Arm1 通过 **{near_pass}/{len(near_rows)}**。\n")
    L.append("| id | in test | Arm1 | 失败摘要 |")
    L.append("|---|---|---|---|")
    for x in near_rows:
        L.append(f"| {x['id']} | {'✓' if x['in_test_split'] else '✗'} | "
                 f"{'pass' if x['arm1_hard'] == 1 else 'fail'} | "
                 f"{(x['arm1_fail_reason_head'] or '').splitlines()[0][:40] if x['arm1_fail_reason_head'] else ''} |")
    L.append("\n关联: [[ablations-consolidated-20261124]] [[phase4a-verdict-20261120]] Issue #93 / #116")
    mp = f"{OUT_DIR}/audit_27_train_items.md"
    open(mp, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(json.dumps(c, ensure_ascii=False, indent=1))
    print("wrote", jp, mp)


if __name__ == "__main__":
    main()
