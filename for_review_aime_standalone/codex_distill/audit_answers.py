"""Read-only audit of saved requests and responses; never call a model."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re

from scoring import score_answer, SCORING_POLICY
from storage import canonical, digest, read_json, write_json, write_text, now


def audit(run, output):
    run = Path(run).resolve()
    output = Path(output).resolve()
    manifest = read_json(run / "manifest.json")
    assert digest((run / "manifest.json").read_bytes()) == (run / "manifest.sha256").read_text().strip()
    for name, sha in manifest["snapshots"].items():
        assert digest((run / name).read_bytes()) == sha, name
    protocol = read_json(run / "snapshots/protocol.json")
    problems = {str(r["id"]): r for r in read_json(run / "snapshots/heldout.json")}
    skills = {"none": "", "s0": (run / "snapshots/s0.md").read_text(),
              "distilled": (run / "distillation/skill.md").read_text()}
    records = []
    failures = []
    models = Counter()
    response_ids = []
    paired = defaultdict(dict)
    for path in sorted(run.glob("evaluation/**/result.json")):
        old = read_json(path)
        attempt = run / old["attempt"]
        request = read_json(attempt.parent / "request.json")
        meta = read_json(attempt / "meta.json")
        for name, sha in meta.get("artifact_sha256", {}).items():
            assert digest((attempt / name).read_bytes()) == sha, str(attempt / name)
        raw = read_json(attempt / "response.json")
        response_ids.append(raw.get("id"))
        models[raw.get("model")] += 1
        choice = raw["choices"][0]
        text = choice["message"].get("content") or ""
        finish = choice["finish_reason"]
        row = problems[str(old["id"])]
        skill = skills[old["arm"]].strip()
        section = "## Skill\n" + skill + "\n" if skill else ""
        expected = {"model": manifest["evaluation"]["model"],
                    "messages": [{"role": "system", "content": protocol["eval_system"].format(skill_section=section)},
                                 {"role": "user", "content": "## Problem\n" + row["problem"]}],
                    "temperature": manifest["evaluation"]["temperature"],
                    "max_tokens": manifest["evaluation"]["max_tokens"], "stream": False}
        if expected != request["payload"] or request["url"] != manifest["evaluation"]["base_url"] + "/chat/completions":
            failures.append({"type": "request_mismatch", "record": str(path.relative_to(run))})
        if text != (attempt / "response.txt").read_text() or finish != meta.get("finish_reason"):
            failures.append({"type": "response_mismatch", "record": str(path.relative_to(run))})
        if int(row["answer"]) != old["correct_answer"]:
            failures.append({"type": "gold_mismatch", "record": str(path.relative_to(run))})
        legacy_tags = re.findall(r"<answer>\s*\\boxed\{\s*(\d{1,3})\s*\}\s*</answer>", text)
        legacy_pred = int(legacy_tags[-1]) if legacy_tags and finish != "length" else None
        new = score_answer(text, row["answer"], finish)
        record = {"id": old["id"], "repeat": old["repeat"], "arm": old["arm"],
                  "attempt": old["attempt"], "previous_record_correct": old["correct"],
                  "legacy_format_score": legacy_pred == int(row["answer"]),
                  "math_score": new["correct"], "corrected_prediction": new["predicted_answer"],
                  "format_valid": new["answer_format_valid"], "truncated": new.get("truncated", False),
                  "gold": int(row["answer"]), "response_sha256": digest(text)}
        records.append(record)
        paired[(str(old["id"]), old["repeat"])][old["arm"]] = record
    common = {k: v for k, v in paired.items() if len(v) == 3}
    summary = {"audited_at": now(), "source_run": str(run), "policy": SCORING_POLICY,
               "request_or_data_errors": failures, "record_count": len(records),
               "unique_response_ids": len(set(response_ids)), "returned_models": dict(models),
               "common_cells": len(common), "arms": {}}
    for arm in skills:
        rows = [r for r in records if r["arm"] == arm]
        matched = [v[arm] for v in common.values()]
        summary["arms"][arm] = {"records": len(rows),
            "legacy_format_correct": sum(x["legacy_format_score"] for x in rows),
            "math_correct": sum(x["math_score"] for x in rows),
            "format_only_false_negatives": sum(x["math_score"] and not x["legacy_format_score"] for x in rows),
            "common_legacy_correct": sum(x["legacy_format_score"] for x in matched),
            "common_math_correct": sum(x["math_score"] for x in matched),
            "common_truncated": sum(x["truncated"] for x in matched)}
    summary["paired_differences"] = [
        {"id": k[0], "repeat": k[1], "s0": v["s0"], "distilled": v["distilled"]}
        for k, v in common.items() if v["s0"]["math_score"] != v["distilled"]["math_score"]]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_json(output / "records.json", records)
    lines = ["# 已保存回复的答案评分审计", "", f"原始实验：{run}", "",
             f"核验了 {len(records)} 份请求与回复，输入/回复/答案映射异常数：{len(failures)}。",
             "原始输出、原评分和冻结 skill 均未修改；本审计没有调用模型。", "",
             "发现：原评分同时要求数字正确和 XML 标签正确，把缺标签的正确 boxed 整数也判错。",
             "数学答案正确与格式合规应分开；以下对三组按同一规则离线重算。", "",
             f"下表仅比较三组均完成的相同题目与重复次数，共 {len(common)} 次/组。", "",
             "| 配置 | 原格式判据正确数 | 数学答案正确数 | 截断 |",
             "|---|---:|---:|---:|"]
    for arm, stats in summary["arms"].items():
        lines.append(f"| {arm} | {stats['common_legacy_correct']}/{len(common)} | "
                     f"{stats['common_math_correct']}/{len(common)} | {stats['common_truncated']} |")
    lines += ["", "截断样本仍作为预算内未完成作答计入分母，不从截断内容提取答案。",
              "只接受明确的最终 boxed 整数或 answer 标签，不在正文中猜数字，也不根据标准答案选择候选值。",
              "评分修正在查看部分输出后发生，原格式指标同时保留；不得把此次数学评分修正描述为预先注册的指标。",
              "这些是部分题目的结果，不能当作完整 AIME25 的迁移结论。"]
    write_text(output / "report.md", "\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.out)
    print(canonical(result))
