#!/usr/bin/env python3
"""Frozen single-arm diagnostic replay, one terminal answer per question.

Reuse the existing endpoint, model, token cap and scoring. Never rerun baseline
arms. Transport failures stop the worker; no automatic model-call retries.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

from backends import DeepSeek
from scoring import SCORING_POLICY, score_answer
from storage import (canonical, cached_response, digest, immutable_json, now,
                     read_json, render_index, run_lock, task_name, write_json,
                     write_text)


HERE = Path(__file__).resolve().parent


def sources():
    return {name: digest((HERE / name).read_bytes()) for name in (
        Path(__file__).name, "backends.py", "storage.py", "scoring.py")}


def progress(run, stage, **details):
    value = {"updated_at": now(), "stage": stage, **details}
    write_json(run / "status.json", value)
    with (run / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(canonical(value).replace("\n", " ") + "\n")
    print(canonical(value), flush=True)


def init(run, baseline, skill, review):
    if run.exists():
        raise RuntimeError("Run folder already exists; use evaluate to resume.")
    m = read_json(baseline / "manifest.json")
    if read_json(baseline / "status.json")["stage"] != "completed":
        raise RuntimeError("Wait for the original evaluation to finish.")
    if m["heldout"]["repeats"] != 1 or m.get("scoring_policy") != SCORING_POLICY:
        raise RuntimeError("Baseline must use one repeat and the same scoring policy.")
    audit = read_json(review / "math_review.json")
    if audit["review"]["verdict"] != "accept" or audit["review"]["issues"]:
        raise RuntimeError("Candidate did not pass its mathematical review.")
    skill_text = skill.read_text(encoding="utf-8")
    if skill_text != (review / "candidate_skill_snapshot.md").read_text(encoding="utf-8"):
        raise RuntimeError("Skill differs from the reviewed snapshot.")
    if not skill_text.startswith("---\n"):
        raise RuntimeError("Expected SKILL.md frontmatter.")
    body = skill_text.split("---", 2)[2].strip() + "\n"
    rows = read_json(baseline / "snapshots/heldout.json")
    controls = {}
    for arm in ("none", "s0", "distilled"):
        controls[arm] = [read_json(baseline / "evaluation" / arm / task_name(r["id"])
                                  / "repeat_01/result.json") for r in rows]
        if any(r["repeat"] != 1 for r in controls[arm]):
            raise RuntimeError("Baseline contains repeated samples.")
    run.mkdir(parents=True)
    write_json(run / "snapshots/questions.json", rows)
    write_json(run / "snapshots/protocol.json", read_json(baseline / "snapshots/protocol.json"))
    write_json(run / "snapshots/baseline_results.json", controls)
    write_json(run / "snapshots/math_review.json", audit)
    write_text(run / "snapshots/SKILL.md", skill_text)
    write_text(run / "snapshots/skill_body.txt", body)
    manifest = {
        "kind": "single_arm_diagnostic_replay", "created_at": now(),
        "sources": sources(), "baseline_run": str(baseline),
        "baseline_manifest_sha256": digest((baseline / "manifest.json").read_bytes()),
        "comparison_evidence": str(review / "comparison.json"),
        "comparison_evidence_sha256": digest((review / "comparison.json").read_bytes()),
        "questions": len(rows), "repeats": 1, "arms": ["candidate"],
        "evaluation": m["evaluation"], "scoring_policy": SCORING_POLICY,
        "automatic_retries": 0,
        "test_informed_candidate": True,
        "interpretation": "Within-dataset diagnostic replay, not an independent generalization test. Historical controls were run earlier; one answer per question cannot establish statistical significance.",
        "snapshots": {p.relative_to(run).as_posix(): digest(p.read_bytes())
                      for p in sorted((run / "snapshots").iterdir())},
    }
    write_json(run / "manifest.json", manifest)
    write_text(run / "manifest.sha256", digest((run / "manifest.json").read_bytes()) + "\n")
    progress(run, "ready", completed=0, total=len(rows))
    report(run)


def verify(run):
    if digest((run / "manifest.json").read_bytes()) != (run / "manifest.sha256").read_text().strip():
        raise RuntimeError("Frozen manifest changed.")
    m = read_json(run / "manifest.json")
    if m["kind"] != "single_arm_diagnostic_replay" or m["sources"] != sources():
        raise RuntimeError("Wrong run type or changed source code.")
    if m["repeats"] != 1 or m["arms"] != ["candidate"]:
        raise RuntimeError("Only one candidate answer per question is allowed.")
    for name, sha in m["snapshots"].items():
        if digest((run / name).read_bytes()) != sha:
            raise RuntimeError("Frozen input changed: " + name)
    return m


def stats(records):
    correct = sum(r["correct"] for r in records)
    return {"completed": len(records), "correct": correct,
            "accuracy": correct / len(records) if records else None,
            "truncated": sum(r.get("truncated", False) for r in records),
            "format_errors": sum(not r["answer_format_valid"] and not r.get("truncated", False)
                                 for r in records)}


def report(run):
    m = read_json(run / "manifest.json")
    current = [read_json(p) for p in sorted((run / "evaluation/candidate").glob("*/repeat_01/result.json"))]
    controls = read_json(run / "snapshots/baseline_results.json")
    outcomes = {str(r["id"]): r for r in current}
    matched = {arm: [r for r in records if str(r["id"]) in outcomes]
               for arm, records in controls.items()}
    summary = {"generated_at": now(), "interpretation": m["interpretation"],
               "complete": len(current) == m["questions"], "planned": m["questions"],
               "candidate": stats(current),
               "matched_baselines": {arm: stats(records) for arm, records in matched.items()},
               "paired_changes": {arm: {
                   "improved_ids": [r["id"] for r in records if not r["correct"] and outcomes[str(r["id"])]["correct"]],
                   "regressed_ids": [r["id"] for r in records if r["correct"] and not outcomes[str(r["id"])]["correct"]],
               } for arm, records in matched.items()}}
    usage = {"attempts": 0, "input_tokens": 0, "output_tokens": 0, "attempts_without_usage": 0}
    for p in run.glob("calls/evaluation/**/meta.json"):
        meta = read_json(p); u = meta.get("usage") or {}
        usage["attempts"] += 1
        usage["attempts_without_usage"] += not bool(u)
        usage["input_tokens"] += u.get("prompt_tokens", 0)
        usage["output_tokens"] += u.get("completion_tokens", 0)
    summary["usage"] = usage
    write_json(run / "summary.json", summary)
    lines = ["新版 skill 单次诊断回测", "", f"更新时间：{summary['generated_at']}",
             f"状态：{'已完成' if summary['complete'] else '尚未完成'}；每题一组一次。",
             "新版参考过 AIME25 作答；这里是同集诊断回测，不是独立泛化验证。",
             "旧组复用历史首次结果，没有新增调用。以下只比较当前新版已完成的相同题目。", ""]
    for arm, s in {**summary["matched_baselines"], "candidate": summary["candidate"]}.items():
        accuracy = f"{s['accuracy']:.1%}" if s["accuracy"] is not None else "—"
        lines.append(f"{arm}: {s['correct']}/{s['completed']} ({accuracy}); 截断 {s['truncated']}; 格式错误 {s['format_errors']}")
    lines += ["", "配对变化（题目 ID 从 0 开始）：", canonical(summary["paired_changes"]),
              "所有达到原输出上限的截断仍计未成功；不依据答案是否碰巧出现改变判分。",
              "单次采样及历史控制无法排除随机波动和时间差，不据此声称统计显著。"]
    write_text(run / "report.txt", "\n".join(lines) + "\n")
    render_index(run)
    return summary


def evaluate(run, client=None):
    m = verify(run)
    rows = read_json(run / "snapshots/questions.json")
    protocol = read_json(run / "snapshots/protocol.json")
    body = (run / "snapshots/skill_body.txt").read_text(encoding="utf-8")
    client = client or DeepSeek(m["evaluation"])
    client.preflight()
    done = 0
    try:
        for row in rows:
            stem = Path("candidate") / task_name(row["id"]) / "repeat_01"
            task = run / "calls/evaluation" / stem
            if list(task.glob("attempt_*/meta.json")) and not cached_response(task, include_truncated=True):
                raise RuntimeError("An earlier nonterminal attempt exists; no automatic repeat call: " + str(task))
            progress(run, "evaluating", completed=done, total=len(rows), current_id=row["id"])
            messages = [
                {"role": "system", "content": protocol["eval_system"].format(skill_section="## Skill\n" + body.strip() + "\n")},
                {"role": "user", "content": "## Problem\n" + row["problem"]},
            ]
            text, attempt = client.call(task, f"AIME25 diagnostic replay / {row['id']} / candidate / once", messages)
            meta = read_json(attempt / "meta.json")
            result = {"id": row["id"], "arm": "candidate", "repeat": 1,
                      **score_answer(text, row["answer"], meta.get("finish_reason")),
                      "attempt": attempt.relative_to(run).as_posix()}
            immutable_json(run / "evaluation" / stem / "result.json", result)
            done += 1
            report(run)
            time.sleep(0.1)
        progress(run, "completed", completed=done, total=len(rows))
    except KeyboardInterrupt:
        progress(run, "interrupted", completed=done, total=len(rows))
        raise
    except Exception as exc:
        progress(run, "failed", completed=done, total=len(rows), error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report(run)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["init", "evaluate", "report", "verify"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--skill", type=Path)
    p.add_argument("--review", type=Path)
    args = p.parse_args(); run = args.run.resolve()
    if args.command == "init":
        if not all([args.baseline, args.skill, args.review]):
            p.error("init requires --baseline, --skill, --review")
        init(run, args.baseline.resolve(), args.skill.resolve(), args.review.resolve())
    elif args.command == "report":
        report(run)
    elif args.command == "verify":
        m = verify(run); print(canonical({"verified": True, "questions": m["questions"], "repeats": 1}))
    else:
        def stop(_sig, _frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, stop)
        with run_lock(run):
            evaluate(run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
