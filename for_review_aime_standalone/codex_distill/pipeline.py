"""Collect -> independently review -> distill -> freeze -> held-out evaluation."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import prompts
from backends import Codex, DeepSeek, ROOT, eval_settings
from scoring import SCORING_POLICY, extract_answer, score_answer
from storage import (canonical, digest, immutable_json, now, read_json,
                     render_index, task_name, write_json, write_text)


HERE = Path(__file__).resolve().parent
VERSION = 1


def code_hashes():
    return {name: digest((HERE / name).read_bytes()) for name in
            ("run.py", "pipeline.py", "backends.py", "prompts.py", "storage.py", "scoring.py")}


def load_data(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(r["id"]) for r in rows]
    problems = [digest(" ".join(r["problem"].split())) for r in rows]
    if not rows or len(set(ids)) != len(ids) or len(set(problems)) != len(problems):
        raise RuntimeError(f"数据为空或存在重复 ID / 题目: {path}")
    for r in rows:
        if not r["problem"].strip() or not str(r["answer"]).strip().isdigit() or not 0 <= int(r["answer"]) <= 999:
            raise RuntimeError(f"题目或答案无效: {r['id']}")
    return rows


def init_run(run, *, limit=30, samples=1, repeats=3, binary="codex", model="gpt-6.1-sol",
             effort="high", timeout=600, eval_timeout=300, temperature=0.0, max_tokens=8192):
    run = Path(run).resolve()
    if run.exists():
        raise RuntimeError("实验文件夹已存在；请选择新名称，或对原目录执行 collect/distill/evaluate。")
    train = load_data(ROOT / "data/aime24.jsonl")
    test = load_data(ROOT / "data/aime25.jsonl")
    if not 1 <= limit <= len(train) or samples < 1 or repeats < 1:
        raise RuntimeError("limit、samples 或 repeats 无效。")
    if min(timeout, eval_timeout, max_tokens) <= 0 or not 0 <= temperature <= 2:
        raise RuntimeError("超时、输出上限或温度无效。")
    train_hashes = {digest(" ".join(r["problem"].split())) for r in train}
    if train_hashes & {digest(" ".join(r["problem"].split())) for r in test}:
        raise RuntimeError("训练集和留出集存在重复题目。")
    train = train[:limit]
    run.mkdir(parents=True)
    write_json(run / "snapshots/train.json", train)
    write_json(run / "snapshots/heldout.json", test)
    write_text(run / "snapshots/s0.md", (ROOT / "data/skill_aime_initial_v2.md").read_text(encoding="utf-8"))
    write_json(run / "snapshots/protocol.json", prompts.protocol())
    final = eval_settings()
    manifest = {
        "version": VERSION, "created_at": now(), "sources": code_hashes(),
        "scoring_policy": SCORING_POLICY,
        "snapshots": {p.relative_to(run).as_posix(): digest(p.read_bytes())
                      for p in sorted((run / "snapshots").iterdir())},
        "training": {"dataset": "AIME24", "questions": len(train), "samples": samples},
        "heldout": {"dataset": "AIME25", "questions": len(test), "repeats": repeats,
                    "arms": ["none", "s0", "distilled"]},
        "codex": {"binary": binary, "model": model, "reasoning_effort": effort, "timeout_s": timeout},
        "evaluation": {"base_url": final["base_url"], "model": final["model"],
                       "aigc_user": final["aigc_user"], "temperature": temperature,
                       "max_tokens": max_tokens, "timeout_s": eval_timeout},
        "limits": ["Only CLI-visible events and public solutions are archived, not private reasoning.",
                   "Reviewer is the same model in a separate fresh call, not an independent ground-truth proof checker.",
                   "Repeated API calls are replicate labels; no deterministic seed is requested.",
                   "Public benchmark questions may be present in model pretraining.",
                   "Do not select or revise skills using held-out results."],
    }
    write_json(run / "manifest.json", manifest)
    # Separate digest prevents accidental manifest edits from silently changing a resumed run.
    write_text(run / "manifest.sha256", digest((run / "manifest.json").read_bytes()) + "\n")
    progress(run, "initialized", "实验已建立，尚未调用模型")
    report(run)
    return run


def verify_run(run):
    run = Path(run)
    if digest((run / "manifest.json").read_bytes()) != (run / "manifest.sha256").read_text().strip():
        raise RuntimeError("实验配置被修改，拒绝继续。")
    manifest = read_json(run / "manifest.json")
    if manifest["version"] != VERSION or manifest["sources"] != code_hashes():
        raise RuntimeError("流程代码已变化，请为新版本建立新实验，避免混用结果。")
    for name, sha in manifest["snapshots"].items():
        if digest((run / name).read_bytes()) != sha:
            raise RuntimeError(f"冻结输入被修改: {name}")
    return manifest


def progress(run, stage, message, **details):
    status = {"updated_at": now(), "stage": stage, "message": message, **details}
    write_json(Path(run) / "status.json", status)
    with (Path(run) / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(status, ensure_ascii=False) + "\n")
        f.flush()
    print(f"[{stage}] {message}", flush=True)


def parse_structured(text, attempt, schema):
    """Validate semantic minimums as well as the CLI's requested JSON schema."""
    def validate(value, spec):
        typ = spec["type"]
        if typ == "object":
            if not isinstance(value, dict) or set(value) != set(spec["required"]):
                raise ValueError("对象字段不完整或含额外字段")
            for k, sub in spec["properties"].items():
                validate(value[k], sub)
        elif typ == "array":
            if not isinstance(value, list) or len(value) < spec.get("minItems", 0):
                raise ValueError("数组缺失或长度不足")
            for item in value:
                validate(item, spec["items"])
        elif typ == "integer":
            if type(value) is not int or not spec.get("minimum", 0) <= value <= spec.get("maximum", 999):
                raise ValueError("整数答案不在允许范围内")
        elif typ == "string":
            if not isinstance(value, str) or not value.strip() or ("enum" in spec and value not in spec["enum"]):
                raise ValueError("字符串为空或不属于允许值")
    try:
        value = json.loads(text)
        validate(value, schema)
        return value
    except (ValueError, TypeError, KeyError) as exc:
        write_json(Path(attempt) / "invalid.json", {"reason": "结构化输出校验失败", "detail": str(exc)})
        raise RuntimeError(f"结构化输出无效，记录保留；再次执行将新建尝试: {attempt}") from None


def collect(run, client=None):
    run = Path(run)
    m = verify_run(run)
    p = read_json(run / "snapshots/protocol.json")
    rows = read_json(run / "snapshots/train.json")
    client = client or Codex(m["codex"])
    client.preflight()
    for i, row in enumerate(rows, 1):
        q = task_name(row["id"])
        for sample in range(1, m["training"]["samples"] + 1):
            label = f"AIME24 题目 {row['id']} / 样本 {sample}"
            stem = f"{q}/sample_{sample:02d}"
            progress(run, "collecting", f"{i}/{len(rows)} {label}：采集解法")
            text, attempt = client.call(run / "calls/solve" / stem, label + " / 解题",
                                        p["solve"].format(problem=row["problem"]), p["solve_schema"])
            solution = parse_structured(text, attempt, p["solve_schema"])
            correct = solution["answer"] == int(row["answer"])
            review, review_attempt = None, None
            if correct:
                progress(run, "reviewing", f"{i}/{len(rows)} {label}：独立复核")
                review_text, review_attempt = client.call(run / "calls/review" / stem, label + " / 复核",
                    p["review"].format(problem=row["problem"], solution=text), p["review_schema"])
                review = parse_structured(review_text, review_attempt, p["review_schema"])
            accepted = bool(correct and review and review["verdict"] == "accept" and review["methods"])
            result = {"id": row["id"], "sample": sample, "predicted_answer": solution["answer"],
                      "correct_answer": int(row["answer"]), "answer_correct": correct,
                      "accepted_for_distillation": accepted, "solution": solution, "review": review,
                      "solve_attempt": attempt.relative_to(run).as_posix(),
                      "review_attempt": review_attempt.relative_to(run).as_posix() if review_attempt else None}
            # Attempt paths may change only when a previous invalid/failed attempt is retried.
            immutable_json(run / "collection" / stem / "result.json", result)
            render_index(run)
    progress(run, "collected", "训练集采集与复核已完成")
    report(run)


def training_results(run, manifest):
    rows = read_json(run / "snapshots/train.json")
    results = []
    for row in rows:
        for sample in range(1, manifest["training"]["samples"] + 1):
            path = run / "collection" / task_name(row["id"]) / f"sample_{sample:02d}" / "result.json"
            if not path.exists():
                raise RuntimeError("训练集尚未全部采集，请先执行 collect。")
            result = read_json(path)
            # Validate the selected response files and result fields before using evidence.
            solve = read_json(run / result["solve_attempt"] / "response.txt")
            if solve != result["solution"] or result["correct_answer"] != int(row["answer"]):
                raise RuntimeError("采集结果与原始回复或答案快照不一致。")
            for key in ("solve_attempt", "review_attempt"):
                if not result[key]:
                    continue
                folder = run / result[key]
                meta = read_json(folder / "meta.json")
                if meta["status"] != "completed" or digest((folder / "response.txt").read_bytes()) != meta["response_sha256"]:
                    raise RuntimeError("采集调用记录被修改。")
            review = read_json(run / result["review_attempt"] / "response.txt") if result["review_attempt"] else None
            if review != result["review"]:
                raise RuntimeError("复核记录与原始回复不一致。")
            expected = bool(solve["answer"] == int(row["answer"]) and review
                            and review["verdict"] == "accept" and review["methods"])
            if expected != result["accepted_for_distillation"]:
                raise RuntimeError("采集入选标记与原始证据不一致。")
            results.append(result)
    return results


def distill(run, client=None):
    run = Path(run)
    m = verify_run(run)
    results = training_results(run, m)
    selected = [r for r in results if r["accepted_for_distillation"]]
    if not selected:
        raise RuntimeError("没有答案正确且复核通过的记录，不能总结 skill。")
    evidence = [{"source_id": r["id"], "sample": r["sample"], "methods": r["review"]["methods"]}
                for r in selected]
    # Provenance IDs are stored locally; synthesis sees only generalized method cards.
    cards = [card for r in selected for card in r["review"]["methods"]]
    evidence_path = run / "distillation/evidence.json"
    immutable_json(evidence_path, evidence)
    evidence_digest = digest(evidence_path.read_bytes())
    freeze_path = run / "distillation/freeze.json"
    if freeze_path.exists():
        verify_freeze(run)
        if read_json(freeze_path)["evidence_sha256"] != evidence_digest:
            raise RuntimeError("冻结后训练依据发生变化。")
        return
    if any((run / "calls/evaluation").glob("**/meta.json")):
        raise RuntimeError("已有留出集评测记录，禁止重新总结 skill。")
    client = client or Codex(m["codex"])
    client.preflight()
    p = read_json(run / "snapshots/protocol.json")
    progress(run, "distilling", f"根据 {len(selected)}/{len(results)} 份合格记录总结 skill")
    skill, attempt = client.call(run / "calls/distill", "训练经验 / 总结 skill",
                                 p["distill"].format(evidence=canonical(cards)))
    if not skill.strip() or len(skill.split()) > 1800:
        write_json(attempt / "invalid.json", {"reason": "skill 为空或超过 1800 词"})
        raise RuntimeError("skill 长度无效；再次执行将新建总结尝试。")
    write_text(run / "distillation/draft.md", skill)
    progress(run, "auditing", "审核 skill 的适用条件、数学有效性和题目记忆风险")
    rows = read_json(run / "snapshots/train.json")
    audit_text, audit_attempt = client.call(run / "calls/audit", "训练经验 / 审核 skill",
        p["audit"].format(skill=skill, evidence=canonical(cards), problems=canonical([r["problem"] for r in rows])),
        p["audit_schema"])
    audit = parse_structured(audit_text, audit_attempt, p["audit_schema"])
    write_json(run / "distillation/audit.json", audit)
    if audit["verdict"] != "accept" or audit["issues"]:
        raise RuntimeError("skill 未通过审核；请查看 distillation/audit.json，留出集尚未调用。")
    write_text(run / "distillation/skill.md", skill)
    write_json(freeze_path, {"frozen_at": now(), "skill_sha256": digest(skill),
        "evidence_sha256": evidence_digest, "audit_sha256": digest(audit_text),
        "distill_attempt": attempt.relative_to(run).as_posix(),
        "audit_attempt": audit_attempt.relative_to(run).as_posix(),
        "selected_samples": len(selected), "selected_questions": len({str(r["id"]) for r in selected}),
        "training_results_sha256": digest(canonical(results))})
    progress(run, "frozen", "skill 已冻结，可以开始 AIME25 三组对照")
    report(run)


def verify_freeze(run):
    run = Path(run)
    freeze = read_json(run / "distillation/freeze.json")
    if digest((run / "distillation/skill.md").read_bytes()) != freeze["skill_sha256"]:
        raise RuntimeError("冻结 skill 被修改，拒绝评测。")
    if digest((run / "distillation/evidence.json").read_bytes()) != freeze["evidence_sha256"]:
        raise RuntimeError("冻结训练依据被修改，拒绝评测。")
    audit = run / freeze["audit_attempt"] / "response.txt"
    if digest(audit.read_bytes()) != freeze["audit_sha256"]:
        raise RuntimeError("冻结审核记录被修改。")
    judgment = read_json(audit)
    if judgment["verdict"] != "accept" or judgment["issues"]:
        raise RuntimeError("skill 审核未通过。")
    return freeze


def evaluate(run, client=None):
    run = Path(run)
    m = verify_run(run)
    freeze = verify_freeze(run)
    results = training_results(run, m)
    if digest(canonical(results)) != freeze["training_results_sha256"]:
        raise RuntimeError("冻结后训练评分发生变化。")
    p = read_json(run / "snapshots/protocol.json")
    rows = read_json(run / "snapshots/heldout.json")
    skills = {"none": "", "s0": (run / "snapshots/s0.md").read_text(encoding="utf-8"),
              "distilled": (run / "distillation/skill.md").read_text(encoding="utf-8")}
    client = client or DeepSeek(m["evaluation"])
    client.preflight()
    done = 0
    total = len(rows) * m["heldout"]["repeats"] * 3
    for question_index, row in enumerate(rows):
        for repeat in range(1, m["heldout"]["repeats"] + 1):
            # Rotate arm order across cells, keeping the three conditions interleaved.
            arms = list(skills)
            offset = (question_index + repeat - 1) % len(arms)
            for arm in arms[offset:] + arms[:offset]:
                done += 1
                label = f"AIME25 题目 {row['id']} / 重复 {repeat} / {arm}"
                progress(run, "evaluating", f"{done}/{total} {label}")
                section = "## Skill\n" + skills[arm].strip() + "\n" if skills[arm].strip() else ""
                messages = [{"role": "system", "content": p["eval_system"].format(skill_section=section)},
                            {"role": "user", "content": "## Problem\n" + row["problem"]}]
                stem = Path(arm) / task_name(row["id"]) / f"repeat_{repeat:02d}"
                text, attempt = client.call(run / "calls/evaluation" / stem, label, messages)
                meta = read_json(attempt / "meta.json")
                result = {"id": row["id"], "arm": arm, "repeat": repeat,
                          **score_answer(text, row["answer"], meta.get("finish_reason")),
                          "attempt": attempt.relative_to(run).as_posix()}
                immutable_json(run / "evaluation" / stem / "result.json", result)
                render_index(run)
                # Sequential pacing, no concurrent bursts or implicit provider retries.
                time.sleep(0.1)
    progress(run, "completed", "AIME25 三组对照已完成")
    report(run)


def report(run):
    run = Path(run)
    m = read_json(run / "manifest.json")
    train = [read_json(p) for p in run.glob("collection/**/result.json")]
    summary = {"generated_at": now(), "training": {
        "completed": len(train), "planned": m["training"]["questions"] * m["training"]["samples"],
        "answer_correct": sum(r["answer_correct"] for r in train),
        "accepted": sum(r["accepted_for_distillation"] for r in train)}, "evaluation": {}, "calls": {}}
    test = read_json(run / "snapshots/heldout.json")
    winners = {}
    planned = len(test) * m["heldout"]["repeats"]
    for arm in m["heldout"]["arms"]:
        cells = [read_json(p) for p in (run / "evaluation" / arm).glob("**/result.json")]
        counts = Counter(str(r["id"]) for r in cells)
        eligible = [r for r in test if counts[str(r["id"])] == m["heldout"]["repeats"]]
        winners[arm] = {}
        for row in eligible:
            votes = Counter(r["predicted_answer"] for r in cells if str(r["id"]) == str(row["id"]))
            most = votes.most_common()
            # Ties count as incorrect; never use the answer key to break a tie.
            winner = most[0][0] if len(most) == 1 or most[0][1] > most[1][1] else None
            winners[arm][str(row["id"])] = winner == int(row["answer"])
        correct = sum(r["correct"] for r in cells)
        strict_correct = sum(r.get("strict_correct", r["correct"] and r["answer_format_valid"]) for r in cells)
        summary["evaluation"][arm] = {
            "completed_samples": len(cells), "planned_samples": planned, "correct_samples": correct,
            "sample_accuracy": correct / len(cells) if cells else None,
            "strict_correct_samples": strict_correct,
            "strict_accuracy": strict_correct / len(cells) if cells else None,
            "complete_questions": len(eligible), "voted_correct": sum(winners[arm].values()),
            "voted_accuracy": sum(winners[arm].values()) / len(eligible) if eligible else None,
            "format_errors": sum(not r["answer_format_valid"] and not r.get("truncated", False) for r in cells),
            "truncated_samples": sum(r.get("truncated", False) for r in cells),
            "complete": len(cells) == planned}
    for p in run.glob("calls/**/meta.json"):
        meta = read_json(p)
        backend = p.relative_to(run / "calls").parts[0]
        group = summary["calls"].setdefault(backend, {"attempts": 0, "completed": 0,
                                                     "input_tokens": 0, "output_tokens": 0,
                                                     "attempts_without_usage": 0,
                                                     "reasoning_output_tokens": 0, "cached_input_tokens": 0})
        group["attempts"] += 1
        group["completed"] += meta["status"] == "completed"
        usage = meta.get("usage") or {}
        # Token totals include only observations actually supplied by the provider.
        # Interrupted calls with no usage must not be interpreted as zero-cost calls.
        group["attempts_without_usage"] += not bool(usage)
        group["input_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        group["output_tokens"] += usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        group["reasoning_output_tokens"] += usage.get("reasoning_output_tokens", 0) or 0
        group["cached_input_tokens"] += usage.get("cached_input_tokens", usage.get("prompt_cache_hit_tokens", 0)) or 0
    complete = all(s["complete"] for s in summary["evaluation"].values())
    summary["complete"] = complete
    if complete:
        summary["paired_changes"] = {}
        for base in ("none", "s0"):
            summary["paired_changes"][base] = {
                "improved_question_ids": [qid for qid in winners[base] if not winners[base][qid] and winners["distilled"][qid]],
                "regressed_question_ids": [qid for qid in winners[base] if winners[base][qid] and not winners["distilled"][qid]]}
    write_json(run / "summary.json", summary)
    def pct(value):
        return f"{value:.1%}" if value is not None else "—"
    accuracy_label = "数学答案正确率" if m.get("scoring_policy", {}).get("name") == SCORING_POLICY["name"] else "旧格式判据正确率"
    lines = ["# Codex 解题经验 → Skill → DeepSeek 对照", "",
             f"生成时间：{summary['generated_at']}", "",
             f"训练：AIME24，{summary['training']['completed']}/{summary['training']['planned']} 份记录完成，"
             f"答案正确 {summary['training']['answer_correct']}，答案正确且复核通过 {summary['training']['accepted']}。", "",
             f"老师：{m['codex']['model']} / {m['codex']['reasoning_effort']} / ChatGPT 登录。",
             f"最终评测：{m['evaluation']['model']} / {m['evaluation']['base_url']}；"
             f"temperature={m['evaluation']['temperature']}，max_tokens={m['evaluation']['max_tokens']}，"
             f"每题每组 {m['heldout']['repeats']} 次。", "",
             "状态：" + ("对照已完成。" if complete else "对照尚未完成，以下仅为进度，不能据此比较增益。"), "",
             f"| 配置 | 已处理次数 | {accuracy_label} | 答对且符合格式 | 完成题数 | 投票正确率 | 格式不合规 | 截断 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, s in summary["evaluation"].items():
        lines.append(f"| {arm} | {s['completed_samples']}/{planned} | {pct(s['sample_accuracy'])} | "
                     f"{pct(s['strict_accuracy'])} | {s['complete_questions']}/{len(test)} | {pct(s['voted_accuracy'])} | {s['format_errors']} | {s['truncated_samples']} |")
    lines += ["", "none = 不带 skill；s0 = 原始 skill；distilled = 本次冻结的 skill。",
              "数学答案评分读取最后一个明确的 boxed 答案或 answer 标签；不从正文任意找数字。"
              "缺少指定 XML 标签单列为格式不合规，不再把正确的 boxed 整数当作数学答错。", "",
              "投票平局或无有效答案按错误计。仅完整重复采样的题目进入投票统计。", "",
              "三组采用相同 token 上限。达到上限的样本单列为截断，作为预算内未完成作答计入总次数和正确率分母；"
              "不重复抽样直到完成，也不从截断内容中提取候选答案。网络异常仍保留日志后等待续跑。", "",
              "## 解释范围", "",
              "这是解题经验向 DeepSeek 的迁移实验，不能区分 Codex harness 与基础模型的贡献。",
              "独立复核仍由同一个模型完成，不能替代形式化证明或人工复核。公开题可能已出现在预训练数据中。",
              "小样本差异只能视为线索；不得根据留出集结果继续修改本次 skill 后宣称仍是留出验证。", "",
              "## 查阅记录", "", "打开 index.html 按阶段和题目查阅；失败、超时和重试均保留在 calls 下。",
              "原始事件是 CLI 公开记录，不保证包含完整内部思考。尚在 running 的记录可能是正在执行或进程被强行终止，须结合 status.json 判断。"]
    write_text(run / "report.md", "\n".join(lines) + "\n")
    render_index(run)
    return summary
