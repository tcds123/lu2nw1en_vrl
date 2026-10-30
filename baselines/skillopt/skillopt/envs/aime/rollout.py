"""AIME rollout — single-shot competition-mathematics reasoning.

Mirrors appsplus/rollout.py but for numerical-answer tasks: one chat_target
call per item, then exact-integer comparison via the AIME evaluator (no code
execution, no stdin/stdout). The skill document is the ONLY thing being
optimized by the SkillOpt loop; it is injected as a "## Skill" section in the
system prompt, exactly the way appsplus injects it — so any lift is
attributable to the skill text alone and NOT to an MCQ label shortcut
(AIME has no choices).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.envs.aime.evaluator import evaluate
from skillopt.model import chat_target
from skillopt.prompts import load_prompt


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    # NOTE: the AIME prompt contains literal LaTeX braces (``\boxed{N}``,
    # ``\boxed{204}``) which would be misread by str.format as fields. Use a
    # single-field substitution that only replaces the ``{skill_section}``
    # marker and leaves all other braces untouched.
    return load_prompt("rollout_system", env="aime").replace(
        "{skill_section}", skill_section
    )


def _build_user(item: dict) -> str:
    return (
        f"## Problem\n{item['question']}\n\n"
        "## Instruction\n"
        "Solve the problem carefully. Show your reasoning, then report the "
        "final integer answer (0-999) inside <answer>\\boxed{N}</answer> tags."
    )


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_turns: int = 1,
    max_completion_tokens: int = 4096,
    exec_timeout: int | None = None,
    **kwargs,
) -> dict:
    item_id = str(item["id"])
    result = {
        "id": item_id,
        "question": item["question"],
        "task_type": "aime",
        "hard": 0,
        "soft": 0.0,
        "pass_rate": -3.0,
        "predicted_answer": "",
        "predicted_int": None,
        "correct_answer": str(item.get("answer", "")),
        "response": "",
        "fail_reason": "",
        "error_type": "",
        "agent_ok": False,
        "n_turns": 0,
    }

    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)

        system = _build_system(skill_content)
        user = _build_user(item)

        response, _ = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="rollout",
            timeout=exec_timeout,
        )

        result["response"] = response
        result["agent_ok"] = True
        result["n_turns"] = 1

        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)

        eval_result = evaluate(response, item["answer"])
        result["hard"] = int(eval_result["em"])
        result["soft"] = float(eval_result["em"])
        result["pass_rate"] = float(eval_result["em"])
        result["predicted_answer"] = eval_result.get("predicted_answer", "")
        result["predicted_int"] = eval_result.get("predicted_int")
        if not result["hard"]:
            result["fail_reason"] = "wrong_integer"
            result["error_type"] = "WrongAnswer"

        eval_detail = (
            f"[EVALUATION RESULT]\n"
            f"correct_int: {eval_result.get('correct_int')}\n"
            f"predicted_int: {result['predicted_int']}\n"
            f"correct (exact integer match): {result['hard']}"
        )
        conversation = [
            {"type": "message", "turn": 1, "content": response},
            {"role": "system", "content": eval_detail},
        ]
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        return result

    except Exception as exc:  # noqa: BLE001
        result["fail_reason"] = f"rollout_exception: {exc!r}"[:500]
        result["error_type"] = "RolloutException"
        return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    *,
    max_turns: int = 1,
    exec_timeout: int = 300,
    workers: int = 8,
    max_completion_tokens: int = 4096,
    **kwargs,
) -> list[dict]:
    os.makedirs(out_root, exist_ok=True)
    results: list[dict] = []

    workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_item = {
            pool.submit(
                process_one,
                item,
                out_root,
                skill_content,
                max_turns=max_turns,
                max_completion_tokens=max_completion_tokens,
                exec_timeout=exec_timeout,
            ): item
            for item in items
        }
        for fut in as_completed(future_to_item):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                item = future_to_item[fut]
                results.append({
                    "id": str(item.get("id")),
                    "task_type": "aime",
                    "hard": 0, "soft": 0.0, "pass_rate": -3.0,
                    "predicted_answer": "", "predicted_int": None,
                    "correct_answer": str(item.get("answer", "")),
                    "response": "",
                    "fail_reason": f"future_exception: {exc!r}"[:500],
                    "error_type": "FutureException",
                    "agent_ok": False, "n_turns": 0,
                })

    results.sort(key=lambda r: str(r.get("id")))
    with open(os.path.join(out_root, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n = len(results)
    n_pass = sum(1 for r in results if r["hard"])
    avg_soft = sum(r["soft"] for r in results) / n if n else 0.0
    summary = {
        "n": n,
        "pass_rate_full": n_pass / n if n else 0.0,
        "avg_soft": avg_soft,
        "agent_ok": sum(1 for r in results if r.get("agent_ok")) / n if n else 0.0,
    }
    with open(os.path.join(out_root, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [aime] {n} items | exact-match={summary['pass_rate_full']:.4f} | avg_soft={summary['avg_soft']:.4f}")
    return results
