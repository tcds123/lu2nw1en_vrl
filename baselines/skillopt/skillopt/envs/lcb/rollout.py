"""APPS-Plus rollout — single-shot competitive-programming code generation.

Same shape as mbpppro/rollout.py: build system prompt (skill section + task
format), one chat_target call per item (max_turns=1), extract code, run the
deterministic stdin/stdout evaluator.

The skill document is the ONLY thing being optimized by the SkillOpt loop;
here it is injected as a "## Skill" section in the system prompt, exactly the
way mbpppro injects it — so any lift is attributable to the skill text alone.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.envs.lcb.evaluator import evaluate
from skillopt.model import chat_target
from skillopt.prompts import load_prompt


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="appsplus").format(skill_section=skill_section)


def _build_user(item: dict) -> str:
    return (
        f"## Problem\n{item['question']}\n\n"
        "## Instruction\n"
        "Write a complete, self-contained Python program that reads from stdin "
        "and writes to stdout. Solve the problem for arbitrary inputs that satisfy "
        "the constraints. Do NOT read from any file. Do NOT use a main guard is "
        "optional, but the program must run top-to-bottom. Output the solution "
        "inside a single ```python ... ``` code block."
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
        "task_type": "apps_plus",
        "hard": 0,
        "soft": 0.0,
        "pass_rate": -3.0,
        "predicted_answer": "",
        "response": "",
        "code": "",
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

        eval_result = evaluate(response, item["test_code"])
        result["hard"] = int(eval_result["hard"])
        result["soft"] = float(eval_result["soft"])
        result["pass_rate"] = float(eval_result["pass_rate"])
        result["code"] = eval_result.get("code", "")
        result["fail_reason"] = eval_result.get("fail_reason", "")
        result["error_type"] = eval_result.get("error_type", "")
        result["predicted_answer"] = "PASS" if result["hard"] else "FAIL"

        eval_detail = (
            f"[EVALUATION RESULT]\n"
            f"pass_rate: {result['pass_rate']}\n"
            f"hard (all pass): {result['hard']}\n"
            f"error_type: {result['error_type']}\n"
            f"fail_reason: {result['fail_reason'][:300]}"
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
                    "task_type": "apps_plus",
                    "hard": 0, "soft": 0.0, "pass_rate": -3.0,
                    "response": "", "code": "",
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
    print(f"  [appsplus] {n} items | full-pass={summary['pass_rate_full']:.4f} | avg_soft={summary['avg_soft']:.4f}")
    return results
