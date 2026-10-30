"""APPS-Plus evaluation helpers — deterministic stdin/stdout execution.

APPS-Plus is a competitive-programming benchmark: each problem has a set of
(stdin, expected_stdout) test pairs. The generated solution is a complete
program that reads from stdin via `input()` and writes to stdout via `print()`.

We run the generated code as a subprocess for each test case, feed the stdin
string on a pipe, capture stdout, and compare against the expected stdout
(after normalising trailing whitespace/newlines — the standard competitive-
programming convention). pass_rate = passed / total test cases.

This mirrors the sandboxing philosophy of mbpppro/evaluator.py (fresh temp dir,
subprocess isolation, hard timeout) but swaps assert-execution for stdin/stdout
comparison. No Docker.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import re
import tempfile


TIMEOUT_SECONDS = 6.0


def extract_code(text: str) -> str:
    """Extract a python code block from the model response."""
    if not text:
        return ""
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match_s = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match_s:
        return match_s.group(1).strip()
    return text.strip()


def _normalize_stdout(s: str) -> str:
    """Competitive-programming comparison: strip trailing whitespace per line
    and trailing newlines. This is the APPS / Codeforces convention."""
    return "\n".join(line.rstrip() for line in s.splitlines()).strip()


def _run_one(code_path: str, stdin: str, queue) -> None:
    import subprocess
    try:
        proc = subprocess.run(
            ["python3", code_path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        queue.put({"ok": True, "stdout": proc.stdout, "returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        queue.put({"ok": False, "status": "timeout"})
    except Exception as e:  # noqa: BLE001
        queue.put({"ok": False, "status": f"error: {e!r}"})


def _unsafe_exec_worker(code: str, test_pairs: list[tuple[str, str]], result_queue) -> None:
    """Write code to a temp file, run it once per stdin, compare stdout."""
    import traceback
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        code_path = os.path.join(temp_dir, "solution.py")
        try:
            os.chdir(temp_dir)
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

            total = len(test_pairs)
            if total == 0:
                result_queue.put({"status": "no_tests", "pass_rate": 0.0})
                return

            passed = 0
            first_failure = None
            for stdin, expected in test_pairs:
                q: multiprocessing.Queue = multiprocessing.Queue()
                p = multiprocessing.Process(target=_run_one, args=(code_path, stdin, q))
                p.start()
                p.join(timeout=TIMEOUT_SECONDS + 2.0)
                if p.is_alive():
                    p.terminate()
                    p.join()
                    if not first_failure:
                        first_failure = "Timeout"
                    continue
                if q.empty():
                    if not first_failure:
                        first_failure = "Worker returned nothing"
                    continue
                res = q.get()
                if not res.get("ok"):
                    if not first_failure:
                        first_failure = res.get("status", "runtime_error")
                    continue
                if res.get("returncode", 0) != 0:
                    if not first_failure:
                        first_failure = f"nonzero returncode={res.get('returncode')}"
                    continue
                got = _normalize_stdout(res.get("stdout", ""))
                want = _normalize_stdout(expected)
                if got == want:
                    passed += 1
                elif not first_failure:
                    first_failure = f"stdout mismatch:\n--- got ---\n{got[:400]}\n--- want ---\n{want[:400]}"

            result_queue.put({
                "status": "success",
                "pass_rate": passed / total,
                "failure_trace": first_failure,
            })
        except Exception:
            result_queue.put({"status": "runtime_error", "msg": traceback.format_exc()})
        finally:
            try:
                os.chdir(original_cwd)
            except Exception:
                pass


def parse_test_pairs(test_code: str) -> list[tuple[str, str]]:
    """The dataloader stores stdin/stdout pairs as a JSON blob in test_code."""
    if not test_code:
        return []
    try:
        obj = json.loads(test_code)
    except json.JSONDecodeError:
        return []
    stdins = obj.get("stdin") or []
    stdouts = obj.get("stdout") or []
    n = min(len(stdins), len(stdouts))
    return [(str(stdins[i]), str(stdouts[i])) for i in range(n)]


def evaluate(prediction_text: str, test_code: str) -> dict:
    """Run the generated program against the (stdin, stdout) test pairs.

    Returns a dict compatible with SkillOpt result schema:
      hard: 1 if all test pairs pass (pass_rate == 1.0), else 0
      soft: pass_rate in [0.0, 1.0] (0 if compile/error)
      pass_rate, fail_reason, error_type, code
    """
    code = extract_code(prediction_text)
    if not code:
        return {
            "hard": 0, "soft": 0.0, "pass_rate": -3.0,
            "fail_reason": "EmptyCode: extraction failed", "error_type": "EmptyCode",
            "code": "",
        }

    test_pairs = parse_test_pairs(test_code)
    if not test_pairs:
        return {
            "hard": 0, "soft": 0.0, "pass_rate": -3.0,
            "fail_reason": "NoTests: test_code is empty", "error_type": "NoTests",
            "code": code,
        }

    queue: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_unsafe_exec_worker, args=(code, test_pairs, queue))
    p.start()
    p.join(timeout=TIMEOUT_SECONDS * len(test_pairs) + 30)

    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "hard": 0, "soft": 0.0, "pass_rate": -1.0,
            "fail_reason": f"Timeout: exceeded {TIMEOUT_SECONDS}s/test", "error_type": "Timeout",
            "code": code,
        }

    if not queue.empty():
        res = queue.get()
        status = res.get("status", "unknown")
        if status == "success":
            pr = float(res.get("pass_rate", 0.0))
            hard = 1 if pr >= 1.0 else 0
            fail_reason = "" if hard else (res.get("failure_trace") or "Some tests failed")
            return {
                "hard": hard, "soft": pr, "pass_rate": pr,
                "fail_reason": fail_reason[:500], "error_type": "" if hard else "PartialFailure",
                "code": code,
            }
        return {
            "hard": 0, "soft": 0.0, "pass_rate": 0.0 if status == "no_tests" else -3.0,
            "fail_reason": f"{status}: {str(res.get('msg') or res.get('failure_trace') or '')[:400]}",
            "error_type": status, "code": code,
        }

    return {
        "hard": 0, "soft": 0.0, "pass_rate": -3.0,
        "fail_reason": "Unknown: worker returned nothing", "error_type": "Unknown",
        "code": code,
    }
