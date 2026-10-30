"""MBPP-Pro evaluation helpers — deterministic unit-test execution.

Faithfully reuses the sandboxed exec worker from the lunwen project's
`rl/train/verl1/custom_reward.py` (`_unsafe_exec_worker`): the generated code
is exec'd in a fresh temp dir, function-name mismatch is auto-bridged, and each
assert line is run in the same scope. pass_rate = passed / total asserts.

This gives the SkillOpt loop a deterministic, Docker-free verifier on code data
that is directly comparable to our historical ablation pass rates.
"""
from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import re
import tempfile
import traceback


TIMEOUT_SECONDS = 4.0


def _unsafe_exec_worker(code: str, unit_tests: list[str], result_queue) -> None:
    """Exec code in a temp dir, run each assert, report pass_rate.

    Ported verbatim (behavior preserved) from custom_reward._unsafe_exec_worker.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                scope: dict = {}
                try:
                    exec(code, scope)
                except Exception:
                    result_queue.put({"status": "compile_error", "msg": traceback.format_exc()})
                    return

                def get_func_name(s: str) -> str | None:
                    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", s)
                    return m.group(1) if m else None

                gen_name = get_func_name(code)
                test_name = None
                if unit_tests:
                    m = re.search(r"(?:assert|print)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", unit_tests[0])
                    test_name = m.group(1) if m else None

                if gen_name and test_name and gen_name != test_name:
                    if gen_name in scope:
                        scope[test_name] = scope[gen_name]

                passed = 0
                total = len(unit_tests)
                if total == 0:
                    result_queue.put({"status": "no_tests", "pass_rate": 0.0})
                    return

                first_failure = None
                for test in unit_tests:
                    try:
                        exec(test.strip(), scope)
                        passed += 1
                    except Exception:
                        if not first_failure:
                            first_failure = traceback.format_exc()

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


def parse_unit_tests(test_code: str) -> list[str]:
    """Split a test_code string into individual assert lines."""
    if not test_code:
        return []
    lines = [ln.strip() for ln in str(test_code).splitlines() if ln.strip()]
    return lines


def evaluate(prediction_text: str, test_code: str) -> dict:
    """Run the generated code against the unit tests.

    Returns a dict compatible with SkillOpt result schema:
      hard: 1 if all tests pass (pass_rate == 1.0), else 0
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

    unit_tests = parse_unit_tests(test_code)
    if not unit_tests:
        return {
            "hard": 0, "soft": 0.0, "pass_rate": -3.0,
            "fail_reason": "NoTests: test_code is empty", "error_type": "NoTests",
            "code": code,
        }

    queue: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_unsafe_exec_worker, args=(code, unit_tests, queue))
    p.start()
    p.join(timeout=TIMEOUT_SECONDS)

    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "hard": 0, "soft": 0.0, "pass_rate": -1.0,
            "fail_reason": f"Timeout: exceeded {TIMEOUT_SECONDS}s", "error_type": "Timeout",
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
