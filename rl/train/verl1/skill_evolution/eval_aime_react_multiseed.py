#!/usr/bin/env python3
"""AIME ReAct evaluation (single-call vs ReAct+execute_code) — the agent-capability test.

Question: does upgrading AIME eval from single-shot generation to a ReAct loop with a
python_executor (self-debug) raise pass-rate? This is the core experiment validating
that the BAgent agent capability (which exists in agent/b_agent/core.py but is never
enabled in eval) actually helps when turned on.

Fairness contract (aligned with eval_bagent_react_multiseed.py):
  - same skill (s0 / v2 / evolver / skillopt), same model (DeepSeek-V3.2), same rate-limit.
  - baseline = single-call: reuse eval_aime24_multiseed cached per_{name}_{hash}_seed{N}.json
    from single_dir; not re-run.
  - treatment = ReAct: system prompt = AIME _build_system(skill) + REACT_SUFFIX (tool spec);
    multi-turn chat_target_messages loop; model may call execute_code to verify; the last
    tool-free reply is the final answer. eval_k, N-seed, identical to baseline.
  - ONLY difference = REACT_SUFFIX + tool loop. skill / model / rate-limit / eval identical.

AIME adaptation vs eval_bagent_react_multiseed.py:
  - data: AIME integer items (data/aime24.jsonl or aime25.jsonl), item={id,problem,answer}
    not livemath choice items (item={id,problem,choices,correct_choice})
  - system: AIME _build_system (wants boxed N) not livemath _build_system (wants choice label)
  - scoring: evaluate_aime (int compare via extract_int_answer) not math_evaluate (choice em)
  - execute_code / parse_tool_calls / ReAct loop skeleton: copied from eval_bagent_react_multiseed.py
    (verified working, guarantees consistency with LiveMath ReAct protocol)

execute_code tool spec adapted for integer problems: sympy verification, small-case
enumeration, concrete-value substitution.

Output: out_dir/react_eval.json
Usage:
  # AIME24 (has single-call cache for baseline comparison)
  python eval_aime_react_multiseed.py --skills s0,evolver --data data/aime24.jsonl \
      --eval_k 3 --seeds 4 --max_steps 6 --out_dir .../aime24_react --single_dir .../multiseed_test
  # AIME25 held-out
  python eval_aime_react_multiseed.py --skills s0,evolver --data data/aime25.jsonl \
      --eval_k 3 --seeds 3 --max_steps 6 --out_dir .../aime25_react
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tf_global_v2_aime import set_temp, EVAL_TEMP, md5  # noqa: E402  (AIME backend)
from eval_aime24_multiseed import (  # noqa: E402
    _build_system, _build_user, evaluate_aime, load_aime,
)
from skillopt.model import chat_target_messages  # noqa: E402


# ================================================================
# execute_code -- copied from agent/b_agent/core.py BAgent._execute_python_code
# (sandboxed subprocess, 10s timeout). Consistent with eval_bagent_react_multiseed.
# ================================================================
def execute_code(code):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python3", tmp_path], capture_output=True, text=True, timeout=10,
        )
        output = result.stdout
        if result.stderr:
            output = output + "\n[stderr]:\n" + result.stderr
        if result.returncode != 0:
            output = output + "\n[Exit code: " + str(result.returncode) + "]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[Timeout] Execution exceeded 10 seconds"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ================================================================
# Tool-call parsing -- copied from eval_bagent_react_multiseed.py (XML-tag format).
# Tag literals are built from chr() so the source contains no literal delimiter
# sequence (avoids editor/harness parsing issues).
# ================================================================
_TOOL_OPEN = chr(60) + "tool_call" + chr(62)
_TOOL_CLOSE = chr(60) + "/" + "tool_call" + chr(62)
_TOOL_CALL_RE = re.compile(
    re.escape(_TOOL_OPEN) + "(.*?)" + re.escape(_TOOL_CLOSE), re.DOTALL
)


def parse_tool_calls(response):
    calls = []
    for m in _TOOL_CALL_RE.finditer(response):
        try:
            data = json.loads(m.group(1))
            name = data.get("name", "")
            args = data.get("arguments", data.get("args", {}))
            if name:
                calls.append((name, args))
        except json.JSONDecodeError:
            continue
    return calls


# Example of a tool call, shown to the model. Built from the chr() tags so no
# literal delimiter appears in source.
_CALL_EX = (
    _TOOL_OPEN
    + '{"name": "execute_code", "arguments": {"code": "<your complete self-contained python code>"}}'
    + _TOOL_CLOSE
)

# answer-tag literals (built from chr() for the same reason)
_ANS_OPEN = chr(60) + "answer" + chr(62)
_ANS_CLOSE = chr(60) + "/" + "answer" + chr(62)
_BS = chr(92)  # backslash, so "\boxed{N}" in source has no literal backslash escaping pitfalls

REACT_SUFFIX = (
    "\n\n## Tool: execute_code\n"
    "You have access to a Python sandbox (execute_code). Use it to VERIFY your mathematical "
    "reasoning before committing to an answer:\n"
    "- Plug in CONCRETE values to test your formula (try edge cases: 0, 1, boundaries, max/min).\n"
    "- Use sympy for symbolic checks (algebra, limits, derivatives, integrals, matrix "
    "determinant/rank, polynomial roots, modular arithmetic).\n"
    "- Enumerate small cases by brute force to detect or refute a pattern, or to cross-check a "
    "closed form.\n"
    "- For counting problems, write a direct brute-force enumeration over all configurations and "
    "compare to your analytical count.\n"
    "The sandbox runs Python 3 with a 10-second timeout; stdout+stderr are returned to you.\n\n"
    "To call the tool, output EXACTLY (and only) on its own line:\n"
    + _CALL_EX
    + "\n\n## Procedure\n"
    "1. Reason about the problem; identify what to compute and how to verify it.\n"
    "2. If verification helps, call execute_code with complete, self-contained Python code "
    "(include all imports).\n"
    "3. Read the tool result, then continue. You may call execute_code again (up to <<MAX_STEPS>> "
    "times total).\n"
    "4. When confident, output your FINAL ANSWER inside "
    + _ANS_OPEN
    + "..."
    + _ANS_CLOSE
    + " tags, containing the integer in "
    + _BS
    + "boxed{N} form (e.g. "
    + _ANS_OPEN
    + _BS
    + "boxed{204}"
    + _ANS_CLOSE
    + "), with NO tool call.\n"
)


def _build_react_system(skill_content, max_steps):
    return _build_system(skill_content) + REACT_SUFFIX.replace("<<MAX_STEPS>>", str(max_steps))


# ================================================================
# ReAct single-problem eval: k samples, each a full ReAct trajectory.
# Returns a dict compatible with run_one_aime (pass_rate / agent_ok / predicted_answer).
# ================================================================
def run_one_aime_react(item, skill_content, k=1, max_steps=6, stage="aime_react"):
    """Evaluate one problem with skill + ReAct loop, k times. pass_rate = n_pass / k."""
    qid = str(item["id"])
    res = {
        "id": qid, "pass_rate": -3.0, "agent_ok": False, "fail_reason": "",
        "response": "", "predicted_answer": None, "correct_answer": item["answer"],
        "n_pass": 0, "n_samples": 0, "n_tool_calls": 0,
    }
    system = _build_react_system(skill_content, max_steps)
    user = _build_user(item)
    n_pass = 0
    last_ev = None
    last_resp = ""
    total_tools = 0
    for _ in range(max(1, k)):
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            final = ""
            resp = ""
            n_tools_this = 0
            for step in range(max_steps):
                resp, _ = chat_target_messages(
                    messages, max_completion_tokens=4096, retries=5,
                    stage=stage, timeout=300,
                )
                tcs = parse_tool_calls(resp)
                if not tcs:
                    final = resp
                    break
                # execute tools, feed results back
                messages.append({"role": "assistant", "content": resp})
                for name, args in tcs:
                    n_tools_this += 1
                    if name == "execute_code":
                        code = args.get("code", "") if isinstance(args, dict) else ""
                        result = execute_code(code)
                    else:
                        result = "Unknown tool: " + name
                    messages.append(
                        {
                            "role": "user",
                            "content": "Tool result for " + name + ":\n" + result,
                        }
                    )
            else:
                # exhausted max_steps without a tool-free reply: nudge for a final answer
                final = resp if resp else ""
                if parse_tool_calls(final) or not final.strip():
                    messages.append(
                        {
                            "role": "user",
                            "content": "Based on the above reasoning and tool results, "
                            "output your FINAL ANSWER inside "
                            + _ANS_OPEN
                            + "..."
                            + _ANS_CLOSE
                            + " tags containing the integer in "
                            + _BS
                            + "boxed{N} form. Do NOT call any tool.",
                        }
                    )
                    final, _ = chat_target_messages(
                        messages, max_completion_tokens=512, retries=5,
                        stage=stage, timeout=300,
                    )
            total_tools += n_tools_this
            ev = evaluate_aime(final, item["answer"])
            last_ev, last_resp = ev, final
            if ev["em"]:
                n_pass += 1
        except Exception as e:
            res["fail_reason"] = "exception: " + repr(e)
    if last_ev is not None:
        res.update(
            {
                "pass_rate": n_pass / max(1, k),
                "agent_ok": True,
                "response": last_resp,
                "predicted_answer": last_ev.get("predicted_answer"),
                "fail_reason": "" if n_pass == 0 and k == 1 else res.get("fail_reason", ""),
            }
        )
    res["n_pass"], res["n_samples"] = n_pass, max(1, k)
    res["n_tool_calls"] = total_tools
    return res


def eval_react_on(items, skill_content, k, seeds, max_steps, max_workers, stage="aime_react"):
    """Return list of {seed, mean_pass, per_qid, n_tool_calls}. Parallel within each seed."""
    runs = []
    for seed in range(1, seeds + 1):
        t0 = time.time()
        per_qid = {}
        n_tools_total = 0

        def _work(item):
            return str(item["id"]), run_one_aime_react(
                item, skill_content, k=k, max_steps=max_steps, stage=stage
            )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_work, it) for it in items]
            for fut in as_completed(futs):
                qid, r = fut.result()
                per_qid[qid] = r["pass_rate"] if r["pass_rate"] >= 0 else 0.0
                n_tools_total += r.get("n_tool_calls", 0)
        mean_pass = sum(per_qid.values()) / max(1, len(per_qid))
        dt = time.time() - t0
        print(
            "  [react seed " + str(seed) + "/" + str(seeds) + "] mean_pass="
            + ("%.4f" % mean_pass)
            + " tools=" + str(n_tools_total)
            + " (%.1f" % (n_tools_total / max(1, len(items))) + "/q) "
            + ("%.0f" % dt) + "s",
            flush=True,
        )
        runs.append(
            {"seed": seed, "mean_pass": mean_pass, "per_qid": per_qid, "n_tool_calls": n_tools_total}
        )
    return runs


def load_single_cache(skill_name, skill_hash, seeds, single_dir):
    """Read single-call baseline (per_{name}_{hash}_seed{N}.json) from single_dir."""
    runs = []
    for seed in range(1, seeds + 1):
        p = os.path.join(
            single_dir, "per_" + skill_name + "_" + skill_hash + "_seed" + str(seed) + ".json"
        )
        if os.path.exists(p):
            per = json.load(open(p, encoding="utf-8"))
            mean_pass = sum(per.values()) / max(1, len(per))
            runs.append({"seed": seed, "mean_pass": mean_pass, "per_qid": per})
    return runs


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def paired_t_test(x_qid, y_qid):
    qids = sorted(set(x_qid) & set(y_qid))
    if len(qids) < 2:
        return None
    diffs = [_mean(x_qid[q]) - _mean(y_qid[q]) for q in qids]
    n = len(diffs)
    m = _mean(diffs)
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else 0.0
    df = n - 1

    def norm_cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    p = 2 * (1 - norm_cdf(abs(t)))
    if df < 30:
        p = min(1.0, p * (1 + 0.5 * (30 - df) / 30))
    return {"mean_diff": m, "t": t, "df": df, "p": p, "n": n}


def collect_per_qid(runs):
    qid = defaultdict(list)
    for r in runs:
        for q, pr in r.get("per_qid", {}).items():
            qid[q].append(pr)
    return qid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/zhuldz/lunwen/data/aime24.jsonl",
                    help="AIME jsonl (aime24 or aime25)")
    ap.add_argument("--skills", default="s0,evolver", help="comma-separated skill names")
    ap.add_argument("--skill_files", default="",
                    help="comma-separated paths aligned with --skills; empty = built-in defaults")
    ap.add_argument("--s0", default="/data/zhuldz/lunwen/data/skill_aime_initial_v2.md")
    ap.add_argument(
        "--v2_best",
        default="/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/run_LATEST/tf_global_skill_v2_aime.md",
    )
    ap.add_argument(
        "--evolver_best",
        default="/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime_v2/run_20261009_213157/best_skill_evolved.md",
    )
    ap.add_argument(
        "--skillopt_best",
        default="/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/skillopt/aime-skillopt-skel-20261010-095515/train/best_skill.md",
    )
    ap.add_argument("--eval_k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=6, help="max ReAct tool-call rounds")
    ap.add_argument("--max_workers", type=int, default=14)
    ap.add_argument(
        "--out_dir",
        default="/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/aime24_react",
    )
    ap.add_argument("--single_dir", default="",
                    help="single-call baseline cache dir; empty = report react only")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_temp(EVAL_TEMP)
    print(
        "AIME ReAct eval | EVAL_TEMP=" + str(EVAL_TEMP)
        + " k=" + str(args.eval_k) + " seeds=" + str(args.seeds)
        + " max_steps=" + str(args.max_steps) + " workers=" + str(args.max_workers)
    )

    test_items = load_aime(args.data)
    print("AIME items: " + str(len(test_items)) + "  data=" + args.data)

    skill_paths = {
        "s0": args.s0, "v2": args.v2_best,
        "evolver": args.evolver_best, "skillopt": args.skillopt_best,
    }
    if args.skill_files:
        names = [s.strip() for s in args.skills.split(",")]
        paths = [p.strip() for p in args.skill_files.split(",")]
        for nm, pt in zip(names, paths):
            skill_paths[nm] = pt

    skills = {}
    for name in args.skills.split(","):
        name = name.strip()
        p = skill_paths.get(name)
        if p and os.path.exists(p):
            txt = open(p, encoding="utf-8").read().strip()
            skills[name] = txt
            print("  " + name + ": " + str(len(txt)) + " chars (hash " + md5(txt)[:8] + ")")
        else:
            print("  WARN " + name + ": path not found " + str(p))

    # ---- run ReAct (treatment), per skill per seed cached separately ----
    results = {name: [] for name in skills}
    for name, txt in skills.items():
        h = md5(txt)[:8]
        print("\n" + "=" * 50 + "\n> ReAct eval: " + name + " (hash " + h + ")\n" + "=" * 50, flush=True)
        for seed in range(1, args.seeds + 1):
            cache = os.path.join(
                args.out_dir, "per_" + name + "_" + h + "_react_seed" + str(seed) + ".json"
            )
            if os.path.exists(cache):
                per = json.load(open(cache, encoding="utf-8"))
                mean_pass = sum(per.values()) / max(1, len(per))
                results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per})
                print(
                    "  [react seed " + str(seed) + "/" + str(args.seeds)
                    + "] CACHED mean_pass=" + "%.4f" % mean_pass, flush=True
                )
                continue
            t0 = time.time()
            per_qid = {}
            n_tools_total = 0

            def _work(item):
                return str(item["id"]), run_one_aime_react(
                    item, txt, k=args.eval_k, max_steps=args.max_steps
                )

            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futs = [ex.submit(_work, it) for it in test_items]
                for fut in as_completed(futs):
                    qid, r = fut.result()
                    per_qid[qid] = r["pass_rate"] if r["pass_rate"] >= 0 else 0.0
                    n_tools_total += r.get("n_tool_calls", 0)
            mean_pass = sum(per_qid.values()) / max(1, len(per_qid))
            json.dump(per_qid, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
            print(
                "  [react seed " + str(seed) + "/" + str(args.seeds) + "] mean_pass="
                + ("%.4f" % mean_pass) + " tools=" + str(n_tools_total)
                + " (%.1f" % (n_tools_total / max(1, len(test_items))) + "/q) "
                + ("%.0f" % (time.time() - t0)) + "s -> " + cache, flush=True
            )
            results[name].append(
                {"seed": seed, "mean_pass": mean_pass, "per_qid": per_qid, "n_tool_calls": n_tools_total}
            )

    # ---- load single-call baseline (optional) ----
    single_runs = {}
    if args.single_dir:
        for name, txt in skills.items():
            h = md5(txt)[:8]
            sr = load_single_cache(name, h, args.seeds, args.single_dir)
            if not sr:
                import glob
                for p in glob.glob(os.path.join(args.single_dir, "per_*_" + h + "_seed*.json")):
                    seed = int(re.search(r"seed(\d+)", p).group(1))
                    if seed <= args.seeds:
                        per = json.load(open(p, encoding="utf-8"))
                        sr.append(
                            {"seed": seed, "mean_pass": sum(per.values()) / max(1, len(per)),
                             "per_qid": per}
                        )
                sr.sort(key=lambda r: r["seed"])
            single_runs[name] = sr
            if sr:
                print(
                    "  single-call " + name + ": " + str(len(sr))
                    + " seed cached, mean=" + "%.4f" % _mean([r["mean_pass"] for r in sr])
                )
            else:
                print("  WARN single-call " + name + ": no cache, no baseline")

    # ---- summarize + compare ----
    summary = {}
    comparison = {}
    for name in skills:
        r_runs = results[name]
        s_runs = single_runs.get(name, [])
        r_pass = _mean([r["mean_pass"] for r in r_runs]) if r_runs else 0.0
        s_pass = _mean([r["mean_pass"] for r in s_runs]) if s_runs else 0.0
        summary[name + "_react"] = {
            "mean_pass": r_pass, "n_seeds": len(r_runs),
            "delta_vs_single": r_pass - s_pass if s_runs else None,
        }
        if s_runs:
            summary[name + "_single"] = {"mean_pass": s_pass, "n_seeds": len(s_runs)}
            tt = paired_t_test(collect_per_qid(r_runs), collect_per_qid(s_runs))
            comparison[name] = {
                "react_pass": r_pass, "single_pass": s_pass,
                "delta": r_pass - s_pass,
                "t": tt["t"] if tt else None, "p": tt["p"] if tt else None,
                "n": tt["n"] if tt else None,
            }
        else:
            summary[name + "_react"]["note"] = "no single-call baseline"

    out = {
        "args": vars(args), "n_test": len(test_items), "eval_k": args.eval_k,
        "seeds": args.seeds, "max_steps": args.max_steps,
        "summary": summary, "comparison": comparison,
        "runs_react": results, "runs_single": single_runs,
    }
    out_path = os.path.join(args.out_dir, "react_eval.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(
        "\n" + "=" * 60 + "\nAIME ReAct result (test=" + str(len(test_items))
        + ", k=" + str(args.eval_k) + ", seeds=" + str(args.seeds) + ")\n" + "=" * 60
    )
    if comparison:
        print("  skill        single     react        D        t        p")
        print("-" * 62)
        for name in skills:
            c = comparison.get(name, {})
            if c:
                sig = "SIG" if c["p"] < 0.05 else ("marg" if c["p"] < 0.1 else "ns")
                print(
                    "  " + name.ljust(10)
                    + " %10.4f %10.4f %10.4f %8.2f %8.4f %s"
                    % (c["single_pass"], c["react_pass"], c["delta"], c["t"], c["p"], sig)
                )
    else:
        for name in skills:
            r_pass = summary.get(name + "_react", {}).get("mean_pass", 0.0)
            print("  " + name + ": react mean_pass=" + "%.4f" % r_pass)
    print("\nDone -> " + out_path)


if __name__ == "__main__":
    main()
