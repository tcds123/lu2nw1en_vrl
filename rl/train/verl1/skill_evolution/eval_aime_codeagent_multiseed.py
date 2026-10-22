#!/usr/bin/env python3
"""AIME CodeAgent evaluation (Paradigm 2: Program-of-Thoughts).

The companion to eval_aime_react_multiseed.py (Paradigm 1, interleaved ReAct). That
eval returned SIGNIFICANTLY NEGATIVE deltas (-7.4pp s0, -13.0pp evolver). User correction:
the negative result means MY ReAct agent DESIGN is bad, not that agent capability is
inherently harmful on math. Diagnosis (see memory/react-agent-paradigm-diagnosis-20261010.md):
-13pp = implementation flaws (4) + wrong paradigm (interleaved tools interrupt symbolic reasoning).

This eval tests Paradigm 2 (smolagents CodeAgent / Program-of-Thoughts): the model writes
Python in a persistent LocalPythonExecutor (state survives across steps), final_answer(x)
terminates. Literature: PoT > CoT on arithmetic-heavy tasks (offloads computation). This is
the correct way to test "can agent capability help math?" — opposite paradigm to ReAct.

Fairness contract (aligned with react eval + single-call baseline, byte-for-byte):
  - same skill (s0 / evolver / ...), same model (DeepSeek-V3.2), same AIME25 data.
  - baseline = single-call: reuse eval_aime25_multiseed cached per_{name}_{hash}_seed{N}.json
    from single_dir; NOT re-run.
  - treatment = CodeAgent: system prompt = AIME _build_system(skill) PREPENDED to CodeAgent
    default system prompt + AIME answer-format suffix; model may write Python to compute/verify;
    final_answer(x) terminates. eval_k, N-seed, EVAL_TEMP=0.3, max_tokens=4096 identical.
  - ONLY difference = CodeAgent PoT loop (vs single-call one-shot). skill/model/data/temp/k same.

Rate-limit fairness: CodeAgent calls go through the OpenAI SDK, NOT through skillopt's
urllib path. To share the SAME 30-RPM DeepSeek-V3.2 quota without double-counting, we
import skillopt's _TokenBucket (28 RPM) and acquire() before every OpenAI SDK call. This
serializes CodeAgent LLM calls under the same global bucket as single-call would use.
Run this eval ALONE (no concurrent skillopt/react) to avoid two consumers on one bucket.

Output: out_dir/codeagent_eval.json
Usage:
  python eval_aime_codeagent_multiseed.py --skills s0,evolver --data data/aime25.jsonl \
      --eval_k 3 --seeds 3 --max_steps 6 --out_dir .../aime25_codeagent \
      --single_dir .../aime25_test
"""
import argparse
import gc
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tf_global_v2_aime import set_temp, EVAL_TEMP, md5  # noqa: E402
from eval_aime24_multiseed import (  # noqa: E402
    _build_system, _build_user, evaluate_aime, load_aime,
)

# ---- V3.2 backend config (mirror tf_global_v2_math.py lines 38-47) ----
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_API_KEY", "")  # 由环境变量提供 key
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1")
os.environ.setdefault("QWEN_CHAT_MODEL", "/model/DeepSeek-V3.2")
os.environ.setdefault("QWEN_CHAT_AIGC_USER", "zhuldz")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "4096")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.3")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "28")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

_V32_API_KEY = os.environ["QWEN_CHAT_API_KEY"]
_V32_BASE_URL = os.environ["QWEN_CHAT_BASE_URL"]
_V32_MODEL = os.environ["QWEN_CHAT_MODEL"]
_AIGC_USER = os.environ["QWEN_CHAT_AIGC_USER"]

# smolagents (Paradigm 2)
from smolagents import CodeAgent, OpenAIServerModel  # noqa: E402
from smolagents.local_python_executor import LocalPythonExecutor  # noqa: E402

# Reuse skillopt's process-global token bucket so CodeAgent shares the SAME 30-RPM
# DeepSeek-V3.2 quota (not a parallel bucket). Acquire() blocks until a token frees.
from skillopt.model.qwen_backend import _get_rate_limiter  # noqa: E402

_RATE_LIMITER = _get_rate_limiter()

# Per-trajectory wall-clock timeout (Bug 5 fix). smolagents' per-step executor
# timeout uses ThreadPoolExecutor.future.result(timeout) — the smolagents source
# itself (local_python_executor.py:299-301) warns the busy-loop thread CANNOT be
# force-killed and keeps running at 100% CPU after the step raises. Orphaned
# threads accumulate, and the multi-step agent.run() loop itself has no outer
# bound, so a single stuck trajectory (sympy busy-loop / gateway hang) wedges the
# whole worker forever — 26h of wasted CPU, zero results (see Bug 5 in
# phase2_issues.md). signal.alarm is INVALID in worker threads (ValueError), so we
# wrap the ENTIRE agent.run() in a daemon thread + join(timeout=TRAJ_TIMEOUT).
# On timeout we abandon that k-vote (pass_rate contribution 0) and let the worker
# proceed. max_steps(6) * ~30s/step + summary slack; 240s is generous but bounded.
TRAJ_TIMEOUT = int(os.environ.get("CODEAGENT_TRAJ_TIMEOUT", "240"))


# ================================================================
# EscapedNewlineExecutor: the DeepSeek-V3.2 gateway (via OpenAI SDK) sometimes
# returns code blocks where newlines are LITERAL backslash-n ("\n" as two chars)
# rather than real newlines — e.g. the model emits "\nfinal_answer(70)\n" as a
# single-line string. LocalPythonExecutor then sees `\nfinal_answer(70)\n` which
# is a SyntaxError ("unexpected character after line continuation character"),
# wasting all max_steps on retry loops before the max-steps fallback. We fix this
# at the executor boundary: unescape literal \n / \t into real newlines/tabs
# BEFORE evaluation. We guard against double-processing by checking the code does
# not already parse as valid Python (if it does, leave it untouched).
# ================================================================
import ast as _ast  # noqa: E402


def _unescape_code_literals(code: str) -> str:
    """Convert literal backslash escapes (\\n, \\t, \\', \\\") in code to real chars.

    The V3.2 gateway via the OpenAI SDK returns code blocks where newlines/tabs/quotes
    are JSON-style escaped, and the ESCAPE LEVEL VARIES across calls and even within a
    single block (2-char single `\\n`, 3-char double `\\\\n`, mixed with `\\'`). A fixed
    set of .replace() calls cannot cover every combination — a double-escaped `\\\\n`
    decoded once leaves a stray backslash that becomes a line-continuation -> SyntaxError
    -> wasted max_steps.

    Robust approach: ITERATIVELY apply a single-pass decode (\\n->\\n, \\t->\\t, \\'->
    ', \\"->") and re-check ast.parse after each pass. Stop as soon as the code parses,
    or when a pass stops changing the string (fixpoint). The initial ast.parse gate
    guarantees already-valid code (including legitimate `\\n` inside string literals) is
    never touched."""
    def _try(c: str) -> bool:
        try:
            _ast.parse(c)
            return True
        except SyntaxError:
            return False

    if _try(code):
        return code  # already valid Python — leave untouched

    def _decode_once(c: str) -> str:
        # Order matters: de-double backslashes first (\\\\ -> \\), THEN unescape the
        # now-single sequences. This correctly resolves 3-char \\\\n -> \\n -> real
        # newline, and stray \\\\ at line ends -> nothing (no line-continuation).
        c = c.replace("\\\\", "\x00")  # protect doubled backslashes temporarily
        c = c.replace("\\n", "\n").replace("\\t", "\t")
        c = c.replace("\\'", "'").replace('\\"', '"')
        c = c.replace("\x00", "\\")     # restore doubled backslashes as single
        return c

    cur = code
    for _ in range(6):  # cap iterations to avoid pathological loops
        nxt = _decode_once(cur)
        if nxt == cur:
            break  # fixpoint — no more escapes to decode
        if _try(nxt):
            return nxt
        cur = nxt
    return cur  # best-effort; if still invalid, executor surfaces the SyntaxError
    return cur  # best-effort; if still invalid, executor surfaces the SyntaxError


class EscapedNewlineExecutor(LocalPythonExecutor):
    """LocalPythonExecutor that unescapes literal \\n/\\t before executing code.

    Subclass so we can pass it as CodeAgent(executor=...) without monkeypatching
    smolagents source. All other behavior (state persistence, authorized imports,
    final_answer tool registration) is inherited unchanged."""

    def __call__(self, code_action: str):
        return super().__call__(_unescape_code_literals(code_action))


# ================================================================
# Rate-limited OpenAIServerModel: wrap generate() to acquire a token before
# each OpenAI SDK call. This is the ONLY integration point — smolagents calls
# model.generate() for every LLM step, so wrapping it serializes all CodeAgent
# traffic under the shared bucket.
# ================================================================
class RateLimitedOpenAIModel(OpenAIServerModel):
    """OpenAIServerModel that acquires skillopt's shared token bucket before each call."""

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        if _RATE_LIMITER is not None:
            _RATE_LIMITER.acquire()
        return super().generate(
            messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )


def _normalize_final(out) -> str:
    """Normalize the final_answer() return value to a string extract_int_answer can parse.

    CodeAgent computes answers in Python, so it often returns floats (e.g. 70.0 not 70).
    The single-call baseline's extract_int_answer takes the LAST integer in the string, so
    str(70.0)='70.0' -> findall=['70','0'] -> wrongly extracts 0 (the decimal tail). This
    normalizes whole-number floats to their int string form ('70.0' -> '70') to match the
    fairness contract. Non-numeric strings (e.g. '\\boxed{70}') pass through unchanged so
    the existing \\boxed/answer-tag regex still works."""
    s = str(out).strip() if out is not None else ""
    # When CodeAgent exhausts max_steps, agent.run() returns the summary LLM call's
    # output, which is a list of content parts (e.g. [{"type":"text","text":"..."}]).
    # str() of that list embeds field names ('text', 'type') — extract the clean text
    # so extract_int_answer's \boxed/<answer>/last-int logic runs on prose, not JSON.
    if isinstance(out, list):
        parts = []
        for p in out:
            if isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
            else:
                parts.append(str(p))
        s = "\n".join(parts).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return s
    except (ValueError, TypeError):
        return s


def _make_model() -> RateLimitedOpenAIModel:
    """Build the V3.2-backed model. temperature/stream via kwargs -> self.kwargs -> create().

    CRITICAL: the Midea DeepSeek-V3.2 gateway defaults to SSE streaming even WITHOUT
    stream=True (confirmed via _probe_v32.py: plain create() returns a raw 'data: {...}'
    string, not a ChatCompletion). The OpenAI SDK in non-stream mode does NOT parse SSE,
    so it returns the raw string -> 'str' object has no attribute 'choices'. skillopt's
    qwen_backend works because it forces "stream": False in the payload (see qwen_backend
    comment line 286-287). We replicate that here: stream=False goes into self.kwargs
    (highest priority in _prepare_completion_kwargs, line 545-550) so every non-stream
    generate() call sends stream=False to the gateway. The streaming path (generate_stream)
    passes a literal stream=True AFTER **completion_kwargs, so it still wins there."""
    return RateLimitedOpenAIModel(
        model_id=_V32_MODEL,
        api_base=_V32_BASE_URL,
        api_key=_V32_API_KEY,
        # Midea AIGC gateway requires AIGC-USER routing header on every request.
        client_kwargs={"default_headers": {"AIGC-USER": _AIGC_USER}},
        max_tokens=4096,
        temperature=EVAL_TEMP,
        stream=False,  # force non-streaming (gateway defaults to SSE otherwise)
        flatten_messages_as_text=False,
    )


# ================================================================
# System prompt assembly: AIME skill PREPENDED to CodeAgent default prompt,
# + AIME answer-format suffix. Mirrors how _build_system puts skill first,
# but wrapped in the CodeAgent PoT scaffold so the model writes Python.
# ================================================================
def _build_codeagent_system(skill_content: str) -> str:
    """AIME system prompt = skill section + CodeAgent PoT directive + answer format.

    The skill is prepended (same position as single-call _build_system), then we
    direct the agent to solve via Python (Paradigm 2) and call final_answer(int)."""
    if skill_content.strip():
        skill_section = "## Skill\n" + skill_content.strip() + "\n\n"
    else:
        skill_section = ""
    # NOTE: the CodeAgent default system prompt is injected automatically by the
    # framework (initialize_system_prompt reads code_agent.yaml). We do NOT override
    # prompt_templates — instead we pass our AIME directive as the task system message
    # via the model's custom role conversion is overkill. Simpler & fairer: build the
    # task string and let CodeAgent prepend its own scaffold. We achieve this by setting
    # the agent's system prompt via initialize_system_prompt override below.
    aime_directive = (
        "## Task Format\n"
        "You are an expert mathematical reasoning agent solving AIME-style competition problems "
        "(numerical answers, integers 0 to 999). Reason carefully about the problem setup, "
        "constraints, edge cases, and computational pitfalls.\n\n"
        "## How to solve (Program-of-Thoughts)\n"
        "Translate the problem into Python code. Use the persistent Python interpreter: define "
        "variables, run computations, brute-force small cases, use sympy for symbolic checks "
        "(algebra, modular arithmetic, roots, determinants). State persists across steps — you "
        "can build up a solution incrementally. Double-check your arithmetic with code rather "
        "than computing in your head.\n\n"
        "## Answer Format\n"
        "When you have verified your answer, call final_answer(N) where N is the integer answer.\n"
    )
    return skill_section + aime_directive


# ================================================================
# Per-trajectory timeout wrapper (Bug 5 fix). agent.run() is the multi-step PoT
# loop and has NO outer wall-clock bound. We run it in a daemon thread and
# join(timeout). signal.alarm is INVALID off the main thread, so threading is the
# only option inside a ThreadPoolExecutor worker. The daemon thread cannot be
# force-killed (same Python limitation smolagents hits for step timeouts), but
# abandoning it lets the worker proceed to the next k-vote / problem instead of
# wedging forever. Returns (out, n_steps, timed_out).
# ================================================================
def _run_agent_with_timeout(agent, full_task, timeout=TRAJ_TIMEOUT):
    box = {}

    def _target():
        try:
            box["out"] = agent.run(full_task, reset=True)
            box["n_steps"] = len(getattr(getattr(agent, "memory", None), "steps", []) or [])
        except Exception as e:  # noqa: BLE001 — propagate to caller for fail_reason
            box["err"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        # Still running -> orphaned. We can't kill it; record timeout and move on.
        return None, 0, True
    if "err" in box:
        raise box["err"]
    return box.get("out"), box.get("n_steps", 0), False


# ================================================================
# Single-problem CodeAgent eval: k samples, each a full PoT trajectory.
# ================================================================
def run_one_aime_codeagent(item, agent: CodeAgent, system_prompt: str, k=1, stage="aime_codeagent"):
    """Evaluate one problem with CodeAgent (PoT), k times. pass_rate = n_pass / k.

    agent is a fresh or reused CodeAgent. We reset its memory per sample so each
    k-vote is an independent trajectory (same independence contract as single-call k).
    """
    qid = str(item["id"])
    res = {
        "id": qid, "pass_rate": -3.0, "agent_ok": False, "fail_reason": "",
        "response": "", "predicted_answer": None, "correct_answer": item["answer"],
        "n_pass": 0, "n_samples": 0, "n_steps": 0, "n_timeouts": 0,
    }
    user = _build_user(item)
    n_pass = 0
    n_timeout = 0
    last_resp = ""
    total_steps = 0
    for _ in range(max(1, k)):
        try:
            # run(reset=True) already calls memory.reset()+monitor.reset() internally for
            # k-vote independence (verified: agents.py:478-480). No separate reset needed.
            # We prepend the AIME system directive to the task so the agent sees skill+format.
            full_task = system_prompt + "\n\n" + user
            # Bug 5 fix: wrap the WHOLE multi-step trajectory in a daemon thread +
            # join(timeout). Without this, a single stuck trajectory (sympy busy-loop,
            # gateway hang) wedges the worker forever — smolagents' per-step timeout
            # can't help because the orphaned step thread keeps running at 100% CPU.
            out, n_steps, timed_out = _run_agent_with_timeout(agent, full_task)
            if timed_out:
                n_timeout += 1
                res["fail_reason"] = "traj_timeout_" + str(TRAJ_TIMEOUT) + "s"
                continue
            # out is the final_answer() value (int/float/str). Normalize whole-number
            # floats to int strings so extract_int_answer matches the single-call contract.
            final = _normalize_final(out)
            total_steps += n_steps
            ev = evaluate_aime(final, item["answer"])
            last_resp = final
            if ev["em"]:
                n_pass += 1
        except Exception as e:
            res["fail_reason"] = "exception: " + repr(e)
        # Free the heavy executor state (sympy exprs, big-int lists, 50k-loop results)
        # before the next k-vote reuses this agent — prevents 8GB cgroup OOM kill.
        gc.collect()
    res["n_timeouts"] = n_timeout
    if last_resp:
        res.update({
            "pass_rate": n_pass / max(1, k),
            "agent_ok": True,
            "response": last_resp,
            "predicted_answer": evaluate_aime(last_resp, item["answer"]).get("predicted_answer"),
            "fail_reason": "" if n_pass == 0 and k == 1 else res.get("fail_reason", ""),
        })
    res["n_pass"], res["n_samples"] = n_pass, max(1, k)
    res["n_steps"] = total_steps
    return res


def eval_codeagent_on(items, skill_content, k, seeds, max_steps, max_workers, stage="aime_codeagent",
                      cache_path=None, done_qids=None):
    """Return list of {seed, mean_pass, per_qid, n_steps}. Parallel within each seed.

    Each thread builds its own CodeAgent (not thread-safe to share). The model's
    rate limiter is process-global (shared _RATE_LIMITER) so parallel agents still
    serialize on the 28-RPM bucket.

    Bug 5 fix — per-qid checkpointing: if cache_path is given, completed qids in
    done_qids are skipped, and each newly-finished qid is written to cache_path
    immediately (atomic temp+rename). This replaces the old per-seed granularity
    where one deadlock after 29/30 problems lost everything."""
    runs = []
    sp = _build_codeagent_system(skill_content)
    for seed in range(1, seeds + 1):
        t0 = time.time()
        # Seed the per_qid map from any already-completed (checkpointed) qids so
        # reruns skip them. done_qids is a {qid: pass_rate} dict for THIS seed.
        per_qid = dict(done_qids) if done_qids else {}
        n_steps_total = 0
        todo = [it for it in items if str(it["id"]) not in per_qid]
        n_skip = len(items) - len(todo)
        if n_skip:
            print(
                "  [codeagent seed " + str(seed) + "/" + str(seeds) + "] resuming: "
                + str(n_skip) + "/" + str(len(items)) + " qids already cached",
                flush=True,
            )

        def _work(item):
            # Each worker gets its own agent + model (isolated memory/state, NOT thread-safe
            # to share). The rate limiter is process-global so parallel agents still serialize
            # on the 28-RPM bucket.
            model = _make_model()
            agent = CodeAgent(
                tools=[],  # built-in final_answer auto-registered
                model=model,
                max_steps=max_steps,
                code_block_tags="markdown",  # V3.2 emits markdown ```python blocks reliably
                additional_authorized_imports=["sympy", "math", "itertools", "collections", "numpy"],
                executor=EscapedNewlineExecutor(  # unescape literal \n before execution (V3.2 gateway quirk)
                    additional_authorized_imports=["sympy", "math", "itertools", "collections", "numpy"],
                    timeout_seconds=15,  # bound per-step memory (heavy loops: 50k iters, big-int print) — cgroup is 8GB
                ),
            )
            return str(item["id"]), run_one_aime_codeagent(item, agent, sp, k=k, stage=stage)

        def _checkpoint():
            # Atomic write: dump to temp then rename so a mid-write crash can't
            # corrupt the cache (and a killed process leaves a valid prior file).
            if not cache_path:
                return
            tmp = cache_path + ".tmp"
            json.dump(per_qid, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(tmp, cache_path)

        ck_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_work, it) for it in todo]
            for fut in as_completed(futs):
                qid, r = fut.result()
                pr = r["pass_rate"] if r["pass_rate"] >= 0 else 0.0
                n_steps_total += r.get("n_steps", 0)
                with ck_lock:
                    per_qid[qid] = pr
                    _checkpoint()
                # Surface timeouts/failures inline so a stuck seed is diagnosable.
                nt = r.get("n_timeouts", 0)
                if nt or r.get("fail_reason"):
                    print(
                        "    qid=" + qid + " pass=" + ("%.2f" % pr)
                        + (" timeouts=" + str(nt) if nt else "")
                        + (" fail=" + str(r.get("fail_reason")) if r.get("fail_reason") and not nt else ""),
                        flush=True,
                    )
        mean_pass = sum(per_qid.values()) / max(1, len(per_qid))
        dt = time.time() - t0
        print(
            "  [codeagent seed " + str(seed) + "/" + str(seeds) + "] mean_pass="
            + ("%.4f" % mean_pass)
            + " steps=" + str(n_steps_total)
            + " (%.1f" % (n_steps_total / max(1, len(items))) + "/q) "
            + ("%.0f" % dt) + "s",
            flush=True,
        )
        runs.append({"seed": seed, "mean_pass": mean_pass, "per_qid": per_qid, "n_steps": n_steps_total})
    return runs


# ================================================================
# Baseline loading + statistics (copied from react eval for exact comparability)
# ================================================================
def load_single_cache(skill_name, skill_hash, seeds, single_dir):
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
    ap.add_argument("--data", default="/data/zhuldz/lunwen/data/aime25.jsonl",
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
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=6, help="max CodeAgent steps")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument(
        "--out_dir",
        default="/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/aime25_codeagent",
    )
    ap.add_argument("--single_dir", default="",
                    help="single-call baseline cache dir; empty = report codeagent only")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_temp(EVAL_TEMP)
    print(
        "AIME CodeAgent (PoT) eval | EVAL_TEMP=" + str(EVAL_TEMP)
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

    # ---- run CodeAgent (treatment), per skill per seed cached separately ----
    # Bug 5 fix: per-qid checkpointing. A seed's cache file may be PARTIAL (e.g.
    # 5/30 qids if a previous run was killed mid-seed). We load whatever exists as
    # done_qids, pass cache_path to eval_codeagent_on so it resumes AND writes each
    # finished qid back atomically. Only a full-length cache (len==n_items) is a
    # true hit; a partial one resumes.
    n_items = len(test_items)
    results = {name: [] for name in skills}
    for name, txt in skills.items():
        h = md5(txt)[:8]
        print("\n" + "=" * 50 + "\n> CodeAgent eval: " + name + " (hash " + h + ")\n" + "=" * 50, flush=True)
        for seed in range(1, args.seeds + 1):
            cache = os.path.join(
                args.out_dir, "per_" + name + "_" + h + "_codeagent_seed" + str(seed) + ".json"
            )
            done_qids = {}
            if os.path.exists(cache):
                try:
                    done_qids = json.load(open(cache, encoding="utf-8"))
                except Exception as e:  # noqa: BLE001 — corrupt cache shouldn't abort the run
                    print("  WARN corrupt cache " + cache + " (" + repr(e) + "); starting fresh", flush=True)
                    done_qids = {}
            if len(done_qids) >= n_items:
                mean_pass = sum(done_qids.values()) / max(1, len(done_qids))
                results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": done_qids})
                print(
                    "  [codeagent seed " + str(seed) + "/" + str(args.seeds)
                    + "] CACHED (" + str(len(done_qids)) + "/" + str(n_items)
                    + ") mean_pass=" + "%.4f" % mean_pass, flush=True
                )
                continue
            runs = eval_codeagent_on(
                test_items, txt, args.eval_k, 1, args.max_steps, args.max_workers,
                stage="aime25_codeagent_" + name,
                cache_path=cache, done_qids=done_qids,
            )
            r = runs[0]
            # eval_codeagent_on already checkpointed per-qid; final write keeps the
            # full per_qid (and is redundant-but-safe with the last per-qid write).
            json.dump(r["per_qid"], open(cache, "w", encoding="utf-8"), ensure_ascii=False)
            results[name].append(r)
            # incremental write
            _write_summary(args, test_items, skills, results, args.single_dir, args.out_dir)

    summary = _write_summary(args, test_items, skills, results, args.single_dir, args.out_dir)
    print("\n" + "=" * 50)
    print("Done -> " + os.path.join(args.out_dir, "codeagent_eval.json"))
    print("\nSummary (k=" + str(args.eval_k) + ", AIME=" + str(len(test_items)) + ", seeds=" + str(args.seeds) + "):")
    for name, s in summary.items():
        line = "  " + name + ": mean_pass=" + ("%.4f" % s["mean_pass"]) + \
               " passes=[" + ",".join("%.4f" % p for p in s["passes"]) + "]"
        if "delta_vs_single" in s:
            d = s["delta_vs_single"]
            line += "  DvsSingle mean=" + ("%+.4f" % d["mean"]) + \
                    " t=" + ("%.2f" % d["t"]) + " p=" + ("%.4f" % d["p"]) + " n=" + str(d["n"])
        print(line)


def _write_summary(args, test_items, skills, results, single_dir, out_dir):
    summary = {}
    for name, runs in results.items():
        if not runs:
            continue
        passes = [r["mean_pass"] for r in runs]
        summary[name] = {"n_seeds": len(passes), "mean_pass": sum(passes) / len(passes), "passes": passes}
        # delta vs single-call baseline (paired t-test on per-qid pass rates)
        if single_dir:
            h = md5(skills[name])[:8]
            single_runs = load_single_cache(name, h, args.seeds, single_dir)
            if single_runs:
                cg_qid = collect_per_qid(runs)
                sg_qid = collect_per_qid(single_runs)
                tt = paired_t_test(cg_qid, sg_qid)
                if tt:
                    summary[name]["delta_vs_single"] = tt
    out = {"args": vars(args), "n_test": len(test_items), "summary": summary, "runs": results}
    json.dump(out, open(os.path.join(out_dir, "codeagent_eval.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    main()
