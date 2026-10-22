#!/usr/bin/env python3
"""Smoke test: run CodeAgent on ONE AIME25 problem (q0, answer=70), k=1, max_steps=4.
Verifies V3.2 connection + rate limiter + final_answer termination + answer extraction."""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eval_aime_codeagent_multiseed import (
    _make_model, _build_codeagent_system, run_one_aime_codeagent, EVAL_TEMP,
    EscapedNewlineExecutor,
)
from eval_aime24_multiseed import load_aime, evaluate_aime
from smolagents import CodeAgent
from tf_global_v2_aime import set_temp

set_temp(EVAL_TEMP)
items = load_aime("/data/zhuldz/lunwen/data/aime25.jsonl")
item = items[0]  # q0, answer=70
print(f"Smoke: q{item['id']} answer={item['answer']}")
print(f"  problem: {item['problem'][:120]}...")

skill = open("/data/zhuldz/lunwen/data/skill_aime_initial_v2.md", encoding="utf-8").read().strip()
sp = _build_codeagent_system(skill)
model = _make_model()
agent = CodeAgent(
    tools=[],
    model=model,
    max_steps=4,
    code_block_tags="markdown",
    additional_authorized_imports=["sympy", "math", "itertools", "collections", "numpy"],
    executor=EscapedNewlineExecutor(
        additional_authorized_imports=["sympy", "math", "itertools", "collections", "numpy"],
    ),
)
t0 = time.time()
r = run_one_aime_codeagent(item, agent, sp, k=1, stage="smoke")
dt = time.time() - t0
print(f"\n--- RESULT ({dt:.0f}s) ---")
print(f"  pass_rate={r['pass_rate']}  predicted={r['predicted_answer']}  correct={r['correct_answer']}")
print(f"  agent_ok={r['agent_ok']}  n_steps={r['n_steps']}  fail_reason={r['fail_reason']}")
print(f"  response(final_answer value): {r['response'][:200]}")
