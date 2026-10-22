#!/usr/bin/env python3
"""Smoke test for eval_aime_react_multiseed: 2 AIME25 problems, s0, k=1, max_steps=4.
Verifies the ReAct loop end-to-end before a full run. Run with the QWEN_CHAT_* env exported."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_aime_react_multiseed as M
from tf_global_v2_aime import set_temp, EVAL_TEMP, md5

set_temp(EVAL_TEMP)
items = M.load_aime("/data/zhuldz/lunwen/data/aime25.jsonl")[:2]
skill = open("/data/zhuldz/lunwen/data/skill_aime_initial_v2.md", encoding="utf-8").read().strip()
print(f"smoke: {len(items)} problems, skill_hash={md5(skill)[:8]}, EVAL_TEMP={EVAL_TEMP}", flush=True)
for it in items:
    print(f"\n--- problem {it['id']} (answer={it['answer']}) ---", flush=True)
    t0 = time.time()
    r = M.run_one_aime_react(it, skill, k=1, max_steps=4)
    print(
        f"  pass_rate={r['pass_rate']} agent_ok={r['agent_ok']} "
        f"n_tool_calls={r['n_tool_calls']} elapsed={time.time()-t0:.1f}s", flush=True
    )
    print(
        f"  predicted={r['predicted_answer']} correct={r['correct_answer']} "
        f"fail_reason={r.get('fail_reason','')}", flush=True
    )
    resp = r.get('response', '')
    print(f"  final_resp_tail: ...{resp[-160:]}", flush=True)
print("\nSMOKE DONE", flush=True)
