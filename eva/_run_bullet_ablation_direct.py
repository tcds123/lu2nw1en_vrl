#!/usr/bin/env python3
"""Single bullet ablation: S₀ vs Skill A (pure verify [1]) vs Skill B (full Record2 [0-4]) on AIME24.
Paired eval, k=4, seed=0, V3.2. S₀ reuses existing cache.
"""
import json, os, sys, hashlib, numpy as np

sys.path.insert(0, "/data/zhuldz/skillopt")
sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
sys.path.insert(0, "/data/zhuldz/lunwen/for_review/code")

import _env
from eval_aime24_multiseed import load_aime, run_one_aime
from tf_global_v2_math import bootstrap_ci_lower

items = load_aime("/data/zhuldz/lunwen/data/aime24.jsonl")
print(f"Loaded {len(items)} problems")

s0_cache_path = "rl/train/verl1/outputs/crossmodel_baseline/tf_aime_trunc_efl_bgt_both_gates_v3/run_trunc_efl_bgt_dual_trunc_v3_20261119_140908/run_20261119_140908/s0_cache.json"
with open(s0_cache_path) as f:
    s0_cache = json.load(f)
print(f"S0 cache loaded: {len(s0_cache)} entries")

with open("outputs/bullet_ablation_skill_A_pure_verify.md") as f:
    skill_A = f.read()
with open("outputs/bullet_ablation_skill_B_full_record2.md") as f:
    skill_B = f.read()

k = 4
seed = 0

print(f"Skill A (pure verify): {hashlib.sha1(skill_A.encode()).hexdigest()[:10]}, {len(skill_A)} chars")
print(f"Skill B (full Record2): {hashlib.sha1(skill_B.encode()).hexdigest()[:10]}, {len(skill_B)} chars")

results = {"s0": {}, "skill_A": {}, "skill_B": {}}

# Get S0 pass rates
for item in items:
    qid = str(item["id"])
    s0_entry = s0_cache.get(qid, {})
    results["s0"][qid] = {"pass_rate": s0_entry.get("pass_rate", -1)}

for skill_name, skill_content in [("skill_A", skill_A), ("skill_B", skill_B)]:
    passes = []
    for item in items:
        qid = str(item["id"])
        s0_pass = results["s0"][qid]["pass_rate"]

        result = run_one_aime(item, skill_content, stage=f"ablation_{skill_name}", k=k)
        skill_pass = result["pass_rate"]

        results[skill_name][qid] = {
            "pass_rate": skill_pass,
            "s0_pass": s0_pass,
            "delta": skill_pass - s0_pass if s0_pass >= 0 else None,
            "n_samples": result.get("n_samples", k),
            "n_pass": result.get("n_pass", 0),
            "samples": [
                {"finish_reason": s.get("finish_reason", "?"),
                 "truncated": s.get("truncated", False),
                 "em": s.get("em", False)}
                for s in result.get("samples", [])
            ],
        }
        passes.append(skill_pass)

        deltas_done = [v["delta"] for v in results[skill_name].values() if v["delta"] is not None]
        mean_d = np.mean(deltas_done) if deltas_done else 0
        n = len(passes)
        print(f"  [{skill_name}] {n:2d}/{len(items)}: mean_pass={np.mean(passes):.4f} delta={mean_d:+.4f}", flush=True)

    mean_pass = np.mean(passes)
    s0_pass_dict = results["s0"]
    deltas = [results[skill_name][str(item["id"])]["pass_rate"] - s0_pass_dict[str(item["id"])]["pass_rate"]
              for item in items
              if s0_pass_dict[str(item["id"])]["pass_rate"] >= 0]
    mean_delta = np.mean(deltas)
    ci_low = bootstrap_ci_lower(deltas, n_bootstrap=10000)
    print(f"\n  [{skill_name}] FINAL: mean_pass={mean_pass:.4f} delta={mean_delta:+.4f} CI_lower={ci_low:+.4f} CI>0={ci_low>0}", flush=True)

    # Truncation
    n_trunc = sum(1 for item in items
                  for s in results[skill_name][str(item["id"])].get("samples", [])
                  if s.get("truncated") or s.get("finish_reason") == "length")
    n_total = sum(len(results[skill_name][str(item["id"])].get("samples", [])) for item in items)
    print(f"  [{skill_name}] trunc: {n_trunc}/{n_total} ({n_trunc/max(n_total,1):.1%})", flush=True)

# Summary
print("\n" + "=" * 60)
print("BULLET ABLATION SUMMARY")
print("=" * 60)
for skill_name in ["skill_A", "skill_B"]:
    passes_out = [results[skill_name][str(item["id"])]["pass_rate"] for item in items]
    s0_passes = [results["s0"][str(item["id"])]["pass_rate"] for item in items]
    deltas_out = [passes_out[i] - s0_passes[i] for i in range(len(items)) if s0_passes[i] >= 0]
    mean_d = np.mean(deltas_out) if deltas_out else 0
    ci_l = bootstrap_ci_lower(deltas_out, n_bootstrap=10000) if deltas_out else float('nan')
    n_trunc = sum(1 for item in items
                  for s in results[skill_name][str(item["id"])].get("samples", [])
                  if s.get("truncated") or s.get("finish_reason") == "length")
    n_total = sum(len(results[skill_name][str(item["id"])].get("samples", [])) for item in items)
    print(f"\n{skill_name}:")
    print(f"  mean_pass: {np.mean(passes_out):.4f}")
    print(f"  delta: {mean_d:+.4f}")
    print(f"  CI_lower: {ci_l:+.4f}")
    print(f"  CI > 0: {ci_l > 0}")
    print(f"  trunc: {n_trunc}/{n_total}")

out_path = "outputs/bullet_ablation_results.json"
with open(out_path, "w") as f:
    json.dump({
        "args": {"k": k, "seed": seed},
        "skills": {
            "s0": {"hash": hashlib.sha1(open("/data/zhuldz/lunwen/data/skill_aime_initial_v2.md").read().encode()).hexdigest()[:10]},
            "skill_A": {"hash": hashlib.sha1(skill_A.encode()).hexdigest()[:10], "desc": "S0 + pure verification bullet[1]"},
            "skill_B": {"hash": hashlib.sha1(skill_B.encode()).hexdigest()[:10], "desc": "S0 + full Record2 (5 bullets)"},
        },
        "results": results,
    }, f, indent=2, ensure_ascii=False)
print(f"\nResults -> {out_path}")
print("=== done ===")
