#!/usr/bin/env bash
# 单 bullet 消融: S₀ vs Skill A (纯验证) vs Skill B (全 Record2) 在 AIME24 上
# paired eval, k=4, seed=0, V3.2 backend. S₀ 复用已有缓存.
set -euo pipefail

cd /data/zhuldz/lunwen

V3_KEY=$(python3 -c "
import yaml
c=yaml.safe_load(open('/data/zhuldz/lunwen/configs/models_aimp.local.yaml'))
print(c['providers']['deepseek-v3.2']['api_key'])
" 2>/dev/null)

export QWEN_CHAT_API_KEY="$V3_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1"
export QWEN_CHAT_MODEL="/model/DeepSeek-V3.2"
export QWEN_CHAT_AIGC_USER="zhuldz"
export QWEN_CHAT_MAX_TOKENS="8192"
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_ENABLE_THINKING="false"
export QWEN_CHAT_TRUNC_RETRY="0"
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="28"
export QWEN_CHAT_BURST="2"
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

EVAL_K=4
SEED=0

source lunwen/bin/activate

python3 - <<PYEOF 2>&1
import json, os, sys, numpy as np, hashlib
sys.path.insert(0, "/data/zhuldz/skillopt")
sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
sys.path.insert(0, "/data/zhuldz/lunwen/for_review/code")

import _env
from eval_aime24_multiseed import load_aime, run_one_aime, _build_system, _build_user
from tf_global_v2_math import bootstrap_ci_lower

# Load AIME24 data
items = load_aime("/data/zhuldz/lunwen/data/aime24.jsonl")
print(f"Loaded {len(items)} problems")

# Load S0 cache (k=3)
s0_cache_path = "rl/train/verl1/outputs/crossmodel_baseline/tf_aime_trunc_efl_bgt_both_gates_v3/run_trunc_efl_bgt_dual_trunc_v3_20261119_140908/run_20261119_140908/s0_cache.json"
with open(s0_cache_path) as f:
    s0_cache = json.load(f)
print(f"S0 cache loaded: {len(s0_cache)} entries")

# Load skills
with open("outputs/bullet_ablation_skill_A_pure_verify.md") as f:
    skill_A = f.read()
with open("outputs/bullet_ablation_skill_B_full_record2.md") as f:
    skill_B = f.read()

k = int(os.environ.get("EVAL_K", 4))
seed = int(os.environ.get("SEED", 0))

print(f"\nRunning eval: k={k}, seed={seed}")
print(f"Skill A (pure verify): {hashlib.sha1(skill_A.encode()).hexdigest()[:10]}, {len(skill_A)} chars")
print(f"Skill B (full Record2): {hashlib.sha1(skill_B.encode()).hexdigest()[:10]}, {len(skill_B)} chars")

results = {"s0": {}, "skill_A": {}, "skill_B": {}}

for skill_name, skill_content in [("skill_A", skill_A), ("skill_B", skill_B)]:
    passes = []
    for item in items:
        qid = str(item["id"])
        # Get S0 pass_rate from cache (k=3)
        s0_entry = s0_cache.get(qid, {})
        s0_pass = s0_entry.get("pass_rate", -1)
        
        result = run_one_aime(item, skill_content, stage=f"bullet_ablation_{skill_name}", k=k)
        skill_pass = result["pass_rate"]
        
        results[skill_name][qid] = {
            "pass_rate": skill_pass,
            "s0_pass": s0_pass,
            "delta": skill_pass - s0_pass if s0_pass >= 0 else None,
            "n_samples": result.get("n_samples", k),
            "n_pass": result.get("n_pass", 0),
            "harness": result.get("harness", {}),
            "samples": [
                {"finish_reason": s.get("finish_reason", "?"),
                 "truncated": s.get("truncated", False),
                 "em": s.get("em", False)}
                for s in result.get("samples", [])
            ],
        }
        
        passes.append(skill_pass)
        
        deltas_with_s0 = [v["delta"] for v in results[skill_name].values() if v["delta"] is not None]
        mean_delta = np.mean(deltas_with_s0) if deltas_with_s0 else 0
        n_done = len(passes)
        print(f"  [{skill_name}] {n_done}/{len(items)}: mean_pass={np.mean(passes):.4f} delta_vs_s0={mean_delta:.4f}", flush=True)
    
    mean_pass = np.mean(passes)
    
    # Also get S0 pass rates
    s0_passes = [s0_cache.get(str(item["id"]), {}).get("pass_rate", -1) for item in items]
    s0_valid = [p for p in s0_passes if p >= 0]
    s0_mean = np.mean(s0_valid) if s0_valid else 0
    
    deltas = [skill_p - s0_p for skill_p, s0_p in zip(passes, s0_passes) if s0_p >= 0]
    mean_delta = np.mean(deltas) if deltas else 0
    ci_low = bootstrap_ci_lower(deltas, n_bootstrap=10000) if deltas else float('nan')
    
    print(f"\n  [{skill_name}] FINAL: mean_pass={mean_pass:.4f} s0_mean={s0_mean:.4f} delta={mean_delta:.4f} CI_lower={ci_low:.4f}")

# Summary
print("\n" + "="*60)
print("BULLET ABLATION SUMMARY")
print("="*60)
for skill_name in ["skill_A", "skill_B"]:
    passes_out = [results[skill_name][str(item["id"])]["pass_rate"] for item in items]
    s0_pass_dict = {str(item["id"]): s0_cache.get(str(item["id"]), {}).get("pass_rate", -1) for item in items}
    deltas_out = [passes_out[i] - s0_pass_dict[str(item["id"])] for i, item in enumerate(items) if s0_pass_dict[str(item["id"])] >= 0]
    mean_d = np.mean(deltas_out) if deltas_out else 0
    ci_l = bootstrap_ci_lower(deltas_out, n_bootstrap=10000) if deltas_out else float('nan')
    
    # truncation count
    n_trunc = sum(1 for item in items 
                  for s in results[skill_name][str(item["id"])].get("samples", [])
                  if s.get("truncated") or s.get("finish_reason") == "length")
    n_total = sum(len(results[skill_name][str(item["id"])].get("samples", [])) for item in items)
    print(f"\n{skill_name}:")
    print(f"  mean_pass: {np.mean(passes_out):.4f}")
    print(f"  delta_vs_s0: {mean_d:.4f}")
    print(f"  CI_lower: {ci_l:.4f}")
    print(f"  CI > 0: {ci_l > 0}")
    print(f"  trunc: {n_trunc}/{n_total} ({n_trunc/max(n_total,1):.1%})")

# Write results
out_path = "outputs/bullet_ablation_results.json"
with open(out_path, "w") as f:
    json.dump({
        "args": {"k": k, "seed": seed},
        "skills": {
            "s0": {"hash": hashlib.sha1(open("/data/zhuldz/lunwen/data/skill_aime_initial_v2.md").read().encode()).hexdigest()[:10]},
            "skill_A": {"hash": hashlib.sha1(skill_A.encode()).hexdigest()[:10], "desc": "S0 + pure verification bullet"},
            "skill_B": {"hash": hashlib.sha1(skill_B.encode()).hexdigest()[:10], "desc": "S0 + full Record2 (5 bullets)"},
        },
        "results": results,
    }, f, indent=2, ensure_ascii=False)
print(f"\nResults written to {out_path}")

PYEOF
echo ""
echo "=== done ==="
