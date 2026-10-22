#!/usr/bin/env python3
"""S₀ cache 重生 (live 路径, 带 manifest) — Phase B item (a) 修复.

旧 launch_s0_regen.sh 调 precompute_s0_1-3-6.py (legacy, temp=0.0, 无 manifest) →
manifest loader 会拒绝. 本脚本走 live tf_global_v2_math.precompute_s0 (EVAL_TEMP=0.3,
k 投票, 写 _manifest), 产物与生产 evolver load_s0_cache_with_manifest 口径完全一致.

口径 (与 launch_arm_on.sh 对齐):
  split_dir = livemathematicianbench_1-3-6_seed42
  select    = train+val = 71 items
  eval_k    = 3
  shuffle_seed = None (evolver 默认, launch 脚本未传 --shuffle_seed)
  split_name = "train+val"
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tf_global_v2_math import precompute_s0, load_split, md5  # noqa: E402

S0 = "/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md"
SPLIT_DIR = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42"
OUT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json"
EVAL_K = 3
SHUFFLE_SEED = None  # 与 evolver 默认 (launch 脚本未传 --shuffle_seed) 一致

if __name__ == "__main__":
    s0 = open(S0, encoding="utf-8").read().strip()
    train = load_split(SPLIT_DIR, "train")
    val = load_split(SPLIT_DIR, "val")
    select = train + val
    print(f"📋 S₀: {len(s0)} chars (hash {md5(s0)[:8]}) | select(train+val): {len(select)} "
          f"(train {len(train)} + val {len(val)}) | k={EVAL_K} | shuffle_seed={SHUFFLE_SEED}", flush=True)
    t0 = time.time()
    cache = precompute_s0(select, s0, EVAL_K,
                          stage="tf_v2_s0_regen",
                          shuffle_seed=SHUFFLE_SEED,
                          split_dir=SPLIT_DIR, split_name="train+val")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(cache, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    manifest = cache.get("_manifest", {})
    n_items = len(cache) - 1  # 减 _manifest 键
    print(f"✅ wrote {OUT} ({n_items} items + manifest) in {time.time()-t0:.0f}s", flush=True)
    print(f"   manifest: k={manifest.get('k')} eval_temp={manifest.get('eval_temp')} "
          f"shuffle_seed={manifest.get('shuffle_seed')} skill_hash={manifest.get('skill_hash','')[:8]} "
          f"split_hash={manifest.get('split_hash','')[:8]} n_items={manifest.get('n_items')}", flush=True)
