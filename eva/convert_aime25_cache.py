#!/usr/bin/env python3
# eva/convert_aime25_cache.py — 把 .verify_tmp 产出的 {id: float} 裸格式 AIME25 S0 cache
# 转成 crossmodel_flash_baseline.py 期望的 {id: {"pass_rate": float}} 嵌套格式。
#
# 输入: per_s0_seed1.json + per_s0_seed2.json (两 seed 的逐题 pass_rate)
# 输出: s0_cache_aime25.json (两 seed 均值, 嵌套格式)
#
# 用法:
#   python eva/convert_aime25_cache.py
import json
from pathlib import Path

BASE = Path("/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/flash_aime25_s0_20261104_100708")
S1 = BASE / "per_s0_seed1.json"
S2 = BASE / "per_s0_seed2.json"
OUT = BASE / "s0_cache_aime25.json"

s1 = json.load(open(S1))
s2 = json.load(open(S2))

# 两 seed 均值; 若某 seed 缺该 id 则用另一 seed 单值
all_ids = set(s1) | set(s2)
cache = {}
for k in sorted(all_ids, key=lambda x: int(x) if str(x).lstrip("-").isdigit() else x):
    vals = [v for v in (s1.get(k), s2.get(k)) if v is not None]
    cache[k] = {"pass_rate": sum(vals) / len(vals)}

json.dump(cache, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
n = len(cache)
mean = sum(c["pass_rate"] for c in cache.values()) / n
full = sum(1 for c in cache.values() if c["pass_rate"] >= 1.0 - 1e-9)
print(f"[convert] {n} 题, mean={mean:.4f}, 满分={full}/{n} → {OUT}")
