#!/usr/bin/env python3
"""RQ3-C #48: 把 Arm1 侧的有效结果合并成一份 #18 可用的输入文件 (0 API)。

来源 (按优先级, 先到先得):
  1. `arm1/run_20261125_154821/test_independent_results.json` — 训练 run 的 final test 段 (同一冻结 skill,
     sha 5be5cdf5…), 只取 agent_ok=True 的条目 (58 条 infra 500 见 Issue #123, 不可作 #18 输入)
  2. `arm1/test_2765_20261127/arm1_test_cache.json` — #48 定向补齐的 31 题 (同 skill / 同口径)

输出 `arm1_test_combined.json` = {id: {hard, pass_rate, agent_ok=True, src, skill_sha256}},
以及 `arm1_test_combined_meta.json` (每来源条数、覆盖、缺失 id)。

用法: python3 scripts/rq3_c_combine_arm1.py [--model deepseek-v4-pro]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

BASE = "/data/zhuldz/lunwen/outputs/rq3"
ARM1_SHA_EXPECTED = "5be5cdf5b47e83a028f862d23d64b9db94af6cb42334c6a44da512057e467a16"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    a = ap.parse_args()
    d = os.path.join(BASE, a.model)
    run_dir = os.path.join(d, "arm1", "run_20261125_154821")
    fill_dir = os.path.join(d, "arm1", "test_2765_20261127")

    skill_path = os.path.join(run_dir, "tf_global_skill.md")
    sha = hashlib.sha256(open(skill_path, "rb").read()).hexdigest()
    if sha != ARM1_SHA_EXPECTED:
        print(f"[combine] BLOCKED: run skill sha={sha[:12]} ≠ 预期 {ARM1_SHA_EXPECTED[:12]}", file=sys.stderr)
        return 2
    fill_meta = json.load(open(os.path.join(fill_dir, "run_meta.json"), encoding="utf-8"))
    if fill_meta.get("skill_sha256") != sha:
        print(f"[combine] BLOCKED: 补齐 run 的 skill sha={str(fill_meta.get('skill_sha256'))[:12]} ≠ {sha[:12]}",
              file=sys.stderr)
        return 2

    src1 = json.load(open(os.path.join(run_dir, "test_independent_results.json"), encoding="utf-8"))
    fill_path = os.path.join(fill_dir, "arm1_test_cache.json")
    src2 = json.load(open(fill_path, encoding="utf-8")) if os.path.exists(fill_path) else {}

    out, prov = {}, {"run_20261125_154821": 0, "test_2765_20261127": 0}
    skipped_infra = []
    for qid, r in src1.items():
        if r.get("agent_ok"):
            out[str(qid)] = {k: r[k] for k in ("hard", "pass_rate", "agent_ok")}
            out[str(qid)]["src"] = "run_20261125_154821"
            prov["run_20261125_154821"] += 1
        else:
            skipped_infra.append(str(qid))
    for qid, r in src2.items():
        if not r.get("agent_ok"):
            continue
        if str(qid) in out:
            print(f"[combine] WARN: {qid} 同时在两个来源中, 保留 run 结果", file=sys.stderr)
            continue
        out[str(qid)] = {k: r[k] for k in ("hard", "pass_rate", "agent_ok")}
        out[str(qid)]["src"] = "test_2765_20261127"
        prov["test_2765_20261127"] += 1

    for v in out.values():
        v["skill_sha256"] = sha
    s0 = json.load(open(os.path.join(d, "test_s0_cache.json"), encoding="utf-8"))
    missing = sorted(set(s0) - set(out), key=int)
    hard_n = sum(int(v["hard"]) for v in out.values())
    meta = {"model": a.model, "skill_path": skill_path, "skill_sha256": sha,
            "sources": prov, "skipped_run_infra_fail": len(skipped_infra),
            "skipped_run_infra_fail_ids": sorted(skipped_infra, key=int),
            "n_arm1_valid": len(out), "hard_pass": hard_n, "fail_pool": len(out) - hard_n,
            "hard_rate": round(hard_n / max(1, len(out)), 6),
            "n_s0_valid": len(s0), "n_s0_minus_arm1": len(missing), "s0_minus_arm1_ids": missing}
    json.dump(out, open(os.path.join(d, "arm1_test_combined.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(meta, open(os.path.join(d, "arm1_test_combined_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: meta[k] for k in ("sources", "n_arm1_valid", "hard_pass", "hard_rate",
                                           "n_s0_valid", "n_s0_minus_arm1")}, ensure_ascii=False))
    print(f"[combine] s0-valid-but-arm1-missing: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
