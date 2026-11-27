#!/usr/bin/env python3
"""RQ3-C #48: 用**冻结的 Arm1 skill** 在全量 test split (2765) 上重评, 产出 #18 配对的 Arm1 侧。

为什么需要 (Issue #123): Arm1 训练 run 的 final test 段 (`arm1/run_20261125_154821/test_independent_results.json`)
实际只评了 `s0_cache_test_d5.json` 那 100 题 × val id 的空交集 —— 其 pass 率不可作 #18 输入。
本脚本以同一冻结 skill (`tf_global_skill.md`, sha 5be5cdf5…) 作 system prompt, 口径与
`rq3_b_eval_test_s0.py` 完全一致 (同 env / 同 run_one / 同 max_tokens), 只把 "skill 内容" 从 S₀ 换成 Arm1。

输出:
  <run-dir>/<arm1-dir>/calls_arm1_test.jsonl  逐题日志 (stage=rq3c_arm1_test)
  <run-dir>/<arm1-dir>/arm1_test_cache.json   {id: run_one 结果}, 只含 agent_ok=True 的定稿条目
  <run-dir>/<arm1-dir>/arm1_test_summary.json

用法:
  RQ3_MODEL=deepseek-v4-pro python3 -u scripts/rq3_arm1_test_eval.py \
      --only-ids-file /tmp/rq3_arm1_leftover31.txt --workers 8 --rpm 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, "/data/zhuldz/lunwen/scripts")
import rq3_b_eval_test_s0 as B  # noqa: E402

DEFAULT_SKILL = "/data/zhuldz/lunwen/outputs/rq3/deepseek-v4-pro/arm1/run_20261125_154821/tf_global_skill.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("RQ3_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--skill", default=DEFAULT_SKILL)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--rpm", type=int, default=20)
    ap.add_argument("--burst", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--only-ids-file", default="")
    ap.add_argument("--dry-run", action="store_true")
    # 端点覆盖 (默认 = pro 的 OpenAI 兼容端点, 行为不变); GLM-5.2 用 aimp-glm/v1
    ap.add_argument("--base-url", default="https://aimpapi.midea.com/t-aigc/llm-openai-api")
    ap.add_argument("--aigc-user", default="")
    a = ap.parse_args()
    if not a.out_dir:
        a.out_dir = os.path.join("/data/zhuldz/lunwen/outputs/rq3", a.model, "arm1", "test_2765_20261127")

    key = os.environ.get("QWEN_CHAT_API_KEY") or os.environ.get("AIMP_KEY")
    if not key:
        print("[rq3c] BLOCKED: 未设置 QWEN_CHAT_API_KEY / AIMP_KEY", file=sys.stderr)
        return 2
    os.environ["QWEN_CHAT_API_KEY"] = key
    os.environ["QWEN_CHAT_BASE_URL"] = a.base_url
    os.environ["QWEN_CHAT_MODEL"] = a.model
    if a.aigc_user:
        os.environ["QWEN_CHAT_AIGC_USER"] = a.aigc_user
    else:
        os.environ.pop("QWEN_CHAT_AIGC_USER", None)
    os.environ["QWEN_CHAT_MAX_TOKENS"] = str(a.max_tokens)
    os.environ["QWEN_CHAT_TEMPERATURE"] = "0.0"
    os.environ["QWEN_CHAT_ENABLE_THINKING"] = "false"
    os.environ["QWEN_CHAT_MAX_RPM"] = str(a.rpm)
    os.environ["QWEN_CHAT_BURST"] = str(a.burst)
    os.environ["QWEN_CHAT_TIMEOUT_SECONDS"] = "600"
    os.environ["TARGET_BACKEND"] = "qwen_chat"
    os.environ["OPTIMIZER_BACKEND"] = "qwen_chat"
    sys.path.insert(0, B.SKILLOPT)

    ev = B._load_evolver()
    from skillopt.model import qwen_backend  # noqa: E402
    qwen_backend.TARGET_CONFIG = qwen_backend._initial_config("target")
    cfg = qwen_backend.TARGET_CONFIG
    token_key = "max_completion_tokens" if cfg.use_max_completion_tokens else "max_tokens"
    serialized = {"model": cfg.deployment, "base_url": cfg.base_url,
                  token_key: min(a.max_tokens, cfg.max_tokens), "stream": False,
                  "temperature": cfg.temperature,
                  "thinking": {"chat_template_kwargs": {"enable_thinking": True}}
                  if cfg.enable_thinking else "server_default"}
    if serialized["model"] != a.model:
        print(f"[rq3c] BLOCKED: 生效 model={serialized['model']} ≠ 请求 {a.model}", file=sys.stderr)
        return 2

    skill = open(a.skill, encoding="utf-8").read()
    test_items = ev.load_split(B.SPLIT_DIR, "test", truncate=True)
    only = set()
    if a.only_ids_file:
        only = {s.strip() for s in open(a.only_ids_file).read().replace("\n", ",").split(",") if s.strip()}
        unknown = only - {str(it["id"]) for it in test_items}
        if unknown:
            print(f"[rq3c] BLOCKED: {len(unknown)} 个 id 不在 test split: {sorted(unknown)[:10]}", file=sys.stderr)
            return 2
    items = [it for it in test_items if not only or str(it["id"]) in only]

    os.makedirs(a.out_dir, exist_ok=True)
    cache_path = os.path.join(a.out_dir, "arm1_test_cache.json")
    log_path = os.path.join(a.out_dir, "calls_arm1_test.jsonl")
    import hashlib
    meta = {"model": a.model, "skill_path": a.skill,
            "skill_sha256": hashlib.sha256(skill.encode("utf-8")).hexdigest(),
            "skill_chars": len(skill), "serialized_request": serialized,
            "rpm": a.rpm, "burst": a.burst, "workers": a.workers,
            "split_dir": B.SPLIT_DIR, "evolver": B.EVOLVER,
            "stage": "rq3c_arm1_test", "started_at": datetime.now(timezone.utc).isoformat()}
    json.dump(meta, open(os.path.join(a.out_dir, "run_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[rq3c] model={a.model} n={len(items)}/{len(test_items)} skill_chars={len(skill)} "
          f"skill_sha={meta['skill_sha256'][:12]}", flush=True)
    print(f"[rq3c] serialized: {json.dumps(serialized, ensure_ascii=False)}", flush=True)
    if a.dry_run:
        print("[rq3c] dry-run: 0 请求")
        return 0

    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
        print(f"[rq3c] resume: 盘上已有 {len(cache)} 题有效结果", flush=True)
    todo = [it for it in items if str(it["id"]) not in cache]

    lock = threading.Lock()

    def _one(it):
        t0 = time.time()
        r = ev.run_one(it, skill, "rq3c_arm1_test")
        row = {"ts": datetime.now(timezone.utc).isoformat(), "stage": "rq3c_arm1_test",
               "item_id": str(it["id"]), "model": serialized["model"],
               "skill_sha256": meta["skill_sha256"][:16],
               "prompt_tokens": None, "completion_tokens": None,
               "latency_s": round(time.time() - t0, 2),
               "hard": int(r["hard"]), "pass_rate": r["pass_rate"], "agent_ok": r.get("agent_ok"),
               "fail_reason": str(r.get("fail_reason", ""))[:200]}
        with lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return r

    def dump():
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    cnt = {"ok": 0, "infra": 0}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, it): str(it["id"]) for it in todo}
        for n, f in enumerate(as_completed(futs), 1):
            qid = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"id": qid, "hard": 0, "pass_rate": -3.0, "agent_ok": False,
                     "fail_reason": f"exception: {e!r}"[:300]}
            if r.get("agent_ok"):
                cache[qid] = r
                cnt["ok"] += 1
            else:
                cnt["infra"] += 1   # 调用层失败不进 cache → resume 时重跑
            if n % 5 == 0 or n == len(todo):
                print(f"[rq3c] {n}/{len(todo)} done ({cnt['ok']} ok, {cnt['infra']} infra) "
                      f"cache={len(cache)}", flush=True)
                dump()
    dump()

    hard_n = sum(int(r["hard"]) for r in cache.values())
    missing = sorted({str(it["id"]) for it in items} - set(cache), key=int)
    out = {"model": a.model, "n_target": len(items), "n_valid": len(cache),
           "n_undetermined": len(missing), "undetermined_ids": missing,
           "coverage": round(len(cache) / max(1, len(items)), 6),
           "hard_pass": hard_n, "fail_pool": len(cache) - hard_n,
           "hard_rate": round(hard_n / max(1, len(cache)), 6), **meta}
    json.dump(out, open(os.path.join(a.out_dir, "arm1_test_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in ("n_target", "n_valid", "n_undetermined", "coverage",
                                          "hard_pass", "fail_pool", "hard_rate")}, ensure_ascii=False), flush=True)
    return 0 if not missing else 4


if __name__ == "__main__":
    sys.exit(main())
