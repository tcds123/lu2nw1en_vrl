#!/usr/bin/env python3
"""RQ3-B1: 目标模型在 val split 上的 S₀ 评测 (Arm1 训练门控信号来源) + 逐调用日志。

背景 (设计决策, 2026-11-25): 生产 Evolver 的门控 cache (--s0_cache_val) 是 **flash 在 100 道
val 题上的 S₀ 缓存**。RQ3 要求"skill 由目标模型自己的失败池进化", 若沿用 flash 的 val cache:
  (a) 门控 turn1 用的是别的模型的通过/失败判定;
  (b) routed 评测里 S₀-pass 题直接复用 flash 的结果, 等于把 flash 能力混进 Arm1 的 val 分。
因此 RQ3 在目标模型上重算 val S₀, 算法/协议 (eval_on_routed + 100 题子集) 不变, 只换 S₀ 来源。

评测题集 = 生产 cache 覆盖的 100 题 (按生产 cache 的 id 集合取), 以保证与 flash run 同口径。
输出为**逐题 dict** (`{id: result}`) — 与 `precompute_s0` 产物同形, 不是 list。

用法:
    bash /tmp/rq3_run_val.sh
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

SKILLOPT = "/data/zhuldz/skillopt"
EVOLVER = "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py"
SPLIT_DIR = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
S0_PATH = "/data/zhuldz/skillopt/skillopt/envs/appsplus/skills/initial.md"
# 生产 val cache (flash S₀, 100 题) — 只用来取"题集"和做对照, 不用其分数
FLASH_VAL_CACHE = ("/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/"
                   "appsplus-skillopt-flash-20261111-v2/train/selection_eval_baseline/results.json")

_log_lock = threading.Lock()


def _load_evolver():
    spec = importlib.util.spec_from_file_location("tf_evolver", EVOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--out-dir", default="outputs/rq3/deepseek-v4-pro")
    # 端点覆盖 (默认 = pro 的 OpenAI 兼容端点, 行为不变)。GLM-5.2 走 AIMP 专有端点 aimp-glm/v1;
    # 评测口径 (run_one / max_tokens / temperature / RPM) 一字不改, 仅换路由。
    ap.add_argument("--base-url", default="https://aimpapi.midea.com/t-aigc/llm-openai-api")
    ap.add_argument("--aigc-user", default="", help="非空则加 AIGC-USER header")
    ap.add_argument("--rpm", type=int, default=20)
    ap.add_argument("--burst", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--only-ids", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("QWEN_CHAT_API_KEY") or os.environ.get("AIMP_KEY")
    if not key:
        print("[rq3b1] BLOCKED: 未设置 QWEN_CHAT_API_KEY / AIMP_KEY", file=sys.stderr)
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
    sys.path.insert(0, SKILLOPT)

    ev = _load_evolver()
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
        print(f"[rq3b1] BLOCKED: 生效 model={serialized['model']} ≠ 请求 {a.model}", file=sys.stderr)
        return 2

    flash_ids = {str(r["id"]) for r in json.load(open(FLASH_VAL_CACHE, encoding="utf-8"))}
    all_val = [it for it in ev.load_split(SPLIT_DIR, "val", truncate=True) if str(it["id"]) in flash_ids]
    only = {s.strip() for s in (a.only_ids or "").split(",") if s.strip()}
    val_items = [it for it in all_val if str(it["id"]) in only] if only else all_val
    s0 = open(S0_PATH, encoding="utf-8").read()

    os.makedirs(a.out_dir, exist_ok=True)
    cache_path = os.path.join(a.out_dir, "val_s0_cache.json")
    log_path = os.path.join(a.out_dir, "calls_val_s0.jsonl")

    print(f"[rq3b1] model={a.model} val_n={len(val_items)} (题集=flash val cache {len(flash_ids)} 题)")
    print(f"[rq3b1] serialized: {json.dumps(serialized, ensure_ascii=False)}")
    if a.dry_run:
        print("[rq3b1] dry-run: 0 请求")
        return 0

    attempts = {"n": 0}
    lock = threading.Lock()
    orig_post = qwen_backend._post_chat_completion

    def _post_counted(payload, timeout, config):
        with lock:
            attempts["n"] += 1
        return orig_post(payload, timeout, config)

    qwen_backend._post_chat_completion = _post_counted

    def _one(it):
        t0 = time.time()
        r = ev.run_one(it, s0, "rq3b1_val_s0")
        usage = qwen_backend.get_token_summary().get("rq3b1_val_s0", {})
        row = {"ts": datetime.now(timezone.utc).isoformat(), "stage": "rq3b1_val_s0",
               "item_id": str(it["id"]), "model": serialized["model"],
               "thinking": serialized["thinking"], token_key: serialized[token_key],
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens"),
               "latency_s": round(time.time() - t0, 2),
               "hard": int(r["hard"]), "pass_rate": r["pass_rate"], "agent_ok": r.get("agent_ok"),
               "fail_reason": str(r.get("fail_reason", ""))[:200]}
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return r

    cache, errs, n = {}, [], 0
    if only and os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
        print(f"[rq3b1] merge bases: 盘上已有 {len(cache)} 题", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, it): str(it["id"]) for it in val_items}
        for fut in as_completed(futs):
            n += 1
            try:
                r = fut.result()
                cache[str(r["id"])] = r
            except Exception as e:  # noqa: BLE001
                errs.append({"id": futs[fut], "err": repr(e)[:300]})
            if n % 10 == 0:
                print(f"[rq3b1] {n}/{len(val_items)} done", flush=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # 与 flash val cache 逐题对位 (0 API)
    flash = {str(r["id"]): r for r in json.load(open(FLASH_VAL_CACHE, encoding="utf-8"))}
    common = sorted(set(cache) & set(flash), key=int)
    both = sum(1 for k in common if int(cache[k]["hard"]) == 1 and int(flash[k]["hard"]) == 1)
    only_us = sum(1 for k in common if int(cache[k]["hard"]) == 1 and int(flash[k]["hard"]) == 0)
    only_flash = sum(1 for k in common if int(cache[k]["hard"]) == 0 and int(flash[k]["hard"]) == 1)
    neither = sum(1 for k in common if int(cache[k]["hard"]) == 0 and int(flash[k]["hard"]) == 0)
    hard_n = sum(int(r["hard"]) for r in cache.values())
    summary = {
        "model": a.model, "serialized_request": serialized,
        "rpm": a.rpm, "burst": a.burst, "split_dir": SPLIT_DIR,
        "val_n": len(cache),
        "hard_pass": hard_n, "fail_pool": len(cache) - hard_n,
        "agent_fail_ids": sorted(k for k, r in cache.items() if not r.get("agent_ok")),
        "only_ids": sorted(only), "errors": errs,
        "vs_flash_val_cache": {"n_common": len(common), "both_pass": both,
                               "only_target_pass": only_us, "only_flash_pass": only_flash,
                               "neither": neither,
                               "flash_hard_pass_all": sum(int(r["hard"]) for r in flash.values()),
                               "note": "flash cache 若为 selection 口径 (median-most-likely) 则与 hard 口径不可直接比"},
        "cache_path": cache_path, "call_log": log_path,
        "attempts_http": attempts["n"],
        "tracker": qwen_backend.get_token_summary().get("rq3b1_val_s0", {}),
    }
    with open(os.path.join(a.out_dir, "val_s0_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
