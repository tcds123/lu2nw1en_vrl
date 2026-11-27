#!/usr/bin/env python3
"""RQ3-A: 目标模型在 train[:80] 上的 S₀ 基线 (失败池来源) + 逐调用日志。

复用生产代码 `tf_skill_optimizer_appsplus.py` 的 `load_split` / `run_one` / 评测契约,
**不修改该文件**。产出与生产 `train_s0_cache.json` 同构, 可直接作为 RQ3-B 的 `--s0_cache_train`,
失败池 = hard==0 题。

每次调用落一条 JSONL (stage / serialized model / max_tokens / thinking / usage / latency),
弥补 `qwen_backend._chat` 不记 attempt 的盲区 (进程内 retry 仍只能"未观测")。

用法:
    set -a; source configs/.env; set +a
    python3 scripts/rq3_a_s0_train.py --model deepseek-v4-pro --out-dir outputs/rq3/deepseek-v4-pro
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

_log_lock = threading.Lock()


def _load_evolver():
    """import 生产模块 (只读); 其 import 期只做 setdefault, 不触网。"""
    spec = importlib.util.spec_from_file_location("tf_evolver", EVOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--out-dir", default="outputs/rq3/deepseek-v4-pro")
    ap.add_argument("--train-n", type=int, default=80)
    # 端点覆盖 (默认 = pro 的 OpenAI 兼容端点, 行为不变)。GLM-5.2 走 AIMP 专有端点 aimp-glm/v1;
    # 评测口径 (run_one / max_tokens / temperature / RPM) 一字不改, 仅换路由。
    ap.add_argument("--base-url", default="https://aimpapi.midea.com/t-aigc/llm-openai-api")
    ap.add_argument("--aigc-user", default="", help="非空则加 AIGC-USER header")
    ap.add_argument("--rpm", type=int, default=20, help="≤30 (用户约束); 与 burst 共同封顶")
    ap.add_argument("--burst", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--only-ids", default="", help="逗号分隔; 只重算这些题, merge 进已有 cache")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划, 发 0 次请求")
    a = ap.parse_args()

    key = os.environ.get("QWEN_CHAT_API_KEY") or os.environ.get("AIMP_KEY")
    if not key:
        print("[rq3a] BLOCKED: 未设置 QWEN_CHAT_API_KEY / AIMP_KEY", file=sys.stderr)
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

    # attempt 计数: 包一层 _post_chat_completion (进程内, 不改只读源码)。
    # 每次真正发出的 HTTP 请求 +1; `_chat_messages_impl` 只在失败时才发下一次,
    # 故 item 的 attempts = 该 item 两次取号之差。
    _attempts = {"n": 0}
    _attempt_lock = threading.Lock()
    _orig_post = qwen_backend._post_chat_completion

    def _post_counted(payload, timeout, config):
        with _attempt_lock:
            _attempts["n"] += 1
        return _orig_post(payload, timeout, config)

    qwen_backend._post_chat_completion = _post_counted

    cfg = qwen_backend.TARGET_CONFIG
    token_key = "max_completion_tokens" if cfg.use_max_completion_tokens else "max_tokens"
    serialized = {
        "model": cfg.deployment,
        "base_url": cfg.base_url,
        token_key: min(a.max_tokens, cfg.max_tokens),
        "stream": False,
        "temperature": cfg.temperature,
        "thinking": {"chat_template_kwargs": {"enable_thinking": True}} if cfg.enable_thinking else "server_default",
    }
    if serialized["model"] != a.model:
        print(f"[rq3a] BLOCKED: 生效 model={serialized['model']} ≠ 请求 {a.model}", file=sys.stderr)
        return 2

    # 续跑: --only-ids 只重算指定题, 结果 merge 进已有 cache (其余题不发请求)。
    all_train = ev.load_split(SPLIT_DIR, "train", truncate=True)[: a.train_n]
    only = {s.strip() for s in (a.only_ids or "").split(",") if s.strip()}
    train_items = [it for it in all_train if str(it["id"]) in only] if only else all_train
    s0 = open(S0_PATH, encoding="utf-8").read()
    os.makedirs(a.out_dir, exist_ok=True)
    cache_path = os.path.join(a.out_dir, "train_s0_cache.json")
    log_path = os.path.join(a.out_dir, "calls_s0.jsonl")

    print(f"[rq3a] model={a.model} train_n={len(train_items)} rpm={a.rpm} burst={a.burst} workers={a.workers}")
    print(f"[rq3a] serialized: {json.dumps(serialized, ensure_ascii=False)}")
    print(f"[rq3a] out: {a.out_dir}")
    if a.dry_run:
        print("[rq3a] dry-run: 0 请求")
        return 0

    def _one(it):
        t0 = time.time()
        with _attempt_lock:
            a0 = _attempts["n"]
        r = ev.run_one(it, s0, "rq3a_s0")
        with _attempt_lock:
            a1 = _attempts["n"]
        usage = qwen_backend.get_token_summary().get("rq3a_s0", {})  # calls/tokens 为累计值
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": "rq3a_s0",
            "item_id": str(it["id"]),
            "model": serialized["model"],
            "thinking": serialized["thinking"],
            token_key: serialized[token_key],
            # 进程内计数 → 并发下 item 间可能串号, 合计才是准数 (见 summary.attempts)
            "attempts_item_upper": a1 - a0,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_s": round(time.time() - t0, 2),
            "hard": int(r["hard"]),
            "pass_rate": r["pass_rate"],
            "agent_ok": r.get("agent_ok"),
        }
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return r

    cache, errs, n = {}, [], 0
    if only and os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
        print(f"[rq3a] merge bases: 盘上已有 {len(cache)} 题", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, it): str(it["id"]) for it in train_items}
        for fut in as_completed(futs):
            n += 1
            try:
                r = fut.result()
                cache[str(r["id"])] = r
            except Exception as e:  # noqa: BLE001
                errs.append({"id": futs[fut], "err": repr(e)[:300]})
            if n % 10 == 0:
                print(f"[rq3a] {n}/{len(train_items)} done", flush=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    hard_n = sum(int(r["hard"]) for r in cache.values())
    fail_ids = sorted(int(k) for k, r in cache.items() if int(r["hard"]) == 0)
    infra_fail = sorted(k for k, r in cache.items() if not r.get("agent_ok"))
    summary = {
        "model": a.model,
        "serialized_request": serialized,
        "rpm": a.rpm, "burst": a.burst,
        "split_dir": SPLIT_DIR, "train_n": len(train_items),
        "n_result": len(cache), "hard_pass": hard_n, "fail_pool": len(cache) - hard_n,
        "fail_ids": fail_ids,
        "infra_fail_ids": infra_fail,  # agent_ok=False: 5 次重发后仍失败, 需重试或剔除
        "only_ids": sorted(only),
        "errors": errs,
        "cache_path": cache_path, "call_log": log_path,
        "attempts_http": _attempts["n"],
        "tracker": qwen_backend.get_token_summary().get("rq3a_s0", {}),
    }
    with open(os.path.join(a.out_dir, "s0_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "fail_ids"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
