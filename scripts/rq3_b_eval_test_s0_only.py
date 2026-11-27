#!/usr/bin/env python3
"""RQ3-B0 补跑器: 只评测 `--only-ids` 给出的题目, 结果 **合并** 进已有 `test_s0_cache.json`。

为什么单独写 (2026-11-27): `rq3_b_eval_test_s0.py` 的 resume 分支 `todo = items if id not in cache`,
与 `--only-ids` 组合时若盘上 cache 已覆盖绝大多数 id, todo 为空 → 0 请求直接退出 (第二遍实测如此)。
本脚本不改评测口径, 直接复用主脚本的 env 设置 / `run_one` / 调用日志格式, 只把调度范围限定为
缺口的 id 集, 并在结束时把结果 merge 回同一个 cache 文件 (其余 id 原样保留)。

用法:
    RQ3_ONLY_IDS=123,456 python3 -u scripts/rq3_b_eval_test_s0_only.py --model deepseek-v4-pro --workers 4
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
import rq3_b_eval_test_s0 as B  # noqa: E402  复用 env 口径与 classify_fail

_log_lock = threading.Lock()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--out-dir", default="/data/zhuldz/lunwen/outputs/rq3/deepseek-v4-pro")
    ap.add_argument("--ids-file", default="/tmp/rq3_resume_ids.txt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=int, default=20)
    ap.add_argument("--burst", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16384)
    a = ap.parse_args()

    key = os.environ.get("QWEN_CHAT_API_KEY") or os.environ.get("AIMP_KEY")
    if not key:
        print("[only] BLOCKED: 未设置 QWEN_CHAT_API_KEY", file=sys.stderr)
        return 2
    os.environ["QWEN_CHAT_API_KEY"] = key
    os.environ["QWEN_CHAT_BASE_URL"] = "https://aimpapi.midea.com/t-aigc/llm-openai-api"
    os.environ["QWEN_CHAT_MODEL"] = a.model
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
        print(f"[only] BLOCKED: 生效 model≠{a.model}", file=sys.stderr)
        return 2

    ids = {s.strip() for s in open(a.ids_file).read().split(",") if s.strip()}
    test_items = ev.load_split(B.SPLIT_DIR, "test", truncate=True)
    items = [it for it in test_items if str(it["id"]) in ids]
    missing_from_split = ids - {str(it["id"]) for it in items}
    if missing_from_split:
        print(f"[only] BLOCKED: {len(missing_from_split)} 个 id 不在 test split 中", file=sys.stderr)
        return 2
    s0 = open(B.S0_PATH, encoding="utf-8").read()

    cache_path = os.path.join(a.out_dir, "test_s0_cache.json")
    log_path = os.path.join(a.out_dir, "calls_test_s0.jsonl")
    cache = json.load(open(cache_path, encoding="utf-8"))
    todo = [it for it in items if str(it["id"]) not in cache]
    print(f"[only] model={a.model} only_ids={len(ids)} todo={len(todo)} workers={a.workers}", flush=True)
    print(f"[only] serialized: {json.dumps(serialized, ensure_ascii=False)}", flush=True)

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
        r = ev.run_one(it, s0, "rq3b0_test_s0_pass2")
        usage = qwen_backend.get_token_summary().get("rq3b0_test_s0_pass2", {})
        row = {"ts": datetime.now(timezone.utc).isoformat(), "stage": "rq3b0_test_s0_pass2",
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

    # 第二遍: 失败一律最多补跑 2 次, 调度上限 workers (题量小, 不需要主脚本的补跑不限并发)
    # 兼容性: 部分 id 的 run_one 在内部卡住时不会返回, 线程永久阻塞 → 后面的结果没人收、不落盘。
    # 因此用 timeout 收集 + 提交节流 (见下), 且定期把已完成的条目写盘。
    resub = {}
    done_n = 0
    item_by_id = {str(it["id"]): it for it in todo}
    emit = 0.0   # 下次提交时间 (RPM/burst 节流, 兜底防突发; backend 自身也有令牌桶)
    interval = (60.0 / a.rpm) if a.rpm > 0 else 0.0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {}
        pending = list(todo)
        while pending or futs:
            while pending and len(futs) < a.workers * 2:
                now = time.monotonic()
                if emit > now:
                    time.sleep(min(emit - now, 1.0))
                    continue
                emit = max(now, emit) + interval / max(1, a.burst)
                it = pending.pop(0)
                futs[ex.submit(_one, it)] = str(it["id"])
            try:
                fin = next(as_completed(list(futs), timeout=15))
            except TimeoutError:      # 本轮无新完成; 继续提交/等待 (卡住的线程不能拖住其它结果)
                continue
            qid = futs.pop(fin)
            try:
                r = fin.result()
            except Exception as e:  # noqa: BLE001
                r = {"id": qid, "hard": 0, "pass_rate": -3.0, "agent_ok": False,
                     "fail_reason": f"exception: {e!r}"[:300]}
            kind = "model" if r.get("agent_ok") else B.classify_fail(r.get("fail_reason", ""))
            if not r.get("agent_ok") and kind != "model" and resub.get(qid, 0) < 2:
                resub[qid] = resub.get(qid, 0) + 1
                pending.append(item_by_id[qid])
            else:
                cache[qid] = r
                if not r.get("agent_ok"):
                    resub.setdefault("_final_fail", []).append(qid)
            done_n += 1
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"[only] {done_n}/{len(todo)} done (cache={len(cache)}, http={attempts['n']}, "
                  f"inflight={len(futs)})", flush=True)

    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    split_ids = {str(it["id"]) for it in test_items}
    missing = sorted(split_ids - set(cache), key=int)
    hard_n = sum(int(r["hard"]) for r in cache.values())
    out = {"test_n_split": len(test_items), "test_n_cache": len(cache),
           "coverage_complete": not missing, "n_missing": len(missing), "missing_ids": missing[:50],
           "hard_pass": hard_n, "fail_pool": len(cache) - hard_n,
           "hard_rate": hard_n / max(1, len(cache)),
           "agent_fail_ids": sorted(k for k, r in cache.items() if not r.get("agent_ok")),
           "attempts_http": attempts["n"], "pass2_ids": len(ids), "pass2_resubmits": len(resub) - (1 if "_final_fail" in resub else 0)}
    json.dump(out, open(os.path.join(a.out_dir, "test_s0_pass2_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return 0 if not missing and not out["agent_fail_ids"] else 4


if __name__ == "__main__":
    sys.exit(main())
