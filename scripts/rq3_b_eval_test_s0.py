#!/usr/bin/env python3
"""RQ3-B0: 目标模型在**全量 test split** (2765 题) 上的 S₀ 评测 → `outputs/rq3/<model>/test_s0_cache.json`。

为什么需要 (Issue #122): RQ3-B launcher 的 `--s0_cache_test` 曾误指 val cache, 导致最终 test 段的
McNemar / base_hard / routed / retention 全部在"test 题 × val id"空交集上计算 (静默产出 p=1.0, g=0)。
本脚本产出与 test split **同 id 集**的 S₀ 逐题结果, 作为 RQ3-C 配对的唯一合法基线。

口径与 `scripts/rq3_b_eval_val.py` / `rq3_a_s0_train.py` 完全一致 (同一 backend 参数, 同一 run_one)。
输出:
  <out-dir>/test_s0_cache.json   {id: run_one 结果 dict, 含 response}   ← 配对基线 + evolver 的 test cache
  <out-dir>/calls_test_s0.jsonl  逐调用日志 (usage / latency / attempt)
  <out-dir>/test_s0_summary.json 通过数 / 失败池 / agent_fail 列表 / 对照信息

注意: 本模型 test S₀ 的 hard=0 题数即 Arm1 的"可救池"上界; 与 flash 的 test 数字不可直接比
(backend/thinking 口径不同, 见 Issue #121)。

用法 (需 QWEN_CHAT_API_KEY 或 AIMP_KEY 已在环境中):
    RQ3_MODEL=deepseek-v4-pro python3 -u scripts/rq3_b_eval_test_s0.py --workers 8 --rpm 20
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
    spec = importlib.util.spec_from_file_location("tf_evolver", EVOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify_fail(fail_reason: str) -> str:
    """区分 API 基础设施失败 (可重试) 与真实的代码/评测失败 (不可重试)。

    只有 API 侧错误算"不健康": HTTP 429 / 5xx / 超时 / 连接断开 / 重试耗尽。
    NameError / stdout mismatch / 编译错误等是模型的真实回答缺陷, 属有效数据, 不重试。
    """
    s = (fail_reason or "").lower()
    if "429" in s or "rate limit" in s or "too many requests" in s:
        return "429"
    if "http 5" in s or "internalservererror" in s or "api_error" in s:
        return "5xx"
    if "timed out" in s or "timeout" in s:
        return "timeout"
    if "remote end closed" in s or "connection" in s or "urlerror" in s:
        return "conn"
    if "qwen chat call failed after" in s:
        return "api_exhausted"
    return "model"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("RQ3_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--rpm", type=int, default=20)
    ap.add_argument("--burst", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    # 降级保护 (用户批准方案 2): 健康则保持 --workers, 连续不健康则逐级降 16→12→8,
    # 单进程内生效, 不重启 → 不重复调用。RPM 令牌桶在进程级共享, 与线程数无关。
    ap.add_argument("--min-workers", type=int, default=8)
    ap.add_argument("--degrade-after-bad", type=int, default=25,
                    help="连续 N 次补跑失败 (非 429) 触发降级")
    ap.add_argument("--pause-after-bad", type=int, default=80,
                    help="连续 N 次补跑失败 → 熔断暂停并落盘 resume")
    ap.add_argument("--health-window", type=int, default=40,
                    help="健康告警: 每 N 次补跑统计 429/5xx/超时占比")
    ap.add_argument("--pause-on-alert", action="store_true",
                    help="健康告警且连续失败时直接熔断 (默认只告警)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--only-ids", default="")
    # 端点覆盖 (默认 = pro 的 OpenAI 兼容端点, 行为不变)。GLM-5.2 走 AIMP 专有端点 `aimp-glm/v1` 且
    # 需 AIGC-USER header (configs/models_aimp.local.yaml 的 api 字段); 评测口径 (run_one / max_tokens /
    # temperature / RPM) 一字不改, 仅换路由。
    ap.add_argument("--base-url", default="https://aimpapi.midea.com/t-aigc/llm-openai-api")
    ap.add_argument("--aigc-user", default="", help="非空则加 AIGC-USER header")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.out_dir:
        a.out_dir = os.path.join("/data/zhuldz/lunwen/outputs/rq3", a.model)

    key = os.environ.get("QWEN_CHAT_API_KEY") or os.environ.get("AIMP_KEY")
    if not key:
        print("[rq3b0] BLOCKED: 未设置 QWEN_CHAT_API_KEY / AIMP_KEY", file=sys.stderr)
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
        print(f"[rq3b0] BLOCKED: 生效 model={serialized['model']} ≠ 请求 {a.model}", file=sys.stderr)
        return 2

    test_items = ev.load_split(SPLIT_DIR, "test", truncate=True)
    only = {s.strip() for s in (a.only_ids or "").split(",") if s.strip()}
    items = [it for it in test_items if str(it["id"]) in only] if only else test_items
    s0 = open(S0_PATH, encoding="utf-8").read()

    os.makedirs(a.out_dir, exist_ok=True)
    cache_path = os.path.join(a.out_dir, "test_s0_cache.json")
    log_path = os.path.join(a.out_dir, "calls_test_s0.jsonl")

    print(f"[rq3b0] model={a.model} test_n={len(items)}/{len(test_items)}")
    print(f"[rq3b0] serialized: {json.dumps(serialized, ensure_ascii=False)}")
    print(f"[rq3b0] out: {cache_path}")
    if a.dry_run:
        print("[rq3b0] dry-run: 0 请求")
        return 0

    attempts = {"n": 0, "429": 0, "5xx": 0, "timeout": 0, "conn": 0}
    lock = threading.Lock()
    orig_post = qwen_backend._post_chat_completion

    def _post_counted(payload, timeout, config):
        # 每次 HTTP 尝试单独计数 (含 429 backoff/5 次重试), 供健康监控与 summary 使用
        with lock:
            attempts["n"] += 1
        try:
            return orig_post(payload, timeout, config)
        except Exception as e:  # noqa: BLE001
            k = classify_fail(repr(e))
            key = {"429": "429", "5xx": "5xx", "timeout": "timeout", "conn": "conn"}.get(k)
            if key:
                with lock:
                    attempts[key] += 1
            raise

    qwen_backend._post_chat_completion = _post_counted

    def _one(it):
        t0 = time.time()
        r = ev.run_one(it, s0, "rq3b0_test_s0")
        usage = qwen_backend.get_token_summary().get("rq3b0_test_s0", {})
        row = {"ts": datetime.now(timezone.utc).isoformat(), "stage": "rq3b0_test_s0",
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

    # 断点续跑: 盘上已有 cache 则跳过已完成 id (与 --only-ids 组合可分批补缺)
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
        print(f"[rq3b0] resume: 盘上已有 {len(cache)} 题", flush=True)
    todo = [it for it in items if str(it["id"]) not in cache]

    # ── 调度 ──
    # - 单写者: 所有 cache 写入都在主线程 (as_completed 循环内), summary 也由主线程落盘
    # - 每题只分发一次: 每个 id 只有在首轮或 (API 失败后的) 补跑轮被提交, 补跑后不再重复
    # - 补跑立即提交 (不等整轮), 避免单个超时题拖满尾延迟; 补跑**不受 workers 上限约束**,
    #   否则慢请求会永久占住槽位把吞吐压到远低于 20 RPM
    item_by_id = {str(it["id"]): it for it in items}
    inflight = {}                     # id -> future (未完成)
    resubmits = {}                    # id -> 已补跑次数
    state = {"workers": a.workers}
    errs, n_done, n_submit = [], 0, 0
    cnt = {"429": 0, "5xx": 0, "timeout": 0, "conn": 0, "api_exhausted": 0, "model": 0, "other": 0}
    streak_bad, since_alert = 0, 0
    last_ts = [time.monotonic()]
    paused = {"v": False}
    # 进度落盘: 盘上 cache 只放"已定稿"条目 (补跑中的 id 在 resubmits 里 → dump 时剔除,
    # resume 时会被重新提交), 补跑链在内存 cache 里只保留最新结果。
    def finalize():
        pend = set(resubmits) & set(cache)
        if pend:
            return {k: v for k, v in cache.items() if k not in pend}
        return cache

    def dump():
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(finalize(), f, ensure_ascii=False)

    def submit(out, it, pool):
        nonlocal n_submit
        n_submit += 1
        out[str(it["id"])] = pool.submit(_one, it)

    ex = ThreadPoolExecutor(max_workers=state["workers"])
    for it in todo:
        submit(inflight, it, ex)

    try:
        while inflight:
            done = next(as_completed(list(inflight.values())))
            qid = next(k for k, f in inflight.items() if f is done)
            del inflight[qid]   # 键 = item id (不是 id(obj), 后者可能被复用)
            n_done += 1
            try:
                r = done.result()
            except Exception as e:  # noqa: BLE001
                r = {"id": qid, "hard": 0, "pass_rate": -3.0, "agent_ok": False,
                     "fail_reason": f"exception: {e!r}"[:300]}
            cache[qid] = r
            since_alert += 1

            kind = "model" if r.get("agent_ok") else classify_fail(r.get("fail_reason", ""))
            cnt[kind] += 1

            if not r.get("agent_ok") and kind != "model":
                # API 失败 → 立即补跑 (最多 3 次, 同一 id 不会并发重复)
                k = resubmits.get(qid, 0)
                if k < 3 and not paused["v"]:
                    resubmits[qid] = k + 1
                    del cache[qid]
                    submit(inflight, item_by_id[qid], ex)   # 补跑不受 workers 限制
                streak_bad += 1
            else:
                streak_bad = 0

            # ── 降级保护: 连续补跑仍失败 → 降 workers (进程内, 不重启) ──
            if streak_bad and streak_bad % a.degrade_after_bad == 0 and state["workers"] > a.min_workers:
                new_w = max(a.min_workers, round(state["workers"] * 0.75))
                print(f"[rq3b0] ⚠️ DEGRADE {state['workers']}→{new_w} workers "
                      f"(连续 {streak_bad} 次补跑失败; 已在跑的请求不受影响)", flush=True)
                state["workers"] = new_w
                ex._max_workers = new_w   # 只在旧线程退出后产生新线程
                streak_bad = 0

            if streak_bad >= a.pause_after_bad:
                print(f"[rq3b0] 🛑 CIRCUIT BREAK: 连续 {streak_bad} 次补跑失败, 暂停并落盘",
                      flush=True)
                paused["v"] = True
                dump()

            # ── 健康告警 ──
            if since_alert >= a.health_window:
                api_bad = cnt["429"] + cnt["5xx"] + cnt["timeout"] + cnt["conn"] + cnt["api_exhausted"]
                rate = api_bad / max(1, sum(cnt.values()))
                now = time.monotonic()
                dt = (now - last_ts[0]) / 60.0
                print(f"[rq3b0] HEALTH {sum(cnt.values()) - api_bad}/{sum(cnt.values())} ok, "
                      f"api_err={rate:.1%} {dict(cnt)}, throughput={since_alert/max(dt,1e-6):.1f}/min, "
                      f"workers={state['workers']}, inflight={len(inflight)}", flush=True)
                if rate > 0.10 and (a.pause_on_alert or streak_bad >= 10):
                    print("[rq3b0] 🛑 HEALTH ALERT → 暂停并落盘 (处理完再 resume)", flush=True)
                    paused["v"] = True
                    dump()
                cnt.update({k: 0 for k in cnt})
                since_alert, last_ts[0] = 0, now

            if n_done % 5 == 0:   # 每 5 题落盘 (进度可观测; 2765 题文件 ~13MB/次, 开销可忽略)
                print(f"[rq3b0] {len(finalize())}/{len(items)} final (in-mem={len(cache)}, "
                      f"submitted={n_submit}, resubmits={sum(resubmits.values())}, "
                      f"inflight={len(inflight)}, workers={state['workers']})", flush=True)
                dump()
    finally:
        ex.shutdown(wait=True)
    dump()
    in_mem_only = len(set(cache) - set(finalize()))
    if in_mem_only:
        print(f"[rq3b0] ⚠️ {in_mem_only} 题补跑耗尽仍失败, 未写入盘上 cache (resume 会重跑)", flush=True)

    split_ids = {str(it["id"]) for it in test_items}
    missing = sorted(split_ids - set(cache), key=int)
    hard_n = sum(int(r["hard"]) for r in cache.values())
    summary = {
        "model": a.model, "serialized_request": serialized, "rpm": a.rpm, "burst": a.burst,
        "workers_start": a.workers, "workers_final": state["workers"], "min_workers": a.min_workers,
        "n_submitted": n_submit, "n_resubmitted": sum(resubmits.values()),
        "n_resubmit_pending": len(set(resubmits) & set(cache)),
        "resubmit_ids": sorted(k for k, v in resubmits.items() if v),
        "final_fail_by_kind": {k: v for k, v in cnt.items() if v},  # 仅最后一轮, 窗口内清零
        "http_errors": dict(attempts), "circuit_broken": paused["v"],
        "split_dir": SPLIT_DIR, "test_n_split": len(test_items), "test_n_cache": len(cache),
        "coverage_complete": not missing, "missing_ids": missing[:50], "n_missing": len(missing),
        "hard_pass": hard_n, "fail_pool": len(cache) - hard_n,
        "hard_rate": hard_n / max(1, len(cache)),
        "agent_fail_ids": sorted(k for k, r in cache.items() if not r.get("agent_ok")),
        "errors": errs, "attempts_http": attempts["n"],
        "tracker": qwen_backend.get_token_summary().get("rq3b0_test_s0", {}),
        "cache_path": cache_path, "call_log": log_path,
        "note": "Issue #122: 本文件是 RQ3-C 配对的唯一合法 S₀ 基线; 不得用 val cache / D5 cache 替代",
    }
    with open(os.path.join(a.out_dir, "test_s0_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in ("test_n_split", "test_n_cache", "coverage_complete",
                                              "n_missing", "hard_pass", "fail_pool", "hard_rate",
                                              "agent_fail_ids", "attempts_http")},
                     ensure_ascii=False, indent=2))
    return 0 if not missing else 4


if __name__ == "__main__":
    sys.exit(main())
