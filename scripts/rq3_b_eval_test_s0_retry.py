#!/usr/bin/env python3
"""RQ3-B0 补跑器 v2 (串行): 逐题重评 `--ids-file` 中的缺口, 结果 merge 进 `test_s0_cache.json`。

背景 (2026-11-27): 第一遍 (16 workers) 结束时 76/2765 题因 API 侧失败 (5xx/超时/连接/耗尽) 未定稿。
线程池版补跑器 (`rq3_b_eval_test_s0_only.py`) 实测不可用: 少数题的 `run_one` 在内部阻塞且不返回,
既拖住全部后续结果的收集/落盘, 也让并发请求互相拖慢 (单题 1300-1500 s)。因此改为:
  - **串行**执行 (一次一题), 每题在**独立子进程**里跑, 超时直接 kill (默认 600 s), 记 infra 失败并继续
    (v3, 2026-11-27: 实测 SIGALRM 打不断卡在 socket 读上的调用, 第 495 题 600 s 到点后进程无响应)
  - 每题完成即写盘 + 打印, 任何时刻中断都不会丢已完成结果
  - 仅 API 侧失败 (classify_fail != model) 才重试, 最多 `--max-attempts` 次

评测口径不变: 复用主脚本 env 设置 / `run_one` / 调用日志格式 (stage=rq3b0_test_s0_pass2)。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "/data/zhuldz/lunwen/scripts")
import rq3_b_eval_test_s0 as B  # noqa: E402


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _Timeout()


WORKER_FLAG = "RQ3_RETRY_WORKER"


req_model = "deepseek-v4-pro"
req_max_tokens = 16384


def _run_item_worker(it, s0, stage):
    """在**独立子进程**里执行单题评测 (题目/提示/判分口径完全不变)。

    为什么: 实测 SIGALRM 无法中断卡在 C 层 socket 读取/urllib 重试睡眠中的线程 —— 第 495 题
    (item_timeout=600 s) 到点后进程既没报超时也没继续, 心跳停在 10:25:51。子进程 + `kill()` 是
    唯一可靠的单题超时手段。子进程返回 dict, 超时则 kill 并按 infra 失败记录。
    注意: 子进程内 token summary 是独立的, 用量统计仍以主进程(此处)拿到不到 → 记 None。
    """
    import subprocess
    pid = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, json; sys.path.insert(0, '/data/zhuldz/lunwen/scripts'); sys.stdout.reconfigure(line_buffering=True);"
         "import rq3_b_eval_test_s0_retry as R; R.worker_main()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True,
        env={**os.environ, WORKER_FLAG: "1"}, start_new_session=True,
    )
    try:
        out, err = pid.communicate(json.dumps({"item": it, "s0": s0, "stage": stage,
                                               "model": req_model, "max_tokens": req_max_tokens}),
                                   timeout=WORKER_ITEM_TIMEOUT)
    except subprocess.TimeoutExpired:
        # kill 掉整个进程组: run_one 内部的沙箱/子进程不会残留
        try:
            os.killpg(pid.pid, signal.SIGKILL)
        except ProcessLookupError:
            pid.kill()
        pid.communicate()
        return {"id": str(it["id"]), "hard": 0, "pass_rate": -3.0, "agent_ok": False,
                "fail_reason": f"item_timeout>{WORKER_ITEM_TIMEOUT}s(process_killed)"}
    last = [l for l in out.strip().splitlines() if l.startswith("{")]
    if not last:
        return {"id": str(it["id"]), "hard": 0, "pass_rate": -3.0, "agent_ok": False,
                "fail_reason": "worker_no_output"}
    return json.loads(last[-1])


WORKER_ITEM_TIMEOUT = 600


def worker_main():
    """子进程入口: 读 stdin 的 {item, s0, stage, max_tokens}, 复用与主脚本相同的 env/TARGET_CONFIG 后跑 run_one。

    子进程内**无限速** (`QWEN_CHAT_MAX_RPM=0`): 父进程串行提交、按 60/rpm 节拍发车, 等效并发 ≤1、速率 ≤rpm。
    这样既不叠加两套令牌桶, 也不让 fork 出来的子进程继承父进程锁状态。
    """
    req = json.loads(sys.stdin.read())
    apply_env(req["model"], req["max_tokens"], rpm=0)
    ev = B._load_evolver()
    from skillopt.model import qwen_backend  # noqa: F401,E402
    qwen_backend.TARGET_CONFIG = qwen_backend._initial_config("target")
    r = ev.run_one(req["item"], req["s0"], req["stage"])
    print(json.dumps(r, ensure_ascii=False))


def apply_env(model, max_tokens, rpm=20):
    """与 `rq3_b_eval_test_s0.py` 第 92-109 行的设置逐项一致 (审计 C 组核对的 12 项口径)。"""
    key = os.environ.get("QWEN_CHAT_API_KEY") or os.environ.get("AIMP_KEY")
    if not key:
        raise SystemExit("[retry] BLOCKED: 未设置 QWEN_CHAT_API_KEY")
    os.environ["QWEN_CHAT_API_KEY"] = key
    os.environ["QWEN_CHAT_BASE_URL"] = "https://aimpapi.midea.com/t-aigc/llm-openai-api"
    os.environ["QWEN_CHAT_MODEL"] = model
    os.environ.pop("QWEN_CHAT_AIGC_USER", None)
    os.environ["QWEN_CHAT_MAX_TOKENS"] = str(max_tokens)
    os.environ["QWEN_CHAT_TEMPERATURE"] = "0.0"
    os.environ["QWEN_CHAT_ENABLE_THINKING"] = "false"
    # RPM 限流放在父进程 (串行 + 按 60/rpm 节拍发车); 子进程 rpm=0 不重复限流
    os.environ["QWEN_CHAT_MAX_RPM"] = str(rpm)
    os.environ["QWEN_CHAT_BURST"] = "2"
    os.environ["TARGET_BACKEND"] = "qwen_chat"
    os.environ["OPTIMIZER_BACKEND"] = "qwen_chat"
    sys.path.insert(0, B.SKILLOPT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--out-dir", default="/data/zhuldz/lunwen/outputs/rq3/deepseek-v4-pro")
    ap.add_argument("--ids-file", default="/tmp/rq3_retry76.txt")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--item-timeout", type=int, default=1200)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--rpm", type=int, default=20)
    # 预注册的尝试计划: 预先声明每一轮的超时/次数, 落盘后再执行 (避免看到结果回调参数)。
    # {"stage": "...", "attempts": [{"item_timeout": 900, "max_attempts": 1}, ...]}
    ap.add_argument("--plan", default="")
    ap.add_argument("--stage", default="rq3b0_test_s0_pass2")
    a = ap.parse_args()
    plan = None
    if a.plan:
        plan = json.load(open(a.plan, encoding="utf-8"))
        if plan.get("stage") != a.stage:
            print("[retry] BLOCKED: plan.stage ≠ --stage", file=sys.stderr)
            return 2
        if not plan.get("attempts"):
            print("[retry] BLOCKED: plan.attempts 为空", file=sys.stderr)
            return 2

    apply_env(a.model, a.max_tokens, rpm=a.rpm)
    global req_model, req_max_tokens
    req_model, req_max_tokens = a.model, a.max_tokens

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
        print(f"[retry] BLOCKED: 生效 model≠{a.model}", file=sys.stderr)
        return 2

    ids = [s.strip() for s in open(a.ids_file).read().replace("\n", ",").split(",") if s.strip()]
    test_items = ev.load_split(B.SPLIT_DIR, "test", truncate=True)
    item_by_id = {str(it["id"]): it for it in test_items}
    missing_from_split = [i for i in ids if i not in item_by_id]
    if missing_from_split:
        print(f"[retry] BLOCKED: {len(missing_from_split)} 个 id 不在 test split", file=sys.stderr)
        return 2
    s0 = open(B.S0_PATH, encoding="utf-8").read()

    cache_path = os.path.join(a.out_dir, "test_s0_cache.json")
    log_path = os.path.join(a.out_dir, "calls_test_s0.jsonl")
    cache = json.load(open(cache_path, encoding="utf-8"))
    todo = [i for i in ids if i not in cache]
    print(f"[retry] model={a.model} ids={len(ids)} todo={len(todo)} serial", flush=True)
    print(f"[retry] serialized: {json.dumps(serialized, ensure_ascii=False)}", flush=True)

    final_fail = []
    global WORKER_ITEM_TIMEOUT               # 逐轮覆写: 子进程超时按计划走
    interval = 60.0 / max(1, a.rpm)   # 发车节拍: 串行执行下等效速率 ≤ rpm, 并发恒为 1
    emit = 0.0
    for n, qid in enumerate(todo, 1):
        it = item_by_id[qid]
        r = None
        wait = emit - time.monotonic()          # 发车节流 (含重试/切换题目)
        if wait > 0:
            time.sleep(wait)
        emit = max(time.monotonic(), emit) + interval
        # 尝试计划: 有 --plan 则按计划逐轮 (轮内 max_attempts 次, 每次 item_timeout); 否则用命令行参数
        rounds = ([(rd.get("item_timeout", a.item_timeout), rd.get("max_attempts", 1))
                   for rd in plan["attempts"]] if plan else [(a.item_timeout, a.max_attempts)])
        attempt = 0
        for round_i, (r_timeout, r_max) in enumerate(rounds, 1):
            WORKER_ITEM_TIMEOUT = r_timeout
            for slot in range(1, r_max + 1):
                attempt += 1
                t0 = time.time()
                # 心跳: 内层 chat_target 有 retries=5, 单题最坏可挂很久; 先打印再调用, 观察进程是否活着
                print(f"[retry] start {n}/{len(todo)} id={qid} round={round_i} slot={slot} "
                      f"(item_timeout={r_timeout}s) {datetime.now().strftime('%H:%M:%S')}", flush=True)
                try:
                    r = _run_item_worker(it, s0, a.stage)
                except Exception as e:  # noqa: BLE001
                    r = {"id": qid, "hard": 0, "pass_rate": -3.0, "agent_ok": False,
                         "fail_reason": f"exception: {e!r}"[:300]}
                row = {"ts": datetime.now(timezone.utc).isoformat(), "stage": a.stage,
                       "item_id": qid, "attempt": attempt, "round": round_i, "slot": slot,
                       "item_timeout": r_timeout, "model": serialized["model"],
                       "prompt_tokens": None, "completion_tokens": None,
                       "latency_s": round(time.time() - t0, 2),
                       "hard": int(r["hard"]), "pass_rate": r["pass_rate"], "agent_ok": r.get("agent_ok"),
                       "fail_reason": str(r.get("fail_reason", ""))[:200]}
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                # 调用层失败 (5xx/超时/连接) 一律按 infra 重试, 即使 run_one 已吞成 agent_ok=False 的占位结果;
                # 只有确定性判分失败 (classify_fail == "model") 才不重试。否则 72 题会单次 shell 落盘成 final。
                # "model" = 有产出但判分不过 (确定性); "infra" = 调用层失败 → 仍需重试
                kind = "model" if r.get("agent_ok") else "infra"
                print(f"[retry] {n}/{len(todo)} id={qid} attempt={attempt} {row['latency_s']}s "
                      f"hard={r['hard']} ok={r.get('agent_ok')} kind={kind}", flush=True)
                if r.get("agent_ok") or kind == "model":
                    break
            if r.get("agent_ok"):   # 提前收工: 不进入下一轮
                break
        cache[qid] = r
        if not r.get("agent_ok"):
            final_fail.append(qid)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    split_ids = {str(it["id"]) for it in test_items}
    missing = sorted(split_ids - set(cache), key=int)
    hard_n = sum(int(r["hard"]) for r in cache.values())
    out = {"test_n_split": len(test_items), "test_n_cache": len(cache),
           "coverage_complete": not missing, "n_missing": len(missing), "missing_ids": missing[:50],
           "hard_pass": hard_n, "fail_pool": len(cache) - hard_n,
           "hard_rate": hard_n / max(1, len(cache)),
           "agent_fail_ids": sorted(k for k, r in cache.items() if not r.get("agent_ok")),
           "retry_todo": len(todo), "retry_final_fail": final_fail,
           "plan": plan, "plan_path": a.plan or None, "stage": a.stage,
           "attempts_http": None,
           "attempts_http_note": "子进程运行, 主进程拿不到 qwen_backend 计数; 以 calls_test_s0.jsonl 中 "
                                 "stage=rq3b0_test_s0_pass2 的行数计"}
    json.dump(out, open(os.path.join(a.out_dir, "test_s0_pass2_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return 0 if not missing and not out["agent_fail_ids"] else 4


if __name__ == "__main__":
    sys.exit(main())
