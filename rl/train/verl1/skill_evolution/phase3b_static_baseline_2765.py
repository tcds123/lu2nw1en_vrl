#!/usr/bin/env python3
"""Phase 3b: 补 APPS Plus 2765 题静态基线 (S₀ / SkillOpt), 独立非路由 + D5 评测.

协议: docs/phase3b_static_baseline_2765_protocol.md (2026-11-20, 首次 API 调用前冻结).
  - 全部 2765 ID; 每题独立生成 (run_one 复刻: _build_system/_build_user → chat_target → _evaluate_d5);
  - 两组请求按 ticket 交替提交到同一 8-worker 池 (共享 20 RPM token bucket);
  - 每块 100 题落盘 chunk_XXXX.json → 断点续跑;
  - 逐请求 trace (<out>/trace/*.jsonl) + 每题 usage/attempts/finish_reason/latency。

用法:
  python3 phase3b_static_baseline_2765.py --drift-check          # 8 题漂移检查 (并入 S₀ chunk0 记录)
  python3 phase3b_static_baseline_2765.py                        # 全量 2765 x {S₀, SkillOpt}
  python3 phase3b_static_baseline_2765.py --finalize             # 合并 chunk → all_results/run_summary
key 运行期从 configs/models_aimp.local.yaml 读取, 不打印.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
# ⚠️ 运行期复刻 Arm1/Arm3 (launch_tf_appsplus_routing.sh: `cd $SCRIPT_DIR` + PYTHONPATH=/data/zhuldz/skillopt,
# `python tf_skill_optimizer_appsplus.py`)。实测该分辨率下:
#   tf_skill_optimizer_appsplus → HERE/tf_skill_optimizer_appsplus.py (当前版, 有 _evaluate_d5)
#   skillopt                    → /data/zhuldz/skillopt/skillopt (上游 566 行 qwen_backend, **无 trace 钩子**)
# 陷阱 (issue #96): HERE 下有残缺 `skillopt/` 包 (只有 model/qwen_backend.py 806 行改动版, 无 __init__)
# 和 `code/` 下的旧 optimizer 副本 (无 _evaluate_d5)。因此 HERE/code 必须先出 sys.path,
# 再显式把上游 `/data/zhuldz/skillopt` 置顶 (schedule 与 arm 的 PYTHONPATH 一致)。
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE
            and not os.path.abspath(p or ".").startswith(os.path.join(HERE, "code"))]
sys.path.insert(0, HERE)   # 与 arm 的 cwd 行为一致
sys.path.insert(0, "/data/zhuldz/skillopt")  # 与 arm 的 PYTHONPATH 一致, 且压过 HERE 残缺 skillopt/

import yaml  # noqa: E402

_PROV = yaml.safe_load(open("/data/zhuldz/lunwen/configs/models_aimp.local.yaml",
                            encoding="utf-8"))["providers"]["deepseek-v4-flash"]
os.environ["QWEN_CHAT_API_KEY"] = _PROV["api_key"]      # 仅供本进程, 不打印/不落盘

import skillopt  # noqa: E402
if not os.path.abspath(skillopt.__file__).startswith("/data/zhuldz/skillopt/skillopt/"):
    raise SystemExit(f"skillopt resolved to {skillopt.__file__}, expected upstream /data/zhuldz/skillopt")
import tf_skill_optimizer_appsplus as T  # noqa: E402   (optimizer 内部先 import skillopt.model)
if not hasattr(T, "_evaluate_d5"):
    raise SystemExit(f"stale tf_skill_optimizer_appsplus loaded: {T.__file__}")
# 与 arm 环境逐项对齐 (launch_tf_appsplus_routing.sh); optimizer 只 setdefault, 这里再兜底核对
for _k, _v in {"TARGET_BACKEND": "qwen_chat", "OPTIMIZER_BACKEND": "qwen_chat",
               "QWEN_CHAT_ENABLE_THINKING": "false", "QWEN_CHAT_MAX_RPM": "20",
               "QWEN_CHAT_BURST": "2"}.items():
    os.environ.setdefault(_k, _v)
from skillopt.model import qwen_backend as _QB  # noqa: E402
from skillopt.model.qwen_backend import TARGET_CONFIG  # noqa: E402
_expected_cfg = {"backend": "qwen_chat", "base_url": "https://aimpapi.midea.com/t-aigc/llm-openai-api",
                 "max_tokens": 16384, "temperature": 0.0, "enable_thinking": False,
                 "deployment": "deepseek-v4-flash", "use_max_completion_tokens": False}
_got_cfg = {"backend": T.TARGET_BACKEND if hasattr(T, "TARGET_BACKEND") else "qwen_chat",
            "base_url": TARGET_CONFIG.base_url, "max_tokens": TARGET_CONFIG.max_tokens,
            "temperature": TARGET_CONFIG.temperature, "enable_thinking": bool(TARGET_CONFIG.enable_thinking),
            "deployment": TARGET_CONFIG.deployment,
            "use_max_completion_tokens": TARGET_CONFIG.use_max_completion_tokens}
for _k, _v in _expected_cfg.items():
    if _got_cfg[_k] != _v:
        raise SystemExit(f"config mismatch {_k}: got {_got_cfg[_k]!r}, expected {_v!r}")

# ── 请求级审计 (不改上游库): 逐请求 jsonl, 记录 response/finish_reason/usage/model ─────────
_TRACE_LOCK = threading.Lock()
_TRACE_PATH = [None, None]      # [dir, 当前请求的 item_id]
_TRACE_PATCHED = [False]


def set_trace_dir(path):
    os.makedirs(path, exist_ok=True)
    _TRACE_PATH[0] = path


def _patch_trace():
    """包一层 _chat_messages_impl: 每次**尝试**经 _post_chat_completion 成功即落一行 trace。"""
    if _TRACE_PATCHED[0]:
        return
    inner_post, inner_impl = _QB._post_chat_completion, _QB._chat_messages_impl
    state = threading.local()
    state.stage = state.role = None
    state.attempt = 0

    def post(payload, timeout, config):
        data = inner_post(payload, timeout, config)
        _TRACE_TS.rows = int(getattr(_TRACE_TS, "rows", 0) or 0) + 1
        d = _TRACE_PATH[0]
        if d:
            row = {"ts": round(time.time(), 3), "stage": getattr(state, "stage", None),
                   "role": getattr(state, "role", None),
                   "tid": threading.get_ident(), "attempt": getattr(state, "attempt", None),
                   "item_id": _TRACE_PATH[1],
                   "request": {"model": payload.get("model"), "temperature": payload.get("temperature"),
                               "max_tokens_key": [k for k in ("max_tokens", "max_completion_tokens")
                                                  if k in payload],
                               "max_tokens": payload.get("max_tokens", payload.get("max_completion_tokens")),
                               "stream": payload.get("stream"),
                               "chat_template_kwargs": payload.get("chat_template_kwargs"),
                               "messages_sha256": hashlib.sha256(
                                   json.dumps(payload.get("messages"), ensure_ascii=False,
                                              sort_keys=True).encode()).hexdigest()},
                   "response_model": data.get("model"),
                   "usage": data.get("usage"),
                   "finish_reason": ((data.get("choices") or [{}])[0]).get("finish_reason"),
                   "content_chars": len(((data.get("choices") or [{}])[0].get("message") or {})
                                        .get("content") or "")}
            with _TRACE_LOCK, open(os.path.join(d, f"phase3b_{getattr(state, 'stage', 'x')}.jsonl"),
                                   "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return data

    def impl(messages, *a, **kw):
        # 线程序号: 同一线程连续调用同一 stage 视为一个请求, attempt 从 1 起递增;
        # stage 变化 (即换了请求) 时归零。8 worker 下每线程同一时刻只有一个在途请求。
        st, ro = kw.get("stage"), kw.get("role")
        if st != getattr(state, "stage", "\0"):
            state.attempt = 0
            _TRACE_TS.rows = 0
        prev = getattr(state, "stage", None), getattr(state, "role", None)
        state.stage, state.role = st, ro
        state.attempt = int(getattr(state, "attempt", 0)) + 1
        try:
            return inner_impl(messages, *a, **kw)
        finally:
            state.stage, state.role = prev
            _TRACE_PATH[1] = None

    _QB._post_chat_completion = post
    _QB._chat_messages_impl = impl
    _TRACE_PATCHED[0] = True


def chat_target(*a, **kw):
    """包装上游 chat_target: 额外从 tracker 增量算该请求的 HTTP attempts (重试也计入)。

    ⚠️ tracker 是进程级全局计数, 8 worker 并发下增量会被其他线程吃掉 → attempts 不可靠 (issue #97)。
    准确次数以 trace jsonl 的 `attempt` 字段为准 (由 _post_chat_completion 包装逐次写入)。
    """
    st = kw.get("stage", "target")
    before = _QB.tracker.summary().get(st, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    text, usage = _QB.chat_target(*a, **kw)
    after = _QB.tracker.summary().get(st, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    usage = dict(usage or {})
    usage["tracker_calls_delta"] = int(after["calls"]) - int(before["calls"])
    return text, usage

BASE = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline"
V2 = os.path.join(BASE, "appsplus-skillopt-flash-20261111-v2/train")
ARM1 = os.path.join(BASE, "tf48-arm1-full-20261117/run_20261118_091052")
ARM3 = os.path.join(BASE, "tf48-arm3-drop-scaffold-20261117/run_20261119_091235")
SPLIT_DIR = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
SKILLS = {
    "s0": "/data/zhuldz/skillopt/skillopt/envs/appsplus/skills/initial.md",
    "skillopt": os.path.join(V2, "best_skill.md"),
}
STAGE = {"s0": "phase3b_s0", "skillopt": "phase3b_skillopt"}
CHUNK = 100
WORKERS = 8
INFRA_RETRIES = 1


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


_TRACE_TS = threading.local()   # 当前线程在途请求的 trace 行数 / 末次 attempt


def _take_attempts() -> int:
    """取本线程刚才那次请求的 HTTP 尝试数 (= trace 行数), 由 _patch_trace 的 impl 记录。"""
    n = int(getattr(_TRACE_TS, "rows", 0) or 0)
    _TRACE_TS.rows = 0
    return n


def gen_one(item, skill_content, tag):
    """run_one 的扩展版: 同调用路径, 额外记录调用级元信息。异常 → agent_ok=False。"""
    qid = str(item["id"])
    res = {"id": qid, "difficulty": item.get("difficulty", "?"),
           "hard": 0, "soft": 0.0, "pass_rate": -3.0, "agent_ok": False, "fail_reason": "",
           "ts_start": None, "ts_end": None, "latency_s": None,
           "usage": {"prompt_tokens": 0, "completion_tokens": 0},
           "finish_reason": None, "http_attempts": 0, "infra_retried": False}
    t0 = time.time()
    res["ts_start"] = round(t0, 3)
    try:
        system = T._build_system(skill_content)
        user = T._build_user(item)
        _TRACE_TS.rows = 0
        _TRACE_PATH[1] = qid
        response, usage = chat_target(system=system, user=user, max_completion_tokens=16384,
                                      retries=5, stage=STAGE[tag], timeout=300)
        # 上游 backend 只在成功时返回 usage; finish_reason 走本文件的 trace 包装 (见 _patch_trace)
        ev = T._evaluate_d5(response, item["test_code"], str(item.get("question", "")))
        u = usage or {}
        res.update({"hard": ev["hard"], "soft": ev["soft"], "pass_rate": ev["pass_rate"],
                    "agent_ok": True, "fail_reason": ev.get("fail_reason", "")[:200],
                    "usage": {k: u.get(k, 0) for k in ("prompt_tokens", "completion_tokens")},
                    "http_attempts": _take_attempts(),
                    "response_chars": len(response or "")})
    except Exception as e:  # noqa: BLE001
        res["fail_reason"] = f"exception: {e!r}"[:300]
    res["ts_end"] = round(time.time(), 3)
    res["latency_s"] = round(res["ts_end"] - t0, 2)
    return res


def run_chunk(items, skill_content, tag, existing: dict) -> dict:
    """块内执行 (断点续跑): infra 重试 1 轮, 与 eval_independent 同口径。"""
    out = dict(existing)
    todo = [it for it in items if str(it["id"]) not in out]
    for rnd in range(1 + INFRA_RETRIES):
        if not todo:
            break
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(gen_one, it, skill_content, tag): str(it["id"]) for it in todo}
            for fut in as_completed(futs):
                qid = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:  # future 级异常
                    r = {"id": qid, "difficulty": "?", "hard": 0, "soft": 0.0,
                         "pass_rate": -3.0, "agent_ok": False,
                         "fail_reason": f"future-exception: {e!r}"[:200],
                         "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
                r["infra_retried"] = rnd > 0
                out[qid] = r
        todo = [it for it in items if not out[str(it["id"])]["agent_ok"]
                and not out[str(it["id"])].get("infra_retried")]
    return out


def build_protocol(items):
    files = {
        "test_items": os.path.join(SPLIT_DIR, "test/items.json"),
        "s0_skill": SKILLS["s0"],
        "skillopt_skill": SKILLS["skillopt"],
        "optimizer": os.path.join(HERE, "tf_skill_optimizer_appsplus.py"),
        "optimizer_resolved": T.__file__,
        "qwen_backend_resolved": _QB.__file__,   # 上游只读库, 运行期按自身 sha 记录 (未修改)
        "arm1_independent": os.path.join(ARM1, "test_independent_results.json"),
        "arm3_independent": os.path.join(ARM3, "test_independent_results.json"),
    }
    cfg = TARGET_CONFIG
    return {
        "protocol_doc": "docs/phase3b_static_baseline_2765_protocol.md",
        "nature": "post-revelation baseline supplement (NOT pre-registered confirmatory experiment)",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "n_items": len(items),
        "item_ids_sha256": hashlib.sha256(
            ",".join(str(it["id"]) for it in items).encode()).hexdigest(),
        "file_sha256": {k: sha(p) for k, p in files.items()},
        "model_call": {
            "backend": _got_cfg["backend"], "base_url": cfg.base_url,
            "deployment": cfg.deployment, "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "payload_token_key": "max_completion_tokens" if cfg.use_max_completion_tokens else "max_tokens",
            "stream": False, "enable_thinking": bool(cfg.enable_thinking),
            "timeout_s": cfg.timeout_seconds,
            "max_rpm": os.environ.get("QWEN_CHAT_MAX_RPM"), "burst": os.environ.get("QWEN_CHAT_BURST"),
            "trunc_retry": os.environ.get("QWEN_CHAT_TRUNC_RETRY"),
            "retries": 5, "max_completion_tokens_arg": 16384, "workers": WORKERS,
            "system_prompt_builder": "_build_system(skill_content)",
            "user_prompt_builder": "_build_user(item)",
            "evaluator": "_evaluate_d5 (D5)",
        },
        "skills": {k: {"path": p, "sha256": sha(p), "chars": len(open(p, encoding="utf-8").read()),
                       "stage": STAGE[k]} for k, p in SKILLS.items()},
        "schedule": {"chunk": CHUNK, "ticket_order": "[s0 id_i, skillopt id_i] alternating, 8 workers, 20 RPM"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--drift-check", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(BASE, "phase3b-static-baseline-2765-20261120",
                                       "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(out_dir, "chunks"), exist_ok=True)
    _patch_trace()
    set_trace_dir(os.path.join(out_dir, "trace"))
    items = T.load_split(SPLIT_DIR, "test")
    if args.limit:
        items = items[:args.limit]
    by_id = {str(it["id"]): it for it in items}
    proto_path = os.path.join(out_dir, "protocol.json")
    if not os.path.exists(proto_path):
        with open(proto_path, "w", encoding="utf-8") as f:
            json.dump(build_protocol(items), f, ensure_ascii=False, indent=2)
    print(f"out={out_dir}\nn={len(items)} proto={proto_path}", flush=True)

    if args.drift_check:
        ids = [str(it["id"]) for it in items[:8]]
        skill = open(SKILLS["s0"], encoding="utf-8").read()
        res = run_chunk([by_id[i] for i in ids], skill, "s0", {})
        stored = {str(r["id"]): r for r in json.load(open(
            os.path.join(V2, "test_eval_baseline/results.rejudged_d5_v2.json"), encoding="utf-8"))}
        cmp_rows, agree = [], 0
        for i in ids:
            h_new, h_old = int(res[i]["hard"]), int(stored[i]["hard"]) if i in stored else None
            same = h_old is not None and h_new == h_old
            agree += int(same)
            cmp_rows.append({"id": i, "hard_new": h_new, "hard_stored": h_old,
                             "pass_rate_new": res[i]["pass_rate"],
                             "pass_rate_stored": stored[i]["pass_rate"] if i in stored else None,
                             "fail_reason_stored": (stored[i].get("fail_reason") or "")[:120] if i in stored else None})
        out = {"generated_by": "phase3b_static_baseline_2765.py --drift-check",
               "ran_at": datetime.now().isoformat(timespec="seconds"),
               "n": len(ids), "hard_agree": agree,
               "stored_basis": os.path.join(V2, "test_eval_baseline/results.rejudged_d5_v2.json"),
               "rows": cmp_rows, "new_records": res}
        with open(os.path.join(out_dir, "drift_check_8.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(json.dumps({"hard_agree": f"{agree}/{len(ids)}",
                          "rows": [{k: r[k] for k in ("id", "hard_new", "hard_stored")} for r in cmp_rows]},
                         ensure_ascii=False, indent=1))
        return

    if args.finalize:
        allres = {t: {} for t in SKILLS}
        for fn in sorted(os.listdir(os.path.join(out_dir, "chunks"))):
            d = json.load(open(os.path.join(out_dir, "chunks", fn), encoding="utf-8"))
            for t in SKILLS:
                allres[t].update(d.get(t, {}))
        with open(os.path.join(out_dir, "all_results.json"), "w", encoding="utf-8") as f:
            json.dump(allres, f, ensure_ascii=False, indent=1)
        summary = {"finished_at": datetime.now().isoformat(timespec="seconds"),
                   "usage_this_process": _QB.tracker.summary()}
        for t in SKILLS:
            rs = list(allres[t].values())
            valid = [r for r in rs if r["agent_ok"]]
            summary[t] = {
                "n": len(rs), "n_valid": len(valid), "n_agent_fail": len(rs) - len(valid),
                "hard": sum(int(r["hard"]) for r in valid),
                "hard_rate": round(sum(int(r["hard"]) for r in valid) / max(1, len(valid)), 6),
                "mean_pass_rate": round(sum(r["pass_rate"] for r in valid) / max(1, len(valid)), 6),
                "truncated": sum(1 for r in rs if r.get("finish_reason") == "length"),
                "attempts_gt1": sum(1 for r in rs if (r.get("attempts") or 0) > 1),
                "infra_retried": sum(1 for r in rs if r.get("infra_retried")),
                "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in rs),
                "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in rs),
                "first_ts": min((r["ts_start"] for r in rs if r.get("ts_start")), default=None),
                "last_ts": max((r["ts_end"] for r in rs if r.get("ts_end")), default=None),
                "response_chars": sum(r.get("response_chars") or 0 for r in rs),
            }
        with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=1)[:3000])
        return

    skills = {t: open(p, encoding="utf-8").read() for t, p in SKILLS.items()}
    n_chunks = (len(items) + CHUNK - 1) // CHUNK
    t_all = time.time()
    for ci in range(n_chunks):
        cpath = os.path.join(out_dir, "chunks", f"chunk_{ci:04d}.json")
        chunk = items[ci * CHUNK:(ci + 1) * CHUNK]
        existing = json.load(open(cpath, encoding="utf-8")) if os.path.exists(cpath) else {}
        done = all(str(it["id"]) in existing.get(t, {}) for t in SKILLS for it in chunk) and existing
        if done:
            print(f"[{ci+1}/{n_chunks}] skip (done)", flush=True)
            continue
        t0 = time.time()
        res = {}
        # ticket 交替: 两组同题号交替提交, 共享 token bucket
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {}
            for t in SKILLS:
                for it in chunk:
                    if str(it["id"]) not in existing.get(t, {}):
                        futs[ex.submit(gen_one, it, skills[t], t)] = (t, str(it["id"]))
            got = {t: dict(existing.get(t, {})) for t in SKILLS}
            for fut in as_completed(futs):
                t, qid = futs[fut]
                try:
                    got[t][qid] = fut.result()
                except Exception as e:  # noqa: BLE001
                    got[t][qid] = {"id": qid, "difficulty": "?", "hard": 0, "soft": 0.0,
                                   "pass_rate": -3.0, "agent_ok": False,
                                   "fail_reason": f"future-exception: {e!r}"[:200],
                                   "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        # infra 重试 1 轮: 只重试 agent_ok=False 且本块首跑失败的题
        for t in SKILLS:
            retry_items = [it for it in chunk if str(it["id"]) in got[t]
                           and not got[t][str(it["id"])]["agent_ok"]
                           and not got[t][str(it["id"])].get("infra_retried")]
            if not retry_items:
                continue
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = {ex.submit(gen_one, it, skills[t], t): str(it["id"]) for it in retry_items}
                for fut in as_completed(futs):
                    qid = futs[fut]
                    try:
                        r = fut.result()
                    except Exception:  # noqa: BLE001
                        continue
                    r["infra_retried"] = True
                    got[t][qid] = r
        for t in SKILLS:
            for it in chunk:
                got[t].setdefault(str(it["id"]), {"id": str(it["id"]), "agent_ok": False,
                                                  "hard": 0, "pass_rate": -3.0,
                                                  "fail_reason": "missing",
                                                  "usage": {"prompt_tokens": 0, "completion_tokens": 0}})
        with open(cpath + ".tmp", "w", encoding="utf-8") as f:
            json.dump(got, f, ensure_ascii=False)
        os.replace(cpath + ".tmp", cpath)
        line = " ".join(f"{t}:{sum(int(r['hard']) for r in got[t].values() if r['agent_ok'])}"
                        f"/{sum(1 for r in got[t].values() if r['agent_ok'])}"
                        for t in SKILLS)
        print(f"[{ci+1}/{n_chunks}] {time.time()-t0:.0f}s {line}", flush=True)
        try:  # 仅进度可见性: trace 行数即已完成 HTTP 请求数 (不改动 chunk/断点文件)
            _tf = [os.path.join(os.path.dirname(os.path.join(out_dir, "x")), "trace", f)
                   for f in os.listdir(os.path.join(out_dir, "trace"))]
            _n = sum(sum(1 for _ in open(f, encoding="utf-8", errors="ignore")) for f in _tf)
            print(f"  [progress] http_requests={_n}", flush=True)
        except Exception:  # noqa: BLE001
            pass
    print(f"ALL DONE in {(time.time()-t_all)/60:.1f} min → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
