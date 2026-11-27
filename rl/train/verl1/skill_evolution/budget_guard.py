"""Budget guard for LiveMath full run (冻结方案 C, 2026-11-26).

用户三条执行边界之一: 60k calls / 36h 墙钟硬停。evolver 现无此机制。
本模块在本项目 git 内 (不改 skillopt 第三方包源码), 通过 monkey-patch
`skillopt.model.qwen_backend._post_chat_completion` 挂载原子预算预留。

挂载点决策 (静态核查见 docs/phase2_issues_202611.md #123):
- 真实 HTTP 收口 = `_post_chat_completion` (urlopen 在其内部)
- `_chat_messages_impl` 的 retry 循环 `for attempt in range(retries)` 每轮调一次 `_post_chat_completion`
- L311 是裸名调用, 运行时查模块 globals → patch 模块属性 `_post_chat_completion` 会被解析到
- 故包装 `_post_chat_completion` = 每次 HTTP attempt (含重试) 都计数, 不漏

线程安全: evolver 单进程 + ThreadPoolExecutor(40). 用 threading.Lock 守模块级计数器, 无需跨进程。
断点续跑: 预算账本 + start_ts 落盘 run 目录, --resume 时加载累计 (不清零)。
不改算法: 包装只计数, 不触碰 payload/response/异常/重试顺序。

启用: evolver 启动时调 `install(out_dir, max_calls, wall_seconds, resume_from=...)`。
预算耗尽: 包装在 urlopen 前抛 `BudgetExceeded` (RuntimeError 子类), 不发请求。
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone

# ── 全局账本 (单进程, 线程安全) ──────────────────────────────────────────
_LOCK = threading.Lock()
_state: dict = {
    "n_requests": 0,        # 已预留 (含已发/进行中/已失败) 的真实 HTTP attempt 数
    "n_attempts_total": 0,  # = n_requests (别名, 语义同, 供日志区分)
    "n_budget_exceeded": 0,  # 因预算耗尽被拒的 attempt 数
    "start_ts": None,       # run 起始 unix 秒 (resume 时沿用最早)
    "max_calls": None,
    "wall_seconds": None,
    "ledger_path": None,
    "installed": False,
    "original_post": None,   # 未包装的 _post_chat_completion (用于卸载/直通)
}


class BudgetExceeded(RuntimeError):
    """预算耗尽或墙钟超限。包装在 urlopen 前抛出, 不发真实 HTTP 请求。
    是 RuntimeError 子类, 被 _chat_messages_impl 的 `except Exception` 捕获后计入 retry,
    但因为预算已满, 后续 attempt 也会立即抛 → 最终 `retries` 耗尽后向上传播 RuntimeError。
    evolver 主循环捕获它做安全停止 (见 install 注入的钩子)。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _wall_elapsed() -> float:
    if _state["start_ts"] is None:
        return 0.0
    return time.time() - _state["start_ts"]


def _persist_locked():
    """已持锁时调用: 把账本落盘 (resume 可恢复累计计数)。"""
    p = _state["ledger_path"]
    if not p:
        return
    snap = {
        "n_requests": _state["n_requests"],
        "n_attempts_total": _state["n_attempts_total"],
        "n_budget_exceeded": _state["n_budget_exceeded"],
        "start_ts": _state["start_ts"],
        "max_calls": _state["max_calls"],
        "wall_seconds": _state["wall_seconds"],
        "wall_elapsed": round(_wall_elapsed(), 1),
        "updated": _now_iso(),
    }
    try:
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)  # 原子写
    except Exception:
        pass  # 落盘失败不影响预算控制主链


def _check_and_reserve_locked() -> bool:
    """已持锁时调用: 检查预算并原子预留一次 attempt。返回 True=已预留可发, False=预算已满。"""
    mc = _state["max_calls"]
    ws = _state["wall_seconds"]
    if mc is not None and _state["n_requests"] >= mc:
        return False
    if ws is not None and _wall_elapsed() >= ws:
        return False
    _state["n_requests"] += 1
    _state["n_attempts_total"] += 1
    return True


def _wrapped_post_chat_completion(payload, timeout, config):
    """包装版 _post_chat_completion: urlopen 前原子预留预算。"""
    # 墙钟检查 (无需预留, 只读, 但放锁内避免与请求计数交错)
    with _LOCK:
        ok = _check_and_reserve_locked()
        if not ok:
            _state["n_budget_exceeded"] += 1
            _persist_locked()
        else:
            # 每 50 次落盘一次 (降低 IO), 关键时刻 (接近上限) 每次落盘
            if _state["n_requests"] % 50 == 0 or (
                _state["max_calls"] and _state["n_requests"] >= _state["max_calls"] - 200
            ):
                _persist_locked()
    if not ok:
        raise BudgetExceeded(
            f"budget exhausted: n_requests={_state['n_requests']} "
            f"max_calls={_state['max_calls']} wall_elapsed={_wall_elapsed():.0f}s "
            f"wall_limit={_state['wall_seconds']}s"
        )
    # 真实 HTTP 请求 (调用原 _post_chat_completion). 不动 payload/timeout/config/异常.
    return _state["original_post"](payload, timeout, config)


def install(out_dir, max_calls=None, wall_seconds=None, resume_from=None):
    """evolver 启动时调用: 安装预算护栏。

    out_dir: run 目录, 账本 budget_ledger.json 落盘于此.
    max_calls: 真实 HTTP attempt 上限 (60k). None=不限 calls.
    wall_seconds: 墙钟上限秒 (36h=129600). None=不限墙钟.
    resume_from: 若 --resume, 传 prior run 目录, 加载其账本累计计数 + 沿用最早 start_ts.

    幂等: 重复调用不重复 patch (检查 _state['installed'])。
    """
    if _state["installed"]:
        return _state
    import skillopt.model.qwen_backend as qb

    ledger_path = os.path.join(out_dir, "budget_ledger.json")
    start_ts = time.time()
    n_requests = 0
    n_budget_exceeded = 0

    # 断点续跑: 加载 prior 账本 (累计计数 + 沿用最早 start_ts)
    if resume_from:
        prior_ledger = os.path.join(resume_from, "budget_ledger.json")
        if os.path.exists(prior_ledger):
            try:
                with open(prior_ledger, encoding="utf-8") as f:
                    prior = json.load(f)
                n_requests = int(prior.get("n_requests", 0))
                n_budget_exceeded = int(prior.get("n_budget_exceeded", 0))
                # 沿用最早 start_ts (墙钟不因重启清零)
                if prior.get("start_ts"):
                    start_ts = float(prior["start_ts"])
                print(f"🛡️ budget_guard resume: n_requests={n_requests} "
                      f"start_ts={datetime.fromtimestamp(start_ts, timezone.utc).strftime('%H:%M:%SZ')} "
                      f"(累计沿用, 不清零)", flush=True)
            except Exception as e:
                print(f"⚠️ budget_guard resume: 加载 prior 账本失败 ({e!r}), 从 0 起", flush=True)

    with _LOCK:
        _state.update({
            "n_requests": n_requests,
            "n_attempts_total": n_requests,
            "n_budget_exceeded": n_budget_exceeded,
            "start_ts": start_ts,
            "max_calls": max_calls,
            "wall_seconds": wall_seconds,
            "ledger_path": ledger_path,
            "original_post": qb._post_chat_completion,
            "installed": True,
        })
        # patch 模块全局符号 (L311 裸名调用会解析到此)
        qb._post_chat_completion = _wrapped_post_chat_completion
        _persist_locked()

    print(f"🛡️ budget_guard installed: max_calls={max_calls} wall_seconds={wall_seconds} "
          f"ledger={ledger_path} (patched skillopt.model.qwen_backend._post_chat_completion)", flush=True)
    return _state


def uninstall():
    """卸载 patch (测试用, 正式 run 不卸载)。"""
    with _LOCK:
        if not _state["installed"]:
            return
        import skillopt.model.qwen_backend as qb
        if _state["original_post"] is not None:
            qb._post_chat_completion = _state["original_post"]
        _state["installed"] = False


def status() -> dict:
    """当前预算状态快照 (线程安全)。供 evolver 每轮记录 + 终局报告。"""
    with _LOCK:
        return {
            "n_requests": _state["n_requests"],
            "n_attempts_total": _state["n_attempts_total"],
            "n_budget_exceeded": _state["n_budget_exceeded"],
            "max_calls": _state["max_calls"],
            "wall_seconds": _state["wall_seconds"],
            "wall_elapsed": round(_wall_elapsed(), 1),
            "start_ts": _state["start_ts"],
            "installed": _state["installed"],
        }


def flush():
    """强制落盘账本 (evolver 每轮结束/终局调用)。"""
    with _LOCK:
        _persist_locked()


def is_exhausted() -> bool:
    """预算是否已耗尽 (evolver 每轮前检查, 提前停止派发)。"""
    with _LOCK:
        mc = _state["max_calls"]
        ws = _state["wall_seconds"]
        if mc is not None and _state["n_requests"] >= mc:
            return True
        if ws is not None and _wall_elapsed() >= ws:
            return True
        return False
