"""Gate trace collector for LiveMath full run (冻结方案 B, 2026-11-26).

用户要求 + codex 指出: per_item_delta 未落盘 = 离线无法精确复算 CI 的真实缺口。
本模块在本项目 git 内, 纯增量观测: 每个候选评测后记录一行 gate_trace.jsonl,
含逐题 delta, 支持离线复算 bootstrap CI + 稳估 σ + 区分生成/选择瓶颈。

不改算法 (用户硬要求#3): 只增加观测和持久化, 不触碰候选选择/随机数/晋升判断/
rescue/pool_filter/任何冻结门控。trace_record 只 append, 不返回值影响主链。

启用: evolver 启动时调 `init(out_dir)`, 各评测点调 `record(...)`, 轮末 `flush()`,
终局 `close()`。也暴露 `record_budget()` 让 evolver 每轮记预算状态 (与 budget_guard 联动)。
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

_LOCK = threading.Lock()
_fp = None
_path = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init(out_dir):
    """evolver 启动时调用: 打开 gate_trace.jsonl (追加模式, 崩溃不丢已写行)。"""
    global _fp, _path
    if _fp is not None:
        return _path
    _path = os.path.join(out_dir, "gate_trace.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    _fp = open(_path, "a", encoding="utf-8")
    return _path


def record(iter_t, operator, facet, parent_hash, cand_hash, cand_skill_text,
           val_delta, val_ci, per_item_delta, not_meta_ci=None, meta_ci=None,
           gate_decision=None, gate_reason=None, extra=None):
    """记录一个候选的完整评测轨迹 (一行 JSON)。

    iter_t: 轮次. operator: facet_edit/overlay/expand/crossover/strengthen/collect_failures.
    facet: 被改 facet 轴. parent_hash/cand_hash: 父子 skill md5.
    cand_skill_text: 候选完整文本 (复现 + 看 evolver 发现了什么规则).
    val_delta/val_ci: 总体 paired delta + bootstrap CI lower (n=select).
    per_item_delta: {qid: [d]} 逐题 delta — 本次补落盘的核心字段.
    not_meta_ci/meta_ci: 子集 CI (分层信号, 有则记).
    gate_decision: PROMOTE/KEEP-PARENT/RESCUE/LOST/POOL-FILTER-REJ.
    gate_reason: 拒绝/晋升原因.
    extra: 额外 dict (如 base_pass, cp, n_error 等).
    """
    if _fp is None:
        return
    row = {
        "ts": _now_iso(),
        "iter": iter_t,
        "operator": operator,
        "facet": facet,
        "parent_hash": parent_hash,
        "cand_hash": cand_hash,
        "cand_skill_text": cand_skill_text,
        "val_delta": round(val_delta, 6) if val_delta is not None else None,
        "val_ci": round(val_ci, 6) if val_ci is not None else None,
        "per_item_delta": {str(q): [round(float(x), 6) for x in v] for q, v in (per_item_delta or {}).items()},
        "n_items": len(per_item_delta or {}),
        "not_meta_ci": round(not_meta_ci, 6) if not_meta_ci is not None else None,
        "meta_ci": round(meta_ci, 6) if meta_ci is not None else None,
        "gate_decision": gate_decision,
        "gate_reason": gate_reason,
        "extra": extra or {},
    }
    line = json.dumps(row, ensure_ascii=False)
    with _LOCK:
        try:
            _fp.write(line + "\n")
            _fp.flush()  # 即时落盘, 崩溃不丢
        except Exception:
            pass


def record_budget(iter_t, budget_status):
    """每轮记录预算状态一行 (与 budget_guard.status() 联动)。"""
    if _fp is None:
        return
    row = {"ts": _now_iso(), "iter": iter_t, "event": "budget_snapshot",
           "budget": budget_status}
    line = json.dumps(row, ensure_ascii=False)
    with _LOCK:
        try:
            _fp.write(line + "\n")
            _fp.flush()
        except Exception:
            pass


def flush():
    """强制 flush (evolver 每轮末尾调用)。"""
    with _LOCK:
        if _fp is not None:
            try:
                _fp.flush()
            except Exception:
                pass


def close():
    """终局关闭 (evolver 退出前调用)。"""
    global _fp
    with _LOCK:
        if _fp is not None:
            try:
                _fp.close()
            except Exception:
                pass
            _fp = None
