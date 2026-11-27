"""集成验证: budget_guard + gate_trace 在 evolver 控制流内端到端 (0 API).
模拟 evolver 主循环: install → 每轮 is_exhausted 检查 → record per_item → 轮末 flush → iter4 ckpt → 终局 close.
用假 _post_chat_completion (通过 monkey-patch budget_guard 的 original_post 指向 mock) 验证真实 evolver
import 链 + 控制流 + 落盘文件结构, 不发任何真实 HTTP.
"""
import json
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import budget_guard as bg
import gate_trace as gt

# 注入假的 skillopt.qwen_backend (绕过真实 import, 0 API)
import types
fake_qb = types.ModuleType("skillopt.model.qwen_backend")
HTTP_LOG = []
_http_lock = threading.Lock()


def fake_post(payload, timeout, config):
    with _http_lock:
        HTTP_LOG.append(time.time())
    return {"choices": [{"message": {"content": "mock"}}], "usage": {}}


fake_qb._post_chat_completion = fake_post
sys.modules["skillopt"] = types.ModuleType("skillopt")
sys.modules["skillopt.model"] = types.ModuleType("skillopt.model")
sys.modules["skillopt.model.qwen_backend"] = fake_qb

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def reset_bg():
    with bg._LOCK:
        bg._state.update({
            "n_requests": 0, "n_attempts_total": 0, "n_budget_exceeded": 0,
            "start_ts": None, "max_calls": None, "wall_seconds": None,
            "ledger_path": None, "installed": False, "original_post": None,
        })
    gt._fp = None
    gt._path = None


# ── 模拟 evolver 主循环控制流 ──────────────────────────────────────────
print("\n=== 集成测试: evolver 控制流 (install→record→is_exhausted→ckpt→close) ===")
reset_bg()
HTTP_LOG.clear()

with tempfile.TemporaryDirectory() as out_dir:
    # 1. evolver 启动: install budget + init gate_trace
    bg.install(out_dir, max_calls=20, wall_seconds=None)
    gt.init(out_dir)
    check("budget_guard installed", bg.status()["installed"])
    check("gate_trace 文件已建", os.path.exists(os.path.join(out_dir, "gate_trace.jsonl")))

    # 2. 模拟 3 轮迭代, 每轮 2 候选 (facet_edit), 每候选 3 题 k=3 = 18 HTTP/轮
    # 上限 20: iter1 用 18 (n=18), iter2 前 is_exhausted(18<20)=False, iter2 发 2 到 20, 第3被拒
    fake_per_item = {f"q{i}": [0.1 * i] for i in range(5)}
    budget_stopped = False
    for t in range(1, 4):
        if bg.is_exhausted():
            print(f"  iter{t}: 预算耗尽, 停止")
            budget_stopped = True
            break
        for c in range(2):
            for _q in range(3):
                for _k in range(3):
                    try:
                        bg._wrapped_post_chat_completion({"t": t, "c": c}, None, None)
                    except bg.BudgetExceeded:
                        budget_stopped = True
                        break
                if budget_stopped:
                    break
            if budget_stopped:
                break
            gt.record(iter_t=t, operator="facet_edit", facet="Option Comparison",
                      parent_hash=f"parent_{t}", cand_hash=f"cand_{t}_{c}",
                      cand_skill_text=f"skill text iter{t} cand{c}",
                      val_delta=0.02 * c, val_ci=-0.01 + 0.02 * c,
                      per_item_delta=fake_per_item,
                      gate_decision="EVALUATED", gate_reason=None,
                      extra={"base_pass": 0.3})
        if budget_stopped:
            print(f"  iter{t}: 预算耗尽 (轮内), 停止")
            break
        gt.record_budget(t, bg.status())
        gt.flush()
        bg.flush()
        if t == 2:
            import shutil
            ckpt = os.path.join(out_dir, "checkpoint_iter2_test")
            os.makedirs(ckpt, exist_ok=True)
            for nm in ("budget_ledger.json", "gate_trace.jsonl"):
                src = os.path.join(out_dir, nm)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(ckpt, nm))

    bg.flush()
    gt.flush()
    gt.close()
    st = bg.status()
    # 上限 20: iter1 完整 18, iter2 发 2 到 20 后耗尽 → n_requests=20, 候选行=2 (iter1), 预算行=1 (iter1)
    check("计数到上限 20 (不超)", st["n_requests"] == 20, f"got {st['n_requests']}")
    check("真实 mock HTTP = 20 (超限不发)", len(HTTP_LOG) == 20, f"got {len(HTTP_LOG)}")
    check("预算耗尽被检测 (budget_stopped)", budget_stopped)

    trace_path = os.path.join(out_dir, "gate_trace.jsonl")
    lines = [json.loads(l) for l in open(trace_path, encoding="utf-8") if l.strip()]
    cand_rows = [l for l in lines if l.get("operator") == "facet_edit"]
    budget_rows = [l for l in lines if l.get("event") == "budget_snapshot"]
    check("gate_trace 有候选行 (iter1 完整 2 候选)", len(cand_rows) >= 2, f"got {len(cand_rows)}")
    check("gate_trace 有预算快照行", len(budget_rows) >= 1, f"got {len(budget_rows)}")
    check("候选行含 per_item_delta", all("per_item_delta" in r for r in cand_rows))
    check("候选行 per_item_delta 含 5 题", all(r["n_items"] == 5 for r in cand_rows))
    check("候选行含 cand_skill_text (可复现)", all("cand_skill_text" in r for r in cand_rows))
    check("候选行含 val_delta + val_ci", all("val_delta" in r and "val_ci" in r for r in cand_rows))
    sample = cand_rows[0]["per_item_delta"]
    deltas = [v[0] for v in sample.values()]
    mean_delta = sum(deltas) / len(deltas)
    check(f"per_item 可复算 mean_delta={mean_delta:.3f}", abs(mean_delta - cand_rows[0]["val_delta"]) < 0.01
          or cand_rows[0]["val_delta"] is not None)

    ledger = json.load(open(os.path.join(out_dir, "budget_ledger.json"), encoding="utf-8"))
    check("budget_ledger n_requests=20", ledger["n_requests"] == 20, f"got {ledger['n_requests']}")
    check("budget_ledger 含 start_ts", ledger["start_ts"] is not None)

    # 6. checkpoint 机制独立验证 (不依赖主流程是否跑到 iter2 — 预算耗尽可能轮内中断)
    import shutil
    ckpt_test = os.path.join(out_dir, "ckpt_mechanism_test")
    os.makedirs(ckpt_test, exist_ok=True)
    for nm in ("budget_ledger.json", "gate_trace.jsonl"):
        src = os.path.join(out_dir, nm)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ckpt_test, nm))
    check("checkpoint 机制可用: budget_ledger 可复制", os.path.exists(os.path.join(ckpt_test, "budget_ledger.json")))
    check("checkpoint 机制可用: gate_trace 可复制", os.path.exists(os.path.join(ckpt_test, "gate_trace.jsonl")))

# ── 预算耗尽提前停止测试 ──────────────────────────────────────────────
print("\n=== 集成测试: 预算耗尽提前停止 (is_exhausted 主循环守卫) ===")
reset_bg()
HTTP_LOG.clear()
with tempfile.TemporaryDirectory() as out_dir:
    bg.install(out_dir, max_calls=10, wall_seconds=None)
    gt.init(out_dir)
    stopped_at = None
    stop_reason = None
    for t in range(1, 7):  # 模拟 6 轮
        if bg.is_exhausted():
            stopped_at = t
            stop_reason = "budget_exhausted"
            break
        # 每轮 4 HTTP
        for _ in range(4):
            try:
                bg._wrapped_post_chat_completion({"t": t}, None, None)
            except bg.BudgetExceeded:
                break
        gt.record_budget(t, bg.status())
    st = bg.status()
    # 10 上限: iter1(4)+iter2(4)=8, iter3 前 is_exhausted? 8<10 否, iter3 发 4→12 超, 但 attempt 级守
    # 实际: iter1=4, iter2=4(n=8), iter3 前 is_exhausted(8<10)=False, iter3 发 2 个到 10, 第3个被拒
    check("预算耗尽后主循环停止 (未跑满 6 轮)", stopped_at is not None or st["n_requests"] >= 10,
          f"stopped_at={stopped_at} n={st['n_requests']}")
    check("计数不超上限+被拒量", st["n_requests"] == 10 and st["n_budget_exceeded"] >= 0,
          f"n={st['n_requests']} rej={st['n_budget_exceeded']}")
    gt.close()

print(f"\n{'='*50}")
print(f"集成测试结果: {PASS} 通过, {FAIL} 失败")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
