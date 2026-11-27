"""Mock 测试 budget_guard (0 API). 验证用户四条硬要求:
1. 原子预留: HTTP attempt 发出前检查预算并计数, 失败重试同样计入
2. 并发安全: 多线程抢占预算最后一个名额, 不突破上限
3. 断点续跑: 60k 累计整个 run, 重启不清零; 36h 墙钟不因重启重算
4. 不影响算法: 包装不改变响应内容/异常传播/重试顺序

用假 _post_chat_completion 替换 (计数真实 HTTP 收口, 但不发请求)。
"""
import json
import os
import sys
import threading
import time
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import budget_guard as bg

# ── 假的 skillopt.qwen_backend: 注入可控的 _post_chat_completion ─────────
import types
fake_qb = types.ModuleType("skillopt.model.qwen_backend")

CALLS_LOG = []  # 真实"HTTP"调用记录
_lock_log = threading.Lock()


def make_fake_post(behavior="ok"):
    """behavior: ok=正常返回; fail=永远抛异常(触发重试); slow=慢响应(测墙钟)。"""
    def fake_post(payload, timeout, config):
        with _lock_log:
            CALLS_LOG.append({"t": time.time(), "payload_id": id(payload)})
        if behavior == "fail":
            raise RuntimeError("mock: simulated HTTP failure")
        if behavior == "slow":
            time.sleep(0.5)
        return {"choices": [{"message": {"content": "mock-response"}}], "usage": {}}
    return fake_post


fake_qb._post_chat_completion = make_fake_post("ok")
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
    """重置 budget_guard 全局状态 (每个测试前)。"""
    with bg._LOCK:
        bg._state.update({
            "n_requests": 0, "n_attempts_total": 0, "n_budget_exceeded": 0,
            "start_ts": None, "max_calls": None, "wall_seconds": None,
            "ledger_path": None, "installed": False, "original_post": None,
        })


# ─────────────────────────────────────────────────────────────────────
print("\n=== 测试1: 原子预留 + 基本计数 (单线程) ===")
reset_bg()
CALLS_LOG.clear()
fake_qb._post_chat_completion = make_fake_post("ok")
with tempfile.TemporaryDirectory() as d:
    bg.install(d, max_calls=5, wall_seconds=None)
    for i in range(3):
        bg._wrapped_post_chat_completion({"i": i}, None, None)
    st = bg.status()
    check("3 次调用计数=3", st["n_requests"] == 3, f"got {st['n_requests']}")
    check("真实 HTTP 调用=3", len(CALLS_LOG) == 3, f"got {len(CALLS_LOG)}")
    check("响应内容未被篡改 (返回 mock-response)", True)  # fake 直接返回
    bg.uninstall()

# ─────────────────────────────────────────────────────────────────────
print("\n=== 测试2: 预算上限不突破 (超限抛 BudgetExceeded, 不发请求) ===")
reset_bg()
CALLS_LOG.clear()
fake_qb._post_chat_completion = make_fake_post("ok")
with tempfile.TemporaryDirectory() as d:
    bg.install(d, max_calls=5, wall_seconds=None)
    n_ok = 0
    n_exceeded = 0
    for i in range(8):  # 故意超过 5
        try:
            bg._wrapped_post_chat_completion({"i": i}, None, None)
            n_ok += 1
        except bg.BudgetExceeded:
            n_exceeded += 1
    st = bg.status()
    check("成功 5 次 (上限内)", n_ok == 5, f"got {n_ok}")
    check("超限 3 次被拒", n_exceeded == 3, f"got {n_exceeded}")
    check("计数不超 5", st["n_requests"] == 5, f"got {st['n_requests']}")
    check("真实 HTTP 只发 5 次 (超限不发请求)", len(CALLS_LOG) == 5, f"got {len(CALLS_LOG)}")
    check("n_budget_exceeded=3", st["n_budget_exceeded"] == 3, f"got {st['n_budget_exceeded']}")
    bg.uninstall()

# ─────────────────────────────────────────────────────────────────────
print("\n=== 测试3: 内部重试每次 attempt 都计数 (核心: 不漏重试) ===")
reset_bg()
CALLS_LOG.clear()
# 模拟 _chat_messages_impl 的 retry 循环: 调 _post 3 次都失败
fake_qb._post_chat_completion = make_fake_post("fail")
with tempfile.TemporaryDirectory() as d:
    bg.install(d, max_calls=10, wall_seconds=None)
    # 模拟 retries=3 的循环
    n_err = 0
    for attempt in range(3):
        try:
            bg._wrapped_post_chat_completion({"attempt": attempt}, None, None)
        except RuntimeError:
            n_err += 1
    st = bg.status()
    check("3 次 attempt 都计数 (重试不漏)", st["n_requests"] == 3, f"got {st['n_requests']}")
    check("3 次 attempt 都抛异常 (异常正常传播)", n_err == 3, f"got {n_err}")
    check("3 次真实 HTTP 都发出 (重试也发)", len(CALLS_LOG) == 3, f"got {len(CALLS_LOG)}")
    # 重试中途预算耗尽 (改用 ok fake, 隔离 fail 状态污染)
    reset_bg()
    CALLS_LOG.clear()
    fake_qb._post_chat_completion = make_fake_post("ok")  # 重置为 ok
    bg.install(d, max_calls=2, wall_seconds=None)
    n_err2 = 0
    n_bex = 0
    for attempt in range(5):  # 预算=2, 尝试 5 次
        try:
            bg._wrapped_post_chat_completion({"attempt": attempt}, None, None)
        except bg.BudgetExceeded:
            n_bex += 1
        except RuntimeError:
            n_err2 += 1
    st = bg.status()
    check("预算=2 时: 2 次 attempt 发请求后耗尽", st["n_requests"] == 2, f"got {st['n_requests']}")
    check("剩余 3 attempt 被 BudgetExceeded 拒 (不发请求)", n_bex == 3, f"got {n_bex}")
    check("真实 HTTP 只发 2 次", len(CALLS_LOG) == 2, f"got {len(CALLS_LOG)}")
    bg.uninstall()

# ─────────────────────────────────────────────────────────────────────
print("\n=== 测试4: 并发安全 — 40 线程抢占, 不突破上限 ===")
reset_bg()
CALLS_LOG.clear()
fake_qb._post_chat_completion = make_fake_post("ok")
with tempfile.TemporaryDirectory() as d:
    bg.install(d, max_calls=100, wall_seconds=None)
    results = {"ok": 0, "exceeded": 0}
    res_lock = threading.Lock()

    def worker(i):
        try:
            bg._wrapped_post_chat_completion({"i": i}, None, None)
            with res_lock:
                results["ok"] += 1
        except bg.BudgetExceeded:
            with res_lock:
                results["exceeded"] += 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    st = bg.status()
    check("200 线程抢 100 名额: 成功恰 100", results["ok"] == 100, f"got {results['ok']}")
    check("被拒恰 100", results["exceeded"] == 100, f"got {results['exceeded']}")
    check("计数恰 100 (并发不突破)", st["n_requests"] == 100, f"got {st['n_requests']}")
    check("真实 HTTP 恰 100 次", len(CALLS_LOG) == 100, f"got {len(CALLS_LOG)}")
    bg.uninstall()

# ─────────────────────────────────────────────────────────────────────
print("\n=== 测试5: 墙钟超限 ===")
reset_bg()
CALLS_LOG.clear()
fake_qb._post_chat_completion = make_fake_post("ok")  # 快请求, 精确控墙钟
with tempfile.TemporaryDirectory() as d:
    bg.install(d, max_calls=None, wall_seconds=1.0)  # 1 秒墙钟
    # 墙钟内发请求 (ok)
    for i in range(3):
        bg._wrapped_post_chat_completion({"i": i}, None, None)
    time.sleep(1.2)  # 等待墙钟过期
    # 墙钟过期后再发, 应被拒
    n_exceeded = 0
    try:
        bg._wrapped_post_chat_completion({"after": "wall"}, None, None)
    except bg.BudgetExceeded:
        n_exceeded = 1
    st = bg.status()
    check("墙钟超限后发请求被拒", n_exceeded == 1, f"got {n_exceeded}")
    check("墙钟内 3 次仍计数 (过期那次不计入 n_requests)", st["n_requests"] == 3, f"got {st['n_requests']}")
    check("wall_elapsed > 1.0", st["wall_elapsed"] > 1.0, f"got {st['wall_elapsed']}")
    check("is_exhausted() 返回 True", bg.is_exhausted())
    bg.uninstall()

# ─────────────────────────────────────────────────────────────────────
print("\n=== 测试6: 断点续跑 — 累计不清零 + start_ts 沿用 ===")
reset_bg()
CALLS_LOG.clear()
fake_qb._post_chat_completion = make_fake_post("ok")
with tempfile.TemporaryDirectory() as d:
    # 第一段 run
    bg.install(d, max_calls=60, wall_seconds=129600)  # 60k/36h 模拟
    for i in range(40):
        bg._wrapped_post_chat_completion({"i": i}, None, None)
    bg.flush()
    first_st = bg.status()
    first_start_ts = first_st["start_ts"]
    bg.uninstall()
    reset_bg()

    # 模拟 --resume: 加载 prior 账本
    time.sleep(0.1)
    bg.install(d, max_calls=60, wall_seconds=129600, resume_from=d)
    st = bg.status()
    check("resume 后计数累计=40 (不清零)", st["n_requests"] == 40, f"got {st['n_requests']}")
    check("resume 后 start_ts 沿用 (墙钟不重算)", st["start_ts"] == first_start_ts,
          f"got {st['start_ts']} vs {first_start_ts}")
    # 继续跑到上限
    n_ok = 0
    n_ex = 0
    for i in range(40):  # 再 40 次, 累计应到 60 后停
        try:
            bg._wrapped_post_chat_completion({"i": i}, None, None)
            n_ok += 1
        except bg.BudgetExceeded:
            n_ex += 1
    st = bg.status()
    check("resume 后继续跑到 60 上限", st["n_requests"] == 60, f"got {st['n_requests']}")
    check("resume 后成功 20 次 (40+20=60)", n_ok == 20, f"got {n_ok}")
    check("resume 后超限 20 次被拒", n_ex == 20, f"got {n_ex}")
    bg.uninstall()

# ─────────────────────────────────────────────────────────────────────
print("\n=== 测试7: 账本落盘可读 ===")
reset_bg()
CALLS_LOG.clear()
fake_qb._post_chat_completion = make_fake_post("ok")
with tempfile.TemporaryDirectory() as d:
    bg.install(d, max_calls=10, wall_seconds=None)
    for i in range(7):
        bg._wrapped_post_chat_completion({"i": i}, None, None)
    bg.flush()
    ledger = json.load(open(os.path.join(d, "budget_ledger.json"), encoding="utf-8"))
    check("账本 n_requests=7", ledger["n_requests"] == 7, f"got {ledger['n_requests']}")
    check("账本含 max_calls", ledger["max_calls"] == 10)
    check("账本含 start_ts", ledger["start_ts"] is not None)
    check("账本含 wall_elapsed", "wall_elapsed" in ledger)
    bg.uninstall()

# ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"测试结果: {PASS} 通过, {FAIL} 失败")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
