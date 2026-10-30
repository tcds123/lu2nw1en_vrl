"""C2 Self-test 3: eval_paired unit test (零 API, mock run_one).

验证 (a) 5-tuple 结构, (b) delta 代数 d_paired = cand_pr − base_pr, (c) 超时放弃的两 skill 都计 0.
不触发真实 API: 用 monkeypatch 把 tf_global_v2_aime.run_one 替换为查表函数.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_global_v2_aime as tfa
import skill_evolver_aime_v2 as ev  # noqa: F401  (确认 import 通, 含 eval_paired)

# ---- mock run_one: 按 (skill 前缀, qid) 查表返回 pass_rate ----
# cand_skill 前缀 "C" → pass_rate 表; base_skill 前缀 "B" → pass_rate 表.
_CAND = {"q1": 0.875, "q2": 0.500, "q3": 0.000}   # k=8 → 7/4/0 pass
_BASE = {"q1": 0.500, "q2": 0.500, "q3": 0.125}   # k=8 → 4/4/1 pass
# 期望 paired delta: q1=+0.375, q2=0.0, q3=−0.125 → mean = (0.375+0−0.125)/3 = 0.0833...

def _mock_run_one(it, skill, stage, k, capture_cb=None):
    # capture_cb: eval_paired L456 无条件转发 capture_cb=cb (10-22 改动), 真 run_one 有该形参
    # (10-20 加, 默认 None); mock 必须收下该 kw 否则 TypeError. 零 API 测试不消费 cb.
    qid = str(it["id"])
    if skill.startswith("C"):
        pr = _CAND.get(qid, 0.0)
    elif skill.startswith("B"):
        pr = _BASE.get(qid, 0.0)
    else:
        pr = 0.0
    return {"id": it["id"], "pass_rate": pr, "n_pass": round(pr * k),
            "n_samples": k, "stage": stage, "ok": True}

tfa.run_one = _mock_run_one   # monkeypatch 模块级符号 (eval_paired 通过模块全局解析 run_one)

ITEMS = [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}]
K = 8

# ---- Test A: 正常配对 (无超时) ----
mp, md, n, per_item, base_per = tfa.eval_paired(
    ITEMS, cand_skill="C-skill", base_skill="B-skill",
    stage="test_paired", k=K, eval_deadline_s=None)

assert isinstance(mp, float), f"mean_pass 应为 float, got {type(mp)}"
assert isinstance(md, float), f"mean_delta 应为 float, got {type(md)}"
assert n == 3, f"n 应=3 (3 items), got {n}"
assert set(per_item.keys()) == {"q1", "q2", "q3"}, f"per_item qids 不全: {per_item.keys()}"
assert set(base_per.keys()) == {"q1", "q2", "q3"}, f"base_per_item qids 不全: {base_per.keys()}"
assert 0.0 <= mp <= 1.0, f"mean_pass 超界: {mp}"
# delta 代数
assert abs(per_item["q1"][0] - (0.875 - 0.500)) < 1e-9, f"q1 paired delta 错: {per_item['q1']}"
assert abs(per_item["q2"][0] - (0.500 - 0.500)) < 1e-9, f"q2 paired delta 错: {per_item['q2']}"
assert abs(per_item["q3"][0] - (0.000 - 0.125)) < 1e-9, f"q3 paired delta 错: {per_item['q3']}"
assert abs(md - (0.375 + 0.0 - 0.125) / 3) < 1e-9, f"mean_delta 错: {md}"
assert abs(base_per["q1"] - 0.500) < 1e-9, f"base_per q1 错: {base_per['q1']}"
print(f"✅ Test A (正常配对): mean_pass={mp:.4f}, mean_delta={md:+.4f}, n={n}, per_item={per_item}")

# ---- Test B: 超时放弃 (mock run_one sleep → deadline 触发) ----
import time
def _slow_run_one(it, skill, stage, k, capture_cb=None):
    time.sleep(5.0)   # 远超 deadline
    return _mock_run_one(it, skill, stage, k)
tfa.run_one = _slow_run_one
mp2, md2, n2, per_item2, base_per2 = tfa.eval_paired(
    ITEMS, cand_skill="C-skill", base_skill="B-skill",
    stage="test_paired_timeout", k=K, eval_deadline_s=2)
# 1′ (2026-10-29) 语义: 超时放弃的题不再计 0, 而是整对剔除 (abandoned flag → _n_drop).
# 全超时 → deltas 空 → n=0, per_item={}, base_per={}. mean_pass/mean_delta 用 max(1,len)=0 兜底 → 0.0.
# (旧行为是计 0 + n=3, delta=0; 新行为是 n=0. 这是 1′ 修 abandoned/全异常拉偏 baseline→抬 Δ 偏置的正确改动.)
assert n2 == 0, f"全超时后所有配对应被剔除 (1′ abandoned-drop), n 应=0, got {n2}"
assert abs(md2) < 1e-9, f"空 deltas 兜底 mean_delta 应=0, got {md2}"
assert abs(mp2) < 1e-9, f"空 deltas 兜底 mean_pass 应=0, got {mp2}"
assert per_item2 == {}, f"全剔除后 per_item 应空 dict, got {per_item2}"
assert base_per2 == {}, f"全剔除后 base_per 应空 dict, got {base_per2}"
print(f"✅ Test B (超时放弃, 1′ abandoned-drop 语义): n={n2} (全剔除), mean_pass={mp2:.4f}, mean_delta={md2:+.4f}")

# ---- Test C: identity 候选 (cand==base, 纯函数级) ----
# 不用真实 API: mock 下 cand==base → pass_rate 相同表 → delta 恒 0
tfa.run_one = _mock_run_one
_SAME = {"q1": 0.625, "q2": 0.375, "q3": 0.875}
def _ident_run_one(it, skill, stage, k, capture_cb=None):
    qid = str(it["id"]); pr = _SAME.get(qid, 0.0)
    return {"id": it["id"], "pass_rate": pr, "n_pass": round(pr*k), "n_samples": k}
tfa.run_one = _ident_run_one
mp3, md3, n3, per_item3, _ = tfa.eval_paired(
    ITEMS, cand_skill="X", base_skill="Y",   # 同表 → cand_pr==base_pr
    stage="test_identity", k=K, eval_deadline_s=None)
assert abs(md3) < 1e-9, f"identity (mock 同表) mean_delta 应=0, got {md3}"
for q in ("q1", "q2", "q3"):
    assert abs(per_item3[q][0]) < 1e-9, f"identity {q} delta 应=0, got {per_item3[q]}"
# bootstrap_ci_lower 在恒 0 delta 上 → 应≈0 (不抛异常)
ci = ev.bootstrap_ci_lower(per_item3)
assert ci <= 1e-6, f"identity paired → bootstrap CI 应≈0, got {ci}"
print(f"✅ Test C (identity mock): mean_delta={md3:+.4f}, bootstrap_ci={ci:+.4f} (零噪声下 CI≈0)")

print("\n✅ eval_paired unit test 全通过 (3/3)")
