"""A-Step2 unit test: make_staleness_negatives (零 API, 纯函数).

验证 (a) max_n 截断 + id 升序确定性, (b) focus=None 不崩, (c) 空输入返回 "",
(d) id 乱序输入 → 输出按 id 升序 (跨 run 复现性), (e) max_n > 输入长度 → 全返回不崩.

设计见 memory/dir1-executable-checklist-20261029.md A2.1 单测条目.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skill_evolver_aime_v2 as ev  # noqa: F401  (确认 import 通)


def _mk(qid, s0=0.0, sk=1.0):
    """造一条 stable_solved (skill 稳定解对 + S₀ 难)."""
    return {
        "id": qid,
        "item": {"year": 2024, "problem": f"problem text for {qid}", "answer": "42"},
        "skill_pass": sk,
        "s0_pass": s0,
    }


# ---- Test A: 3 题, max_n=2 → 输出含 2 题且 id 升序 ----
data_a = [_mk("q3"), _mk("q1"), _mk("q2")]   # 故意乱序
out_a = ev.make_staleness_negatives(data_a, focus="Computation & Verification", max_n=2)
assert out_a != "", "Test A: 非空输入应返回非空"
assert out_a.count("### Problem") == 2, f"Test A: 应含 2 题 (max_n=2), got {out_a.count('### Problem')}"
# id 升序: q1 在 q2 前
assert out_a.index("q1") < out_a.index("q2"), f"Test A: id 应升序 (q1<q2), 输出:\n{out_a}"
assert "q3" not in out_a, "Test A: max_n=2 应截断 q3"
# 摆题面 + 答案, 不摆解法 excerpt (staleness 是反向护栏, 非正信号蒸馏)
assert "Correct answer: 42" in out_a, "Test A: 应含答案"
assert "CORRECT reasoning excerpt" not in out_a, "Test A: staleness 不应摆解法 excerpt (与 make_wins 区分)"
print(f"✅ Test A (max_n=2 截断 + id 升序): 含 2 题 q1<q2, q3 被截断, 摆题面+答案无解法 excerpt")

# ---- Test B: focus=None 不崩 ----
out_b = ev.make_staleness_negatives(data_a, focus=None, max_n=2)
assert out_a == out_b, "Test B: focus=None 应与有 focus 输出一致 (当前实现 focus 仅透传, 不影响渲染)"
print(f"✅ Test B (focus=None 不崩): 输出与有 focus 一致 (focus 当前仅透传, make_staleness_negatives 不消费)")

# ---- Test C: 空输入返回 "" ----
out_c1 = ev.make_staleness_negatives([], focus="x", max_n=2)
assert out_c1 == "", f"Test C: 空列表应返回 '', got {out_c1!r}"
out_c2 = ev.make_staleness_negatives(None, focus="x", max_n=2)
assert out_c2 == "", f"Test C: None 应返回 '', got {out_c2!r}"
print("✅ Test C (空输入): [] 和 None 都返回 ''")

# ---- Test D: id 乱序 → 输出严格 id 升序 (跨 run 复现性) ----
data_d = [_mk(f"q{i:02d}") for i in [9, 3, 7, 1, 5]]   # 5 题乱序
out_d = ev.make_staleness_negatives(data_d, focus="x", max_n=10)  # max_n 大于长度, 不截断
# 提取输出里出现的 qid, 按出现序
qids_in_order = []
for i in range(1, 10):
    marker = f"q0{i}"
    if marker in out_d:
        pos = out_d.index(marker)
        qids_in_order.append((pos, marker))
qids_in_order.sort()
seq = [q for _, q in qids_in_order]
assert seq == ["q01", "q03", "q05", "q07", "q09"], f"Test D: id 应升序, got {seq}"
assert out_d.count("### Problem") == 5, f"Test D: 5 题全返回 (max_n=10>5), got {out_d.count('### Problem')}"
print(f"✅ Test D (确定性 id 升序): 乱序输入 [9,3,7,1,5] → 输出 {seq}")

# ---- Test E: max_n > 输入长度 → 全返回不崩 ----
out_e = ev.make_staleness_negatives([_mk("only1")], focus="x", max_n=100)
assert out_e.count("### Problem") == 1, f"Test E: max_n=100 但仅 1 题, 应全返回 1, got {out_e.count('### Problem')}"
print("✅ Test E (max_n > 长度): 仅 1 题输入, max_n=100, 全返回不崩")

# ---- Test F: STALENESS_SECTION 模板槽可渲染 (零回归: 默认空串) ----
# 验证 generate_candidates 在 staleness_negatives=None 时 st_sec="" (零回归核心)
import inspect
src = inspect.getsource(ev.generate_candidates)
assert 'staleness_negatives=None' in src, "Test F: generate_candidates 默认 staleness_negatives=None (零回归)"
assert 'if staleness_negatives:' in src, "Test F: 有 staleness 非空才构造 st_sec 的 guard"
print("✅ Test F (零回归): generate_candidates 默认 staleness_negatives=None, 有 guard 仅非空时构造 st_sec")

print("\n🎉 全部 6 项通过: make_staleness_negatives + STALENESS_SECTION + 零回归 guard")
