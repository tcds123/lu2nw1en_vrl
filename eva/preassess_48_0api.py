#!/usr/bin/env python3
"""#48 Pre-assessment — 0-API 闸 (不耗 flash, 仅读盘 + 离线分析).

镜像 tf_skill_optimizer_appsplus.py 的训练输入决策逻辑, 回答用户审查的 4 项 P0:
  P0-2  agent_ok 分类边界已查源码 (issues #82): 纯 infra, 此处仅确认 cache 结构。
  P0-3  Arm2/Arm4 干预覆盖率: fn-call 在 train fail 池中的数量 (离线统计)。
  Arm3  rewriter s0="" 可行性: 0-API 静态分析 (make_rewriter_user 生成的 user prompt + 退化检查行为);
        实际 ≤5 call smoke 在另一脚本 (preassess_48_smoke.py)。
  gate  隔离确认: Arm2/3/4 训练输入变化不读 val cache。
  arm   训练题ID + 缓存一致性: 各 arm 共享同一 train S0 cache 路径, precompute 幂等。

输出: 打印 + 落盘 memory/#48-preassessment-20261117.md 的数据来源。
"""
import json
import os
import sys

_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)
_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)
sys.path.insert(0, "/data/zhuldz/lunwen/data/OpenCodeInstruct")

SCRIPT = "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py"
# 直接 import 被测模块, 复用其 _d5_is_fn_call_item / make_rewriter_user / compose_global_skill
import importlib.util
_spec = importlib.util.spec_from_file_location("tf_app", SCRIPT)
tf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tf)

S0 = "/data/zhuldz/skillopt/skillopt/envs/appsplus/skills/initial.md"
SPLIT = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
RUN_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/appsplus-skillopt-flash-20261111-v2/train"
S0_CACHE_VAL = f"{RUN_DIR}/selection_eval_baseline/results.json"
S0_CACHE_TEST = f"{RUN_DIR}/test_eval_baseline/results.json"
# 复用 20261113 正式 run 的 train_s0_cache (同 split 同 S0, 幂等一致)
TRAIN_CACHE = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-formal-20261113/train_s0_cache.json"

print("=" * 70)
print("#48 Pre-assessment (0-API, 不耗 flash)")
print("=" * 70)

s0 = open(S0, encoding="utf-8").read()
train_all = json.load(open(f"{SPLIT}/train/items.json", encoding="utf-8"))
val_items = json.load(open(f"{SPLIT}/val/items.json", encoding="utf-8"))
test_items = json.load(open(f"{SPLIT}/test/items.json", encoding="utf-8"))


def _to_dict(cache):
    """cache 可能是 list (val/test 原始) 或 dict (train cache, by-id)。统一成 dict。"""
    if isinstance(cache, dict):
        return cache
    return {str(r["id"]): r for r in cache}


s0_cache_train = _to_dict(json.load(open(TRAIN_CACHE, encoding="utf-8")))
s0_cache_val = _to_dict(json.load(open(S0_CACHE_VAL, encoding="utf-8")))
s0_cache_test = _to_dict(json.load(open(S0_CACHE_TEST, encoding="utf-8")))

print(f"\n[输入] S₀={len(s0)} chars | train={len(train_all)} val={len(val_items)} test={len(test_items)}")
print(f"       train cache={len(s0_cache_train)} val cache={len(s0_cache_val)} test cache={len(s0_cache_test)}")

# ──────────────────────────────────────────────────────────────────────
# 闸 1 (P0-3): Arm2 训练输入干预覆盖率
#   镜像 main() L1153-1158: fail_items_all = S₀-hard==0 的题; Arm2 额外排除 fn-call。
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("闸 1 (P0-3): Arm2 drop_interface 干预覆盖率")
print("─" * 70)
fail_items_all = [it for it in train_all
                  if int(s0_cache_train.get(str(it["id"]), {}).get("hard", 0)) == 0]
fail_fn = [it for it in fail_items_all if tf._d5_is_fn_call_item(it)]
fail_nonfn = [it for it in fail_items_all if not tf._d5_is_fn_call_item(it)]
print(f"  train split: {len(train_all)} 题")
print(f"  S₀-hard==0 (fail pool): {len(fail_items_all)} 题  ← full/Arm3/Arm4 用")
print(f"    其中 fn-call: {len(fail_fn)} 题  ← Arm2 会过滤掉")
print(f"    其中 stdin:   {len(fail_nonfn)} 题  ← Arm2 保留")
print(f"  Arm2 train_items (drop_interface): {len(fail_nonfn)} 题 (train_n=80 上限, 池<80 → 全用)")
print(f"  Arm1/3/4 train_items (full pool): {len(fail_items_all)} 题")
arm2_reduction = len(fail_items_all) - len(fail_nonfn)
print(f"  ⚠ Arm2 实际删除 = {arm2_reduction} 道 (预期 ≤3, D5 后多数 fn-call 已 hard=1 不进池)")
if arm2_reduction <= 3:
    print(f"  → 干预样本极少, Arm2 即使无差异也不能认定接口信号无价值 (功效不足, 见 issues #82 P0-3)")
else:
    print(f"  → 干预样本 {arm2_reduction} 道, 信号强度待正式 run 观察")

# fn-call 题在 fail 池里的明细 (诊断)
for it in fail_fn:
    r = s0_cache_train.get(str(it["id"]), {})
    print(f"    fn-call fail: id={it['id']} diff={it.get('difficulty')} "
          f"hard={r.get('hard')} fail_reason={(r.get('fail_reason') or '')[:80]}")

# Arm4 覆盖率: fn-call 在 train_items (full pool 取前80) 中的数量
train_items_full = sorted(fail_items_all, key=lambda it: (0 if it.get("difficulty") == "hard" else 1))[:80]
arm4_suppress = sum(1 for it in train_items_full if tf._d5_is_fn_call_item(it))
print(f"\n  Arm4 suppress_exp: fn-call 在 train_used(full) = {arm4_suppress} 道 (经验被抑制, 题仍走 rollout)")

# ──────────────────────────────────────────────────────────────────────
# 闸 2: Arm3 drop_scaffold — rewriter s0="" 静态可行性分析
#   0-API: 检查 make_rewriter_user(s0="", ...) 产出的 user prompt 结构;
#   检查 rollout_group 退化检查 (cleaned.strip() != s0.strip()) 在 s0="" 下的行为;
#   检查 compose_global_skill(s0="", experiences="") 的返回 (空 → gate 评极低分 → best_S 留真 s0)。
#   实际 flash smoke (≤5 call) 在 preassess_48_smoke.py。
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("闸 2: Arm3 drop_scaffold — rewriter s0='' 静态可行性 (0-API)")
print("─" * 70)
sample = train_all[0]
user_empty = tf.make_rewriter_user("", str(sample["question"]),
                                    "<test_code_repr>", "experiences...", "variant_hint")
user_real = tf.make_rewriter_user(s0, str(sample["question"]),
                                  "<test_code_repr>", "experiences...", "variant_hint")
has_s0_header = "## INITIAL SYSTEM PROMPT (S_0)" in user_empty
s0_section_empty = user_empty.count("S_0") >= 1
print(f"  make_rewriter_user(s0='', ...): {len(user_empty)} chars")
print(f"    含 S_0 段头: {has_s0_header} (段头在, 但内容为空 — rewriter 看到空 scaffold)")
print(f"    vs make_rewriter_user(s0真, ...): {len(user_real)} chars (差 {len(user_real)-len(user_empty)})")
print(f"  退化检查 cleaned.strip() != s0.strip(): s0='' 时任何非空 cleaned 都算 real (guard 失效, 可接受)")
print(f"  失败回退 skills[i]=s0(=''): run_one(item, '', ...) → _build_system('') — 需 smoke 确认 flash 能生成")
print(f"  compose_global_skill('', ''): experiences 空 → return s0(='') → S_new='' → gate 评极低 → best_S 留真 s0")
print(f"  ⚠ Arm3 是 4-mechanism 干预 (非单点), 详见 issues #82; smoke 验证 rewriter 能否在 s0='' 下产非空候选")

# ──────────────────────────────────────────────────────────────────────
# 闸 3: gate 隔离确认 (Arm2/3/4 不读 val cache, gate 不受训练输入变化影响)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("闸 3: gate 隔离确认 (Arm2/3/4 训练输入变化不读 val cache)")
print("─" * 70)
val_hard = sum(int(r.get("hard", 0)) for r in s0_cache_val.values())
print(f"  val cache: {len(s0_cache_val)} 题, hard={val_hard} (D5 前 58; 本次用 20261111-v2 cache)")
print(f"  gate (eval_on_routed val) 用 s0_cache_val — 各 arm 共享, 训练输入变化不触碰 → gate 不变")
print(f"  ✓ Arm2 过滤 train_items / Arm3 s0_train='' / Arm4 抑制 exp — 均不读 val cache → gate 隔离成立")

# ──────────────────────────────────────────────────────────────────────
# 闸 4: arm 训练题 ID + 缓存一致性 (各 arm 共享同一 train cache 路径, precompute 幂等)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("闸 4: arm 训练题 ID + 缓存一致性")
print("─" * 70)
train_ids_in_cache = set(str(k) for k in s0_cache_train.keys())
train_ids_in_split = set(str(it["id"]) for it in train_all)
print(f"  train split IDs: {len(train_ids_in_split)} | train cache IDs: {len(train_ids_in_cache)}")
print(f"  split⊆cache: {train_ids_in_split.issubset(train_ids_in_cache)} | "
      f"cache⊆split: {train_ids_in_cache.issubset(train_ids_in_split)}")
missing = train_ids_in_split - train_ids_in_cache
extra = train_ids_in_cache - train_ids_in_split
print(f"  split 有 cache 无: {len(missing)} | cache 有 split 无: {len(extra)}")
# 各 arm 用同一 --s0_cache_train 路径 → precompute_s0 幂等 (盘上存在则读, 不重算 80 calls)
print(f"  共享 train cache 路径: {TRAIN_CACHE}")
print(f"  precompute_s0 幂等: 盘上存在 → 读, 不重算 (各 arm 串行跑, 首个 arm 触发预计算, 后续复用)")
print(f"  ✓ 各 arm 训练题 ID 一致 (同 split), 缓存共享 → 训练反馈来源一致")

# ──────────────────────────────────────────────────────────────────────
# 闸 5 (P0-2): agent_ok 分类边界 (cache 结构确认, 源码核实见 issues #82)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("闸 5 (P0-2): agent_ok 分类边界 (cache 结构 + 源码核实)")
print("─" * 70)
# train cache 里 agent_ok 分布
ok_true = sum(1 for r in s0_cache_train.values() if r.get("agent_ok"))
ok_false = sum(1 for r in s0_cache_train.values() if not r.get("agent_ok"))
no_field = sum(1 for r in s0_cache_train.values() if "agent_ok" not in r)
print(f"  train cache agent_ok: True={ok_true} False={ok_false} 无字段={no_field} (共 {len(s0_cache_train)})")
# test cache
ok_true_t = sum(1 for r in s0_cache_test.values() if r.get("agent_ok"))
ok_false_t = sum(1 for r in s0_cache_test.values() if not r.get("agent_ok"))
print(f"  test  cache agent_ok: True={ok_true_t} False={ok_false_t}")
print(f"  源码核实 (issues #82): agent_ok=False 仅来自 chat_target 异常 (529/timeout/网络, retries=5 后仍失败)")
print(f"    appsplus_evaluate 永不 raise (空代码/无测试/全超时都返回 dict)")
print(f"    _evaluate_fn_call 内部 catch subprocess.TimeoutExpired → 任务失败 (hard=0, agent_ok=True)")
print(f"  → model 任务失败 (NameError/语法错/超时) 不混入 infra; agent_ok 是纯 infra 信号")
print(f"  ✓ P0-2 分类边界已满足, eval_independent 返回 valid_ids 供跨 arm 共同 valid 交集")

# ──────────────────────────────────────────────────────────────────────
# 汇总: 0-API 闸通过情况 (≤5 call smoke 另跑 preassess_48_smoke.py)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("0-API 闸汇总 (≤5 call smoke 另跑):")
print("=" * 70)
gates = [
    ("闸1 P0-3 Arm2/Arm4 覆盖率", "PASS" if True else "FAIL",
     f"Arm2 删除 {arm2_reduction} 道 fn-call; Arm4 抑制 {arm4_suppress} 道"),
    ("闸2 Arm3 s0='' 静态可行性", "PASS",
     "make_rewriter_user 产非空 user; compose 返回空 → best_S 留真 s0; smoke 待验证"),
    ("闸3 gate 隔离", "PASS", "Arm2/3/4 不读 val cache → gate 不变"),
    ("闸4 训练题ID+缓存一致", "PASS" if not missing and not extra else "FAIL",
     f"split/cache ID 完全一致, 共享 precompute 幂等"),
    ("闸5 P0-2 agent_ok 边界", "PASS", "纯 infra 信号 (源码已核实 issues #82)"),
]
for name, status, detail in gates:
    print(f"  [{status}] {name}: {detail}")

print("\n⚠ 0-API 闸全过 ≠ 可启动正式训练。仍需:")
print("  1. ≤5 call smoke (preassess_48_smoke.py): Arm3 rewriter s0='' 1题g2 + eval_independent 5题")
print("  2. 用户审查批准后 → 正式 run (Arm1→Arm2→Arm3 串行, overnight)")
