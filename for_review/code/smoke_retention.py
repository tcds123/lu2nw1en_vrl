#!/usr/bin/env python3
"""通道 B 冒烟: 在 5 道 S₀-pass 题上独立重跑候选 skill, 验证 eval_retention 走通。

不跑 evolver 主循环 — 直接复用已产出的 best_skill (formal run) + SkillOpt v2 test cache。
验证项:
  1. cand_by_id 5 条, 每条含 id/difficulty/hard/soft/pass_rate/agent_ok/fail_reason
  2. n_preserved + n_regress == N (5)  (S₀-pass 题集上 cand_only 结构性 = 0)
  3. mcnemar 方向: a_only == n_regress
  4. retention_rate + regress_rate == 1.0
"""
import json
import os
import sys

# ── 路径自定位 (standalone: for_review/code/ 下平铺; 源仓库则用 /data/zhuldz 安装路径) ──
_HERE = os.path.dirname(os.path.abspath(__file__))           # for_review/code/ (或源 skill_evolution/)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PKG_ROOT = os.path.dirname(_HERE)                            # for_review/
_SKILLOPT_FALLBACK = "/data/zhuldz/skillopt"
if os.path.isdir(os.path.join(_SKILLOPT_FALLBACK, "skillopt")) and _SKILLOPT_FALLBACK not in sys.path:
    sys.path.insert(0, _SKILLOPT_FALLBACK)
_OC_FALLBACK = "/data/zhuldz/lunwen/data/OpenCodeInstruct"
if os.path.isdir(_OC_FALLBACK) and _OC_FALLBACK not in sys.path:
    sys.path.insert(0, _OC_FALLBACK)

os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")

from tf_skill_optimizer_appsplus import (  # noqa: E402
    eval_retention,
    mcnemar_paired,
    load_split,
)

# 包内默认; 源仓库对应 /data/zhuldz/skillopt/data/... 与 .../crossmodel_baseline/...
SPLIT_DIR = os.environ.get(
    "APPSPLUS_DATA_DIR", os.path.join(_PKG_ROOT, "data", "appsplus_capped_seed42"))
_V2_CACHE = os.environ.get(
    "APPSPLUS_V2_CACHE", os.path.join(_PKG_ROOT, "outputs", "appsplus_v2_cache"))
S0_CACHE_TEST = os.environ.get(
    "APPSPLUS_S0_CACHE_TEST", os.path.join(_V2_CACHE, "test_eval_baseline", "results.json"))
# BEST_SKILL: 我们 evolver 正式跑产物 (tf_global_skill.md); standalone 暂用 SkillOpt 基线 best_skill.md 兜底。
# 若 evolver 正式 run 已落盘进包, 用 --best_skill 或环境变量 APPSPLUS_BEST_SKILL 指向它。
_BEST_SKILL_DEFAULT = os.path.join(_V2_CACHE, "best_skill.md")
BEST_SKILL = os.environ.get("APPSPLUS_BEST_SKILL", _BEST_SKILL_DEFAULT)

# ── 加载 ──────────────────────────────────────────────────────────────────────
test_items = load_split(SPLIT_DIR, "test")
s0_cache_test = {str(r["id"]): r for r in json.load(open(S0_CACHE_TEST, encoding="utf-8"))}
with open(BEST_SKILL, encoding="utf-8") as f:
    best_S = f.read().strip()

n_s0pass = sum(1 for r in s0_cache_test.values() if int(r["hard"]) == 1)
print(f"📄 best_skill: {len(best_S)} chars | test cache: {len(s0_cache_test)} 题 (S₀-pass={n_s0pass})")
print(f"   冒烟: 取前 5 道 S₀-pass 独立重跑候选 skill (flash, cap=16384, temp=0)...\n")

# ── 冒烟: eval_retention limit=5 ─────────────────────────────────────────────
ret = eval_retention(test_items, best_S, s0_cache_test, stage="tf_smoke_retention", limit=5)

# ── 核对 ──────────────────────────────────────────────────────────────────────
bp_r, bf_r, s0_only, cand_only, n_common, p = ret["mcnemar"]
ok = True

# 1. cand_by_id 条数 (全部样本都有结果, 含 infra)
n_cand = len(ret["cand_by_id"])
check1 = (n_cand == 5)
print(f"[1] cand_by_id 条数 = {n_cand} (期望 5, 含 infra) → {'✅' if check1 else '❌'}")
ok &= check1

# 1b. 每条字段完整
if n_cand:
    sample = next(iter(ret["cand_by_id"].values()))
    need = {"id", "difficulty", "hard", "soft", "pass_rate", "agent_ok", "fail_reason"}
    have = set(sample.keys())
    check1b = need.issubset(have)
    print(f"[1b] 逐条字段含必需 → {'✅' if check1b else '❌'}")
    ok &= check1b

# 2. 闭合性: n_valid + n_agent_fail + n_missing == N
check2 = (ret["n_valid"] + ret["n_agent_fail"] + ret["n_missing"] == ret["n_s0_pass"])
print(f"[2] 闭合 valid({ret['n_valid']})+fail({ret['n_agent_fail']})+missing({ret['n_missing']})==N({ret['n_s0_pass']}) → {'✅' if check2 else '❌'}")
ok &= check2

# 3. McNemar 只在 valid 上: n_preserved + n_regress == n_valid
check3 = (ret["n_preserved"] + ret["n_regress"] == ret["n_valid"])
print(f"[3] McNemar 分母: preserved({ret['n_preserved']})+regress({ret['n_regress']})==n_valid({ret['n_valid']}) → {'✅' if check3 else '❌'}")
ok &= check3

# 4. cand_only == 0 (S₀-pass 题集上 cand 不可能 pass 而 S₀ fail, 结构性 0)
check4 = (cand_only == 0)
print(f"[4] cand_only (S₀fail→candpass) = {cand_only} (结构性 0) → {'✅' if check4 else '❌'}")
ok &= check4

# 5. mcnemar 方向: a_only (S₀pass→candfail) == n_regress
check5 = (s0_only == ret["n_regress"])
print(f"[5] mcnemar a_only({s0_only}) == n_regress({ret['n_regress']}) → {'✅' if check5 else '❌'}")
ok &= check5

# 6. retention_rate + regress_rate == 1.0 (在 n_valid 分母上)
check6 = abs(ret["retention_rate"] + ret["regress_rate"] - 1.0) < 1e-9
print(f"[6] retention({ret['retention_rate']:.4f})+regress({ret['regress_rate']:.4f})==1.0 (valid 分母) → {'✅' if check6 else '❌'}")
ok &= check6

# 7. infra 隔离: n_agent_fail 样本不在 McNemar 分母里
check7 = (ret["n_agent_fail"] == 0 or ret["n_valid"] < ret["n_s0_pass"] or ret["n_agent_fail"] == 0)
# 若有 infra_fail, 则 n_valid < N (已排除); 若无 infra_fail, n_valid==N。两种都正常
infra_excluded = (ret["n_agent_fail"] == 0) or (ret["n_valid"] == ret["n_s0_pass"] - ret["n_agent_fail"] - ret["n_missing"])
print(f"[7] infra 隔离: n_agent_fail={ret['n_agent_fail']} valid={ret['n_valid']} (infra 已排除出 McNemar) → {'✅' if infra_excluded else '❌'}")
ok &= infra_excluded

print(f"\n{'='*50}")
print(f"  冒烟结果: N={ret['n_s0_pass']} valid={ret['n_valid']} infra_fail={ret['n_agent_fail']} missing={ret['n_missing']}")
print(f"  Retention: preserved={ret['n_preserved']} regress={ret['n_regress']} "
      f"retention={ret['retention_rate']*100:.1f}% regress_rate={ret['regress_rate']*100:.1f}%")
print(f"  McNemar(valid-only): both_pass={bp_r} both_fail={bf_r} regress={s0_only} cand_rescue={cand_only} p={p:.3e} n_common={n_common}")
print(f"  逐题 (id | hard | agent_ok | pass_rate | status | fail_reason):")
for qid, r in sorted(ret["cand_by_id"].items()):
    status = "valid" if r["agent_ok"] else "infra_invalid"
    print(f"    {qid} | hard={r['hard']} | agent_ok={r['agent_ok']} | pass={r['pass_rate']:.3f} | {status} | {r['fail_reason'][:50]}")
print(f"\n  {'✅ 冒烟通过 — eval_retention (infra 隔离 + 闭合 + valid-only McNemar) 走通, 可跑全量' if ok else '❌ 冒烟失败 — 核对项未过'}")
sys.exit(0 if ok else 1)
