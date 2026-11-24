#!/usr/bin/env python3
"""#48 Pre-assessment — ≤5-call (per part) 动态 smoke (耗 flash, 串行).

承接 preassess_48_0api.py (0-API 闸全过)。本脚本做两件 0-API 做不到的事:
  Part A (Arm3 空 S₀ rewriter 可行性): rollout_group(item, s0="", g=2) 在 1 道 train fail 题上。
    验证: rewriter 在 s0="" 下能产出非空 cleaned 候选 (real=True), 不触发 n_real==0 的
    RuntimeError 中断, 且 B-eval (run_one) 能对空-scaffold 衍生的候选正常执行 + _evaluate_d5 评。
    重点: 空 scaffold 不会造成训练流程中断 (plan §A 副作用预案)。
  Part B (eval_independent 端到端): eval_independent 在 5 道混合题 (含 stdin + fn-call) 上跑
    固定 skill (真 S₀)。验证 API 调用 / D5 评测分流 (fn-call→_evaluate_fn_call, stdin→appsplus_evaluate)
    / infra 分类 (agent_ok) / 闭合校验 (valid+infra+missing==N) / 结果结构落盘。

调用预算 (账号级 20 RPM 串行, 必须独占, 见 memory/flash-eval-serialize-20261113.md):
  Part A: 2 rewriter (llm) + 2 B-eval (run_one) = 4 calls
  Part B: 5 run_one (eval_independent) + 可能 ≤5 infra retry (仅 agent_ok=False 触发) = 5~10 calls
  分段计, 各 ≤5 (B 含 retry 可能略超, 记录实际)。
  ⚠ 启动前必须 pgrep 确认无其它 flash 消费者 (trunc evolve / cap 扫描 等)。

env 由 launch shim (或本脚本前的 export 块) 提供: flash key 运行时从 yaml 读, 不打印明文。
"""
import json
import os
import sys
import time
import importlib.util

# ── 路径 (镜像 preassess_48_0api.py) ──────────────────────────────────────────
_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)
_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)
sys.path.insert(0, "/data/zhuldz/lunwen/data/OpenCodeInstruct")

# ⚠ flash env 必须在 import tf 模块前就绪 (模块顶 setdefault 才不会把空串固化)。
# launch shim 已 export; 此处仅 fallback 提示, 不读 key (key 只在 shim 里 python3 -c 读)。
_need = ["QWEN_CHAT_API_KEY", "QWEN_CHAT_BASE_URL", "QWEN_CHAT_MODEL"]
_missing_env = [k for k in _need if not os.environ.get(k)]
if _missing_env:
    # 不在这里读 yaml (避免脚本独立跑时也读, 仍可由 shim 包一层); 直接报错让用户走 shim。
    print(f"ERR: 缺 flash env {_missing_env}。请用 launch shim 或先 source flash env。", file=sys.stderr)
    print("    (shim: rl/train/verl1/skill_evolution/launch_tf_appsplus_routing.sh 的 env 块)", file=sys.stderr)
    sys.exit(2)

SCRIPT = "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py"
_spec = importlib.util.spec_from_file_location("tf_app", SCRIPT)
tf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tf)

# ── 调用计数器 (monkey-patch chat_target, 防止预算失控) ────────────────────────
# 对 #48 smoke 而言 chat_target 是唯一 flash 出口 (rewriter llm + run_one B-eval 都走它)。
_orig_chat_target = tf.chat_target
CALL_LOG = []  # [(stage, t0)] 精确计数


def _counted_chat_target(*args, **kwargs):
    stage = kwargs.get("stage", args[3] if len(args) > 3 else "?")
    CALL_LOG.append(stage)
    return _orig_chat_target(*args, **kwargs)


tf.chat_target = _counted_chat_target
# run_one 内部 import 的是 chat_target 名字; rollout_group/llm 走 tf.llm→tf.chat_target。
# 但 run_one 在模块内引用 chat_target 全局名 → 已被 patch (同模块命名空间)。
# eval_independent 通过 run_one → 同 patch 生效。✓

# ── 输入路径 (同 0-api 脚本) ───────────────────────────────────────────────────
S0_PATH = "/data/zhuldz/skillopt/skillopt/envs/appsplus/skills/initial.md"
SPLIT_DIR = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
RUN_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/appsplus-skillopt-flash-20261111-v2/train"
TRAIN_CACHE = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-formal-20261113/train_s0_cache.json"
OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/smoke_48_20261117"
os.makedirs(OUT_DIR, exist_ok=True)

s0 = open(S0_PATH, encoding="utf-8").read()
train_all = tf.load_split(SPLIT_DIR, "train", truncate=True)
test_all = tf.load_split(SPLIT_DIR, "test", truncate=True)


def _to_dict(cache):
    if isinstance(cache, dict):
        return cache
    return {str(r["id"]): r for r in cache}


s0_cache_train = _to_dict(json.load(open(TRAIN_CACHE, encoding="utf-8")))

print("=" * 72)
print("#48 Pre-assessment — ≤5-call/part 动态 smoke (flash deepseek-v4-flash)")
print("=" * 72)
print(f"[env] model={os.environ.get('QWEN_CHAT_MODEL')} cap={os.environ.get('QWEN_CHAT_MAX_TOKENS')} "
      f"rpm={os.environ.get('QWEN_CHAT_MAX_RPM')} temp={os.environ.get('QWEN_CHAT_TEMPERATURE')} "
      f"AIGC_USER={'set(❌)' if os.environ.get('QWEN_CHAT_AIGC_USER') else 'unset(✓)'}")
print(f"[输入] S₀={len(s0)} chars | train={len(train_all)} test={len(test_all)}")
print(f"[落盘] {OUT_DIR}")
print()

# 选 Part A 的题: train fail 池里取一道 (s0 hard=0)。0-API 已确认全 stdin, 这里取第一道 fail。
fail_items = [it for it in train_all
             if int(s0_cache_train.get(str(it["id"]), {}).get("hard", 0)) == 0]
partA_item = fail_items[0]
partA_s0_pass = s0_cache_train.get(str(partA_item["id"]), {}).get("pass_rate", 0.0)
print(f"[Part A] item id={partA_item['id']} diff={partA_item.get('difficulty')} "
      f"fn_call={tf._d5_is_fn_call_item(partA_item)} s0_hard=0 s0_pass={partA_s0_pass}")
print(f"[Part A] s0_train='' (Arm3 drop_scaffold), g=2, variant_temps=[0.7,0.8]")
print()

# ──────────────────────────────────────────────────────────────────────────────
# Part A: Arm3 空 S₀ — rollout_group(s0="") 不中断 + 产非空候选
# ──────────────────────────────────────────────────────────────────────────────
partA = {"ok": False, "n_real": 0, "abort": False, "variants": [], "calls": 0}
tA0 = time.time()
calls_before_A = len(CALL_LOG)
try:
    qid, variants = tf.rollout_group(
        partA_item, "", "", partA_s0_pass, 2, [0.7, 0.8])
    partA["qid"] = qid
    partA["n_real"] = sum(1 for v in variants if v.get("real"))
    partA["variants"] = [
        {"skill_len": len(v["skill"]), "pass": round(v["pass"], 3),
         "delta": round(v["delta"], 3), "hard": v["hard"],
         "agent_ok": v["agent_ok"], "real": v["real"],
         "rewriter_fail_reason": v.get("rewriter_fail_reason")}
        for v in variants
    ]
    partA["calls"] = len(CALL_LOG) - calls_before_A
    partA["ok"] = partA["n_real"] > 0   # 至少 1 个真实候选 = 未中断
    partA["abort"] = False
    print(f"[Part A] ✓ rollout_group 完成 (无 RuntimeError): n_real={partA['n_real']}/2, "
          f"calls={partA['calls']}, {time.time()-tA0:.1f}s")
    for i, v in enumerate(partA["variants"]):
        print(f"    var{i}: real={v['real']} skill_len={v['skill_len']} "
              f"pass={v['pass']} delta={v['delta']:+.3f} hard={v['hard']} "
              f"agent_ok={v['agent_ok']} reason={v['rewriter_fail_reason']}")
    if partA["n_real"] == 0:
        print("[Part A] ⚠ n_real=0: rewriter 在空 scaffold 下未产真实候选 — 触发 plan §A 预案 (最小 scaffold)")
        partA["ok"] = False
    else:
        print("[Part A] ✓ 空 scaffold 下 rewriter 仍产非空候选 — 训练流程不中断, Arm3 可正式跑")
except RuntimeError as e:
    partA["abort"] = True
    partA["calls"] = len(CALL_LOG) - calls_before_A
    print(f"[Part A] ✗ RuntimeError (n_real==0 中止): {e}")
    print("[Part A] → Arm3 需退化方案 (最小 scaffold), 不能直接跑空 S₀")
except Exception as e:
    partA["abort"] = True
    partA["calls"] = len(CALL_LOG) - calls_before_A
    partA["error"] = f"{type(e).__name__}: {e}"
    print(f"[Part A] ✗ 异常: {partA['error']}")

print()

# ──────────────────────────────────────────────────────────────────────────────
# Part B: eval_independent 端到端 (5 道混合题: stdin + fn-call)
# ──────────────────────────────────────────────────────────────────────────────
# 选 5 道: 至少 2 fn-call + 3 stdin, 从 test split 取 (test 有 30 fn-call)。
fn_call_items = [it for it in test_all if tf._d5_is_fn_call_item(it)]
stdin_items = [it for it in test_all if not tf._d5_is_fn_call_item(it)]
partB_items = (fn_call_items[:2] + stdin_items[:3])[:5]
print(f"[Part B] eval_independent 5 题 (fn-call={sum(tf._d5_is_fn_call_item(it) for it in partB_items)}, "
      f"stdin={sum(not tf._d5_is_fn_call_item(it) for it in partB_items)}), skill=真 S₀")
for it in partB_items:
    print(f"    id={it['id']} diff={it.get('difficulty')} fn_call={tf._d5_is_fn_call_item(it)}")
print()

partB = {"ok": False, "calls": 0}
tB0 = time.time()
calls_before_B = len(CALL_LOG)
try:
    res = tf.eval_independent(partB_items, s0, stage="tf_smoke_indep", infra_retries=1)
    partB["calls"] = len(CALL_LOG) - calls_before_B
    # 闭合校验已在函数内 assert; 此处取结果
    partB["n_total"] = res["n_total"]
    partB["n_valid"] = res["n_valid"]
    partB["n_agent_fail"] = res["n_agent_fail"]
    partB["n_missing"] = res["n_missing"]
    partB["hard_count"] = res["hard_count"]
    partB["pass_rate"] = round(res["pass_rate"], 4)
    partB["valid_ids"] = res["valid_ids"]
    partB["infra_ids"] = res["infra_ids"]
    partB["missing_ids"] = res["missing_ids"]
    partB["closure"] = (res["n_valid"] + res["n_agent_fail"] + res["n_missing"] == res["n_total"])
    # 每题分流记录
    partB["per_item"] = []
    fn_eval = stdin_eval = 0
    for it in partB_items:
        qid = str(it["id"])
        r = res["cand_by_id"].get(qid, {})
        is_fn = tf._d5_is_fn_call_item(it)
        if is_fn:
            fn_eval += 1
        else:
            stdin_eval += 1
        partB["per_item"].append({
            "id": qid, "fn_call": is_fn,
            "hard": r.get("hard"), "pass_rate": round(r.get("pass_rate", -3), 3),
            "agent_ok": r.get("agent_ok"),
            "evaluator_version": r.get("evaluator_version", ""),
            "fail_reason": (r.get("fail_reason") or "")[:120],
        })
    partB["fn_evaluated"] = fn_eval
    partB["stdin_evaluated"] = stdin_eval
    partB["ok"] = partB["closure"] and res["n_missing"] == 0
    print(f"[Part B] ✓ eval_independent 完成: valid={res['n_valid']} infra={res['n_agent_fail']} "
          f"missing={res['n_missing']} hard={res['hard_count']}/{res['n_valid']} "
          f"closure={partB['closure']} calls={partB['calls']} {time.time()-tB0:.1f}s")
    print(f"    分流: fn-call 评测={fn_eval} stdin 评测={stdin_eval} (D5 dispatch 自动)")
    for p in partB["per_item"]:
        print(f"    id={p['id']} fn={p['fn_call']} hard={p['hard']} pass={p['pass_rate']} "
              f"agent_ok={p['agent_ok']} reason={p['fail_reason']}")
    if res["n_agent_fail"] > 0:
        print(f"    ⚠ infra={res['n_agent_fail']} 题 agent_ok=False (529/timeout/网络), 已计入排除集")
    if not partB["closure"]:
        print(f"    ✗ 闭合校验失败! valid+infra+missing≠total")
    else:
        print("[Part B] ✓ 闭合校验通过 (valid+infra+missing==N), 落盘结构完整")
except AssertionError as e:
    partB["calls"] = len(CALL_LOG) - calls_before_B
    partB["ok"] = False
    partB["error"] = f"闭合 assert: {e}"
    print(f"[Part B] ✗ 闭合校验失败: {e}")
except Exception as e:
    partB["calls"] = len(CALL_LOG) - calls_before_B
    partB["ok"] = False
    partB["error"] = f"{type(e).__name__}: {e}"
    print(f"[Part B] ✗ 异常: {partB['error']}")

# ── 汇总 + 落盘 ──────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("smoke 汇总")
print("=" * 72)
total_calls = len(CALL_LOG)
stage_breakdown = {}
for s in CALL_LOG:
    stage_breakdown[s] = stage_breakdown.get(s, 0) + 1
print(f"  总 flash 调用 = {total_calls}  (按 stage: {stage_breakdown})")
print(f"  Part A (Arm3 空 S₀ rewriter): {'PASS' if partA.get('ok') else 'FAIL'} "
      f"(n_real={partA.get('n_real')}/2, abort={partA.get('abort')}, calls={partA.get('calls')})")
print(f"  Part B (eval_independent):     {'PASS' if partB.get('ok') else 'FAIL'} "
      f"(closure={partB.get('closure')}, calls={partB.get('calls')})")

smoke_result = {
    "ts": "2026-11-17", "total_calls": total_calls,
    "stage_breakdown": stage_breakdown,
    "partA_arm3_empty_s0": partA,
    "partB_eval_independent": partB,
    "verdict": {
        "arm3_empty_s0_feasible": bool(partA.get("ok")),
        "eval_independent_end_to_end": bool(partB.get("ok")),
        "can_proceed_to_formal": bool(partA.get("ok") and partB.get("ok")),
    },
}
out_path = os.path.join(OUT_DIR, "smoke_result.json")
json.dump(smoke_result, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\n落盘: {out_path}")
print()
if partA.get("ok") and partB.get("ok"):
    print("✅ smoke 通过 → 可启动正式 Arm1 → Arm3 (需用户批准 + flash 独占)")
else:
    print("❌ smoke 未全过 → 查看上面 FAIL 项, 修后再跑; 不得直接启动正式训练")
