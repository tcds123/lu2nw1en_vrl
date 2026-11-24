#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 修复冒烟验收 (task #50 / codex 2026-11-14 review)
====================================================

最小闭合验证: rollout_group rewriter 是否真正执行, 温度是否真正生效。

验收 4 点 (用户 2026-11-16 明确):
  1. 两次真实 rewriter API 请求 (非 S₀, 非 TypeError 静默回退)
  2. 两次请求分别使用预期温度 (0.7 / 0.8, 即 variant_temps[0]/[1])
  3. 候选通过 validity check (clean_skill_text 后非空; real=True)
  4. 候选进入下游 eval, 未被 fallback 覆盖 (results 非空, pass_rate 有值)

不要求两个候选彼此不同 (温度差异不保证输出不同, flash temp≠0 但非完美发散)。

证据落盘:
  - smoke_d2_request.log   : D2-debug 温度轨迹 (env-gated, 临界区实测 applied/saved/restored)
  - smoke_d2_candidates.json: g 个候选完整记录 (skill/pass/delta/temp/real/fail_reason)

用法:
  cd rl/train/verl1/skill_evolution
  source /data/zhuldz/lunwen/lunwen/bin/activate
  bash <(python3 - <<'PY'
... (env 见 launch shim, 此处由外部 wrapper 设)
PY
  )
或直接: 先 source launch shim 的 env (用 -c 读 key), 再 TF_D2_DEBUG=1 python smoke_d2_rewriter_fix.py
"""
import json
import os
import sys

# ── 路径 (与主脚本同目录) ───────────────────────────────────────────────────────
SKILLOPT = "/data/zhuldz/skillopt"
_LUNWEN = "/data/zhuldz/lunwen"
for p in (SKILLOPT, _LUNWEN, "/data/zhuldz/lunwen/data/OpenCodeInstruct"):
    if p not in sys.path:
        sys.path.insert(0, p)

# env (与主脚本 setdefault 一致; launch shim 已 export, 此处仅 fallback)
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/llm-openai-api")
os.environ.setdefault("QWEN_CHAT_MODEL", "deepseek-v4-flash")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "16384")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "20")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

# 开启 D2-debug (临界区温度轨迹)
os.environ["TF_D2_DEBUG"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "_smoke_d2_out")
os.makedirs(OUT_DIR, exist_ok=True)

# 重定向 D2-debug print 到日志文件 (llm 里用 print+flush)
REQ_LOG = os.path.join(OUT_DIR, "smoke_d2_request.log")
_req_fp = open(REQ_LOG, "w", encoding="utf-8")

class _TeeStdout:
    """把 stdout 同时写原终端和日志 (捕获 llm 的 D2-debug print)。"""
    def __init__(self, real, fp):
        self.real = real
        self.fp = fp
        self._closed = False
    def close(self):
        self._closed = True
        try:
            self.fp.close()
        except Exception:
            pass
    def write(self, s):
        self.real.write(s)
        if not self._closed:
            try:
                self.fp.write(s)
                self.fp.flush()
            except Exception:
                pass
        return len(s)
    def flush(self):
        self.real.flush()
        if not self._closed:
            try:
                self.fp.flush()
            except Exception:
                pass

sys.stdout = _TeeStdout(sys.__stdout__, _req_fp)

# ── import (在 env 之后) ──────────────────────────────────────────────────────
from skillopt.model.qwen_backend import TARGET_CONFIG  # noqa: E402
import tf_skill_optimizer_appsplus as M  # noqa: E402


def main():
    print("=" * 60)
    print("D2 修复冒烟验收 — rollout_group rewriter (g=2, 1 train-fail 题)")
    print("=" * 60)

    # 验收前置: TARGET_CONFIG 是模块级单例, 导入时从 QWEN_CHAT_TEMPERATURE=0.0 冻结
    print(f"\n[PRE] TARGET_CONFIG.temperature (import-frozen) = {TARGET_CONFIG.temperature}")
    print(f"[PRE] 期望 = 0.0 (launch shim QWEN_CHAT_TEMPERATURE=0.0)")

    # 取一道真实 S₀-fail 题 (id=6937, 来自历史 train cache, hard=0)
    items = json.load(open(
        "/data/zhuldz/skillopt/data/appsplus_capped_seed42/train/items.json", encoding="utf-8"))
    item = next(it for it in items if str(it["id"]) == "6937")
    # 截断 test_code (与 load_split 一致)
    item["test_code"] = M.truncate_test_code(item.get("test_code", ""))

    s0 = open("/data/zhuldz/skillopt/skillopt/envs/appsplus/skills/initial.md", encoding="utf-8").read().strip()
    s0_pass = 0.0  # S₀-fail 题, s0_pass=0
    g = 2
    variant_temps = [0.7 + 0.1 * (i % 3) for i in range(g)]  # [0.7, 0.8]
    print(f"\n[INPUT] qid={item['id']} difficulty={item.get('difficulty')} "
          f"question_len={len(str(item['question']))}")
    print(f"[INPUT] g={g} variant_temps={variant_temps}")
    print(f"[INPUT] s0_len={len(s0)}")

    # ── 执行 rollout_group (修复后: 串行 rewriter + 显式失败 + abort) ───────────
    print(f"\n[RUN] 调用 rollout_group (修复后)...")
    try:
        qid, variants = M.rollout_group(item, s0, experiences="", s0_pass=s0_pass,
                                        g=g, variant_temps=variant_temps)
    except RuntimeError as e:
        print(f"\n[ABORT] rollout_group 中止 (0 真实候选): {e}")
        print("  → 这是修复后的预期行为 (历史 bug 会静默继续全 S₀); 冒烟需检查 API key/网络。")
        sys.stdout.close()   # 关闭 tee (防 sys.exit flush 误写已关文件)
        sys.exit(2)

    print(f"\n[DONE] qid={qid} 返回 {len(variants)} 个 variants")

    # ── 验收 4 点 ────────────────────────────────────────────────────────────────
    verdict = {}
    print("\n" + "=" * 60)
    print("验收")
    print("=" * 60)

    # 点1: 两次真实 rewriter API 请求 (非 S₀ 回退)
    n_real = sum(1 for v in variants if v.get("real"))
    c1 = n_real >= 1  # 至少 1 个真实候选 (不要求 2 个都不同, 但要求 ≥1 真实非 TypeError-回退)
    verdict["1_real_api_calls"] = {
        "pass": bool(c1), "n_real": n_real, "g": g,
        "note": "≥1 真实 rewriter 候选 (非 TypeError 静默回退 S₀); 全失败会被 abort 拦截, 不会到这",
    }
    print(f"\n[1] 真实 API 请求: n_real={n_real}/{g}  → {'✅ PASS' if c1 else '❌ FAIL'}")
    for i, v in enumerate(variants):
        is_s0 = (v["skill"].strip() == s0.strip())
        print(f"    var[{i}] real={v['real']} is_s0={is_s0} "
              f"skill_len={len(v['skill'])} fail_reason={v.get('rewriter_fail_reason')}")

    # 点2: 两次请求分别使用预期温度 (0.7 / 0.8) — 从 D2-debug 日志解析 applied temp
    applied_temps = []
    with open(REQ_LOG, encoding="utf-8") as f:
        log_text = f.read()
    import re
    # 匹配: applied=0.7 / applied=0.8 (tf_rewriter stage only)
    for m in re.finditer(r"\[D2-debug\] llm stage=tf_rewriter req_temp=([\d.]+) applied=([\d.]+) saved=([\d.]+)", log_text):
        applied_temps.append({"req": float(m.group(1)), "applied": float(m.group(2)), "saved": float(m.group(3))})
    c2 = (len(applied_temps) == g and
          all(a["applied"] == a["req"] for a in applied_temps) and
          all(a["saved"] == 0.0 for a in applied_temps))
    verdict["2_temperature_matched"] = {
        "pass": bool(c2), "applied_temps": applied_temps, "expected": variant_temps,
        "note": "每个候选 applied==req (温度真生效), 且 saved==0.0 (原值正确保存)",
    }
    print(f"\n[2] 温度匹配: {len(applied_temps)} 条 rewriter 温度记录, expected={variant_temps}")
    for a in applied_temps:
        print(f"    req={a['req']} applied={a['applied']} saved={a['saved']} "
              f"{'✅' if a['applied']==a['req'] and a['saved']==0.0 else '❌'}")
    print(f"    → {'✅ PASS' if c2 else '❌ FAIL'}")

    # 点3: 候选通过 validity check (real=True, clean 后非空)
    c3 = all(v.get("real") for v in variants) and all(len(v["skill"]) > 0 for v in variants)
    verdict["3_validity_check"] = {
        "pass": bool(c3),
        "all_real": all(v.get("real") for v in variants),
        "all_nonempty": all(len(v["skill"]) > 0 for v in variants),
    }
    print(f"\n[3] validity: all_real={all(v.get('real') for v in variants)} "
          f"all_nonempty={all(len(v['skill'])>0 for v in variants)} → {'✅ PASS' if c3 else '❌ FAIL'}")

    # 点4: 候选进入下游 eval, 未被 fallback 覆盖 (results 非空, pass_rate 有值)
    # ⚠️ variant dict 的键是 "pass" (rollout_group 返回契约), 不是 "pass_rate"
    #    (run_one 返回 "pass_rate", 但 rollout_group 包成 variant 时改名 "pass")
    c4 = all(v.get("pass") is not None and v.get("hard") is not None for v in variants)
    verdict["4_downstream_eval"] = {
        "pass": bool(c4),
        "pass_rates": [v["pass"] for v in variants],
        "hards": [v["hard"] for v in variants],
        "deltas": [v["delta"] for v in variants],
    }
    print(f"\n[4] 下游 eval: pass_rates={[v['pass'] for v in variants]} "
          f"hards={[v['hard'] for v in variants]} deltas={[round(v['delta'],3) for v in variants]}")
    print(f"    → {'✅ PASS' if c4 else '❌ FAIL'}")

    # ── 还原检查: 临界区结束后 TARGET_CONFIG.temperature 应恢复 0.0 ─────────────
    restored = TARGET_CONFIG.temperature
    c5 = (restored == 0.0)
    verdict["5_config_restored"] = {"pass": bool(c5), "restored_temp": restored, "expected": 0.0}
    print(f"\n[5] 临界区后还原: TARGET_CONFIG.temperature={restored} (期望 0.0) "
          f"→ {'✅ PASS' if c5 else '❌ FAIL'}")

    # ── 落盘候选记录 ─────────────────────────────────────────────────────────────
    cand_path = os.path.join(OUT_DIR, "smoke_d2_candidates.json")
    record = {
        "qid": qid, "g": g, "variant_temps": variant_temps,
        "variants": [{"i": i, "skill": v["skill"][:500] + ("..." if len(v["skill"]) > 500 else ""),
                      "skill_len": len(v["skill"]), "pass_rate": v["pass"],
                      "hard": v["hard"], "delta": v["delta"], "temp": v["temp"],
                      "real": v["real"], "rewriter_fail_reason": v.get("rewriter_fail_reason")}
                     for i, v in enumerate(variants)],
        "verdict": verdict,
        "all_pass": all(v["pass"] for v in verdict.values()),
    }
    json.dump(record, open(cand_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    all_pass = all(v["pass"] for v in verdict.values())
    print(f"总验收: {'✅ ALL PASS' if all_pass else '❌ HAS FAIL'}")
    print(f"  request log : {REQ_LOG}")
    print(f"  candidates  : {cand_path}")
    print("=" * 60)

    sys.stdout.close()   # 关闭 tee (防 sys.exit flush 误写已关文件)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
