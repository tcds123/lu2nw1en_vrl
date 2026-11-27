#!/usr/bin/env python
# SkillOpt 同协议对照评测: skill_v0002 vs S₀ on test=106 (2026-11-27).
# 与 Run A/B 完全同口径: V3.2, k=3, shuffle_seed=0, paired_rescore, EVAL_TEMP, 同 split(1-3-6).
# 评测任意 skill 路径 (SkillOpt 臂 / 旧 Evolver / 任意 baseline skill), 输出独立目录 + 独立 budget 账本.
#
# 用法: bash (自动载入 key + _env.sh)
#   bash rl/train/verl1/skill_evolution/launch_eval_skillopt.sh
# ⚠️ 付费 API (~636 calls: 106题×k3×2臂 paired). Run B 完成后启动 (不抢 28RPM 限流).
import json
import os
import sys
import time

_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")

from tf_global_v2_math import (  # noqa: E402
    eval_paired, bootstrap_ci_lower, load_split, is_meta_item,
    set_temp, EVAL_TEMP, md5,
)
import skill_evolver_math as se  # noqa: E402
SPLIT_DEFAULT = se.SPLIT_DEFAULT
S0_DEFAULT = se.S0_DEFAULT
import budget_guard  # noqa: E402
import gate_trace  # noqa: E402

# === 评测目标 skill (可被 argv[1] 覆盖) ===
CAND_SKILL = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md"
CAND_LABEL = sys.argv[2] if len(sys.argv) > 2 else "skillopt_v0002"
_TS = time.strftime("%Y%m%d_%H%M%S")
OUT_BASE = sys.argv[3] if len(sys.argv) > 3 else \
    f"/data/zhuldz/lunwen/rl/train/verl1/outputs/livemath_{CAND_LABEL}_eval_{_TS}"

K = 3
SHUFFLE_SEED = 0


def main():
    # 创建独立输出目录 (不与任何 run 共用, 独立 budget 账本)
    OUT_DIR = OUT_BASE
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"out_dir: {OUT_DIR}")
    print(f"cand_skill: {CAND_SKILL}  label={CAND_LABEL}")

    cand_skill = open(CAND_SKILL, encoding="utf-8").read().strip()
    s0_skill = open(S0_DEFAULT, encoding="utf-8").read().strip()
    print(f"cand: {len(cand_skill)} chars md5={md5(cand_skill)[:12]}")
    print(f"S₀:   {len(s0_skill)} chars md5={md5(s0_skill)[:12]}")
    print(f"改动: {md5(cand_skill)[:12] != md5(s0_skill)[:12]}")

    test_items = load_split(SPLIT_DEFAULT, "test", shuffle_seed=SHUFFLE_SEED)
    print(f"\ntest 集: {len(test_items)} 题 (shuffle_seed={SHUFFLE_SEED})")

    # 独立 budget 账本 (resume_from=None, 不累加 Run B 的账)
    budget_guard.install(OUT_DIR, max_calls=60000, wall_seconds=129600, resume_from=None)
    gate_trace.init(OUT_DIR)

    print(f"\n=== paired 评测 {CAND_LABEL} vs S₀ on test={len(test_items)} (k={K}) ===")
    mean_pass, mean_delta, n, per_item, base_per = eval_paired(
        test_items, cand_skill, s0_skill,
        stage=f"{CAND_LABEL}_test_paired", k=K, eval_deadline_s=5400)
    ci = bootstrap_ci_lower(per_item)
    print(f"结果: mean_pass={mean_pass:.4f} delta={mean_delta:+.4f} n={n} CI_lower={ci:+.4f}")
    if ci > 0:
        print(f"✅ {CAND_LABEL} 强成功: Δ>0 且 CI_lower>0")
    elif mean_delta > 0:
        print(f"🟡 {CAND_LABEL} 中等: Δ>0 但 CI 跨零")
    else:
        print(f"🔴 {CAND_LABEL} 弱/失败: Δ≤0")

    # meta/not_meta 分层 (零额外 API)
    meta_deltas, nm_deltas = [], []
    for qid, ds in per_item.items():
        it = next((it for it in test_items if str(it["id"]) == qid), None)
        if it is None:
            continue
        if is_meta_item(it):
            meta_deltas.append(sum(ds) / len(ds))
        else:
            nm_deltas.append(sum(ds) / len(ds))
    meta_mean = sum(meta_deltas) / len(meta_deltas) if meta_deltas else 0
    nm_mean = sum(nm_deltas) / len(nm_deltas) if nm_deltas else 0
    meta_ci = bootstrap_ci_lower({str(it["id"]): per_item[str(it["id"])] for it in test_items if is_meta_item(it) and str(it["id"]) in per_item})
    nm_ci = bootstrap_ci_lower({str(it["id"]): per_item[str(it["id"])] for it in test_items if not is_meta_item(it) and str(it["id"]) in per_item})
    print(f"  meta    (n={len(meta_deltas)}): mean={meta_mean:+.4f} CI_lower={meta_ci:+.4f}")
    print(f"  not_meta(n={len(nm_deltas)}): mean={nm_mean:+.4f} CI_lower={nm_ci:+.4f}")

    result = {
        "eval": "skillopt_baseline_paired",
        "cand_skill_path": CAND_SKILL,
        "cand_label": CAND_LABEL,
        "cand_md5": md5(cand_skill)[:12],
        "s0_md5": md5(s0_skill)[:12],
        "test_n": len(test_items),
        "k": K, "shuffle_seed": SHUFFLE_SEED,
        "L2": {"mean_pass": round(mean_pass, 6), "delta": round(mean_delta, 6),
               "ci_lower": round(ci, 6), "n": n,
               "verdict": "strong" if ci > 0 else ("medium" if mean_delta > 0 else "weak")},
        "L3": {"meta": {"n": len(meta_deltas), "mean": round(meta_mean, 6), "ci_lower": round(meta_ci, 6)},
               "not_meta": {"n": len(nm_deltas), "mean": round(nm_mean, 6), "ci_lower": round(nm_ci, 6)}},
        "per_item_delta": {str(q): [round(float(x), 6) for x in v] for q, v in per_item.items()},
        "budget_final": budget_guard.status(),
    }
    out_path = os.path.join(OUT_DIR, "skillopt_L2_L3_test_results.json")
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n📄 结果已落盘: {out_path}")
    budget_guard.flush()
    gate_trace.close()
    print(f"🛡️ 预算终局: {budget_guard.status()}")


if __name__ == "__main__":
    main()
