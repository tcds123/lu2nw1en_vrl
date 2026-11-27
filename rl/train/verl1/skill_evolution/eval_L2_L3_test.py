#!/usr/bin/env python
# L2/L3 独立评测: best_skill_evolved.md vs S₀ 在 test=106 上 paired 评测 (2026-11-26).
# 冻结方案 L2 = test=106 独立确认增益, L3 = meta/not_meta 分层 (从 L2 per_item 切, 零额外 API).
# 与 run 同口径: V3.2, k=3, shuffle_seed=0, paired_rescore, EVAL_TEMP.
#
# 用法: bash (自动载入 key + _env.sh). 评测结果落 run 目录.
# ⚠️ 付费 API (已授权 run+L2/L3 全范围).
import json
import os
import sys

_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")

# 同 evolver 的 import 链
from tf_global_v2_math import (  # noqa: E402
    eval_paired, bootstrap_ci_lower, load_split, is_meta_item,
    set_temp, EVAL_TEMP, md5,
)
# split/s0 默认从 evolver 拿
import skill_evolver_math as se  # noqa: E402
SPLIT_DEFAULT = se.SPLIT_DEFAULT
S0_DEFAULT = se.S0_DEFAULT
import budget_guard  # noqa: E402
import gate_trace  # noqa: E402

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/zhuldz/lunwen/rl/train/verl1/outputs/livemath_fullrun_v32_20261126_161211/run_20261126_161212"
K = 3
SHUFFLE_SEED = 0


def main():
    best_skill = open(os.path.join(RUN_DIR, "best_skill_evolved.md"), encoding="utf-8").read().strip()
    s0_skill = open(S0_DEFAULT, encoding="utf-8").read().strip()
    print(f"best_skill: {len(best_skill)} chars md5={md5(best_skill)[:12]}")
    print(f"S₀:         {len(s0_skill)} chars md5={md5(s0_skill)[:12]}")
    print(f"改动: {md5(best_skill)[:12] != md5(s0_skill)[:12]}")

    # 加载 test=106 (与 run 同 shuffle_seed)
    test_items = load_split(SPLIT_DEFAULT, "test", shuffle_seed=SHUFFLE_SEED)
    print(f"\ntest 集: {len(test_items)} 题 (shuffle_seed={SHUFFLE_SEED})")

    # 安装 budget_guard (沿用 run 的账本累计, 不清零)
    budget_guard.install(RUN_DIR, max_calls=60000, wall_seconds=129600, resume_from=RUN_DIR)
    gate_trace.init(RUN_DIR)

    # L2: paired 评测 best vs S₀ on test=106
    print(f"\n=== L2: paired 评测 best vs S₀ on test={len(test_items)} (k={K}) ===")
    mean_pass, mean_delta, n, per_item, base_per = eval_paired(
        test_items, best_skill, s0_skill,
        stage="L2_test_paired", k=K, eval_deadline_s=5400)
    ci = bootstrap_ci_lower(per_item)
    print(f"L2 结果: mean_pass={mean_pass:.4f} delta={mean_delta:+.4f} n={n} CI_lower={ci:+.4f}")
    if ci > 0:
        print(f"✅ L2 强成功: Δ>0 且 CI_lower>0 (独立评测显著正向)")
    elif mean_delta > 0:
        print(f"🟡 L2 中等: Δ>0 但 CI 跨零 (有信号待加固)")
    else:
        print(f"🔴 L2 弱/失败: Δ≤0")

    # L3: meta/not_meta 分层 (从 per_item 切, 零额外 API)
    print(f"\n=== L3: meta/not_meta 分层 (从 L2 per_item 切) ===")
    meta_items = [it for it in test_items if is_meta_item(it)]
    nm_items = [it for it in test_items if not is_meta_item(it)]
    print(f"test 分层: meta={len(meta_items)} not_meta={len(nm_items)}")

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
    meta_ci = bootstrap_ci_lower({str(it["id"]): per_item[str(it["id"])] for it in meta_items if str(it["id"]) in per_item})
    nm_ci = bootstrap_ci_lower({str(it["id"]): per_item[str(it["id"])] for it in nm_items if str(it["id"]) in per_item})
    print(f"  meta    (n={len(meta_deltas)}): mean={meta_mean:+.4f} CI_lower={meta_ci:+.4f}")
    print(f"  not_meta(n={len(nm_deltas)}): mean={nm_mean:+.4f} CI_lower={nm_ci:+.4f}")

    # 落盘 L2/L3 结果
    result = {
        "eval": "L2_L3_test_independent",
        "best_skill_md5": md5(best_skill)[:12],
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
    out_path = os.path.join(RUN_DIR, "L2_L3_test_results.json")
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n📄 L2/L3 结果已落盘: {out_path}")
    budget_guard.flush()
    gate_trace.close()
    print(f"🛡️ 预算终局: {budget_guard.status()}")


if __name__ == "__main__":
    main()
