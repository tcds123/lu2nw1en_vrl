#!/usr/bin/env python3
"""五位置轮换分析 — 从 per_{name}_pos{p}_rot.json 计算 per-skill drift.

复用 eval_test_5pos_rotation 的 stratify / META_TEXT / meta-set 探测逻辑,
对每个 arm 计算:
  - 5 位置的 overall mean (A-E)
  - meta_46 / other_60 分层 mean (5 位置)
  - drift = max−min, sd across 5 positions
  - 位置-A (原 all-A 泄漏布局) vs B-E 均值差 (位置捷径信号)

rule_meta_guess 的 other_60 drift 是结构性的 (sha256 索引偏置, 非模型位置偏差),
作为 baseline 读; 真信号看模型臂 (s0/on/off/skillopt).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 直接复用 eval harness 的 meta_qids (含 .lower() 探测, 保证与正式 eval 完全一致)
from eval_test_5pos_rotation import meta_qids  # noqa: E402

OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math_poolfilter/phaseA_shuffle_20260917"
SPLIT_DIR = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42"


def _load_meta_set():
    """从 test split 重建 meta-set (与 eval harness 完全一致的 meta_qids)."""
    items = json.load(open(os.path.join(SPLIT_DIR, "test", "items.json"), encoding="utf-8"))
    return meta_qids(items), len(items)


def _arm_files(name):
    """返回某 arm 的 5 个 pos 文件路径 (存在的). rule 臂文件名固定, 模型臂带 hash glob."""
    import glob
    files = []
    for p in range(5):
        if name == "rule_meta_guess":
            f = os.path.join(OUT_DIR, f"per_rule_meta_guess_rule_pos{p}_rot.json")
            files.append(f if os.path.exists(f) else None)
        else:
            # 模型臂: per_{name}_{hash8}_pos{p}_rot.json
            hits = glob.glob(os.path.join(OUT_DIR, f"per_{name}_*_pos{p}_rot.json"))
            files.append(hits[0] if hits else None)
    return files


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def analyze():
    meta_set, n_test = _load_meta_set()
    print(f"meta-set: {len(meta_set)} / {n_test} test items (META_TEXT detected)")
    print(f"  → meta_46 actually {len(meta_set)}, other_{n_test - len(meta_set)}\n")

    arms = ["s0", "on", "off", "skillopt", "rule_meta_guess"]
    rows = []
    for name in arms:
        files = _arm_files(name)
        if not all(files):
            missing = [p for p, f in enumerate(files) if not f]
            print(f"  {name}: ⏳ missing pos {missing} — skip")
            continue
        overall, meta_means, other_means = [], [], []
        for p, f in enumerate(files):
            d = json.load(open(f, encoding="utf-8"))
            per = [v for k, v in d.items() if k != "_manifest"]
            overall.append(_mean(per))
            meta_means.append(_mean([d[q] for q in d if q in meta_set]))
            other_means.append(_mean([d[q] for q in d if q not in meta_set]))
        rng = max(overall) - min(overall)
        import statistics
        sd = statistics.pstdev(overall)
        a_vs_rest = overall[0] - _mean(overall[1:])
        rows.append({
            "name": name, "overall": overall, "meta_46": meta_means,
            "other_60": other_means, "range": rng, "sd": sd, "a_vs_rest": a_vs_rest,
        })

    # 打印表
    print(f"{'arm':<16} {'A':>7} {'B':>7} {'C':>7} {'D':>7} {'E':>7} "
          f"{'range':>7} {'sd':>6} {'A-vs-BCDE':>10}")
    for r in rows:
        m = r["overall"]
        print(f"{r['name']:<16} " + " ".join(f"{x:>7.4f}" for x in m) +
              f" {r['range']:>7.4f} {r['sd']:>6.4f} {r['a_vs_rest']:>+10.4f}")
    print()
    print("meta_46 (should be ≈1.0 × all 5 pos for rule; model arms show real meta handling):")
    print(f"{'arm':<16} " + " ".join(f"{'pos'+chr(65+p):>7}" for p in range(5)))
    for r in rows:
        print(f"{r['name']:<16} " + " ".join(f"{x:>7.4f}" for x in r["meta_46"]))
    print()
    print("other_60 (rule drift = structural sha256 bias, NOT position bias; model arms = real signal):")
    print(f"{'arm':<16} " + " ".join(f"{'pos'+chr(65+p):>7}" for p in range(5)))
    for r in rows:
        print(f"{r['name']:<16} " + " ".join(f"{x:>7.4f}" for x in r["other_60"]))

    # 输出 JSON 供 RESPONSE 文档引用
    out = os.path.join(OUT_DIR, "rotation_5pos_analysis.json")
    json.dump({"meta_count": len(meta_set), "n_test": n_test, "arms": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    analyze()
