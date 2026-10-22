#!/usr/bin/env python3
"""AIME25 held-out 测试评测: S0(AIME) vs v2(AIME) vs evolver(AIME), 同条件 k 投票多 seed.

镜像 eval_aime24_multiseed.py, 但:
  - 数据: AIME25 (yentinglin/aime_2025, 30 题) - 完全 held-out, 从未参与 AIME24 train/val 优化
  - skill: AIME 从头优化的产物 (skill_aime_initial / tf_global_skill_v2_aime / evolver_aime best)
            而非 livemath 选择题 skill (eval_aime24_multiseed 测的是泛化, 这里测的是同域 from-scratch)
  - 评测后端完全复用: run_one_aime / eval_skill_on / extract_int_answer (整数比对)

这是 Task2 的终极判据: 我们的免训练优化方法在 AIME 上从 S0 跑到 evolver, 是否在 held-out AIME25 上
相对 S0(AIME) 有提升. 无论正负都记录 (正=方法在新数据集有效; 负=方法未迁移到 AIME 难度/形态).

输出: out_dir/aime25_eval.json + 每 seed 的 per_qid 明细 (断点续跑缓存).
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tf_global_v2_aime import set_temp, EVAL_TEMP, md5  # noqa: E402  (AIME backend, not math)
from eval_aime24_multiseed import eval_skill_on, load_aime  # noqa: E402

AIME_DATA = "/data/zhuldz/lunwen/data/aime25.jsonl"
S0_AIME = "/data/zhuldz/lunwen/data/skill_aime_initial_v2.md"
V2_AIME = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/run_LATEST/tf_global_skill_v2_aime.md"
EVOLVER_AIME = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime/run_LATEST/best_skill_evolved.md"
SKILLOPT_AIME = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/skillopt/aime-skillopt-20261008/train/best_skill.md"
OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/aime25_test"


def _write_summary(args, test_items, skills, results, out_dir):
    summary = {}
    for name, runs in results.items():
        if not runs:
            continue
        passes = [r["mean_pass"] for r in runs]
        summary[name] = {"n_seeds": len(passes), "mean_pass": sum(passes) / len(passes), "passes": passes}
    if "s0" in results and results["s0"]:
        s0_runs = results["s0"]
        for name in skills:
            if name == "s0" or name not in results or not results[name]:
                continue
            n = min(len(s0_runs), len(results[name]))
            deltas = [results[name][i]["mean_pass"] - s0_runs[i]["mean_pass"] for i in range(n)]
            summary[name]["delta_vs_s0"] = {"per_seed": deltas, "mean": sum(deltas) / len(deltas) if deltas else 0.0}
    out = {"args": vars(args), "n_test": len(test_items), "summary": summary, "runs": results}
    out_path = os.path.join(out_dir, "aime25_eval.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=AIME_DATA)
    ap.add_argument("--s0", default=S0_AIME)
    ap.add_argument("--v2_best", default=V2_AIME)
    ap.add_argument("--evolver_best", default=EVOLVER_AIME)
    ap.add_argument("--skillopt_best", default=SKILLOPT_AIME,
                    help="skillopt 训练产物 best_skill.md 路径 (与我们 evolver 公平对比)")
    ap.add_argument("--eval_k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=4, help="重复次数 (多 seed = 多次 k 投票独立采样)")
    ap.add_argument("--skills", default="s0,v2,evolver", help="逗号分隔, 选 s0/v2/evolver/skillopt")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_items = load_aime(args.data)
    print(f"📋 AIME25 items: {len(test_items)} | k={args.eval_k} | seeds={args.seeds}")

    skills = {}
    if "s0" in args.skills:
        skills["s0"] = open(args.s0, encoding="utf-8").read().strip()
    if "v2" in args.skills:
        skills["v2"] = open(args.v2_best, encoding="utf-8").read().strip()
    if "evolver" in args.skills:
        skills["evolver"] = open(args.evolver_best, encoding="utf-8").read().strip()
    if "skillopt" in args.skills:
        skills["skillopt"] = open(args.skillopt_best, encoding="utf-8").read().strip()
    for name, txt in skills.items():
        print(f"  {name}: {len(txt)} chars (hash {md5(txt)[:8]})")

    results = {name: [] for name in skills}

    def _per_path(name, seed, txt=None):
        if txt is None:
            txt = skills[name]
        return os.path.join(args.out_dir, f"per_{name}_{md5(txt)[:8]}_seed{seed}.json")

    for seed in range(1, args.seeds + 1):
        print(f"\n{'='*50}\nSeed {seed}/{args.seeds}\n{'='*50}", flush=True)
        for name, txt in skills.items():
            per_path = _per_path(name, seed, txt)
            if os.path.exists(per_path):
                per = json.load(open(per_path, encoding="utf-8"))
                mean_pass = sum(per.values()) / max(1, len(per))
                results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per})
                print(f"  [{name}] seed{seed} CACHED pass={mean_pass:.4f}", flush=True)
                continue
            t0 = time.time()
            done = [0]

            def cb(d, total, qid, pr, _name=name):
                json.dump({"done": d, "total": total},
                          open(per_path + ".progress", "w", encoding="utf-8"), ensure_ascii=False)
                if d % 5 == 0 or d == total:
                    print(f"    [{_name} seed{seed}] {d}/{total}  last {qid} pr={pr:.2f}  "
                          f"elapsed {time.time()-t0:.0f}s", flush=True)

            mean_pass, per = eval_skill_on(test_items, txt, args.eval_k,
                                           stage=f"aime25_{name}", progress_cb=cb)
            json.dump(per, open(per_path, "w", encoding="utf-8"), ensure_ascii=False)
            results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per})
            print(f"  [{name}] seed{seed} AIME25 pass={mean_pass:.4f} ({time.time()-t0:.0f}s)", flush=True)
            _write_summary(args, test_items, skills, results, out_dir=args.out_dir)

    summary = _write_summary(args, test_items, skills, results, out_dir=args.out_dir)
    out_path = os.path.join(args.out_dir, "aime25_eval.json")
    print(f"\n{'='*50}\n✅ Done -> {out_path}")
    print(f"\n汇总 (k={args.eval_k}, AIME25={len(test_items)}, seeds={args.seeds}):")
    for name, s in summary.items():
        line = f"  {name}: mean_pass={s['mean_pass']:.4f}  passes={[round(p,4) for p in s['passes']]}"
        if "delta_vs_s0" in s:
            d = s["delta_vs_s0"]
            line += f"  ΔvsS0 mean={d['mean']:+.4f} per_seed={[round(x,4) for x in d['per_seed']]}"
        print(line)


if __name__ == "__main__":
    main()
