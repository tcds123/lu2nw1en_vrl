#!/usr/bin/env python3
"""test=106 多 seed 复测: S0 vs v2 best vs evolver best, 同条件 k=5 投票.

坐实 v2 修复后 test +7.0pp 是否可复现 (task1), 同时用 k=5 同条件重评测 evolver 旧 best
(task2, 评测 bug 修复前跑的, 数字被采样污染).

输出: out_dir/multiseed_test_eval.json + 每 seed 的 pass 明细.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tf_global_v2_math import (  # noqa: E402
    run_one, set_temp, load_split, EVAL_TEMP, GEN_TEMP,
    S0_DEFAULT, md5,
)

SPLIT_136 = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42"
V2_BEST = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2/run_20260728_131934/tf_global_skill_v2_math.md"
EVOLVER_BEST = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math/run_20260727_163814/best_skill_evolved.md"
SKILLOPT_BEST = "/data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md"  # SkillOpt math best, test=124 = 0.5161 (full_best_0724 eval_summary)
MERGE_BEST = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math/merge_skillopt_evolver_manual.md"  # 手工合并: SkillOpt 全文 + evolver 2-iter 非meta-option failure modes (badcase互补假设)
EVO_SEED_BEST = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math/run_20260803_184657/skill_iter2.md"  # seeded-evolver 2-iter 产物 (种在 SkillOpt 上, select delta +0.390, hash 72a95d13)
OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test"


def eval_skill_on(items, skill_content, k, stage, max_workers=40, progress_cb=None):
    """对 items 用 skill 评测 k 次投票 (EVAL_TEMP), 返回 (mean_pass, per_qid_pass).
    progress_cb(done, total, qid, pr) 每完成一题回调 (即时进度 + 落盘)."""
    set_temp(EVAL_TEMP)
    per = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one, it, skill_content, stage, k): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            r = fut.result()
            pr = max(0.0, r["pass_rate"]) if r["pass_rate"] >= 0 else 0.0
            per[str(it["id"])] = pr
            done += 1
            if progress_cb:
                progress_cb(done, len(items), str(it["id"]), pr)
    mean_pass = sum(per.values()) / max(1, len(per))
    return mean_pass, per


def _write_summary(args, test_items, skills, results, out_dir):
    """从 results 构建汇总并落盘, 返回 summary dict. 在循环内/结束各调一次."""
    summary = {}
    for name, runs in results.items():
        if not runs:
            continue
        passes = [r["mean_pass"] for r in runs]
        summary[name] = {
            "n_seeds": len(passes),
            "mean_pass": sum(passes) / len(passes),
            "passes": passes,
        }
    if "s0" in results and results["s0"]:
        s0_runs = results["s0"]
        for name in skills:
            if name == "s0" or name not in results or not results[name]:
                continue
            n = min(len(s0_runs), len(results[name]))
            deltas = [results[name][i]["mean_pass"] - s0_runs[i]["mean_pass"] for i in range(n)]
            summary[name]["delta_vs_s0"] = {
                "per_seed": deltas,
                "mean": sum(deltas) / len(deltas) if deltas else 0.0,
            }
    out = {"args": vars(args), "n_test": len(test_items), "summary": summary, "runs": results}
    out_path = os.path.join(out_dir, "multiseed_test_eval.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default=SPLIT_136)
    ap.add_argument("--s0", default=S0_DEFAULT)
    ap.add_argument("--v2_best", default=V2_BEST)
    ap.add_argument("--evolver_best", default=EVOLVER_BEST)
    ap.add_argument("--skillopt_best", default=SKILLOPT_BEST)
    ap.add_argument("--merge_best", default=MERGE_BEST)
    ap.add_argument("--evo_seed_best", default=EVO_SEED_BEST)
    # ── 机制消融 skill 槽 (leave-one-out; FULL = evolver 主结果 12633201, 已 4-seed 缓存) ──
    # 默认空, 路径提供且文件存在才加载 (时间戳动态, 由 run_ablation_multiseed.sh 传入).
    ap.add_argument("--abl_failure_best", default="", help="A2 ablate_failure 产物 best_skill_evolved.md")
    ap.add_argument("--abl_gepa_best", default="", help="A1 ablate_gepa 产物")
    ap.add_argument("--abl_consolidate_best", default="", help="A3 ablate_consolidate 产物")
    ap.add_argument("--abl_antideg_best", default="", help="A4 ablate_antideg 产物")
    ap.add_argument("--eval_k", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3, help="重复次数 (多 seed = 多次 k 投票独立采样)")
    ap.add_argument("--skills", default="s0,v2,evolver", help="逗号分隔, 选 s0/v2/evolver/skillopt/merge/evo_seed/abl_failure/abl_gepa/abl_consolidate/abl_antideg")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_items = load_split(args.split_dir, "test")
    print(f"📋 test items: {len(test_items)} | k={args.eval_k} | seeds={args.seeds}")

    skills = {}
    if "s0" in args.skills:
        skills["s0"] = open(args.s0, encoding="utf-8").read().strip()
    if "v2" in args.skills:
        skills["v2"] = open(args.v2_best, encoding="utf-8").read().strip()
    if "evolver" in args.skills:
        skills["evolver"] = open(args.evolver_best, encoding="utf-8").read().strip()
    if "skillopt" in args.skills:
        skills["skillopt"] = open(args.skillopt_best, encoding="utf-8").read().strip()
    if "merge" in args.skills:
        skills["merge"] = open(args.merge_best, encoding="utf-8").read().strip()
    if "evo_seed" in args.skills:
        skills["evo_seed"] = open(args.evo_seed_best, encoding="utf-8").read().strip()
    # 机制消融槽: 路径提供且文件存在才加载, 否则告警跳过 (不崩, 便于部分跑)
    for _slot, _path in [("abl_failure", args.abl_failure_best),
                         ("abl_gepa", args.abl_gepa_best),
                         ("abl_consolidate", args.abl_consolidate_best),
                         ("abl_antideg", args.abl_antideg_best)]:
        if _slot in args.skills:
            if _path and os.path.exists(_path):
                skills[_slot] = open(_path, encoding="utf-8").read().strip()
            else:
                print(f"  ⚠️ {_slot}: 路径未提供或不存在 ({_path or '空'}), 跳过")
    for name, txt in skills.items():
        print(f"  {name}: {len(txt)} chars (hash {md5(txt)[:8]})")

    results = {name: [] for name in skills}  # list of (mean_pass, per_qid)
    # 断点续跑: 每 (skill,seed) 单独落盘 per_qid, 已存在则跳过.
    # 缓存 key 含 skill hash 前缀, 避免 skill 内容变更后误用旧缓存 (07-30 踩坑: evolver 换 skill 仍命中旧 per_evolver_seed*.json).
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
                done[0] = d
                # 即时落盘 (覆盖写, 断点续跑基础)
                json.dump({"done": d, "total": total, "per_qid_so_far": None},
                          open(per_path + ".progress", "w"), ensure_ascii=False)
                if d % 10 == 0 or d == total:
                    print(f"    [{_name} seed{seed}] {d}/{total}  last {qid} pr={pr:.2f}  "
                          f"elapsed {time.time()-t0:.0f}s", flush=True)

            mean_pass, per = eval_skill_on(test_items, txt, args.eval_k,
                                           stage=f"multiseed_{name}", progress_cb=cb)
            json.dump(per, open(per_path, "w", encoding="utf-8"), ensure_ascii=False)
            results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per})
            print(f"  [{name}] seed{seed} test pass={mean_pass:.4f} ({time.time()-t0:.0f}s)", flush=True)
            # 每完成一个 skill-seed 即更新汇总
            _write_summary(args, test_items, skills, results, out_dir=args.out_dir)

    # 最终汇总
    summary = _write_summary(args, test_items, skills, results, out_dir=args.out_dir)
    out_path = os.path.join(args.out_dir, "multiseed_test_eval.json")
    print(f"\n{'='*50}\n✅ Done → {out_path}")
    print(f"\n汇总 (k={args.eval_k}, test={len(test_items)}, seeds={args.seeds}):")
    for name, s in summary.items():
        line = f"  {name}: mean_pass={s['mean_pass']:.4f}  passes={[round(p,4) for p in s['passes']]}"
        if "delta_vs_s0" in s:
            d = s["delta_vs_s0"]
            line += f"  ΔvsS0 mean={d['mean']:+.4f} per_seed={[round(x,4) for x in d['per_seed']]}"
        print(line)


if __name__ == "__main__":
    main()
