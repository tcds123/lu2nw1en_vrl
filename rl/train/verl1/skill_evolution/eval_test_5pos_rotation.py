#!/usr/bin/env python3
"""阶段 A 实验 2：五位置轮换敏感性检查 (CF F01 建议).

诊断目的：当*仅*正确答案的位置变化 (所有 gold → 固定位置 p∈{A,B,C,D,E}) 时，
成绩是否漂移？
  - 高漂移 = 位置依赖 (捷径信号)
  - 低漂移 = 位置无关 (真算法信号)

与 shuffle 的区别：shuffle 按 (qid,seed) 随机打乱每题；rotation 把*所有*题的正确选项
强制放到同一位置 p，是位置偏置的受控对照。位置 p=0(A) = 原始全 A 泄漏口径 → 若算法
在该位置显著高于 B-E，则位置偏置存在。

5 个臂: s0 / on / off / skillopt / rule_meta_guess
× 5 位置 × 1 seed × k=3。

**仅诊断，不用于重选 best test 配置** (CF F01 明确要求)。

复用 eval_test_multiseed_shuffle 的 eval_skill_on / stratify / rule_meta_guess。
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tf_global_v2_math import (  # noqa: E402
    run_one, set_temp, load_split, EVAL_TEMP, md5,
)
from skillopt.envs.livemathematicianbench.dataloader import (  # noqa: E402
    LiveMathematicianBenchDataLoader, _CHOICE_LABELS, _normalize_label,
)

SPLIT_136 = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42"
META_TEXT = "one of the remaining options is correct, but a stronger result can be proven"
OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math_poolfilter/phaseA_shuffle_20260917"


# ===================================================================
# rotate：把正确选项强制放到固定位置 pos_idx (0=A..4=E)，其余按原顺序填占位
# ===================================================================
def rotate_to_position(item: dict, pos_idx: int) -> dict:
    """把 correct_choice 放到 index pos_idx，其余 distractors 按原相对顺序填充，remap A-E.

    与 shuffle_item 的随机化对照：这里位置完全确定 (由 pos_idx 决定)，用于位置偏置诊断。
    """
    choices = [dict(c) for c in item["choices"]]
    original_correct_label = _normalize_label(item["correct_choice"]["label"])

    correct_choice = None
    distractors = []
    for c in choices:
        if _normalize_label(c["label"]) == original_correct_label:
            correct_choice = c
        else:
            distractors.append(c)
    if correct_choice is None:
        # 兜底：按 correct_choice.text 匹配
        ctext = str(item["correct_choice"].get("text", "")).strip().lower()
        for c in choices:
            if str(c.get("text", "")).strip().lower() == ctext:
                correct_choice = c
            else:
                distractors.append(c)
        if correct_choice is None:
            correct_choice = choices[0]
            distractors = [c for c in choices[1:]]

    new_order = [None] * len(choices)
    new_order[pos_idx] = correct_choice
    di = 0
    for i in range(len(choices)):
        if new_order[i] is None:
            new_order[i] = distractors[di]
            di += 1

    remapped = []
    new_correct = dict(item["correct_choice"])
    for idx, c in enumerate(new_order):
        new_label = _CHOICE_LABELS[idx]
        if _normalize_label(c["label"]) == original_correct_label:
            new_correct = {"label": new_label, "text": c["text"]}
        remapped.append({"label": new_label, "text": c["text"]})

    out = dict(item)
    out["choices"] = remapped
    out["correct_choice"] = new_correct
    return out


def rotate_batch(items, pos_idx):
    return [rotate_to_position(it, pos_idx) for it in items]


def is_meta(item: dict) -> bool:
    return any(META_TEXT in str(c.get("text", "")).lower() for c in item["choices"])


def meta_qids(items):
    return {str(it["id"]) for it in items if is_meta(it)}


def eval_skill_on(items, skill_content, k, stage, max_workers=40, progress_cb=None):
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


def rule_meta_guess(item: dict, seed: int) -> str:
    choices = item["choices"]
    for c in choices:
        if META_TEXT in str(c.get("text", "")).lower():
            return _normalize_label(c["label"])
    digest = hashlib.sha256(f"{seed}:{item['id']}".encode("utf-8")).hexdigest()
    idx = int(digest[:16], 16) % len(choices)
    return _normalize_label(choices[idx]["label"])


def eval_rule_on(items, rule_fn, seed):
    per = {}
    for it in items:
        pred = rule_fn(it, seed)
        gold = _normalize_label(it["correct_choice"]["label"])
        per[str(it["id"])] = 1.0 if pred == gold else 0.0
    mean_pass = sum(per.values()) / max(1, len(per))
    return mean_pass, per


def stratify(per: dict, meta_set: set):
    meta_vals = [v for q, v in per.items() if q in meta_set]
    other_vals = [v for q, v in per.items() if q not in meta_set]
    return {
        "meta_46": sum(meta_vals) / max(1, len(meta_vals)),
        "meta_n": len(meta_vals),
        "other_60": sum(other_vals) / max(1, len(other_vals)),
        "other_n": len(other_vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default=SPLIT_136)
    ap.add_argument("--s0", default="/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md")
    ap.add_argument("--on", default="/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math_poolfilter/run_20260916_181857/best_skill_evolved.md")
    ap.add_argument("--off", default="/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math_poolfilter_off/run_20260916_181237/best_skill_evolved.md")
    ap.add_argument("--skillopt", default="/data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md")
    ap.add_argument("--eval_k", type=int, default=3)
    ap.add_argument("--skills", default="s0,on,off,skillopt,rule_meta_guess",
                    help="逗号分隔, 选 s0/on/off/skillopt/rule_meta_guess")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="冒烟用: 只跑前 N 题 (0=全部)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_items = load_split(args.split_dir, "test")
    if args.limit > 0:
        test_items = test_items[:args.limit]
    meta_set = meta_qids(test_items)
    n_pos = 5
    print(f"📋 test items: {len(test_items)} | meta: {len(meta_set)} | other: {len(test_items)-len(meta_set)}")
    print(f"   k={args.eval_k} | positions=A,B,C,D,E (fixed per position) | arms={args.skills}")

    skills = {}
    if "s0" in args.skills:
        skills["s0"] = open(args.s0, encoding="utf-8").read().strip()
    if "on" in args.skills:
        skills["on"] = open(args.on, encoding="utf-8").read().strip()
    if "off" in args.skills:
        skills["off"] = open(args.off, encoding="utf-8").read().strip()
    if "skillopt" in args.skills:
        skills["skillopt"] = open(args.skillopt, encoding="utf-8").read().strip()
    rule_names = [n for n in ["rule_meta_guess"] if n in args.skills]
    for name, txt in skills.items():
        print(f"  {name}: {len(txt)} chars (hash {md5(txt)[:8]})")

    # 结构: results[name][pos_idx] = {mean_pass, per_qid, stratified}
    results = {name: {} for name in list(skills.keys()) + rule_names}

    def _per_path(name, pos, txt=None):
        if txt is None:
            txt = skills.get(name, "")
        h = md5(txt)[:8] if txt else "rule"
        # rotation cache 后缀 _rot (与 shuffle 的 _shuf 区分)
        return os.path.join(args.out_dir, f"per_{name}_{h}_pos{pos}_rot.json")

    for pos in range(n_pos):
        pos_label = _CHOICE_LABELS[pos]
        print(f"\n{'='*50}\nPosition {pos} ({pos_label})\n{'='*50}", flush=True)
        rot_items = rotate_batch(test_items, pos)
        # 抽查：position 0(A) 应与原始全 A 一致；其他位置 gold 应恒为该位置
        if rot_items:
            g0 = rot_items[0]["correct_choice"]["label"]
            print(f"  [rotate check] qid={rot_items[0]['id']} pos{pos} gold={g0} (应恒={pos_label})", flush=True)

        for name, txt in skills.items():
            per_path = _per_path(name, pos, txt)
            if os.path.exists(per_path):
                per = json.load(open(per_path, encoding="utf-8"))
                mean_pass = sum(per.values()) / max(1, len(per))
                results[name][pos] = {"mean_pass": mean_pass, "per_qid": per,
                                      "stratified": stratify(per, meta_set)}
                print(f"  [{name}] pos{pos} CACHED pass={mean_pass:.4f}", flush=True)
                continue
            t0 = time.time()
            done = [0]

            def cb(d, total, qid, pr, _name=name):
                done[0] = d
                if d % 10 == 0 or d == total:
                    print(f"    [{_name} pos{pos}] {d}/{total}  last {qid} pr={pr:.2f}  "
                          f"elapsed {time.time()-t0:.0f}s", flush=True)

            mean_pass, per = eval_skill_on(rot_items, txt, args.eval_k,
                                           stage=f"phaseA_rot_p{pos}_{name}", progress_cb=cb)
            json.dump(per, open(per_path, "w", encoding="utf-8"), ensure_ascii=False)
            results[name][pos] = {"mean_pass": mean_pass, "per_qid": per,
                                  "stratified": stratify(per, meta_set)}
            st = results[name][pos]["stratified"]
            print(f"  [{name}] pos{pos} pass={mean_pass:.4f} ({time.time()-t0:.0f}s)  "
                  f"meta={st['meta_46']:.3f}/other={st['other_60']:.3f}", flush=True)

        for rn in rule_names:
            per_path = _per_path(rn, pos)
            if os.path.exists(per_path):
                per = json.load(open(per_path, encoding="utf-8"))
                mean_pass = sum(per.values()) / max(1, len(per))
                results[rn][pos] = {"mean_pass": mean_pass, "per_qid": per,
                                    "stratified": stratify(per, meta_set)}
                print(f"  [{rn}] pos{pos} CACHED pass={mean_pass:.4f}", flush=True)
                continue
            rule_fn = {"rule_meta_guess": rule_meta_guess}[rn]
            mean_pass, per = eval_rule_on(rot_items, rule_fn, seed=1)
            json.dump(per, open(per_path, "w", encoding="utf-8"), ensure_ascii=False)
            results[rn][pos] = {"mean_pass": mean_pass, "per_qid": per,
                               "stratified": stratify(per, meta_set)}
            st = results[rn][pos]["stratified"]
            print(f"  [{rn}] pos{pos} pass={mean_pass:.4f}  meta={st['meta_46']:.3f}/other={st['other_60']:.3f}", flush=True)

        # 每 position 落盘一次汇总
        _write_summary(args, test_items, list(skills.keys()) + rule_names, results, meta_set, args.out_dir)

    summary = _write_summary(args, test_items, list(skills.keys()) + rule_names, results, meta_set, args.out_dir)
    out_path = os.path.join(args.out_dir, "rotation_5pos_eval.json")
    print(f"\n{'='*50}\n✅ Done → {out_path}")
    print(f"\n汇总 (k={args.eval_k}, test={len(test_items)}, meta={len(meta_set)}, positions=5):")
    print(f"  {'name':<16} {'A':>7} {'B':>7} {'C':>7} {'D':>7} {'E':>7} {'range':>7} {'sd':>6}")
    for name in list(skills.keys()) + rule_names:
        s = summary.get(name, {})
        if not s:
            continue
        means = s["per_position_mean"]
        rng = max(means) - min(means)
        mean_of_means = sum(means) / len(means)
        sd = (sum((m - mean_of_means) ** 2 for m in means) / len(means)) ** 0.5
        print(f"  {name:<16} " + " ".join(f"{m:>7.4f}" for m in means) +
              f" {rng:>7.4f} {sd:>6.4f}")
    print("\n判读:")
    print("  - rule_meta_guess meta_46 应在所有 5 位置 ≈1.0 (meta 按文本检测, 非位置); other_60 应 ≈0.2 (位置无关)")
    print("  - 演化 skill 在 A 位置 (原始全 A 泄漏口径) 显著高于 B-E → 位置偏置存在")
    print("  - 跨 5 位置 range < 0.03 且 sd < 0.015 → 位置无关 (真信号)")


def _write_summary(args, test_items, all_names, results, meta_set, out_dir):
    """构建按位置汇总 (含 per-position mean + drift)."""
    summary = {}
    for name in all_names:
        pos_entries = results.get(name, {})
        if not pos_entries:
            continue
        per_position_mean = []
        per_position_strat = {"meta_46": [], "other_60": []}
        for pos in range(5):
            e = pos_entries.get(pos)
            if e:
                per_position_mean.append(e["mean_pass"])
                st = e.get("stratified", {})
                per_position_strat["meta_46"].append(st.get("meta_46", float("nan")))
                per_position_strat["other_60"].append(st.get("other_60", float("nan")))
            else:
                per_position_mean.append(float("nan"))
                per_position_strat["meta_46"].append(float("nan"))
                per_position_strat["other_60"].append(float("nan"))
        valid = [m for m in per_position_mean if m == m]  # filter NaN
        rng = (max(valid) - min(valid)) if valid else 0.0
        mean_of_means = sum(valid) / len(valid) if valid else 0.0
        sd = (sum((m - mean_of_means) ** 2 for m in valid) / len(valid)) ** 0.5 if valid else 0.0
        summary[name] = {
            "per_position_mean": per_position_mean,
            "per_position_strat": per_position_strat,
            "range": rng,
            "sd": sd,
            "position_labels": list(_CHOICE_LABELS[:5]),
        }
    out = {"args": vars(args), "n_test": len(test_items), "n_meta": len(meta_set),
           "meta_text": META_TEXT, "summary": summary, "results": results}
    out_path = os.path.join(out_dir, "rotation_5pos_eval.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    main()
