#!/usr/bin/env python3
"""阶段 A：shuffle 重测先定性（答案位置打乱 + meta 位置随机化 + 分层报告）.

GPT review 暴露 P0 评测捷径（已零 API 坐实，见 docs/phase2_issues.md 2026-09-17）:
- test 106 题答案全是 A（evolver load_split 直接读 items.json，绕过 dataloader 的 shuffle_choices=True）
- meta 选项 "One of the remaining options is correct, but a stronger result can be proven." 100% 正确（46/46）
- ON−S0 提升 +0.2925 的 77% 来自这 46 道 meta 题

本脚本复用 eval_test_multiseed.py 骨架，在 run_one 前对 item dict 按 (qid, seed) 打乱 choices + remap
correct_choice.label（复用 dataloader 的 _item_shuffle_seed 静态方法，保证与参考 dataloader 字节一致）。
run_one/evaluator 无需改动即可看到打乱后的选项、按打乱后的 gold 判分 → "选 A" 位置捷径失效。

新增:
- rule_meta_guess（零 API 规则基线）: 含 meta 文本→选 meta 选项 label；否则按 sha256(seed:qid) 确定性 1/5 猜。
  作为"无算法下限"参照线（meta_46 应≈1.0, other_60 应≈0.2）。
- 分层报告: 每个 skill 按 meta 题 qid 集合分 meta_46 / other_60 两组算 mean_pass，落盘 multiseed_test_eval.json。

5 个 skill 原样跑（不删任何段——核查已证 ON L16-20 非 test 泄漏、L4 meta 规则 shuffle-invariant 合法）。
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)

from tf_global_v2_math import (  # noqa: E402
    run_one, set_temp, load_split, EVAL_TEMP, md5,
)
from skillopt.envs.livemathematicianbench.dataloader import (  # noqa: E402
    LiveMathematicianBenchDataLoader, _CHOICE_LABELS, _normalize_label,
)

SPLIT_136 = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42"

# meta 选项判定文本（小写子串匹配，与 P0 复现脚本一致）
META_TEXT = "one of the remaining options is correct, but a stronger result can be proven"

OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math_poolfilter/phaseA_shuffle_20260917"


# ===================================================================
# shuffle：复用 dataloader 的 _shuffle_item_choices 逻辑（字节一致）
# ===================================================================
def shuffle_item(item: dict, seed: int) -> dict:
    """对单题按 (qid, seed) 确定性打乱 choices + remap correct_choice.label.

    复用 LiveMathematicianBenchDataLoader._item_shuffle_seed（staticmethod，无需实例化）
    作 random.Random 种子，与 dataloader 默认 shuffle_choices=True 路径完全一致。
    """
    shuffled = [dict(c) for c in item["choices"]]
    rng = random.Random(LiveMathematicianBenchDataLoader._item_shuffle_seed(str(item["id"]), seed))
    rng.shuffle(shuffled)

    original_correct = _normalize_label(item["correct_choice"]["label"])
    remapped = []
    new_correct = dict(item["correct_choice"])
    for idx, c in enumerate(shuffled):
        new_label = _CHOICE_LABELS[idx]
        old_label = _normalize_label(c["label"])
        remapped.append({"label": new_label, "text": c["text"]})
        if old_label == original_correct:
            new_correct = {"label": new_label, "text": c["text"]}

    out = dict(item)
    out["choices"] = remapped
    out["correct_choice"] = new_correct
    return out


def shuffle_batch(items, seed):
    return [shuffle_item(it, seed) for it in items]


def is_meta(item: dict) -> bool:
    return any(META_TEXT in str(c.get("text", "")).lower() for c in item["choices"])


def meta_qids(items):
    """返回 meta 题的 qid 集合（用未打乱 items 判定，与 shuffle 无关——meta 是文本属性）。"""
    return {str(it["id"]) for it in items if is_meta(it)}


# ===================================================================
# LLM 评测（复用 eval_test_multiseed 的 eval_skill_on）
# ===================================================================
def eval_skill_on(items, skill_content, k, stage, max_workers=40, progress_cb=None):
    """对 items 用 skill 评测 k 次投票，返回 (mean_pass, per_qid_pass)."""
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


# ===================================================================
# 规则基线（零 API，纯逻辑）
# ===================================================================
def rule_meta_guess(item: dict, seed: int) -> str:
    """有 meta 选项→选 meta 选项 label；否则按 sha256(seed:qid) 确定性选一个 label.

    shuffle 后 meta 选项位置已随机化，但 rule 仍按*文本*检测 meta → meta_46 仍应 100% 对。
    非 meta 题 1/5 确定性猜 → other_60 应 ≈0.2。
    """
    choices = item["choices"]
    for c in choices:
        if META_TEXT in str(c.get("text", "")).lower():
            return _normalize_label(c["label"])
    # 无 meta：确定性 1/5 猜
    digest = hashlib.sha256(f"{seed}:{item['id']}".encode("utf-8")).hexdigest()
    idx = int(digest[:16], 16) % len(choices)
    return _normalize_label(choices[idx]["label"])


def eval_rule_on(items, rule_fn, seed):
    """对 items 跑规则基线（无 API），返回 (mean_pass, per_qid_pass)."""
    per = {}
    for it in items:
        pred = rule_fn(it, seed)
        gold = _normalize_label(it["correct_choice"]["label"])
        per[str(it["id"])] = 1.0 if pred == gold else 0.0
    mean_pass = sum(per.values()) / max(1, len(per))
    return mean_pass, per


# ===================================================================
# 分层 + 汇总
# ===================================================================
def stratify(per: dict, meta_set: set):
    """把 per_qid 按是否 meta 题分两组算 mean_pass."""
    meta_vals = [v for q, v in per.items() if q in meta_set]
    other_vals = [v for q, v in per.items() if q not in meta_set]
    return {
        "meta_46": sum(meta_vals) / max(1, len(meta_vals)),
        "meta_n": len(meta_vals),
        "other_60": sum(other_vals) / max(1, len(other_vals)),
        "other_n": len(other_vals),
    }


def _write_summary(args, test_items, skills, results, meta_set, out_dir):
    """构建汇总（含分层）并落盘."""
    summary = {}
    for name, runs in results.items():
        if not runs:
            continue
        passes = [r["mean_pass"] for r in runs]
        # 分层：跨 seed 取 per_qid 平均后再分层（更稳），这里用各 seed 分层后取均值
        strat_per_seed = [r.get("stratified") for r in runs if r.get("stratified")]
        entry = {
            "n_seeds": len(passes),
            "mean_pass": sum(passes) / len(passes),
            "passes": passes,
        }
        if strat_per_seed:
            entry["stratified"] = {
                "meta_46": sum(s["meta_46"] for s in strat_per_seed) / len(strat_per_seed),
                "other_60": sum(s["other_60"] for s in strat_per_seed) / len(strat_per_seed),
            }
        summary[name] = entry
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
    out = {"args": vars(args), "n_test": len(test_items), "n_meta": len(meta_set),
           "meta_text": META_TEXT, "summary": summary, "runs": results}
    out_path = os.path.join(out_dir, "multiseed_test_eval.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default=SPLIT_136)
    ap.add_argument("--s0", default="/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md")
    ap.add_argument("--on", default="/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math_poolfilter/run_20260916_181857/best_skill_evolved.md")
    ap.add_argument("--on_noimpl", default="/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math_poolfilter/run_20260916_181857/best_skill_on_noimpl.md",
                    help="ON skill 删除 ## Implication Direction Check 段后（去泄漏诊断）")
    ap.add_argument("--off", default="/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math_poolfilter_off/run_20260916_181237/best_skill_evolved.md")
    ap.add_argument("--skillopt", default="/data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md")
    ap.add_argument("--eval_k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=4, help="重复次数 (每 seed 不同 shuffle)")
    ap.add_argument("--skills", default="s0,on,off,skillopt,rule_meta_guess",
                    help="逗号分隔, 选 s0/on/on_noimpl/off/skillopt/rule_meta_guess")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="冒烟用: 只跑前 N 题 (0=全部)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_items = load_split(args.split_dir, "test")
    if args.limit > 0:
        test_items = test_items[:args.limit]
    meta_set = meta_qids(test_items)
    print(f"📋 test items: {len(test_items)} | meta: {len(meta_set)} | other: {len(test_items)-len(meta_set)}")
    print(f"   k={args.eval_k} | seeds={args.seeds} | shuffle=ON (per qid,seed 确定性)")

    # 加载 skill 文本
    skills = {}
    if "s0" in args.skills:
        skills["s0"] = open(args.s0, encoding="utf-8").read().strip()
    if "on" in args.skills:
        skills["on"] = open(args.on, encoding="utf-8").read().strip()
    if "on_noimpl" in args.skills:
        skills["on_noimpl"] = open(args.on_noimpl, encoding="utf-8").read().strip()
    if "off" in args.skills:
        skills["off"] = open(args.off, encoding="utf-8").read().strip()
    if "skillopt" in args.skills:
        skills["skillopt"] = open(args.skillopt, encoding="utf-8").read().strip()
    rule_names = [n for n in ["rule_meta_guess"] if n in args.skills]
    for name, txt in skills.items():
        print(f"  {name}: {len(txt)} chars (hash {md5(txt)[:8]})")
    for rn in rule_names:
        print(f"  {rn}: zero-API rule baseline")

    results = {name: [] for name in list(skills.keys()) + rule_names}

    def _per_path(name, seed, txt=None):
        # cache key 加 _shuf 后缀，避免与未打乱旧 cache 混淆
        if txt is None:
            txt = skills.get(name, "")
        h = md5(txt)[:8] if txt else "rule"
        return os.path.join(args.out_dir, f"per_{name}_{h}_seed{seed}_shuf.json")

    for seed in range(1, args.seeds + 1):
        print(f"\n{'='*50}\nSeed {seed}/{args.seeds}\n{'='*50}", flush=True)
        # 本 seed 的打乱（同 seed 内所有臂面对相同打乱 → paired 成立）
        shuf_items = shuffle_batch(test_items, seed)
        # 抽查：seed1 第 1 题 gold 是否仍为 A（应已 remap）
        if seed == 1 and shuf_items:
            g0 = shuf_items[0]["correct_choice"]["label"]
            print(f"  [shuffle check] qid={shuf_items[0]['id']} seed{seed} gold={g0} (原始恒 A, 打乱后应非恒 A)", flush=True)

        # LLM 臂
        for name, txt in skills.items():
            per_path = _per_path(name, seed, txt)
            if os.path.exists(per_path):
                per = json.load(open(per_path, encoding="utf-8"))
                mean_pass = sum(per.values()) / max(1, len(per))
                results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per,
                                      "stratified": stratify(per, meta_set)})
                print(f"  [{name}] seed{seed} CACHED pass={mean_pass:.4f}", flush=True)
                continue
            t0 = time.time()
            done = [0]

            def cb(d, total, qid, pr, _name=name):
                done[0] = d
                if d % 10 == 0 or d == total:
                    print(f"    [{_name} seed{seed}] {d}/{total}  last {qid} pr={pr:.2f}  "
                          f"elapsed {time.time()-t0:.0f}s", flush=True)

            mean_pass, per = eval_skill_on(shuf_items, txt, args.eval_k,
                                           stage=f"phaseA_shuf_{name}", progress_cb=cb)
            json.dump(per, open(per_path, "w", encoding="utf-8"), ensure_ascii=False)
            results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per,
                                  "stratified": stratify(per, meta_set)})
            print(f"  [{name}] seed{seed} pass={mean_pass:.4f} ({time.time()-t0:.0f}s)  "
                  f"meta={results[name][-1]['stratified']['meta_46']:.3f}/other={results[name][-1]['stratified']['other_60']:.3f}", flush=True)
            _write_summary(args, test_items, list(skills.keys()) + rule_names, results, meta_set, args.out_dir)

        # 规则基线（零 API，极快）
        for rn in rule_names:
            per_path = _per_path(rn, seed)
            if os.path.exists(per_path):
                per = json.load(open(per_path, encoding="utf-8"))
                mean_pass = sum(per.values()) / max(1, len(per))
                results[rn].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per,
                                    "stratified": stratify(per, meta_set)})
                print(f"  [{rn}] seed{seed} CACHED pass={mean_pass:.4f}", flush=True)
                continue
            rule_fn = {"rule_meta_guess": rule_meta_guess}[rn]
            mean_pass, per = eval_rule_on(shuf_items, rule_fn, seed)
            json.dump(per, open(per_path, "w", encoding="utf-8"), ensure_ascii=False)
            results[rn].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per,
                                "stratified": stratify(per, meta_set)})
            st = results[rn][-1]["stratified"]
            print(f"  [{rn}] seed{seed} pass={mean_pass:.4f}  meta={st['meta_46']:.3f}/other={st['other_60']:.3f}", flush=True)
            _write_summary(args, test_items, list(skills.keys()) + rule_names, results, meta_set, args.out_dir)

    # 最终汇总
    all_names = list(skills.keys()) + rule_names
    summary = _write_summary(args, test_items, all_names, results, meta_set, args.out_dir)
    out_path = os.path.join(args.out_dir, "multiseed_test_eval.json")
    print(f"\n{'='*50}\n✅ Done → {out_path}")
    print(f"\n汇总 (k={args.eval_k}, test={len(test_items)}, meta={len(meta_set)}, other={len(test_items)-len(meta_set)}, seeds={args.seeds}):")
    print(f"  {'name':<18} {'mean':>7} {'meta_46':>8} {'other_60':>9}  passes")
    for name in all_names:
        s = summary.get(name, {})
        if not s:
            continue
        st = s.get("stratified", {})
        m = st.get("meta_46", float("nan"))
        o = st.get("other_60", float("nan"))
        print(f"  {name:<18} {s['mean_pass']:>7.4f} {m:>8.4f} {o:>9.4f}  {[round(p,4) for p in s['passes']]}")
    print("\n判读:")
    print("  - rule_meta_guess 在 meta_46 应≈1.0、other_60 应≈0.2（实现自检；偏离则有 bug）")
    print("  - 演化 skill 在 other_60 上 > s0 且跨 seed 稳定 → 这部分才是真算法贡献")
    print("  - rule_meta_guess ≥ 演化 skill 在 other_60 → 非 meta 题无真实算法增益")


if __name__ == "__main__":
    main()
