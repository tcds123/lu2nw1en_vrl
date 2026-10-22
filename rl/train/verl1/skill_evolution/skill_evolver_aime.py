#!/usr/bin/env python3
"""Skill 自进化迭代器 - AIME 版 (免训练).

镜像 skill_evolver_math.py, 评测后端换为 AIME 数值题. 5 机制不变:
  - M1 候选种群 + 帕累托选择 (GEPA): 每轮 K 候选, val bootstrap CI 选前沿
  - M2 失败驱动局部 patch: 只做 add/replace 小改, patch 锚定当前 skill 仍错题的具体失败
  - M3 慢更新: 快周期局部 patch + 每 N 轮蒸馏整合去冗余
  - M4 防退化 + LOST 监控 + 多样性 (轮换失败类型侧重)
  - M5 test-free 选择, 最终 test (AIME25) 验证

AIME 适配: collect_failures/make_diagnostics 用 problem/answer (无数值 choices/theorem);
FAILURE_FOCI_AIME 锚定数值题失败类型 (计算/case/边界/答案形式/residue).

机制消融 leave-one-out flag 与 math 版完全一致 (--ablate_gepa/failure/consolidate/antideg),
无 flag = FULL, 便于 AIME 上复现机制贡献分析.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 通用层 + AIME 评测层都从 v2_aime 导入 (v2_aime 已从 math v2 导入通用层).
from tf_global_v2_aime import (  # noqa: E402
    run_one, eval_on, precompute_s0, bootstrap_ci_lower,
    set_temp, EVAL_TEMP, GEN_TEMP, md5, load_aime_split,
)
from tf_global_v2_math import llm, clean_skill_text as clean_skill  # noqa: E402

S0_DEFAULT = "/data/zhuldz/lunwen/data/skill_aime_initial.md"
V2_SKILL_DEFAULT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/run_LATEST/tf_global_skill_v2_aime.md"
AIME24_DATA = "/data/zhuldz/lunwen/data/aime24.jsonl"
OUT_DIR_DEFAULT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime"

# 数值题失败类型轮换 (覆盖长尾), 对应 math 版的 quantifier/equality foci
FAILURE_FOCI_AIME = [
    "Misreading the asked quantity (count vs value vs sum vs max vs min)",
    "Case enumeration completeness (missing or overlapping cases in combinatorics)",
    "Off-by-one / boundary (inclusive vs exclusive counting, endpoint inclusion)",
    "Equation / setup errors (wrong variable, wrong formula, misread constraint)",
    "Answer-form conversion (unreduced fraction, missing mod 0-999, wrong rounding)",
    "Inclusion-exclusion / overcounting (double-counting, unaccounted symmetry)",
    "Number-theoretic case analysis (missing residue classes, unbounded solution range)",
    "Arithmetic / sign / computational slips in multi-step calculation",
]

PATCHER_SYSTEM = (
    "You are an expert System Prompt Engineer for a mathematical-reasoning LLM solving AIME-style "
    "competition problems (numerical answers, integers 0-999). You make SMALL, SURGICAL edits to an "
    "existing skill to fix SPECIFIC failure modes - you do NOT rewrite it. You add concrete, "
    "operational DECISION PROCEDURES (not generic checklists) anchored to real errors."
)


def _excerpt(text: str, n: int = 700) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[:n] + " …[truncated]"


# ===================================================================
# M2: 失败驱动局部 patch (AIME 数值诊断)
# ===================================================================
def collect_failures(val_items, skill_content, s0_cache, n_diag=8, k=1):
    """在 val 上跑当前 skill (k 投票, EVAL_TEMP), 收集仍错题 (含 pred/correct/response) 做诊断样本."""
    set_temp(EVAL_TEMP)
    results = {}
    with ThreadPoolExecutor(max_workers=40) as ex:
        futs = {ex.submit(run_one, it, skill_content, "evolver_aime_diag", k): it for it in val_items}
        for fut in as_completed(futs):
            it = futs[fut]
            results[str(it["id"])] = (it, fut.result())
    failures = []
    for qid, (it, r) in results.items():
        if r["pass_rate"] == 0 and r["agent_ok"]:  # 仍错 (k 次全错)
            failures.append({
                "id": qid, "item": it,
                "pred": r.get("predicted_label", ""),
                "correct": r.get("correct_label", ""),
                "response_excerpt": _excerpt(r.get("response", ""), 600),
                "s0_pass": s0_cache.get(qid, {}).get("pass_rate", 0.0),
            })
    failures.sort(key=lambda f: f["s0_pass"])  # 优先 S0 也错的 (真 headroom)
    return failures[:n_diag], len(failures)


PATCH_TEMPLATE_AIME = """You are improving a system prompt (skill) for a mathematical-reasoning LLM solving AIME-style competition problems.

The CURRENT skill is below. It already works well on many problems but STILL FAILS on the diagnostic cases shown. Your job: make SMALL, SURGICAL edits to fix the SPECIFIC failure mode - add a concrete DECISION PROCEDURE (a step-by-step test the model can run), not a generic checklist.

Rules:
1. Do NOT rewrite the skill. Keep its structure. Add at most ONE new short subsection (or replace/extend one existing bullet) that addresses the failure mode.
2. The new content must be a CONCRETE, OPERATIONAL decision procedure anchored to the diagnostic cases - e.g. "For 'number of' questions, before counting, list the cases in a fixed order and verify pairwise disjointness by checking shared elements." NOT "count carefully".
3. Keep total skill under ~5000 chars. If near the limit, compress redundant existing bullets rather than appending.
4. Do NOT reference specific problem answers or this problem's particular numbers.
5. Focus area this round: {focus}
6. Rejected patches to avoid (do not repeat these ideas): {rejected}

## CURRENT SKILL
{skill}

## DIAGNOSTIC FAILURES (current skill still gets these wrong)
{diagnostics}

## YOUR OUTPUT
Output ONLY the edited skill text (full skill with your small edit applied). No meta commentary. START directly with the skill.
"""

# A2 消融 (no-failure): 通用 patch, 不喂诊断失败. 与 PATCH_TEMPLATE_AIME 唯一差别 = 无 DIAGNOSTIC FAILURES 节 +
# 无 rejected 反面教材, 隔离"失败驱动"机制的贡献. 仍保持小改 + 具体决策程序纪律, 公平对照.
GENERIC_PATCH_TEMPLATE_AIME = """You are improving a system prompt (skill) for a mathematical-reasoning LLM solving AIME-style competition problems.

The CURRENT skill is below. Make a SMALL, SURGICAL edit to improve it - add a concrete DECISION PROCEDURE (a step-by-step test the model can run) for the focus area, not a generic checklist.

Rules:
1. Do NOT rewrite the skill. Keep its structure. Add at most ONE new short subsection (or replace/extend one existing bullet) that addresses the focus area.
2. The new content must be a CONCRETE, OPERATIONAL decision procedure - e.g. "For 'number of' questions, before counting, list the cases in a fixed order and verify pairwise disjointness." NOT "count carefully".
3. Keep total skill under ~5000 chars. If near the limit, compress redundant existing bullets rather than appending.
4. Do NOT reference specific problem answers or this problem's particular numbers.
5. Focus area this round: {focus}

## CURRENT SKILL
{skill}

## YOUR OUTPUT
Output ONLY the edited skill text (full skill with your small edit applied). No meta commentary. START directly with the skill.
"""


def make_diagnostics(failures):
    parts = []
    for f in failures:
        it = f["item"]
        parts.append(
            f"### Problem {f['id']}\n"
            f"Problem: {str(it['problem'])[:700]}\n"
            f"Model predicted: {f['pred']!r} | Correct: {f['correct']!r}\n"
            f"Model reasoning excerpt:\n{f['response_excerpt']}"
        )
    return "\n\n".join(parts)


def generate_candidates(current_skill, failures, focus, rejected, K, temps, failure_driven=True):
    """生成 K 个候选 patch. 多样性靠 GEN_TEMP 真采样 + per-candidate 文字角度提示."""
    set_temp(GEN_TEMP)
    if failure_driven:
        diag = make_diagnostics(failures) if failures else "(no diagnostic failures this round - make a diversifying exploratory edit)"
        base_user = PATCH_TEMPLATE_AIME.format(skill=current_skill, diagnostics=diag,
                                               focus=focus, rejected=rejected or "(none)")
    else:
        base_user = GENERIC_PATCH_TEMPLATE_AIME.format(skill=current_skill, focus=focus)
    angle_hints = [
        "Emphasize the most concrete step-by-step test you can run on the diagnostic cases.",
        "Focus on the failure mode that recurs across the most diagnostic cases.",
        "Prefer a minimal edit (extend one existing bullet) over adding a new subsection.",
        "Target the case-enumeration / boundary distinction specifically.",
    ]
    cands = []
    with ThreadPoolExecutor(max_workers=min(K, 8)) as ex:
        futs = {}
        for i in range(K):
            hint = angle_hints[i % len(angle_hints)]
            user = f"Editor angle #{i+1}: {hint}\n\n" + base_user
            futs[ex.submit(llm, PATCHER_SYSTEM, user,
                           max_tokens=4096, stage="evolver_aime_patch")] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                cands.append((i, clean_skill(fut.result(), current_skill)))
            except Exception as e:
                print(f"  ⚠️ cand{i} patch 生成失败: {e!r}", flush=True)
                cands.append((i, current_skill))
    cands.sort(key=lambda x: x[0])
    return [c for _, c in cands]


# ===================================================================
# M3: 慢更新 - 蒸馏整合去冗余
# ===================================================================
CONSOLIDATE_TEMPLATE_AIME = """You are consolidating a system prompt (skill) for a mathematical-reasoning LLM solving AIME-style competition problems.

Over several iterations, multiple patches have been applied. The skill below may now be redundant, bloated, or internally repetitive. Consolidate it WITHOUT losing any operational decision procedure.

Rules:
1. Merge redundant/overlapping bullets into the strongest single version.
2. Remove pure padding or restatements of the same idea.
3. Keep EVERY concrete decision procedure (step-by-step test) - these are the valuable parts.
4. Preserve the section structure. Keep under ~4500 chars.
5. Do NOT weaken or remove content that addresses a specific failure mode.
6. Output ONLY the consolidated skill text. No meta commentary.

## SKILL TO CONSOLIDATE
{skill}
"""


def consolidate(skill):
    set_temp(GEN_TEMP)
    try:
        out = llm(PATCHER_SYSTEM, CONSOLIDATE_TEMPLATE_AIME.format(skill=skill),
                  max_tokens=4096, stage="evolver_aime_consolidate")
        c = clean_skill(out, skill)
        if len(c) < len(skill) * 0.7:
            return skill
        return c
    except Exception:
        return skill


def main():
    ap = argparse.ArgumentParser(description="Skill 自进化迭代器 - AIME 版 (免训练)")
    ap.add_argument("--init_skill", default=V2_SKILL_DEFAULT, help="起点 skill (默认 v2_aime 产物)")
    ap.add_argument("--s0", default=S0_DEFAULT)
    ap.add_argument("--data", default=AIME24_DATA)
    ap.add_argument("--train_n", type=int, default=18)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--s0_cache", default="", help="S₀ cache json (同条件 k 投票连续值), 缺省现场算")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--K", type=int, default=4, help="每轮候选数")
    ap.add_argument("--eval_k", type=int, default=3, help="评测每题采样投票次数")
    ap.add_argument("--n_diag", type=int, default=8, help="每轮诊断失败题数")
    ap.add_argument("--consolidate_every", type=int, default=3, help="每 N 轮慢更新")
    ap.add_argument("--patience", type=int, default=3, help="连续 N 轮无提升则探索/停")
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    # ── 机制消融 (leave-one-out, 互斥; 无 flag = FULL) ──
    ap.add_argument("--ablate_gepa", action="store_true", help="A1: K=1 单候选 (M1)")
    ap.add_argument("--ablate_failure", action="store_true", help="A2: 通用 patch 不喂诊断失败 (M2)")
    ap.add_argument("--ablate_consolidate", action="store_true", help="A3: 禁用慢更新 (M3)")
    ap.add_argument("--ablate_antideg", action="store_true", help="A4: 去 LOST/rejected/focus 轮换 (M4)")
    args = ap.parse_args()

    K_eff = 1 if args.ablate_gepa else args.K
    _abl = {"gepa": args.ablate_gepa, "failure": args.ablate_failure,
            "consolidate": args.ablate_consolidate, "antideg": args.ablate_antideg}
    ablation_label = "FULL" if not any(_abl.values()) else "ablate_" + "+".join(k for k, v in _abl.items() if v)
    print(f"🧪 [AIME] ablation={ablation_label} | K_eff={K_eff} | failure_driven={not args.ablate_failure} | "
          f"consolidate={not args.ablate_consolidate} | antideg={not args.ablate_antideg}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    init_skill = open(args.init_skill, encoding="utf-8").read().strip()
    s0 = open(args.s0, encoding="utf-8").read().strip()
    train_items, val_items = load_aime_split(args.data, args.train_n, args.split_seed)
    select_items = train_items + val_items

    if args.s0_cache and os.path.exists(args.s0_cache):
        s0_cache = json.load(open(args.s0_cache, encoding="utf-8"))
        missing = [str(it["id"]) for it in select_items if str(it["id"]) not in s0_cache]
        if missing:
            print(f"⚠️ s0_cache 缺 {len(missing)} 题, 补算 ...")
            extra = precompute_s0([it for it in select_items if str(it["id"]) in missing], s0, args.eval_k)
            s0_cache.update(extra)
            json.dump(s0_cache, open(os.path.join(out_dir, "s0_cache.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
    else:
        print(f"⏳ precomputing S₀ on select set (k={args.eval_k}) ...")
        s0_cache = precompute_s0(select_items, s0, args.eval_k)
        json.dump(s0_cache, open(os.path.join(out_dir, "s0_cache.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print(f"📄 init_skill: {len(init_skill)} chars | select(train+val): {len(select_items)} | "
          f"S₀ cache: {len(s0_cache)} | eval_k={args.eval_k}")

    base_pass, base_delta, _, base_per = eval_on(select_items, init_skill, s0_cache,
                                                 stage="evolver_aime_base", k=args.eval_k)
    base_ci = bootstrap_ci_lower(base_per)
    print(f"📊 init skill select: pass={base_pass:.3f} delta={base_delta:+.3f} CI_lower={base_ci:+.3f}")

    best_skill = init_skill
    best_val_delta = base_delta
    best_val_ci = base_ci
    rejected_patches = []
    no_improve = 0
    log = {"ts": ts, "env": "aime24", "init_skill": args.init_skill, "init_skill_hash": md5(init_skill),
           "ablation": ablation_label,
           "baseline_val": {"pass": base_pass, "delta": base_delta, "ci": base_ci},
           "iters": [], "args": vars(args)}

    for t in range(1, args.iters + 1):
        t0 = time.time()
        focus = "General mathematical reasoning improvement" if args.ablate_antideg else FAILURE_FOCI_AIME[(t - 1) % len(FAILURE_FOCI_AIME)]
        print(f"\n{'='*50}\nIter {t}/{args.iters}  focus={focus}\n{'='*50}")

        current_skill = best_skill
        current_hash = md5(current_skill)

        if args.ablate_failure:
            failures, n_fail = [], 0
            print(f"  [ablate_failure] 跳过 collect_failures (通用 patch, 无诊断)")
        else:
            failures, n_fail = collect_failures(select_items, best_skill, s0_cache,
                                                n_diag=args.n_diag, k=args.eval_k)
            print(f"  best skill select failures: {n_fail} (diagnosing {len(failures)})")

        explore = (not args.ablate_antideg) and no_improve >= args.patience
        if explore:
            temps = [0.9, 1.0, 1.1, 0.8][:K_eff]
            print(f"  ⚠️ no_improve={no_improve}, EXPLORATION mode (bolder angle hints)")
        else:
            temps = [0.4 + 0.15 * (i % 3) for i in range(K_eff)]

        rejected_feed = "" if args.ablate_antideg else "; ".join(rejected_patches[-3:])
        cands = generate_candidates(best_skill, failures, focus, rejected_feed,
                                    K_eff, temps, failure_driven=not args.ablate_failure)
        n_novel = sum(1 for c in cands if md5(c) != current_hash)
        print(f"  generated {len(cands)} candidates, {n_novel} novel (hash != current)")

        scored = []
        for ci_idx, cand in enumerate(cands):
            cp, cd, _, citem = eval_on(select_items, cand, s0_cache,
                                       stage="evolver_aime_val", k=args.eval_k)
            cl = bootstrap_ci_lower(citem)
            scored.append({"idx": ci_idx, "skill": cand, "hash": md5(cand),
                           "pass": cp, "delta": cd, "ci": cl, "len": len(cand)})
            print(f"  cand{ci_idx}: delta={cd:+.3f} CI={cl:+.3f} pass={cp:.3f} len={len(cand)} "
                  f"novel={md5(cand) != current_hash}", flush=True)

        novel_positive = [s for s in scored if s["hash"] != current_hash and s["ci"] > 0]
        reject_reason = None
        if novel_positive:
            winner = max(novel_positive, key=lambda s: s["delta"])
        else:
            novel_all = [s for s in scored if s["hash"] != current_hash]
            if novel_all:
                winner = max(novel_all, key=lambda s: s["delta"])
                reject_reason = (f"no novel candidate with CI>0 "
                                 f"(best novel delta={winner['delta']:+.3f} CI={winner['ci']:+.3f})")
            else:
                winner = None
                reject_reason = "no novel candidate generated (all == current)"

        improved = False
        if winner is not None:
            improved = (winner["delta"] > best_val_delta and winner["ci"] > 0
                        and winner["hash"] != current_hash)
            if not args.ablate_antideg and winner["pass"] < base_pass - 0.06:
                print(f"  ⚠️ winner pass {winner['pass']:.3f} << base {base_pass:.3f}, REJECT (LOST risk)")
                improved = False
                reject_reason = f"LOST: pass drop to {winner['pass']:.3f}"
                rejected_patches.append(f"iter{t}: caused pass drop to {winner['pass']:.3f}")

        if improved:
            assert md5(winner["skill"]) != current_hash, "accept 需产物已改动"
            best_skill = winner["skill"]
            best_val_delta = winner["delta"]
            best_val_ci = winner["ci"]
            no_improve = 0
            print(f"  ✅ ACCEPT iter{t}: delta={winner['delta']:+.3f} CI={winner['ci']:+.3f} "
                  f"hash {current_hash}->{winner['hash']}")
        else:
            no_improve += 1
            if not args.ablate_antideg and winner is not None and reject_reason and "LOST" not in reject_reason:
                rejected_patches.append(f"iter{t} focus={focus}: {reject_reason}")
            print(f"  ✗ KEEP PREV (no_improve={no_improve}) reason={reject_reason}")

        consolidated = False
        if not args.ablate_consolidate and t % args.consolidate_every == 0 and len(best_skill) > 4000:
            before = len(best_skill)
            before_hash = md5(best_skill)
            best_skill = consolidate(best_skill)
            consolidated = True
            print(f"  🔧 consolidated: {before} -> {len(best_skill)} chars hash {before_hash}->{md5(best_skill)}")

        with open(os.path.join(out_dir, f"skill_iter{t}.md"), "w", encoding="utf-8") as f:
            f.write(best_skill)
        log["iters"].append({
            "t": t, "secs": round(time.time() - t0, 1), "focus": focus,
            "n_val_failures": n_fail, "n_candidates": len(cands),
            "n_novel_candidates": n_novel, "current_hash": current_hash,
            "winner_hash": winner["hash"] if winner else None,
            "reject_reason": reject_reason,
            "best_val_delta": round(best_val_delta, 4), "best_val_ci": round(best_val_ci, 4),
            "improved": improved, "no_improve": no_improve, "consolidated": consolidated,
            "skill_len": len(best_skill), "skill_hash": md5(best_skill),
        })
        json.dump(log, open(os.path.join(out_dir, "evolver_log.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        if no_improve >= args.patience * 2:
            print(f"  ⏹ early stop: no_improve={no_improve}")
            break

    with open(os.path.join(out_dir, "best_skill_evolved.md"), "w", encoding="utf-8") as f:
        f.write(best_skill)
    json.dump(log, open(os.path.join(out_dir, "evolver_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    final_hash = md5(best_skill)
    init_hash = md5(init_skill)
    changed = final_hash != init_hash
    print(f"\n{'='*50}\n✅ Done. init val delta={base_delta:+.3f} -> best val delta={best_val_delta:+.3f} (CI={best_val_ci:+.3f})")
    print(f"   skill: {len(init_skill)} -> {len(best_skill)} chars | hash {init_hash} -> {final_hash}")
    if not changed:
        print(f"   ⚠️ 产物未改动 (best == init)! 优化器未产出有效 patch, 需检查生成/门控.")
    else:
        print(f"   ✅ 产物已改动 (hash 不同), 可进 AIME25 test 复评.")
    print(f"   产出: {out_dir}/best_skill_evolved.md")
    print(f"   评测 AIME25: python rl/train/verl1/eval_aime25_multiseed.py --evolver_best {out_dir}/best_skill_evolved.md")


if __name__ == "__main__":
    main()
