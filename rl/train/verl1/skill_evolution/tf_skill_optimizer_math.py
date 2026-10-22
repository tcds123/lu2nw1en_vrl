#!/usr/bin/env python3
"""
Training-Free GRPO for Skill Optimization — 数学版 (免训练线, LiveMathematicianBench).

对应 memory/math-trial-direction-20260724.md + memory/math-baseline-samecond-20260724.md。
零 GPU: rewriter = DeepSeek-V3.2 API。与 LCB 版 (tf_skill_optimizer.py) 同算法, 仅 B/C 接口换成数学 env:
  - B: livemathematicianbench.rollout._build_system / _build_user
  - C: livemathematicianbench.evaluator.evaluate (确定性 <answer> 标签验证, em=0/1)
  - 无 test_code: rewriter 针对 question + choices 改写, 不引用 stdin/stdout
  - S_0 = skillopt/envs/livemathematicianbench/skills/initial.md (同条件 baseline 33.1%)

真问题: 数学上 TF-global 能否超过 S_0? (代码上 TF-global 反而比 S_0 差, 见 tf-global-reversal)

算法 (同 LCB 版, 复用 experience/extractor 群组蒸馏):
  S ← S_0
  for iter t = 1..T:
    for each q in train: 采样 G 个 skill 变体 → B 生成答案 → em → delta = em - s0_em[q]
    组内好坏对比 → GROUP_ADVANTAGE_TEMPLATE 蒸馏经验
    跨题合并 → 更新 experience_pool
    S_new ← Rewriter(S_0, experience_pool) 合成全局 skill
    门控: val 上 delta(S_new) >= delta(S) 才接受, 否则保留 S
  产出: tf_global_skill_math.md + experience_pool_tf_math.json + 每轮快照

环境变量: QWEN_CHAT_* (新 key msk-4601b27), TARGET_BACKEND/OPTIMIZER_BACKEND=qwen_chat。
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)
_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)

# 必须在 import skillopt.model 前设
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_API_KEY", "")  # 由环境变量提供 key
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1")
os.environ.setdefault("QWEN_CHAT_MODEL", "/model/DeepSeek-V3.2")
os.environ.setdefault("QWEN_CHAT_AIGC_USER", "zhuldz")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "4096")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "28")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

from skillopt.envs.livemathematicianbench.rollout import _build_system, _build_user  # noqa: E402
from skillopt.envs.livemathematicianbench.evaluator import evaluate as math_evaluate  # noqa: E402
from skillopt.model import chat_target  # noqa: E402

# 复用经验蒸馏模板
from experience.templates import GROUP_ADVANTAGE_TEMPLATE, CROSS_PROBLEM_MERGE_TEMPLATE  # noqa: E402

S0_DEFAULT = "/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md"
SPLIT_DEFAULT = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_2-1-7_seed42"
OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math"


# ===================================================================
# LLM 调用
# ===================================================================
def llm(system: str, user: str, *, max_tokens=4096, temperature=0.7, stage="tf_rewriter") -> str:
    resp, _ = chat_target(
        system=system, user=user,
        max_completion_tokens=max_tokens, retries=5, stage=stage,
        reasoning_effort=None, timeout=300,
    )
    return resp


REWRITER_SYSTEM = (
    "You are an expert System Prompt Engineer optimizing a mathematical-reasoning AI for "
    "multiple-choice questions (LiveMathematicianBench). "
    "You rewrite an INITIAL system prompt (S_0) into a BETTER skill for a specific task, "
    "guided by distilled experience."
)


def _format_choices(choices):
    return "\n".join(f"{c['label']}. {c['text']}" for c in choices)


def make_rewriter_user(s0: str, question: str, choices_repr: str, experiences: str, variant_hint: str) -> str:
    exp_section = (
        f"## DISTILLED EXPERIENCE (apply these where relevant)\n{experiences}\n\n"
        if experiences.strip() else ""
    )
    return (
        f"Improve the INITIAL system prompt below for the TARGET MATH MCQ TASK, using the distilled experience.\n\n"
        f"Rules:\n"
        f"1. KEEP what works in S_0. ADD task-specific PROCEDURAL instructions targeting LIKELY FAILURE MODES "
        f"for theorem-level math MCQs: compare ALL options before committing; track exact quantifiers "
        f"('there exists' vs 'for every', 'iff'); check equality cases and extremal wording; verify hypotheses "
        f"and domain; distinguish a theorem from a weakening (dropped characterization/equality) or an "
        f"overstatement (upgraded regularity/removed scale restriction); MANDATORY final answer as the single "
        f"choice label inside <answer>...</answer> tags.\n"
        f"2. Be CONCRETE, not generic. Cite the specific option distinction or invariant in this task.\n"
        f"3. Incorporate the distilled experience where it applies; ignore irrelevant experience.\n"
        f"4. Do NOT solve the problem. Do NOT state which option is correct. Do NOT reference answer labels.\n"
        f"5. {variant_hint}\n\n"
        f"{exp_section}"
        f"## INITIAL SYSTEM PROMPT (S_0)\n{s0}\n\n"
        f"## TARGET TASK\n### Problem\n{question}\n\n"
        f"### Choices\n{choices_repr}\n\n"
        f"## YOUR OUTPUT\nOutput ONLY the improved system prompt text. No reasoning, no meta commentary. "
        f"START directly with the improved prompt.\n"
    )


def clean_skill_text(s: str, fallback: str) -> str:
    s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL).strip()
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()
    return s if s else fallback


def extract_experiences(text: str) -> str:
    m = re.search(r'<Experiences>(.*?)</Experiences>', text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ===================================================================
# B 模型生成 + 验证 (数学: <answer> 标签确定性验证)
# ===================================================================
def run_one(item: dict, skill_content: str, stage: str = "tf_b") -> dict:
    qid = str(item["id"])
    res = {"id": qid, "theorem_type": item.get("theorem_type", "?"),
           "pass_rate": -3.0, "agent_ok": False, "fail_reason": ""}
    try:
        system = _build_system(skill_content)
        user = _build_user(item)
        response, _ = chat_target(system=system, user=user, max_completion_tokens=4096,
                                  retries=5, stage=stage, timeout=300)
        ev = math_evaluate(response, item["correct_choice"], item["choices"])
        res.update({"pass_rate": float(ev["em"]), "agent_ok": True,
                    "predicted_label": ev.get("predicted_label", ""),
                    "correct_label": ev.get("correct_label", ""),
                    "fail_reason": "" if ev["em"] else "wrong"})
    except Exception as e:
        res["fail_reason"] = f"exception: {e!r}"
    return res


# ===================================================================
# 核心循环
# ===================================================================
def load_split(split_dir, split):
    path = os.path.join(split_dir, split, "items.json")
    return json.load(open(path, encoding="utf-8"))


def rollout_group(item, s0, experiences, s0_pass, g, variant_temps):
    """对单题生成 g 个 skill 变体并评测, 返回 (qid, [ {skill, pass, delta, temp} ])."""
    qid = str(item["id"])
    question = str(item["question"])
    choices_repr = _format_choices(item["choices"])
    hints = [f"Produce variation #{i+1}: take a slightly different angle on the failure modes." for i in range(g)]
    with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
        futs = {
            ex.submit(llm, REWRITER_SYSTEM,
                      make_rewriter_user(s0, question, choices_repr, experiences, hints[i]),
                      dict(max_tokens=4096, temperature=variant_temps[i], stage="tf_rewriter")): i
            for i in range(g)
        }
        skills = [None] * g
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                skills[i] = clean_skill_text(fut.result(), s0)
            except Exception:
                skills[i] = s0
    with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
        futs = {ex.submit(run_one, item, skills[i]): i for i in range(g)}
        results = [None] * g
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
    variants = []
    for i in range(g):
        pr = results[i]["pass_rate"] if results[i]["pass_rate"] >= 0 else 0.0
        variants.append({"skill": skills[i], "pass": pr, "delta": pr - s0_pass,
                         "agent_ok": results[i]["agent_ok"],
                         "pred": results[i].get("predicted_label", ""),
                         "gold": results[i].get("correct_label", ""),
                         "temp": variant_temps[i]})
    return qid, variants


def distill_experiences(problem, variants, num_exp=3):
    ranked = sorted(variants, key=lambda v: v["delta"], reverse=True)
    summaries = []
    for idx, v in enumerate(ranked):
        summaries.append(
            f"### Rollout {idx+1} (delta={v['delta']:+.3f}, pass={v['pass']:.3f})\n"
            f"<system_prompt>\n{v['skill'][:1500]}\n</system_prompt>"
        )
    prompt = GROUP_ADVANTAGE_TEMPLATE.format(
        num_rollouts=len(ranked),
        problem=problem[:2000],
        rollout_summaries="\n\n".join(summaries),
        learning_objective="Identify prompt-writing techniques that increased answer accuracy (delta vs S_0 baseline).",
        num_experiences=num_exp,
    )
    try:
        out = llm("You are an expert in prompt engineering for mathematical reasoning.",
                  prompt, max_tokens=1500, temperature=0.3, stage="tf_distill")
        return extract_experiences(out)
    except Exception as e:
        return f"(distill failed: {e!r})"


def merge_experiences(all_exp, current_pool, max_exp=12):
    prompt = CROSS_PROBLEM_MERGE_TEMPLATE.format(
        num_problems=len(all_exp),
        candidate_experiences="\n\n---\n\n".join(all_exp),
        current_pool=current_pool,
        max_experiences=max_exp,
    )
    try:
        out = llm("You are managing a pool of prompt engineering best practices for mathematical reasoning.",
                  prompt, max_tokens=2000, temperature=0.3, stage="tf_merge")
        return extract_experiences(out)
    except Exception as e:
        return current_pool


def compose_global_skill(s0, experiences):
    if not experiences.strip():
        return s0
    user = (
        f"Rewrite the INITIAL system prompt into a stronger GLOBAL skill by incorporating the distilled experience.\n"
        f"Keep it concise and concrete. Output ONLY the improved system prompt text, no meta commentary.\n\n"
        f"## INITIAL SYSTEM PROMPT (S_0)\n{s0}\n\n"
        f"## DISTILLED EXPERIENCE\n{experiences}\n\n"
        f"## YOUR OUTPUT\n"
    )
    try:
        out = llm(REWRITER_SYSTEM, user, max_tokens=4096, temperature=0.3, stage="tf_compose")
        return clean_skill_text(out, s0)
    except Exception:
        return s0


def eval_on(items, skill_content, s0_cache, stage="tf_val"):
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, it, skill_content, stage): it for it in items}
        for fut in as_completed(futs):
            results.append((futs[fut]["id"], fut.result()))
    deltas = []
    for qid, r in results:
        if r["pass_rate"] >= 0:
            deltas.append(r["pass_rate"] - s0_cache.get(qid, {}).get("pass_rate", 0.0))
    mean_pass = sum(r["pass_rate"] for _, r in results if r["pass_rate"] >= 0) / max(1, len(deltas))
    mean_delta = sum(deltas) / max(1, len(deltas))
    return mean_pass, mean_delta, len(deltas)


# ===================================================================
# S_0 预计算 (若 s0_cache 不存在则现场算)
# ===================================================================
def precompute_s0(items, stage="tf_s0"):
    """现场算 S_0 每题 pass_rate, 写入 cache dict {qid: {pass_rate}}."""
    cache = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, it, S0_CONTENT, stage): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            r = fut.result()
            cache[str(it["id"])] = {"pass_rate": max(0.0, r["pass_rate"])}
    return cache


S0_CONTENT = ""  # main 里填


def main():
    global S0_CONTENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", default=S0_DEFAULT)
    ap.add_argument("--split_dir", default=SPLIT_DEFAULT)
    ap.add_argument("--s0_cache", default="", help="可选 S_0 cache json, 缺省现场算")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--train_n", type=int, default=35, help="每轮 train 题数 (数学 train=35)")
    ap.add_argument("--num_exp", type=int, default=3)
    ap.add_argument("--max_pool", type=int, default=12)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.s0, encoding="utf-8") as f:
        s0 = f.read().strip()
    S0_CONTENT = s0
    train_items = load_split(args.split_dir, "train")
    val_items = load_split(args.split_dir, "val")
    train_items = train_items[: args.train_n]

    # S_0 cache
    if args.s0_cache and os.path.exists(args.s0_cache):
        s0_cache = json.load(open(args.s0_cache, encoding="utf-8"))
    else:
        print("⏳ precomputing S_0 baseline on train+val ...")
        all_items = train_items + val_items
        s0_cache = precompute_s0(all_items)
        with open(os.path.join(out_dir, "s0_cache_math.json"), "w", encoding="utf-8") as f:
            json.dump(s0_cache, f, ensure_ascii=False, indent=2)
    print(f"📄 S_0: {len(s0)} chars | s0_cache: {len(s0_cache)} 题 | train: {len(train_items)} | val: {len(val_items)}")

    experience_pool = ""
    S_global = s0
    base_pass, base_delta, _ = eval_on(val_items, s0, s0_cache, stage="tf_val_s0")
    print(f"📊 val baseline S_0: pass={base_pass:.3f} delta={base_delta:+.3f}")
    best_val_delta = base_delta
    best_S = s0

    log = {"ts": ts, "env": "livemathematicianbench", "baseline_val": {"pass": base_pass, "delta": base_delta},
           "iters": [], "args": vars(args)}

    for t in range(1, args.iters + 1):
        t0 = time.time()
        print(f"\n{'='*50}\nIter {t}/{args.iters}  (experience_pool={len(experience_pool)} chars)\n{'='*50}")
        all_exp = []
        iter_deltas = []
        for qi, item in enumerate(train_items):
            qid, variants = rollout_group(item, s0, experience_pool,
                                          s0_cache.get(str(item["id"]), {}).get("pass_rate", 0.0),
                                          args.g, [0.7 + 0.1 * (i % 3) for i in range(args.g)])
            best_d = max(v["delta"] for v in variants)
            iter_deltas.append(best_d)
            exp = distill_experiences(str(item["question"]), variants, num_exp=args.num_exp)
            all_exp.append(exp)
            print(f"  [{qi+1}/{len(train_items)}] {qid} {item.get('theorem_type','?')}: best_delta={best_d:+.3f} "
                  f"(deltas={[round(v['delta'],2) for v in variants]})", flush=True)
        experience_pool = merge_experiences(all_exp, experience_pool, max_exp=args.max_pool)
        S_new = compose_global_skill(s0, experience_pool)
        new_pass, new_delta, _ = eval_on(val_items, S_new, s0_cache, stage="tf_val")
        accepted = new_delta >= best_val_delta
        print(f"  iter {t}: val new delta={new_delta:+.3f} vs best {best_val_delta:+.3f} → "
              f"{'ACCEPT' if accepted else 'KEEP PREV'}")
        if accepted:
            best_val_delta = new_delta
            best_S = S_new
        S_global = S_new
        with open(os.path.join(out_dir, f"skill_iter{t}.md"), "w", encoding="utf-8") as f:
            f.write(S_global)
        with open(os.path.join(out_dir, f"experience_iter{t}.md"), "w", encoding="utf-8") as f:
            f.write(experience_pool)
        log["iters"].append({
            "t": t, "secs": round(time.time() - t0, 1),
            "train_mean_best_delta": round(sum(iter_deltas) / max(1, len(iter_deltas)), 4),
            "val_new_delta": round(new_delta, 4), "val_best_delta": round(best_val_delta, 4),
            "accepted": accepted,
        })

    with open(os.path.join(out_dir, "tf_global_skill_math.md"), "w", encoding="utf-8") as f:
        f.write(best_S)
    with open(os.path.join(out_dir, "experience_pool_tf_math.json"), "w", encoding="utf-8") as f:
        json.dump({"pool": experience_pool, "log": log}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "tf_run_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}\n✅ Done. best_val_delta={best_val_delta:+.3f} (baseline {base_delta:+.3f})")
    print(f"   产出: {out_dir}/tf_global_skill_math.md")
    print(f"   评测 test=124: bash run_midea.sh evalskill {out_dir}/tf_global_skill_math.md")


if __name__ == "__main__":
    main()
