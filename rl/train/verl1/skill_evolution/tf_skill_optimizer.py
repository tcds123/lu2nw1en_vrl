#!/usr/bin/env python3
"""
Training-Free GRPO for Skill Optimization (免训练线).

对应 docs/training_vs_training_free_parallel_plan.md §2。零 GPU: rewriter = DeepSeek-V3.2 API。
与训练线 (custom_reward_lcb.py GRPO) 共用同一套 B/C/evaluator/S_0/s0_cache/split, 结果可直接同表对位。

算法 (复用 experience/extractor.py 群组蒸馏思想 + training_free_grpo_fusion_design.md):
  S ← S_0
  for iter t = 1..T:
    for each q in train: 采样 G 个 skill 变体 → B 生成代码 → 单测 → delta = pass - s0_pass[q]
    组内好坏轨迹对比 → GROUP_ADVANTAGE_TEMPLATE 蒸馏经验
    跨题合并 → 更新 experience_pool
    S_new ← Rewriter(S_0, experience_pool) 合成全局 skill
    门控: val 上 delta(S_new) >= delta(S) 才接受 (学 RSEA/GRASP 防回归), 否则保留 S
  产出: tf_global_skill.md + experience_pool_tf.json + 每轮快照

eval_lcb_skillopt.py --skill_source tf_global 用产出 skill 评 test=143。

环境变量: 与 train_lcb_skillopt.sh 一致 (QWEN_CHAT_*, TARGET_BACKEND/OPTIMIZER_BACKEND=qwen_chat)。
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
sys.path.insert(0, "/data/zhuldz/lunwen/data/OpenCodeInstruct")

# 必须在 import skillopt.model 前设 (见 precompute_s0_baseline.py 注释)
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_API_KEY", "")  # 旧 key msk-bf2324.. 2026-07-23 失效, 换新
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1")
os.environ.setdefault("QWEN_CHAT_MODEL", "/model/DeepSeek-V3.2")
os.environ.setdefault("QWEN_CHAT_AIGC_USER", "zhuldz")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "4096")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "28")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

from skillopt.envs.lcb.rollout import _build_system, _build_user  # noqa: E402
from skillopt.envs.lcb.evaluator import evaluate as lcb_evaluate  # noqa: E402
from skillopt.model import chat_target  # noqa: E402

from json2parquet_skillopt import (  # noqa: E402
    construct_training_prompt,
    _format_test_code_for_prompt,
    truncate_test_code,
)

# 复用经验蒸馏模板 (Group Advantage + Cross-problem merge)
from experience.templates import GROUP_ADVANTAGE_TEMPLATE, CROSS_PROBLEM_MERGE_TEMPLATE  # noqa: E402

S0_DEFAULT = "/data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md"
SPLIT_DEFAULT = "/data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42"
S0_CACHE_DEFAULT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/s0_baseline_lcb.json"
OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill"


# ===================================================================
# LLM 调用 (rewriter / 蒸馏, 复用 chat_target, stage 区分)
# ===================================================================
def llm(system: str, user: str, *, max_tokens=4096, temperature=0.7, stage="tf_rewriter") -> str:
    """单次 LLM 调用。rewriter/蒸馏用高 temp 采样变体; B 模型用 0 temp (run_one 单独调)。"""
    resp, _ = chat_target(
        system=system, user=user,
        max_completion_tokens=max_tokens, retries=5, stage=stage,
        reasoning_effort=None, timeout=300,
    )
    return resp


REWRITER_SYSTEM = (
    "You are an expert System Prompt Engineer optimizing a code-generation AI for competitive programming (stdin/stdout). "
    "You rewrite an INITIAL system prompt (S_0) into a BETTER skill for a specific task, guided by distilled experience."
)


def make_rewriter_user(s0: str, question: str, test_code_repr: str, experiences: str, variant_hint: str) -> str:
    """构造 rewriter prompt。复用 construct_training_prompt 的 SePO 风格指令, 注入经验池。"""
    exp_section = (
        f"## DISTILLED EXPERIENCE (apply these where relevant)\n{experiences}\n\n"
        if experiences.strip() else ""
    )
    return (
        f"Improve the INITIAL system prompt below for the TARGET TASK, using the distilled experience.\n\n"
        f"Rules:\n"
        f"1. KEEP what works in S_0. ADD task-specific PROCEDURAL instructions targeting LIKELY FAILURE MODES "
        f"(mandatory hand-trace of sample I/O before coding, anti-looping discipline, off-by-one/boundary checks, "
        f"output-format discipline, complexity check vs constraints).\n"
        f"2. Be CONCRETE, not generic. Cite the specific edge case or invariant.\n"
        f"3. Incorporate the distilled experience where it applies to this task; ignore irrelevant experience.\n"
        f"4. Do NOT include solution code. Do NOT reference test case values.\n"
        f"5. {variant_hint}\n\n"
        f"{exp_section}"
        f"## INITIAL SYSTEM PROMPT (S_0)\n{s0}\n\n"
        f"## TARGET TASK\n### Problem\n{question}\n\n"
        f"### Test I/O contract (representative subset)\n{test_code_repr}\n\n"
        f"## YOUR OUTPUT\nOutput ONLY the improved system prompt text. No reasoning, no meta commentary. "
        f"START directly with the improved prompt.\n"
    )


def clean_skill_text(s: str, fallback: str) -> str:
    """清理 thinking/HTML 注释; 空则回退 fallback。"""
    s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL).strip()
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()
    return s if s else fallback


def extract_experiences(text: str) -> str:
    """从 LLM 输出抽 <Experiences>...</Experiences> 块; 没有则用全文。"""
    m = re.search(r'<Experiences>(.*?)</Experiences>', text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ===================================================================
# B 模型生成 + 单测 (与 custom_reward_lcb.py / precompute 一致)
# ===================================================================
def run_one(item: dict, skill_content: str, stage: str = "tf_b") -> dict:
    qid = str(item["id"])
    res = {"id": qid, "difficulty": item.get("difficulty", "?"),
           "hard": 0, "soft": 0.0, "pass_rate": -3.0, "agent_ok": False, "fail_reason": ""}
    try:
        system = _build_system(skill_content)
        user = _build_user(item)
        response, _ = chat_target(system=system, user=user, max_completion_tokens=4096,
                                  retries=5, stage=stage, timeout=300)
        ev = lcb_evaluate(response, item["test_code"])
        res.update({"hard": ev["hard"], "soft": ev["soft"], "pass_rate": ev["pass_rate"],
                    "agent_ok": True, "fail_reason": ev.get("fail_reason", "")[:200]})
    except Exception as e:
        res["fail_reason"] = f"exception: {e!r}"
    return res


# ===================================================================
# 核心循环
# ===================================================================
def load_split(split_dir, split, truncate=True):
    path = os.path.join(split_dir, split, "items.json")
    items = json.load(open(path, encoding="utf-8"))
    if truncate:
        for it in items:
            it["test_code"] = truncate_test_code(it.get("test_code", ""))
    return items


def rollout_group(item, s0, experiences, s0_pass, g, variant_temps):
    """对单题生成 g 个 skill 变体并评测, 返回 (qid, [ {skill, pass, delta, temp} ])."""
    qid = str(item["id"])
    question = str(item["question"])
    test_repr = _format_test_code_for_prompt(item["test_code"])
    variants = []
    # 生成 g 个变体 (并发)
    hints = [f"Produce variation #{i+1}: take a slightly different angle on the failure modes." for i in range(g)]
    with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
        futs = {
            ex.submit(llm, REWRITER_SYSTEM,
                      make_rewriter_user(s0, question, test_repr, experiences, hints[i]),
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
    # 评测每个变体 (B 模型 + 单测)
    with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
        futs = {ex.submit(run_one, item, skills[i]): i for i in range(g)}
        results = [None] * g
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
    for i in range(g):
        pr = results[i]["pass_rate"] if results[i]["pass_rate"] >= 0 else 0.0
        variants.append({"skill": skills[i], "pass": pr, "delta": pr - s0_pass,
                         "hard": results[i]["hard"], "agent_ok": results[i]["agent_ok"],
                         "temp": variant_temps[i]})
    return qid, variants


def distill_experiences(problem, variants, num_exp=3):
    """组内好坏对比蒸馏经验 (GROUP_ADVANTAGE_TEMPLATE)."""
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
        learning_objective="Identify prompt-writing techniques that increased unit-test pass-rate (delta vs S_0 baseline).",
        num_experiences=num_exp,
    )
    try:
        out = llm("You are an expert in prompt engineering for code generation.",
                  prompt, max_tokens=1500, temperature=0.3, stage="tf_distill")
        return extract_experiences(out)
    except Exception as e:
        return f"(distill failed: {e!r})"


def merge_experiences(all_exp, current_pool, max_exp=12):
    """跨题合并去重 (CROSS_PROBLEM_MERGE_TEMPLATE)."""
    prompt = CROSS_PROBLEM_MERGE_TEMPLATE.format(
        num_problems=len(all_exp),
        candidate_experiences="\n\n---\n\n".join(all_exp),
        current_pool=current_pool,
        max_experiences=max_exp,
    )
    try:
        out = llm("You are managing a pool of prompt engineering best practices for code generation.",
                  prompt, max_tokens=2000, temperature=0.3, stage="tf_merge")
        return extract_experiences(out)
    except Exception as e:
        return current_pool


def compose_global_skill(s0, experiences):
    """用经验池合成新一轮全局 skill (S_0 + 经验合并)."""
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
    """在 item 集上评 skill 平均 pass-rate, 返回 (mean_pass, mean_delta, n)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", default=S0_DEFAULT)
    ap.add_argument("--split_dir", default=SPLIT_DEFAULT)
    ap.add_argument("--s0_cache", default=S0_CACHE_DEFAULT)
    ap.add_argument("--iters", type=int, default=3, help="T 离线迭代轮数")
    ap.add_argument("--g", type=int, default=4, help="每题采样变体数")
    ap.add_argument("--train_n", type=int, default=41, help="每轮采样 train 题数 (≤41)")
    ap.add_argument("--num_exp", type=int, default=3, help="每题蒸馏经验条数")
    ap.add_argument("--max_pool", type=int, default=12, help="经验池上限")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.s0, encoding="utf-8") as f:
        s0 = f.read().strip()
    s0_cache = json.load(open(args.s0_cache, encoding="utf-8"))
    train_all = load_split(args.split_dir, "train")
    val_items = load_split(args.split_dir, "val")
    # 限定 train_n (按 hard 优先采样, hard 是 headroom 所在)
    hard_first = sorted(train_all, key=lambda it: (0 if it.get("difficulty") == "hard" else 1))
    train_items = hard_first[: args.train_n]
    print(f"📄 S_0: {len(s0)} chars | s0_cache: {len(s0_cache)} 题 | train: {len(train_items)} (hard优先) | val: {len(val_items)}")

    experience_pool = ""
    S_global = s0
    # baseline: val 上 S_0 的 delta
    base_pass, base_delta, _ = eval_on(val_items, s0, s0_cache, stage="tf_val_s0")
    print(f"📊 val baseline S_0: pass={base_pass:.3f} delta={base_delta:+.3f}")
    best_val_delta = base_delta
    best_S = s0

    log = {"ts": ts, "baseline_val": {"pass": base_pass, "delta": base_delta},
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
            print(f"  [{qi+1}/{len(train_items)}] {qid} {item.get('difficulty')}: best_delta={best_d:+.3f} "
                  f"(deltas={[round(v['delta'],2) for v in variants]})", flush=True)
        # 跨题合并
        experience_pool = merge_experiences(all_exp, experience_pool, max_exp=args.max_pool)
        # 合成全局 skill
        S_new = compose_global_skill(s0, experience_pool)
        # 门控: val 上比较
        new_pass, new_delta, _ = eval_on(val_items, S_new, s0_cache, stage="tf_val")
        accepted = new_delta >= best_val_delta
        print(f"  iter {t}: val new delta={new_delta:+.3f} vs best {best_val_delta:+.3f} → "
              f"{'ACCEPT' if accepted else 'KEEP PREV'}")
        if accepted:
            best_val_delta = new_delta
            best_S = S_new
        S_global = S_new
        # 快照
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

    # 产出最终
    with open(os.path.join(out_dir, "tf_global_skill.md"), "w", encoding="utf-8") as f:
        f.write(best_S)
    with open(os.path.join(out_dir, "experience_pool_tf.json"), "w", encoding="utf-8") as f:
        json.dump({"pool": experience_pool, "log": log}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "tf_run_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}\n✅ Done. best_val_delta={best_val_delta:+.3f} (baseline {base_delta:+.3f})")
    print(f"   产出: {out_dir}/tf_global_skill.md (+ experience_pool_tf.json)")
    print(f"   评测: python rl/train/verl1/eval_lcb_skillopt.py --skill_source tf_global "
          f"--tf_skill {out_dir}/tf_global_skill.md --output ...")


if __name__ == "__main__":
    main()
