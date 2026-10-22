#!/usr/bin/env python3
"""
方向3 剪枝评估 — 逐条 Skill 的 Baseline 对照实验

对每条 skill 在固定任务集上做 paired 对比:
  无此 skill (baseline) vs 仅注入此条 skill (single_skill)

输出每条 skill 的 per-task delta，用于剪枝决策。

使用:
    # 快速评估 (10 tasks, top 20 skills)
    python -m agent.b_agent.prune_eval --n_tasks 10 --top_skills 20 --dataset bigcodebench

    # 全量评估 (20 tasks, all skills)
    python -m agent.b_agent.prune_eval --n_tasks 20 --dataset humaneval_pro
"""

import sys, os, json, time, random, argparse, logging, re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

PROJECT_ROOT = "/data/zhuldz/lunwen"
sys.path.insert(0, PROJECT_ROOT)

from agent.b_agent.core import BAgent
from agent.b_agent.skills import SkillLibrary, Skill
from agent.b_agent.ablation_test import (
    load_training_pool, extract_real_query, compute_execution_pass_rate, call_c_judge
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SKILLS_DIR = "/data/zhuldz/lunwen/agent/b_agent/skills"
OUTPUT_DIR = "/data/zhuldz/lunwen/agent/b_agent/ablation_results"


@dataclass
class SkillEvalResult:
    """单条 skill 在单个 task 上的评估"""
    skill_id: str
    task: str
    task_id: int
    baseline_score: float = 0.0       # 无此 skill
    skill_score: float = 0.0          # 仅此 skill
    delta: float = 0.0               # skill_score - baseline_score
    baseline_pass_rate: float = 0.0
    skill_pass_rate: float = 0.0
    baseline_c_score: float = 0.0
    skill_c_score: float = 0.0


def build_system_prompt_with_skill(base_prompt: str, skill: Skill) -> str:
    """将单条 skill 注入 system prompt，带清晰的标记"""
    skill_text = skill.to_markdown() if hasattr(skill, 'to_markdown') else f"[{skill.id}] {skill.content}"
    return (
        f"{base_prompt}\n\n"
        f"## Active Skill\n"
        f"{skill_text}\n\n"
        f"IMPORTANT: Apply the skill above when relevant to the task."
    )


def evaluate_single_skill_on_task(
    agent: BAgent,
    skill: Skill,
    task_query: str,
    gt: str,
    unit_tests: List[str],
    base_system_prompt: str,
    task_id: int,
    skill_id: str,
    baseline_score: float,
    baseline_pass_rate: float,
    baseline_c_score: float,
) -> SkillEvalResult:
    """对单个 task 评估单条 skill"""
    result = SkillEvalResult(
        skill_id=skill_id,
        task=task_query[:200],
        task_id=task_id,
        baseline_score=baseline_score,
        baseline_pass_rate=baseline_pass_rate,
        baseline_c_score=baseline_c_score,
    )

    try:
        full_sys = build_system_prompt_with_skill(base_system_prompt, skill)
        code = agent.generate_code(task=task_query, system_prompt=full_sys)
        pass_rate = compute_execution_pass_rate(code, unit_tests)
        c_score = call_c_judge(code, gt, base_system_prompt, task_query, hash(f"{task_id}_{skill_id}") % 100000)
        exec_score = pass_rate if pass_rate >= 0 else -3.0
        result.skill_score = 0.7 * max(0, exec_score) * 10 + 0.3 * c_score
        result.skill_pass_rate = pass_rate
        result.skill_c_score = c_score
        result.delta = result.skill_score - baseline_score
    except Exception as e:
        logger.warning(f"  [{skill_id}] task {task_id} error: {e}")
        result.skill_score = -10.0  # 标记为错误

    return result


def run_prune_evaluation(
    n_tasks: int = 20,
    top_skills: int = None,
    dataset: str = "bigcodebench",
    seed: int = 42,
    skip_skills_with_zero_ate: bool = True,
) -> Dict:
    """
    逐条 skill 评估。

    Args:
        n_tasks: 任务数量
        top_skills: 只评估前 N 条 skill (None = 全部)
        dataset: 数据集名称
        seed: 随机种子
        skip_skills_with_zero_ate: 跳过 score_delta=0 的 skill
    """
    random.seed(seed)
    np.random.seed(seed)

    # 加载
    pool = load_training_pool(dataset=dataset, n=n_tasks)
    skill_lib = SkillLibrary(SKILLS_DIR)
    agent = BAgent(workspace="/data/zhuldz/lunwen/agent/b_agent")

    logger.info(f"Prune Evaluation: {n_tasks} tasks, {skill_lib.count} skills, dataset={dataset}")

    base_system_prompt = (
        "You are an expert Python programmer. Write correct, efficient, and well-structured code. "
        "Follow PEP 8 guidelines. Include proper docstrings. Handle edge cases explicitly. "
        "Output only the complete, runnable function."
    )

    # --- Step 1: 跑所有 task 的 baseline (无 skill) ---
    logger.info("=" * 60)
    logger.info("Step 1: Running baseline (no skills) on all tasks")

    task_data = []  # List of {query, gt, ut, baseline_score, baseline_pr, baseline_c}
    for idx, (raw_input, gt, ut) in enumerate(pool):
        query = extract_real_query(raw_input, dataset=dataset)
        try:
            code = agent.generate_code(task=query, system_prompt=base_system_prompt)
            pr = compute_execution_pass_rate(code, ut[:5])
            c_score = call_c_judge(code, gt, base_system_prompt, query, idx * 1000)
            exec_score = pr if pr >= 0 else -3.0
            score = 0.7 * max(0, exec_score) * 10 + 0.3 * c_score
        except Exception as e:
            logger.error(f"Baseline task {idx} error: {e}")
            pr, c_score, score = -3.0, 0.0, -3.0

        task_data.append({
            "idx": idx, "query": query, "gt": gt, "ut": ut[:5],
            "baseline_score": score, "baseline_pr": pr, "baseline_c": c_score,
        })
        logger.info(f"  Task {idx}: baseline_score={score:.1f}, pr={pr:.2f}")

    task_data.sort(key=lambda x: x["baseline_score"])  # 先跑低分任务

    # --- Step 2: 筛选要评估的 skill ---
    all_skills = skill_lib.all  # sorted by id
    if skip_skills_with_zero_ate:
        # 保留 score_delta != 0 的，以及 G0 和最近 20 条
        non_zero = [s for s in all_skills if s.score_delta != 0.0]
        zero = [s for s in all_skills if s.score_delta == 0.0]
        # 从零 ATE 中随机采样 30 条（不能让全部被跳过）
        random.seed(seed)
        sampled_zero = random.sample(zero, min(30, len(zero)))
        skills_to_eval = non_zero + sampled_zero
        logger.info(f"Skills to eval: {len(non_zero)} non-zero ATE + {len(sampled_zero)} sampled zero-ATE = {len(skills_to_eval)}")
    else:
        skills_to_eval = all_skills

    if top_skills and top_skills < len(skills_to_eval):
        # 按 score_delta 排序取 top
        sorted_skills = sorted(skills_to_eval, key=lambda s: s.score_delta, reverse=True)
        skills_to_eval = sorted_skills[:top_skills]
        logger.info(f"Capped to top {top_skills} skills")

    n_skills = len(skills_to_eval)
    logger.info(f"Step 2: Evaluating {n_skills} skills on {n_tasks} tasks")

    # --- Step 3: 逐条 skill × 每个 task 评估 ---
    all_results: List[SkillEvalResult] = []

    for si, skill in enumerate(skills_to_eval):
        for td in task_data:
            result = evaluate_single_skill_on_task(
                agent=agent, skill=skill,
                task_query=td["query"], gt=td["gt"], unit_tests=td["ut"],
                base_system_prompt=base_system_prompt,
                task_id=td["idx"], skill_id=skill.id,
                baseline_score=td["baseline_score"],
                baseline_pass_rate=td["baseline_pr"],
                baseline_c_score=td["baseline_c"],
            )
            all_results.append(result)

        if (si + 1) % 10 == 0:
            logger.info(f"  Progress: {si+1}/{n_skills} skills evaluated")

    # --- Step 4: 聚合统计 ---
    skill_stats = defaultdict(lambda: {
        "deltas": [], "pass_rates_with": [], "pass_rates_without": [],
        "c_scores_with": [], "c_scores_without": [],
        "n_tasks": 0, "n_positive": 0, "n_negative": 0, "n_tie": 0,
    })

    for r in all_results:
        stats = skill_stats[r.skill_id]
        stats["deltas"].append(r.delta)
        stats["pass_rates_with"].append(r.skill_pass_rate)
        stats["pass_rates_without"].append(r.baseline_pass_rate)
        stats["c_scores_with"].append(r.skill_c_score)
        stats["c_scores_without"].append(r.baseline_c_score)
        stats["n_tasks"] += 1
        if r.delta > 0.01:
            stats["n_positive"] += 1
        elif r.delta < -0.01:
            stats["n_negative"] += 1
        else:
            stats["n_tie"] += 1

    # --- Step 5: 排序 & 输出 ---
    skill_summary = []
    for skill_id, stats in skill_stats.items():
        deltas = [d for d in stats["deltas"] if d > -10.0]  # 排除错误标记
        if not deltas:
            continue
        skill_summary.append({
            "skill_id": skill_id,
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)) if len(deltas) > 1 else 0.0,
            "max_delta": float(np.max(deltas)),
            "min_delta": float(np.min(deltas)),
            "n_positive": stats["n_positive"],
            "n_negative": stats["n_negative"],
            "n_tie": stats["n_tie"],
            "n_tasks": stats["n_tasks"],
            "mean_pass_rate_with": float(np.mean([p for p in stats["pass_rates_with"] if p >= 0])),
            "mean_pass_rate_without": float(np.mean([p for p in stats["pass_rates_without"] if p >= 0])),
        })

    skill_summary.sort(key=lambda x: x["mean_delta"], reverse=True)

    # 输出报告
    print("\n" + "=" * 80)
    print("SKILL PRUNE EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nDataset: {dataset}, Tasks: {n_tasks}, Skills evaluated: {len(skill_summary)}")
    print(f"\n{'Skill':>8}  {'Δ':>8}  {'±std':>6}  {'+':>4}  {'-':>4}  {'tie':>4}  {'Tasks':>5}  {'PR_with':>8}  {'PR_wo':>8}")
    print("-" * 80)

    for s in skill_summary[:40]:  # top 40
        print(f"{s['skill_id']:>8}  {s['mean_delta']:>+7.2f}  {s['std_delta']:>5.2f}  "
              f"{s['n_positive']:>4}  {s['n_negative']:>4}  {s['n_tie']:>4}  {s['n_tasks']:>5}  "
              f"{s['mean_pass_rate_with']:>7.2f}  {s['mean_pass_rate_without']:>7.2f}")

    print("-" * 80)

    # 分类统计
    beneficial = [s for s in skill_summary if s["mean_delta"] > 0.1]
    harmful = [s for s in skill_summary if s["mean_delta"] < -0.1]
    neutral = [s for s in skill_summary if -0.1 <= s["mean_delta"] <= 0.1]

    print(f"\nBeneficial (Δ>0.1):  {len(beneficial)} skills  mean Δ={np.mean([s['mean_delta'] for s in beneficial]):+.2f}" if beneficial else "\nBeneficial: 0")
    print(f"Harmful (Δ<-0.1):    {len(harmful)} skills  mean Δ={np.mean([s['mean_delta'] for s in harmful]):+.2f}" if harmful else "Harmful: 0")
    print(f"Neutral (-0.1~0.1):  {len(neutral)} skills")

    # 建议剪枝列表
    prune_candidates = [s for s in skill_summary if s["mean_delta"] < 0]
    print(f"\nPrune candidates (Δ<0): {len(prune_candidates)} skills")
    for s in prune_candidates[:20]:
        print(f"  {s['skill_id']}: Δ={s['mean_delta']:+.2f}  (+{s['n_positive']}/-{s['n_negative']}/{s['n_tasks']} tasks)")

    return {
        "dataset": dataset, "n_tasks": n_tasks, "n_skills_evaluated": len(skill_summary),
        "n_beneficial": len(beneficial), "n_harmful": len(harmful), "n_neutral": len(neutral),
        "prune_candidates": [s["skill_id"] for s in prune_candidates],
        "top_beneficial": [s["skill_id"] for s in skill_summary[:20]],
        "skill_details": skill_summary,
        "per_task_results": [
            {"skill_id": r.skill_id, "task_id": r.task_id, "baseline_score": r.baseline_score,
             "skill_score": r.skill_score, "delta": r.delta}
            for r in all_results if r.delta > -10.0
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Prune Evaluation — per-skill baseline comparison")
    parser.add_argument("--n_tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--top_skills", type=int, default=None, help="Only evaluate top N skills")
    parser.add_argument("--dataset", type=str, default="bigcodebench",
                        choices=["opencode", "humaneval_pro", "mbpp_pro", "bigcodebench"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_skills", action="store_true",
                        help="Evaluate ALL skills including zero-ATE ones")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running")
    args = parser.parse_args()

    if args.dry_run:
        skill_lib = SkillLibrary(SKILLS_DIR)
        pool = load_training_pool(dataset=args.dataset, n=args.n_tasks)
        n_skills = skill_lib.count
        n_non_zero = len([s for s in skill_lib.all if s.score_delta != 0.0])
        n_api_calls = args.n_tasks * (1 + min(args.top_skills or n_skills, n_skills))
        print(f"Dry run plan:")
        print(f"  Tasks: {len(pool)}, Skills to eval: ~{args.top_skills or n_skills}")
        print(f"  Non-zero ATE skills: {n_non_zero}/{n_skills}")
        print(f"  Estimated API calls: {n_api_calls} (baseline + per-skill)")
        print(f"  Estimated time: {n_api_calls * 3 / 60:.0f} min (at ~20 calls/min)")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start = time.time()
    results = run_prune_evaluation(
        n_tasks=args.n_tasks,
        top_skills=args.top_skills,
        dataset=args.dataset,
        seed=args.seed,
        skip_skills_with_zero_ate=not args.all_skills,
    )
    elapsed = time.time() - start

    # 保存
    output_path = args.output or os.path.join(
        OUTPUT_DIR,
        f"prune_eval_{args.dataset}_n{args.n_tasks}_s{len(results['skill_details'])}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "dataset": args.dataset, "n_tasks": args.n_tasks,
                "top_skills": args.top_skills, "seed": args.seed,
                "all_skills": args.all_skills,
            },
            "summary": {k: v for k, v in results.items() if k != "skill_details" and k != "per_task_results"},
            "skill_details": results["skill_details"],
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
