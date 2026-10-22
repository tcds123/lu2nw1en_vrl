"""
Phase 2 — Skill ATE 评分器

消费 custom_reward.py 产出的 readable_trace 日志 (含 skills_used 字段),
计算每条 skill 的 ATE (Average Treatment Effect)，回写到 SkillLibrary。

与 scorer.py 的区别:
  - scorer.py 消费旧格式 trace (exp_ids 在 A 模型输入中)
  - score_skills.py 消费新格式 trace (skills_used 在 group_tests[*] 中)
  - 直接更新 BAgent 的 SkillLibrary (skills.py), 而非 experience_pool.json

使用方式:
    python -m experience.score_skills --trace_dir <path> [--alpha 0.1] [--min_uses 3]
    python -m experience.score_skills --trace_file <readable_trace.log> [--dry-run]
"""

import os
import sys
import re
import json
import math
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

# 确保项目根在 path 中
PROJECT_ROOT = "/data/zhuldz/lunwen"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ================================================================
# 数据结构
# ================================================================
@dataclass
class SkillScore:
    """一条 skill 的评分统计"""
    skill_id: str
    present_scores: List[float] = None   # 使用该 skill 时的得分
    absent_scores: List[float] = None    # 未使用时的得分
    within_problem_diffs: List[float] = None  # 同问题 diff
    problem_ids: List[str] = None        # 涉及的问题
    usage_count: int = 0

    def __post_init__(self):
        self.present_scores = self.present_scores or []
        self.absent_scores = self.absent_scores or []
        self.within_problem_diffs = self.within_problem_diffs or []
        self.problem_ids = self.problem_ids or []

    @property
    def ate(self) -> float:
        """Average Treatment Effect"""
        if self.within_problem_diffs:
            return sum(self.within_problem_diffs) / len(self.within_problem_diffs)
        elif self.present_scores and self.absent_scores:
            return (sum(self.present_scores) / len(self.present_scores) -
                    sum(self.absent_scores) / len(self.absent_scores))
        return 0.0

    @property
    def n_present(self) -> int:
        return len(self.present_scores)

    @property
    def n_absent(self) -> int:
        return len(self.absent_scores)

    @property
    def n_problems(self) -> int:
        return len(set(self.problem_ids))

    @property
    def reliable(self) -> bool:
        """样本量足够"""
        return self.n_present >= 3 and self.n_absent >= 3 and self.n_problems >= 1


# ================================================================
# Trace 解析
# ================================================================
def parse_phase2_trace(trace_path: str) -> List[dict]:
    """
    解析 Phase 2 格式的 readable_trace 日志。

    格式: 每个 block 由 ========== 分隔, 内容是 JSON。
    每个 block 的 group_tests[*] 包含 "skills_used" 字段。
    """
    rollouts = []

    if not os.path.exists(trace_path):
        logger.error(f"Trace file not found: {trace_path}")
        return rollouts

    with open(trace_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按 ====== 分隔每个 trace block
    blocks = re.split(r'={30,}.*?={30,}', content)

    for block in blocks:
        block = block.strip()
        if not block or not block.startswith('{'):
            continue

        # 找到 JSON 起始位置 (有时 block 前有其他字符)
        start = block.find('{')
        if start < 0:
            continue

        try:
            trace = json.loads(block[start:])
        except json.JSONDecodeError:
            continue

        # 提取 group_tests 中的每个 test
        group_tests = trace.get("group_tests", [])
        a_input = trace.get("a_model", {}).get("input", "")

        for test in group_tests:
            skills_used = test.get("skills_used", [])
            if not skills_used:
                continue

            score = test.get("final_case_score", 0.0)
            test_type = test.get("test_type", "unknown")

            # 提取 problem key (用于 within-problem diff)
            problem_key = _extract_problem_key(a_input, test_type)

            rollouts.append({
                "skills_used": skills_used,
                "score": float(score),
                "problem_key": problem_key,
                "test_type": test_type,
            })

    logger.info(f"Parsed {len(rollouts)} rollouts from {trace_path} "
                f"(avg {len(rollouts)/max(1,len(set(r['problem_key'] for r in rollouts))):.1f} per problem)")

    return rollouts


def parse_all_traces(trace_dir: str) -> List[dict]:
    """解析 trace_dir 下所有 readable_trace 日志"""
    all_rollouts = []
    trace_dir = os.path.abspath(trace_dir)

    if not os.path.isdir(trace_dir):
        # 如果是文件, 直接解析
        return parse_phase2_trace(trace_dir)

    for fname in sorted(os.listdir(trace_dir)):
        if fname.startswith("readable_trace") and fname.endswith(".log"):
            fpath = os.path.join(trace_dir, fname)
            rollouts = parse_phase2_trace(fpath)
            all_rollouts.extend(rollouts)

    logger.info(f"Total: {len(all_rollouts)} rollouts from {trace_dir}")
    return all_rollouts


def _extract_problem_key(a_input: str, test_type: str) -> str:
    """从 A 模型输入 + test_type 中提取问题标识"""
    # 尝试从 a_input 提取函数名
    match = re.search(r'def\s+(\w+)', a_input)
    if match:
        return f"{match.group(1)}_{test_type.replace(' ', '_')[:30]}"

    # fallback: hash 前 200 字符
    return f"p{hash(a_input[:200] + test_type) % 100000}"


# ================================================================
# ATE 估计
# ================================================================
def estimate_skill_effects(
    rollouts: List[dict],
    min_uses: int = 3,
    min_problems: int = 1,
) -> Dict[str, SkillScore]:
    """
    估计每条 skill 的因果效应。
    Within-problem difference-in-differences。
    """
    # 1. 按问题分组
    problem_groups = defaultdict(list)
    for r in rollouts:
        problem_groups[r["problem_key"]].append(r)

    logger.info(f"{len(problem_groups)} problems, {len(rollouts)} total rollouts")

    # 2. 收集所有 skill IDs
    all_skill_ids = set()
    for r in rollouts:
        all_skill_ids.update(r["skills_used"])
    logger.info(f"{len(all_skill_ids)} unique skill IDs found")

    # 3. 计算每条 skill 的 ATE
    scores: Dict[str, SkillScore] = {}

    for skill_id in all_skill_ids:
        score = SkillScore(skill_id=skill_id)

        for problem_key, problem_rollouts in problem_groups.items():
            present = [r["score"] for r in problem_rollouts if skill_id in r["skills_used"]]
            absent = [r["score"] for r in problem_rollouts if skill_id not in r["skills_used"]]

            score.present_scores.extend(present)
            score.absent_scores.extend(absent)

            if present and absent:
                diff = sum(present)/len(present) - sum(absent)/len(absent)
                score.within_problem_diffs.append(diff)
                score.problem_ids.append(problem_key)

        score.usage_count = len(score.present_scores)

        # 过滤: 至少 min_uses 次
        if score.n_present >= min_uses and score.n_problems >= min_problems:
            scores[skill_id] = score

    # 统计
    positive = sum(1 for s in scores.values() if s.ate > 0)
    negative = sum(1 for s in scores.values() if s.ate < 0)
    reliable = sum(1 for s in scores.values() if s.reliable)

    logger.info(f"Estimated effects for {len(scores)}/{len(all_skill_ids)} skills")
    logger.info(f"  Positive: {positive}, Negative: {negative}, Reliable (n>=3): {reliable}")
    logger.info(f"  Mean ATE: {sum(s.ate for s in scores.values())/max(1,len(scores)):.4f}")
    logger.info(f"  Top positive: {_top_positive_str(scores, 3)}")
    logger.info(f"  Top negative: {_top_negative_str(scores, 3)}")

    return scores


def _top_positive_str(scores, k=3):
    pos = sorted([s for s in scores.values() if s.ate > 0], key=lambda x: x.ate, reverse=True)[:k]
    return ", ".join(f"{s.skill_id}:{s.ate:.3f}" for s in pos)


def _top_negative_str(scores, k=3):
    neg = sorted([s for s in scores.values() if s.ate < 0], key=lambda x: x.ate)[:k]
    return ", ".join(f"{s.skill_id}:{s.ate:.3f}" for s in neg)


# ================================================================
# 回写到 SkillLibrary
# ================================================================
def update_skill_library(
    scores: Dict[str, SkillScore],
    skills_dir: str = "/data/zhuldz/lunwen/agent/b_agent/skills",
    alpha: float = 0.1,
    prune_threshold: float = -0.5,
    prune_min_uses: int = 5,
    dry_run: bool = False,
) -> dict:
    """
    将 ATE 评分回写到 SkillLibrary。
    - EMA 更新 score_delta
    - 剪枝: 删除 ATE < prune_threshold 且使用次数 >= prune_min_uses 的 skill
    """
    from agent.b_agent.skills import SkillLibrary

    lib = SkillLibrary(skills_dir)

    stats = {"updated": 0, "pruned": 0, "skipped": 0, "total_skills": lib.count}

    for skill_id, skill_score in scores.items():
        skill = lib.get(skill_id)
        if skill is None:
            stats["skipped"] += 1
            continue

        # 旧 score_delta
        old_score = skill.score_delta
        # EMA 更新
        new_score = (1 - alpha) * old_score + alpha * skill_score.ate
        skill.score_delta = new_score

        if not dry_run:
            lib.update_score(skill_id, skill_score.ate, alpha=alpha)

        stats["updated"] += 1

    # 剪枝: 删除负效应 skill
    if not dry_run:
        n_pruned = lib.prune_negative(min_usage=prune_min_uses, threshold=prune_threshold)
        stats["pruned"] = n_pruned

    logger.info(f"Update complete: {stats}")

    return stats


# ================================================================
# CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 2 Skill ATE Scorer")
    parser.add_argument("--trace_file", type=str, default=None,
                        help="单条 readable_trace 日志路径")
    parser.add_argument("--trace_dir", type=str,
                        default="/data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout",
                        help="trace 日志目录")
    parser.add_argument("--skills_dir", type=str,
                        default="/data/zhuldz/lunwen/agent/b_agent/skills",
                        help="SkillLibrary 目录")
    parser.add_argument("--alpha", type=float, default=0.1, help="EMA 更新速率")
    parser.add_argument("--min_uses", type=int, default=3, help="最少使用次数")
    parser.add_argument("--prune", action="store_true", help="启用剪枝")
    parser.add_argument("--prune_threshold", type=float, default=-0.3,
                        help="ATE 低于此值触发剪枝")
    parser.add_argument("--dry_run", action="store_true", help="仅打印, 不实际修改文件")
    parser.add_argument("--top_k", type=int, default=10, help="打印多少条 top skill")
    args = parser.parse_args()

    # 1. 解析 trace
    trace_source = args.trace_file or args.trace_dir

    if os.path.isfile(trace_source):
        rollouts = parse_phase2_trace(trace_source)
    else:
        rollouts = parse_all_traces(trace_source)

    if not rollouts:
        logger.error("No rollouts found. Waiting for more training data.")
        return

    # 2. 估计 ATE
    scores = estimate_skill_effects(rollouts, min_uses=args.min_uses)

    # 3. Show top skills
    sorted_scores = sorted(scores.values(), key=lambda s: s.ate, reverse=True)
    print(f"\n{'='*60}")
    print(f"Top {args.top_k} Positive Skills:")
    print(f"{'='*60}")
    for s in sorted_scores[:args.top_k]:
        if s.ate > 0:
            print(f"  [{s.skill_id}] ATE={s.ate:+.4f}  uses={s.n_present}  "
                  f"problems={s.n_problems}  reliable={s.reliable}")

    print(f"\n{'='*60}")
    print(f"Top {args.top_k} Negative Skills (potential prune candidates):")
    print(f"{'='*60}")
    neg_sorted = sorted(scores.values(), key=lambda s: s.ate)
    for s in neg_sorted[:args.top_k]:
        if s.ate < 0:
            print(f"  [{s.skill_id}] ATE={s.ate:+.4f}  uses={s.n_present}  "
                  f"problems={s.n_problems}")

    # 4. 更新 SkillLibrary
    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"Updating SkillLibrary...")
        print(f"{'='*60}")
        stats = update_skill_library(
            scores, skills_dir=args.skills_dir,
            alpha=args.alpha, prune_threshold=args.prune_threshold,
            dry_run=False,
        )
        print(f"Result: {stats}")
    else:
        print(f"\n[Dry run] No files modified. Use --no-dry_run to update.")


if __name__ == "__main__":
    main()
