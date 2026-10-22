"""
经验因果效应估计器 — Improvement A: Empirical Experience Scoring (v3: 跨题稳定性)

从 GRPO trace 日志中估计每条经验的因果效应 (ATE):
  ATE(g) = E[R | g present] - E[R | g absent]

v3 新增:
  - Cross-problem stability: 要求经验在多个不同问题上效果一致
  - Leave-one-problem-out (LOPO): 验证 ATE 符号的稳定性
  - 自动剪枝不稳定经验
"""

import os
import sys
import json
import re
import math
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experience.cache import ExperienceCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperienceScore:
    """经验的评分统计 (v3: 增加跨题稳定性指标)"""
    exp_id: str
    present_scores: List[float]     # 注入时的 score
    absent_scores: List[float]      # 未注入时的 score
    within_problem_diffs: List[float]  # 同问题内 score 差异
    problem_ids: List[str]          # 涉及的问题 ID (v3 新增)

    @property
    def ate(self) -> float:
        """Average Treatment Effect (within-problem mean diff)"""
        if self.within_problem_diffs:
            return sum(self.within_problem_diffs) / len(self.within_problem_diffs)
        elif self.present_scores and self.absent_scores:
            return (sum(self.present_scores) / len(self.present_scores) -
                    sum(self.absent_scores) / len(self.absent_scores))
        return 0.0

    @property
    def std_error(self) -> float:
        """ATE 的标准误"""
        if len(self.within_problem_diffs) < 2:
            return float('inf')
        mean = self.ate
        variance = sum((d - mean) ** 2 for d in self.within_problem_diffs) / (len(self.within_problem_diffs) - 1)
        return math.sqrt(variance / len(self.within_problem_diffs))

    @property
    def t_statistic(self) -> float:
        """t 统计量: ATE / SE"""
        se = self.std_error
        if se == 0:
            return float('inf') if self.ate != 0 else 0.0
        return self.ate / se

    @property
    def num_problems(self) -> int:
        """涉及的唯一问题数"""
        return len(set(self.problem_ids))

    @property
    def cross_problem_stability(self) -> float:
        """
        跨题稳定性指标 (v3 新增)。

        定义为 within-problem diffs 的变异系数 (CV):
          stability = 1 - CV = 1 - (std(diffs) / |mean(diffs)|)

        值域: 0 (极不稳定) ~ 1 (完全稳定)
        只在至少有 2 个不同问题时有意义。
        """
        if len(self.within_problem_diffs) < 2:
            return 0.0
        mean = self.ate
        if abs(mean) < 1e-8:
            return 0.0
        std = math.sqrt(
            sum((d - mean) ** 2 for d in self.within_problem_diffs) /
            (len(self.within_problem_diffs) - 1)
        )
        cv = std / abs(mean)
        return max(0.0, 1.0 - min(cv, 1.0))

    @property
    def effect_direction_consistency(self) -> float:
        """
        效应方向一致性 (v3 新增)。

        在同问题的 diff 中，有多少比例和 ATE 同方向。
        如果 ATE > 0 但只有 50% 的问题上效果为正 → 说明不一致。
        """
        if not self.within_problem_diffs:
            return 0.0
        consistent = sum(
            1 for d in self.within_problem_diffs
            if (d > 0 and self.ate > 0) or (d < 0 and self.ate < 0)
        )
        return consistent / len(self.within_problem_diffs)

    @property
    def significant(self, alpha: float = 0.1) -> bool:
        """是否统计显著 (v3: 需要同时满足跨题稳定性)"""
        if len(self.within_problem_diffs) < 3:
            return False
        if self.num_problems < 2:
            return False
        return abs(self.t_statistic) > 1.645

    @property
    def generalizable(self) -> bool:
        """
        是否可泛化 (v3 新增): 同时满足:
        1. 统计显著 (|t| > 1.645)
        2. 跨题稳定 (stability > 0.3)
        3. 方向一致 (consistency > 0.6)
        4. 足够多的问题 (n_problems >= 2)
        """
        return (
            self.significant and
            self.cross_problem_stability > 0.3 and
            self.effect_direction_consistency > 0.6 and
            self.num_problems >= 2
        )


class ExperienceScorer:
    """从 trace 日志中估计经验的因果效应 (v3: 跨题稳定性)"""

    def __init__(self, cache_path: str = "/data/zhuldz/lunwen/experience/experience_pool.json"):
        self.cache = ExperienceCache(cache_path)

    # ============================================================
    # Trace 解析
    # ============================================================
    def parse_trace_with_exp_ids(self, trace_path: str) -> List[dict]:
        """解析 trace 日志, 同时提取每个 rollout 中包含的经验 ID"""
        rollouts = []

        with open(trace_path, 'r') as f:
            content = f.read()

        blocks = re.split(r'={30,}.*?={30,}', content)
        for block in blocks:
            block = block.strip()
            if not block or not block.startswith('{'):
                continue

            try:
                trace = json.loads(block)
            except json.JSONDecodeError:
                start = block.find('{')
                if start >= 0:
                    try:
                        trace = json.loads(block[start:])
                    except:
                        continue
                else:
                    continue

            entry = self._extract_with_exp_ids(trace)
            if entry:
                rollouts.append(entry)

        logger.info(f"Parsed {len(rollouts)} rollouts with experience IDs")
        return rollouts

    def _extract_with_exp_ids(self, trace: dict) -> Optional[dict]:
        """从单条 trace 中提取信息, 包括经验 ID"""
        try:
            a_model = trace.get("a_model", {})
            a_input = a_model.get("input", "")
            system_prompt = a_model.get("output", "")

            exp_ids = []
            for line in str(a_input).split('\n'):
                match = re.match(r'\s*\[(G\d+)\]', line)
                if match:
                    exp_ids.append(match.group(1))

            group_tests = trace.get("group_tests", [])
            if not group_tests:
                return None

            test = group_tests[0]
            final_score = test.get("final_case_score", 0.0)

            problem_key = self._extract_problem_key(str(a_input))

            return {
                "exp_ids": exp_ids,
                "score": final_score,
                "problem_key": problem_key,
                "system_prompt_len": len(str(system_prompt)),
            }
        except Exception:
            return None

    @staticmethod
    def _extract_problem_key(a_input: str) -> str:
        """从 A 模型输入中提取问题标识"""
        match = re.search(r'Ground Truth.*?```(?:python)?\s*(.*?)\s*```', a_input, re.DOTALL)
        if match:
            code = match.group(1)[:300].strip()
            sig_match = re.search(r'def\s+(\w+)', code)
            if sig_match:
                return f"fn_{sig_match.group(1)}_{hash(code) % 10000}"
            return f"code_{hash(code) % 100000}"

        match = re.search(r'User Query:\s*(.*?)(?:\n|Ground Truth)', a_input, re.DOTALL)
        if match:
            query = match.group(1).strip()[:200]
            return f"q_{hash(query) % 100000}"

        return f"unknown_{hash(a_input[:500]) % 100000}"

    # ============================================================
    # 因果效应估计 (v3: 增加跨题稳定性)
    # ============================================================
    def estimate_causal_effects(
        self,
        rollouts: List[dict],
        min_samples_per_exp: int = 3,
        min_problems_per_exp: int = 2,
    ) -> Dict[str, ExperienceScore]:
        """
        估计每条经验的因果效应 (v3: 跨题稳定性)。

        Method: Within-problem difference-in-differences
        - 对每个问题 p, 计算: diff(g, p) = mean(score | g, p) - mean(score | ¬g, p)
        - ATE(g) = mean_p diff(g, p)
        - 额外计算: cross_problem_stability, effect_direction_consistency
        """
        # 1. 按问题分组
        problem_groups = defaultdict(list)
        for r in rollouts:
            problem_groups[r["problem_key"]].append(r)

        logger.info(f"Grouped into {len(problem_groups)} problems, "
                    f"avg {len(rollouts)/max(1,len(problem_groups)):.1f} rollouts/problem")

        # 2. 收集所有经验 ID
        all_exp_ids = set()
        for r in rollouts:
            all_exp_ids.update(r["exp_ids"])
        logger.info(f"Found {len(all_exp_ids)} unique experience IDs in rollouts")

        # 3. 对每条经验, 计算 within-problem score difference
        scores: Dict[str, ExperienceScore] = {}

        for exp_id in all_exp_ids:
            score = ExperienceScore(
                exp_id=exp_id,
                present_scores=[],
                absent_scores=[],
                within_problem_diffs=[],
                problem_ids=[],
            )

            for problem_key, problem_rollouts in problem_groups.items():
                present = [r["score"] for r in problem_rollouts if exp_id in r["exp_ids"]]
                absent = [r["score"] for r in problem_rollouts if exp_id not in r["exp_ids"]]

                score.present_scores.extend(present)
                score.absent_scores.extend(absent)

                if present and absent:
                    diff = sum(present) / len(present) - sum(absent) / len(absent)
                    score.within_problem_diffs.append(diff)
                    score.problem_ids.append(problem_key)

            # 只保留有足够跨题样本的经验
            if (len(score.within_problem_diffs) >= min_problems_per_exp and
                len(score.present_scores) + len(score.absent_scores) >= min_samples_per_exp):
                scores[exp_id] = score

        logger.info(f"Estimated effects for {len(scores)}/{len(all_exp_ids)} experiences "
                    f"(min {min_problems_per_exp} problems, {min_samples_per_exp} samples)")
        logger.info(f"  Mean ATE: {sum(s.ate for s in scores.values())/max(1,len(scores)):.4f}")
        logger.info(f"  Significant (p<0.1, n_problems>=2): {sum(1 for s in scores.values() if s.significant)}")
        logger.info(f"  Generalizable (all checks): {sum(1 for s in scores.values() if s.generalizable)}")
        logger.info(f"  Positive effect: {sum(1 for s in scores.values() if s.ate > 0)}")
        logger.info(f"  Negative effect: {sum(1 for s in scores.values() if s.ate < 0)}")

        return scores

    # ============================================================
    # Leave-One-Problem-Out 验证 (v3 新增)
    # ============================================================
    def leave_one_problem_out_validation(
        self,
        scores: Dict[str, ExperienceScore],
    ) -> Dict[str, dict]:
        """
        LOPO 验证: 对每条经验，逐一去掉一个问题后重新计算 ATE。
        用于检验 ATE 是否被某个单一问题过度影响。

        Returns:
            {exp_id: {
                "ate_full": float,
                "ate_lopo_min": float,
                "ate_lopo_max": float,
                "sign_flips": int,  # ATE 符号翻转次数
                "is_stable": bool,   # 符号始终一致
            }}
        """
        results = {}
        for exp_id, score in scores.items():
            diffs = score.within_problem_diffs
            if len(diffs) < 3:
                continue

            ate_full = sum(diffs) / len(diffs)
            lopo_ates = []
            for i in range(len(diffs)):
                lopo_diffs = diffs[:i] + diffs[i+1:]
                lopo_ates.append(sum(lopo_diffs) / len(lopo_diffs))

            sign_full = 1 if ate_full > 0 else -1
            sign_flips = sum(1 for a in lopo_ates if (a > 0) != (ate_full > 0))

            results[exp_id] = {
                "ate_full": ate_full,
                "ate_lopo_min": min(lopo_ates),
                "ate_lopo_max": max(lopo_ates),
                "sign_flips": sign_flips,
                "is_stable": sign_flips == 0,
            }

        return results

    # ============================================================
    # 评分与更新 (v3: 跨题稳定性过滤器)
    # ============================================================
    def score_and_update(
        self,
        trace_path: str,
        alpha: float = 0.1,
        min_rollouts: int = 100,
        min_cross_problems: int = 2,
        require_generalizable: bool = True,
    ) -> Dict[str, ExperienceScore]:
        """
        完整流程: 解析 trace → 估计因果效应 → 更新经验池 (v3)

        Args:
            trace_path: readable_trace 日志路径
            alpha: EMA 更新速率
            min_rollouts: 最少 rollout 数量
            min_cross_problems: 最少跨问题数 (v3 新增)
            require_generalizable: 是否只更新通过泛化检验的经验 (v3 新增)

        Returns:
            {exp_id: ExperienceScore} 评分结果
        """
        # 1. 解析
        logger.info(f"Parsing trace: {trace_path}")
        rollouts = self.parse_trace_with_exp_ids(trace_path)

        if len(rollouts) < min_rollouts:
            logger.warning(f"Only {len(rollouts)} rollouts < {min_rollouts} minimum, skipping")
            return {}

        # 2. 估计因果效应
        scores = self.estimate_causal_effects(
            rollouts,
            min_problems_per_exp=min_cross_problems,
        )

        # 3. LOPO 验证 (v3)
        lopo_results = self.leave_one_problem_out_validation(scores)
        lopo_stable = {eid for eid, r in lopo_results.items() if r["is_stable"]}
        logger.info(f"LOPO validation: {len(lopo_stable)}/{len(lopo_results)} experiences sign-stable")

        # 4. 更新经验池 (v3: 选择性更新)
        updated = 0
        skipped_unstable = 0
        skipped_specific = 0

        for exp_id, score in scores.items():
            if len(score.within_problem_diffs) < min_cross_problems:
                skipped_specific += 1
                continue

            if require_generalizable and not score.generalizable:
                skipped_unstable += 1
                logger.debug(f"  Skipping {exp_id}: stability={score.cross_problem_stability:.2f}, "
                            f"consistency={score.effect_direction_consistency:.2f}, "
                            f"n_problems={score.num_problems}")
                continue

            # LOPO 检查: 如果符号不稳定，降低 alpha (更保守)
            effective_alpha = alpha
            if exp_id not in lopo_stable:
                effective_alpha = alpha * 0.3  # 符号不稳定时更保守地更新

            self.cache.update_score_ema(exp_id, score.ate, effective_alpha)
            updated += 1

        logger.info(f"Updated {updated}/{len(scores)} experiences in pool")
        logger.info(f"  Skipped: {skipped_unstable} unstable, {skipped_specific} too-few-problems")
        logger.info(f"Pool stats: {self.cache.stats()}")

        # 5. 剪枝
        pruned_neg = self.cache.prune_negative(min_usage=3, threshold=-0.5)
        if pruned_neg:
            logger.info(f"Pruned {pruned_neg} negative-effect experiences")

        pruned_unstable = self.cache.prune_unstable(min_usage=4)
        if pruned_unstable:
            logger.info(f"Pruned {pruned_unstable} unstable (near-zero effect) experiences")

        # 6. 语义聚类去重
        merged = self.cache.cluster_and_dedup(similarity_threshold=0.55)
        if merged:
            logger.info(f"Semantic clustering merged {merged} redundant experiences")

        return scores

    # ============================================================
    # 报告 (v3: 增加跨题稳定性列)
    # ============================================================
    def print_report(self, scores: Dict[str, ExperienceScore], top_n: int = 20):
        """打印经验评分报告 (v3 增强版)"""
        if not scores:
            print("No scores available.")
            return

        ranked = sorted(scores.items(), key=lambda x: x[1].ate, reverse=True)

        print(f"\n{'='*100}")
        print(f"EXPERIENCE SCORING REPORT (v3: Cross-Problem Stability)")
        print(f"{'='*100}")
        header = (f"{'Rank':<6} {'ID':<8} {'ATE':>8} {'SE':>8} {'t':>7} "
                  f"{'Stab':>6} {'Cons':>6} {'#P':>4} {'Gen?':>5} {'Sig':>4}")
        print(header)
        print(f"{'-'*100}")

        for rank, (exp_id, score) in enumerate(ranked[:top_n], 1):
            exp = self.cache.get(exp_id)
            text = exp.text[:60] if exp else "(not in pool)"
            sig = "***" if score.significant and score.ate > 0 else \
                  " **" if score.significant and score.ate < 0 else \
                  "  *" if abs(score.t_statistic) > 1.0 else "   "
            gen = " ✓" if score.generalizable else "  "
            print(f"{rank:<6} {exp_id:<8} {score.ate:>+8.4f} {score.std_error:>8.4f} "
                  f"{score.t_statistic:>7.2f} {score.cross_problem_stability:>6.3f} "
                  f"{score.effect_direction_consistency:>6.3f} {score.num_problems:>4} "
                  f"{gen:>5} {sig:>4}")
            if exp:
                print(f"       └─ {text}...")

        print(f"{'='*100}")

        # Distribution summary
        ates = [s.ate for s in scores.values()]
        print(f"\nSummary: {len(scores)} experiences scored")
        print(f"  ATE range: [{min(ates):+.4f}, {max(ates):+.4f}]")
        print(f"  ATE mean:  {sum(ates)/len(ates):+.4f}")
        print(f"  Positive:  {sum(1 for a in ates if a > 0)} ({100*sum(1 for a in ates if a > 0)/len(ates):.0f}%)")
        print(f"  Negative:  {sum(1 for a in ates if a < 0)} ({100*sum(1 for a in ates if a < 0)/len(ates):.0f}%)")
        print(f"  Significant (|t|>1.645, n_problems>=2): {sum(1 for s in scores.values() if s.significant)}")
        print(f"  Generalizable (all checks passed): {sum(1 for s in scores.values() if s.generalizable)}")
        print(f"  Cross-problem stable (stability>0.3): "
              f"{sum(1 for s in scores.values() if s.cross_problem_stability > 0.3)}")
        print(f"  Direction consistent (consistency>0.6): "
              f"{sum(1 for s in scores.values() if s.effect_direction_consistency > 0.6)}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate causal effects of prompt engineering experiences from GRPO traces (v3)")
    parser.add_argument("--trace_path", type=str, required=True,
                        help="Path to readable_trace log")
    parser.add_argument("--cache_path", type=str,
                        default="/data/zhuldz/lunwen/experience/experience_pool.json",
                        help="Experience pool JSON path")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="EMA update rate for score_delta")
    parser.add_argument("--min_rollouts", type=int, default=100,
                        help="Minimum rollouts required")
    parser.add_argument("--min_cross_problems", type=int, default=2,
                        help="Minimum cross-problem count for an experience to be considered (v3)")
    parser.add_argument("--no_generalizable_filter", action="store_true",
                        help="Disable generalizability filter (update all experiences)")
    parser.add_argument("--report_only", action="store_true",
                        help="Only print report, don't update pool")
    args = parser.parse_args()

    scorer = ExperienceScorer(cache_path=args.cache_path)

    # 解析并估计
    rollouts = scorer.parse_trace_with_exp_ids(args.trace_path)
    if len(rollouts) < args.min_rollouts:
        print(f"Insufficient rollouts: {len(rollouts)} < {args.min_rollouts}")
        sys.exit(1)

    scores = scorer.estimate_causal_effects(
        rollouts,
        min_problems_per_exp=args.min_cross_problems,
    )

    if args.report_only:
        scorer.print_report(scores)
    else:
        scorer.print_report(scores)
        # 更新经验池
        updated = 0
        skipped = 0
        for exp_id, score in scores.items():
            if len(score.within_problem_diffs) >= args.min_cross_problems:
                if args.no_generalizable_filter or score.generalizable:
                    scorer.cache.update_score_ema(exp_id, score.ate, args.alpha)
                    updated += 1
                else:
                    skipped += 1

        logger.info(f"Updated {updated}/{len(scores)} experiences ({skipped} skipped by generalizability filter)")
        pruned = scorer.cache.prune_negative(min_usage=3)
        if pruned:
            logger.info(f"Pruned {pruned} negative-effect experiences")

        merged = scorer.cache.cluster_and_dedup()
        if merged:
            logger.info(f"Semantic clustering merged {merged} redundant experiences")
