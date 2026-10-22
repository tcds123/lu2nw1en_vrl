"""
Skill 闭环进化流水线 (v4) — 修复 Trace2Skill Gap G2

Automates the closed loop:
  Trace logs → Parse skills_used → Estimate ATE → Update skill scores → Prune negative skills

This should be run periodically (e.g., after each GRPO training session produces new traces).

使用方式:
    # 从最新 trace 自动评分并更新
    python -m experience.skill_loop

    # 指定 trace 文件
    python -m experience.skill_loop --trace_file <path>

    # 只报告，不修改
    python -m experience.skill_loop --dry_run

    # 从所有历史 trace 累计评分
    python -m experience.skill_loop --trace_dir <dir> --all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = "/data/zhuldz/lunwen"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRACE_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout"
DEFAULT_SKILLS_DIR = "/data/zhuldz/lunwen/agent/b_agent/skills"


class SkillLoop:
    """Skill 闭环进化管理器"""

    def __init__(
        self,
        trace_dir: str = DEFAULT_TRACE_DIR,
        skills_dir: str = DEFAULT_SKILLS_DIR,
    ):
        self.trace_dir = Path(trace_dir)
        self.skills_dir = Path(skills_dir)

    def find_latest_trace(self) -> Optional[str]:
        """找到最新的 readable_trace 日志"""
        if not self.trace_dir.is_dir():
            logger.error(f"Trace directory not found: {self.trace_dir}")
            return None

        candidates = []
        for f in sorted(self.trace_dir.iterdir()):
            if f.name.startswith("readable_trace") and f.name.endswith(".log"):
                candidates.append(f)

        if not candidates:
            logger.error("No trace files found")
            return None

        # 按修改时间排序，取最新
        candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest = str(candidates[0])
        logger.info(f"Latest trace: {latest} ({candidates[0].stat().st_size / 1024 / 1024:.1f} MB)")
        return latest

    def check_trace_has_skills(self, trace_path: str) -> bool:
        """检查 trace 是否包含 skills_used 字段"""
        import json, re
        with open(trace_path, 'r') as f:
            content = f.read()

        blocks = re.split(r'={30,}.*?={30,}', content)
        checked = 0
        has_skills = 0

        for block in blocks[:20]:  # 只检查前20个block
            block = block.strip()
            if not block or not block.startswith('{'):
                continue
            start = block.find('{')
            if start < 0:
                continue
            try:
                trace = json.loads(block[start:])
            except json.JSONDecodeError:
                continue

            for t in trace.get("group_tests", []):
                checked += 1
                if t.get("skills_used"):
                    has_skills += 1

        if checked == 0:
            logger.warning("No rollouts found in trace")
            return False

        ratio = has_skills / checked if checked > 0 else 0
        logger.info(f"Trace skills_used coverage: {has_skills}/{checked} ({ratio:.0%})")

        if ratio < 0.1:
            logger.warning(
                "⚠️  Trace has very low skills_used coverage. "
                "This likely means: \n"
                "  1. The trace was produced by old custom_reward.py (pre-Phase-2)\n"
                "  2. OR the BAgentInterface is not returning skills_used correctly\n\n"
                "Solution: restart GRPO training with the current custom_reward.py"
            )
            return False

        return True

    def run(
        self,
        trace_path: str = None,
        dry_run: bool = False,
        alpha: float = 0.1,
    ) -> dict:
        """
        执行一次完整的闭环更新。

        Returns:
            stats dict: {"updated": int, "pruned": int, "rollouts": int, "skills_with_data": int}
        """
        # 1. 找 trace
        if trace_path is None:
            trace_path = self.find_latest_trace()
        if trace_path is None:
            return {"error": "No trace found"}

        # 2. 检查 skills_used 覆盖
        if not self.check_trace_has_skills(trace_path):
            return {"error": "Trace has no skills_used data", "action": "restart_training"}

        # 3. 解析 + 估计 ATE
        from experience.score_skills import parse_phase2_trace, estimate_skill_effects, update_skill_library

        logger.info(f"Parsing trace: {trace_path}")
        rollouts = parse_phase2_trace(trace_path)
        logger.info(f"Parsed {len(rollouts)} rollouts")

        if len(rollouts) < 50:
            logger.warning(f"Only {len(rollouts)} rollouts — need more data for reliable ATE estimation")
            # 仍然运行，但 alpha 降低
            effective_alpha = alpha * 0.3
        else:
            effective_alpha = alpha

        # 4. 估计因果效应
        scores = estimate_skill_effects(rollouts, min_uses=3, min_problems=1)

        if not scores:
            logger.info("No skills met the minimum data threshold. Need more training steps.")
            return {"updated": 0, "rollouts": len(rollouts), "skills_with_data": 0}

        # 5. 更新 SkillLibrary
        stats = update_skill_library(
            scores,
            skills_dir=str(self.skills_dir),
            alpha=effective_alpha,
            dry_run=dry_run,
        )

        # 6. 统计 summary
        positive = sum(1 for s in scores.values() if s.ate > 0)
        negative = sum(1 for s in scores.values() if s.ate < 0)
        top_pos = sorted([s for s in scores.values() if s.ate > 0], key=lambda x: x.ate, reverse=True)[:5]
        top_neg = sorted([s for s in scores.values() if s.ate < 0], key=lambda x: x.ate)[:5]

        logger.info(f"\n{'='*60}")
        logger.info(f"Skill Loop Complete")
        logger.info(f"{'='*60}")
        logger.info(f"  Rollouts processed: {len(rollouts)}")
        logger.info(f"  Skills with data:   {len(scores)}")
        logger.info(f"  Positive effect:    {positive} ({100*positive/max(1,len(scores)):.0f}%)")
        logger.info(f"  Negative effect:    {negative} ({100*negative/max(1,len(scores)):.0f}%)")
        logger.info(f"  Skills updated:     {stats.get('updated', 0)}")
        logger.info(f"  Skills pruned:      {stats.get('pruned', 0)}")

        if top_pos:
            logger.info(f"\n  Top 5 positive skills:")
            for s in top_pos:
                logger.info(f"    {s.skill_id}: ATE={s.ate:+.4f}, uses={s.n_present}, problems={s.n_problems}")

        if top_neg:
            logger.info(f"\n  Top 5 negative skills (prune candidates):")
            for s in top_neg:
                logger.info(f"    {s.skill_id}: ATE={s.ate:+.4f}, uses={s.n_present}")

        return {
            "updated": stats.get('updated', 0),
            "pruned": stats.get('pruned', 0),
            "rollouts": len(rollouts),
            "skills_with_data": len(scores),
            "positive": positive,
            "negative": negative,
        }


def main():
    parser = argparse.ArgumentParser(description="Skill closed-loop evolution pipeline (v4)")
    parser.add_argument("--trace_file", type=str, default=None,
                        help="Specific trace file (default: auto-detect latest)")
    parser.add_argument("--trace_dir", type=str, default=DEFAULT_TRACE_DIR,
                        help="Trace directory")
    parser.add_argument("--skills_dir", type=str, default=DEFAULT_SKILLS_DIR,
                        help="Skills directory")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="EMA update rate for skill score_delta")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report only, don't modify files")
    parser.add_argument("--force", action="store_true",
                        help="Run even if trace has no skills_used field")
    args = parser.parse_args()

    loop = SkillLoop(trace_dir=args.trace_dir, skills_dir=args.skills_dir)

    if args.trace_file:
        trace = args.trace_file
    else:
        trace = loop.find_latest_trace()

    if trace is None:
        print("❌ No trace found. Run GRPO training first.")
        return

    result = loop.run(trace_path=trace, dry_run=args.dry_run, alpha=args.alpha)

    if result.get("error"):
        print(f"\n❌ {result['error']}")
        if result.get("action") == "restart_training":
            print("→ Action required: restart GRPO training to produce traces with skills_used data")
    elif args.dry_run:
        print(f"\n[Dry run] Would update {result['updated']} skills")
        print("Remove --dry_run to actually modify files.")
    else:
        print(f"\n✅ Updated {result['updated']} skills, pruned {result.get('pruned', 0)}")


if __name__ == "__main__":
    main()
