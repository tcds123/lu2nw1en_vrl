"""
Skill 批量增强器 (v4) — 修复 Trace2Skill Gap G4: 将 thin skills 升级为结构化指南

将现有的 401 条一句话 skill 通过 LLM 批量升级为结构化格式:
  WHEN / WHAT / HOW (Example) / WHY

使用方式:
    # 预览模式 (不实际修改文件):
    python -m experience.enhance_skills --dry_run

    # 批量增强 (分批处理, 每批 10 条):
    python -m experience.enhance_skills --batch_size 10 --start_from 0

    # 只增强特定范围:
    python -m experience.enhance_skills --start_from 1 --end_at 20
"""

import os
import sys
import re
import json
import time
import argparse
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = "/data/zhuldz/lunwen"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experience.templates import SKILL_ENHANCEMENT_TEMPLATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SKILLS_DIR = "/data/zhuldz/lunwen/agent/b_agent/skills"
ENHANCED_BACKUP_DIR = "/data/zhuldz/lunwen/agent/b_agent/skills_backup_20260703"

# LLM 配置 (DeepSeek-V3.2, 用于 skill 增强 — 非代码生成任务的 LLM)
LLM_MODEL = "/model/DeepSeek-V3.2"
LLM_BASE_URL = "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1"
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


class SkillEnhancer:
    """将 thin skills 批量升级为结构化指南"""

    def __init__(self, skills_dir: str = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        if not self.skills_dir.exists():
            raise FileNotFoundError(f"Skills directory not found: {skills_dir}")

    def load_thin_skills(self, start_from: int = 0, end_at: int = None) -> List[Tuple[str, str]]:
        """加载所有 thin skills，返回 [(skill_id, strategy_text), ...]"""
        from agent.b_agent.skills import SkillLibrary
        lib = SkillLibrary(str(self.skills_dir))

        all_skills = sorted(lib.all, key=lambda s: s.id)
        if end_at is not None:
            all_skills = all_skills[start_from:end_at]
        else:
            all_skills = all_skills[start_from:]

        thin_skills = []
        for skill in all_skills:
            # 只增强内容较短且无结构的 skill
            strategy = skill.strategy.strip()
            has_structure = any(marker in strategy for marker in [
                "**When to apply**", "**What to do**", "**Example**", "**Why it works**"
            ])
            if not has_structure:
                thin_skills.append((skill.id, strategy))

        logger.info(f"Found {len(thin_skills)} thin skills to enhance "
                     f"(out of {len(all_skills)} total)")
        return thin_skills

    def enhance_batch(self, skills: List[Tuple[str, str]], batch_size: int = 10,
                      dry_run: bool = False) -> Dict[str, str]:
        """
        批量增强: 每 batch_size 条一起喂给 LLM 升级。
        每批完成后立即保存到磁盘（防止中途崩溃丢失进度）。

        Returns:
            {skill_id: enhanced_markdown_text}
        """
        enhanced = {}

        # 分批处理
        for batch_start in range(0, len(skills), batch_size):
            batch = skills[batch_start:batch_start + batch_size]
            batch_ids = [sid for sid, _ in batch]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(skills) + batch_size - 1) // batch_size

            logger.info(f"Batch {batch_num}/{total_batches}: enhancing skills {batch_ids[0]}...{batch_ids[-1]} "
                        f"({len(batch)} skills)")

            # 格式化当前 batch
            skills_text = "\n---\n".join(
                f"[{sid}]. Current skill (one-liner): {strategy}"
                for sid, strategy in batch
            )

            prompt = SKILL_ENHANCEMENT_TEMPLATE.format(skills_batch=skills_text)

            response = self._call_llm(prompt, max_tokens=3000, temperature=0.3)
            logger.debug(f"Response length: {len(response)} chars")

            # 解析增强后的 skills
            parsed = self._parse_enhanced_skills(response, batch_ids)
            enhanced.update(parsed)

            logger.info(f"  Parsed {len(parsed)}/{len(batch)} enhanced skills from response")

            # 立即保存本批（防止中途崩溃丢失进度）
            if len(parsed) > 0:
                batch_saved = self.save_enhanced_skills(parsed, dry_run=dry_run)
                logger.info(f"  Saved {batch_saved}/{len(parsed)} skills to disk")

            # 速率限制: 批次间隔 2 秒
            if batch_start + batch_size < len(skills):
                time.sleep(2)

        return enhanced

    def _parse_enhanced_skills(self, response: str, expected_ids: List[str]) -> Dict[str, str]:
        """
        从 LLM 输出中解析增强后的结构化 skill。

        格式: [G<N>]. <Title>\n\n**When to apply**: ...\n...
        Skills 之间用 `---` 分隔。
        """
        enhanced = {}

        # 按 `[G<id>]` 分割
        # 匹配模式: [G<number>]. <rest>
        pattern = r'\[(G\d+)\]\.\s+(.*?)(?=\n\n\*\*When to apply\*\*)'
        blocks = re.split(r'(?=\[G\d+\]\.)', response)

        current_id = None
        current_content = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # 提取 skill ID
            id_match = re.match(r'\[(G\d+)\]\.', block)
            if id_match:
                if current_id and current_content:
                    enhanced[current_id] = '\n'.join(current_content).strip()
                current_id = id_match.group(1)
                current_content = [block]
            elif current_id:
                current_content.append(block)

        # 最后一个
        if current_id and current_content:
            enhanced[current_id] = '\n'.join(current_content).strip()

        # 如果解析出的结果太少，尝试按 `---` 分隔
        if len(enhanced) < len(expected_ids) * 0.5:
            logger.warning(f"Only parsed {len(enhanced)} skills, expected ~{len(expected_ids)}. "
                           f"Trying fallback parsing...")
            alt_blocks = re.split(r'\n---\n|\n---|\n\*\*\*\n', response)
            for block in alt_blocks:
                id_match = re.search(r'\[(G\d+)\]', block)
                if id_match:
                    enhanced[id_match.group(1)] = block.strip()

        return enhanced

    def save_enhanced_skills(
        self,
        enhanced: Dict[str, str],
        dry_run: bool = False,
    ) -> int:
        """
        将增强后的 skill 写回文件。

        保留原有 YAML frontmatter (id, version, score_delta, usage_count, triggers),
        只替换正文 strategy 部分。

        Returns:
            成功写入的 skill 数量
        """
        from agent.b_agent.skills import SkillLibrary, Skill
        import yaml

        saved = 0

        for skill_id, enhanced_text in enhanced.items():
            skill_file = self.skills_dir / f"{skill_id}.md"
            if not skill_file.exists():
                logger.warning(f"Skill file not found: {skill_file}")
                continue

            # 读取原有 YAML frontmatter
            original = skill_file.read_text(encoding="utf-8")
            fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", original, re.DOTALL)
            if not fm_match:
                logger.warning(f"No YAML frontmatter in {skill_file}")
                continue

            try:
                metadata = yaml.safe_load(fm_match.group(1))
            except Exception:
                logger.warning(f"Failed to parse YAML from {skill_file}")
                continue

            # 构造新的 SKILL.md: 保留 frontmatter, 替换正文
            triggers = metadata.get("triggers", [])
            trigger_lines = "\n".join(f"  - {t}" for t in triggers) if triggers else "  - coding task"

            # 从 enhanced_text 中提取纯 strategy (去掉 [G#]. 前缀)
            clean_text = re.sub(r'^\[G\d+\]\.\s*', '', enhanced_text, count=1)

            new_content = f"""---
id: {skill_id}
version: {metadata.get('version', 1)}
score_delta: {metadata.get('score_delta', 0.0)}
usage_count: {metadata.get('usage_count', 0)}
triggers:
{trigger_lines}
---

{clean_text}
"""
            if dry_run:
                logger.info(f"[DRY RUN] Would update: {skill_id} "
                            f"({len(original)} → {len(new_content)} chars)")
                # 显示新旧内容的关键片段
                old_strategy = original.split("---\n")[-1].strip()[:80]
                new_strategy = clean_text[:120]
                logger.info(f"  Old: {old_strategy}...")
                logger.info(f"  New: {new_strategy}...")
            else:
                skill_file.write_text(new_content, encoding="utf-8")
                logger.info(f"Enhanced: {skill_id} ({len(original)} → {len(new_content)} chars)")

            saved += 1

        return saved

    def _call_llm(self, prompt: str, max_tokens: int = 3000, temperature: float = 0.3) -> str:
        """调用 LLM API"""
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{LLM_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {LLM_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                logger.warning(f"LLM API status {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"LLM attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        return ""


def main():
    parser = argparse.ArgumentParser(description="Bulk enhance thin skills to structured format (v4)")
    parser.add_argument("--skills_dir", type=str, default=SKILLS_DIR,
                        help="Skills directory")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Skills per LLM call (larger = fewer API calls but slower per call)")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Skip first N skills (for resuming)")
    parser.add_argument("--end_at", type=int, default=None,
                        help="Stop after processing this many skills")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview only, don't write files")
    parser.add_argument("--backup", action="store_true", default=True,
                        help="Backup original skills before modifying (default: True)")
    args = parser.parse_args()

    enhancer = SkillEnhancer(skills_dir=args.skills_dir)

    # 1. 备份
    if not args.dry_run and args.backup:
        import shutil
        backup_dir = Path(ENHANCED_BACKUP_DIR)
        if not backup_dir.exists():
            shutil.copytree(args.skills_dir, backup_dir)
            logger.info(f"Backed up skills to {backup_dir} ({len(list(backup_dir.iterdir()))} files)")

    # 2. 加载 thin skills
    thin_skills = enhancer.load_thin_skills(
        start_from=args.start_from,
        end_at=args.end_at,
    )

    if not thin_skills:
        logger.info("All skills are already structured. Nothing to enhance.")
        return

    logger.info(f"Enhancing {len(thin_skills)} thin skills in batches of {args.batch_size}...")
    logger.info(f"Estimated API calls: {(len(thin_skills) + args.batch_size - 1) // args.batch_size}")

    # 3. 批量增强 (每批完成后立即保存到磁盘)
    enhanced = enhancer.enhance_batch(thin_skills, batch_size=args.batch_size, dry_run=args.dry_run)

    if args.dry_run:
        logger.info(f"\n[DRY RUN] Would enhance {len(enhanced)}/{len(thin_skills)} skills")
        logger.info("Remove --dry_run to actually modify files.")
    else:
        logger.info(f"\n✅ Enhanced {len(enhanced)}/{len(thin_skills)} skills")
        logger.info(f"   Backup saved to: {ENHANCED_BACKUP_DIR}")
        logger.info(f"   Next: re-run ablation test to measure impact")


if __name__ == "__main__":
    main()
