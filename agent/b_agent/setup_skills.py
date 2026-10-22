#!/usr/bin/env python3
"""
Phase 2 初始化 — 将 experience_pool.json 的经验批量导入为 SKILL.md 文件
运行一次即可，后续手动维护 skill 文件
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.b_agent.skills import SkillLibrary, Skill

SKILLS_DIR = "/data/zhuldz/lunwen/agent/b_agent/skills"
EXPERIENCE_POOL = "/data/zhuldz/lunwen/experience/experience_pool.json"


def main():
    lib = SkillLibrary(SKILLS_DIR)

    print(f"Before import: {lib.count} skills")
    lib.import_from_experience_pool(EXPERIENCE_POOL)
    print(f"After import: {lib.count} skills")

    print("\n--- All Skills ---")
    for skill in lib.all:
        print(f"  [{skill.id}] {skill.name[:80]} | ATE={skill.score_delta:.2f} | uses={skill.usage_count} | triggers={skill.triggers}")

    print("\n--- Active (ATE >= 0) ---")
    for skill in lib.active:
        print(f"  [{skill.id}] {skill.name[:80]} | ATE={skill.score_delta:.2f}")

    print(f"\nDone. Skills directory: {SKILLS_DIR}")


if __name__ == "__main__":
    main()
