"""
ContextBuilder - 从多个 evolvable 组件组装 Agent 的 system prompt

借鉴 nanobot 的组件化思想 + sema-code-core 的 AgentConfig 模式。
每个组件是独立的文件/Markdown，可独立追踪、独立版本管理、独立进化。

组件加载优先级:
  1. 角色定义 (SOUL.md)
  2. 任务相关的技能 (skills/)
  3. 长期记忆 (memory/)
  4. 可用工具描述 (动态生成)
  5. 行为规范 (rules)
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class ContextComponent:
    """一个可进化的 context 组件"""
    name: str            # 组件名
    content: str         # 组件文本内容
    source: str          # 来源 (soul/skill/memory/tool/rule)
    version: int = 1     # 版本号
    enabled: bool = True # 是否启用

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.content) // 4)

    @property
    def fingerprint(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:8]


class ContextBuilder:
    """
    从多个 evolvable 组件组装 Agent 的 system prompt。

    使用方式:
        builder = ContextBuilder(workspace="/path/to/agent")
        system_prompt = builder.build(task="Write a sorting function")
    """

    def __init__(self, workspace: str = None):
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self._components: Dict[str, ContextComponent] = {}
        self._ensure_dirs()

    def _ensure_dirs(self):
        (self.workspace / "skills").mkdir(parents=True, exist_ok=True)
        (self.workspace / "memory").mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 组件加载
    # ================================================================
    def load_soul(self) -> Optional[ContextComponent]:
        """加载角色定义 (借鉴 nanobot SOUL.md)"""
        path = self.workspace / "SOUL.md"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            return ContextComponent(name="soul", content=content, source="soul")
        return None

    def load_skill(self, skill_name: str) -> Optional[ContextComponent]:
        """加载单个技能文件 (借鉴 nanobot SKILL.md + sema AgentConfig)"""
        path = self.workspace / "skills" / skill_name / "SKILL.md"
        if not path.exists():
            path = self.workspace / "skills" / f"{skill_name}.md"
        if path.exists():
            content = self._strip_frontmatter(path.read_text(encoding="utf-8"))
            return ContextComponent(name=skill_name, content=content, source="skill")
        return None

    def load_all_skills(self) -> List[ContextComponent]:
        """加载所有可用技能"""
        skills = []
        skills_dir = self.workspace / "skills"
        if not skills_dir.exists():
            return skills

        for entry in skills_dir.iterdir():
            if entry.is_dir():
                skill_file = entry / "SKILL.md"
                if skill_file.exists():
                    name = entry.name
                    content = self._strip_frontmatter(skill_file.read_text(encoding="utf-8"))
                    skills.append(ContextComponent(name=name, content=content, source="skill"))
            elif entry.suffix == ".md":
                name = entry.stem
                content = self._strip_frontmatter(entry.read_text(encoding="utf-8"))
                skills.append(ContextComponent(name=name, content=content, source="skill"))

        return skills

    def load_memory(self) -> Optional[ContextComponent]:
        """加载长期记忆 (借鉴 nanobot MEMORY.md)"""
        path = self.workspace / "memory" / "MEMORY.md"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            # 只取最近的部分（避免无限增长）
            sections = content.split("\n## ")
            recent = sections[:5]  # 取前 5 个 section
            trimmed = "\n## ".join(recent)
            return ContextComponent(name="memory", content=trimmed, source="memory")
        return None

    # ================================================================
    # Context 组装
    # ================================================================
    def build(
        self,
        task: str = "",
        tools_description: str = "",
        max_tokens: int = 4000,
    ) -> str:
        """
        组装最终的 system prompt。

        顺序: SOUL -> Task -> Skills -> Tools -> Memory -> Rules
        """
        parts = []

        # 1. 角色定义 (最高优先级，不可被覆盖)
        soul = self.load_soul()
        if soul:
            parts.append(soul.content)
        else:
            parts.append(self._default_soul())

        # 2. 当前任务
        if task:
            parts.append(f"## Current Task\n{task}")

        # 3. 工具描述
        if tools_description:
            parts.append(f"## Available Tools\n{tools_description}")

        # 4. Skills 不在此处加载 - 由 core.py 通过 format_all_active_for_context() 注入
        #    (对标 Trace2Skill: 全量 compact 注入, 由 agent 自主判断适用性)

        # 5. 长期记忆
        memory = self.load_memory()
        if memory:
            parts.append(f"## Memory\n{memory.content}")

        # 6. 行为规范
        parts.append(self._default_rules())

        system_prompt = "\n\n".join(parts)

        # Token 预算保护
        if len(system_prompt) // 4 > max_tokens:
            # 先裁剪 skills（最长、且可牺牲）
            system_prompt = self._trim_to_budget(system_prompt, max_tokens)

        return system_prompt

    def build_stateless(self, task: str, tools_description: str = "", skills: List[str] = None) -> str:
        """无文件依赖的快速组装 (用于测试/API 调用)"""
        parts = [self._default_soul()]

        if task:
            parts.append(f"## Current Task\n{task}")

        if tools_description:
            parts.append(f"## Available Tools\n{tools_description}")

        if skills:
            skill_texts = []
            for name in skills:
                skill = self.load_skill(name)
                if skill:
                    skill_texts.append(f"### {name}\n{skill.content}")
            if skill_texts:
                parts.append("## Skills\n" + "\n\n".join(skill_texts))

        parts.append(self._default_rules())
        return "\n\n".join(parts)

    # ================================================================
    # 组件追踪 (自进化用)
    # ================================================================
    def snapshot(self) -> Dict[str, str]:
        """获取所有组件的指纹快照 (用于版本追踪和归因)"""
        snap = {}
        for comp in self._list_all_components():
            snap[comp.name] = comp.fingerprint
        return snap

    def _list_all_components(self) -> List[ContextComponent]:
        components = []
        soul = self.load_soul()
        if soul: components.append(soul)
        components.extend(self.load_all_skills())
        memory = self.load_memory()
        if memory: components.append(memory)
        return components

    # ================================================================
    # 辅助方法
    # ================================================================
    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """移除 YAML frontmatter，保留正文 (prompt 部分)"""
        if not content.startswith("---"):
            return content.strip()
        match = re.match(r"^---\s*\n.*?\n---\s*\n?", content, re.DOTALL)
        if match:
            return content[match.end():].strip()
        return content.strip()

    @staticmethod
    def _default_soul() -> str:
        return (
            "You are an expert Python programmer. Write correct, efficient, and well-structured code. "
            "Follow PEP 8 guidelines. Include proper docstrings. Handle edge cases explicitly. "
            "Output only the complete, runnable function in a ```python code block."
        )

    @staticmethod
    def _meta_skill_guidance() -> str:
        """Meta-Skill 引导语: 教会 agent 如何选择性使用 skill (减少负迁移)。

        对标 SkillLens 的 meta-skill 实验: 给 agent 一个"选择性忽略"的许可,
        避免 skill 在不适用任务上被硬套导致负迁移。零 API 成本。
        b_agent_interface.py 绕过 context.build() 时也复用此方法, 保证训练路径
        与 BAgent (core.py) 路径的 Meta-Skill 引导语一致。
        """
        return """## How to Use Guidelines & Skills
The following guidelines are distilled from past coding experiences. They are **suggestions**, not hard rules.

When reading a guideline:
- First, check if it's actually relevant to your current task. If your task doesn't involve the scenario described, you MUST IGNORE that guideline.
- A guideline should HELP you write correct code - never follow a guideline if it would make your code incorrect.
- Prioritize writing correct, runnable code that passes the tests. Guidelines are secondary.

Not every guideline applies to every task. Skip the ones that don't fit."""

    @staticmethod
    def _rules() -> str:
        """基础行为规范"""
        return """## Rules
- Think step by step, but be concise.
- Prefer editing existing code over creating new files.
- Return complete, runnable code in your final answer.
- If the user's instructions contradict these rules, follow the user."""

    @staticmethod
    def _default_rules() -> str:
        """Meta-Skill 引导语 + 基础规范 (供 context.build 使用)"""
        return ContextBuilder._meta_skill_guidance() + "\n\n" + ContextBuilder._rules()

    @staticmethod
    def _trim_to_budget(text: str, max_tokens: int) -> str:
        """按 token 预算裁剪"""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        # 保留 SOUL + Rules，裁剪中间
        parts = text.split("\n\n")
        head = parts[0] if parts else ""
        tail = parts[-1] if len(parts) > 1 else ""

        budget = max_chars - len(head) - len(tail) - 200
        if budget <= 0:
            return head[:max_chars]

        middle_parts = parts[1:-1] if len(parts) > 2 else []
        trimmed_middle = []
        for p in middle_parts:
            if len("\n\n".join([head] + trimmed_middle + [p] + [tail])) <= max_chars:
                trimmed_middle.append(p)
            else:
                break

        return "\n\n".join([head] + trimmed_middle + [tail])
