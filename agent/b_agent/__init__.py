"""
B Agent — 自进化 Agent 框架

架构:
  BAgent (ReAct 循环, 72B API)
    ├── ContextBuilder (从多个 evolvable 组件组装 system prompt)
    ├── SkillLibrary (可进化的技能管理)
    ├── ToolRegistry (工具注册和调用)
    └── AgentMemory (长期记忆)
"""

from .core import BAgent
from .context import ContextBuilder
from .skills import SkillLibrary
from .memory import AgentMemory

__all__ = ["BAgent", "ContextBuilder", "SkillLibrary", "AgentMemory"]
