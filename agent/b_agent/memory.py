"""
AgentMemory — 长期记忆管理

借鉴 nanobot 的 MEMORY.md + history.jsonl 模式。
短期的 ReAct step 记忆由 smolagents AgentMemory 管理。
这里管理跨 session 的长期知识积累。

存储:
  memory/
    MEMORY.md          ← 结构化长期记忆 (Markdown, 按主题分类)
    history.jsonl      ← 归因日志 (每条记录: 任务, 使用了哪些 skill, 结果, 归因)
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """一条长期记忆"""
    timestamp: str
    task_summary: str           # 任务摘要
    skills_used: List[str]      # 使用的技能 ID
    outcome: str                # "success" | "partial" | "failure"
    score: float                # 客观评分
    attribution: str            # 成功/失败的归因
    lesson: str                 # 学到的教训

    def to_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class AgentMemory:
    """长期记忆管理器

    使用方式:
        memory = AgentMemory(workspace="/path/to/agent")
        memory.record(task="sort a list", skills=["G1"], outcome="success", score=8.5,
                      attribution="G1 (input anchoring) prevented hallucination")
    """

    def __init__(self, workspace: str = None):
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self._ensure_files()

    def _ensure_files(self):
        if not self.memory_file.exists():
            self.memory_file.write_text("# Agent Long-term Memory\n\n", encoding="utf-8")
        if not self.history_file.exists():
            self.history_file.write_text("", encoding="utf-8")

    # ================================================================
    # 记录
    # ================================================================
    def record(
        self,
        task: str,
        skills_used: List[str] = None,
        outcome: str = "success",
        score: float = 0.0,
        attribution: str = "",
        lesson: str = "",
    ):
        """记录一条归因日志"""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            task_summary=task[:200],
            skills_used=skills_used or [],
            outcome=outcome,
            score=score,
            attribution=attribution,
            lesson=lesson,
        )
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.to_line() + "\n")

    def consolidate(self, max_entries: int = 50):
        """
        记忆巩固: 将归因日志压缩为 MEMORY.md 中的结构化长期记忆。
        类似于 nanobot 的 Dream consolidation 概念，
        但这里用规则而非 LLM 调用做压缩 (节省成本)。
        """
        entries = self._read_history()
        if not entries:
            return

        # 只处理最近的 entry
        recent = entries[-max_entries:]

        # 按 skill 分组统计
        skill_stats: Dict[str, List[MemoryEntry]] = {}
        for e in recent:
            for sid in e.get("skills_used", []):
                if sid not in skill_stats:
                    skill_stats[sid] = []
                skill_stats[sid].append(e)

        # 生成统计摘要
        lines = ["# Agent Long-term Memory", "", f"Last consolidated: {datetime.now().isoformat()}", ""]

        for skill_id, skill_entries in skill_stats.items():
            successes = [e for e in skill_entries if e.get("outcome") == "success"]
            total = len(skill_entries)
            success_rate = len(successes) / total if total > 0 else 0
            avg_score = sum(e.get("score", 0) for e in skill_entries) / total if total > 0 else 0

            lines.append(f"## {skill_id}")
            lines.append(f"- Success rate: {success_rate:.0%} ({len(successes)}/{total})")
            lines.append(f"- Average score: {avg_score:.2f}")

            # 示例归因
            for e in skill_entries[:2]:
                if e.get("lesson"):
                    lines.append(f"  - {e['lesson']}")
            lines.append("")

        self.memory_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Memory consolidated: {len(recent)} entries → {len(skill_stats)} skill stats")

    # ================================================================
    # 读取
    # ================================================================
    def read(self) -> str:
        """读取长期记忆"""
        return self.memory_file.read_text(encoding="utf-8")

    def get_skill_memory(self, skill_id: str) -> Optional[Dict]:
        """获取特定 skill 的历史表现"""
        entries = self._read_history()
        skill_entries = [
            e for e in entries
            if skill_id in e.get("skills_used", [])
        ]
        if not skill_entries:
            return None

        scores = [e.get("score", 0) for e in skill_entries]
        return {
            "skill_id": skill_id,
            "total_uses": len(skill_entries),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "recent_lessons": [e.get("lesson", "") for e in skill_entries[-3:] if e.get("lesson")],
        }

    def _read_history(self) -> List[Dict]:
        if not self.history_file.exists():
            return []
        entries = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    # ================================================================
    # 统计
    # ================================================================
    def stats(self) -> Dict:
        entries = self._read_history()
        if not entries:
            return {"total": 0}
        scores = [e.get("score", 0) for e in entries]
        return {
            "total": len(entries),
            "avg_score": sum(scores) / len(scores),
            "success_rate": sum(1 for e in entries if e.get("outcome") == "success") / len(entries),
            "unique_skills": len(set(
                sid for e in entries for sid in e.get("skills_used", [])
            )),
        }
