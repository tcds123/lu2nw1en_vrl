"""
SkillLibrary — 可进化的技能管理

借鉴 nanobot 的 SKILL.md 格式 + sema-code-core 的 AgentConfig 模式。

每项技能是一个 Markdown 文件，包含:
  - YAML frontmatter: id, score_delta, usage_count, triggers, evidence
  - 正文: 策略描述

技能生命周期:
  创建 → 评分(ATE) → 强化/削弱 → 剪枝/合并 → 退役
"""

import os
import re
import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 线程锁: Step 2 在线评分回写 (custom_reward 多线程并发) 保护 read-modify-write
_skill_lock = threading.Lock()

# SKILL.md 模板 (YAML frontmatter + 正文)
SKILL_TEMPLATE = """---
id: {skill_id}
version: {version}
score_delta: {score_delta}
usage_count: {usage_count}
online_success_rate: {online_success_rate}
triggers:{trigger_lines}
---

{strategy}
"""


@dataclass
class Skill:
    """一项可进化的技能"""
    id: str                       # 唯一标识 (inherited from experience_pool.json G-IDs)
    name: str                     # 技能名称 (简短描述)
    strategy: str                 # 策略内容 (正文)
    compact: str = ""             # 紧凑版本 (1-2句，用于全量注入context)
    score_delta: float = 0.0      # 因果效应估计 (ATE)
    usage_count: int = 0          # 使用次数
    online_success_rate: float = 0.0  # 在线观测成功率 (EMA of pass_rate when injected); 非因果ATE
    version: int = 1              # 版本号
    triggers: List[str] = field(default_factory=list)  # 触发条件
    file_path: Optional[Path] = None  # 文件路径

    @property
    def fingerprint(self) -> str:
        return hashlib.md5(self.strategy.encode()).hexdigest()[:8]

    def to_markdown(self) -> str:
        if self.triggers:
            trigger_lines = "\n".join(f"  - {t}" for t in self.triggers)
            trigger_lines = "\n" + trigger_lines  # 前面加换行，YAML 格式正确
        else:
            trigger_lines = '\n  - coding task'
        return SKILL_TEMPLATE.format(
            skill_id=self.id,
            version=self.version,
            score_delta=self.score_delta,
            usage_count=self.usage_count,
            online_success_rate=self.online_success_rate,
            trigger_lines=trigger_lines,
            strategy=self.strategy,
        )


class SkillLibrary:
    """技能库管理器

    技能文件存储:
      skills/
        prompt_executable_output.md
        prompt_input_anchoring.md
        error_handling_patterns.md
        ...

    每项技能可从 experience_pool.json 导入，或手动创建。
    """

    def __init__(self, skills_dir: str = None):
        self.skills_dir = Path(skills_dir) if skills_dir else Path.cwd() / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._skills: Dict[str, Skill] = {}
        self._load_all()

    @property
    def all(self) -> List[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.score_delta, reverse=True)

    @property
    def active(self) -> List[Skill]:
        return [s for s in self.all if s.score_delta >= 0]

    @property
    def count(self) -> int:
        return len(self._skills)

    # ================================================================
    # 加载/保存
    # ================================================================
    def _load_all(self):
        """加载所有技能文件"""
        if not self.skills_dir.exists():
            return

        for entry in self.skills_dir.iterdir():
            if entry.is_dir():
                skill_file = entry / "SKILL.md"
            elif entry.suffix == ".md":
                skill_file = entry
            else:
                continue

            if not skill_file.exists():
                continue

            try:
                skill = self._parse_skill_file(skill_file)
                if skill:
                    self._skills[skill.id] = skill
            except Exception as e:
                logger.warning(f"Failed to parse skill file {skill_file}: {e}")

        logger.info(f"Loaded {self.count} skills from {self.skills_dir}")

    def _parse_skill_file(self, path: Path) -> Optional[Skill]:
        """解析 SKILL.md 文件 (YAML frontmatter + 正文)"""
        content = path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if not match:
            return None

        import yaml
        try:
            metadata = yaml.safe_load(match.group(1))
        except Exception:
            return None

        if not isinstance(metadata, dict):
            return None

        strategy = match.group(2).strip()
        skill_id = metadata.get("id", path.stem)

        return Skill(
            id=skill_id,
            name=metadata.get("name", path.stem),
            strategy=strategy,
            score_delta=float(metadata.get("score_delta", 0.0)),
            usage_count=int(metadata.get("usage_count", 0)),
            online_success_rate=float(metadata.get("online_success_rate", 0.0)),
            version=int(metadata.get("version", 1)),
            triggers=list(metadata.get("triggers", [])),
            file_path=path,
        )

    def save(self, skill: Skill):
        """保存技能到文件"""
        path = skill.file_path or (self.skills_dir / f"{skill.id}.md")
        path.write_text(skill.to_markdown(), encoding="utf-8")
        skill.file_path = path

    # ================================================================
    # CRUD
    # ================================================================
    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def add(self, skill: Skill) -> str:
        """添加新技能"""
        self._skills[skill.id] = skill
        self.save(skill)
        logger.info(f"Added skill: {skill.id} - {skill.name}")
        return skill.id

    def update_score(self, skill_id: str, delta: float, alpha: float = 0.1):
        """EMA 更新技能的因果效应评分"""
        skill = self._skills.get(skill_id)
        if skill:
            skill.score_delta = (1 - alpha) * skill.score_delta + alpha * delta
            skill.usage_count += 1
            self.save(skill)


    def record_outcome(self, skill_ids: List[str], pass_rate: float, alpha: float = 0.1):
        """[Step 2 在线反馈] 记录一次生成结果到所用 skill 的 online_success_rate (EMA)。

        全量注入下, 同一次生成的所有 co-injected skill 共享同一 pass_rate,
        因此这是观测性信号 (success-rate-when-injected), 非因果 ATE。
        因果 ATE 由 offline prune_eval / skill_loop 写入 score_delta;
        本方法刻意不触碰 score_delta, 避免污染因果估计 (否则难任务里被
        co-inject 的 skill 会被误判有害而遭 prune_negative 剪掉)。

        刻意不递增 usage_count: usage_count 原语义是 "ATE 评估次数" (唯一写者
        是 update_score), prune_negative 用它做门控 (min_usage)。若 record_outcome
        也递增, 注入次数会膨胀 usage_count, 使 prune 门控被注入次数而非 ATE 评估
        次数越过 -> 单次 ATE 即可能误剪。在线注入频率已由 online_success_rate EMA
        隐式捕获, 无需另存计数。

        线程安全: 由 _skill_lock 保护 (custom_reward 用 ThreadPoolExecutor 并发)。
        不立即 save (高频调用, IO 开销大); 调用方定期调 flush() 持久化。
        """
        with _skill_lock:
            for sid in skill_ids:
                skill = self._skills.get(sid)
                if skill:
                    skill.online_success_rate = (1 - alpha) * skill.online_success_rate + alpha * pass_rate

    def flush(self):
        """[Step 2] 把内存中所有 skill 的 online_success_rate / usage_count 写回磁盘。"""
        with _skill_lock:
            for skill in list(self._skills.values()):
                try:
                    self.save(skill)
                except Exception as e:
                    logger.warning(f"flush save failed for {skill.id}: {e}")

    def remove(self, skill_id: str):
        """删除技能"""
        skill = self._skills.pop(skill_id, None)
        if skill and skill.file_path and skill.file_path.exists():
            skill.file_path.unlink()
            logger.info(f"Removed skill: {skill_id}")

    # ================================================================
    # 检索
    # ================================================================
    def retrieve(self, task: str, top_k: int = 5, diversity: bool = True) -> List[Skill]:
        """
        根据任务检索相关技能 (关键词匹配 + 多样性惩罚 + ATE 加权)

        Args:
            task: 任务描述
            top_k: 返回技能数
            diversity: 启用多样性惩罚 (避免总是返回相同的高频 skill)
        """
        task_lower = task.lower()
        task_words = set(task_lower.split())

        scored = []
        for skill in self.all:
            base_score = self._relevance(task_lower, task_words, skill)
            if base_score <= 0:
                continue

            # ATE 加权: score_delta 高的 skill 提升排名
            ate_bonus = max(0, skill.score_delta) * 0.5

            # 多样性惩罚: 基于 usage_count（被使用越多次，越降权）
            # 这样低频但相关的 skill 有机会被检索到
            diversity_penalty = 0.0
            if diversity and skill.usage_count > 0:
                # log penalty: 使用 100 次 → ~0.6 惩罚因子
                diversity_penalty = min(0.5, 0.05 * __import__('math').log(max(1, skill.usage_count)))

            final_score = base_score * (1.0 - diversity_penalty) + ate_bonus
            scored.append((skill, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:top_k]]

    def _relevance(self, task_lower: str, task_words: set, skill: Skill) -> float:
        """计算技能与任务的相关性"""
        # 触发词匹配
        trigger_hits = 0
        for t in skill.triggers:
            # 防御性处理：确保 trigger 是字符串 (YAML 可能把含冒号的文本解析成 dict)
            if not isinstance(t, str):
                t = str(t)
            t_words = set(t.lower().split())
            if t_words & task_words:  # 交集非空
                trigger_hits += 1
            # 完整短语匹配（更精准）
            if t.lower() in task_lower:
                trigger_hits += 2

        # 策略正文关键词匹配
        strategy_words = set(skill.strategy.lower().split())
        common = strategy_words & task_words
        strategy_score = len(common) / max(1, len(strategy_words))

        return trigger_hits * 0.5 + strategy_score * 30  # 降低关键词匹配权重，减少噪音

    def format_for_context(self, skills: List[Skill], compact: bool = True) -> str:
        """格式化技能列表为 context 文本 [已废弃 — 请使用 format_all_active_for_context]

        Args:
            compact: 如果为 True，只提取 What to do 部分（减小 token 消耗）
                     如果为 False，输出完整结构化内容
        """
        if not skills:
            return ""
        lines = []
        for skill in skills:
            strategy = skill.strategy
            if compact:
                # 提取核心内容: title + What to do + Example（省略 When/Why 以控制 token）
                compact_text = self._compact_skill(strategy)
                if compact_text:
                    lines.append(f"- {compact_text}")
                else:
                    # Fallback: 如果解析失败，取前 200 字符
                    lines.append(f"- {strategy[:200]}")
            else:
                lines.append(f"### [{skill.id}] {skill.name}")
                lines.append(strategy)
            lines.append("")
        return "\n".join(lines)

    def format_all_active_for_context(self, max_skills: int = 50, max_tokens: int = 3000) -> Tuple[str, List["Skill"]]:
        """
        对标 Trace2Skill: 将全部 active skill 的 compact 版本注入 context。

        Trace2Skill 的核心洞察: 全部注入 experience, 由 agent 自主判断适用性,
        而非用粗粒度的触发词匹配替 agent 做选择。

        Args:
            max_skills: 最多注入多少条 (按 score_delta 排序取 top)
            max_tokens: token 预算上限

        Returns:
            (skill_text, selected_skills):
              - skill_text: 格式化的 skill 列表文本, 每行一条 "[id]. compact_text"
              - selected_skills: 实际被选中注入的 Skill 列表
            调用方应基于 selected_skills 记录 skills_used, 避免"记了402条但只注入50条"
            的归因偏差 (Step 2 ATE 评分需要准确的 usage 归属)。
        """
        # 自动生成 compact 版本（如果还没有）
        for skill in self._skills.values():
            if not skill.compact:
                skill.compact = self._generate_compact(skill.strategy)

        # 按 score_delta 降序排列，取 top-N
        active = [s for s in self._skills.values() if s.score_delta >= 0]
        active.sort(key=lambda s: s.score_delta, reverse=True)

        selected = []
        total_est_tokens = 0
        for skill in active:
            est = max(1, len(skill.compact) // 4)
            if len(selected) < max_skills and total_est_tokens + est <= max_tokens:
                selected.append(skill)
                total_est_tokens += est

        if not selected:
            return "", []

        lines = [f"[{s.id}] {s.compact}" for s in selected]
        return "\n".join(lines), selected

    @staticmethod
    def _generate_compact(strategy: str) -> str:
        """
        从结构化 strategy 生成 1-2 句的 compact 版本。

        优先提取 What to do（最核心的可操作建议），
        如果太短则补充标题作为上下文。
        """
        # 提取 What to do 部分（最核心）
        what_match = re.search(r'\*\*What to do\*\*:\s*(.*?)(?=\n\*\*|\Z)', strategy, re.DOTALL)
        if what_match:
            what = what_match.group(1).strip()
            # 限制 150 字符 — 足够清晰又不冗长
            if len(what) > 150:
                what = what[:150].rsplit('.', 1)[0].strip() + "."
            return what

        # Fallback: 提取标题 + 前两句话
        lines = strategy.strip().split('\n')
        title = lines[0].strip() if lines else ""
        title = re.sub(r'^\[G\d+\]\.\s*', '', title)

        # 收集正文中前几个非空行（跳过空行和markdown标记）
        body_lines = []
        for line in lines[1:]:
            stripped = line.strip()
            if stripped and not stripped.startswith('**'):
                body_lines.append(stripped)
            if len(' '.join(body_lines)) > 100:
                break

        body = ' '.join(body_lines)
        if body:
            combined = f"{title}: {body}"
            if len(combined) > 200:
                combined = combined[:200].rsplit('.', 1)[0].strip() + "."
            return combined

        return title[:200] if title else strategy[:200]

    @staticmethod
    def _compact_skill(strategy: str) -> str:
        """将结构化 skill 压缩为紧凑的单行格式 [保留兼容旧代码]"""
        # 提取标题（第一行）
        lines = strategy.strip().split('\n')
        title = lines[0].strip()
        # 去掉可能的 [G#]. 前缀
        title = re.sub(r'^\[G\d+\]\.\s*', '', title)

        # 提取 **What to do** 部分
        what_match = re.search(r'\*\*What to do\*\*:\s*(.*?)(?=\n\*\*|\Z)', strategy, re.DOTALL)
        if what_match:
            what = what_match.group(1).strip()
            # 限制长度
            if len(what) > 250:
                what = what[:250] + "..."
            return f"**{title}**: {what}"

        # Fallback: 取前 200 字符
        if len(strategy) > 200:
            return strategy[:200] + "..."
        return strategy

    # ================================================================
    # 剪枝
    # ================================================================
    def prune_negative(self, min_usage: int = 3, threshold: float = -0.5) -> int:
        """剪枝负效应技能"""
        to_remove = [
            sid for sid, skill in self._skills.items()
            if skill.score_delta < threshold and skill.usage_count >= min_usage
        ]
        for sid in to_remove:
            self.remove(sid)
        if to_remove:
            logger.info(f"Pruned {len(to_remove)} negative-effect skills")
        return len(to_remove)

    def import_from_experience_pool(self, pool_path: str):
        """从 experience_pool.json 导入技能 (v3 迁移)"""
        import json
        if not os.path.exists(pool_path):
            return

        with open(pool_path, 'r') as f:
            pool = json.load(f)

        imported = 0
        for exp_id, exp_data in pool.items():
            if exp_id in self._skills:
                continue

            text = exp_data.get("text", "")
            # 解析 [G#]. 前缀
            text = re.sub(r'^\[G\d+\]\.\s*', '', text).strip()

            skill = Skill(
                id=exp_id,
                name=text[:60] + ("..." if len(text) > 60 else ""),
                strategy=text,
                score_delta=float(exp_data.get("score_delta", 0.0)),
                usage_count=int(exp_data.get("usage_count", 0)),
                triggers=["code generation", "prompt writing"],
            )
            self.add(skill)
            imported += 1

        logger.info(f"Imported {imported} skills from {pool_path}")
