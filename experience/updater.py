"""
Skill 池四操作闭环 (Step 3) - 对标 Trace2Skill 的 ExperienceUpdater

将"离线一次性蒸馏 + skill 冻结"改造为"在线 ADD/UPDATE/DELETE/NONE 闭环"。

四操作模型 (借鉴 Trace2Skill 的 group_update + batch_update):
  ADD:    候选 skill 是全新的 -> 加入池子 (分配新 G-id)
  UPDATE: 候选 skill refine/extend 已有 skill -> 合并到 target_id
  DELETE: 候选 skill 揭示已有 skill 有害/过时 -> 移除 target_id
  NONE:   候选 skill 已被现有池充分覆盖 -> 跳过

批量调和 (batch_update): 多个候选 UPDATE 同一 target_id 时, 合并它们的 merged_text。

数据源: 新蒸馏出的候选 skill (e.g., 从最新 GRPO trace 蒸馏, 或手工输入)。
触发: 每 N 步 GRPO 后 / 每次消融实验后 / 手动 `python -m experience.updater`。

使用方式:
    # 从 JSON 文件读候选 skill ({"candidates": ["text1", "text2", ...]})
    python -m experience.updater --candidates new_candidates.json

    # 预览模式 (不修改文件)
    python -m experience.updater --candidates new_candidates.json --dry_run

    # 从 experience_pool.json 增量导入新经验 (与现有池去重)
    python -m experience.updater --from_pool experience/experience_pool.json
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

from agent.b_agent.skills import SkillLibrary, Skill, _skill_lock

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SKILLS_DIR = "/data/zhuldz/lunwen/agent/b_agent/skills"

# LLM 配置 (DeepSeek-V3.2, 用于 skill 合并决策 - 非代码生成任务)
LLM_MODEL = "/model/DeepSeek-V3.2"
LLM_BASE_URL = "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1"
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

DECISION_TEMPLATE = """You are a skill-library curator for a code-generation agent.

You maintain a pool of "skills" (guidelines distilled from past coding experience). A NEW candidate skill has been distilled. Decide how it relates to the EXISTING pool.

## Existing Pool (id: compact description)
{pool_compact}

## Candidate Skill
{candidate}

## Task
Decide ONE operation:
- ADD: the candidate is genuinely new (not covered by any existing skill). A new perspective/rule.
- UPDATE: the candidate refines, extends, or corrects an existing skill. Merge them into a better version.
- DELETE: the candidate reveals an existing skill is harmful/obsolete/contradictory and should be removed.
- NONE: the candidate is already fully covered by an existing skill (redundant).

Respond in STRICT JSON only (no markdown fences):
{{
  "operation": "ADD" | "UPDATE" | "DELETE" | "NONE",
  "target_id": "<existing skill id for UPDATE/DELETE, or null>",
  "merged_text": "<merged skill text for UPDATE, or null>",
  "new_name": "<short name for ADD, or null>",
  "reason": "<one sentence>"
}}

Notes:
- For UPDATE, produce a COMPLETE merged strategy text (preserve the WHEN/WHAT/Example/WHY structure if present).
- For ADD, leave merged_text null; we will generate structure separately.
- Be conservative: prefer NONE/UPDATE over ADD to avoid pool bloat. Prefer NONE over DELETE unless the existing skill is clearly harmful."""


def call_llm(prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
    """调用 DeepSeek-V3.2 API (复用 enhance_skills.py 的模式)。"""
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
                return resp.json()["choices"][0]["message"]["content"]
            logger.warning(f"LLM API status {resp.status_code}: {resp.text[:200]}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"LLM attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return ""


def parse_decision(response: str) -> Optional[Dict]:
    """从 LLM 响应中解析 JSON 决策 (容错: 去除 markdown fence, 找第一个 {...})。"""
    if not response:
        return None
    text = response.strip()
    # 去 markdown fence
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    # 找第一个 {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        dec = json.loads(m.group(0))
        if "operation" not in dec:
            return None
        dec["operation"] = dec["operation"].upper().strip()
        if dec["operation"] not in ("ADD", "UPDATE", "DELETE", "NONE"):
            return None
        return dec
    except json.JSONDecodeError:
        return None


class SkillUpdater:
    """Skill 池四操作闭环管理器 (对标 Trace2Skill ExperienceUpdater)。"""

    def __init__(self, skills_dir: str = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        self.lib = SkillLibrary(str(self.skills_dir))
        logger.info(f"SkillUpdater: loaded {self.lib.count} skills")

    def _next_skill_id(self) -> str:
        """分配下一个 G-id (max+1)。"""
        max_n = 0
        for sid in self.lib._skills.keys():
            m = re.match(r"G(\d+)", sid)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"G{max_n + 1}"

    def _pool_compact(self, max_skills: int = 400) -> str:
        """生成现有池的 compact 表示 (id: compact_text), 供 LLM 对比。"""
        # 用 SkillLibrary 的 compact 生成逻辑
        lines = []
        for skill in self.lib.all[:max_skills]:
            if not skill.compact:
                skill.compact = self.lib._generate_compact(skill.strategy)
            lines.append(f"[{skill.id}] {skill.compact}")
        return "\n".join(lines)

    def decide_one(self, candidate: str, pool_compact: str = None) -> Dict:
        """对单条候选 skill 做 ADD/UPDATE/DELETE/NONE 决策。"""
        if pool_compact is None:
            pool_compact = self._pool_compact()
        prompt = DECISION_TEMPLATE.format(pool_compact=pool_compact, candidate=candidate)
        response = call_llm(prompt, max_tokens=2000, temperature=0.2)
        dec = parse_decision(response)
        if dec is None:
            logger.warning(f"决策解析失败, 默认 NONE. 响应: {response[:200]}")
            return {"operation": "NONE", "reason": "parse_failed", "raw": response[:200]}
        return dec

    def update_pool(self, candidates: List[str], dry_run: bool = False,
                    backup: bool = True) -> Dict[str, int]:
        """
        对一批候选 skill 执行四操作闭环。

        Args:
            candidates: 候选 skill 文本列表
            dry_run: 预览模式, 不修改文件
            backup: 修改前备份 skills 目录

        Returns:
            stats: {added, updated, deleted, none, parse_failed}
        """
        if backup and not dry_run:
            self._backup_skills()

        # 一次性生成 pool_compact (复用, 避免每候选重算)
        pool_compact = self._pool_compact()
        logger.info(f"Pool compact: {len(pool_compact)} chars, {len(candidates)} candidates")

        # 第一遍: 对每个候选做决策
        decisions = []
        for i, cand in enumerate(candidates):
            cand = cand.strip()
            if not cand:
                continue
            logger.info(f"[{i+1}/{len(candidates)}] 决策中: {cand[:60]}...")
            dec = self.decide_one(cand, pool_compact=pool_compact)
            dec["_candidate"] = cand
            decisions.append(dec)
            logger.info(f"  -> {dec['operation']} {dec.get('target_id') or ''} : {dec.get('reason', '')[:80]}")
            time.sleep(0.5)  # 速率限制

        # 第二遍: 批量调和 (多个 UPDATE 同一 target_id -> 合并)
        updates_by_target: Dict[str, List[Dict]] = {}
        for dec in decisions:
            if dec["operation"] == "UPDATE":
                tid = dec.get("target_id")
                if tid:
                    updates_by_target.setdefault(tid, []).append(dec)

        stats = {"added": 0, "updated": 0, "deleted": 0, "none": 0, "parse_failed": 0}

        # 执行 ADD
        for dec in decisions:
            if dec["operation"] == "ADD":
                new_id = self._next_skill_id() if not dry_run else f"G_NEW_{stats['added']}"
                name = dec.get("new_name") or (dec["_candidate"][:60] + "...")
                skill = Skill(id=new_id, name=name, strategy=dec["_candidate"],
                              triggers=["code generation"])
                if not dry_run:
                    with _skill_lock:
                        self.lib.add(skill)
                stats["added"] += 1
                logger.info(f"  ADD -> {new_id}: {name[:50]}")

        # 执行 UPDATE (合并同 target 的多条)
        for tid, decs in updates_by_target.items():
            target = self.lib.get(tid)
            if target is None:
                logger.warning(f"  UPDATE target {tid} 不存在, 跳过")
                continue
            # 合并: 取最后一条 merged_text (或拼接多条)
            merged_texts = [d.get("merged_text") for d in decs if d.get("merged_text")]
            if not merged_texts:
                merged_texts = [d["_candidate"] for d in decs]
            if len(merged_texts) == 1:
                new_strategy = merged_texts[0]
            else:
                # 多条合并: 让 LLM 再做一次合并
                new_strategy = self._llm_merge(tid, target.strategy, merged_texts)
            if not dry_run:
                with _skill_lock:
                    target.strategy = new_strategy
                    target.version += 1
                    target.compact = ""  # 失效, 下次重新生成
                    self.lib.save(target)
            stats["updated"] += 1
            logger.info(f"  UPDATE {tid} (v{target.version if not dry_run else '?'}): {len(decs)} candidates merged")

        # 执行 DELETE
        for dec in decisions:
            if dec["operation"] == "DELETE":
                tid = dec.get("target_id")
                if tid and self.lib.get(tid):
                    if not dry_run:
                        with _skill_lock:
                            self.lib.remove(tid)
                    stats["deleted"] += 1
                    logger.info(f"  DELETE {tid}: {dec.get('reason', '')[:80]}")

        # NONE + parse_failed (parse_failed 优先于 NONE, 否则被吞掉)
        for dec in decisions:
            if dec.get("reason") == "parse_failed":
                stats["parse_failed"] += 1
            elif dec["operation"] == "NONE":
                stats["none"] += 1

        logger.info(f"\n{'='*50}\nSkillUpdater 完成 (dry_run={dry_run})")
        logger.info(f"  ADD:    {stats['added']}")
        logger.info(f"  UPDATE: {stats['updated']}")
        logger.info(f"  DELETE: {stats['deleted']}")
        logger.info(f"  NONE:   {stats['none']}")
        logger.info(f"  解析失败: {stats['parse_failed']}")
        logger.info(f"  池子大小: {self.lib.count} (变更后)")
        return stats

    def _llm_merge(self, target_id: str, old_strategy: str, new_texts: List[str]) -> str:
        """多条候选合并到同一 target 时, 用 LLM 做最终合并。"""
        prompt = f"""Merge these skill updates into ONE coherent skill strategy.

Existing skill [{target_id}]:
{old_strategy}

Update candidates:
""" + "\n---\n".join(new_texts) + """

Produce the FINAL merged strategy text (preserve WHEN/WHAT/Example/WHY structure if present). Output ONLY the merged strategy text, no commentary."""
        merged = call_llm(prompt, max_tokens=1500, temperature=0.2)
        return merged.strip() if merged else old_strategy

    def _backup_skills(self):
        """修改前备份 skills 目录 (带时间戳)。"""
        import shutil
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.skills_dir.parent / f"skills_backup_updater_{ts}"
        shutil.copytree(self.skills_dir, backup_dir)
        logger.info(f"备份 skills -> {backup_dir}")


def load_candidates_from_json(path: str) -> List[str]:
    """从 JSON 文件读候选 skill。支持 {"candidates": [...]} 或 ["...", ...] 或 {id: text}。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        if "candidates" in data:
            return [str(x) for x in data["candidates"]]
        # {id: text} 形式
        return [str(v) for v in data.values()]
    return []


def main():
    parser = argparse.ArgumentParser(description="Skill 池四操作闭环 (Step 3, 对标 Trace2Skill)")
    parser.add_argument("--candidates", type=str, required=False,
                        help="候选 skill 的 JSON 文件 (list 或 {candidates: [...]} 或 {id: text})")
    parser.add_argument("--from_pool", type=str, required=False,
                        help="从 experience_pool.json 增量导入 (与现有池去重)")
    parser.add_argument("--skills_dir", type=str, default=SKILLS_DIR)
    parser.add_argument("--dry_run", action="store_true", help="预览, 不修改文件")
    parser.add_argument("--no_backup", action="store_true", help="跳过备份")
    args = parser.parse_args()

    updater = SkillUpdater(skills_dir=args.skills_dir)

    # 收集候选
    candidates = []
    if args.candidates:
        candidates = load_candidates_from_json(args.candidates)
    elif args.from_pool:
        with open(args.from_pool, "r", encoding="utf-8") as f:
            pool = json.load(f)
        # 增量: 只取池中不存在的 id (去重)
        for exp_id, exp_data in pool.items():
            if exp_id not in updater.lib._skills:
                text = exp_data.get("text", "")
                text = re.sub(r"^\[G\d+\]\.\s*", "", text).strip()
                if text:
                    candidates.append(text)
        logger.info(f"从 {args.from_pool} 增量导入 {len(candidates)} 条新候选 (已去重)")
    else:
        parser.error("需要 --candidates 或 --from_pool")

    if not candidates:
        logger.info("无候选 skill, 退出")
        return

    stats = updater.update_pool(candidates, dry_run=args.dry_run, backup=not args.no_backup)

    if args.dry_run:
        print(f"\n[Dry run] 预览结果: {stats}")
        print("去掉 --dry_run 实际执行。")
    else:
        print(f"\n✅ 完成: {stats}")


if __name__ == "__main__":
    main()
