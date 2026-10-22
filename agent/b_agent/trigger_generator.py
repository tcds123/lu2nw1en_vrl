"""
触发词批量生成器 — 借鉴 SkillOpt 的验证门控 + Bayesian-Agent 的可审计改写

对 SkillLibrary 中每条 skill，分析其 strategy 内容，生成精准触发词。

两种模式:
  basic: 修复 YAML 格式，使用 skill strategy 的关键词作为触发词 (无 API 调用)
  llm:   用 LLM 分析 strategy 内容，生成 3-5 个专用触发词 (需 API 调用)

用法:
  python -m agent.b_agent.trigger_generator --mode basic
  python -m agent.b_agent.trigger_generator --mode llm --dry-run
  python -m agent.b_agent.trigger_generator --mode llm
"""

import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRIGGER_GENERATION_PROMPT = """You are analyzing a skill/guideline used by a coding agent.

Below is a skill that the agent uses to improve code generation:

<skill id="{skill_id}">
{skill_content}
</skill>

## Task
Generate 3-5 specific trigger phrases that would help retrieve this skill
when a user's task matches its domain.

Requirements:
1. Each trigger should be 1-4 words, VERY SPECIFIC to this skill's domain
2. Triggers should cover different query phrasings users might use
3. Avoid generic phrases like "code generation" or "prompt writing"
4. Include technical keywords that appear naturally in coding tasks

Examples of GOOD triggers:
  For a skill about "explicit function signatures":
    - "function signature", "type annotations", "parameter types", "return type", "method definition"

  For a skill about "edge case handling":
    - "edge cases", "boundary check", "empty input", "null handling", "corner case"

  For a skill about "performance optimization":
    - "performance", "optimization", "time complexity", "efficient algorithm", "O(n)"

Output format: JSON array only, no other text.
["trigger1", "trigger2", "trigger3", "trigger4", "trigger5"]

Your triggers:"""

# 验证门控规则 (借鉴 SkillOpt)
MAX_TRIGGER_WORDS = 6      # 触发词最多单词数
MIN_TRIGGER_LENGTH = 2     # 触发词最少字符数
MAX_TRIGGERS = 5           # 每条 skill 最多触发词数
BANNED_TRIGGERS = {        # 禁止的通用触发词
    "code generation", "prompt writing", "coding", "programming",
    "code", "prompt", "generation", "writing", "developer",
}


def extract_keywords_from_skill(skill_text: str, max_triggers: int = 5) -> List[str]:
    """从 skill 策略文本中提取关键词作为触发词 (basic 模式，无 API 调用)

    策略:
    1. 提取首句中的关键概念
    2. 提取技术关键词 (function, algorithm, recursion, etc.)
    3. 提取编程范式关键词
    """
    triggers = []
    text_lower = skill_text.lower()

    # 技术概念关键词列表
    tech_keywords = [
        # 算法与数据结构
        ("sorting", "sorting algorithm"),
        ("searching", "search algorithm"),
        ("recursion", "recursion"),
        ("dynamic programming", "dynamic programming"),
        ("dp", "dp"),
        ("greedy", "greedy algorithm"),
        ("backtracking", "backtracking"),
        ("tree", "tree traversal"),
        ("graph", "graph algorithm"),
        ("hash", "hash table"),
        ("linked list", "linked list"),
        ("array", "array manipulation"),
        ("string", "string manipulation"),
        ("stack", "stack data structure"),
        ("queue", "queue data structure"),
        ("heap", "heap data structure"),
        ("binary", "binary search"),
        ("two pointer", "two pointer technique"),
        ("sliding window", "sliding window"),
        ("prefix sum", "prefix sum"),
        ("interval", "interval handling"),
        ("matrix", "matrix manipulation"),
        # 代码质量
        ("edge case", "edge case handling"),
        ("corner case", "corner case handling"),
        ("boundary", "boundary check"),
        ("validation", "input validation"),
        ("error handling", "error handling"),
        ("null check", "null check"),
        ("empty input", "empty input handling"),
        ("exception", "exception handling"),
        ("type hint", "type hints"),
        ("function signature", "function signature"),
        ("return type", "return type"),
        ("annotation", "type annotations"),
        ("docstring", "documentation"),
        ("comment", "code comments"),
        ("readability", "code readability"),
        ("naming", "variable naming"),
        ("modular", "modular design"),
        ("refactor", "code refactoring"),
        ("clean code", "clean code"),
        ("dry", "DRY principle"),
        # 性能
        ("complexity", "time complexity"),
        ("performance", "performance optimization"),
        ("optimization", "optimization"),
        ("memory", "memory efficiency"),
        ("space", "space complexity"),
        ("O(n)", "time O(n)"),
        ("efficient", "efficient algorithm"),
        ("loop", "loop optimization"),
        ("caching", "caching strategy"),
        ("memoization", "memoization"),
        # 编程范式
        ("functional", "functional programming"),
        ("object-oriented", "object-oriented"),
        ("class", "class design"),
        ("iterator", "iterator pattern"),
        ("generator", "generator function"),
        ("lambda", "lambda function"),
        ("closure", "closure"),
        ("decorator", "decorator pattern"),
        ("regex", "regex"),
        ("parsing", "input parsing"),
        ("serialization", "serialization"),
        ("deserialization", "deserialization"),
        # 测试
        ("test case", "test case"),
        ("unit test", "unit test"),
        ("debug", "debugging"),
        ("assertion", "assertion"),
        ("verification", "code verification"),
        # 数据
        ("file i/o", "file I/O"),
        ("stdin", "stdin reading"),
        ("stdout", "stdout writing"),
        ("format", "output formatting"),
        ("precision", "precision handling"),
        ("floating", "floating point"),
        ("integer", "integer arithmetic"),
        ("overflow", "integer overflow"),
        ("negative", "negative number"),
    ]

    found_triggers = set()
    for keyword, trigger in tech_keywords:
        if keyword in text_lower and trigger not in found_triggers:
            if not any(t in BANNED_TRIGGERS for t in [trigger.lower()]):
                found_triggers.add(trigger)
                if len(found_triggers) >= max_triggers:
                    break

    # 如果关键词不足，从首句提取
    if len(found_triggers) < 3:
        first_sentence = skill_text.split(".")[0]
        # 提取名词短语 (简化: 取前几个有意义的词)
        words = first_sentence.split()
        phrase = " ".join(w for w in words[:4] if len(w) > 3 and w.lower() not in BANNED_TRIGGERS)
        if phrase and phrase not in found_triggers:
            found_triggers.add(phrase.lower())

    triggers = list(found_triggers)
    return triggers if triggers else ["coding task"]


def validate_triggers(triggers: List[str]) -> Tuple[List[str], List[str]]:
    """验证门控：过滤无效触发词 (借鉴 SkillOpt)

    Returns:
        (valid_triggers, rejected_triggers)
    """
    valid = []
    rejected = []

    for t in triggers:
        t_stripped = t.strip().lower()

        # 检查是否在禁止列表中
        if t_stripped in BANNED_TRIGGERS:
            rejected.append(f"'{t}' -> banned generic trigger")
            continue

        # 检查长度
        if len(t_stripped) < MIN_TRIGGER_LENGTH:
            rejected.append(f"'{t}' -> too short (<{MIN_TRIGGER_LENGTH} chars)")
            continue

        # 检查单词数
        if len(t_stripped.split()) > MAX_TRIGGER_WORDS:
            rejected.append(f"'{t}' -> too many words (>{MAX_TRIGGER_WORDS})")
            continue

        # 去重
        if t_stripped in [v.lower() for v in valid]:
            rejected.append(f"'{t}' -> duplicate")
            continue

        valid.append(t_stripped)

    return valid, rejected


def regenerate_skills_basic(skills_dir: str, dry_run: bool = False, batch_limit: int = None):
    """Basic 模式: 修复 YAML 格式，使用关键词提取作为触发词"""
    skills_path = Path(skills_dir)
    if not skills_path.exists():
        logger.error(f"Skills directory not found: {skills_dir}")
        return

    skill_files = sorted(skills_path.glob("G*.md"))
    if batch_limit:
        skill_files = skill_files[:batch_limit]

    logger.info(f"Found {len(skill_files)} skill files to process"
                f"{' (dry-run)' if dry_run else ''}")

    fixed_count = 0
    stats = {"total_triggers": 0, "rejected": 0, "unique_skills": 0}

    for md_file in skill_files:
        content = md_file.read_text(encoding="utf-8")
        if not content.startswith("---"):
            continue

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if not match:
            continue

        try:
            import yaml
            metadata = yaml.safe_load(match.group(1))
        except Exception:
            continue

        if not isinstance(metadata, dict):
            continue

        strategy = match.group(2).strip()
        skill_id = metadata.get("id", md_file.stem)

        # 提取关键词作为触发词
        raw_triggers = extract_keywords_from_skill(strategy, max_triggers=MAX_TRIGGERS)

        # 验证门控
        valid_triggers, rejected = validate_triggers(raw_triggers)
        stats["rejected"] += len(rejected)

        if not valid_triggers:
            valid_triggers = ["coding task"]

        stats["total_triggers"] += len(valid_triggers)
        if len(valid_triggers) >= 3:
            stats["unique_skills"] += 1

        # 构建正确的 YAML 内容
        trigger_lines = "\n".join(f"  - {t}" for t in valid_triggers)
        new_content = f"""---
id: {skill_id}
version: {metadata.get("version", 1)}
score_delta: {metadata.get("score_delta", 0.0)}
usage_count: {metadata.get("usage_count", 0)}
triggers:
{trigger_lines}
---

{strategy}
"""
        # 检查是否真的有变化
        if dry_run:
            logger.info(f"  [{skill_id}] would set triggers: {valid_triggers}")
            if rejected:
                logger.info(f"    rejected: {rejected}")
            continue

        if new_content != content:
            md_file.write_text(new_content, encoding="utf-8")
            fixed_count += 1
            if fixed_count <= 5:
                logger.info(f"Fixed {md_file.name}: {valid_triggers}")

    logger.info(f"Basic mode: fixed {fixed_count}/{len(skill_files)} files")
    logger.info(f"Stats: {stats['total_triggers']} total triggers, "
                f"{stats['rejected']} rejected, "
                f"{stats['unique_skills']} skills with >=3 triggers")


def parse_sse_response(response_text: str) -> str:
    """解析 SSE (Server-Sent Events) 响应，拼接所有 content 片段"""
    full_content = ""
    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            data_str = line[6:]  # 去掉 "data: " 前缀
            if data_str == "[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                full_content += content
            except json.JSONDecodeError:
                continue
    return full_content.strip()


def llm_generate_triggers(
    skill_id: str,
    skill_content: str,
    api_key: str,
    base_url: str = "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1/chat/completions",
    model: str = "DeepSeek-V3.2-Fast",
) -> List[str]:
    """LLM 模式: 用 LLM 分析 skill 内容生成触发词"""
    import requests

    prompt = TRIGGER_GENERATION_PROMPT.format(
        skill_id=skill_id,
        skill_content=skill_content,
    )

    try:
        resp = requests.post(
            base_url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
                "stream": False,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "AIGC-USER": "zhuldz",
            },
            timeout=30,
        )

        if resp.status_code != 200:
            logger.warning(f"LLM non-200 for {skill_id}: {resp.status_code} {resp.text[:300]}")
            return []

        raw_text = resp.text

        # 处理 SSE 响应 (API 可能忽略 stream=False)
        if "text/event-stream" in resp.headers.get("Content-Type", ""):
            raw = parse_sse_response(raw_text)
        elif raw_text.strip().startswith("{"):
            try:
                data = json.loads(raw_text)
                raw = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except json.JSONDecodeError:
                raw = ""
        else:
            raw = raw_text.strip()

        if not raw:
            logger.warning(f"LLM empty response for {skill_id}")
            return []

        # 提取 JSON 数组
        json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if json_match:
            triggers = json.loads(json_match.group(0))
            return triggers if isinstance(triggers, list) else []
        else:
            logger.warning(f"LLM response without JSON array for {skill_id}: {raw[:200]}")

    except Exception as e:
        logger.warning(f"LLM generation failed for {skill_id}: {e}")

    return []


def regenerate_skills_llm(
    skills_dir: str,
    api_key: str,
    base_url: str = "https://aimpapi.midea.com/v1",
    model: str = "deepseek-v3.2",
    dry_run: bool = False,
    batch_limit: int = None,
    only_ids: List[str] = None,
    retry_delay: float = 0.5,
):
    """LLM 模式: 用 LLM 分析每条 skill，生成专用触发词"""
    skills_path = Path(skills_dir)
    skill_files = sorted(skills_path.glob("G*.md"))

    # 如果指定了 only_ids，只处理这些 skill
    if only_ids:
        only_ids_set = set(only_ids)
        skill_files = [f for f in skill_files if f.stem in only_ids_set]
        logger.info(f"Filtered to {len(skill_files)} skills by only_ids")

    if batch_limit:
        skill_files = skill_files[:batch_limit]

    logger.info(f"LLM mode: processing {len(skill_files)} skills"
                f"{' (dry-run)' if dry_run else ''}")

    updated_count = 0
    stats = {"api_calls": 0, "api_failures": 0, "triggers_generated": 0}

    for md_file in skill_files:
        content = md_file.read_text(encoding="utf-8")
        if not content.startswith("---"):
            continue

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if not match:
            continue

        try:
            import yaml
            metadata = yaml.safe_load(match.group(1))
        except Exception:
            continue

        strategy = match.group(2).strip()
        skill_id = metadata.get("id", md_file.stem)

        # LLM 生成触发词
        raw_triggers = llm_generate_triggers(
            skill_id, strategy, api_key, base_url, model
        )
        stats["api_calls"] += 1

        if not raw_triggers:
            stats["api_failures"] += 1
            # 回退到 basic 模式
            raw_triggers = extract_keywords_from_skill(strategy)

        # 验证门控
        valid_triggers, rejected = validate_triggers(raw_triggers)

        if not valid_triggers:
            valid_triggers = ["coding task"]

        stats["triggers_generated"] += len(valid_triggers)

        if dry_run:
            logger.info(f"  [{skill_id}] would set triggers: {valid_triggers}")
            if rejected:
                logger.info(f"    rejected: {rejected}")
            continue

        # 写入文件
        trigger_lines = "\n".join(f"  - {t}" for t in valid_triggers)
        new_content = f"""---
id: {skill_id}
version: {metadata.get("version", 1)}
score_delta: {metadata.get("score_delta", 0.0)}
usage_count: {metadata.get("usage_count", 0)}
triggers:
{trigger_lines}
---

{strategy}
"""
        if new_content != content:
            md_file.write_text(new_content, encoding="utf-8")
            updated_count += 1
            if updated_count <= 5:
                logger.info(f"Updated {md_file.name}: {valid_triggers}")

        # 速率限制
        import time
        time.sleep(0.3)

    logger.info(f"LLM mode done: {updated_count} updated, "
                f"{stats['api_calls']} API calls, "
                f"{stats['api_failures']} failures, "
                f"{stats['triggers_generated']} triggers generated")


def main():
    parser = argparse.ArgumentParser(description="Skill 触发词批量生成器")
    parser.add_argument("--mode", choices=["basic", "llm"], default="basic",
                        help="basic: 关键词提取 (无API), llm: LLM分析生成")
    parser.add_argument("--skills-dir", default=None,
                        help="技能文件目录 (默认: agent/b_agent/skills/)")
    parser.add_argument("--api-key", default=None,
                        help="LLM API key (llm 模式必需)")
    parser.add_argument("--base-url", default="https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1/chat/completions",
                        help="LLM API URL (完整路径)")
    parser.add_argument("--model", default="DeepSeek-V3.2-Fast",
                        help="LLM 模型名")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅分析不写入")
    parser.add_argument("--batch-limit", type=int, default=None,
                        help="限制处理的 skill 数量 (测试用)")
    parser.add_argument("--only-ids", type=str, default=None,
                        help="只处理指定的 skill ID (逗号分隔，如 G1,G2,G10)")
    parser.add_argument("--retry-delay", type=float, default=0.5,
                        help="API 调用之间的延迟 (秒)")
    args = parser.parse_args()

    # 确定 skills 目录
    if args.skills_dir:
        skills_dir = args.skills_dir
    else:
        # 从当前目录或脚本所在目录推断
        script_dir = Path(__file__).resolve().parent
        skills_dir = script_dir / "skills"
        if not skills_dir.exists():
            # 尝试从工作目录
            skills_dir = Path.cwd() / "agent" / "b_agent" / "skills"

    skills_dir = str(skills_dir)
    logger.info(f"Skills directory: {skills_dir}")

    if args.mode == "basic":
        regenerate_skills_basic(
            skills_dir,
            dry_run=args.dry_run,
            batch_limit=args.batch_limit,
        )
    elif args.mode == "llm":
        # 获取 API key
        api_key = args.api_key
        if not api_key:
            # 尝试从环境变量获取
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            # 从 core.py 中获取默认 key
            try:
                import importlib.util
                # 这里硬编码默认 key，与 core.py 保持一致
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            except Exception:
                pass
        only_ids = args.only_ids.split(",") if args.only_ids else None
        regenerate_skills_llm(
            skills_dir, api_key, args.base_url, args.model,
            dry_run=args.dry_run, batch_limit=args.batch_limit,
            only_ids=only_ids, retry_delay=args.retry_delay,
        )


if __name__ == "__main__":
    main()
