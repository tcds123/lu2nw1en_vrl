"""
经验注入器 — 将蒸馏出的经验指南注入 A 模型的输入 prompt
控制 token 预算，确保不超出 A 模型的上下文窗口 (4096 tokens)
"""

import os
import sys
from typing import Optional

from experience.cache import ExperienceCache

# A 模型上下文窗口
MAX_CONTEXT_TOKENS = 4096
# 经验注入的 token 预算 (留足余量给问题+参考解+ICL)
EXPERIENCE_TOKEN_BUDGET = 500

# A 模型基础 prompt 模板
A_MODEL_BASE_PROMPT = """I will provide you with some examples of generating system prompts. Please carefully study and understand the content and structure of these examples, and learn how to generate high-quality system prompts.

## Task
Given a problem description and a reference solution, generate a system prompt that will guide a code generation model (B model) to produce correct and efficient code for similar problems.

## Requirements
- The system prompt should be SPECIFIC and ACTIONABLE
- Include clear instructions about input/output format, edge cases, and coding style
- Reference the problem domain without copying the solution verbatim
- Output ONLY the system prompt text, no explanations or markdown formatting

{icl_section}

{experiences_section}

## Input
Original Problem: {problem}

Reference Solution:
{ground_truth}

## Output
Generate your system prompt below:"""


def load_icl_examples(example_file: str = None) -> str:
    """加载 few-shot ICL 示例"""
    if example_file is None:
        example_file = "/data/zhuldz/lunwen/rl/train/verl1/sysprompt_icl.txt"

    if os.path.exists(example_file):
        with open(example_file, 'r') as f:
            examples = f.read().strip()
        return f"## Examples\n{examples}"
    return ""


def format_experiences(cache: ExperienceCache,
                       max_experiences: int = 10,
                       token_budget: int = EXPERIENCE_TOKEN_BUDGET) -> str:
    """格式化经验为注入文本，控制 token 预算"""
    raw = cache.format_for_injection(max_experiences)

    # 简单 token 估算: ~1.3 tokens per word for Chinese/English mixed
    estimated_tokens = len(raw.split()) * 1.3
    if estimated_tokens > token_budget:
        # 裁剪: 减少经验条数
        for n in range(max_experiences - 1, 2, -1):
            shorter = cache.format_for_injection(n)
            est = len(shorter.split()) * 1.3
            if est <= token_budget:
                return shorter

    return raw


def build_a_model_input(
    problem: str,
    ground_truth: str,
    experiences: str = "",
    icl_file: str = None,
    cache: Optional[ExperienceCache] = None,
) -> str:
    """
    构造 A 模型的完整输入 prompt (含经验注入)

    Args:
        problem: 原始问题
        ground_truth: 参考解
        experiences: 预格式化的经验文本 (如果提供，优先使用)
        icl_file: ICL 示例文件路径
        cache: 经验池 (如果提供了 cache 但没提供 experiences，自动从 cache 读取)

    Returns:
        完整的 A 模型输入字符串
    """
    # 经验部分
    if experiences:
        exp_section = f"## Important Guidelines (learned from experience)\n{experiences}"
    elif cache and cache.count > 0:
        formatted = format_experiences(cache)
        if formatted:
            exp_section = f"## Important Guidelines (learned from experience)\n{formatted}"
        else:
            exp_section = ""
    else:
        exp_section = ""

    # ICL 部分
    icl_section = load_icl_examples(icl_file)

    # 组装
    prompt = A_MODEL_BASE_PROMPT.format(
        icl_section=icl_section,
        experiences_section=exp_section,
        problem=problem,
        ground_truth=ground_truth,
    )

    # Token 预算检查
    est_tokens = len(prompt.split()) * 1.3
    if est_tokens > MAX_CONTEXT_TOKENS * 0.9:  # 90% = 3686 tokens
        # 紧急裁剪: 缩短参考解
        max_gt_words = int((MAX_CONTEXT_TOKENS * 0.9 - len(prompt.split()) * 1.3 + len(ground_truth.split()) * 1.3))
        if max_gt_words > 100:
            gt_words = ground_truth.split()
            truncated_gt = ' '.join(gt_words[:max_gt_words])
            prompt = A_MODEL_BASE_PROMPT.format(
                icl_section=icl_section,
                experiences_section=exp_section,
                problem=problem,
                ground_truth=truncated_gt + "\n... (truncated)",
            )

    return prompt


# ============================================================
# 快捷函数: 从 cache 直接注入
# ============================================================
def inject_experiences(
    problem: str,
    ground_truth: str,
    cache_path: str = "/data/zhuldz/lunwen/experience/experience_pool.json",
) -> str:
    """一行调用: 从 cache 读取经验并注入到 A 模型输入"""
    cache = ExperienceCache(cache_path)
    return build_a_model_input(problem, ground_truth, cache=cache)


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 测试: 不含经验的 baseline 输入
    base = build_a_model_input(
        problem="Write a function to check if a string is a palindrome.",
        ground_truth="def is_palindrome(s): return s == s[::-1]",
    )
    print("=== Baseline (no experiences) ===")
    print(f"Estimated tokens: {len(base.split()) * 1.3:.0f}")
    print(base[:500])
    print("...")

    # 测试: 含经验的输入
    cache = ExperienceCache()
    if cache.count > 0:
        enhanced = build_a_model_input(
            problem="Write a function to check if a string is a palindrome.",
            ground_truth="def is_palindrome(s): return s == s[::-1]",
            cache=cache,
        )
        print("\n=== With Experiences ===")
        print(f"Estimated tokens: {len(enhanced.split()) * 1.3:.0f}")
        print(enhanced[:800])
