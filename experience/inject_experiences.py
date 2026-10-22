#!/usr/bin/env python3
"""
经验注入脚本: 将 experience_pool.json 中的经验指南注入到 A 模型训练数据的 prompt 中

用法:
  python inject_experiences.py \
    --parquet /data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax.parquet \
    --cache /data/zhuldz/lunwen/experience/experience_pool.json \
    --output /data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax_exp.parquet \
    --max_experiences 8
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from typing import List, Optional

# 确保可以导入 experience 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experience.cache import ExperienceCache


EXPERIENCE_INJECT_TEMPLATE = """
## PROVEN PROMPT-WRITING GUIDELINES (from training experience)
The following rules have been learned from comparing successful vs. failed system prompts.
Follow them closely to generate HIGH-QUALITY system prompts:

{experiences}

## CRITICAL: Output a COMPLETE and DETAILED system prompt (>300 characters)
Never output just one word or an incomplete phrase. Your prompt must be a full,
actionable instruction set for the code generation model.
"""


def inject_into_prompt(original_input: str, experiences_text: str) -> str:
    """
    将经验指南注入到 A 模型的 input prompt 中
    在 'Now, generate' 之后插入经验块
    """
    if not experiences_text:
        return original_input

    experience_block = EXPERIENCE_INJECT_TEMPLATE.format(experiences=experiences_text)

    # 尝试在 'Now, generate the System Prompt' 之后插入
    markers = [
        "Now, generate the System Prompt for the following case.",
        "Now, generate the System Prompt",
        "generate the System Prompt",
    ]

    for marker in markers:
        if marker in original_input:
            idx = original_input.find(marker) + len(marker)
            return original_input[:idx] + "\n" + experience_block + "\n" + original_input[idx:]

    # Fallback: 在 【Target Case】 之前插入
    if "【Target Case】" in original_input:
        idx = original_input.find("【Target Case】")
        return original_input[:idx] + experience_block + "\n\n" + original_input[idx:]

    # 最后兜底: 在 prompt 末尾添加
    return original_input + "\n\n" + experience_block


def inject_into_input_column(input_value, experiences_text: str):
    """
    将经验注入到 input 列的数据中。
    支持 numpy array 格式 (包含 dict 带 content/role 键)
    """
    if not experiences_text:
        return input_value

    if isinstance(input_value, np.ndarray):
        new_items = []
        for item in input_value:
            if isinstance(item, dict):
                new_item = dict(item)
                if 'content' in new_item:
                    new_item['content'] = inject_into_prompt(str(new_item['content']), experiences_text)
                new_items.append(new_item)
            else:
                new_items.append(item)
        return np.array(new_items, dtype=object)
    elif isinstance(input_value, list):
        new_items = []
        for item in input_value:
            if isinstance(item, dict):
                new_item = dict(item)
                if 'content' in new_item:
                    new_item['content'] = inject_into_prompt(str(new_item['content']), experiences_text)
                new_items.append(new_item)
            else:
                new_items.append(item)
        return new_items
    else:
        return inject_into_prompt(str(input_value), experiences_text)


def _extract_content_text(input_value) -> str:
    """从 input 列中提取 content 文本（用于展示）"""
    if isinstance(input_value, np.ndarray):
        for item in input_value:
            if isinstance(item, dict) and 'content' in item:
                return str(item['content'])
    elif isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, dict) and 'content' in item:
                return str(item['content'])
    return str(input_value)


def _extract_exp_ids(experiences_text: str) -> List[str]:
    """从注入的经验文本中提取经验 ID 列表 (每行只取第一个 [G#])"""
    import re
    ids = []
    for line in experiences_text.split('\n'):
        match = re.match(r'\s*\[(G\d+)\]', line)
        if match:
            ids.append(match.group(1))
    return ids


def main():
    parser = argparse.ArgumentParser(description="注入经验到训练数据")
    parser.add_argument("--parquet", required=True, help="原始 parquet 文件路径")
    parser.add_argument("--cache", default="/data/zhuldz/lunwen/experience/experience_pool.json",
                        help="经验池 JSON 路径")
    parser.add_argument("--output", required=True, help="输出 parquet 路径")
    parser.add_argument("--max_experiences", type=int, default=8,
                        help="每个 prompt 注入的最大经验条数")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印第一条的效果，不实际生成文件")
    parser.add_argument("--randomize", action="store_true", default=True,
                        help="随机化注入 (构造自然实验, 用于因果效应估计)")
    parser.add_argument("--no_randomize", dest="randomize", action="store_false",
                        help="所有 prompt 注入相同经验 (baseline)")
    parser.add_argument("--selection", choices=["random", "ucb", "fixed"], default="random",
                        help="经验选择策略: random/ucb/fixed")
    args = parser.parse_args()

    # 1. 加载经验池
    cache = ExperienceCache(args.cache)
    print(f"经验池: {cache.count} 条经验")

    if cache.count == 0:
        print("经验池为空，无法注入。请先运行 run_experience_extraction.sh")
        return

    # 注入前做一次语义聚类去重
    if cache.count > 5:
        merged = cache.cluster_and_dedup(similarity_threshold=0.55)
        if merged:
            print(f"  注入前聚类: 合并了 {merged} 条冗余经验")

    experiences_text = cache.format_for_injection(args.max_experiences)
    print(f"格式化经验: {len(experiences_text)} 字符")

    # 2. 加载训练数据
    print(f"加载训练数据: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"  共 {len(df)} 条数据, columns={list(df.columns)}")

    if 'input' not in df.columns:
        print("数据中缺少 'input' 列")
        return

    # 3. 根据注入策略生成经验文本
    if args.randomize:
        print(f"\n随机化注入模式 (per-prompt randomization)")
        print(f"选择策略: {args.selection}")

        def make_experience_text():
            """每条 prompt 独立采样经验"""
            if args.selection == "random":
                exp_text, _ = cache.format_randomized_injection(args.max_experiences)
            elif args.selection == "ucb":
                exp_text, _ = cache.format_ucb_injection(args.max_experiences)
            else:
                exp_text = experiences_text  # fixed
            return exp_text

        # Dry run
        first_input = df.iloc[0]['input']
        orig_text = _extract_content_text(first_input)
        sample_exp = make_experience_text()
        enhanced = inject_into_input_column(first_input, sample_exp)
        enhanced_text = _extract_content_text(enhanced)
        print(f"  原始长度: {len(orig_text)} chars")
        print(f"  注入后长度: {len(enhanced_text)} chars (sample)")
        print(f"  示例经验IDs: {_extract_exp_ids(sample_exp)}")

        if args.dry_run:
            print("\nDry run - 未生成文件。")
            return

        # 逐条注入 (每条随机)
        print(f"\n注入经验到 {len(df)} 条数据 (每条独立随机)...")
        import random
        def inject_one(x):
            exp_text = make_experience_text()
            return inject_into_input_column(x, exp_text)
        df['input'] = df['input'].apply(inject_one)
    else:
        # 固定注入 (所有 prompt 相同经验)
        print(f"\n固定注入模式 (所有 prompt 相同经验)")
        first_input = df.iloc[0]['input']
        orig_text = _extract_content_text(first_input)
        enhanced = inject_into_input_column(first_input, experiences_text)
        enhanced_text = _extract_content_text(enhanced)
        print(f"  原始长度: {len(orig_text)} chars")
        print(f"  注入后长度: {len(enhanced_text)} chars")

        if args.dry_run:
            print("\nDry run - 未生成文件。")
            return

        print(f"\n注入经验到 {len(df)} 条数据...")
        df['input'] = df['input'].apply(lambda x: inject_into_input_column(x, experiences_text))

    # 5. 保存
    print(f"保存到: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"完成! 增强后的训练数据已保存到 {args.output}")

    # 6. 统计
    print(f"\n统计:")
    print(f"  注入经验条数: {args.max_experiences}")
    print(f"  每条平均增加 ~{len(experiences_text)} 字符")


if __name__ == "__main__":
    main()
