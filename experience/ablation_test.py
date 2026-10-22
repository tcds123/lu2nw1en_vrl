#!/usr/bin/env python3
"""
经验注入消融测试 — 纯推理，不做 GRPO 训练

对比:
  A组: base A 模型 (无经验)
  B组: base A 模型 + 经验注入
  C组: base A 模型 + 经验注入 + 强约束模板

测量:
  - 系统提示词长度分布
  - B 模型代码通过率
  - C 模型主观评分
  - 提示词多样性 (避免所有输出雷同)

这剥离了 GRPO 噪声 — 如果 B 组显著优于 A 组，
说明经验本身有效，问题出在 GRPO 的 reward signal。
"""

import os
import sys
import json
import time
import argparse
import re
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd

PROJECT_ROOT = "/data/zhuldz/lunwen"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experience"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rl/train/verl1"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "judge"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experience.cache import ExperienceCache
from experience.inject_experiences import inject_into_prompt, EXPERIENCE_INJECT_TEMPLATE


# ============================================================
# 配置
# ============================================================
A_MODEL_PATH = "/data/zhuldz/lunwen/models/Qwen3-4B-Instruct-2507"
CACHE_PATH = "/data/zhuldz/lunwen/experience/experience_pool.json"
TEST_DATA_PATH = "/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax.parquet"
N_TEST_SAMPLES = 30  # 每组测试问题数
MAX_EXPERIENCES = 8

# A 模型生成参数
GEN_TEMP = 0.4
GEN_TOP_P = 0.9
GEN_MAX_TOKENS = 1024


# ============================================================
# A 模型加载
# ============================================================
def load_a_model():
    print(f"Loading A model: {A_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(A_MODEL_PATH, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        A_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def generate_prompt(model, tokenizer, input_text: str) -> str:
    """A 模型推理: 输入 → 系统提示词"""
    messages = [{"role": "user", "content": input_text}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=GEN_MAX_TOKENS,
            temperature=GEN_TEMP,
            top_p=GEN_TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


# ============================================================
# B + C 评估 (复用现有)
# ============================================================
def evaluate_prompt(system_prompt: str, user_query: str, ground_truth: str,
                    unit_tests: list) -> Dict:
    """用 B 模型生成代码 + C 模型评分"""
    result = {
        "system_prompt_len": len(system_prompt),
        "code": "",
        "pass_rate": 0.0,
        "obj_score": 0.0,
        "sub_score": 0.0,
        "final_score": 0.0,
    }

    try:
        # B 模型生成代码
        from b_model_api import BModelAPI
        b_model = BModelAPI()
        code = b_model.generate_code(user_query, system_prompt)

        if not code:
            result["final_score"] = -10.0
            return result

        result["code"] = code[:2000]

        # 单元测试
        from custom_reward import compute_execution_pass_rate
        pass_rate = compute_execution_pass_rate(code, unit_tests)

        # C 模型评分
        from c_interface import load_c_model, compute_reward
        c_judge = load_c_model()
        sub_score, c_details = compute_reward(
            c_judge, code, ground_truth, system_prompt, user_query, user_query
        )

        obj_score = 10.0 if pass_rate >= 1.0 else (pass_rate * 7.5 if pass_rate > 0 else 0.0)
        result["pass_rate"] = pass_rate
        result["obj_score"] = obj_score
        result["sub_score"] = sub_score
        result["final_score"] = obj_score * 0.7 + sub_score * 0.3

    except Exception as e:
        result["error"] = str(e)
        result["final_score"] = -5.0

    return result


# ============================================================
# 三组对比测试
# ============================================================
@dataclass
class GroupResult:
    name: str
    samples: List[Dict] = field(default_factory=list)

    @property
    def avg_score(self): return np.mean([s["final_score"] for s in self.samples])
    @property
    def avg_len(self): return np.mean([s["system_prompt_len"] for s in self.samples])
    @property
    def avg_pass_rate(self): return np.mean([s["pass_rate"] for s in self.samples])
    @property
    def len_std(self): return np.std([s["system_prompt_len"] for s in self.samples])
    @property
    def collapse_rate(self):
        """< 20 字符的退化比例"""
        return sum(1 for s in self.samples if s["system_prompt_len"] < 20) / len(self.samples)
    @property
    def short_rate(self):
        """< 100 字符的比例"""
        return sum(1 for s in self.samples if s["system_prompt_len"] < 100) / len(self.samples)


def build_a_input(problem_text: str, ground_truth: str, experiences_text: str = "",
                  strong_constraint: bool = False) -> str:
    """构造 A 模型输入 (对齐训练时的格式)"""
    base = f"""You are an expert System Prompt Engineer. Your goal is to optimize the performance of a code generation AI.
Task: Based on the provided User Query and Ground Truth Code, write a high-quality 'System Prompt' that will guide another AI to generate the correct code.
The System Prompt you write should:
1. Define a clear persona (e.g., Expert Coder).
2. Set strict coding standards (e.g., performance, robustness, modularity).
3. NOT contain the solution code itself.

Now, generate the System Prompt for the following case.
IMPORTANT Rules:
- Output ONLY the final System Prompt content.
- Do not output any 'Here is the prompt', 'Let me think' or reasoning process.
- START directly with the system prompt text.

"""

    if experiences_text:
        exp_block = EXPERIENCE_INJECT_TEMPLATE.format(experiences=experiences_text)
        base += f"\n{exp_block}\n"

    if strong_constraint:
        base += """
## MANDATORY OUTPUT FORMAT
Your output MUST:
- Be at least 500 characters long
- Contain specific instructions about: input format, output format, edge cases, algorithm approach, and coding style
- Use imperative language (e.g., "You must...", "Ensure that...")
- NOT be a single generic sentence
"""

    base += f"""
【Target Case】
User Query: {problem_text}

Ground Truth Code:
```python
{ground_truth}
```
"""
    return base


def load_test_data(n: int) -> List[Tuple[str, str, list]]:
    """加载测试问题 (随机采样 n 个)"""
    df = pd.read_parquet(TEST_DATA_PATH)
    indices = random.sample(range(len(df)), min(n, len(df)))

    samples = []
    for idx in indices:
        row = df.iloc[idx]
        raw_input = str(row.get("raw_input", ""))

        # 提取 ground_truth 和 unit_tests
        output = row.get("output", "{}")
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except:
                output = {"ground_truth": output}

        gt = output.get("ground_truth", "")
        ut = output.get("unit_tests", [])

        if isinstance(ut, str):
            try:
                ut = json.loads(ut)
            except:
                ut = [ut] if ut else []

        if raw_input and gt:
            samples.append((raw_input, gt, ut))

    return samples[:n]


def run_ablation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=N_TEST_SAMPLES)
    parser.add_argument("--skip_eval", action="store_true",
                        help="仅生成提示词不评估 (快速测试)")
    args = parser.parse_args()

    # 1. 加载 A 模型
    model, tokenizer = load_a_model()

    # 2. 加载经验池
    cache = ExperienceCache(CACHE_PATH)
    print(f"Experience pool: {cache.count} entries")
    experiences_text = cache.format_for_injection(MAX_EXPERIENCES)

    # 3. 加载测试数据
    test_samples = load_test_data(args.n_samples)
    print(f"Test samples: {len(test_samples)}")

    # 4. 运行三组
    groups = {
        "A_baseline": GroupResult(name="A: Baseline (no experiences)"),
        "B_experiences": GroupResult(name="B: With experiences"),
        "C_strong": GroupResult(name="C: Experiences + strong constraints"),
    }

    print(f"\n{'='*60}")
    print("Running ablation test...")
    print(f"{'='*60}")

    for i, (problem, gt, ut) in enumerate(test_samples):
        print(f"\n[{i+1}/{len(test_samples)}] Problem: {problem[:60]}...")

        for group_key, config in [
            ("A_baseline", {"experiences_text": "", "strong_constraint": False}),
            ("B_experiences", {"experiences_text": experiences_text, "strong_constraint": False}),
            ("C_strong", {"experiences_text": experiences_text, "strong_constraint": True}),
        ]:
            a_input = build_a_input(problem, gt, **config)
            sys_prompt = generate_prompt(model, tokenizer, a_input)

            result = {
                "problem": problem[:200],
                "system_prompt": sys_prompt[:3000],
                "system_prompt_len": len(sys_prompt),
            }

            if not args.skip_eval:
                eval_result = evaluate_prompt(sys_prompt, problem, gt, ut)
                result.update(eval_result)
                print(f"  {group_key}: len={len(sys_prompt)}, "
                      f"pass={eval_result.get('pass_rate',0):.2f}, "
                      f"score={eval_result['final_score']:.2f}")
            else:
                print(f"  {group_key}: len={len(sys_prompt)}, "
                      f"preview={sys_prompt[:80].replace(chr(10),' ')}...")

            groups[group_key].samples.append(result)

    # 5. 报告
    print(f"\n{'='*80}")
    print("ABLATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'A: Baseline':>15} {'B: +Experiences':>15} {'C: +Strong':>15}")
    print(f"{'-'*80}")

    for metric_name, attr in [
        ("Avg Final Score", "avg_score"),
        ("Avg Pass Rate", "avg_pass_rate"),
        ("Avg Prompt Length (chars)", "avg_len"),
        ("Std Prompt Length", "len_std"),
        ("Collapse Rate (<20 chars)", "collapse_rate"),
        ("Short Rate (<100 chars)", "short_rate"),
    ]:
        vals = [f"{getattr(g, attr):.2f}" if "rate" in attr else f"{getattr(g, attr):.1f}"
                for g in groups.values()]
        print(f"{metric_name:<30} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")

    a_score = groups["A_baseline"].avg_score
    b_score = groups["B_experiences"].avg_score
    c_score = groups["C_strong"].avg_score
    a_len = groups["A_baseline"].avg_len
    b_len = groups["B_experiences"].avg_len

    if b_score > a_score:
        print(f"✅ Experiences HELP: +{b_score - a_score:.2f} score improvement over baseline")
    else:
        print(f"❌ Experiences do NOT help in inference-only mode ({b_score:.2f} vs {a_score:.2f})")

    if b_len > a_len:
        print(f"✅ Experiences increase prompt length: {a_len:.0f} → {b_len:.0f} chars")
    else:
        print(f"⚠️  Experiences do NOT increase length: {a_len:.0f} → {b_len:.0f} chars")

    if groups["A_baseline"].collapse_rate > 0:
        print(f"⚠️  Baseline collapses in {groups['A_baseline'].collapse_rate*100:.0f}% of cases")

    # 保存结果
    output_path = "/data/zhuldz/lunwen/experience/outputs/ablation_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": {"n_samples": args.n_samples, "temperature": GEN_TEMP, "max_experiences": MAX_EXPERIENCES},
            "A_baseline": {"avg_score": groups["A_baseline"].avg_score, "avg_len": groups["A_baseline"].avg_len,
                          "samples": groups["A_baseline"].samples},
            "B_experiences": {"avg_score": groups["B_experiences"].avg_score, "avg_len": groups["B_experiences"].avg_len,
                             "samples": groups["B_experiences"].samples},
            "C_strong": {"avg_score": groups["C_strong"].avg_score, "avg_len": groups["C_strong"].avg_len,
                        "samples": groups["C_strong"].samples},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_ablation()
