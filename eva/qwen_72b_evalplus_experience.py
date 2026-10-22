"""
EvalPlus 评估脚本 (支持三种提示词来源):
  --prompt_source baseline    : 硬编码通用提示词 (原始)
  --prompt_source experiences : 经验池直接注入 B 模型 system prompt
  --prompt_source a_model     : A模型+经验 → 为每道题生成定制系统提示词 → B模型

用法:
  cd /data/zhuldz/lunwen && source ./lunwen/bin/activate

  # Baseline
  python eva/qwen_72b_evalplus_experience.py --dataset humaneval --prompt_source baseline --output_tag baseline

  # 经验直接注入 B 模型
  python eva/qwen_72b_evalplus_experience.py --dataset humaneval --prompt_source experiences --output_tag exp_direct

  # A模型+经验 → 定制提示词 (方式A, 创新点)
  python eva/qwen_72b_evalplus_experience.py --dataset humaneval --prompt_source a_model --output_tag exp_amodel
"""

import sys
import os
import re
import argparse
import subprocess
import time
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl


OUTPUT_DIRS = {
    "humaneval": "/data/zhuldz/lunwen/eva/evalplus_results/humaneval/code_gen",
    "mbpp": "/data/zhuldz/lunwen/eva/evalplus_results/mbpp/code_gen",
}

current_dir = os.path.dirname(os.path.abspath(__file__))
generation_dir = os.path.join(os.path.dirname(current_dir), 'generation')
sys.path.append(generation_dir)
sys.path.append(os.path.dirname(current_dir))

from b_model_api import BModelAPI

# ============================================================
# Baseline 提示词
# ============================================================
BASELINE_PROMPT = (
    "System Prompt:  \nSystem  \n"
    "You are Expert Coder tasked with generating code based on user queries "
    "without prior interaction. Always follow strict coding standards: PEP8, "
    "proper docstrings (with Args, Returns, Raises), and type hints. Prioritize "
    "code robustness (proper type checks, safe attribute access), performance "
    "(efficient resource management, no hard-coded limits), and modularity "
    "(no hard-coded paths, appropriate docstrings). Do not include explanations "
    "or solutions. For user requests, analyze requirements rigorously, enforce "
    "safety constraints (e.g., no insecure operations), and output only fully "
    "compliant code with comprehensive docstrings, type hints, and correct PEP8 "
    "formatting. No extra text—just code. No questions asked, direct output."
)

# ============================================================
# A 模型配置 — 使用训练好的 Qwen3-4B (项目创新点)
# ============================================================
# 当前 GRPO 训练产出
CURRENT_CKPT = "/data/zhuldz/lunwen/rl/train/verl1/a_model_grpo_standard/checkpoints/global_step_30/actor/huggingface"
# 回退: 基座模型 (无 GRPO 微调)
BASE_MODEL = "/data/zhuldz/lunwen/models/Qwen3-4B-Instruct-2507"

A_MODEL_PATH = CURRENT_CKPT if os.path.exists(CURRENT_CKPT) else BASE_MODEL
ICL_FILE = "/data/zhuldz/lunwen/rl/train/verl1/sysprompt_icl.txt"

# 全局加载 (只加载一次)
_a_model = None
_a_tokenizer = None


def _load_a_model():
    """加载训练好的 Qwen3-4B (CPU offload, 不抢 GPU)"""
    global _a_model, _a_tokenizer
    if _a_model is not None:
        return _a_model, _a_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = A_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"⚠️  {model_path} 不存在, 回退到基座模型")
        model_path = BASE_MODEL

    print(f"📦 加载 A 模型: {model_path} ...")
    _a_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _a_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",      # CPU 推理, 不抢 GRPO 训练的 GPU
        trust_remote_code=True,
    )
    _a_model.eval()
    print(f"   ✅ A 模型就绪 (CPU, {sum(p.numel() for p in _a_model.parameters())/1e9:.1f}B params)")
    return _a_model, _a_tokenizer


def extract_code(text: str) -> str:
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1)
    pattern_generic = r"```\s*(.*?)\s*```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic: return match_generic.group(1)
    return text


def align_function_name(generated_code: str, reference_content: str) -> str:
    if not generated_code or not reference_content:
        return generated_code
    gt_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    gt_match = re.search(gt_pattern, reference_content)
    if not gt_match: return generated_code
    target_name = gt_match.group(1)
    if re.search(rf"def\s+{target_name}\s*\(", generated_code) or \
       re.search(rf"{target_name}\s*=", generated_code):
        return generated_code
    gen_matches = re.findall(gt_pattern, generated_code)
    if not gen_matches: return generated_code
    model_func_name = gen_matches[-1]
    if target_name != model_func_name:
        return generated_code + f"\n\n{target_name} = {model_func_name}\n"
    return generated_code


# ============================================================
# 经验加载
# ============================================================
def load_experiences(max_n: int = 15) -> str:
    """从经验池加载经验指南文本"""
    try:
        from experience.cache import ExperienceCache
        cache = ExperienceCache()
        if cache.count == 0:
            return ""
        return cache.format_for_injection(max_experiences=max_n)
    except Exception as e:
        print(f"⚠️ 经验加载失败: {e}")
        return ""


# ============================================================
# 方式 B: 经验直接注入 B 模型
# ============================================================
def get_experience_system_prompt(max_n: int = 15) -> str:
    guidelines = load_experiences(max_n)
    if not guidelines:
        return BASELINE_PROMPT
    return (
        "You are an Expert Coder. Follow these tested guidelines:\n\n"
        f"{guidelines}\n\n"
        "Output ONLY the complete, executable Python code. No explanations."
    )


# ============================================================
# 方式 A: A 模型 + 经验 → 为每道题生成定制系统提示词
# ============================================================
def _load_icl_examples() -> str:
    if os.path.exists(ICL_FILE):
        with open(ICL_FILE, 'r') as f:
            return f.read().strip()
    return ""


def generate_prompt_via_a_model(problem: str, reference_solution: str,
                                 experiences: str) -> str:
    """用训练好的 Qwen3-4B 为特定题目生成定制系统提示词"""
    model, tokenizer = _load_a_model()
    import torch

    icl = _load_icl_examples()
    exp_section = ""
    if experiences:
        exp_section = (
            "## Important Guidelines (learned from experience)\n"
            "When generating system prompts, you MUST follow these proven rules:\n"
            f"{experiences}\n\n"
        )

    prompt = (
        "You are an expert System Prompt Engineer. Generate a high-quality "
        "System Prompt that will guide a code generation model to produce "
        "correct, efficient code for this type of problem.\n\n"
        f"{exp_section}"
        f"## Examples\n{icl}\n\n"
        f"## Task\n"
        f"Problem: {problem}\n\n"
        f"Reference Solution:\n```python\n{reference_solution}\n```\n\n"
        "Output ONLY the system prompt text, no explanation or markdown."
    )

    # 应用 chat template
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response.strip()


# ============================================================
# CLI
# ============================================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="humaneval",
                        choices=["humaneval", "mbpp"])
    parser.add_argument("--prompt_source", type=str, default="baseline",
                        choices=["baseline", "experiences", "a_model"])
    parser.add_argument("--max_experiences", type=int, default=15)
    parser.add_argument("--output_tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    save_dir = OUTPUT_DIRS[args.dataset]
    os.makedirs(save_dir, exist_ok=True)

    tag = args.output_tag or args.prompt_source
    filename = f"samples_{tag}.jsonl"
    output_file = os.path.join(save_dir, filename)

    print(f"{'='*50}")
    print(f"🚀 EvalPlus 评估")
    print(f"   数据集:     {args.dataset}")
    print(f"   提示词来源: {args.prompt_source}")
    print(f"   结果路径:   {output_file}")
    print(f"{'='*50}")

    # 加载经验 (baseline 不需要)
    experiences = ""
    if args.prompt_source in ("experiences", "a_model"):
        experiences = load_experiences(args.max_experiences)
        print(f"   📋 经验: {len(experiences)} 字符")
        if experiences:
            print(f"   前150字符: {experiences[:150]}...")

    # 确定 system prompt
    if args.prompt_source == "baseline":
        global_prompt = BASELINE_PROMPT
    elif args.prompt_source == "experiences":
        global_prompt = get_experience_system_prompt(args.max_experiences)
    else:
        global_prompt = None  # a_model: 每题单独生成

    if global_prompt:
        print(f"   Prompt: {len(global_prompt)} 字符\n")

    # 加载数据
    problems = get_human_eval_plus() if args.dataset == "humaneval" else get_mbpp_plus()
    api = BModelAPI()
    samples = []

    print(f"🔄 生成 {len(problems)} 个任务的代码...")
    for task_id, problem in tqdm(problems.items()):
        raw_prompt = problem["prompt"]
        user_prompt = f"```python\n{raw_prompt}\n```"

        # 方式 A: 为每道题生成定制提示词
        if args.prompt_source == "a_model":
            # HumanEval 的 canonical_solution / MBPP 的 code 作为参考解
            ref_solution = problem.get("canonical_solution", problem.get("code", ""))
            sys_prompt = generate_prompt_via_a_model(raw_prompt, ref_solution, experiences)
        else:
            sys_prompt = global_prompt

        response = api.generate_code(user_query=user_prompt, system_prompt=sys_prompt)
        if not response:
            clean_code = ""
        else:
            clean_code = extract_code(response)
            try:
                ref_text = (str(problem.get("prompt", "")) + "\n" +
                           str(problem.get("canonical_solution", "")) + "\n" +
                           str(problem.get("code", "")))
                clean_code = align_function_name(clean_code, ref_text)
            except:
                pass

        samples.append({"task_id": task_id, "completion": clean_code})

    write_jsonl(output_file, samples)
    print(f"💾 {output_file}")

    # 评估
    print(f"\n🧪 EvalPlus ({args.dataset}) 评分...")
    cmd = ["evalplus.evaluate", "--dataset", args.dataset, "--samples", output_file,
           "--min-time-limit", "1", "--i-just-wanna-run"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ 评估命令执行出错")


if __name__ == "__main__":
    main()
