
import sys
import os
import re
import argparse
import subprocess
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl


OUTPUT_DIRS = {
    "humaneval": "/data/zhuldz/lunwen/eva/evalplus_results/humaneval/code_gen",
    "mbpp": "/data/zhuldz/lunwen/eva/evalplus_results/mbpp/code_gen"
}
# 定义保存的文件名 (在上述目录中)
FILENAME = "samples_b_72b_1.jsonl"
# ===========================================

# --- 1. 动态导入 generation 目录下的 BModelAPI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
generation_dir = os.path.join(os.path.dirname(current_dir), 'generation')
sys.path.append(generation_dir)

try:
    from b_model_api import BModelAPI
    print("✅ 成功导入 BModelAPI")
except ImportError:
    print(f"❌ 无法导入 BModelAPI，请确认 {generation_dir}/b_model_api.py 存在")
    exit(1)

def extract_code(text: str) -> str:
    """清洗 API 返回的 Markdown，提取纯代码"""
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1)
    
    pattern_generic = r"```\s*(.*?)\s*```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic: return match_generic.group(1)
    
    return text

def get_args():
    parser = argparse.ArgumentParser(description="BModel EvalPlus Evaluation Script")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="humaneval", 
        choices=["humaneval", "mbpp"],
        help="选择数据集: humaneval 或 mbpp"
    )
    return parser.parse_args()

def main():
    args = get_args()
    dataset_name = args.dataset
    
    save_dir = OUTPUT_DIRS.get(dataset_name)
    
    # 2. 确保目录存在 (如果不存在则自动创建)
    if not os.path.exists(save_dir):
        print(f"📁 目录不存在，正在创建: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        
    # 3. 拼接完整文件路径
    output_file = os.path.join(save_dir, FILENAME)
    
    print(f"🚀 启动 B模型 (Qwen-72B) EvalPlus 评估流程")
    print(f"🎯 目标数据集: {dataset_name}")
    print(f"📂 结果将保存至: {output_file}")
    
    if os.path.exists(output_file):
        print(f"\n✨ 发现已存在的生成结果，跳过生成步骤...")
    
    else:
        # --- 2. 初始化 API ---
        api = BModelAPI()
        
        # --- 3. 加载数据 ---
        print(f"📚 正在加载 {dataset_name} 数据集...")
        if dataset_name == "humaneval":
            problems = get_human_eval_plus()
        else:
            problems = get_mbpp_plus()
            
        samples = []
        print(f"🔄 开始生成 {len(problems)} 个任务的代码...")
        
        # --- 4. 遍历生成 ---
        for task_id, problem in tqdm(problems.items()):
            raw_prompt = problem["prompt"]
            
            # 构造 Prompt
            # instruct_prompt = (
            #     f"Please complete the following Python function based on the provided docstring or description.\n"
            #     f"Do not include any explanation, just the code.\n\n"
            #     f"```python\n{raw_prompt}\n```"
            # )
            instruct_prompt = (
                f"```python\n{raw_prompt}\n```"
            )
            # instruct_prompt = (
            #     f"System Prompt:  \nSystem  \nYou are Expert Coder tasked with generating code based on user queries without prior interaction. Always follow strict coding standards: PEP8, proper docstrings (with Args, Returns, Raises), and type hints. Prioritize code robustness (proper type checks, safe attribute access), performance (efficient resource management, no hard-coded limits), and modularity (no hard-coded paths, appropriate docstrings). Do not include explanations or solutions. For user requests, analyze requirements rigorously, enforce safety constraints (e.g., no insecure operations), and output only fully compliant code with comprehensive docstrings, type hints, and correct PEP8 formatting. No extra text—just code. No questions asked, direct output."
            #     f"```python\n{raw_prompt}\n```"
            # )
            
            response = api.generate_code(instruct_prompt)
            
            if not response:
                print(f"⚠️ Task {task_id} 生成失败/为空")
                clean_code = ""
            else:
                clean_code = extract_code(response)
            
            samples.append({
                "task_id": task_id,
                "completion": clean_code
            })

        # --- 5. 保存结果 ---
        write_jsonl(output_file, samples)
        print(f"💾 生成结果已保存至: {output_file}")
    
    # --- 6. 调用评估器 ---
    print("\n" + "="*40)
    print(f"🧪 开始运行 EvalPlus ({dataset_name}) 评分...")
    print("="*40)
    
    cmd = [
        "evalplus.evaluate",
        "--dataset", dataset_name,
        "--samples", output_file,   # 使用完整的绝对路径
        "--min-time-limit", "1",
        "--i-just-wanna-run" 
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ 评估命令执行出错，请检查环境或手动运行。")

if __name__ == "__main__":
    main()