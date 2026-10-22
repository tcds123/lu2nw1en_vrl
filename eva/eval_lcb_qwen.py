import sys
import os
import json
import re
import argparse
import subprocess
from tqdm import tqdm

# --- 配置路径 ---
# 这一步非常关键：将你上传的 LiveCodeBench 源码所在目录加入系统路径
# 假设 livecodebench 文件夹就在当前目录的上一级或同级，请根据实际情况修改
# 目标是指向包含 'lcb_runner' 文件夹的那个父目录
LCB_SOURCE_ROOT = "/data/zhuldz/lunwen/eva/LiveCodeBench"  # 请修改为实际绝对路径
sys.path.append(LCB_SOURCE_ROOT)

# 尝试导入 datasets
try:
    from datasets import load_dataset
except ImportError:
    print("❌ 需要安装 datasets 库: pip install datasets")
    exit(1)

# --- 1. 动态导入 BModelAPI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
generation_dir = os.path.join(os.path.dirname(current_dir), 'generation')
sys.path.append(generation_dir)

try:
    from b_model_api import BModelAPI
    print("✅ 成功导入 BModelAPI")
except ImportError:
    print(f"❌ 无法导入 BModelAPI，请确认 {generation_dir}/b_model_api.py 存在")
    exit(1)

# 定义输出路径
OUTPUT_DIR = "/data/zhuldz/lunwen/eva/results/livecodebench/code_gen"
FILENAME = "samples_b_72b_lcb.json" # 注意：后缀是 .json 不是 .jsonl

def extract_code(text: str) -> str:
    """提取 Markdown 中的代码块"""
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1)
    
    pattern_generic = r"```\s*(.*?)\s*```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic: return match_generic.group(1)
    
    return text

def main():
    # 1. 准备环境
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, FILENAME)

    # 2. 生成代码 (如果文件不存在)
    if os.path.exists(output_path):
        print(f"✨ 发现已有结果文件 {output_path}，跳过生成，直接评测...")
    else:
        print("🚀 开始 LiveCodeBench 代码生成...")
        api = BModelAPI()
        
        # 加载 LCB 数据集 (使用 HuggingFace)
        # release_date 用于过滤题目，test split 包含比赛题
        print("📚 加载 livecodebench/lcb_v1 数据集...")
        ds = load_dataset("livecodebench/lcb_v1", split="test")
        
        results_for_eval = []
        
        for problem in tqdm(ds, desc="Generating"):
            question_id = problem["question_id"]
            
            # LCB 的题目通常包含 content (描述) 和 sample_io
            # 构造一个类似竞赛的 Prompt
            prompt_content = problem.get("question_content", "") or problem.get("content", "")
            
            # 构造 Prompt
            instruct_prompt = (
                f"You are an expert programmer participating in a coding competition.\n"
                f"Solve the following problem using Python.\n\n"
                f"Problem:\n{prompt_content}\n\n"
                f"Write the solution inside a Python code block.\n"
                f"```python\n"
            )
            
            # 调用模型
            response = api.generate_code(instruct_prompt)
            
            if response:
                clean_code = extract_code(response)
            else:
                clean_code = ""
            
            # --- 关键修改：构造符合 custom_evaluator 要求的字典格式 ---
            # 必须包含 question_id 和 code_list (list[str])
            results_for_eval.append({
                "question_id": question_id,
                "code_list": [clean_code] 
            })
            
        # 3. 保存为标准 JSON Array (不是 JSONL)
        print(f"💾 保存结果到: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_for_eval, f, indent=4)

    # 4. 调用你本地的 custom_evaluator.py 进行评测
    print("\n" + "="*40)
    print("🧪 开始调用本地 LiveCodeBench 评测脚本...")
    print("="*40)
    
    # 定位 custom_evaluator 的路径
    evaluator_script = os.path.join(LCB_SOURCE_ROOT, "lcb_runner", "runner", "custom_evaluator.py")
    
    if not os.path.exists(evaluator_script):
        print(f"❌ 找不到评测脚本: {evaluator_script}")
        print("请检查 LCB_SOURCE_ROOT 配置是否正确指向了包含 lcb_runner 的父目录。")
        return

    # 构造评测命令
    # 对应 custom_evaluator.py 中的 args
    cmd = [
        "python", evaluator_script,
        "--custom_output_file", output_path,
        "--scenario", "codegeneration",  # 对应 scenarios.py 中的 Enum
        "--evaluate" # 确保触发 get_metrics
    ]
    
    # 设置 PYTHONPATH 环境变量，确保子进程能找到 lcb_runner
    env = os.environ.copy()
    env["PYTHONPATH"] = LCB_SOURCE_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"\n✅ 评测完成！结果文件生成在: {output_path.replace('.json', '_eval_all.json')}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 评测过程中出错: {e}")
        print("请检查是否安装了必要的依赖 (如: pip install scipy numpy)")

if __name__ == "__main__":
    main()