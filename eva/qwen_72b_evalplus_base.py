
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
FILENAME = "samples_0206_72b_1.jsonl"
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

def align_function_name(generated_code: str, reference_content: str) -> str:
    """
    对比生成代码和参考内容(Prompt/GroundTruth)，如果函数名不一致，自动添加别名映射。
    """
    if not generated_code or not reference_content:
        return generated_code

    # 1. 从参考内容中寻找目标函数名 (Target Name)
    # 匹配 def FunctionName(
    gt_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    gt_match = re.search(gt_pattern, reference_content)
    
    if not gt_match:
        return generated_code # 参考代码里没找到函数定义，跳过
    
    target_name = gt_match.group(1)

    # 2. 检查生成的代码中是否已经正确定义了该函数，或者已经有了别名
    # 如果代码里已经有 def TargetName( 或 TargetName = ... 则不需要修复
    if re.search(rf"def\s+{target_name}\s*\(", generated_code) or \
       re.search(rf"{target_name}\s*=", generated_code):
        return generated_code

    # 3. 寻找生成代码里的主函数名 (Model Name)
    # 我们假设模型生成的最后一个顶层函数是主函数
    gen_matches = re.findall(gt_pattern, generated_code)
    
    if not gen_matches:
        return generated_code

    model_func_name = gen_matches[-1] # 取最后一个定义的函数

    # 4. 如果名字不同，添加别名
    if target_name != model_func_name:
        print(f"🔧 [Auto-Fix] Aligning function name: {target_name} = {model_func_name} (Task ID context)")
        return generated_code + f"\n\n# Alias for evaluation\n{target_name} = {model_func_name}\n"
    
    return generated_code

def get_args():
    parser = argparse.ArgumentParser(description="BModel EvalPlus Evaluation Script")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="mbpp", 
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
            # instruct_prompt = (
            #     f"```python\n{raw_prompt}\n```"
            # )
            # instruct_prompt = (
            #     f"System Prompt:  \nSystem  \nYou are Expert Coder tasked with generating code based on user queries without prior interaction. Always follow strict coding standards: PEP8, proper docstrings (with Args, Returns, Raises), and type hints. Prioritize code robustness (proper type checks, safe attribute access), performance (efficient resource management, no hard-coded limits), and modularity (no hard-coded paths, appropriate docstrings). Do not include explanations or solutions. For user requests, analyze requirements rigorously, enforce safety constraints (e.g., no insecure operations), and output only fully compliant code with comprehensive docstrings, type hints, and correct PEP8 formatting. No extra text—just code. No questions asked, direct output."
            #     f"```python\n{raw_prompt}\n```"
            # )
            # instruct_prompt = (
            #     f"System Prompt:  \nYou are an Expert Coder tasked with instructing the AI to generate accurate and well-structured code. The output must strictly adhere to the following principles:  \n1. **Clarity and Correctness**: The generated code must correctly solve the problem described, with no logical errors or missing edge-case handling.  \n2. **Code Quality**: The code must use proper Python syntax, meaningful variable names, and clean formatting. Avoid hardcoding magic numbers or bypassing normal control flow.  \n3. **Documentation**: Include a concise docstring explaining the function's purpose, input/output types, parameter descriptions, and key behavioral constraints. Avoid redundant explanations or unnecessary details.  \n4. **Modularity**: Keep the solution minimal and focused, avoiding unused imports, global variables, external dependencies, or commented-out lines. Avoid adding helper functions or contextual comments.  \n5. **Styling**: Use consistent Python formatting, correct indentation, and valid type hints. Avoid hard-to-read practices like missing newlines, incorrect spacing, or invalid escape sequences.  \n6. **Safety**: Do not add external explanations, comments, tests, examples, benchmarks, licenses, or other non-code elements. The output must be a fully executable standalone function.  \n\nAdhere strictly to these guidelines when generating code. Do not add extra justification, remarks, or placeholders. The final output must be a correctly formatted Python code snippet."
            #     f"```python\n{raw_prompt}\n```"
            # )
            # instruct_prompt = (
            #     f"Complete precise code without commentary"
            #     f"```python\n{raw_prompt}\n```"
            # )
            
            # 1. 定义你的系统提示词 (可以在循环外定义，也可以配置化)
            #my_system_prompt = "Complete precise code without commentary" 
            #my_system_prompt = "System Prompt:  \nYou are an Expert Coder tasked with instructing the AI to generate accurate and well-structured code. The output must strictly adhere to the following principles:  \n1. **Clarity and Correctness**: The generated code must correctly solve the problem described, with no logical errors or missing edge-case handling.  \n2. **Code Quality**: The code must use proper Python syntax, meaningful variable names, and clean formatting. Avoid hardcoding magic numbers or bypassing normal control flow.  \n3. **Documentation**: Include a concise docstring explaining the function's purpose, input/output types, parameter descriptions, and key behavioral constraints. Avoid redundant explanations or unnecessary details.  \n4. **Modularity**: Keep the solution minimal and focused, avoiding unused imports, global variables, external dependencies, or commented-out lines. Avoid adding helper functions or contextual comments.  \n5. **Styling**: Use consistent Python formatting, correct indentation, and valid type hints. Avoid hard-to-read practices like missing newlines, incorrect spacing, or invalid escape sequences.  \n6. **Safety**: Do not add external explanations, comments, tests, examples, benchmarks, licenses, or other non-code elements. The output must be a fully executable standalone function.  \n\nAdhere strictly to these guidelines when generating code. Do not add extra justification, remarks, or placeholders. The final output must be a correctly formatted Python code snippet." 
            #my_system_prompt = "Please complete the following Python function based on the provided docstring or description.Do not include any explanation, just the code." 
            my_system_prompt = "System Prompt:  \nSystem  \nYou are Expert Coder tasked with generating code based on user queries without prior interaction. Always follow strict coding standards: PEP8, proper docstrings (with Args, Returns, Raises), and type hints. Prioritize code robustness (proper type checks, safe attribute access), performance (efficient resource management, no hard-coded limits), and modularity (no hard-coded paths, appropriate docstrings). Do not include explanations or solutions. For user requests, analyze requirements rigorously, enforce safety constraints (e.g., no insecure operations), and output only fully compliant code with comprehensive docstrings, type hints, and correct PEP8 formatting. No extra text—just code. No questions asked, direct output."

            # 2. 构造纯净的用户提示词 (只包含题目本身)
            user_prompt = f"```python\n{raw_prompt}\n```"

            # 3. [关键修改] 调用时显式传入 system_prompt
            response = api.generate_code(
                user_query=user_prompt,       # 对应 BModelAPI 的 user_query
                system_prompt=my_system_prompt # 对应 BModelAPI 的 system_prompt
            )

            
            if not response:
                print(f"⚠️ Task {task_id} 生成失败/为空")
                clean_code = ""
            else:
                clean_code = extract_code(response)
            
                # =========================================================
                # [新增] 函数名自动对齐修复
                # =========================================================
                # 我们把题目中的 prompt (含函数头) 和 standard solution 拼在一起
                # 作为一个 reference，让 align_function_name 去里面找正确的函数名
                try:
                    ref_text = (
                        str(problem.get("prompt", "")) + "\n" + 
                        str(problem.get("canonical_solution", "")) + "\n" + 
                        str(problem.get("code", ""))
                    )
                    
                    # 调用你在文件头部定义好的 align_function_name
                    clean_code = align_function_name(clean_code, ref_text)
                    
                except Exception as e:
                    # 捕获异常，防止因为修复逻辑出错导致整个任务失败
                    print(f"⚠️ [Name Fix Error] Task {task_id}: {e}")
                # =========================================================

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