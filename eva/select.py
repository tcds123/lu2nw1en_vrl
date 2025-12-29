import pandas as pd
import json
import concurrent.futures
from tqdm import tqdm
import sys
import os
import re
import time

# ================= 路径配置区域 =================
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 计算项目根目录 (向上两级)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

# 构造依赖路径
generation_path = os.path.join(project_root, 'generation')

# 添加到系统路径
if generation_path not in sys.path:
    sys.path.append(generation_path)

print(f"🔧 当前工作目录: {current_dir}")
print(f"🔗 依赖路径:\n  - {generation_path}")

# ================= 模块导入区域 =================
try:
    from b_model_api import BModelAPI
    print("✅ 成功导入 BModelAPI")
except ImportError as e:
    print(f"❌ 导入 BModelAPI 失败: {e}")
    print("请确保 generation/b_model_api.py 存在。")
    sys.exit(1)

# ================= 参数配置 =================
INPUT_FILE = os.path.join(current_dir, "datajson/train_testexample.json") 
OUTPUT_FILE = os.path.join(current_dir, "datajson/filtered_hard_data.json")

MAX_WORKERS = 8       # 并发数
API_KEY = None        # 使用默认Key

# ================= 核心逻辑 =================

def extract_python_code(text):
    """
    专门用于提取 Python 代码的清洗函数
    替代原来的 clean_model_response
    """
    if not text: return ""
    
    # 1. 尝试提取 Markdown 代码块 ```python ... ```
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # 2. 尝试提取通用代码块 ``` ... ```
    pattern_generic = r"```(.*?)```"
    matches_gen = re.findall(pattern_generic, text, re.DOTALL)
    if matches_gen:
        return matches_gen[0].strip()
        
    # 3. 如果没有 Markdown 标记，尝试直接返回（假设纯代码）
    # 去除可能存在的 "Here is the code:" 前缀
    text = re.sub(r"^(Here is|This is|Below is).*?:\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()

def execute_code_with_tests(code, unit_tests_data):
    """
    执行代码并运行测试用例
    """
    try:
        if not code or not code.strip(): return 0.0
        
        # 解析测试用例
        unit_tests = []
        if isinstance(unit_tests_data, str):
            try:
                unit_tests = json.loads(unit_tests_data)
            except:
                if "assert" in unit_tests_data:
                    unit_tests = [unit_tests_data]
                else:
                    return 0.0
        elif isinstance(unit_tests_data, list):
            unit_tests = unit_tests_data
        
        if not unit_tests: return 0.0

        # 定义沙盒环境
        global_scope = {}
        try:
            exec(code, global_scope)
        except Exception:
            return 0.0 # 语法错误

        passed_count = 0
        total_tests = len(unit_tests)
        
        for test_case in unit_tests:
            try:
                exec(test_case, global_scope)
                passed_count += 1
            except Exception:
                pass
        
        return passed_count / total_tests if total_tests > 0 else 0.0
    except Exception:
        return 0.0

def process_row(api_client, row):
    """
    单行处理逻辑
    """
    try:
        prompt = row.get('input') or row.get('question') or row.get('prompt')
        unit_tests = row.get('unit_tests') or row.get('tests')
        
        if not prompt: return None

        # 1. 调用 B 模型 (裸跑)
        raw_code = api_client.generate_code(user_query=prompt, system_prompt=None)
        
        if not raw_code: return None

        # 2. 清洗代码 (使用本地定义的函数)
        cleaned_code = extract_python_code(raw_code)
        
        # 3. 运行测试
        pass_rate = execute_code_with_tests(cleaned_code, unit_tests)
        
        # 4. 筛选：Pass Rate < 1.0 的保留 (困难样本)
        if pass_rate < 1.0:
            row['b_baseline_pass_rate'] = pass_rate
            # 也可以选择把生成的错误代码存下来，后续可用于分析
            # row['b_failed_code'] = cleaned_code 
            return row
        else:
            return None # 简单样本，丢弃

    except Exception as e:
        return None

def load_data(file_path):
    print(f"📂 读取文件: {file_path}")
    try:
        df = pd.read_json(file_path)
        return df
    except ValueError:
        try:
            df = pd.read_json(file_path, lines=True)
            return df
        except Exception as e:
            print(f"❌ 读取 JSON 失败: {e}")
            return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    df = load_data(INPUT_FILE)
    if df is None or len(df) == 0:
        print("❌ 数据集为空或读取失败")
        return

    rows = df.to_dict(orient='records')
    print(f"📊 总样本数: {len(rows)}")
    
    print("🤖 初始化 B模型 API...")
    api_client = BModelAPI(api_key=API_KEY)
    
    hard_samples = []
    
    print(f"🚀 开始筛选 (并发数: {MAX_WORKERS})...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, api_client, row) for row in rows]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(rows)):
            result = future.result()
            if result is not None:
                hard_samples.append(result)

    total = len(rows)
    kept = len(hard_samples)
    dropped = total - kept
    
    print("\n" + "="*40)
    print(f"✅ 筛选完成")
    print(f"📥 总数: {total}")
    print(f"🗑️ 丢弃 (太简单): {dropped} ({dropped/total*100:.1f}%)")
    print(f"💾 保留 (困难): {kept} ({kept/total*100:.1f}%)")
    print("="*40)
    
    if hard_samples:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(hard_samples, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存至: {OUTPUT_FILE}")
        
        # 备份 Parquet
        parquet_file = OUTPUT_FILE.replace('.json', '.parquet')
        pd.DataFrame(hard_samples).to_parquet(parquet_file)
        print(f"💾 Parquet 备份: {parquet_file}")
    else:
        print("⚠️ 警告: 没有保留下任何样本。")

if __name__ == "__main__":
    main()