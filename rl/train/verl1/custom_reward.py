import os
import sys
import json
import logging
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import traceback
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================================================================
# 0. 路径修复
# ===================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ===================================================================
# 1. 导入接口 (修复 NameError 问题)
# ===================================================================
B_MODEL = None
B_TOKENIZER = None
C_JUDGE = None
MODELS_LOADED = False

try:
    from verl1.b_interface import load_b_model, generate_code_from_prompt
    from verl1.c_interface import load_c_model, compute_reward as compute_c_reward
except Exception as e:
    # [关键修复] Python 3 会在 except 块结束后删除 e，必须先转存为字符串
    err_msg = str(e)
    print(f"⚠️ [CustomReward] Import Failed: {err_msg}") # 立即打印，方便调试
    
    def load_b_model(**kwargs): raise ImportError(f"b_interface import failed: {err_msg}")
    def load_c_model(**kwargs): raise ImportError(f"c_interface import failed: {err_msg}")
    def generate_code_from_prompt(**kwargs): return ["Error: Import failed"]
    def compute_c_reward(**kwargs): return 0.0, {}

# 全局配置
ROLLOUT_DATA_DIR = "./a_model_grpo_standard/rollouts"
ITERATION_LOG_DIR = "./a_model_grpo_standard/iterations"
K_CODE_GEN = 2
ROLLOUT_N = 5 
CURRENT_ITERATION = 0
_CONFIG_INITIALIZED = False
SAMPLE_COUNT = 0


MAX_CONCURRENCY = 55  
api_semaphore = threading.Semaphore(MAX_CONCURRENCY)

# ===================================================================
# 2. 动态加载 C 模型模板
# ===================================================================
C_PROMPT_FILE_PATH = "/data/zhuldz/lunwen/judge/c_model_prompt.txt"

def _get_c_model_template():
    """每次调用时从文件读取最新的 Prompt 模板，支持热修改"""
    try:
        if os.path.exists(C_PROMPT_FILE_PATH):
            with open(C_PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            # 默认模板作为兜底
            return "Error: c_model_prompt.txt not found. Please check path."
    except Exception as e:
        return f"Error reading prompt file: {e}"

# ===================================================================
# 3. 核心清洗函数
# ===================================================================
def _clean_prompt_content(dirty_text: str) -> str:
    """
    从包含ICL示例和指令的复杂文本中，提取出原始的用户问题。
    """
    if not isinstance(dirty_text, str): return str(dirty_text)
    
    # 标记必须与 convert_data.py 中的模板一致
    start_marker = "Original prompt:"
    end_marker = "Correct code:"
    
    start_idx = dirty_text.find(start_marker)
    end_idx = dirty_text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # 提取中间部分并去除空白
        return dirty_text[start_idx + len(start_marker):end_idx].strip()
    
    return dirty_text

# ===================================================================
# 4. 模型懒加载
# ===================================================================
def _ensure_models_loaded():
    global B_MODEL, B_TOKENIZER, C_JUDGE, MODELS_LOADED
    if MODELS_LOADED: return True
    try:
        # [关键] 多卡环境下 Ray 会屏蔽 CUDA_VISIBLE_DEVICES，导致子进程看不到卡
        if "CUDA_VISIBLE_DEVICES" in os.environ and not os.environ["CUDA_VISIBLE_DEVICES"]:
            del os.environ["CUDA_VISIBLE_DEVICES"]
            
        if B_MODEL is None: B_MODEL, B_TOKENIZER = load_b_model()
        if C_JUDGE is None: C_JUDGE = load_c_model()
        
        if B_MODEL is not None and C_JUDGE is not None:
            MODELS_LOADED = True
            return True
        return False
    except Exception as e:
        # 这里捕获的是 load_b_model 抛出的 ImportError
        print(f"❌ [CustomReward] Model Init Failed: {e}")
        traceback.print_exc()
        return False

def _initialize_globals_from_config(config):
    global _CONFIG_INITIALIZED, K_CODE_GEN, ROLLOUT_DATA_DIR, CURRENT_ITERATION
    if _CONFIG_INITIALIZED: return
    if config:
        CURRENT_ITERATION = config.get('current_iteration', 0) if isinstance(config, dict) else getattr(config, 'current_iteration', 0)
        if hasattr(config, 'iterative_rl'):
            K_CODE_GEN = config.iterative_rl.code_generation_count or 2
            ROLLOUT_DATA_DIR = config.iterative_rl.rollout_data_dir
        elif isinstance(config, dict) and 'iterative_rl' in config:
            K_CODE_GEN = config['iterative_rl'].get('code_generation_count', 2)
            ROLLOUT_DATA_DIR = config['iterative_rl'].get('rollout_data_dir', ROLLOUT_DATA_DIR)
    os.makedirs(ROLLOUT_DATA_DIR, exist_ok=True)
    _CONFIG_INITIALIZED = True

# ===================================================================
# 5. 核心奖励函数
# ===================================================================
def compute_custom_reward(**kwargs):
    global SAMPLE_COUNT
    _ensure_models_loaded()
    
    # --- 1. 提取 Prompts (A输入) ---
    prompts = kwargs.get('data_sources')
    if prompts is None: prompts = kwargs.get('prompts')
    if prompts is None: prompts = kwargs.get('input_ids')
    if prompts is None: prompts = kwargs.get('inputs')
        
    # --- 2. 提取 Responses (A输出) ---
    responses = kwargs.get('solution_strs')
    if responses is None: responses = kwargs.get('responses')
    if responses is None: responses = kwargs.get('predictions')
    
    # --- 3. 提取真值 (兼容多种 Key) ---
    ground_truths = None
    if 'canonical_solution' in kwargs: ground_truths = kwargs['canonical_solution']
    elif 'output' in kwargs: ground_truths = kwargs['output']
    elif 'ground_truth' in kwargs: ground_truths = kwargs['ground_truth']
    elif 'ground_truths' in kwargs: ground_truths = kwargs['ground_truths']
    elif 'references' in kwargs: ground_truths = kwargs['references']
    
    if ground_truths is None and 'reward_model' in kwargs:
        rm = kwargs['reward_model']
        if isinstance(rm, dict):
            ground_truths = rm.get('ground_truth', rm.get('output'))
        elif isinstance(rm, (list, np.ndarray)) and len(rm) > 0 and isinstance(rm[0], dict):
             ground_truths = [item.get('ground_truth', item.get('output')) for item in rm]

    # --- 4. 提取 raw_input (B输入) ---
    raw_inputs = kwargs.get('raw_input')

    # 快速失败
    if prompts is None or responses is None:
        return {"reward_tensor": torch.zeros(1), "reward_extra_info": {}}

    config = kwargs.get('config')
    _initialize_globals_from_config(config)

    # 格式化列表
    if isinstance(prompts, (np.ndarray, torch.Tensor)): prompts = prompts.tolist()
    if isinstance(raw_inputs, (np.ndarray, torch.Tensor)): raw_inputs = raw_inputs.tolist()
    if isinstance(responses, (np.ndarray, torch.Tensor)): responses = responses.tolist()
    if ground_truths is not None and isinstance(ground_truths, (np.ndarray, torch.Tensor)):
        ground_truths = ground_truths.tolist()
    
    # 确保长度匹配
    if ground_truths is None: ground_truths = [None] * len(prompts)
    
    # --- [关键逻辑] 计算 N_Ratio (数据对齐) ---
    num_responses = len(responses)
    num_prompts = len(prompts)
    
    n_ratio = 1
    if num_prompts > 0 and num_responses > num_prompts:
        n_ratio = num_responses // num_prompts
    
    # 准备任务列表 (Flattened)
    tasks = []
    current_template = _get_c_model_template()

    for i in range(num_responses):
        # 回溯索引
        p_idx = i if num_prompts == num_responses else i // n_ratio
        
        # 获取对应的数据
        if p_idx < len(prompts):
            original_prompt_blob = prompts[p_idx]
        else:
            original_prompt_blob = "Error: Prompt index out of bounds"

        # 获取真值
        if ground_truths and p_idx < len(ground_truths):
            gt = ground_truths[p_idx]
        else:
            gt = None
        gt_str = str(gt) if gt else "N/A"

        # 获取/清洗原始输入
        if raw_inputs and p_idx < len(raw_inputs):
            actual_user_query = str(raw_inputs[p_idx])
        else:
            actual_user_query = _clean_prompt_content(str(original_prompt_blob))
            
        system_prompt = responses[i]

        tasks.append({
            "index": i,
            "original_prompt_blob": original_prompt_blob,
            "gt_str": gt_str,
            "actual_user_query": actual_user_query,
            "system_prompt": system_prompt,
            "template": current_template
        })

    # --- 并发处理函数 ---
    def process_single_sample(task):
        sys_p = task["system_prompt"]
        query = task["actual_user_query"]
        gt = task["gt_str"]
        tmpl = task["template"]
        
        raw_sys_p = str(sys_p)
        clean_sys_p = re.sub(r'<think>.*?</think>', '', raw_sys_p, flags=re.DOTALL).strip()
        if not clean_sys_p: clean_sys_p = "Generate python code."

        with api_semaphore:
            trace_entry = {
                "timestamp": datetime.now().isoformat(),
                "iteration": CURRENT_ITERATION,
                "a_model": {
                    "input": str(task["original_prompt_blob"]), 
                    "output": clean_sys_p
                },
                "b_model": [], 
                "c_model": {} 
            }
            avg_score = 0.0
            
            if MODELS_LOADED:
                try:
                    # A -> B
                    codes_dict_list = _generate_codes(clean_sys_p, query)
                    
                    # B -> C
                    avg_score, _, full_results = _evaluate_codes(codes_dict_list, gt, query)
                    
                    b_logs = []
                    c_log_input = ""
                    
                    for k, code_item in enumerate(codes_dict_list):
                        b_in = str(code_item.get('b_model_input', ''))
                        b_out = str(code_item.get('code', ''))
                        b_logs.append({"input": b_in, "output": b_out})
                        
                        if k == 0:
                            c_log_input = tmpl.format(
                                prompt=str(clean_sys_p),
                                b_model_input=b_in,
                                generated_code=b_out,
                                code_ground_truth=gt 
                            )

                    trace_entry["b_model"] = b_logs
                    trace_entry["c_model"] = {
                        "input": c_log_input,
                        "output": full_results,
                        "avg_score": float(avg_score)
                    }
                except Exception as e:
                    print(f"❌ Pipeline Error: {e}")
                    trace_entry["b_model"] = [{"input": "Error", "output": f"Pipeline Error: {e}"}]
            else:
                trace_entry["b_model"] = [{"input": "Error", "output": "Models Not Loaded"}]
            
            return float(avg_score), json.dumps(trace_entry, ensure_ascii=False)

    # --- 执行线程池 ---
    all_scores = [0.0] * len(responses)
    all_traces = ["{}"] * len(responses)
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY + 4) as executor:
        future_to_idx = {executor.submit(process_single_sample, task): task["index"] for task in tasks}
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                score, trace = future.result()
                all_scores[idx] = score
                all_traces[idx] = trace
            except Exception as e:
                print(f"Task {idx} failed: {e}")

    return {
        "reward_tensor": torch.tensor(all_scores, dtype=torch.float32),
        "reward_extra_info": {"abc_trace": all_traces}
    }

# ===================================================================
# 6. 辅助函数
# ===================================================================
def _generate_codes(system_prompt: str, prompt: str) -> List[Dict]:
    codes = []
    if B_MODEL is None:
        return [{'code': "Error: B_MODEL is None", 'b_model_input': "", 'system_prompt': system_prompt}] * K_CODE_GEN

    try:
        sys_p = str(system_prompt) if system_prompt is not None else ""
        usr_p = str(prompt) if prompt is not None else ""
        generated_codes = generate_code_from_prompt(
            B_MODEL, B_TOKENIZER, generated_prompt=sys_p, original_prompt=usr_p, k=K_CODE_GEN
        )
    except Exception as e:
        print(f"Error in Model B: {e}")
        generated_codes = [f"Error: {e}"] * K_CODE_GEN
        
    for code in generated_codes:
        b_input = f"{system_prompt}\n{prompt}"
        codes.append({
            'code': code, 
            'b_model_input': b_input, 
            'system_prompt': system_prompt 
        })
    return codes
    
def _evaluate_codes(codes: List[Dict], ground_truth: str, original_prompt: str) -> tuple:
    scores = []
    full_results = []
    
    if C_JUDGE is None: return 0.0, [0.0]*len(codes), []
    if not codes: return 0.0, [], []
    
    gt_str = str(ground_truth) if ground_truth is not None else ""

    for code_item in codes:
        try:
            score_val, full_res = compute_c_reward(
                c_judge=C_JUDGE,
                generated_code=code_item['code'],
                canonical_solution=gt_str,
                generated_prompt=code_item.get('system_prompt', ''), 
                b_model_input=code_item['b_model_input']
            )
        except Exception as e: 
            print(f"Error in Model C: {e}")
            score_val = 0.0
            full_res = {"error": str(e)}
            
        scores.append(score_val)
        full_results.append(full_res)
    
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, scores, full_results