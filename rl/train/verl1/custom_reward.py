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
# 1. 导入接口
# ===================================================================
B_MODEL = None
B_TOKENIZER = None
C_JUDGE = None
MODELS_LOADED = False

try:
    from verl1.b_interface import load_b_model, generate_code_from_prompt
    from verl1.c_interface import load_c_model, compute_reward as compute_c_reward
except Exception as e:
    def load_b_model(**kwargs): raise ImportError(f"b_interface import failed: {e}")
    def load_c_model(**kwargs): raise ImportError(f"c_interface import failed: {e}")
    def generate_code_from_prompt(**kwargs): return ["Error: Import failed"]
    def compute_c_reward(**kwargs): return 0.0

# 全局配置
ROLLOUT_DATA_DIR = "./a_model_grpo_standard/rollouts"
ITERATION_LOG_DIR = "./a_model_grpo_standard/iterations"
K_CODE_GEN = 2
ROLLOUT_N = 5 
CURRENT_ITERATION = 0
_CONFIG_INITIALIZED = False
SAMPLE_COUNT = 0

# [关键修改] 全局并发控制：设置为 60
# 即使有多个进程，总并发量也建议通过这里控制（如果是单机多卡，这个是进程内的限制；如果是多机，需要注意总和）
MAX_CONCURRENCY = 60 
api_semaphore = threading.Semaphore(MAX_CONCURRENCY)

# ===================================================================
# 2. C模型评判模板
# ===================================================================
C_MODEL_TEMPLATE = """你是专业代码评判模型（C模型），需要对生成代码和其对应的提示词进行评分。请严格按照以下规则评估：

【评估对象】
- 提示词（A模型输出）：{prompt}
- B模型输入（合并提示词）：{b_model_input}
- 生成代码（B模型输出）：{generated_code}
- 代码真值（参考标准）：{code_ground_truth}

【评估维度及权重】
1. 符合提示词度（30%）：生成代码是否严格遵循B模型输入的要求（功能、格式、约束等）
2. 代码质量（30%）：代码内容语法正确性、逻辑完整性、可读性（命名规范、注释等）
3. 功能一致性（40%）：与真值代码的核心功能是否一致（输入输出、处理逻辑）
4. 模型生成代码含有自然语言内容，在总分基础上扣两分。

【评分规则】
- 总分：-10 ~ +10（分数越高表示提示词效果越好，A模型应被奖励）
  - +7~+10：优秀（完全符合B模型输入，代码质量高，功能一致）
  - +3~+6：良好（基本符合B模型输入，少量问题，功能基本一致）
  - -2~+2：一般（部分符合B模型输入，有明显问题，功能有偏差）
  - -6~-3：较差（很少符合B模型输入，严重问题，功能偏差大）
  - -10~-7：极差（完全不符合B模型输入，无法运行，功能错误）
- 必须给出具体扣分/加分理由，禁止模糊评价

【输出格式】（严格按照JSON格式输出，键名不可修改）
{{
  "total_score": 具体分数（整数）,
  "match_prompt": 布尔值（true/false，是否符合B模型输入）,
  "score_details": {{
    "prompt_match_score": 维度1得分（-4~+4）,
    "code_quality_score": 维度2得分（-3~+3）,
    "function_consistency_score": 维度3得分（-3~+3）
  }},
  "reason": "具体评价理由（分点说明）"
}}"""

# ===================================================================
# 3. 核心清洗函数
# ===================================================================
def _clean_prompt_content(dirty_text: str) -> str:
    if not isinstance(dirty_text, str): return str(dirty_text)
    start_marker = "Original prompt:"
    end_marker = "Correct code:"
    start_idx = dirty_text.find(start_marker)
    end_idx = dirty_text.find(end_marker)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return dirty_text[start_idx + len(start_marker):end_idx].strip()
    return dirty_text

# ===================================================================
# 4. 模型懒加载
# ===================================================================
def _ensure_models_loaded():
    global B_MODEL, B_TOKENIZER, C_JUDGE, MODELS_LOADED
    if MODELS_LOADED: return True
    try:
        if "CUDA_VISIBLE_DEVICES" in os.environ and not os.environ["CUDA_VISIBLE_DEVICES"]:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        if B_MODEL is None: B_MODEL, B_TOKENIZER = load_b_model()
        if C_JUDGE is None: C_JUDGE = load_c_model()
        if B_MODEL is not None and C_JUDGE is not None:
            MODELS_LOADED = True
            return True
        return False
    except Exception as e:
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
# 5. 核心奖励函数 (并发优化版)
# ===================================================================
def compute_custom_reward(**kwargs):
    global SAMPLE_COUNT
    _ensure_models_loaded()
    
    prompts = kwargs.get('data_sources') or kwargs.get('prompts') or kwargs.get('input_ids') or kwargs.get('inputs')
    responses = kwargs.get('solution_strs') or kwargs.get('responses') or kwargs.get('predictions')
    
    # 提取真值
    ground_truths = None
    if 'canonical_solution' in kwargs: ground_truths = kwargs['canonical_solution']
    elif 'output' in kwargs: ground_truths = kwargs['output']
    elif 'ground_truth' in kwargs: ground_truths = kwargs['ground_truth']
    elif 'reward_model' in kwargs:
        rm = kwargs['reward_model']
        if isinstance(rm, dict):
            ground_truths = rm.get('ground_truth', rm.get('output'))
        elif isinstance(rm, (list, np.ndarray)) and len(rm) > 0 and isinstance(rm[0], dict):
             ground_truths = [item.get('ground_truth', item.get('output')) for item in rm]

    # 提取 raw_input
    raw_inputs = kwargs.get('raw_input')
    if raw_inputs is None and prompts is not None:
        raw_inputs = [_clean_prompt_content(str(p)) for p in prompts]

    if prompts is None or responses is None:
        return {"reward_tensor": torch.zeros(1), "reward_extra_info": {}}

    config = kwargs.get('config')
    _initialize_globals_from_config(config)

    # 格式化
    if isinstance(prompts, (np.ndarray, torch.Tensor)): prompts = prompts.tolist()
    if isinstance(raw_inputs, (np.ndarray, torch.Tensor)): raw_inputs = raw_inputs.tolist()
    if isinstance(responses, (np.ndarray, torch.Tensor)): responses = responses.tolist()
    if ground_truths is not None and isinstance(ground_truths, (np.ndarray, torch.Tensor)):
        ground_truths = ground_truths.tolist()
    
    if ground_truths is None: ground_truths = [None] * len(prompts)
    if raw_inputs is None: raw_inputs = [""] * len(prompts)

    n = 1
    if len(prompts) > 0 and len(responses) > len(prompts):
        n = len(responses) // len(prompts)
    
    # --- [修改] 准备任务列表 ---
    tasks = []
    response_index = 0
    
    # 为了并发安全，先分配好每个任务的参数
    for i, original_prompt_blob in enumerate(prompts):
        gt = ground_truths[i] if i < len(ground_truths) else None
        gt_str = str(gt) if gt else "N/A (Truth Not Found)"
        actual_user_query = raw_inputs[i] if i < len(raw_inputs) else str(original_prompt_blob)
        
        end_index = min(response_index + n, len(responses))
        system_prompts_from_policy = responses[response_index : end_index]
        response_index += n
        
        for system_prompt in system_prompts_from_policy:
            tasks.append({
                "original_prompt_blob": original_prompt_blob,
                "gt_str": gt_str,
                "actual_user_query": actual_user_query,
                "system_prompt": system_prompt
            })

    # --- [修改] 并发处理函数 ---
    def process_single_sample(task):
        global SAMPLE_COUNT
        # 增加计数 (注意: 多线程下这可能不准确，但只是log ID不太重要)
        # SAMPLE_COUNT += 1 
        
        system_prompt = task["system_prompt"]
        original_prompt_blob = task["original_prompt_blob"]
        actual_user_query = task["actual_user_query"]
        gt_str = task["gt_str"]

        # 1. 清洗 <think>
        raw_system_prompt = str(system_prompt)
        cleaned_system_prompt = re.sub(r'<think>.*?</think>', '', raw_system_prompt, flags=re.DOTALL).strip()
        if not cleaned_system_prompt:
            cleaned_system_prompt = "Generate python code based on the user request."

        # 2. 并发限制控制
        with api_semaphore:
            trace_entry = {
                "timestamp": datetime.now().isoformat(),
                "sample_id": 0, # 暂不精确追踪ID
                "iteration": CURRENT_ITERATION,
                "a_model": {
                    "input": str(original_prompt_blob), 
                    "raw_output": raw_system_prompt,
                    "output": cleaned_system_prompt
                },
                "b_model": [], 
                "c_model": {} 
            }
            avg_score = 0.0
            full_c_input_log = ""
            
            if MODELS_LOADED:
                try:
                    # A -> B
                    codes_dict_list = _generate_codes(cleaned_system_prompt, str(actual_user_query))
                    # B -> C
                    avg_score, _, full_results = _evaluate_codes(codes_dict_list, gt_str, str(actual_user_query))
                    
                    b_logs = []
                    for k_idx, code_item in enumerate(codes_dict_list):
                        b_input = str(code_item.get('b_model_input', ''))
                        gen_code = str(code_item.get('code', ''))
                        b_logs.append({"input": b_input, "output": gen_code})
                        if k_idx == 0:
                            full_c_input_log = C_MODEL_TEMPLATE.format(
                                prompt=str(cleaned_system_prompt),
                                b_model_input=b_input,
                                generated_code=gen_code,
                                code_ground_truth=gt_str 
                            )

                    trace_entry["b_model"] = b_logs
                    trace_entry["c_model"] = {
                        "input": full_c_input_log if full_c_input_log else "Error: No code generated",
                        "output": full_results, 
                        "avg_score": float(avg_score)
                    }
                except Exception as e:
                    print(f"❌ Pipeline Error: {e}")
                    trace_entry["b_model"] = [{"input": "Error", "output": f"Pipeline Error: {e}"}]
            else:
                trace_entry["b_model"] = [{"input": "Error", "output": "Models Not Loaded"}]
            
            return float(avg_score), json.dumps(trace_entry, ensure_ascii=False)

    # --- [修改] 使用线程池执行 ---
    all_scores = [0.0] * len(responses)
    all_traces = ["{}"] * len(responses)
    
    # 映射回原来的顺序
    # 这里的 tasks 列表顺序对应 all_scores 的顺序，因为是按顺序 append 的
    
    # 使用 ThreadPoolExecutor 提高并发
    # max_workers 设置为 60 或略高，受 semaphore 控制实际请求
    with ThreadPoolExecutor(max_workers=64) as executor:
        # 提交任务
        future_to_idx = {executor.submit(process_single_sample, task): i for i, task in enumerate(tasks)}
        
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
# 6. 辅助函数 (保持不变)
# ===================================================================
def _generate_codes(system_prompt: str, prompt: str) -> List[Dict]:
    codes = []
    if B_MODEL is None:
        return [{'code': "Error: B_MODEL is None", 'b_model_input': "", 'system_prompt': system_prompt}] * K_CODE_GEN

    try:
        sys_p = str(system_prompt) if system_prompt is not None else ""
        usr_p = str(prompt) if prompt is not None else ""
        # 注意：这里内部如果是调用本地模型，本身可能不支持多线程并发（CUDA流冲突）
        # 但如果是调用 API (b_model_api.py)，则支持多线程。
        # 根据你的文件列表，你使用了 b_model_api.py，所以这里多线程是有效的。
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
            # 调用 C 模型 API
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