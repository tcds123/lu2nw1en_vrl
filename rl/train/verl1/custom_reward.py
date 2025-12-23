import os
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd  # [新增] 引入 pandas 读取 parquet
from datetime import datetime
from typing import Dict, List, Any, Tuple
import traceback
import re
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================================================================
# 0. 路径修复 (保持原样)
# ===================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ===================================================================
# 1. 导入接口 (保持原样)
# ===================================================================
B_MODEL = None
B_TOKENIZER = None
C_JUDGE = None
MODELS_LOADED = False

try:
    from verl1.b_interface import load_b_model, generate_code_from_prompt
    from verl1.c_interface import load_c_model, compute_reward as compute_c_reward
except Exception as e:
    err_msg = str(e)
    print(f"⚠️ [CustomReward] Import Failed: {err_msg}")
    
    def load_b_model(**kwargs): raise ImportError(f"b_interface import failed: {err_msg}")
    def load_c_model(**kwargs): raise ImportError(f"c_interface import failed: {err_msg}")
    def generate_code_from_prompt(**kwargs): return ["Error: Import failed"]
    def compute_c_reward(**kwargs): return 0.0, {}

# 全局配置 (保持原样)
ROLLOUT_DATA_DIR = "./a_model_grpo_standard/rollouts"
ITERATION_LOG_DIR = "./a_model_grpo_standard/iterations"
K_CODE_GEN = 1  
ROLLOUT_N = 5 
CURRENT_ITERATION = 0
_CONFIG_INITIALIZED = False
SAMPLE_COUNT = 0

MAX_CONCURRENCY = 10  
api_semaphore = threading.Semaphore(MAX_CONCURRENCY)

# ===================================================================
# [核心配置修改] 指向 Parquet 数据
# ===================================================================
# 优先从环境变量读取，如果没有则使用默认路径
TRAIN_SET_PATH = os.environ.get("VERL_DATA_PATH", "/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_cleaned_2.parquet")

# 组的大小 (1 Center + 2 Aux)
GROUP_SIZE = 3

FAIL_FAST_THRESHOLD = -5.0 

# 全量数据池与索引 (替代原本的 GENERAL_TEST_SET)
TRAINING_POOL_LIST: List[Tuple[str, str]] = [] 
PROMPT_TO_INDEX: Dict[str, int] = {}

USER_OVERLAP_WEIGHT = 15.0       
GT_LEAKAGE_THRESHOLD = 0.15      
MIN_REWARD_FLOOR = -15.0  

# ===================================================================
# 1.4 加载全量训练数据 (修复索引匹配问题)
# ===================================================================
def _load_training_pool():
    global TRAINING_POOL_LIST, PROMPT_TO_INDEX
    if TRAINING_POOL_LIST: return

    print(f"⏳ [CustomReward] 正在加载全量训练数据集(Parquet): {TRAIN_SET_PATH} ...")
    try:
        if os.path.exists(TRAIN_SET_PATH):
            TRAINING_POOL_LIST = []
            PROMPT_TO_INDEX = {}
            
            # 使用 pandas 读取 parquet
            df = pd.read_parquet(TRAIN_SET_PATH)
            
            # 自动探测列名
            cols = df.columns
            # [关键] 优先找 raw_input，因为它是不带 Template 的纯净 Query
            prompt_col = next((c for c in ['raw_input', 'input', 'instruction', 'prompt', 'question', 'content'] if c in cols), None)
            gt_col = next((c for c in ['output', 'response', 'target', 'solution', 'code', 'answer', 'canonical_solution'] if c in cols), None)
            
            if not prompt_col or not gt_col:
                print(f"⚠️ [CustomReward] 无法自动识别列名。现有列: {cols}")
                # 尝试强制指定，防止报错
                prompt_col = prompt_col or cols[0]
                gt_col = gt_col or cols[1]
                print(f"⚠️ [CustomReward] 强制使用: Prompt='{prompt_col}', GT='{gt_col}'")
            else:
                print(f"✅ [CustomReward] 使用列名: Prompt='{prompt_col}', GT='{gt_col}'")

            # 遍历加载
            for _, row in df.iterrows():
                try:
                    q = str(row[prompt_col]).strip()
                    gt = str(row[gt_col]).strip()
                    
                    if q and gt:
                        # 存入列表
                        current_list_idx = len(TRAINING_POOL_LIST)
                        TRAINING_POOL_LIST.append((q, gt))
                        # 建立映射: Query -> 全局List索引 (用于快速查找 Center 的位置)
                        PROMPT_TO_INDEX[q] = current_list_idx
                except Exception:
                    continue
                        
            print(f"✅ [CustomReward] 数据集加载完成，共 {len(TRAINING_POOL_LIST)} 条。")
        else:
            print(f"❌ [CustomReward] 文件未找到: {TRAIN_SET_PATH}")
            TRAINING_POOL_LIST = [("Write a python function", "pass")]
            PROMPT_TO_INDEX = {TRAINING_POOL_LIST[0][0]: 0}
            
    except Exception as e:
        print(f"❌ [CustomReward] 加载失败: {e}")
        TRAINING_POOL_LIST = [("Error loading data", "pass")]
        PROMPT_TO_INDEX = {}

# ===================================================================
# 1.5 & 1.6 本地检测 (保持原样)
# ===================================================================
def calculate_overlap_penalty(generated_prompt: str, original_prompt: str, tokenizer) -> float:
    sp_clean = generated_prompt.strip().lower()
    up_clean = original_prompt.strip().lower()
    if not sp_clean: return 0.0

    if tokenizer:
        try:
            sp_tokens = set(tokenizer.encode(sp_clean, add_special_tokens=False))
            up_tokens = set(tokenizer.encode(up_clean, add_special_tokens=False))
        except Exception:
            sp_tokens = set(sp_clean.split())
            up_tokens = set(up_clean.split())
    else:
        sp_tokens = set(sp_clean.split())
        up_tokens = set(up_clean.split())
    
    if not sp_tokens: return 0.0
    shared = len(sp_tokens.intersection(up_tokens))
    return shared / len(sp_tokens)

def check_ground_truth_leakage(generated_text: str, ground_truth: str) -> float:
    if not generated_text or not ground_truth: return 0.0
    gen_lines = generated_text.split('\n')
    gt_lines = ground_truth.split('\n')
    
    def clean_lines(lines):
        cleaned = set()
        for line in lines:
            s = line.strip()
            if len(s) > 3: cleaned.add(s)
        return cleaned

    gen_set = clean_lines(gen_lines)
    gt_set = clean_lines(gt_lines)
    if not gen_set: return 0.0
    
    intersection = gen_set.intersection(gt_set)
    return len(intersection) / len(gen_set)

# ===================================================================
# 2. 模板加载 (保持原样)
# ===================================================================
C_PROMPT_FILE_PATH = "/data/zhuldz/lunwen/judge/c_model_prompt.txt"

def _get_c_model_template():
    try:
        if os.path.exists(C_PROMPT_FILE_PATH):
            with open(C_PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "Error: c_model_prompt.txt not found."
    except Exception as e:
        return f"Error reading prompt file: {e}"

# ===================================================================
# 3. 清洗 [关键修改] 严格提取 User Query
# ===================================================================
def _clean_prompt_content(dirty_text: str) -> str:
    """
    从完整的模型输入中提取 User Query。
    输入示例: ...【Target Case】\nUser Query: You are given ...\nReference Solution: ...
    目标: 提取 "You are given ..."
    """
    if not isinstance(dirty_text, str): return str(dirty_text)
    
    # 严格的起始标记
    # 注意：JSONL里的换行符可能是 \\n 也可能是实际换行，这里用 User Query: 定位比较稳
    start_marker = "User Query:"
    end_marker = "Reference Solution:" 
    
    start_idx = dirty_text.find(start_marker)
    
    if start_idx != -1:
        # 内容实际开始位置
        content_start = start_idx + len(start_marker)
        
        # 查找结束位置
        end_idx = dirty_text.find(end_marker, content_start)
        
        if end_idx != -1:
            return dirty_text[content_start:end_idx].strip()
        else:
            # 如果没找到结束标记，取剩下所有
            return dirty_text[content_start:].strip()
            
    # 如果没找到标记，返回原文本（可能是纯 Query）
    return dirty_text.strip()

# ===================================================================
# 4. 加载 (保持原样)
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
            _load_training_pool() # [修改] 触发加载池
            return True
        return False
    except Exception as e:
        print(f"❌ Model Init Failed: {e}")
        return False

def _initialize_globals_from_config(config):
    global _CONFIG_INITIALIZED, K_CODE_GEN, ROLLOUT_DATA_DIR, CURRENT_ITERATION
    if _CONFIG_INITIALIZED: return
    if config:
        CURRENT_ITERATION = config.get('current_iteration', 0) if isinstance(config, dict) else getattr(config, 'current_iteration', 0)
        if isinstance(config, dict) and 'iterative_rl' in config:
            ROLLOUT_DATA_DIR = config['iterative_rl'].get('rollout_data_dir', ROLLOUT_DATA_DIR)
    os.makedirs(ROLLOUT_DATA_DIR, exist_ok=True)
    _CONFIG_INITIALIZED = True

# ===================================================================
# 5. 核心奖励函数 (修改为 1 Center + 2 Aux)
# ===================================================================
def compute_custom_reward(**kwargs):
    _ensure_models_loaded()
    
    # 提取 Prompts
    prompts = kwargs.get('data_sources')
    if prompts is None: prompts = kwargs.get('prompts')
    if prompts is None: prompts = kwargs.get('input_ids')
    if prompts is None: prompts = kwargs.get('inputs')
        
    responses = kwargs.get('solution_strs')
    if responses is None: responses = kwargs.get('responses')
    if responses is None: responses = kwargs.get('predictions')
    
    ground_truths = None
    if 'canonical_solution' in kwargs: ground_truths = kwargs['canonical_solution']
    elif 'output' in kwargs: ground_truths = kwargs['output']
    elif 'ground_truth' in kwargs: ground_truths = kwargs['ground_truth']
    elif 'reward_model' in kwargs:
        rm = kwargs['reward_model']
        if isinstance(rm, dict): ground_truths = rm.get('ground_truth', rm.get('output'))
        elif isinstance(rm, list) and len(rm) > 0: ground_truths = [item.get('ground_truth', item.get('output')) for item in rm]

    if prompts is None or responses is None:
        return {"reward_tensor": torch.zeros(1), "reward_extra_info": {}}

    config = kwargs.get('config')
    _initialize_globals_from_config(config)

    if isinstance(prompts, (np.ndarray, torch.Tensor)): prompts = prompts.tolist()
    if isinstance(responses, (np.ndarray, torch.Tensor)): responses = responses.tolist()
    if ground_truths is not None and isinstance(ground_truths, (np.ndarray, torch.Tensor)): ground_truths = ground_truths.tolist()
    if not ground_truths: ground_truths = [None] * len(prompts)
    
    num_responses = len(responses)
    tasks = []
    current_template = _get_c_model_template()

    # 再次检查数据池
    if not TRAINING_POOL_LIST:
        _load_training_pool()

    for i in range(num_responses):
        p_idx = i % len(prompts) if len(prompts) > 0 else 0
        
        original_prompt_blob = prompts[p_idx]
        
        # 清洗 Prompt，用于去索引里查
        current_query_str = _clean_prompt_content(str(original_prompt_blob))
        current_gt_str = str(ground_truths[p_idx]) if p_idx < len(ground_truths) else "N/A"

        tasks.append({
            "index": i,
            "original_input": original_prompt_blob,
            "current_query": current_query_str, # 清洗后的 Query
            "current_gt": current_gt_str,
            "system_prompt": responses[i],
            "template": current_template
        })

    def process_single_sample_batch(task):
        global B_TOKENIZER, TRAINING_POOL_LIST, PROMPT_TO_INDEX
        
        sys_p = task["system_prompt"]
        raw_sys_p = str(sys_p)
        clean_sys_p = re.sub(r'<think>.*?</think>', '', raw_sys_p, flags=re.DOTALL).strip()
        if not clean_sys_p: clean_sys_p = "Generate python code."
        tmpl = task["template"]

        with api_semaphore:
            ts_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            
            trace_entry = {
                "timestamp": ts_str, 
                "iteration": CURRENT_ITERATION,
                "a_model": {"input": str(task["original_input"]), "output": clean_sys_p},
                "group_tests": [],
                "final_stats": {}
            }
            
            total_score = 0.0
            test_count = 0
            
            # [核心逻辑] 查找当前题目的 index
            current_q_clean = task["current_query"].strip()
            center_idx = PROMPT_TO_INDEX.get(current_q_clean)
            
            test_queue = []
            
            if center_idx is not None:
                # 找到索引，构建 [center, center+1, center+2]
                group_indices = [center_idx + k for k in range(GROUP_SIZE)]
                # 过滤越界
                group_indices = [i for i in group_indices if i < len(TRAINING_POOL_LIST)]
                
                for i in group_indices:
                    q_i, gt_i = TRAINING_POOL_LIST[i]
                    t_type = "Center Task" if i == center_idx else f"Aux Task (+{i-center_idx})"
                    test_queue.append({
                        "query": q_i,
                        "gt": gt_i,
                        "type": t_type
                    })
            else:
                # 没找到索引 (兜底方案)
                print(f"⚠️ [Reward] Query not found in index. First 50: {current_q_clean[:50]}...")
                test_queue.append({
                    "query": task["current_query"],
                    "gt": task["current_gt"],
                    "type": "Center Task (Unmatched)"
                })
            
            # 执行测试
            if MODELS_LOADED:
                for t_idx, test_case in enumerate(test_queue):
                    test_query = test_case["query"]
                    test_gt = test_case["gt"]
                    test_type = test_case["type"]
                    
                    try:
                        local_leakage_ratio = check_ground_truth_leakage(clean_sys_p, test_gt)
                        
                        if local_leakage_ratio > GT_LEAKAGE_THRESHOLD:
                            case_score = MIN_REWARD_FLOOR
                            case_info = {
                                "status": "Hard Leakage Detected",
                                "leakage_ratio": local_leakage_ratio,
                                "final_case_score": case_score,
                                "reason": f"Severe code leakage in {test_type}"
                            }
                        else:
                            # A -> B
                            codes_list = _generate_codes(clean_sys_p, test_query, k=1)
                            
                            # B -> C
                            avg_val, _, results = _evaluate_codes(
                                codes_list, test_gt, test_query, 
                                test_index=t_idx, 
                                timestamp=ts_str
                            )
                            
                            # 本地惩罚
                            overlap = calculate_overlap_penalty(clean_sys_p, test_query, B_TOKENIZER)
                            p_user_overlap = overlap * USER_OVERLAP_WEIGHT
                            
                            # 算分
                            case_score = avg_val - p_user_overlap
                            case_score = max(case_score, MIN_REWARD_FLOOR)
                            
                            # 记录
                            b_out = codes_list[0].get('code', '') if codes_list else ''
                            c_res = results[0] if results else {}
                            
                            try:
                                c_log_input = tmpl.format(
                                    prompt=clean_sys_p,
                                    user_prompt=test_query,
                                    generated_code=b_out,
                                    code_ground_truth=test_gt
                                )
                            except Exception:
                                c_log_input = f"[Format Error] {clean_sys_p[:20]}..."

                            case_info = {
                                "test_index": t_idx,
                                "test_type": test_type,
                                "full_judge_context": c_log_input,
                                "c_model_raw_score": float(avg_val),
                                "c_model_details": c_res,
                                "local_penalties": {"user_prompt_overlap": -float(p_user_overlap)},
                                "final_case_score": float(case_score)
                            }

                        total_score += case_score
                        test_count += 1
                        trace_entry["group_tests"].append(case_info)
                        
                        # Fail-Fast
                        if case_score < FAIL_FAST_THRESHOLD:
                            remaining = len(test_queue) - 1 - t_idx
                            if remaining > 0:
                                total_score += (remaining * MIN_REWARD_FLOOR)
                                test_count += remaining
                                trace_entry["final_stats"]["fail_fast_triggered"] = True
                                trace_entry["group_tests"].append({"info": f"Fail-fast at {test_type}, skipped {remaining} tests"})
                                break
                        
                    except Exception as e:
                        print(f"Test case {t_idx} failed: {e}")
                        trace_entry["group_tests"].append({"error": str(e)})
            else:
                trace_entry["error"] = "Models not loaded"
            
            # 平均分
            final_avg_score = (total_score / max(1, test_count)) 
            final_avg_score = max(final_avg_score, MIN_REWARD_FLOOR)
            
            trace_entry["final_stats"].update({
                "tests_run": test_count,
                "total_score": float(total_score),
                "final_average_reward": float(final_avg_score)
            })
            
            return float(final_avg_score), json.dumps(trace_entry, ensure_ascii=False)

    # --- 执行 ---
    all_scores = [0.0] * len(responses)
    all_traces = ["{}"] * len(responses)
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY + 2) as executor:
        future_to_idx = {executor.submit(process_single_sample_batch, task): task["index"] for task in tasks}
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                score, trace = future.result()
                all_scores[idx] = score
                all_traces[idx] = trace
            except Exception as e:
                print(f"Task {idx} failed: {e}")
                traceback.print_exc()

    return {
        "reward_tensor": torch.tensor(all_scores, dtype=torch.float32),
        "reward_extra_info": {"abc_trace": all_traces}
    }

# ===================================================================
# 6. 辅助函数 (保持原样)
# ===================================================================
def _generate_codes(system_prompt: str, prompt: str, k: int = 1) -> List[Dict]:
    codes = []
    if B_MODEL is None:
        return [{'code': "Error: B_MODEL is None", 'b_model_input': "", 'system_prompt': system_prompt}] * k

    try:
        sys_p = str(system_prompt) if system_prompt is not None else ""
        usr_p = str(prompt) if prompt is not None else ""
        generated_codes = generate_code_from_prompt(
            B_MODEL, B_TOKENIZER, generated_prompt=sys_p, original_prompt=usr_p, k=k
        )
    except Exception as e:
        print(f"Error in Model B: {e}")
        generated_codes = [f"Error: {e}"] * k
        
    for code in generated_codes:
        b_input = f"{system_prompt}\n{prompt}"
        codes.append({
            'code': code, 
            'b_model_input': b_input, 
            'system_prompt': system_prompt 
        })
    return codes
    
def _evaluate_codes(codes: List[Dict], ground_truth: str, original_prompt: str, test_index: int = None, timestamp: str = None) -> tuple:
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
                b_model_input=code_item['b_model_input'],
                user_prompt=original_prompt,
                index=test_index,
                timestamp=timestamp
            )
        except Exception as e: 
            print(f"Error in Model C: {e}")
            score_val = 0.0
            full_res = {"error": str(e)}
            
        scores.append(score_val)
        full_results.append(full_res)
    
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, scores, full_results