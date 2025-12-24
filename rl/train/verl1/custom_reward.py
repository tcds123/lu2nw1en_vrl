import os
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd  # 必须引入 pandas 处理 parquet
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

# 全局配置
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
# [核心配置] 数据源 & 奖励参数
# ===================================================================
# 优先从环境变量读取，如果没有则使用默认路径
TRAIN_SET_PATH = os.environ.get("VERL_DATA_PATH", "/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_cleaned_2.parquet")

GROUP_SIZE = 3
FAIL_FAST_THRESHOLD = -5.0 

TRAINING_POOL_LIST: List[Tuple[str, str]] = [] 
PROMPT_TO_INDEX: Dict[str, int] = {}

USER_OVERLAP_WEIGHT = 15.0       
GT_LEAKAGE_THRESHOLD = 0.15      
MIN_REWARD_FLOOR = -10.0  

# [新增] 系统提示词长度限制配置
SYS_PROMPT_MIN_LEN = 50         # 最小 Token 数
SYS_PROMPT_MAX_LEN = 100        # 最大 Token 数
LENGTH_BONUS_SCORE = 1.0        # 达标奖励 (+1.0)
LENGTH_PENALTY_COEFF = 0.07    # 偏离惩罚系数 (3.0/45 ≈ 0.067, 50-5=45)
MAX_LENGTH_PENALTY = 3.0        # 最大惩罚封顶 (最多扣3分)

# ===================================================================
# 1.4 加载全量训练数据 (Parquet)
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
            
            cols = df.columns
            # 优先找 raw_input，因为它是不带 Template 的纯净 Query
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

            for _, row in df.iterrows():
                try:
                    q = str(row[prompt_col]).strip()
                    gt = str(row[gt_col]).strip()
                    
                    if q and gt:
                        current_list_idx = len(TRAINING_POOL_LIST)
                        TRAINING_POOL_LIST.append((q, gt))
                        # 建立映射: Query -> 全局List索引
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
# 1.5 & 1.6 & 1.7 辅助检测函数
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

# [新增] 长度奖励计算逻辑
def _calculate_length_reward(text: str, tokenizer) -> Tuple[float, Dict[str, Any]]:
    reward = 0.0
    details = {"length": 0, "status": "unknown"}
    
    if not text:
        # 空内容直接给最大惩罚
        return -MAX_LENGTH_PENALTY, {"length": 0, "status": "empty"} 

    try:
        # 优先使用 B模型的 Tokenizer 计算长度
        if tokenizer:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            length = len(tokens)
        else:
            length = len(text.split())
    except Exception:
        length = len(text.split())

    details["length"] = length

    # 1. 达标奖励 (Target Zone: 50-100)
    if SYS_PROMPT_MIN_LEN <= length <= SYS_PROMPT_MAX_LEN:
        reward = LENGTH_BONUS_SCORE
        details["status"] = "perfect"
        details["modifier"] = reward
    # 2. 偏离惩罚
    else:
        if length < SYS_PROMPT_MIN_LEN:
            diff = SYS_PROMPT_MIN_LEN - length
            status = "too_short"
        else:
            diff = length - SYS_PROMPT_MAX_LEN
            status = "too_long"
        
        # 计算基础惩罚
        raw_penalty = diff * LENGTH_PENALTY_COEFF
        
        # 应用封顶：最大扣 3.0 分
        penalty = min(raw_penalty, MAX_LENGTH_PENALTY)
        
        reward = -penalty
        details["status"] = status
        details["modifier"] = reward
        details["deviation"] = diff
        details["raw_penalty"] = raw_penalty

    return reward, details

# ===================================================================
# 2. 模板加载
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
# 3. 清洗
# ===================================================================
def _clean_prompt_content(dirty_text: str) -> str:
    """提取 User Query: 和 Reference Solution: 之间的内容"""
    if not isinstance(dirty_text, str): return str(dirty_text)
    
    start_marker = "User Query:"
    end_marker = "Reference Solution:" 
    
    start_idx = dirty_text.find(start_marker)
    if start_idx != -1:
        content_start = start_idx + len(start_marker)
        end_idx = dirty_text.find(end_marker, content_start)
        if end_idx != -1:
            return dirty_text[content_start:end_idx].strip()
        else:
            return dirty_text[content_start:].strip()
            
    return dirty_text.strip()

# ===================================================================
# 4. 加载
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
            _load_training_pool() # 触发 Parquet 加载
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
# 5. 核心奖励函数
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

    # 再次检查数据池是否为空
    if not TRAINING_POOL_LIST:
        _load_training_pool()

    for i in range(num_responses):
        p_idx = i % len(prompts) if len(prompts) > 0 else 0
        original_prompt_blob = prompts[p_idx]
        current_query_str = _clean_prompt_content(str(original_prompt_blob))
        current_gt_str = str(ground_truths[p_idx]) if p_idx < len(ground_truths) else "N/A"

        tasks.append({
            "index": i,
            "original_input": original_prompt_blob,
            "current_query": current_query_str,
            "current_gt": current_gt_str,
            "system_prompt": responses[i],
            "template": current_template
        })

    def process_single_sample_batch(task):
        global B_TOKENIZER, TRAINING_POOL_LIST, PROMPT_TO_INDEX
        
        sys_p = task["system_prompt"]
        clean_sys_p = re.sub(r'<think>.*?</think>', '', str(sys_p), flags=re.DOTALL).strip()
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
            
            # 1. 计算长度奖惩
            len_reward, len_details = _calculate_length_reward(clean_sys_p, B_TOKENIZER)
            trace_entry["length_check"] = len_details
            
            total_score = 0.0
            test_count = 0
            
            current_q_clean = task["current_query"].strip()
            center_idx = PROMPT_TO_INDEX.get(current_q_clean)
            
            test_queue = []
            
            if center_idx is not None:
                # 1 Center + 2 Aux
                group_indices = [center_idx + k for k in range(GROUP_SIZE)]
                group_indices = [i for i in group_indices if i < len(TRAINING_POOL_LIST)]
                for i in group_indices:
                    q_i, gt_i = TRAINING_POOL_LIST[i]
                    t_type = "Center Task" if i == center_idx else f"Aux Task (+{i-center_idx})"
                    test_queue.append({"query": q_i, "gt": gt_i, "type": t_type})
            else:
                print(f"⚠️ [Reward] Query not found in index. First 50: {current_q_clean[:50]}")
                test_queue.append({"query": task["current_query"], "gt": task["current_gt"], "type": "Center Task (Unmatched)"})
            
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
                                "final_case_score": case_score
                            }
                        else:
                            codes_list = _generate_codes(clean_sys_p, test_query, k=1)
                            avg_val, _, results = _evaluate_codes(
                                codes_list, test_gt, test_query, 
                                test_index=t_idx, 
                                timestamp=ts_str
                            )
                            
                            overlap = calculate_overlap_penalty(clean_sys_p, test_query, B_TOKENIZER)
                            p_user_overlap = overlap * USER_OVERLAP_WEIGHT
                            
                            # [综合算分] 基础分 - 重复惩罚 + 长度奖惩
                            raw_score = avg_val - p_user_overlap + len_reward
                            
                            # [新增] 封顶逻辑：不允许超过 10.0 分
                            raw_score = min(raw_score, 10.0)
                            
                            # 兜底逻辑：不允许低于最低分
                            case_score = max(raw_score, MIN_REWARD_FLOOR)
                            
                            b_out = codes_list[0].get('code', '') if codes_list else ''
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
                                "test_type": test_type,
                                "c_model_score": float(avg_val),
                                "length_modifier": float(len_reward),
                                "final_case_score": float(case_score),
                                "full_context": c_log_input
                            }

                        total_score += case_score
                        test_count += 1
                        trace_entry["group_tests"].append(case_info)
                        
                        if case_score < FAIL_FAST_THRESHOLD:
                            break
                        
                    except Exception as e:
                        print(f"Test case {t_idx} failed: {e}")
                        trace_entry["group_tests"].append({"error": str(e)})
            else:
                trace_entry["error"] = "Models not loaded"
            
            final_avg_score = (total_score / max(1, test_count)) 
            final_avg_score = max(final_avg_score, MIN_REWARD_FLOOR)
            
            trace_entry["final_stats"] = {"total_score": float(total_score), "avg_score": float(final_avg_score)}
            return float(final_avg_score), json.dumps(trace_entry, ensure_ascii=False)

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
# 6. 辅助函数 (保持原样，全部保留)
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