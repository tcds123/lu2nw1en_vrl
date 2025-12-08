import os
import sys
import json
import logging
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
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
    err_msg = str(e)
    print(f"⚠️ [CustomReward] Import Failed: {err_msg}")
    
    def load_b_model(**kwargs): raise ImportError(f"b_interface import failed: {err_msg}")
    def load_c_model(**kwargs): raise ImportError(f"c_interface import failed: {err_msg}")
    def generate_code_from_prompt(**kwargs): return ["Error: Import failed"]
    def compute_c_reward(**kwargs): return 0.0, {}

# 全局配置
ROLLOUT_DATA_DIR = "./a_model_grpo_standard/rollouts"
ITERATION_LOG_DIR = "./a_model_grpo_standard/iterations"
K_CODE_GEN = 1  # 配合通用性测试，每个题目只生成1个代码，控制计算成本
ROLLOUT_N = 5 
CURRENT_ITERATION = 0
_CONFIG_INITIALIZED = False
SAMPLE_COUNT = 0

# 并发控制
MAX_CONCURRENCY = 10  
api_semaphore = threading.Semaphore(MAX_CONCURRENCY)

# ===================================================================
# [核心配置] 通用性测试集与惩罚参数
# ===================================================================
# 测试集路径
TEST_SET_PATH = "/data/zhuldz/lunwen/generation/humaneval_pro.json"
# 每次校验使用的题目数量
NUM_TEST_CASES = 4
# 全局测试集缓存
GENERAL_TEST_SET: List[Tuple[str, str]] = []

# 惩罚权重
USER_OVERLAP_WEIGHT = 15.0       # 复读用户题目惩罚权重
GT_LEAKAGE_THRESHOLD = 0.15      # 本地硬熔断阈值 (15%)
MIN_REWARD_FLOOR = -10.0         # 地板分

# ===================================================================
# 1.4 加载通用测试集
# ===================================================================
def _load_general_test_set():
    global GENERAL_TEST_SET
    if GENERAL_TEST_SET: return

    print(f"⏳ [CustomReward] 正在加载通用测试集 (Top {NUM_TEST_CASES})...")
    try:
        if os.path.exists(TEST_SET_PATH):
            with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict): data = [data]
            
            count = 0
            for item in data:
                q = item.get('raw_problem', '').strip()
                gt = item.get('raw_solution', '').strip()
                if q and gt:
                    GENERAL_TEST_SET.append((q, gt))
                    count += 1
                    if count >= NUM_TEST_CASES:
                        break
            print(f"✅ [CustomReward] 加载完成，共 {len(GENERAL_TEST_SET)} 条。")
        else:
            print(f"❌ [CustomReward] 文件未找到: {TEST_SET_PATH}")
            GENERAL_TEST_SET = [("Print hello world", "print('hello world')")]
    except Exception as e:
        print(f"❌ [CustomReward] 加载失败: {e}")
        GENERAL_TEST_SET = [("Error loading data", "pass")]

# ===================================================================
# 1.5 & 1.6 本地检测函数
# ===================================================================
def calculate_overlap_penalty(generated_prompt: str, original_prompt: str, tokenizer) -> float:
    """计算 A模型输出 与 用户原始问题 之间的 Token 重叠率 (防止复读机)"""
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
    """本地快速行重叠检测 (Layer 1 防御)"""
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
# 2. 动态加载模板
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
# 3. 清洗函数
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
# 4. 模型加载
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
            _load_general_test_set()
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
# 5. 核心奖励函数 (通用性测试 + 反幻觉支持)
# ===================================================================
def compute_custom_reward(**kwargs):
    _ensure_models_loaded()
    
    # 提取 A 模型输出 (System Prompts)
    prompts = kwargs.get('data_sources') or kwargs.get('prompts') or kwargs.get('input_ids')
    responses = kwargs.get('solution_strs') or kwargs.get('responses') or kwargs.get('predictions')
    
    if prompts is None or responses is None:
        return {"reward_tensor": torch.zeros(1), "reward_extra_info": {}}

    config = kwargs.get('config')
    _initialize_globals_from_config(config)

    if isinstance(prompts, (np.ndarray, torch.Tensor)): prompts = prompts.tolist()
    if isinstance(responses, (np.ndarray, torch.Tensor)): responses = responses.tolist()
    
    num_responses = len(responses)
    tasks = []
    current_template = _get_c_model_template()

    for i in range(num_responses):
        p_idx = i % len(prompts) if len(prompts) > 0 else 0
        tasks.append({
            "index": i,
            "original_input": prompts[p_idx],
            "system_prompt": responses[i],
            "template": current_template
        })

    # --- 并发处理函数 ---
    def process_single_sample_general(task):
        global B_TOKENIZER, GENERAL_TEST_SET
        
        sys_p = task["system_prompt"]
        raw_sys_p = str(sys_p)
        # 去除思维链标记
        clean_sys_p = re.sub(r'<think>.*?</think>', '', raw_sys_p, flags=re.DOTALL).strip()
        if not clean_sys_p: clean_sys_p = "Generate python code."

        with api_semaphore:
            trace_entry = {
                "timestamp": datetime.now().isoformat(),
                "iteration": CURRENT_ITERATION,
                "a_model": {"input": str(task["original_input"]), "output": clean_sys_p},
                "generalization_tests": [],
                "final_stats": {}
            }
            
            total_score = 0.0
            test_count = 0
            
            if MODELS_LOADED and GENERAL_TEST_SET:
                # 遍历每条测试数据
                for t_idx, (test_query, test_gt) in enumerate(GENERAL_TEST_SET):
                    try:
                        # [Layer 1] 本地硬性熔断检测
                        # 如果 System Prompt 大段照抄了当前题目的答案，直接 -10
                        local_leakage_ratio = check_ground_truth_leakage(clean_sys_p, test_gt)
                        
                        if local_leakage_ratio > GT_LEAKAGE_THRESHOLD:
                            case_score = MIN_REWARD_FLOOR
                            case_info = {
                                "status": "Hard Leakage Detected",
                                "leakage_ratio": local_leakage_ratio,
                                "final_case_score": case_score
                            }
                        else:
                            # [生成阶段] A -> B
                            codes_list = _generate_codes(clean_sys_p, test_query, k=1)
                            
                            # [评判阶段] B -> C (这里会调用 c_interface 的反幻觉核实)
                            # avg_val 已经是经过 C 模型核实后的 (Base Score - Verified Penalties)
                            avg_val, _, results = _evaluate_codes(codes_list, test_gt, test_query)
                            
                            # [本地惩罚] 复读机检测 (检查 System Prompt 是否复读了题目)
                            # 这一步 Judge 很难判断，因为 Judge 可能会认为复读题目是“相关性高”
                            # 所以我们在本地扣分
                            overlap = calculate_overlap_penalty(clean_sys_p, test_query, B_TOKENIZER)
                            p_user_overlap = overlap * USER_OVERLAP_WEIGHT
                            
                            # [最终单题得分]
                            # C模型分 (含质量+泄露扣分) - 本地复读扣分
                            case_score = avg_val - p_user_overlap
                            case_score = max(case_score, MIN_REWARD_FLOOR)
                            
                            # 记录
                            b_out = codes_list[0].get('code', '') if codes_list else ''
                            c_res = results[0] if results else {}
                            
                            case_info = {
                                "query_snippet": test_query[:30] + "...",
                                "b_output_snippet": b_out[:30] + "...",
                                "c_model_raw_score": float(avg_val),
                                "c_model_details": c_res, # 包含 violation_report 和 核实结果
                                "local_penalties": {"user_prompt_overlap": -float(p_user_overlap)},
                                "final_case_score": float(case_score)
                            }

                        total_score += case_score
                        test_count += 1
                        trace_entry["generalization_tests"].append(case_info)
                        
                    except Exception as e:
                        print(f"Test case {t_idx} failed: {e}")
                        trace_entry["generalization_tests"].append({"error": str(e)})
            else:
                trace_entry["error"] = "Models not loaded or Test Set empty"
            
            # 计算平均分
            final_avg_score = (total_score / test_count) if test_count > 0 else MIN_REWARD_FLOOR
            final_avg_score = max(final_avg_score, MIN_REWARD_FLOOR)
            
            trace_entry["final_stats"] = {
                "tests_run": test_count,
                "total_score": float(total_score),
                "final_average_reward": float(final_avg_score)
            }
            
            return float(final_avg_score), json.dumps(trace_entry, ensure_ascii=False)

    # --- 执行 ---
    all_scores = [0.0] * len(responses)
    all_traces = ["{}"] * len(responses)
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY + 2) as executor:
        future_to_idx = {executor.submit(process_single_sample_general, task): task["index"] for task in tasks}
        
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
# 6. 辅助函数
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
    
def _evaluate_codes(codes: List[Dict], ground_truth: str, original_prompt: str) -> tuple:
    scores = []
    full_results = []
    
    if C_JUDGE is None: return 0.0, [0.0]*len(codes), []
    if not codes: return 0.0, [], []
    
    gt_str = str(ground_truth) if ground_truth is not None else ""

    for code_item in codes:
        try:
            # 这里调用 c_interface.compute_reward
            # 里面已经包含了 反幻觉核实 和 分数计算逻辑
            score_val, full_res = compute_c_reward(
                c_judge=C_JUDGE,
                generated_code=code_item['code'],
                canonical_solution=gt_str,
                generated_prompt=code_item.get('system_prompt', ''), 
                b_model_input=code_item['b_model_input'],
                user_prompt=original_prompt
            )
        except Exception as e: 
            print(f"Error in Model C: {e}")
            score_val = 0.0
            full_res = {"error": str(e)}
            
        scores.append(score_val)
        full_results.append(full_res)
    
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, scores, full_results