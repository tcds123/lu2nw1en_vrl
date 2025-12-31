import os
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
import re
import threading
import multiprocessing
import contextlib
import io
import time
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================================================================
# 1. 基础配置
# ===================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

B_MODEL = None
B_TOKENIZER = None
C_JUDGE = None
MODELS_LOADED = False

try:
    from verl1.b_interface import load_b_model, generate_code_from_prompt
    from verl1.c_interface import load_c_model, compute_reward as compute_c_reward
except Exception as e:
    print(f"⚠️ Import Failed: {e}")
    def load_b_model(**kwargs): return None, None
    def load_c_model(**kwargs): return None
    def generate_code_from_prompt(**kwargs): return ["Error"]
    def compute_c_reward(**kwargs): return 0.0, {}

# ===================================================================
# 2. 全局参数
# ===================================================================
TRAIN_SET_PATH = os.environ.get("VERL_DATA_PATH", "/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_cleaned_2.parquet")
MAX_CONCURRENCY = 8
api_semaphore = threading.Semaphore(MAX_CONCURRENCY)

TRAINING_POOL_LIST: List[Tuple[str, str]] = [] 
PROMPT_TO_INDEX: Dict[str, int] = {}

# 评分配置
EXEC_FATAL_ERROR = -5.0
EXEC_ZERO_PASS = -3.0
EXEC_FULL_PASS = 5.0
JUDGE_SCALE = 0.5
PENALTY_LEAKAGE = -10.0      
PENALTY_EMPTY = -10.0        
WEIGHT_OVERLAP = 5.0         
WEIGHT_LENGTH = 0.05         
PASS_RATE_GATING = 0.1 

# ===================================================================
# 3. 工具函数
# ===================================================================

def _extract_user_query_fallback(meta_prompt: str) -> str:
    try:
        pattern = r"User Query:\s*(.*?)\s*Reference Solution:"
        match = re.search(pattern, meta_prompt, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        pattern_backup = r"User Query:\s*(.*)"
        match_backup = re.search(pattern_backup, meta_prompt, re.DOTALL | re.IGNORECASE)
        if match_backup: return match_backup.group(1).strip().split("Reference Solution")[0].strip()
        return meta_prompt
    except: return meta_prompt

def extract_function_name_from_code(code):
    try:
        match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
        return match.group(1) if match else None
    except: return None

def extract_function_name_from_test(test_case):
    try:
        match = re.search(r"assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", test_case)
        return match.group(1) if match else None
    except: return None

def _unsafe_exec_worker(code, unit_tests, result_queue):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            scope = {}
            try: exec(code, scope)
            except Exception:
                result_queue.put({"status": "compile_error"})
                return
            gen_func = extract_function_name_from_code(code)
            exp_func = extract_function_name_from_test(unit_tests[0]) if unit_tests else None
            if gen_func and exp_func and gen_func != exp_func:
                if gen_func in scope: scope[exp_func] = scope[gen_func]
            passed = 0
            total = len(unit_tests)
            if total == 0:
                result_queue.put({"status": "no_tests", "pass_rate": 0.0})
                return
            for test in unit_tests:
                try:
                    exec(test, scope)
                    passed += 1
                except Exception: pass
            result_queue.put({"status": "success", "pass_rate": passed / total})
    except Exception: result_queue.put({"status": "runtime_error"})

def compute_execution_pass_rate(code_str: str, unit_tests_str: str) -> Tuple[float, str]:
    if not code_str: return -1.0, "no_code"
    try:
        clean_code = code_str
        if "```python" in code_str:
            clean_code = code_str.split("```python")[1].split("```")[0].strip()
        elif "```" in code_str:
            clean_code = code_str.split("```")[1].split("```")[0].strip()

        unit_tests = []
        if not unit_tests_str: return -2.0, "ut_empty_string"

        try:
            unit_tests = json.loads(unit_tests_str)
        except:
            try: unit_tests = ast.literal_eval(unit_tests_str)
            except: return -2.0, "ut_parse_fail"

        if not isinstance(unit_tests, list) or not unit_tests:
            return -2.0, f"ut_not_list_or_empty: {type(unit_tests)}"

        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_unsafe_exec_worker, args=(clean_code, unit_tests, queue))
        p.start()
        p.join(timeout=2.0)
        
        if p.is_alive():
            p.terminate(); p.join()
            return -1.0, "timeout"
        if not queue.empty():
            res = queue.get()
            if res["status"] == "success": return res["pass_rate"], "ok"
            elif res["status"] == "compile_error": return -1.0, "compile_error"
        return -1.0, "exec_unknown"
    except Exception as e: return -1.0, f"wrapper_err:{str(e)}"

# ... (辅助函数保持不变) ...
def calculate_overlap_penalty(generated_prompt, original_prompt, tokenizer):
    sp_clean = generated_prompt.strip().lower()
    up_clean = original_prompt.strip().lower()
    if not sp_clean: return 0.0
    sp_tokens = set(sp_clean.split())
    up_tokens = set(up_clean.split())
    if not sp_tokens: return 0.0
    return len(sp_tokens.intersection(up_tokens)) / len(sp_tokens)

def check_ground_truth_leakage(generated_text, ground_truth):
    if not generated_text or not ground_truth: return 0.0
    gen_lines = set([l.strip() for l in generated_text.split('\n') if len(l.strip())>3])
    gt_lines = set([l.strip() for l in ground_truth.split('\n') if len(l.strip())>3])
    if not gen_lines: return 0.0
    return len(gen_lines.intersection(gt_lines)) / len(gen_lines)

def _calculate_rule_penalties(sys_prompt, original_prompt, tokenizer):
    penalties = 0.0
    length = len(sys_prompt.split())
    if length < 30: penalties -= min((30 - length) * WEIGHT_LENGTH, 2.0)
    elif length > 200: penalties -= min((length - 200) * WEIGHT_LENGTH, 2.0)
    overlap_ratio = calculate_overlap_penalty(sys_prompt, original_prompt, tokenizer)
    penalties -= (overlap_ratio * WEIGHT_OVERLAP)
    return penalties

def _get_c_model_template():
    path = os.path.join(project_root, "judge/c_model_prompt.txt")
    if os.path.exists(path):
        with open(path, "r") as f: return f.read().strip()
    return "{prompt}\n{user_prompt}\n{generated_code}"

def _ensure_models_loaded():
    global B_MODEL, B_TOKENIZER, C_JUDGE, MODELS_LOADED
    if MODELS_LOADED: return
    if B_MODEL is None: B_MODEL, B_TOKENIZER = load_b_model()
    if C_JUDGE is None: C_JUDGE = load_c_model()
    MODELS_LOADED = True
    _load_training_pool()

def _load_training_pool():
    global TRAINING_POOL_LIST, PROMPT_TO_INDEX
    if TRAINING_POOL_LIST: return
    try:
        if os.path.exists(TRAIN_SET_PATH):
            df = pd.read_parquet(TRAIN_SET_PATH)
            cols = df.columns
            prompt_col = next((c for c in ['raw_input', 'input', 'prompt'] if c in cols), None)
            if prompt_col:
                for idx, row in df.iterrows(): PROMPT_TO_INDEX[str(row[prompt_col]).strip()] = idx
    except: pass

def _generate_codes(system_prompt: str, prompt: str, k: int = 1) -> List[Dict]:
    codes = []
    if B_MODEL is None: return [{'code': "Error", 'b_model_input': "", 'system_prompt': system_prompt}] * k
    try:
        sys_p = str(system_prompt) if system_prompt else ""
        usr_p = str(prompt) if prompt else ""
        generated_codes = generate_code_from_prompt(B_MODEL, B_TOKENIZER, generated_prompt=sys_p, original_prompt=usr_p, k=k)
    except Exception: generated_codes = ["Error"] * k
    for code in generated_codes:
        codes.append({'code': code, 'b_model_input': f"{system_prompt}\n{prompt}", 'system_prompt': system_prompt})
    return codes
    
def _evaluate_codes(codes: List[Dict], ground_truth: str, original_prompt: str, test_index: int = None, timestamp: str = None):
    if C_JUDGE is None: return 0.0, []
    scores = []
    details = []
    for code_item in codes:
        try:
            res = compute_c_reward(c_judge=C_JUDGE, generated_code=code_item['code'], canonical_solution=ground_truth,
                                    generated_prompt=code_item.get('system_prompt', ''), b_model_input=code_item['b_model_input'],
                                    user_prompt=original_prompt, index=test_index, timestamp=timestamp)
            if isinstance(res, (tuple, list)):
                s = float(res[0]); r = res[1] if len(res) > 1 else {}
            else:
                s = float(res); r = {}
        except Exception as e: s = 0.0; r = {"error": str(e)}
        scores.append(s); details.append(r)
    avg_score = sum(scores)/len(scores) if scores else 0.0
    return avg_score, details

# ===================================================================
# 4. 核心入口 (从 Ground Truth 包裹中解析数据)
# ===================================================================
_DEBUG_PRINTED = False

def compute_custom_reward(**kwargs):
    global _DEBUG_PRINTED
    _ensure_models_loaded()
    
    prompts = kwargs.get('data_sources') or kwargs.get('prompts')
    responses = kwargs.get('solution_strs') or kwargs.get('responses')
    
    # [关键] Verl 框架传入的 output 列数据
    ground_truths_raw = kwargs.get('ground_truth', []) 

    # [DEBUG] 检查是否收到了打包数据
    if not _DEBUG_PRINTED and len(ground_truths_raw) > 0:
        print("\n" + "="*50)
        print("🔍 [DEBUG] First Ground Truth Item:")
        print(str(ground_truths_raw[0])[:200] + "...")
        _DEBUG_PRINTED = True

    n_prompts = len(prompts)
    n_responses = len(responses)
    
    # 解析打包数据
    parsed_items = []
    for item in ground_truths_raw:
        try:
            # 这里的 item 应该是一个 JSON 字符串
            parsed = json.loads(item)
            if not isinstance(parsed, dict): 
                # 如果不是字典，说明数据格式不对（可能是纯代码）
                parsed = {"ground_truth": str(item)}
        except:
            # 解析失败，说明就是纯代码
            parsed = {"ground_truth": str(item)}
        parsed_items.append(parsed)

    ground_truths_aligned = []
    unit_tests_aligned = []
    raw_inputs_aligned = []
    
    ratio = n_responses // n_prompts if n_prompts > 0 else 1
    
    for i in range(n_prompts):
        item = parsed_items[i]
        
        gt = item.get('ground_truth', '')
        # 鲁棒获取 unit_tests
        ut = item.get('unit_tests') or item.get('tests', '[]')
        if not isinstance(ut, str): ut = json.dumps(ut)
        
        raw_in = item.get('raw_input', '')
        
        ground_truths_aligned.extend([gt] * ratio)
        unit_tests_aligned.extend([ut] * ratio)
        raw_inputs_aligned.extend([raw_in] * ratio)
    
    rem = n_responses - len(ground_truths_aligned)
    if rem > 0:
        item = parsed_items[0]
        gt = item.get('ground_truth', '')
        ut = item.get('unit_tests') or item.get('tests', '[]')
        if not isinstance(ut, str): ut = json.dumps(ut)
        raw_in = item.get('raw_input', '')
        ground_truths_aligned.extend([gt] * rem)
        unit_tests_aligned.extend([ut] * rem)
        raw_inputs_aligned.extend([raw_in] * rem)

    tasks = []
    tmpl = _get_c_model_template()

    for i in range(n_responses):
        p_idx = i // ratio if ratio > 0 else 0
        p_idx = min(p_idx, n_prompts - 1)
        tasks.append({
            "index": i,
            "meta_prompt": str(prompts[p_idx]),
            "raw_input": raw_inputs_aligned[i],
            "current_gt": str(ground_truths_aligned[i]),
            "unit_tests": str(unit_tests_aligned[i]),
            "system_prompt": responses[i],
            "template": tmpl
        })

    def process_single_sample_batch(task):
        sys_p = task["system_prompt"]
        clean_sys_p = re.sub(r'<think>.*?</think>', '', str(sys_p), flags=re.DOTALL).strip()
        
        trace_data = {
            "test_type": "unknown", "generated_code": "", "real_user_query": "", 
            "full_context": clean_sys_p, "c_model_details": {}, 
            "c_model_score": 0.0, "length_modifier": 0.0, "pass_rate": 0.0, 
            "exec_score": 0.0, "judge_contrib": 0.0, "penalty_score": 0.0, 
            "final_case_score": 0.0, "msg": "ok"
        }

        if not clean_sys_p:
            trace_data["msg"] = "empty_prompt"
            trace_data["final_case_score"] = PENALTY_EMPTY
            return PENALTY_EMPTY, json.dumps(trace_data, ensure_ascii=False)

        with api_semaphore:
            try:
                leakage = check_ground_truth_leakage(clean_sys_p, task["current_gt"])
                if leakage > 0.15:
                    trace_data["msg"] = "leakage"
                    trace_data["final_case_score"] = PENALTY_LEAKAGE
                    return PENALTY_LEAKAGE, json.dumps(trace_data, ensure_ascii=False)
                
                rule_penalties = _calculate_rule_penalties(clean_sys_p, task["raw_input"] or task["meta_prompt"], B_TOKENIZER)
                trace_data["length_modifier"] = rule_penalties 
                trace_data["penalty_score"] = rule_penalties

                real_user_query = task["raw_input"]
                if not real_user_query:
                    real_user_query = _extract_user_query_fallback(task["meta_prompt"])
                trace_data["real_user_query"] = real_user_query

                codes_list = _generate_codes(clean_sys_p, real_user_query, k=1)
                generated_code = codes_list[0].get('code', '')
                trace_data["generated_code"] = generated_code

                pass_rate, exec_msg = compute_execution_pass_rate(generated_code, task["unit_tests"])
                trace_data["pass_rate"] = pass_rate
                
                if pass_rate < 0:
                    exec_score = EXEC_FATAL_ERROR
                    trace_data["msg"] = f"gated_neg ({exec_msg})"
                    trace_data["test_type"] = "fail"
                else:
                    exec_score = EXEC_ZERO_PASS + (pass_rate * (EXEC_FULL_PASS - EXEC_ZERO_PASS))
                    trace_data["test_type"] = "success"
                
                trace_data["exec_score"] = exec_score

                judge_raw = 0.0
                c_details = []
                if pass_rate >= 0:
                    judge_raw, c_details = _evaluate_codes(
                        codes_list, task["current_gt"], real_user_query, 
                        test_index=task["index"], timestamp=datetime.now().strftime("%T")
                    )
                
                trace_data["c_model_score"] = judge_raw
                trace_data["c_model_details"] = c_details[0] if c_details else {}

                subjective_composite = judge_raw + rule_penalties
                subjective_contrib = subjective_composite * JUDGE_SCALE
                
                if pass_rate < PASS_RATE_GATING:
                    if subjective_contrib > 0:
                        subjective_contrib = 0.0
                        trace_data["msg"] = f"gated_pos ({trace_data['msg']})"
                    else:
                        trace_data["msg"] = f"gated_neg ({trace_data['msg']})"
                
                trace_data["judge_contrib"] = subjective_contrib

                final_score = exec_score + subjective_contrib
                final_score = max(-10.0, min(10.0, final_score))
                trace_data["final_case_score"] = final_score
                
                return float(final_score), json.dumps(trace_data, ensure_ascii=False)

            except Exception as e:
                trace_data["msg"] = f"error: {str(e)}"
                trace_data["final_case_score"] = -10.0
                return -10.0, json.dumps(trace_data, ensure_ascii=False)

    all_scores = [0.0] * n_responses
    all_traces = ["{}"] * n_responses
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY + 2) as executor:
        future_to_idx = {executor.submit(process_single_sample_batch, task): task["index"] for task in tasks}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                s, t = future.result()
                all_scores[idx] = s; all_traces[idx] = t
            except Exception: 
                all_scores[idx] = -10.0; all_traces[idx] = json.dumps({"msg": "thread_error"})

    return {"reward_tensor": torch.tensor(all_scores, dtype=torch.float32), "reward_extra_info": {"abc_trace": all_traces}}