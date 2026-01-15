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
import tempfile
import random
import traceback
import inspect 
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================================================================
# 1. 基础配置
# ===================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# 1. 拆分出目录路径
LOG_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout"
os.makedirs(LOG_DIR, exist_ok=True)

# 2. 构造带时间戳的文件名，例如: readable_trace_20240111_223000.log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
READABLE_LOG_PATH = os.path.join(LOG_DIR, f"readable_trace_{timestamp}.log")

print(f"📄 [Log] 本次日志文件: {READABLE_LOG_PATH}")

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
# 2. 核心参数
# ===================================================================
TRAIN_SET_PATH = os.environ.get("VERL_DATA_PATH", "/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax.parquet")
VALIDATION_SET_PATH = os.path.join(REPO_ROOT, "generation/humaneval_pro.json")

MAX_CONCURRENCY = 8
api_semaphore = threading.Semaphore(MAX_CONCURRENCY)

TRAINING_POOL_LIST: List[Tuple[str, str, Any]] = [] 
CLEAN_PROMPT_POOL: List[str] = [] 
PROMPT_TO_INDEX: Dict[str, int] = {} 
VALIDATION_POOL: List[Tuple[str, List[str]]] = []

# --- 权重配置 ---
GROUP_SIZE = 3               
WEIGHT_OBJ = 0.7             
WEIGHT_SUB = 0.3

# 关闭门控
PASS_RATE_GATING = -1.0       
FAIL_FAST_THRESHOLD = -15.0  

# --- 奖惩参数 ---
GT_LEAKAGE_THRESHOLD = 0.15
MIN_REWARD_FLOOR = -10.0

WEIGHT_LENGTH = 0.0          
BONUS_PERFECT_PASS = 3.0
PENALTY_SYNTAX_ERROR = -10.0 
PENALTY_BAD_FORMAT = -5.0    

# [修改] 重叠惩罚的最大权重 (满额重叠扣5分)
WEIGHT_OVERLAP_MAX = 5.0

# --- 月考 ---
HIGH_SCORE_THRESHOLD = 8.5   
TRIGGER_COUNT = 10           
EXAM_SIZE = 5                
EXAM_PASS_THRESHOLD = 0.6    

_consecutive_high_scores = 0
_counter_lock = threading.Lock()

# ===================================================================
# 3. 辅助函数
# ===================================================================

def _log_readable(data):
    try:
        with open(READABLE_LOG_PATH, "a", encoding="utf-8", buffering=1) as f:
            f.write(f"\n{'='*40} {datetime.now().strftime('%H:%M:%S')} {'='*40}\n")
            f.write(json.dumps(data, indent=2, ensure_ascii=False))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"❌ LOG ERROR: {e}")

def _extract_real_query(full_text: str) -> str:
    if not isinstance(full_text, str): return str(full_text)
    pattern = r"User Query:\s*(.*?)\s*Reference Solution:"
    match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    if "User Query:" in full_text:
        return full_text.split("User Query:")[-1].split("Reference Solution")[0].strip()
    return full_text.strip()

def _robust_clean(text: str) -> str:
    real_query = _extract_real_query(text)
    clean_text = re.sub(r'[^a-z0-9]', '', real_query.lower())
    return clean_text

def _find_index_brute_force(input_text: str) -> int:
    input_clean = _robust_clean(input_text)
    if not input_clean: return None
    key = input_clean[:100]
    if key in PROMPT_TO_INDEX: return PROMPT_TO_INDEX[key]
    for idx, pool_clean in enumerate(CLEAN_PROMPT_POOL):
        if (input_clean in pool_clean) or (pool_clean in input_clean):
            if len(pool_clean) > 20 and len(input_clean) > 20:
                return idx
    return None

def check_ground_truth_leakage(text, gt):
    if not text or not gt: return 0.0
    t_lines = set([l.strip() for l in text.split('\n') if len(l.strip()) > 10])
    g_lines = set([l.strip() for l in gt.split('\n') if len(l.strip()) > 10])
    if not t_lines: return 0.0
    return len(t_lines.intersection(g_lines)) / len(t_lines)

# [修改] 线性重叠惩罚计算
def calculate_overlap_penalty_value(sys_p, query):
    if not sys_p or not query: return 0.0
    
    # 1. 简单的分词并转小写
    s_tokens = set(sys_p.lower().split())
    q_tokens = set(query.lower().split())
    
    # 2. 移除停用词 (避免 "the", "a", "is" 导致误伤)
    # 这些词不包含任务信息，重叠了也没关系
    stopwords = {
        'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'is', 'are', 
        'be', 'was', 'were', 'it', 'that', 'this', 'you', 'your', 'my', 'python', 'code', 'write'
    }
    s_tokens = s_tokens - stopwords
    q_tokens = q_tokens - stopwords
    
    if not s_tokens: return 0.0
    
    # 3. 计算重叠的唯一词数量
    intersection = s_tokens.intersection(q_tokens)
    
    # 4. 计算比例：System Prompt 中有多少词是来自 User Query 的
    ratio = len(intersection) / len(s_tokens)
    
    # 5. 线性映射到 0 ~ 5 分
    # 比例越高，扣分越多
    penalty = ratio * WEIGHT_OVERLAP_MAX
    return penalty

def _unsafe_exec_worker(code, unit_tests, result_queue):
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                scope = {}
                try: exec(code, scope)
                except: 
                    result_queue.put({"status": "compile_error"})
                    return

                def get_func_name(s):
                    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", s)
                    return m.group(1) if m else None
                
                gen_name = get_func_name(code)
                test_name = None
                if unit_tests:
                    m = re.search(r"(?:assert|print)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", unit_tests[0])
                    test_name = m.group(1) if m else None

                if gen_name and test_name and gen_name != test_name:
                    if gen_name in scope: scope[test_name] = scope[gen_name]
                
                passed = 0
                total = len(unit_tests)
                if total == 0:
                    result_queue.put({"status": "no_tests", "pass_rate": 0.0})
                    return
                
                for test in unit_tests:
                    try:
                        exec(test.strip(), scope)
                        passed += 1
                    except: pass
                
                result_queue.put({"status": "success", "pass_rate": passed / total})
        except: 
            result_queue.put({"status": "runtime_error"})
        finally:
            try: os.chdir(original_cwd)
            except: pass

def compute_execution_pass_rate(code_str: str, unit_tests_data: Any) -> float:
    if not code_str: return -1.0
    try:
        clean_code = code_str
        if "```python" in code_str:
            clean_code = code_str.split("```python")[1].split("```")[0].strip()
        elif "```" in code_str:
            clean_code = code_str.split("```")[1].split("```")[0].strip()

        unit_tests = []
        if isinstance(unit_tests_data, list):
            unit_tests = unit_tests_data
        elif isinstance(unit_tests_data, str):
            str_data = unit_tests_data.strip()
            try: unit_tests = json.loads(str_data)
            except:
                try: unit_tests = ast.literal_eval(str_data)
                except: unit_tests = [line for line in str_data.split('\n') if line.strip()]
        
        if not unit_tests: return -2.0

        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_unsafe_exec_worker, args=(clean_code, unit_tests, queue))
        p.start()
        p.join(timeout=2.0)
        
        if p.is_alive(): p.terminate(); p.join(); return -1.0
        if not queue.empty():
            res = queue.get()
            if res["status"] == "success": return res["pass_rate"]
            
        return -1.0
    except: return -1.0

# ===================================================================
# 4. 数据加载
# ===================================================================

def _load_training_pool():
    global TRAINING_POOL_LIST, PROMPT_TO_INDEX, CLEAN_PROMPT_POOL
    if TRAINING_POOL_LIST: return
    
    print(f"⏳ Loading Dataset: {TRAIN_SET_PATH}")
    try:
        df = pd.read_parquet(TRAIN_SET_PATH)
        p_col = 'raw_input' if 'raw_input' in df.columns else 'prompt'
        
        count = 0
        skipped = 0
        
        for idx, row in df.iterrows():
            p = str(row[p_col]) if p_col in row else ""
            if not p: 
                skipped += 1
                continue
            
            gt = ""
            ut = []
            
            output_val = row.get('output')
            data_package = {}
            if isinstance(output_val, str):
                try: data_package = json.loads(output_val)
                except: data_package = {"ground_truth": output_val}
            elif isinstance(output_val, dict):
                data_package = output_val
            
            rm_val = row.get('reward_model')
            if isinstance(rm_val, dict):
                data_package.update(rm_val)
            elif isinstance(rm_val, str):
                try: data_package.update(json.loads(rm_val))
                except: pass

            gt = data_package.get('ground_truth', '')
            raw_ut = data_package.get('unit_tests', [])

            if isinstance(raw_ut, str):
                try: ut = json.loads(raw_ut)
                except:
                    try: ut = ast.literal_eval(raw_ut)
                    except: ut = []
            elif isinstance(raw_ut, list):
                ut = raw_ut
            
            if not ut:
                skipped += 1
                continue
            
            pool_idx = len(TRAINING_POOL_LIST)
            TRAINING_POOL_LIST.append((p, gt, ut))
            
            clean_p = _robust_clean(p)
            CLEAN_PROMPT_POOL.append(clean_p)
            key = clean_p[:100]
            PROMPT_TO_INDEX[key] = pool_idx
            
            count += 1
            
        print(f"✅ Pool Loaded: {count} items (Skipped {skipped})")
        if count > 0:
            print(f"🔑 Sample Clean Key [0]: {CLEAN_PROMPT_POOL[0][:50]}...")
            
    except Exception as e:
        print(f"❌ Load Failed: {e}\n{traceback.format_exc()}")

def _ensure_models_loaded():
    global B_MODEL, B_TOKENIZER, C_JUDGE, MODELS_LOADED
    if MODELS_LOADED: return
    if B_MODEL is None: B_MODEL, B_TOKENIZER = load_b_model()
    if C_JUDGE is None: C_JUDGE = load_c_model()
    MODELS_LOADED = True
    _load_training_pool()

def _load_validation_pool():
    global VALIDATION_POOL
    if VALIDATION_POOL: return
    try:
        if os.path.exists(VALIDATION_SET_PATH):
            with open(VALIDATION_SET_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    prompt = item.get('new_problem', item.get('prompt', ''))
                    raw_tests = item.get('test_code', item.get('test', ''))
                    tests = []
                    if isinstance(raw_tests, str):
                        tests = [line.strip() for line in raw_tests.split('\n') if line.strip()]
                    elif isinstance(raw_tests, list):
                        tests = raw_tests
                    if prompt and tests:
                        VALIDATION_POOL.append((prompt, tests))
    except: pass

def _run_monthly_exam(system_prompt: str) -> float:
    if not VALIDATION_POOL: return 0.0
    exam_questions = random.sample(VALIDATION_POOL, min(EXAM_SIZE, len(VALIDATION_POOL)))
    total_pass = 0.0
    clean_sys_p = re.sub(r'<think>.*?</think>', '', str(system_prompt), flags=re.DOTALL).strip()
    for q_prompt, q_tests in exam_questions:
        try:
            codes = generate_code_from_prompt(B_MODEL, B_TOKENIZER, clean_sys_p, q_prompt, k=1)
            pr = compute_execution_pass_rate(codes[0], q_tests)
            if pr > 0: total_pass += pr
        except: pass
    return total_pass / len(exam_questions)

# ===================================================================
# 5. 生成与评测 Wrappers
# ===================================================================

def _generate_codes(sys_prompt, prompt, k=1):
    try:
        codes = generate_code_from_prompt(B_MODEL, B_TOKENIZER, sys_prompt, prompt, k=k)
        return [{'code': c} for c in codes]
    except Exception as e:
        print(f"Gen Error: {e}")
        return [{'code': ''}] * k

def _evaluate_codes(codes, gt, sys_prompt, user_query):
    """
    [智能适配版] 自动探测函数签名，并使用关键字传参
    """
    scores = []
    results = []
    
    sys_prompt_str = str(sys_prompt) if sys_prompt else "No System Prompt"
    user_query_str = str(user_query) if user_query else "No User Query"
    gt_str = str(gt)
    
    try:
        sig = inspect.signature(compute_c_reward)
        params = list(sig.parameters.keys())
    except: params = []

    for item in codes:
        try:
            code_str = str(item['code'])
            
            # 方案 A: 关键字匹配
            if 'user_prompt' in params:
                kwargs = {
                    'user_prompt': user_query_str,
                    'system_prompt': sys_prompt_str
                }
                # 兼容可能的中间参数
                for p in params:
                    if p in ['context', 'history', 'dialogue']:
                        kwargs[p] = ""
                
                # 假设前三个是固定的 (Model, Code, GT)
                s, r = compute_c_reward(C_JUDGE, code_str, gt_str, **kwargs)
                
            else:
                # 方案 B: 回退到位置传参 (假设中间有一个空位)
                s, r = compute_c_reward(
                    C_JUDGE, code_str, gt_str, sys_prompt_str, "", user_query_str
                )

            val = float(s) if isinstance(s, (int, float)) else 0.0
            scores.append(val)
            results.append(r)
            
        except TypeError as te:
            # 方案 C: 如果上面都不行，尝试硬传6个参数
            try:
                s, r = compute_c_reward(C_JUDGE, code_str, gt_str, sys_prompt_str, "", user_query_str)
                val = float(s) if isinstance(s, (int, float)) else 0.0
                scores.append(val)
                results.append(r)
            except Exception as e2:
                err_msg = f"C-Model Call Failed. Sig: {params}. Error: {e2}"
                print(f"❌ [C-Model Fatal]: {err_msg}")
                scores.append(0.0)
                results.append({"error": err_msg})
                
        except Exception as e:
            print(f"❌ [C-Model Error]: {str(e)}")
            scores.append(0.0)
            results.append({"error": str(e)})
            
    avg = sum(scores)/len(scores) if scores else 0.0
    return avg, scores, results

def _get_c_model_template():
    path = os.path.join(REPO_ROOT, "judge/c_model_prompt.txt")
    if os.path.exists(path):
        with open(path, "r", encoding='utf-8') as f: return f.read().strip()
    return "{prompt}\n{user_prompt}\n{generated_code}\n{code_ground_truth}"

# ===================================================================
# 6. 主函数
# ===================================================================
def compute_custom_reward(**kwargs):
    _ensure_models_loaded()
    
    prompts = kwargs.get('data_sources') or kwargs.get('prompts')
    responses = kwargs.get('solution_strs') or kwargs.get('responses')
    config = kwargs.get('config', {})
    CURRENT_ITERATION = config.get("current_iteration", 0)

    tasks = []
    tmpl = _get_c_model_template()

    for i in range(len(responses)):
        p_idx = i % len(prompts)
        raw_p_obj = prompts[p_idx]
        current_q = raw_p_obj.get('raw_input', raw_p_obj.get('prompt', str(raw_p_obj))) if isinstance(raw_p_obj, dict) else str(raw_p_obj)

        tasks.append({
            "index": i,
            "original_input": current_q,
            "system_prompt": responses[i],
            "template": tmpl
        })

    def process_single_sample_batch(task):
        try:
            global _consecutive_high_scores
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
                
                total_score = 0.0
                test_count = 0
                
                raw_input_str = str(task["original_input"])
                center_idx = _find_index_brute_force(raw_input_str)
                
                test_queue = []
                if center_idx is not None:
                    group_indices = [center_idx + k for k in range(GROUP_SIZE)]
                    group_indices = [i for i in group_indices if i < len(TRAINING_POOL_LIST)]
                    for i in group_indices:
                        q_i, gt_i, ut_i = TRAINING_POOL_LIST[i]
                        t_type = "Center Task" if i == center_idx else f"Aux Task (+{i-center_idx})"
                        test_queue.append({"query": q_i, "gt": gt_i, "ut": ut_i, "type": t_type})
                else:
                    real_q = _extract_real_query(raw_input_str)
                    print(f"⚠️ Unmatched! Fallback using: {real_q[:50]}...")
                    test_queue.append({"query": real_q, "gt": "", "ut": [], "type": "Center Task (Unmatched)"})
                
                for t_idx, test_case in enumerate(test_queue):
                    test_query = test_case["query"]
                    test_gt = test_case["gt"]
                    test_ut = test_case["ut"]
                    test_type = test_case["type"]
                    
                    leakage = check_ground_truth_leakage(clean_sys_p, test_gt)
                    if leakage > GT_LEAKAGE_THRESHOLD:
                        case_score = MIN_REWARD_FLOOR
                        b_out = "[Leakage]"
                        c_res = {}
                        pass_rate = 0.0
                        c_log_input = "Skipped"
                    else:
                        codes_list = _generate_codes(clean_sys_p, test_query, k=1)
                        b_out = codes_list[0].get('code', '')
                        
                        pass_rate = compute_execution_pass_rate(b_out, test_ut)
                        
                        c_score_raw = 0.0
                        c_res = {}
                        c_log_input = ""
                        
                        try:
                            c_log_input = tmpl.format(prompt=clean_sys_p, user_prompt=test_query, generated_code=b_out, code_ground_truth=test_gt)
                            avg_val, _, results = _evaluate_codes(codes_list, test_gt, clean_sys_p, test_query)
                            c_score_raw = avg_val
                            c_res = results[0] if results else {}
                        except Exception as e:
                            print(f"C-Eval Block Error: {e}")
                            c_res = {"error": str(e)}
                            c_score_raw = 0.0 # 出错保底

                        # =======================================================
                        # [Step Reward Logic] 阶梯式客观评分 + 聚合主观评分
                        # =======================================================

                        # --- 1. 计算客观分 (Objective Score) ---
                        # 逻辑：如果不完美(pass_rate < 1.0)，最高只能拿 5 分；只有完美才给 10 分
                        if pass_rate < 0:
                            # 语法错误/无法执行 -> -10 分
                            obj_raw = PENALTY_SYNTAX_ERROR 
                        elif pass_rate < 1.0 - 1e-6:
                            # 不完美 -> 线性映射到 0 ~ 5 分 (和完美拉开差距)
                            obj_raw = pass_rate * 5.0
                        else:
                            # 完美 -> 10 分 (不再叠加 Bonus)
                            obj_raw = 10.0

                        # --- 2. 计算主观分 (Subjective Score) ---
                        # 包含：C模型打分 + 格式惩罚 + 重叠惩罚
                        sub_raw = float(c_score_raw)
                        

                        # [Penalty] 重叠惩罚 (移入主观分)
                        overlap_val = calculate_overlap_penalty_value(clean_sys_p, test_query)
                        sub_raw -= overlap_val

                        # [Clipping] 限制主观分在 [-10, 10] 之间
                        sub_raw = max(-10.0, min(10.0, sub_raw))

                        # --- 3. 最终加权求和 ---
                        # Final = (Obj * 0.7) + (Sub * 0.3)
                        raw_score = (obj_raw * WEIGHT_OBJ) + (sub_raw * WEIGHT_SUB)
                        
                        # --- 4. 地板与天花板 ---
                        case_score = max(raw_score, MIN_REWARD_FLOOR)
                        case_score = min(case_score, 10.0)

                    # 用于日志记录
                    case_info = {
                        "test_type": test_type,
                        "generated_code": b_out, 
                        "c_model_details": c_res,
                        "pass_rate": pass_rate,
                        "scores": {
                            "obj_raw": obj_raw,
                            "sub_raw": sub_raw,
                            "overlap_penalty": overlap_val
                        },
                        "final_case_score": float(case_score)
                    }
                    total_score += case_score
                    test_count += 1
                    trace_entry["group_tests"].append(case_info)
                    if case_score < FAIL_FAST_THRESHOLD: break

                avg_score = total_score / max(1, test_count)
                
                exam_log = "Skipped"
                is_triggered = False
                with _counter_lock:
                    if avg_score >= HIGH_SCORE_THRESHOLD:
                        _consecutive_high_scores += 1
                        if _consecutive_high_scores >= TRIGGER_COUNT:
                            is_triggered = True
                            _consecutive_high_scores = 0 
                    else: _consecutive_high_scores = 0 
                
                if is_triggered:
                    print(f"🚀 [Monthly Exam] Triggered. Avg Score: {avg_score:.2f}")
                    exam_pass_rate = _run_monthly_exam(clean_sys_p)
                    if exam_pass_rate < EXAM_PASS_THRESHOLD:
                        avg_score += -10.0
                        exam_log = f"FAILED ({exam_pass_rate:.2f})"
                    else:
                        avg_score += 3.0
                        exam_log = f"PASSED ({exam_pass_rate:.2f})"
                    avg_score = max(MIN_REWARD_FLOOR, min(15.0, avg_score))

                trace_entry["final_stats"] = {"avg_score": avg_score, "count": test_count, "monthly_exam": exam_log}
                _log_readable(trace_entry)
                return avg_score, json.dumps(trace_entry, ensure_ascii=False)
        
        except Exception as e:
            print(f"CRITICAL ERROR: {e}\n{traceback.format_exc()}")
            return MIN_REWARD_FLOOR, json.dumps({"error": str(e)})

    all_scores = [0.0] * len(tasks)
    all_traces = ["{}"] * len(tasks)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        futures = {executor.submit(process_single_sample_batch, t): t["index"] for t in tasks}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                s, t = fut.result()
                all_scores[idx] = s
                all_traces[idx] = t
            except:
                all_scores[idx] = MIN_REWARD_FLOOR
                all_traces[idx] = "{}"

    return {"reward_tensor": torch.tensor(all_scores), "reward_extra_info": {"abc_trace": all_traces}}