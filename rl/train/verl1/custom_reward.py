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
#from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

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

# ===================================================================
# [新增] 单元测试错误日志配置
# ===================================================================
# 这里的 LOG_DIR 应该复用你之前定义的 "/data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout"
ERROR_LOG_PATH = os.path.join(LOG_DIR, f"unittest_errors_{timestamp}.jsonl")
print(f"📄 [Log] 单元测试错误日志: {ERROR_LOG_PATH}")

def _log_unit_test_error(entry: Dict):
    """
    将错误信息追加写入 JSONL 文件
    """
    try:
        # 使用 json.dumps 保证单行原子性，通常不需要文件锁
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"❌ Failed to write error log: {e}")

B_MODEL = None
B_TOKENIZER = None
C_JUDGE = None
MODELS_LOADED = False

# Phase 2: 默认使用 BAgent 接口 (带 skill curation)
# 如需回退到旧 API 模式, 取消下面注释并将上一行注释:
#   from verl1.b_interface import load_b_model, generate_code_from_prompt
from verl1.b_agent_interface import load_b_model, generate_code_from_prompt, generate_code_with_skills_used
from verl1.c_interface import load_c_model, compute_reward as compute_c_reward

# ===================================================================
# 2. 核心参数
# ===================================================================
TRAIN_SET_PATH = os.environ.get("VERL_DATA_PATH", "/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax.parquet")
VALIDATION_SET_PATH = os.path.join(REPO_ROOT, "generation/humaneval_pro.json")

MAX_CONCURRENCY = 30  # B/C API 各上限 30
# 拆分信号量: B 和 C 独立控制，实现流水线化
b_semaphore = threading.Semaphore(MAX_CONCURRENCY)
c_semaphore = threading.Semaphore(MAX_CONCURRENCY)

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

# --- [新增] Self-Debug 奖励 (Task #5) ---
BONUS_SELF_DEBUG_FIX = 2.0        # self-debug 成功修正错误
BONUS_SELF_DEBUG_EFFICIENT = 1.0  # 1-2 轮修正成功 (效率高)
PENALTY_SELF_DEBUG_FAIL = -1.0    # 用了 self-debug 但最终仍失败

# --- [新增] 系统提示词质量惩罚 ---
# A模型输出退化的三种级别:
#   Level 1 (完全崩溃): "draft" / "" 等  → -15.0, 直接无效
#   Level 2 (严重退化): <100 chars      → -5.0
#   Level 3 (轻度可疑): <200 chars      → -1.0
PENALTY_EMPTY_OUTPUT = -15.0      # < 20 chars 直接判死
PENALTY_TOO_SHORT = -1.0          # < 100 chars (Phase 2: 放宽, A 输出可能是简洁的 curation 指令)
PENALTY_SUSPICIOUS = 0.0          # < 200 chars (Phase 2: 不惩罚, focus 在 curation 质量)
REPETITION_MAX_RATIO = 0.4        # 重复子串占比超过此值视为退化

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

# [新增] 系统提示词质量检测: 防退化/崩溃
def check_prompt_quality(sys_prompt: str) -> Tuple[float, str]:
    """
    检测系统提示词是否退化。返回 (penalty, reason)
    penalty: 负数表示扣分, 0 表示正常
    """
    if not sys_prompt or not isinstance(sys_prompt, str):
        return PENALTY_EMPTY_OUTPUT, "empty_or_none"

    clean = sys_prompt.strip()
    length = len(clean)

    # Level 1: 完全崩溃 (< 20 chars)
    if length < 20:
        return PENALTY_EMPTY_OUTPUT, f"collapsed_output_len={length}"

    # Level 2: 严重退化 (< 100 chars)
    if length < 100:
        return PENALTY_TOO_SHORT, f"too_short_len={length}"

    # Level 3: 轻度可疑 (< 200 chars)
    if length < 200:
        return PENALTY_SUSPICIOUS, f"suspicious_short_len={length}"

    # 重复检测: 检查最有问题的重复模式
    # 例如 "draft draft draft" 或 "draftdraftdraft"
    for window in [3, 5, 10]:
        if length >= window * 3:
            pattern = clean[:window]
            repeats = clean.count(pattern)
            if repeats >= 5:
                return PENALTY_TOO_SHORT, f"repetition_pattern_len={window}_repeats={repeats}"

    # 乱码检测: Unicode 脚本数量
    try:
        import unicodedata
        scripts_found = set()
        sample = clean[:500]
        for char in sample:
            if not char.isalpha():
                continue
            try:
                name = unicodedata.name(char, "").upper()
            except ValueError:
                continue
            if "LATIN" in name: scripts_found.add("LATIN")
            elif "CJK" in name: scripts_found.add("CJK")
            elif "CYRILLIC" in name: scripts_found.add("CYRILLIC")
            elif "ARABIC" in name: scripts_found.add("ARABIC")
            elif "HEBREW" in name: scripts_found.add("HEBREW")
            elif "GREEK" in name: scripts_found.add("GREEK")
            elif "THAI" in name: scripts_found.add("THAI")
            elif "HANGUL" in name: scripts_found.add("KOREAN")
            elif "HIRAGANA" in name or "KATAKANA" in name: scripts_found.add("JAPANESE")

        if len(scripts_found) >= 4:
            return PENALTY_BAD_FORMAT, f"unicode_chaos_scripts={len(scripts_found)}"
    except:
        pass

    return 0.0, "ok"

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
    import traceback
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            # 重定向 stdout/stderr 以保持控制台清洁
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                scope = {}
                # 1. 尝试编译和运行代码主体
                try: 
                    exec(code, scope)
                except Exception:
                    # [关键] 捕获语法错误或顶层运行时错误
                    err_msg = traceback.format_exc()
                    result_queue.put({"status": "compile_error", "msg": err_msg})
                    return

                # 2. 简单的函数名映射逻辑 (保持不变)
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
                
                # 3. 执行测试用例并收集第一个报错
                first_failure = None
                for test in unit_tests:
                    try:
                        exec(test.strip(), scope)
                        passed += 1
                    except Exception:
                        if not first_failure:
                            first_failure = traceback.format_exc()
                
                result_queue.put({
                    "status": "success", 
                    "pass_rate": passed / total,
                    "failure_trace": first_failure # 如果有失败，回传第一个报错
                })
        except Exception: 
            # 兜底捕获（如目录切换失败等）
            result_queue.put({"status": "runtime_error", "msg": traceback.format_exc()})
        finally:
            try: os.chdir(original_cwd)
            except: pass

def compute_self_debug_reward(debug_info: Dict, final_pass_rate: float) -> float:
    """
    计算 self-debug 相关奖励 (借鉴 ReSkill 的迭代验证奖励设计)。

    Args:
        debug_info: 从 BAgentInterface.generate_code_with_self_debug() 返回的 debug_info
        final_pass_rate: 最终代码的单元测试通过率 (0.0 ~ 1.0)

    Returns:
        self-debug 相关的奖励分数
    """
    if not debug_info or not debug_info.get("debug_trace"):
        return 0.0

    debug_trace = debug_info.get("debug_trace", [])
    had_debug = any(t.get("action") == "fix" for t in debug_trace)
    n_rounds = len(debug_trace)

    if not had_debug:
        return 0.0  # 没用 self-debug，不奖励不惩罚

    if final_pass_rate >= 1.0:
        # 最终全部通过
        if n_rounds <= 2:
            return BONUS_SELF_DEBUG_EFFICIENT + BONUS_SELF_DEBUG_FIX  # +3.0
        else:
            return BONUS_SELF_DEBUG_FIX  # +2.0
    elif final_pass_rate > 0:
        # 部分通过 — 给部分奖励
        return BONUS_SELF_DEBUG_FIX * final_pass_rate  # +0 ~ 2.0
    else:
        # 最终完全失败
        return PENALTY_SELF_DEBUG_FAIL  # -1.0


def compute_execution_pass_rate(code_str: str, unit_tests_data: Any, log_context: Dict = None) -> float:
    """
    log_context: 包含 'test_type', 'task_index' 等信息的字典，用于日志对齐
    """
    # 1. 基础判空
    if not code_str: 
        # 记录空代码错误
        if log_context:
            _log_unit_test_error({
                **log_context,
                "timestamp": datetime.now().isoformat(),
                "pass_rate": -3.0,
                "error_type": "EmptyCode",
                "error_msg": "Code string is empty or extraction failed",
                "code_snippet": ""
            })
        return -3.0

    TIMEOUT_SECONDS = 4.0 

    try:
        # ... (中间的代码清洗逻辑保持不变，请保留正则部分) ...
        # 为节省篇幅，这里假设 clean_code 已生成
        import re
        clean_code = code_str
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, code_str, re.DOTALL | re.IGNORECASE)
        if match: clean_code = match.group(1).strip()
        else:
            match_s = re.search(r"```\s*(.*?)\s*```", code_str, re.DOTALL)
            if match_s: clean_code = match_s.group(1).strip()

        if not clean_code: return -3.0 # 清洗后为空

        # 2. 解析测试用例 (保持不变)
        unit_tests = []
        if isinstance(unit_tests_data, list): unit_tests = unit_tests_data
        elif isinstance(unit_tests_data, str):
            try: unit_tests = json.loads(unit_tests_data)
            except: 
                try: unit_tests = ast.literal_eval(unit_tests_data)
                except: unit_tests = [l for l in unit_tests_data.split('\n') if l.strip()]
        
        if not unit_tests: return -3.0

        # 3. 执行
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_unsafe_exec_worker, args=(clean_code, unit_tests, queue))
        p.start()
        p.join(timeout=TIMEOUT_SECONDS)
        
        result = -3.0
        error_details = {"type": "Unknown", "msg": "Unknown Error"}

        # 4. 结果判定与错误信息提取
        if p.is_alive():
            p.terminate(); p.join()
            result = -1.0 # Timeout
            error_details = {"type": "Timeout", "msg": f"Execution exceeded {TIMEOUT_SECONDS}s"}
            
        elif not queue.empty():
            res = queue.get()
            status = res.get("status", "unknown")
            
            if status == "success":
                result = res["pass_rate"]
                if result == 0.0:
                    # 虽然运行成功但通过率为0，记录逻辑错误
                    error_details = {"type": "AllTestsFailed", "msg": res.get("failure_trace", "Wrong Answer")}
                elif result < 1.0:
                    # 部分通过，也可以记录一下
                    error_details = {"type": "PartialFailure", "msg": res.get("failure_trace", "Some tests failed")}
                    
            elif status == "compile_error":
                result = -3.0
                error_details = {"type": "CompileError", "msg": res.get("msg", "Syntax Error")}
            else:
                result = -2.0
                error_details = {"type": "RuntimeError", "msg": res.get("msg", "Runtime Error")}
        else:
            result = -3.0
            error_details = {"type": "ProcessCrash", "msg": "Process crashed (OOM?)"}

        # =========================================================
        # 📝 [核心] 写入错误日志 (当 pass_rate <= 0 时)
        # =========================================================
        if result <= 0.0 and log_context:
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": log_context.get("test_type", "Unknown"),
                "task_index": log_context.get("index", -1),
                "pass_rate": result,
                "error_summ": error_details["type"],
                "error_trace": error_details["msg"], # 具体的报错信息
                "code_snippet": clean_code[:1000]    # 截取前1000字符
            }
            _log_unit_test_error(log_entry)

        # 资源清理 (保持不变)
        try: queue.close(); queue.join_thread()
        except: pass
        try: p.close()
        except: pass

        return result

    except Exception as e:
        return -3.0

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
        skipped_py_file = 0 # 统计一下因为 .py 文件被过滤掉的数量
        
        for idx, row in df.iterrows():
            p = str(row[p_col]) if p_col in row else ""
            if not p: 
                skipped += 1
                continue
            
            # ... (中间的数据解析逻辑保持不变) ...
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

            # =========================================================
            # 🛑 [新增过滤逻辑] 过滤掉所有包含 .py 调用的测试用例
            # =========================================================
            # 将 list 转为 string 进行快速检查
            ut_str_check = str(ut).lower()
            
            # 检查关键词：.py, analyze.py, solution.py 等
            # 注意：我们要避开简单的 import xxx，主要针对 python xxx.py 这种命令
            if ".py" in ut_str_check:
                # 再次确认不是误伤（例如只是 print(".py")）
                # 这里我们稍微激进一点，只要测试用例里出现了 .py 文件名引用，就过滤掉
                # 因为这通常意味着它依赖文件系统操作
                skipped_py_file += 1
                continue
            # =========================================================
            
            pool_idx = len(TRAINING_POOL_LIST)
            TRAINING_POOL_LIST.append((p, gt, ut))
            
            clean_p = _robust_clean(p)
            CLEAN_PROMPT_POOL.append(clean_p)
            key = clean_p[:100]
            PROMPT_TO_INDEX[key] = pool_idx
            
            count += 1
            
        # 显式清理内存
        del df
        import gc
        gc.collect()
            
        print(f"✅ Pool Loaded: {count} items")
        print(f"   - Skipped (Empty/No Tests): {skipped}")
        print(f"   - Skipped (File Dependency): {skipped_py_file}") # 打印过滤数量
        
        if count > 0:
            print(f"🔑 Sample Clean Key [0]: {CLEAN_PROMPT_POOL[0][:50]}...")
            
    except Exception as e:
        print(f"❌ Load Failed: {e}\n{traceback.format_exc()}")

def _ensure_models_loaded():
    global B_MODEL, B_TOKENIZER, C_JUDGE, MODELS_LOADED
    if MODELS_LOADED: return
    if B_MODEL is None: B_MODEL, B_TOKENIZER = load_b_model()  # B_TOKENIZER 在 API 模式下为 None
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
    
    # 打印日志，确认进入函数
    print(f"📝 [Monthly Exam] Starting exam (Size={EXAM_SIZE})...", flush=True)
    
    exam_questions = random.sample(VALIDATION_POOL, min(EXAM_SIZE, len(VALIDATION_POOL)))
    total_pass = 0.0
    
    try:
        clean_sys_p = re.sub(r'<think>.*?</think>', '', str(system_prompt), flags=re.DOTALL).strip()
    except Exception as e:
        print(f"⚠️ [Monthly Exam] Regex Error: {e}", flush=True)
        clean_sys_p = "Generate python code."

    for idx, (q_prompt, q_tests) in enumerate(exam_questions):
        # [新增] 详细进度日志
        print(f"   📝 [Exam Q{idx+1}/{len(exam_questions)}] Generating...", flush=True)
        
        try:
            # [新增] B 模型超时保护 (60秒)
            def _gen_exam_code():
                return generate_code_from_prompt(B_MODEL, B_TOKENIZER, clean_sys_p, q_prompt, k=1)
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_gen_exam_code)
                try:
                    codes = future.result(timeout=60) # 强制 60s 超时
                except TimeoutError:
                    print(f"   ⏰ [Exam Q{idx+1}] Timeout! Skipping.", flush=True)
                    continue # 跳过这一题
                except Exception as e:
                    print(f"   ❌ [Exam Q{idx+1}] Gen Error: {e}", flush=True)
                    continue

            # 执行测试 (已有内部超时，无需额外加)
            if not codes: continue
            pr = compute_execution_pass_rate(codes[0], q_tests)
            if pr > 0: total_pass += pr
            
        except Exception as e:
            print(f"   ⚠️ [Exam Q{idx+1}] Unexpected Error: {e}", flush=True)
            pass

    score = total_pass / len(exam_questions)
    print(f"📝 [Monthly Exam] Finished. Score: {score:.2f}", flush=True)
    return score

# ===================================================================
# 5. 生成与评测 Wrappers
# ===================================================================

def _generate_codes(sys_prompt, prompt, k=1):
    """
    Phase 2: 使用 BAgentInterface 生成代码 (自动 skill curation)
    - sys_prompt: A 模型生成的 system prompt (作为任务特定 context 注入)
    - prompt: 用户原始问题 (用于 skill 检索 + 代码生成)

    Returns:
        (codes_list, skills_used) — codes_list 是 [{'code': str}, ...], skills_used 是 List[str]
    """
    try:
        codes, skills_all = generate_code_with_skills_used(B_MODEL, B_TOKENIZER, sys_prompt, prompt, k=k)
        # skills_all 是 List[List[str]] (第 i 个 code 对应一组 skill IDs)。
        # Step 1 全量注入下各组相同 (都注入同一批 active skill), 取并集展平为 List[str],
        # 否则下游 record_outcome(skills_ids_used) 会把整个内层 list 当成一个 sid
        # 传入 _skills.get() -> TypeError: unhashable type: 'list' (被 try/except 吞掉,
        # Step 2 在线反馈静默失效)。
        skills_used = list(dict.fromkeys(s for group in skills_all for s in group)) if skills_all else []
        return [{'code': c} for c in codes], skills_used
    except Exception as e:
        print(f"Gen Error: {e}")
        return [{'code': ''}] * k, []


def _generate_codes_with_self_debug(sys_prompt, prompt, max_debug_rounds=3):
    """
    Phase 3 (Task #5): 使用 BAgentInterface 的 self-debug 模式生成代码.

    Returns:
        (codes_list, debug_info)
        - codes_list: [{'code': str}]
        - debug_info: {"debug_trace": [...], "n_rounds": int, ...}
    """
    try:
        # 获取 BAgentInterface 实例
        from verl1.b_agent_interface import get_b_agent_interface
        b_agent_interface = get_b_agent_interface()

        code, debug_info = b_agent_interface.generate_code_with_self_debug(
            generated_prompt=sys_prompt,
            original_prompt=prompt,
            max_debug_rounds=max_debug_rounds,
        )
        return [{'code': code}], debug_info
    except Exception as e:
        print(f"Self-Debug Gen Error: {e}")
        return [{'code': ''}], {"debug_trace": [], "n_rounds": 0, "had_self_debug": False, "errors_fixed": 0, "skills_used": []}

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
            
            # [日志] 1. 开始处理 (截取前20字符作为ID)
            task_id_short = str(task["original_input"])[:20].replace('\n', '')
            print(f"▶️ [Start] Processing {task_id_short}...", flush=True)

            print(f"   [Debug] Running Regex for {task_id_short}...", flush=True)
            clean_sys_p = re.sub(r'<think>.*?</think>', '', str(sys_p), flags=re.DOTALL).strip()
            if not clean_sys_p: clean_sys_p = "Generate python code."
            print(f"   [Debug] Regex Done. Waiting for Semaphore...", flush=True)

            # [新增] 系统提示词质量检测: 退化/崩溃直接返回惩罚, 跳过B/C模型调用
            quality_penalty, quality_reason = check_prompt_quality(clean_sys_p)
            if quality_penalty <= PENALTY_EMPTY_OUTPUT:
                print(f"🚫 [Quality] Degenerate prompt ({quality_reason}), returning {quality_penalty}", flush=True)
                return quality_penalty, json.dumps(
                    {"error": f"degenerate_prompt:{quality_reason}", "score": quality_penalty},
                    ensure_ascii=False
                )
            elif quality_penalty < 0:
                print(f"⚠️ [Quality] Suboptimal prompt ({quality_reason}), penalty={quality_penalty}", flush=True)

            tmpl = task["template"]

            # 拆分 B/C 信号量实现流水线: B生成→释放→本地单测→C评判
            print(f"   [Debug] Starting pipeline for {task_id_short}...", flush=True)
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
                test_queue.append({"query": real_q, "gt": "", "ut": [], "type": "Center Task (Unmatched)"})

            for t_idx, test_case in enumerate(test_queue):
                test_query = test_case["query"]
                test_gt = test_case["gt"]
                test_ut = test_case["ut"]
                test_type = test_case["type"]

                # [Step 2] 兜底初始化 (leakage 分支也需要, 避免 case_info 引用未定义变量)
                skills_ids_used = []
                self_debug_info = None
                obj_raw = 0.0
                sub_raw = 0.0
                leakage = check_ground_truth_leakage(clean_sys_p, test_gt)

                if leakage > GT_LEAKAGE_THRESHOLD:
                    case_score = MIN_REWARD_FLOOR
                    b_out = "[Leakage]"
                    c_res = {}
                    pass_rate = 0.0
                    print(f"🚫 [{t_idx}] Leakage Detected!", flush=True)
                else:
                    # =====================================================
                    # [B-Model] 仅 B 调用受 b_semaphore 控制 (30并发)
                    # =====================================================
                    print(f"   [{t_idx}] Generating code (Timeout=60s)...", flush=True)
                    b_out = ""
                    codes_list = [{'code': ''}]
                    self_debug_info = None  # Phase 3 (Task #5): self-debug trace

                    with b_semaphore:
                        gen_executor = ThreadPoolExecutor(max_workers=1)
                        skills_ids_used = []
                        try:
                            # Phase 3: 支持 self-debug 模式
                            use_self_debug = config.get("use_self_debug", False)
                            if use_self_debug:
                                def _run_gen():
                                    return _generate_codes_with_self_debug(clean_sys_p, test_query)
                                future = gen_executor.submit(_run_gen)
                                result = future.result(timeout=120)  # self-debug 需要更长时间
                                codes_list, self_debug_info = result
                                skills_ids_used = self_debug_info.get("skills_used", [])
                            else:
                                def _run_gen():
                                    return _generate_codes(clean_sys_p, test_query, k=1)
                                future = gen_executor.submit(_run_gen)
                                result = future.result(timeout=60)
                                codes_list, skills_ids_used = result

                            b_out = codes_list[0].get('code', '')
                        except TimeoutError:
                            print(f"⏰ [{t_idx}] B-Model Gen TIMEOUT! Skipping...", flush=True)
                        except Exception as e:
                            print(f"❌ [{t_idx}] B-Model Error: {e}", flush=True)
                        finally:
                            gen_executor.shutdown(wait=False)


                        # =====================================================
                        # [Step] 执行单元测试 (修改版：带错误日志记录)
                        # =====================================================
                        
                        # 1. 构造日志上下文 (Log Context)
                        # 这些信息会被写入到 JSONL 文件中，方便你定位是哪条数据的哪个任务出错了
                        log_ctx = {
                            "test_type": test_case["type"],   # 例如 "Center Task" 或 "Aux Task"
                            "index": task["index"],           # 数据的全局索引
                            "iteration": CURRENT_ITERATION,   # 当前训练迭代轮次
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        # 2. 调用升级后的判分函数 (传入 log_context)
                        # 注意：你必须先按上一步修改好 compute_execution_pass_rate 的定义，让它支持这个参数
                        pass_rate = compute_execution_pass_rate(b_out, test_ut, log_context=log_ctx)
                        
                        # =====================================================
                        
                        c_score_raw = 0.0
                        c_res = {}
                        
                        # =====================================================
                        # [C-Model] 仅 C 调用受 c_semaphore 控制 (30并发, 与B独立流水线)
                        # =====================================================
                        print(f"⚖️ [{t_idx}] Calling C-Model (Timeout=45s)...", flush=True)
                        c_executor = ThreadPoolExecutor(max_workers=1)
                        with c_semaphore:
                            try:
                                c_log_input = tmpl.format(prompt=clean_sys_p, user_prompt=test_query, generated_code=b_out, code_ground_truth=test_gt)

                                def _run_c_eval():
                                    return _evaluate_codes(codes_list, test_gt, clean_sys_p, test_query)

                                future = c_executor.submit(_run_c_eval)
                                avg_val, _, results = future.result(timeout=45)

                                c_score_raw = avg_val
                                c_res = results[0] if results else {}
                                print(f"✅ [{t_idx}] C-Model Finished.", flush=True)

                            except TimeoutError:
                                print(f"⏰ [{t_idx}] C-Model TIMEOUT! Skipping...", flush=True)
                                c_score_raw = 0.0
                                c_res = {"error": "Timeout"}
                            except Exception as e:
                                print(f"❌ [{t_idx}] C-Eval Error: {e}", flush=True)
                                c_score_raw = 0.0
                                c_res = {"error": str(e)}
                            finally:
                                c_executor.shutdown(wait=False)

                        # =====================================================
                        # [Score] 细粒度分数映射 (Error Hierarchy)
                        # =====================================================
                        
                        # 1. 💀 语法错误 (-3.0) -> 最低分 -10
                        if pass_rate <= -2.5:
                            obj_raw = PENALTY_SYNTAX_ERROR # -10.0
                        
                        # 2. ⚠️ 运行时错误 (-2.0) -> 次低分 -4
                        # 代码能编译但崩了
                        elif pass_rate <= -1.5:
                            obj_raw = -4.0 
                        
                        # 3. ⏳ 超时 (-1.0) -> 微惩罚 -2
                        # 代码没崩但慢
                        elif pass_rate <= -0.5:
                            obj_raw = -2.0
                        
                        # 4. 📈 正常运行 (>= 0.0)
                        else:
                            if pass_rate < 1e-6:
                                # ❌ 逻辑错误 (0/N) -> 0 分
                                obj_raw = 0.0
                                
                            elif pass_rate < 1.0 - 1e-6:
                                # 📈 部分正确 -> 线性映射到 8.0 分 (突变前夜)
                                # 比如 0.9 * 8.0 = 7.2 分
                                # 这样给模型一个信号：你离完美还有差距，不要停在 0.9
                                obj_raw = pass_rate * 7.5
                                
                            else:
                                # 🎉 全对 -> 满分 10 (突变!)
                                # 从 0.99(约8分) 跳变到 10分，巨大的正向激励
                                obj_raw = 10.0

                        sub_raw = float(c_score_raw)
                        overlap_val = calculate_overlap_penalty_value(clean_sys_p, test_query)
                        sub_raw -= overlap_val
                        sub_raw = max(-10.0, min(10.0, sub_raw))

                        raw_score = (obj_raw * WEIGHT_OBJ) + (sub_raw * WEIGHT_SUB)
                        # [新增] 应用系统提示词质量惩罚
                        raw_score += quality_penalty
                        # [新增] Phase 3 (Task #5): 应用 self-debug 奖励
                        if self_debug_info:
                            sd_reward = compute_self_debug_reward(self_debug_info, pass_rate)
                            raw_score += sd_reward
                            if sd_reward != 0:
                                print(f"🔧 Self-Debug reward: {sd_reward:.1f}", flush=True)
                        case_score = max(raw_score, MIN_REWARD_FLOOR)
                        case_score = min(case_score, 10.0)
                        # [Step 2 在线反馈] 把本次 pass_rate 回写到所用 skill 的 online_success_rate (EMA)
                        # (不触碰 score_delta; 因果 ATE 由 offline prune_eval 负责)
                        if config.get("enable_skill_scoring", True) and skills_ids_used and B_MODEL is not None and hasattr(B_MODEL, "skill_lib"):
                            try:
                                B_MODEL.skill_lib.record_outcome(skills_ids_used, pass_rate)
                            except Exception as _e:
                                print(f"⚠️ record_outcome failed: {_e}", flush=True)

                    case_info = {
                        "test_type": test_type,
                        "generated_code": b_out,
                        "c_model_details": c_res,
                        "pass_rate": pass_rate,
                        "scores": {"obj_raw": obj_raw, "sub_raw": sub_raw},
                        "skills_used": skills_ids_used,       # Phase 2: 使用的 skill IDs
                        "debug_trace": (self_debug_info or {}).get("debug_trace", []),  # [Step 4] self-debug 多步轨迹, 供蒸馏
                        "final_case_score": float(case_score)
                    }
                    total_score += case_score
                    test_count += 1
                    trace_entry["group_tests"].append(case_info)
                    if case_score < FAIL_FAST_THRESHOLD: break

                avg_score = total_score / max(1, test_count)
                
                exam_log = "Disabled_Debug"
                # # [Monthly Exam] 逻辑保持不变
                # exam_log = "Skipped"
                # is_triggered = False
                # with _counter_lock:
                #     if avg_score >= HIGH_SCORE_THRESHOLD:
                #         _consecutive_high_scores += 1
                #         if _consecutive_high_scores >= TRIGGER_COUNT:
                #             is_triggered = True
                #             _consecutive_high_scores = 0 
                #     else: _consecutive_high_scores = 0 
                
                # if is_triggered:
                #     print(f"🚀 [Monthly Exam] Triggered. Avg Score: {avg_score:.2f}", flush=True)
                #     exam_pass_rate = _run_monthly_exam(clean_sys_p)
                #     if exam_pass_rate < EXAM_PASS_THRESHOLD:
                #         avg_score += -10.0
                #         exam_log = f"FAILED ({exam_pass_rate:.2f})"
                #     else:
                #         avg_score += 3.0
                #         exam_log = f"PASSED ({exam_pass_rate:.2f})"
                #     avg_score = max(MIN_REWARD_FLOOR, min(15.0, avg_score))

                trace_entry["final_stats"] = {"avg_score": avg_score, "count": test_count, "monthly_exam": exam_log}
                _log_readable(trace_entry)
                
                # 打印最终完成日志
                print(f"🏁 [Done] Score: {avg_score:.2f}", flush=True)
                return avg_score, json.dumps(trace_entry, ensure_ascii=False)
        
        except Exception as e:
            print(f"🔥 CRITICAL ERROR in Task: {e}\n{traceback.format_exc()}", flush=True)
            return MIN_REWARD_FLOOR, json.dumps({"error": str(e)})

    all_scores = [0.0] * len(tasks)
    all_traces = ["{}"] * len(tasks)
    #import traceback
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)
    try:
        futures = {executor.submit(process_single_sample_batch, t): t["index"] for t in tasks}
        
        try:
            # ⏳ [关键修改] 加上 timeout=240 (4分钟)
            # 如果 Step 271 再次死锁，这里会强制抛出 TimeoutError
            for fut in as_completed(futures, timeout=240): 
                idx = futures[fut]
                try:
                    s, t = fut.result(timeout=1) # 结果已就绪，取值应该很快
                    all_scores[idx] = s
                    all_traces[idx] = t
                except Exception as e:
                    print(f"❌ [Error] Task {idx} Failed inside thread: {e}", flush=True)
                    traceback.print_exc()
                    all_scores[idx] = MIN_REWARD_FLOOR
                    all_traces[idx] = json.dumps({"error": str(e)})
                    
        except TimeoutError:
            # 🚨 这就是解决“卡在271”的救命稻草
            print(f"🔥 [CRITICAL] Batch Processing TIMEOUT (Deadlock suspected)!", flush=True)
            print(f"   Skipping remaining tasks in Step to keep training alive.", flush=True)
            # 没跑完的任务会自动保持为 0.0，训练将继续进入 Step 272
            
    except Exception as e:
            print(f"🔥 [CRITICAL] Batch Executor Error: {e}", flush=True)
            traceback.print_exc()
    
    finally:
        # 🛑 [核心] wait=False 表示：我不等你们了，直接关机
        # 这会立即释放主线程，让训练继续。僵尸线程会在后台慢慢消亡（或被系统回收）
        # 即使僵尸线程还活着，因为没有持有信号量，它们也不会卡住下一轮训练！
        executor.shutdown(wait=False)

    # [Step 2] 批末持久化在线评分 (online_success_rate / usage_count) 到磁盘
    try:
        if B_MODEL is not None and hasattr(B_MODEL, "skill_lib"):
            B_MODEL.skill_lib.flush()
    except Exception as _e:
        print(f"⚠️ skill_lib flush failed: {_e}", flush=True)

    return {"reward_tensor": torch.tensor(all_scores), "reward_extra_info": {"abc_trace": all_traces}}