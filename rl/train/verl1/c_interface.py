import sys
import os
import json
import re
from typing import Optional, Tuple, Dict

# ===================================================================
# 1. 路径修复
# ===================================================================
JUDGE_DIR = "/data/zhuldz/lunwen/judge"
if JUDGE_DIR not in sys.path:
    sys.path.insert(0, JUDGE_DIR)

try:
    from qwen_235B import CModelJudge
    print(f"✅ 成功从 {JUDGE_DIR} 导入 CModelJudge")
except ImportError as e:
    print(f"❌ 导入 CModelJudge 失败: {e}")
    class CModelJudge:
        def __init__(self, **kwargs): pass
        def judge(self, **kwargs): return {"valid": False, "error": f"Import Failed: {e}"}

# ===================================================================
# 2. 核心逻辑：反幻觉核实与详细算分
# ===================================================================

def _verify_claim(claim_flag: bool, evidence: str, source_text: str, label: str, index: int = None, timestamp: str = None) -> bool:
    """
    通用核实函数
    timestamp: 用于日志定位的唯一时间戳
    """
    # 构建日志前缀 [Time: 2025-...] [Case: 0]
    log_prefix = ""
    if timestamp: log_prefix += f"[Time: {timestamp}] "
    if index is not None: log_prefix += f"[Case: {index}] "

    if not claim_flag: return False
    
    if not evidence or not isinstance(evidence, str) or evidence.lower() == "null":
        print(f"{log_prefix}⚠️ [Hallucination] 模型判定 {label}=True 但未提供证据 -> 修正为 False")
        return False
    
    def clean(s): return str(s).replace("\n", "").replace(" ", "").replace("`", "").strip()
    
    if clean(evidence) in clean(source_text):
        print(f"{log_prefix}🔴 [Verified {label}] 证据核实通过: {evidence[:30]}...")
        return True
    else:
        # 这里的日志现在会带上时间戳，方便您直接搜索
        print(f"{log_prefix}⚠️ [Hallucination] 模型判定 {label}=True 但证据与原文不符 -> 修正为 False")
        return False

def calculate_final_reward_with_verification(
    judge_result_json: dict, 
    system_prompt: str,
    generated_code: str,
    index: int = None,
    timestamp: str = None # [新增]
) -> float:
    """
    计算最终得分
    """
    q_scores = judge_result_json.get("quality_scores", {})
    def get_float(k): return float(q_scores.get(k, 0))
    s1 = get_float("prompt_fluency")
    s2 = get_float("code_quality")
    s3 = get_float("function_consistency")
    base_score = s1 + s2 + s3

    v_report = judge_result_json.get("violation_report", {})
    deductions = 0.0
    trace_log = {}

    # A. 泄露核实
    is_leakage = _verify_claim(
        v_report.get("has_code_leakage", False),
        v_report.get("leakage_quote"),
        system_prompt, "Leakage", index, timestamp
    )
    if is_leakage:
        p = float(v_report.get("leakage_penalty_score", 0))
        d_val = max(10.0, min(15.0, p))
        deductions += d_val
        trace_log["leakage_deduction"] = -d_val
    else:
        trace_log["leakage_deduction"] = 0.0

    # B. 自然语言核实
    is_nl = _verify_claim(
        v_report.get("has_natural_language_in_code", False),
        v_report.get("natural_language_quote"),
        generated_code, "NL_in_Code", index, timestamp
    )
    if is_nl:
        deductions += 4.0
        trace_log["nl_in_code_deduction"] = -4.0
    else:
        trace_log["nl_in_code_deduction"] = 0.0

    # C. 乱码核实
    is_unreadable = _verify_claim(
        v_report.get("is_prompt_unreadable", False),
        v_report.get("unreadable_quote"),
        system_prompt, "Unreadable", index, timestamp
    )
    if is_unreadable:
        deductions += 5.0
        trace_log["unreadable_deduction"] = -5.0
    else:
        trace_log["unreadable_deduction"] = 0.0

    final_score = base_score - deductions
    final_score = max(-15.0, min(10.0, final_score))

    trace_log["base_score_sum"] = base_score
    trace_log["total_deductions"] = -deductions
    trace_log["final_score_clipped"] = final_score
    judge_result_json["calculation_trace"] = trace_log

    return final_score

# ===================================================================
# 3. 接口定义
# ===================================================================
def load_c_model(
    api_url: str = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1/chat/completions",
    auth_token: str = "msk-8a895e7fa53a8785f9cc4dc0364fae9064ccc540bbd419b5ba7cde8340ec2af8"
) -> CModelJudge:
    try:
        judge = CModelJudge(api_url=api_url, auth_token=auth_token)
        return judge
    except Exception as e:
        print(f"C模型评判器加载失败：{str(e)}")
        raise

def compute_reward(
    c_judge: CModelJudge,
    generated_code: str,
    canonical_solution: str,
    generated_prompt: str,
    b_model_input: str,
    user_prompt: str,
    model_c: str = "/model/qwen3-235b-a22b",
    index: int = None,
    timestamp: str = None # [新增] 接收时间戳
) -> Tuple[float, Dict]:
    try:
        judge_result = c_judge.judge(
            model_c=model_c,
            prompt=generated_prompt,
            generated_code=generated_code,
            code_ground_truth=canonical_solution,
            b_model_input=b_model_input,
            user_prompt=user_prompt
        )

        def parse_json_safely(text):
            try:
                if not text: return None
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end != 0:
                    return json.loads(text[start:end])
            except: pass
            return None

        if judge_result.get("valid", False):
            result_json = judge_result["result"]
        else:
            err_msg = judge_result.get("error", "")
            raw_text = ""
            if "原始JSON部分" in err_msg:
                try: raw_text = err_msg.split("原始JSON部分")[-1].strip(":").strip()
                except: pass
            
            rescued_json = parse_json_safely(raw_text)
            if rescued_json and "quality_scores" in rescued_json:
                # 只有成功抢救才打印，避免刷屏，带上时间戳方便定位
                print(f"[Time: {timestamp}] [Case: {index}] ✅ [JSON Rescue] 成功抢救分数")
                result_json = rescued_json
                judge_result["result"] = rescued_json
            else:
                return 0.0, {"error": err_msg, "valid": False}

        final_score = calculate_final_reward_with_verification(
            result_json, 
            system_prompt=generated_prompt,
            generated_code=generated_code,
            index=index,
            timestamp=timestamp # [传递]
        )
        
        result_json["final_calculated_score"] = final_score
        return final_score, result_json

    except Exception as e:
        print(f"[Time: {timestamp}] [Case: {index}] Pipeline Error: {e}")
        return 0.0, {"error": str(e), "valid": False}