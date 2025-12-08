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
# 2. 核心逻辑：反幻觉核实与算分 (新增)
# ===================================================================

def _verify_claim(claim_flag: bool, evidence: str, source_text: str, label: str) -> bool:
    """
    通用核实函数：验证模型提取的证据是否存在于源文本中
    """
    if not claim_flag:
        return False
        
    if not evidence or not isinstance(evidence, str) or evidence.lower() == "null":
        print(f"⚠️ [Hallucination] 模型判定 {label}=True 但未提供证据 -> 修正为 False")
        return False
    
    # 简单的去空白处理，提高匹配鲁棒性
    def clean(s): return str(s).replace("\n", "").replace(" ", "").replace("`", "").strip()
    
    # 检查证据是否在原文中
    if clean(evidence) in clean(source_text):
        print(f"🔴 [Verified {label}] 证据核实通过: {evidence[:30]}...")
        return True
    else:
        print(f"⚠️ [Hallucination] 模型判定 {label}=True 但证据与原文不符 -> 修正为 False")
        print(f"   - 伪造证据: {evidence[:50]}")
        return False

def calculate_final_reward_with_verification(
    judge_result_json: dict, 
    system_prompt: str,
    generated_code: str
) -> float:
    """
    计算最终得分：基础分 - 核实后的违规扣分
    """
    # 1. 提取常规质量分 (Quality Scores)
    q_scores = judge_result_json.get("quality_scores", {})
    # 兼容可能返回字符串的情况
    def get_float(k): return float(q_scores.get(k, 0))
    
    s1 = get_float("prompt_fluency")
    s2 = get_float("code_quality")
    s3 = get_float("function_consistency")
    base_score = s1 + s2 + s3

    # 2. 提取并核实违规报告 (Violation Report)
    v_report = judge_result_json.get("violation_report", {})
    deductions = 0.0
    
    # --- A. 真值泄露核实 (查 System Prompt) ---
    is_leakage = _verify_claim(
        v_report.get("has_code_leakage", False),
        v_report.get("leakage_quote"),
        source_text=system_prompt,
        label="Leakage"
    )
    if is_leakage:
        penalty = float(v_report.get("leakage_penalty_score", 0))
        # 限制扣分在 5-10 分之间
        deductions += max(5.0, min(10.0, penalty))

    # --- B. 代码含自然语言核实 (查 Generated Code) ---
    is_nl_in_code = _verify_claim(
        v_report.get("has_natural_language_in_code", False),
        v_report.get("natural_language_quote"),
        source_text=generated_code,
        label="NL_in_Code"
    )
    if is_nl_in_code:
        deductions += 4.0 # 固定扣4分

    # --- C. 提示词不可读核实 (查 System Prompt) ---
    is_unreadable = _verify_claim(
        v_report.get("is_prompt_unreadable", False),
        v_report.get("unreadable_quote"),
        source_text=system_prompt,
        label="Unreadable"
    )
    if is_unreadable:
        deductions += 5.0 # 固定扣5分

    # 3. 计算最终分
    final_score = base_score - deductions
    
    # 4. 兜底限制 (-10 ~ 10)
    return max(-10.0, min(10.0, final_score))

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
    model_c: str = "/model/qwen3-235b-a22b"
) -> Tuple[float, Dict]:
    """
    调用C模型计算奖励，包含完整反幻觉机制
    """
    try:
        judge_result = c_judge.judge(
            model_c=model_c,
            prompt=generated_prompt,
            generated_code=generated_code,
            code_ground_truth=canonical_solution,
            b_model_input=b_model_input,
            user_prompt=user_prompt
        )

        # 辅助函数：尝试从文本中提取JSON
        def parse_json_safely(text):
            try:
                if not text: return None
                # 尝试找到最外层大括号
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end != 0:
                    return json.loads(text[start:end])
            except:
                pass
            return None

        # 处理 API 返回结果
        if judge_result.get("valid", False):
            result_json = judge_result["result"]
        else:
            # 尝试抢救损坏的JSON
            err_msg = judge_result.get("error", "")
            raw_text = ""
            if "原始JSON部分" in err_msg:
                # 提取错误信息中的原始内容
                try:
                    raw_text = err_msg.split("原始JSON部分")[-1].strip(":").strip()
                except:
                    pass
            
            rescued_json = parse_json_safely(raw_text)
            
            if rescued_json and "quality_scores" in rescued_json:
                print("✅ [JSON Rescue] 成功从错误信息中提取有效JSON")
                result_json = rescued_json
                # 将抢救回来的结果写回，以便后续记录日志
                judge_result["result"] = rescued_json 
                judge_result["valid"] = True
            else:
                print(f"❌ 评判失败且无法修复: {err_msg[:100]}...")
                return 0.0, {"error": err_msg, "valid": False}

        # [关键步骤] 调用核实逻辑计算最终分数
        # 不再读取 total_score，而是根据分项和证据核实重新计算
        final_score = calculate_final_reward_with_verification(
            result_json, 
            system_prompt=generated_prompt,
            generated_code=generated_code
        )
        
        # 将计算后的真实分数写回 result 方便日志记录
        result_json["final_calculated_score"] = final_score
        
        return final_score, result_json

    except Exception as e:
        print(f"Pipeline Error: {e}")
        return 0.0, {"error": str(e), "valid": False}