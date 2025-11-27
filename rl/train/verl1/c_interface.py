import sys
import json
# 将C模型脚本（qwen_235B.py）所在目录添加到Python路径
sys.path.append("/data/zhuldz/lunwen/judge")

try:
    from qwen_235B import CModelJudge
except ImportError:
    class CModelJudge:
        def __init__(self, **kwargs): pass
        def judge(self, **kwargs): return {"valid": False, "error": "Mock Judge: Import Failed"}

from typing import Optional, Tuple, Dict

def load_c_model(
    api_url: str = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1/chat/completions",
    auth_token: str = "msk-8a895e7fa53a8785f9cc4dc0364fae9064ccc540bbd419b5ba7cde8340ec2af8"
) -> CModelJudge:
    """加载C模型评判器"""
    try:
        judge = CModelJudge(api_url=api_url, auth_token=auth_token)
        print("C模型评判器加载成功")
        return judge
    except Exception as e:
        print(f"C模型评判器加载失败：{str(e)}")
        raise

# --- JSON 自动修复函数 ---
def _try_fix_json_error(error_msg: str) -> Optional[float]:
    try:
        if "原始JSON部分" not in error_msg:
            return None
        if "原始JSON部分：" in error_msg:
            raw_json = error_msg.split("原始JSON部分：")[-1].strip()
        else:
            raw_json = error_msg.split("原始JSON部分:")[-1].strip()
            
        for i in range(5):
            candidate = raw_json if i == 0 else raw_json[:-i]
            if not candidate: continue
            try:
                data = json.loads(candidate)
                if "total_score" in data:
                    score = float(data["total_score"])
                    print(f"⚠️ [Auto-Fix] 成功修复JSON格式错误，挽回分数: {score}")
                    return score
            except json.JSONDecodeError:
                continue
        return None
    except Exception:
        return None

def compute_reward(
    c_judge: CModelJudge,
    generated_code: str,
    canonical_solution: str,
    generated_prompt: str,
    b_model_input: str,
    model_c: str = "/model/qwen3-235b-a22b"
) -> Tuple[float, Dict]:  # <--- [修改] 返回类型变为元组
    """
    调用C模型计算奖励
    Returns:
        (score, full_result_dict)
    """
    try:
        judge_result = c_judge.judge(
            model_c=model_c,
            prompt=generated_prompt,
            generated_code=generated_code,
            code_ground_truth=canonical_solution,
            b_model_input=b_model_input
        )

        # 1. 正常情况
        if judge_result.get("valid", False):
            total_score = float(judge_result["result"].get("total_score", 0))
            # 返回分数和完整的 result 字典（包含 reason 等）
            return total_score, judge_result["result"]
        
        # 2. 异常情况
        else:
            error_msg = judge_result.get("error", "未知错误")
            
            # 尝试修复
            fixed_score = _try_fix_json_error(error_msg)
            if fixed_score is not None:
                # 构造一个修复后的伪结果用于日志
                fixed_result = {
                    "total_score": fixed_score,
                    "reason": f"Auto-fixed from error: {error_msg[:100]}...",
                    "valid": True
                }
                return fixed_score, fixed_result

            print(f"评判无效，奖励设为0（错误：{error_msg[:100]}...）")
            # 返回 0 分和错误信息
            return 0.0, {"error": error_msg, "valid": False}

    except Exception as e:
        print(f"计算奖励失败，奖励设为0（异常：{str(e)}）")
        return 0.0, {"error": str(e), "valid": False}

if __name__ == "__main__":
    # 测试代码
    print("Test passed")