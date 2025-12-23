import requests
import json
from typing import Dict, Optional, Any
import time
import uuid
from pathlib import Path


# --------------------------
# 从txt文件加载提示词模板（模块级常量）
# --------------------------
def load_prompt_template() -> str:
    """从同目录的txt文件加载C模型评判提示词模板"""
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "c_model_prompt.txt"  # 提示词文件路径

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Error: c_model_prompt.txt not found."
    except Exception as e:
        raise Exception(f"读取提示词模板失败：{str(e)}")


# 加载模板（仅在模块初始化时执行一次）
C_MODEL_PROMPT_TEMPLATE = load_prompt_template()


# --------------------------
# C模型评判类（适配Qwen3 API）
# --------------------------
class CModelJudge:
    def __init__(self, api_url: str, auth_token: str):
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "AIGC-USER": "zhuldz"
        }
        self.timeout = 60  # API超时时间（秒）
        self.retry_times = 5  # 重试次数

    def _send_request(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """发送API请求，带重试机制"""
        retry_count = 0
        while retry_count < self.retry_times:
            try:
                response = requests.post(
                    url=self.api_url,
                    headers=self.headers,
                    json=request_body,
                    timeout=self.timeout,
                    stream=False
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                retry_count += 1
                wait_time = 5 * (2 ** retry_count)
                print(f"请求失败（{retry_count}/{self.retry_times}）：{str(e)}")
                if remaining := self.retry_times - retry_count > 0:
                    print(f"将在{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"超过最大重试次数（{self.retry_times}次）：{str(e)}")

    def judge(
        self,
        model_c: str,
        prompt: str,
        generated_code: str,
        code_ground_truth: str,
        b_model_input: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """调用C模型进行评判，返回结构化结果"""
        # 1. 填充提示词模板
        judge_prompt = C_MODEL_PROMPT_TEMPLATE.format(
            prompt=prompt,
            generated_code=generated_code,
            code_ground_truth=code_ground_truth,
            b_model_input=b_model_input,
            user_prompt=user_prompt
        )

        # 2. 构造请求体
        request_body = {
            "model": model_c,
            "messages": [{"role": "user", "content": judge_prompt}],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.2,
            "max_tokens": 1024  # 增加Token数以容纳新的JSON结构
        }

        request_id = f"judge-{uuid.uuid4().hex[:10]}"
        
        try:
            response_json = self._send_request(request_body)
            return self._parse_judge_result(response_json, request_id)
        except Exception as e:
            return {
                "request_id": request_id,
                "valid": False,
                "error": f"评判请求失败：{str(e)}",
                "raw_response": None
            }

    def _parse_judge_result(self, response: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """解析Qwen3 API的响应，适配新的JSON结构"""
        try:
            if "choices" not in response: raise ValueError("响应中缺少'choices'字段")
            choices = response["choices"]
            if not isinstance(choices, list) or len(choices) == 0: raise ValueError("'choices'字段异常")
            
            message = choices[0].get("message", {})
            judge_text = message.get("content")
            
            # 兼容推理模型返回 reasoning_content 的情况
            if judge_text is None:
                judge_text = message.get("reasoning_content")
                if judge_text:
                    print("🔄 [API] 使用 reasoning_content 作为评判内容")
                else:
                    print(f"⚠️ [API异常] content为None. 完整响应: {json.dumps(response, ensure_ascii=False)[:500]}")
                    raise ValueError("API返回的 content 为空")

            if not isinstance(judge_text, str): judge_text = str(judge_text)
            judge_text = judge_text.strip()

            # 提取并解析JSON
            json_start = judge_text.find("{")
            json_end = judge_text.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError(f"输出不包含JSON，原始文本：{judge_text[:200]}...")
            
            try:
                judge_json = json.loads(judge_text[json_start:json_end])
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON格式错误：{str(e)}，原始片段：{judge_text[json_start:json_end][:100]}...")

            # [关键修改] 验证字段适配新结构
            # 新结构: quality_scores, violation_report, reason
            required_fields = ["quality_scores", "violation_report", "reason"]
            missing_fields = [f for f in required_fields if f not in judge_json]
            
            if missing_fields:
                # 兼容旧结构检查 (自动适配)
                if "total_score" in judge_json:
                    print("⚠️ [兼容] 检测到旧版 JSON 结构，正在进行自动适配...")
                    return {
                        "request_id": request_id,
                        "valid": True,
                        "result": {
                            "quality_scores": judge_json.get("score_details", {}),
                            "violation_report": {},
                            "reason": judge_json.get("reason", "")
                        },
                        "raw_response": response
                    }
                raise ValueError(f"评判结果缺少必要字段：{missing_fields}")

            return {
                "request_id": request_id,
                "valid": True,
                "result": judge_json,
                "raw_response": response
            }

        except Exception as e:
            return {
                "request_id": request_id,
                "valid": False,
                "error": str(e),
                "raw_response": response
            }


# --------------------------
# 批量处理JSON文件的函数（保留原功能）
# --------------------------
def process_json_evaluation(
    source_json_path: str,
    judge: CModelJudge,
    model_c: str,
    sleep_time: int = 3
):
    """读取源JSON文件，逐条评分后在当前目录生成新文件（源文件不变）"""
    source_path = Path(source_json_path)
    if not source_path.exists():
        raise FileNotFoundError(f"源JSON文件不存在：{source_json_path}")
    
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError(f"JSON数据必须是列表或字典，实际为：{type(data)}")
    
    total = len(data)
    print(f"\n=== 开始批量评分 ===")
    print(f"总数据量：{total}条，每条评分后休眠{sleep_time}秒")

    for i, item in enumerate(data, 1):
        print(f"\n--- 处理第{i}/{total}条数据 ---")
        
        # 检查必要字段
        # 注意：这里可能需要根据实际数据结构调整，比如 user_prompt 在哪里
        prompt = item.get("raw_problem", "").strip()
        generated_code = item.get("predict_code", "").strip()
        ground_truth = item.get("raw_solution", "").strip()
        # 这里假设 B模型输入 和 User Prompt 可能是同一个，或者从 item 中获取
        # 如果您的 json 里没有单独的 b_model_input，可以用 raw_problem 代替
        b_model_input = prompt 
        user_prompt = prompt 

        if not prompt or not generated_code:
            print("跳过：缺少关键内容")
            item["evaluation_result"] = {"valid": False, "error": "内容缺失"}
            continue
        
        try:
            eval_result = judge.judge(
                model_c=model_c,
                prompt=prompt,
                generated_code=generated_code,
                code_ground_truth=ground_truth,
                b_model_input=b_model_input,
                user_prompt=user_prompt
            )
            item["evaluation_result"] = eval_result
            print(f"第{i}条评分完成（ID：{eval_result.get('request_id')}）")
        except Exception as e:
            error_msg = f"评分失败：{str(e)}"
            print(error_msg)
            item["evaluation_result"] = {"valid": False, "error": error_msg}
        
        if i < total:
            time.sleep(sleep_time)
    
    # 生成新文件
    new_filename = f"evaluated_{source_path.name}"
    new_file_path = source_path.parent / new_filename

    with open(new_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n所有数据处理完成，新文件已生成：{new_file_path}")

if __name__ == "__main__":
    # 配置参数
    # API_URL = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1/chat/completions"
    # AUTH_TOKEN = "msk-8a895e7fa53a8785f9cc4dc0364fae9064ccc540bbd419b5ba7cde8340ec2af8"
    # MODEL_C = "/model/qwen3-235b-a22b"  # C模型标识
    # API_URL = "https://aimpapi.midea.com/t-aigc/aimp-deepseek-r1/v1/chat/completions"
    # AUTH_TOKEN = "msk-ca04571203246b31eec2dae635521ea079ca23818fdbe1f2177e17934382d378"
    # MODEL_C = "/model/DeepSeek-R1"  # C模型标识
    API_URL = "https://aimpapi.midea.com/t-aigc/f-devops-qwen3-coder-480b-a35b-instruct/v1/chat/completions"
    AUTH_TOKEN = "msk-4b8773bf749c892f2c9803aa69ef94b8b96e7cf807da78cbfdf8606ed919adef"
    MODEL_C = "f-devops-qwen3-coder-480b-a35b-instruct"  # C模型标识
    SOURCE_JSON_PATH = "/data/zhuldz/lunwen/generation/humaneval_pro.json"  # 替换为你的源JSON文件路径
    SLEEP_TIME = 5  # 每条数据评分后的休眠时间（秒）

    try:
        c_judge = CModelJudge(API_URL, AUTH_TOKEN)
    except Exception as e:
        print(f"初始化评判器失败：{str(e)}")
        exit(1)

    # 批量处理并生成新文件
    try:
        process_json_evaluation(
            source_json_path=SOURCE_JSON_PATH,
            judge=c_judge,
            model_c=MODEL_C,
            sleep_time=SLEEP_TIME
        )
    except Exception as e:
        print(f"处理失败：{str(e)}")
        exit(1)