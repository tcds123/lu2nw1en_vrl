import requests
import json
import logging
import time
from typing import List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class BModelAPI:
    """基于API调用的B模型替代方案"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or "msk-1aeb9a660245cc33b21ee04e67fb88da012808b548b794e07aba1cde1591f8c6"
        self.base_url = base_url or "https://aimpapi.midea.com/t-aigc/aimp-qwen2-5-72b-ascend/v1/chat/completions"
        self.model_name = "Qwen25-72B-Instruct"
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "AIGC-USER":"zhuldz"
        }
        
        self.generation_params = {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 2048,
            "n": 1,
            "stream": False
        }
        
        logging.info("B模型API客户端初始化完成")

    # [修改] 重构生成函数，支持传入 system_prompt
    def generate_code(self, user_query: str, system_prompt: str = None) -> str:
        """
        调用API生成代码
        Args:
            user_query: 用户的原始问题
            system_prompt: A模型生成的系统提示词 (如果为None，则使用默认)
        """
        max_retries = 5
        base_wait_time = 5
        
        # 如果没有提供A模型的prompt，使用默认兜底，但在RL训练中通常都有
        final_system_prompt = system_prompt if system_prompt else "You are a professional code generation assistant. Generate correct and efficient code."

        for attempt in range(max_retries):
            try:
                # [核心修正] 正确构建消息结构
                messages = [
                    {
                        "role": "system", 
                        "content": final_system_prompt  # <--- 这里填入 A 模型的输出
                    },
                    {
                        "role": "user",
                        "content": user_query           # <--- 这里填入 原始问题
                    }
                ]
                
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    **self.generation_params
                }
                
                start_time = time.time()
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=data,
                    timeout=60
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        generated_code = result["choices"][0]["message"]["content"].strip()
                        return generated_code
                    else:
                        logging.error(f"API响应格式异常: {result}")
                        return ""
                
                elif response.status_code == 429:
                    logging.warning(f"触发限流 (429)，尝试重试... ({attempt + 1}/{max_retries})")
                    time.sleep(base_wait_time * (2 ** attempt))
                    continue
                
                else:
                    logging.error(f"API调用失败 - 状态码: {response.status_code}")
                    time.sleep(2) 
                    continue

            except Exception as e:
                logging.error(f"请求异常: {str(e)}，正在重试... ({attempt + 1}/{max_retries})")
                time.sleep(base_wait_time * (2 ** attempt))
                continue
        
        logging.error("达到最大重试次数，生成失败")
        return ""

    # [修改] 接口适配：不再进行 smart_combine_prompts 这种带标签的拼接
    def generate_code_from_prompt(self, generated_prompt: str, original_prompt: str = None, k: int = 1, max_tokens: int = 4000) -> List[str]:
        """
        调用B模型生成k个代码样本
        Args:
            generated_prompt: A模型生成的 System Prompt
            original_prompt: 原始 User Query
        """
        codes = []
        
        # 确保 user_query 存在
        user_query = original_prompt if original_prompt else "Write code based on instructions."
        
        for i in range(k):
            # 直接将 generated_prompt 作为 system_prompt 传入
            code = self.generate_code(user_query=user_query, system_prompt=generated_prompt)
            codes.append(code.strip())
            
            if i < k - 1:
                time.sleep(1)
        
        return codes

    def load_b_model(self, *args, **kwargs):
        return self, None

    # [保留但弃用] 为了兼容性保留空实现，或者直接返回
    def smart_combine_prompts(self, generated_prompt: str, original_prompt: str, max_tokens: int = 4000) -> str:
        # 仅仅为了兼容 b_interface.py 的调用而不报错
        # 实际上我们现在的逻辑已经在 generate_code_from_prompt 里解耦了
        return f"{generated_prompt}\n{original_prompt}"

# 创建全局实例
b_model_api = BModelAPI()

# 兼容性函数导出
def load_b_model(model_path=None, device_id=None):
    return b_model_api, None

def generate_code_from_prompt(model, tokenizer, generated_prompt, original_prompt=None, k=1, max_tokens=4000):
    return b_model_api.generate_code_from_prompt(generated_prompt, original_prompt, k, max_tokens)

def smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens=4000):
    return b_model_api.smart_combine_prompts(generated_prompt, original_prompt, max_tokens)

if __name__ == "__main__":
    api_client = BModelAPI()
    print("Testing B Model API with correct System/User roles...")
    
    # 测试：A模型让B模型用Python写，且要打印 'Hello'
    sys_p = "You are a Python expert. Always print 'Hello' at the start of your code."
    user_p = "Write a function to add two numbers."
    
    codes = api_client.generate_code_from_prompt(sys_p, user_p, k=1)
    print(f"Code:\n{codes[0]}")