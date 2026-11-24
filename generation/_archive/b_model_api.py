import os
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
    """基于 API 调用的 B 模型 — Qwen2.5-72B-Instruct"""

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1/chat/completions"
        self.model_name = "/model/DeepSeek-V3.2"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "AIGC-USER":"zhuldz"
        }
        
        self.generation_params = {
            "temperature": 0.05,
            "top_p": 0.9,
            "max_tokens": 2048,
            "n": 1,
            "stream": False
        }
        
        logging.info("B模型API客户端初始化完成")

    def generate_code(self, user_query: str, system_prompt: str = None) -> str:
        """
        调用 API 生成代码。

        Args:
            user_query: 用户的原始问题
            system_prompt: A 模型生成的 system prompt (None 则使用默认)
        """
        max_retries = 5
        base_wait_time = 5

        messages = []

        if system_prompt is None:
            messages.append({
                "role": "system",
                "content": "You are a professional code generation assistant. Generate correct and efficient code."
            })
        elif system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_query})

        for attempt in range(max_retries):
            try:
                # 构造请求数据
                data = {
                    "model": self.model_name,
                    "messages": messages, # <--- 直接使用上面构造好的列表
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

    def generate_code_from_prompt(
        self, generated_prompt: str, original_prompt: str = None, k: int = 1, max_tokens: int = 4000
    ) -> List[str]:
        """调用 B 模型生成 k 个代码样本"""
        codes = []
        user_query = original_prompt or "Write code based on instructions."

        for i in range(k):
            code = self.generate_code(user_query=user_query, system_prompt=generated_prompt)
            codes.append(code.strip())
            if i < k - 1:
                time.sleep(1)

        return codes

    def load_b_model(self, *args, **kwargs):
        """兼容性接口：返回自身作为 'model'，tokenizer 为 None"""
        return self, None

    def smart_combine_prompts(self, generated_prompt: str, original_prompt: str, max_tokens: int = 4000) -> str:
        """简单拼接 system prompt 和 user query"""
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