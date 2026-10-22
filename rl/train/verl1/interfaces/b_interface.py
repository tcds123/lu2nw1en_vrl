"""
B 模型接口 — 统一使用 API 调用 Qwen2.5-72B-Instruct
不再支持本地加载小模型
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

sys.path.append("/data/zhuldz/lunwen/generation")

from b_model_api import (
    load_b_model,
    generate_code_from_prompt,
    smart_combine_prompts,
)

logging.info("B 模型接口已加载 (API 模式: Qwen2.5-72B-Instruct)")


if __name__ == "__main__":
    try:
        print("正在连接 B 模型 API...")
        model, tokenizer = load_b_model()
        print(f"  模型: {model.model_name}")
        print(f"  API: {model.base_url}")

        test_sys_prompt = "You are a Python expert. Always print 'Hello API' at the start."
        test_user_query = "Write a function to add two numbers."

        print("\n正在生成代码...")
        codes = generate_code_from_prompt(
            model, tokenizer,
            generated_prompt=test_sys_prompt,
            original_prompt=test_user_query,
            k=1
        )

        print(f"\n代码生成结果:\n{codes[0]}")

    except Exception as e:
        print(f"测试出错：{str(e)}")
        import traceback
        traceback.print_exc()
