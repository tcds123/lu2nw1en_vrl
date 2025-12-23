import sys
import torch
import logging
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# =============================================
# 模型选择配置区域
# =============================================

# 选项1: 使用本地B模型
# USE_LOCAL_MODEL = True

# 选项2: 使用API版本的B模型 (当前启用)
USE_API_MODEL = True

# =============================================
# 本地B模型实现
# =============================================
if 'USE_LOCAL_MODEL' in locals() and USE_LOCAL_MODEL:
    sys.path.append("/data/zhuldz/lunwen/generation")
    from gen_code import (
        load_model_and_tokenizer,
        generate_code as b_model_generate_code
    )

    def load_b_model(model_path="/data/zhuldz/lunwen/models/Qwen2.5-Coder-7B-Instruct", device_id=1):
        target_device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_model_and_tokenizer(model_path)
        if torch.cuda.is_available():
            model = model.to(target_device)
        logging.info(f"本地B模型已加载到设备: {target_device}")
        return model, tokenizer

    def smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens=4000):
        original_tokens = tokenizer.encode(original_prompt, add_special_tokens=False)
        if len(original_tokens) >= max_tokens:
            truncated_original = tokenizer.decode(original_tokens[:max_tokens-100], skip_special_tokens=True)
            return f"[Original (truncated)]{truncated_original}"
        
        remaining_tokens = max_tokens - len(original_tokens) - 100
        if remaining_tokens <= 0:
            return f"[Original]{original_prompt}"
        
        generated_tokens = tokenizer.encode(generated_prompt, add_special_tokens=False)
        if len(generated_tokens) <= remaining_tokens:
            return f"[Generated]{generated_prompt}\n[Original]{original_prompt}"
        else:
            truncated_generated = tokenizer.decode(generated_tokens[:remaining_tokens], skip_special_tokens=True)
            return f"[Generated (truncated)]{truncated_generated}\n[Original]{original_prompt}"

    def generate_code_from_prompt(model, tokenizer, generated_prompt, original_prompt=None, k=1, max_tokens=4000):
        codes = []
        if original_prompt is not None:
            combined_prompt = smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens)
        else:
            tokens = tokenizer.encode(generated_prompt, add_special_tokens=False)
            if len(tokens) > max_tokens:
                combined_prompt = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            else:
                combined_prompt = generated_prompt
        
        for _ in range(k):
            code = b_model_generate_code(model, tokenizer, combined_prompt)
            codes.append(code.strip())
        return codes

# =============================================
# API版本B模型实现 (已修复参数传递)
# =============================================
elif 'USE_API_MODEL' in locals() and USE_API_MODEL:
    sys.path.append("/data/zhuldz/lunwen/generation")

    from b_model_api import load_b_model as api_load_b_model
    from b_model_api import generate_code_from_prompt as api_generate_code
    from b_model_api import smart_combine_prompts as api_smart_combine
    
    logging.info("🚀 [Mode] 已启用 API 版本 B 模型 (参数透传模式)")

    def load_b_model(model_path=None, device_id=None):
        return api_load_b_model(model_path, device_id)

    def generate_code_from_prompt(model, tokenizer, generated_prompt, original_prompt=None, k=1, max_tokens=4000):
        """
        [修复] 必须显式传递 model 和 tokenizer 给底层函数，
        哪怕底层并不使用它们（为了满足 b_model_api.py 中定义的函数签名）。
        """
        return api_generate_code(
            model=model,          # <--- [核心修复] 补上这两个参数
            tokenizer=tokenizer,  # <--- [核心修复] 补上这两个参数
            generated_prompt=generated_prompt,
            original_prompt=original_prompt,
            k=k,
            max_tokens=max_tokens
        )

    def smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens=4000):
        return api_smart_combine(generated_prompt, original_prompt, max_tokens)

# =============================================
# 默认配置
# =============================================
else:
    sys.path.append("/data/zhuldz/lunwen/generation")
    from gen_code import (
        load_model_and_tokenizer,
        generate_code as b_model_generate_code
    )

    def load_b_model(model_path="/data/zhuldz/lunwen/models/Qwen2.5-Coder-7B-Instruct", device_id=1):
        target_device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_model_and_tokenizer(model_path)
        if torch.cuda.is_available():
            model = model.to(target_device)
        return model, tokenizer

    def smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens=4000):
        original_tokens = tokenizer.encode(original_prompt, add_special_tokens=False)
        if len(original_tokens) >= max_tokens:
            truncated_original = tokenizer.decode(original_tokens[:max_tokens-100], skip_special_tokens=True)
            return f"[Original (truncated)]{truncated_original}"
        remaining_tokens = max_tokens - len(original_tokens) - 100
        if remaining_tokens <= 0:
            return f"[Original]{original_prompt}"
        generated_tokens = tokenizer.encode(generated_prompt, add_special_tokens=False)
        if len(generated_tokens) <= remaining_tokens:
            return f"[Generated]{generated_prompt}\n[Original]{original_prompt}"
        else:
            truncated_generated = tokenizer.decode(generated_tokens[:remaining_tokens], skip_special_tokens=True)
            return f"[Generated (truncated)]{truncated_generated}\n[Original]{original_prompt}"

    def generate_code_from_prompt(model, tokenizer, generated_prompt, original_prompt=None, k=1, max_tokens=4000):
        codes = []
        if original_prompt is not None:
            combined_prompt = smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens)
        else:
            tokens = tokenizer.encode(generated_prompt, add_special_tokens=False)
            if len(tokens) > max_tokens:
                combined_prompt = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            else:
                combined_prompt = generated_prompt
        for _ in range(k):
            code = b_model_generate_code(model, tokenizer, combined_prompt)
            codes.append(code.strip())
        return codes

if __name__ == "__main__":
    try:
        print("正在加载B模型...")
        model, tokenizer = load_b_model()
        
        test_sys_prompt = "You are a Python expert. Always print 'Hello API' at the start."
        test_user_query = "Write a function to add two numbers."
        
        print("\n正在生成代码...")
        codes = generate_code_from_prompt(
            model, tokenizer, 
            generated_prompt=test_sys_prompt,
            original_prompt=test_user_query,
            k=1
        )
        
        if 'USE_API_MODEL' in locals() and USE_API_MODEL:
            print("当前使用：API版本的B模型")
        else:
            print("当前使用：本地版本的B模型")
            
        print(f"\n代码生成结果:\n{codes[0]}")
        
    except Exception as e:
        print(f"测试出错：{str(e)}")
        import traceback
        traceback.print_exc()