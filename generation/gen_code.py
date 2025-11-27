import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from pathlib import Path  # 用于处理路径

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("b_model_generation.log"),  # 日志也保存在当前目录
        logging.StreamHandler()
    ]
)


def load_model_and_tokenizer(model_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.float16,
            # attn_implementation="flash_attention_2"  # 按需启用
        )
        model.eval()
        logging.info(f"模型加载成功：{model_path}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"模型加载失败：{str(e)}", exc_info=True)
        raise

def is_valid_python_code(code: str) -> bool:
    """检查生成的代码是否是有效的Python代码"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def generate_code(model, tokenizer, prompt: str) -> str:
    """统一的代码生成函数"""
    try:
        messages = [
            {"role": "system", "content": "You are a professional code generation assistant. Generate correct and efficient code."},
            {"role": "user", "content": prompt}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        attention_mask = torch.ones_like(input_ids)

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 2048,  # 增加生成长度避免截断
            "temperature": 0.3,     # 降低温度提高稳定性
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
        }

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

        generated_ids = outputs[:, input_ids.shape[1]:]
        generated_code = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        ).strip()

        logging.info(f"代码生成完成，长度: {len(generated_code)}字符")
        return generated_code
        
    except Exception as e:
        logging.error(f"代码生成失败: {str(e)}")
        return ""
    
    # 添加代码有效性检查
    if not is_valid_python_code(generated_code):
        logging.warning("生成的代码存在语法错误，尝试重新生成")
        # 可以添加重试逻辑
        return ""
    
    return generated_code


def process_json_file(json_path: str, model, tokenizer):
    # 检查输入文件是否存在
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"输入JSON文件不存在：{json_path}")

    # 读取原始数据
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 统一数据格式为列表
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError(f"JSON数据必须是列表或字典，实际为：{type(data)}")

    total = len(data)
    logging.info(f"开始处理JSON数据（共{total}条，提示词键名为raw_problem）")

    # 处理每条数据
    for i, item in enumerate(tqdm(data, desc="生成代码进度")):
        if "raw_problem" not in item:
            logging.warning(f"第{i+1}条数据缺少'raw_problem'键，跳过")
            continue

        prompt = item["raw_problem"].strip()
        if not prompt:
            logging.warning(f"第{i+1}条数据的'raw_problem'为空，跳过")
            item["predict_code"] = ""
            continue

        logging.info(f"处理第{i+1}/{total}条：raw_problem前100字符：{prompt[:100]}...")
        generated_code = generate_code(model, tokenizer, prompt)
        item["predict_code"] = generated_code


    current_script_dir = Path(__file__).parent 
    json_filename = os.path.basename(json_path)  
    output_json_path = current_script_dir / json_filename
    backup_path = current_script_dir / f"{json_filename}.backup"

    if not backup_path.exists():
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"原始数据已备份至当前脚本目录：{backup_path}")

    # 保存处理后的结果到当前脚本目录
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"处理完成，结果已保存至当前脚本目录：{output_json_path}")


def main():
    INPUT_JSON_PATH = "/data/zhuldz/lunwen/data/CodeEval-Pro/dataset/humaneval_pro.json"
    MODEL_PATH = "/data/zhuldz/lunwen/models/Qwen2.5-Coder-7B-Instruct"

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    process_json_file(INPUT_JSON_PATH, model, tokenizer)

if __name__ == "__main__":
    main()