import os
import json
import requests
import time
import random
import uuid
from typing import List, Dict, Optional, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qwen_chat.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --------------------------
# Qwen API配置
# --------------------------
API_URL = "https://aimpapi.midea.com/t-aigc/f-devops-qwen3-coder-480b-a35b-instruct/v1/chat/completions"
AUTH_TOKEN = "msk-4b8773bf749c892f2c9803aa69ef94b8b96e7cf807da78cbfdf8606ed919adef"  
TIMEOUT = 300  # 延长超时（代码生成可能较慢）
RETRY_TIMES = 3


class QwenChat:
    def __init__(self):
        """初始化Qwen对话客户端（非流式）"""
        self.headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json"
        }
        self.chat_history: List[Dict[str, str]] = []  # 保存对话历史
        logging.info("Qwen非流式对话客户端初始化完成")

    def send_message(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """非流式发送消息：一次性获取完整回复（带重试机制）"""
        # 构建消息列表（包含系统提示词和用户输入）
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})

        # 构造请求体
        request_body = {
            "model": "/model/qwen3-235b-a22b",  # Qwen模型路径
            "messages": messages,
            "stream": False,  # 非流式
            "temperature": 0.7,  # 控制多样性
            "top_p": 1.0,
            "max_tokens": 2048  # 足够生成系统提示词
        }

        # 重试机制
        for retry in range(RETRY_TIMES):
            try:
                response = requests.post(
                    url=API_URL,
                    headers=self.headers,
                    json=request_body,
                    timeout=TIMEOUT
                )
                response.raise_for_status()  # 检查HTTP错误
                response_json = response.json()

                # 解析Qwen响应（根据API返回结构提取内容）
                # Qwen响应通常在 choices[0].message.content 中
                if (
                    "choices" in response_json and
                    len(response_json["choices"]) > 0 and
                    "message" in response_json["choices"][0] and
                    "content" in response_json["choices"][0]["message"]
                ):
                    assistant_reply = response_json["choices"][0]["message"]["content"].strip()
                    logging.debug(f"Qwen回复成功（长度：{len(assistant_reply)}）")
                    return assistant_reply
                else:
                    raise ValueError(f"Qwen响应结构异常：{json.dumps(response_json, ensure_ascii=False)[:500]}")

            except Exception as e:
                logging.warning(f"第{retry+1}次调用失败: {str(e)}")
                if retry < RETRY_TIMES - 1:
                    wait_time = 2 ** retry  # 指数退避
                    time.sleep(wait_time)
                continue

        logging.error("达到最大重试次数，请求失败")
        return ""

    def clear_history(self) -> None:
        self.chat_history = []
        logging.info("对话历史已清空")


# 读取示例文件
def read_example_file(example_path: str) -> str:
    try:
        with open(example_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"读取示例文件失败: {str(e)}")
        return ""


# 读取数据集（兼容标准JSON数组和JSON Lines）
def read_dataset(dataset_path: str) -> List[Dict]:
    """读取数据集，支持标准JSON数组和JSON Lines格式，过滤有效样本"""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            # 尝试解析为标准JSON数组
            try:
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    logging.info(f"成功加载标准JSON数组，共{len(data)}个样本")
                    return filter_valid_samples(data)
                else:
                    logging.warning("标准JSON解析成功，但不是有效对象列表，尝试逐行解析")
                    f.seek(0)
                    raise json.JSONDecodeError("非对象列表格式", "", 0)
            except json.JSONDecodeError:
                # 逐行解析JSON Lines
                f.seek(0)
                data = []
                line_number = 0
                for line in f:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_obj = json.loads(line)
                        if isinstance(json_obj, dict):
                            data.append(json_obj)
                        else:
                            logging.warning(f"第{line_number}行不是JSON对象，跳过")
                    except json.JSONDecodeError as e:
                        error_snippet = line[:100].replace('\n', ' ')
                        logging.warning(f"第{line_number}行格式错误，跳过：{str(e)}（片段：{error_snippet}...）")
                logging.info(f"逐行解析完成，共读取{len(data)}个原始样本")
                return filter_valid_samples(data)
    except Exception as e:
        logging.error(f"读取数据集失败: {str(e)}")
        return []


def filter_valid_samples(data: List[Dict]) -> List[Dict]:
    """过滤包含id、input、output必要字段的样本"""
    required_keys = {'id', 'input', 'output'}
    valid_samples = []
    for idx, item in enumerate(data):
        if required_keys.issubset(item.keys()):
            valid_samples.append(item)
        else:
            missing_keys = required_keys - set(item.keys())
            logging.warning(f"过滤无效样本（索引{idx}）：缺少字段{missing_keys}")
    logging.info(f"过滤后有效样本数：{len(valid_samples)}/{len(data)}")
    return valid_samples


# 生成系统提示词（循环处理）
def generate_system_prompts_loop(
    qwen: QwenChat,
    example_content: str,
    dataset: List[Dict],
    save_path: str,
    loop_count: int = 3
) -> None:
    # 多样性提示侧重点
    diversity_foci = [
        "注重代码的正确性和遵循编程规范，严格匹配输入需求",
        "强调生成代码的执行效率和性能优化，减少冗余计算",
        "突出代码的可读性、可维护性，要求添加清晰注释",
        "引导模型优先考虑边界情况和异常处理，增强鲁棒性",
        "要求代码模块化设计，便于后续扩展和功能复用",
        "侧重简洁性，用最少的代码实现需求，避免过度设计",
        "强调代码的兼容性，考虑不同环境下的运行可能性",
        "引导模型先分析需求逻辑，再分步实现代码"
    ]
    
    # 加载已生成结果（去重）
    existing_results = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    existing_results[item["id"]] = item
        except Exception as e:
            logging.warning(f"加载已有结果失败，将重新生成: {str(e)}")
            existing_results = {}
    
    # 循环生成
    for loop in range(loop_count):
        logging.info(f"开始第{loop+1}/{loop_count}轮生成")
        current_results = []
        
        for idx, item in enumerate(dataset):
            item_id = item.get('id')
            input_prompt = item.get('input', '')
            output_code = item.get('output', '')
            
            if not all([item_id, input_prompt, output_code]):
                logging.warning(f"样本 {item_id} 缺少必要字段，跳过")
                continue
            
            # 打印进度
            print(f"\n===== 第{loop+1}轮，处理样本 {idx+1}/{len(dataset)} (ID: {item_id}) =====")
            
            # 随机选择侧重点
            focus = random.choice(diversity_foci)
            
            # 构造生成提示
            user_message = f"""任务：生成用于代码生成任务指令微调的系统提示词，需模仿示例风格并保持多样性。

示例参考：
{example_content}

规则说明：
- 将"input"作为Original prompt，"output"作为Correct code
- 系统提示词需明确引导模型根据输入生成符合要求的代码
- 本次生成侧重：{focus}
- 语言专业、简洁，符合代码生成任务的指令微调场景

当前数据：
Original prompt（input）：{input_prompt}
Correct code（output）：{output_code}

请生成对应的系统提示词："""
            
            # 调用Qwen生成
            qwen.clear_history()
            generated_prompt = qwen.send_message(user_input=user_message)
            
            # 保存结果
            if generated_prompt and "未获取到有效回复" not in generated_prompt:
                result_key = f"{item_id}_loop{loop+1}"
                current_results.append({
                    "id": result_key,
                    "original_id": item_id,
                    "input": input_prompt,
                    "output": output_code,
                    "generated_system_prompt": generated_prompt,
                    "focus": focus,
                    "loop": loop + 1
                })
                logging.info(f"样本 {item_id} 第{loop+1}轮生成成功")
            else:
                logging.warning(f"样本 {item_id} 第{loop+1}轮生成失败")
            
            # 控制请求频率
            time.sleep(random.uniform(2, 4))
        
        # 合并并保存结果
        all_results = list(existing_results.values()) + current_results
        existing_results.update({item["id"]: item for item in current_results})
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logging.info(f"第{loop+1}轮结果已保存至 {save_path}（共{len(all_results)}条）")
        except Exception as e:
            logging.error(f"保存第{loop+1}轮结果失败: {str(e)}")
        
        # 轮次间休息
        if loop < loop_count - 1:
            sleep_time = random.uniform(30, 60)
            logging.info(f"第{loop+1}轮结束，休息{sleep_time:.1f}秒")
            time.sleep(sleep_time)


def main():
    # 配置参数
    DATASET_PATH = "/data/zhuldz/lunwen/data/OpenCodeInstruct/datajson/small_test1.json"
    EXAMPLE_PATH = "/data/zhuldz/lunwen/judge/sysprompt_icl.txt"
    SAVE_PATH = "/data/zhuldz/lunwen/judge/generated_system_prompts_qwen.json"
    LOOP_COUNT = 2  # 循环生成次数
    
    # 初始化Qwen客户端
    qwen = QwenChat()
    
    # 读取依赖文件
    example_content = read_example_file(EXAMPLE_PATH)
    if not example_content:
        logging.error("示例文件读取失败，程序退出")
        return
    
    dataset = read_dataset(DATASET_PATH)
    if not dataset:
        logging.error("数据集读取失败，程序退出")
        return
    
    # 开始自动循环生成
    logging.info(f"开始自动循环生成（共{LOOP_COUNT}轮）")
    generate_system_prompts_loop(
        qwen=qwen,
        example_content=example_content,
        dataset=dataset,
        save_path=SAVE_PATH,
        loop_count=LOOP_COUNT
    )
    
    logging.info("所有生成任务完成")
    print("自动生成流程已结束，结果已保存")


if __name__ == "__main__":
    main()