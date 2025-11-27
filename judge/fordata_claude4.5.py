import os
import json
import requests
import time
import random
from typing import List, Dict, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('claude_chat.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ClaudeChat:
    def __init__(self, api_key: str, model_id: str = "anthropic.claude-opus-4-20250514-v1:0"):
        """初始化Claude对话客户端（非流式）"""
        self.api_key = api_key
        self.model_id = model_id
        self.api_url = "https://aimpapi.midea.com/t-aigc/f-devops-qwen3-coder-480b-a35b-instruct/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": api_key,
            "Aimp-Biz-Id": model_id,
            "AIGC-USER": "songzx28"
        }
        self.chat_history: List[Dict[str, str]] = []  # 保存对话历史
        logging.info("Claude非流式对话客户端初始化完成")

    def send_message(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """非流式发送消息：一次性获取完整回复（带重试机制）"""
        max_retries = 3  # 最大重试次数
        retry_delay = 5  # 重试间隔（秒）
        
        for retry in range(max_retries):
            try:
                # 重置本次对话历史
                self.chat_history = [{"role": "user", "content": [{"text": user_input}]}]
                
                # 构造请求体
                payload = {
                    "modelId": self.model_id,
                    "model": "claude",
                    "messages": self.chat_history,
                    "stream": False
                }
                
                # 设置系统提示词
                if system_prompt:
                    payload["system"] = [{"text": system_prompt}]
                
                # 发送请求
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                full_response = response.json()
                
                # 提取回复
                assistant_reply = ""
                if (
                    "output" in full_response and
                    isinstance(full_response["output"], dict) and
                    "message" in full_response["output"] and
                    isinstance(full_response["output"]["message"], dict) and
                    "content" in full_response["output"]["message"] and
                    isinstance(full_response["output"]["message"]["content"], list) and
                    len(full_response["output"]["message"]["content"]) > 0 and
                    "text" in full_response["output"]["message"]["content"][0]
                ):
                    assistant_reply = full_response["output"]["message"]["content"][0]["text"].strip()
                else:
                    assistant_reply = ""
                
                if assistant_reply:
                    self.chat_history.append({
                        "role": "assistant",
                        "content": [{"text": assistant_reply}]
                    })
                    return assistant_reply
                else:
                    raise ValueError("API返回空内容")
                
            except Exception as e:
                logging.warning(f"第{retry+1}次调用失败: {str(e)}")
                if retry < max_retries - 1:
                    time.sleep(retry_delay * (retry + 1))  # 指数退避重试
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


# 读取数据集（采用双重解析策略，与成功代码逻辑对齐）
def read_dataset(dataset_path: str) -> List[Dict]:
    """
    读取数据集，兼容两种格式：
    1. 标准JSON数组（[{"id":1}, {"id":2}]）
    2. JSON Lines（每行一个JSON对象）
    自动跳过错误行，返回有效数据列表
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            # 策略1：尝试解析为标准JSON数组
            try:
                data = json.load(f)
                # 验证是否为包含字典的列表
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    logging.info(f"成功加载标准JSON数组，共{len(data)}个样本")
                    # 过滤缺少必要字段的样本
                    return filter_valid_samples(data)
                else:
                    logging.warning("标准JSON解析成功，但不是有效的对象列表，将尝试逐行解析")
                    f.seek(0)  # 重置文件指针，准备逐行解析
                    raise json.JSONDecodeError("非对象列表格式", "", 0)  # 触发策略2
            except json.JSONDecodeError:
                # 策略2：逐行解析（JSON Lines格式）
                f.seek(0)  # 确保指针在文件开头
                data = []
                line_number = 0
                for line in f:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    try:
                        json_obj = json.loads(line)
                        if isinstance(json_obj, dict):
                            data.append(json_obj)
                            logging.debug(f"成功解析第{line_number}行")
                        else:
                            logging.warning(f"第{line_number}行不是JSON对象，跳过")
                    except json.JSONDecodeError as e:
                        # 输出错误行片段，方便定位问题
                        error_snippet = line[:100].replace('\n', ' ')
                        logging.warning(f"第{line_number}行格式错误，跳过：{str(e)}（片段：{error_snippet}...）")
                logging.info(f"逐行解析完成，共读取{len(data)}个原始样本")
                # 过滤缺少必要字段的样本
                return filter_valid_samples(data)
    except Exception as e:
        logging.error(f"读取数据集失败: {str(e)}")
        return []


def filter_valid_samples(data: List[Dict]) -> List[Dict]:
    """过滤出包含id、input、output必要字段的样本"""
    required_keys = {'id', 'input', 'output'}
    valid_samples = []
    for idx, item in enumerate(data):
        if required_keys.issubset(item.keys()):
            valid_samples.append(item)
        else:
            missing_keys = required_keys - set(item.keys())
            logging.warning(f"过滤无效样本（索引{idx}）：缺少必要字段{missing_keys}")
    logging.info(f"过滤后有效样本数：{len(valid_samples)}/{len(data)}")
    return valid_samples


# 生成系统提示词（循环处理）
def generate_system_prompts_loop(
    claude: ClaudeChat,
    example_content: str,
    dataset: List[Dict],
    save_path: str,
    loop_count: int = 3  # 循环生成次数（增加多样性）
) -> None:
    # 多样性提示侧重点（扩展更多维度）
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
    
    # 加载已生成的结果（避免重复生成）
    existing_results = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    existing_results[item["id"]] = item
        except Exception as e:
            logging.warning(f"加载已有结果失败，将重新生成: {str(e)}")
            existing_results = {}
    
    # 循环生成（多次循环增加多样性）
    for loop in range(loop_count):
        logging.info(f"开始第{loop+1}/{loop_count}轮生成")
        current_results = []
        
        for idx, item in enumerate(dataset):
            item_id = item.get('id')
            input_prompt = item.get('input', '')
            output_code = item.get('output', '')
            
            # 再次校验（双重保险）
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
            
            # 调用API生成
            claude.clear_history()
            generated_prompt = claude.send_message(user_input=user_message)
            
            # 保存结果
            if generated_prompt and "未获取到有效回复" not in generated_prompt:
                result_key = f"{item_id}_loop{loop+1}"  # 区分不同轮次的生成结果
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
            
            # 控制请求频率（避免触发限流）
            time.sleep(random.uniform(2, 4))  # 随机延迟2-4秒，减少规律性请求
        
        # 合并并保存结果
        all_results = list(existing_results.values()) + current_results
        existing_results.update({item["id"]: item for item in current_results})  # 去重
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logging.info(f"第{loop+1}轮结果已保存至 {save_path}（共{len(all_results)}条）")
        except Exception as e:
            logging.error(f"保存第{loop+1}轮结果失败: {str(e)}")
        
        # 轮次间休息（避免长时间连续请求）
        if loop < loop_count - 1:
            sleep_time = random.uniform(30, 60)  # 每轮结束休息30-60秒
            logging.info(f"第{loop+1}轮结束，休息{sleep_time:.1f}秒")
            time.sleep(sleep_time)


def main():
    # 配置参数
    API_KEY = "msk-4b8773bf749c892f2c9803aa69ef94b8b96e7cf807da78cbfdf8606ed919adef"
    DATASET_PATH = "/data/zhuldz/lunwen/data/OpenCodeInstruct/datajson/small_test1.json"  # 支持.json和.jsonl格式
    EXAMPLE_PATH = "/data/zhuldz/lunwen/judge/sysprompt_icl.txt"
    SAVE_PATH = "/data/zhuldz/lunwen/judge/generated_system_prompts.json"
    LOOP_COUNT = 3  # 循环生成次数
    
    # 初始化客户端
    claude = ClaudeChat(api_key=API_KEY)
    
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
        claude=claude,
        example_content=example_content,
        dataset=dataset,
        save_path=SAVE_PATH,
        loop_count=LOOP_COUNT
    )
    
    logging.info("所有生成任务完成")
    print("自动生成流程已结束，结果已保存")


if __name__ == "__main__":
    main()