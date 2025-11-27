import os
import json
import requests
import time
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
        self.api_url = "https://aimpapi.midea.com/t-aigc/mip-chat-app/claude/official/standard/sync/v2/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": api_key,
            "Aimp-Biz-Id": model_id,
            "AIGC-USER": "songzx28"
        }
        self.chat_history: List[Dict[str, str]] = []  # 保存对话历史
        logging.info("Claude非流式对话客户端初始化完成")

    def send_message(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """非流式发送消息：一次性获取完整回复"""
        # 更新对话历史
        self.chat_history.append({"role": "user", "content": [{"text": user_input}]})
        
        # 构造请求体（关闭流式）
        payload = {
            "modelId": self.model_id,
            "model": "claude",
            "messages": self.chat_history,
            "stream": False  # 关键：关闭流式响应
        }
        
        # 首次对话设置系统提示词
        if system_prompt and len(self.chat_history) == 1:
            payload["system"] = [{"text": system_prompt}]
        
        try:
            # 发送非流式请求（stream=False，无需逐行处理）
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=300  # 超时时间（5分钟）
            )
            response.raise_for_status()  # 检查HTTP错误（如401、500）
            
            # 解析完整响应JSON
            full_response = response.json()
            
            # 提取回复内容（根据API返回结构调整）
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
                assistant_reply = "未获取到有效回复（响应结构不匹配）"
            
            # 打印完整回复
            print(f"\nClaude回复：{assistant_reply}\n")
            
            # 更新对话历史
            self.chat_history.append({
                "role": "assistant",
                "content": [{"text": assistant_reply}]
            })
            return assistant_reply
        
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求失败: {str(e)}"
        except json.JSONDecodeError:
            error_msg = "响应格式错误，无法解析为JSON"
        except Exception as e:
            error_msg = f"处理响应时出错: {str(e)}"
        
        logging.error(error_msg)
        print(f"\n错误: {error_msg}")
        # 移除本次失败的用户输入（避免污染历史）
        if self.chat_history and self.chat_history[-1]["role"] == "user":
            self.chat_history.pop()
        return ""

    def clear_history(self) -> None:
        self.chat_history = []
        logging.info("对话历史已清空")
        print("对话历史已重置，可开始新对话")

    def save_history(self, file_path: str = "chat_history.json") -> None:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            logging.info(f"对话历史已保存至: {file_path}")
            print(f"对话历史已保存到 {file_path}")
        except Exception as e:
            logging.error(f"保存对话历史失败: {e}")

    def load_history(self, file_path: str = "chat_history.json") -> None:
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    self.chat_history = json.load(f)
                logging.info(f"已加载对话历史: {file_path}")
                print(f"已加载对话历史（共{len(self.chat_history)}条消息）")
            else:
                print(f"未找到对话历史文件: {file_path}")
        except Exception as e:
            logging.error(f"加载对话历史失败: {e}")


def main():
    # 配置API密钥（替换为有效密钥）
    API_KEY = "msk-b6d97dcc6c2213ebd83ca3e9292f25a02b6cf5af3be7ec42e1b20b2630defcc5"
    
    # 初始化客户端
    claude = ClaudeChat(api_key=API_KEY)
    
    # 系统提示词
    system_prompt = "你是一个友好的AI助手，能解答各种问题，语言简洁明了。"
    
    # 对话循环
    print("欢迎使用Claude非流式对话工具（输入 'exit' 退出，'clear' 清空，'save' 保存，'load' 加载）")
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() == "exit":
            print("对话结束，再见！")
            break
        elif user_input.lower() == "clear":
            claude.clear_history()
            continue
        elif user_input.lower() == "save":
            claude.save_history()
            continue
        elif user_input.lower() == "load":
            claude.load_history()
            continue
        elif not user_input:
            print("请输入内容...")
            continue
        
        # 发送消息（非流式）
        claude.send_message(
            user_input=user_input,
            system_prompt=system_prompt if len(claude.chat_history) == 0 else None
        )
        
        time.sleep(1)  # 避免请求过于频繁


if __name__ == "__main__":
    main()