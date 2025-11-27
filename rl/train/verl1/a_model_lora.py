import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import List
import os
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

class AModelLoRA:
    """支持LoRA的A模型：基于SystemPromptGenerator的系统提示词生成器"""
    
    def __init__(self, model_path: str, device_id: int = 0, example_file: str = "/data/zhuldz/lunwen/rl/train/verl1/sysprompt_icl.txt", 
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1,
                 init_lora_weights: bool = True):
        """初始化支持LoRA的A模型 - 修复初始化问题"""
        # 设置目标设备
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|end_of_solution|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        # 加载基础模型
        logging.info(f"加载基础模型到设备: {self.device}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": self.device} if torch.cuda.is_available() else None,
            trust_remote_code=True, # 显式传递tokenizer配置以确保一致性
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # 应用LoRA适配器
        self.model = self._setup_lora_adapter(lora_r, lora_alpha, lora_dropout, init_lora_weights)
        
        # 确保模型在正确设备上
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        # 启用梯度检查点以节省内存
        self._enable_gradient_checkpointing()
        
        # 设置为训练模式
        self.model.train()
        
        # 加载示例文件
        self.example_content = self._load_example_file(example_file)
        
        # 新增：样本计数器，用于控制示例显示
        self.sample_counter = 0
        self.max_example_samples = 5  # 默认前5个样本显示示例
        
        # 检查LoRA适配器初始化状态
        self._check_lora_initialization()
        
        logging.info("支持LoRA的A模型加载完成")

    def _setup_lora_adapter(self, r, alpha, dropout, init_lora_weights):
        """设置LoRA适配器 - 修复初始化问题"""
        # 定义LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            init_lora_weights=init_lora_weights,  # 确保启用权重初始化
        )
        
        # 应用LoRA适配器
        lora_model = get_peft_model(self.base_model, lora_config)
        
        # 打印可训练参数信息
        lora_model.print_trainable_parameters()
        
        logging.info(f"LoRA配置: r={r}, alpha={alpha}, dropout={dropout}")
        
        return lora_model

    def _check_lora_initialization(self):
        """检查LoRA适配器初始化状态"""
        try:
            from model_utils import check_lora_adapter_initialization
            lora_ok = check_lora_adapter_initialization(self.model, "A模型")
            if lora_ok:
                logging.info("LoRA适配器初始化检查通过")
            else:
                logging.warning("LoRA适配器初始化可能有问题，但继续训练")
        except Exception as e:
            logging.warning(f"LoRA适配器初始化检查失败: {str(e)}")

    def _enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logging.info("A模型已启用梯度检查点")
        else:
            logging.warning("A模型不支持梯度检查点")

    def _load_example_file(self, example_file):
        """加载示例文件"""
        try:
            if os.path.exists(example_file):
                with open(example_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logging.warning(f"示例文件 {example_file} 不存在")
                return ""
        except Exception as e:
            logging.error(f"加载示例文件失败: {str(e)}")
            return ""

    def generate_prompt(self, original_prompt: str, canonical_solution: str, m: int = 1, max_tokens: int = 1024) -> List[str]:
        """生成系统提示词（推理模式）"""
        # 切换到推理模式
        self.model.eval()
        
        try:
            # 构建提示模板
            prompt_template = self._build_prompt_template(original_prompt, canonical_solution)
            
            # 编码输入
            inputs = self.tokenizer(
                prompt_template,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                return_token_type_ids=False
            ).to(self.device)
            
            # 生成提示词
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_return_sequences=m,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    repetition_penalty=1.3
                )
            
            # 解码生成的提示词
            generated_prompts = []
            for output in outputs:
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                if "【model_Output】" in full_text:
                    prompt = full_text.split("【model_Output】")[-1].strip()
                else:
                    prompt = full_text.strip()
                generated_prompts.append(prompt)
            
            return generated_prompts
            
        finally:
            # 切换回训练模式
            self.model.train()

    def generate_prompts(self, original_prompt: str, canonical_solution: str, m: int = 2, max_tokens: int = 1024):
        """生成系统提示词（推理模式）- 适配GRPOTrainerWrapper接口，默认生成2个提示词"""
        # 切换到推理模式
        self.model.eval()
        
        try:
            # 构建提示模板
            prompt_template = self._build_prompt_template(original_prompt, canonical_solution)
            
            # 编码输入
            inputs = self.tokenizer(
                prompt_template,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                return_token_type_ids=False
            ).to(self.device)
            
            # 检查输入张量是否为空或维度不正确
            if inputs['input_ids'].numel() == 0:
                raise ValueError("输入张量为空，无法生成提示词")
            
            # 检查输入张量维度
            if inputs['input_ids'].dim() != 2 or inputs['input_ids'].size(0) == 0:
                raise ValueError(f"输入张量维度不正确: {inputs['input_ids'].shape}")
            
            # 生成提示词 - 默认生成2个序列
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_return_sequences=m,  # 默认生成2个提示词
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    repetition_penalty=1.1
                )
            
            # 检查输出张量是否为空或维度不正确
            if outputs.numel() == 0:
                raise ValueError("输出张量为空，生成失败")
            
            if outputs.dim() != 2 or outputs.size(0) == 0:
                raise ValueError(f"输出张量维度不正确: {outputs.shape}")
            
            # 解码生成的提示词
            generated_prompts = []
            for output in outputs:
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                # 关键修复：安全处理分隔符，避免索引越界
                if "【model_Output】" in full_text:
                    split_parts = full_text.split("【model_Output】")
                    # 确保分割后有内容才取最后一个部分
                    if split_parts:
                        prompt = split_parts[-1].strip()
                    else:
                        prompt = full_text.strip()
                else:
                    prompt = full_text.strip()
                generated_prompts.append(prompt)
            
            # 检查生成的提示词是否有效
            if len(generated_prompts) == 0:
                raise ValueError("解码后未生成任何提示词")
            
            # 确保生成的提示词数量正确
            if len(generated_prompts) < m:
                logging.warning(f"生成的提示词数量不足，期望{m}个，实际{len(generated_prompts)}个")
                # 不补充默认提示词，而是使用实际生成的提示词
            
            # 适配GRPOTrainerWrapper接口，返回三个值
            return generated_prompts, inputs, outputs
            
        except Exception as e:
            logging.error(f"生成提示词过程中发生错误: {str(e)}")
            # 重新抛出异常，让调用者处理
            raise e
            
        finally:
            # 切换回训练模式
            self.model.train()

    def _build_prompt_template(self, original_prompt, canonical_solution):
        """构建提示模板"""
        instruction = "I will provide you with some examples of generating system prompts. Please carefully study and understand the content and structure of these examples.\n\n"
        
        # 新增：根据样本计数器控制示例显示
        example_block = ""
        if self.example_content and self.sample_counter < self.max_example_samples:
            example_block = f"Examples:\n{self.example_content}\n\n"
            logging.info(f"样本 {self.sample_counter + 1}: 显示示例内容")
        else:
            logging.info(f"样本 {self.sample_counter + 1}: 不显示示例内容（已超过前{self.max_example_samples}个样本）")
        
        generation_instruction = (
            "Based on the examples above, generate an English system prompt for the following input (follow the same format as examples),"
            "IMPORTANT RULES:\n"
            "Output ONLY the final system prompt, with NO intermediate thinking, explanations, or reasoning.\n"
            "Do NOT include phrases like 'Let me think', 'First, I need to', or any similar thought process.\n"
            "It is not allowed to output any thinking and explanatory statements, only the generated system prompts:\n"
        )
        
        task_input = (
            f"【Input】\n"
            f"Original prompt: {original_prompt}\n"
            f"Correct code: {canonical_solution}\n"
            f"【model_Output】\n"
        )
        
        # 新增：递增样本计数器
        self.sample_counter += 1
        
        return f"{instruction}{example_block}{generation_instruction}{task_input}"

    def reset_sample_counter(self):
        """重置样本计数器"""
        self.sample_counter = 0
        logging.info("样本计数器已重置")

    def set_max_example_samples(self, max_samples: int):
        """设置最大显示示例的样本数量"""
        self.max_example_samples = max_samples
        logging.info(f"设置最大显示示例样本数为: {max_samples}")

    def save_lora_weights(self, output_dir):
        """保存LoRA权重"""
        try:
            self.model.save_pretrained(output_dir)
            logging.info(f"LoRA权重已保存到: {output_dir}")
        except Exception as e:
            logging.error(f"保存LoRA权重失败: {str(e)}")

    def load_lora_weights(self, lora_path):
        """加载LoRA权重"""
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.base_model, lora_path)
            self.model.train()
            logging.info(f"LoRA权重已从 {lora_path} 加载")
        except Exception as e:
            logging.error(f"加载LoRA权重失败: {str(e)}")