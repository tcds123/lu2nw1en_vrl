# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


@register("batch")
class BatchRewardManager(AbstractRewardManager):
    """
    A batch reward manager that computes rewards for a batch of data.
    """

    def __init__(
        self, tokenizer, num_examine, compute_score: RawRewardFn, reward_fn_key="data_source", prompt_key="prompt", config=None, **reward_kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.config = config
        self.prompt_key = prompt_key
        self.reward_kwargs = reward_kwargs

    def _get_scores_from_custom_fn(self, data: DataProto) -> (torch.Tensor, dict, torch.Tensor):
        # 1. 获取基础数据
        # prompt_key 对应 Parquet 中的 'input' (此时是包含 ICL 的长文本)
        prompts_str = data.non_tensor_batch[self.prompt_key] 
        
        # reward_fn_key 对应 Parquet 中的 'reward_model' 或 'output'
        # 这里获取到的是真值 (Ground Truth)
        ground_truths = data.non_tensor_batch[self.reward_fn_key] 
        
        # --- [关键修改] 尝试提取 raw_input ---
        # 如果数据中有这个列，就提取出来传给奖励函数
        raw_inputs = None
        if 'raw_input' in data.non_tensor_batch:
            raw_inputs = data.non_tensor_batch['raw_input']
        
        # 2. 解码模型生成的 Responses
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        prompt_len = data.batch["prompts"].shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        # 3. 调用自定义奖励函数 (custom_reward.py)
        # 我们显式传递 raw_input 和 ground_truth
        scores = self.compute_score(
            prompts=prompts_str,              # A模型的输入 (含ICL)
            responses=responses_str,          # A模型的输出 (System Prompt)
            canonical_solution=ground_truths, # 兼容旧名称
            ground_truth=ground_truths,       # [新增] 显式传递 ground_truth
            raw_input=raw_inputs,             # [新增] 传递清洗过的原始输入
            config=self.config,
            tokenizer=self.tokenizer,
            return_dict=True,
            **self.reward_kwargs,
        )

        if isinstance(scores, torch.Tensor):
             return scores, {}, valid_response_lengths
        
        reward_tensor = scores.get("reward_tensor")
        reward_extra_info = scores.get("reward_extra_info", {})

        if not isinstance(reward_tensor, (list, torch.Tensor)) or len(reward_tensor) != len(data):
            raise TypeError(f"Custom reward function returned wrong type/size. Expected list or 1D tensor of size {len(data)}, but got {type(reward_tensor)} of size {len(reward_tensor)}")
        
        return reward_tensor, reward_extra_info, valid_response_lengths

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        responses = data.batch["responses"]
        reward_tensor_2d = torch.zeros_like(responses, dtype=torch.float32) 
        
        scores_1d, reward_extra_info, valid_response_lengths = self._get_scores_from_custom_fn(data)

        for i in range(len(data)):
            length = valid_response_lengths[i].item() 
            reward = scores_1d[i].item() if isinstance(scores_1d[i], torch.Tensor) else scores_1d[i] 

            if length > 0:
                reward_tensor_2d[i, length - 1] = reward
        
        # Debug print
        already_printed = {}
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, ["unknown"] * len(data))
        prompts_str = data.non_tensor_batch[self.prompt_key]
        ground_truths = data.non_tensor_batch[self.reward_fn_key]
        
        for i in range(len(data)):
            data_source = str(data_sources[i])
            if already_printed.get(data_source, 0) < self.num_examine:
                # print("[prompt]", prompts_str[i]) 
                # print("[ground_truth]", ground_truths[i])
                # print("[score]", scores_1d[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1
        
        if return_dict:
            return {"reward_tensor": reward_tensor_2d, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor_2d

    def verify(self, data):
        raise NotImplementedError("Verify is deprecated, logic is in __call__")

    # def verify(self, data):
    #     #关键修复
    #     #prompt_ids = data.batch["prompts"]
    #     prompt_ids = data.non_tensor_batch[self.prompt_key]
    #     response_ids = data.batch["responses"]
    #     attention_mask = data.batch["attention_mask"]

    #     prompt_len = prompt_ids.shape[-1]
    #     valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

    #     responses_str = []
    #     for i in range(len(data)):
    #         valid_len = valid_response_lengths[i]
    #         valid_response_ids = response_ids[i][:valid_len]
    #         response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    #         responses_str.append(response_str)

    #     ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
    #     data_sources = data.non_tensor_batch[self.reward_fn_key]
    #     rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
    #     extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])

    #     for i in range(len(data)):
    #         extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]

    #     scores = self.compute_score(
    #         data_sources=data_sources,
    #         solution_strs=responses_str,
    #         ground_truths=ground_truths,
    #         extra_infos=extras,
    #         **self.reward_kwargs,
    #     )

    #     return scores

    # def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
    #     # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
    #     if "rm_scores" in data.batch.keys():
    #         if return_dict:
    #             reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
    #             reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
    #             return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
    #         else:
    #             return data.batch["rm_scores"]

    #     reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    #     reward_extra_info = defaultdict(list)
    #     prompt_ids = data.batch["prompts"]
    #     prompt_len = prompt_ids.shape[-1]
    #     attention_mask = data.batch["attention_mask"]
    #     valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
    #     data_sources = data.non_tensor_batch[self.reward_fn_key]

    #     scores = self.verify(data)
    #     rewards = []
    #     already_printed: dict[str, Any] = {}

    #     for i in range(len(data)):
    #         length = valid_response_lengths[i].item()
    #         score = scores[i]

    #         if isinstance(score, dict):
    #             reward = score["score"]
    #             for key, value in score.items():
    #                 reward_extra_info[key].append(value)
    #         else:
    #             reward = score

    #         rewards.append(reward)
    #         reward_tensor[i, length - 1] = reward

    #         data_source = data_sources[i]
    #         if already_printed.get(data_source, 0) < self.num_examine:
    #             response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
    #             prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
    #             ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
    #             print("[prompt]", prompt_str)
    #             print("[response]", response_str)
    #             print("[ground_truth]", ground_truth)
    #             print("[score]", scores[i])
    #             already_printed[data_source] = already_printed.get(data_source, 0) + 1

    #     data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

    #     if return_dict:
    #         return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
    #     else:
    #         return reward_tensor
