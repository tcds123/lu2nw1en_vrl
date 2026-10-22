---
name: grpo-skill-smoke-passed-20260722
description: 🔥GRPO优化skill方法端到端1-step冒烟通过 - A改写skill→B生成代码→LCB单测→delta reward→GRPO更新全链路打通, hard题delta>0信号清晰
metadata:
  type: project
---

# GRPO 优化 Skill 方法: 端到端 1-step 冒烟通过 (2026-07-22) 🔥

## 里程碑
本机 1×L40 + Qwen3-0.6B, verl GRPO 1-step 完整跑通:
`✅ [System] Step 1 COMPLETE` + `🏁 [Iter 0] n=20 mean_reward=2.35 delta: +8/0:11/-0 mean_delta=0.3268`

整条链路打通: **A 模型 (GRPO 改写 skill) → B 模型 (DeepSeek-V3.2 生成代码) → LCB stdin/stdout 单测 → delta pass-rate reward → GRPO 更新 A**。这是 [[our-grpo-skill-method-design-20260721]] 阶段 0 端到端打通的核心验收。

## 关键信号 (方法有效性初证)
reward trace (verl 真实数据流, 非 0.2 单测):
- **abc380_g (hard)**: S_0 全错 (s0=0), A 改写的 skill 有 2 条让 B 全对 (skill_pass=1.0, delta=1.0, reward=10.0) — **改写 skill 真的能让 B 做出 S_0 做不出的 hard 题!**
- **abc375_g (hard)**: S_0=0.333, 2 条改写到 1.0 (delta=0.667, reward=6.67)
- 整批 20 轨迹: 8 正 delta / 11 零 / 0 负, mean_delta=0.327 — delta 有区分度, GRPO 有信号可学

GRPO 更新指标: actor/pg_loss 正常, ppo_kl=0.0059 (KL 约束生效), grad_norm=1.05, advantages 有正有负 (max1.79/min-1.58), critic/rewards max10/min-15 (退化惩罚 -15 到 正奖励 +10 都触发了)。

## timing (1 卡 0.6B, 20 轨迹/step)
gen:19.5s, **reward:178s (79%)**, update_actor:15s, ref:8.9s, step总:225s。
reward 占 79% (B API 28RPM 限流 + evaluate), 与 [[grpo-readiness-20260713]] "API 是瓶颈非 GPU" 一致。GPU reward 期间空闲。

## 今天修复的 4 个阻塞 (按定位顺序)
1. **S_0 预计算 backend env bug**: `precompute_s0_baseline.py` 只设 TARGET_BACKEND 未设 QWEN_CHAT_* → 61题全报 Azure 错误。补全 setdefault 修复。重跑 61/61 agent_ok=True, 平均 pass_rate=0.6407。
2. **reward override key 路径**: hydra 应为 `custom_reward_function.path` (顶层), 非脚本写的 `reward_model.custom_reward_function.path` (报 ConfigCompositionException)。改两个 train 脚本。
3. **parquet prompt 爆 max_length**: LCB 网格题 80MB test_code 进 A 的 input → sequence_length=105243>4096。`_format_test_code_for_prompt` 改为按大小选最小4对+每对截断800字符。重生成 parquet, 所有题 <4096 token。
4. **(最关键) test_code 截断 + delta 可比性**: py-spy 定位 "Training Progress 0/10 卡住" 实为 reward 卡在 LCB evaluator 跑 80MB test_code (单题最坏 246s, 20轨迹×5采样=卡死)。修复: `truncate_test_code(max_pairs=8, max_case_bytes=4096)` 丢弃大用例取最小8对, unit_tests 80MB→max4.3KB, evaluate 最坏78s。**S_0 预计算 import 同款 truncate_test_code 保证 delta 可比** (S_0 与 skill_pass 用同一份 test_code)。
5. **reward 返回类型**: verl batch reward_manager 以 `return_dict=True` 调用, 期望 dict (含 reward_tensor/reward_extra_info); 旧代码返回 tuple → `'tuple' object has no attribute 'get'`。改 `compute_custom_reward` 支持 return_dict 返回 dict。

## 截断后 S_0 分层 pass_rate (delta 可比性已保证)
easy 0.964 (13/14全对, delta≈0 锚点) / medium 0.705 (14全对/3部分/5全错, **主信号源**) / hard 0.403 (8全对/4部分/13全错, headroom最大 stretch)。

## 状态
- 阶段 0 (端到端打通) ✅ 完成。0.1 S0预计算 ✅, 0.2 reward单测 ✅, 0.3 verl 1-step ✅。
- 下一步: 阶段 1 — 申请托管 8×L40, Qwen3-4B 跑正式训练 (全量41题天然分布起步, max_iter=5, val监控过拟合)。用 `train_lcb_skillopt.sh` (已修 reward key), 拓扑 DP=8 (7-13 validate_config 验证过)。
- 正式训练前: 可选本机 0.6B 跑更长 (max_iter=5 完整) 验证训练曲线 + 过拟合监控逻辑, 但 0.6B 不出论文结果, 仅验流程。

Related: [[our-grpo-skill-method-design-20260721]], [[gpu-strategy-1card-smoke-8card-prod-20260722]], [[grpo-readiness-20260713]], [[api-concurrency-deepseek-20260716]]
