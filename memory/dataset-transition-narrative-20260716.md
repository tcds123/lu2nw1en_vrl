---
name: dataset-transition-narrative-20260716
description: mbpp/humaneval null 结果的战略重定位 - 作为大小论文衔接的过渡证据(模型scaling饱和+纯模型→agent架构), 不试humaneval_pro, 换更难代码基准后尽快进入方法优化
metadata:
  type: project
---

# 数据集转换的论文叙事 (2026-07-16, 用户明确)

## 战略重定位: null 结果是过渡证据, 不是失败
此前 mbpp_pro 双 split 全 null (2-1-7 best 0.5132<noskill 0.5283; 1:2:7 best 0.4604<baseline 0.4679) 一直当作负结果。用户 2026-07-16 重定位: **这些数据有意义, 用于大小论文衔接的过渡论证**。

## 衔接论证 (回答"小模型→大模型为何不沿用 humaneval/mbpp")
当审稿/答辩问"为什么从小模型(A, Qwen3-4B)跳到大模型(B, DeepSeek-V3.2)时, 不用相同的 humaneval/mbpp 做优化"——用这些数据回答两点:
1. **模型参数变大, 能力提升 → 旧基准饱和**: B 在 mbpp_pro noskill 已 0.53、humaneval_pro 0.41, skill 优化的 headroom 被模型自身能力吃掉 (val 有效但 test 不泛化, 见 [[skillopt-code-result-20260715]]、[[code-benchmark-headroom-20260715]])。小模型时代有 headroom 的基准, 对大模型已不够难。
2. **从纯模型到 agent 架构的提升**: 纯单次生成 + skill 注入在饱和基准上零收益, 说明要继续涨分必须换 agent 架构 (ReAct/self-debug/多步) 而非继续在单函数题上磨 skill。

⇒ 这两点共同证明**数据集转换的合理性**: 旧基准既被模型能力饱和, 又被单函数域限制, 必须换更难、更适配 agent 架构发挥的代码基准。

## 决策 (用户 2026-07-16)
- ❌ **不再试 humaneval_pro**: headroom 虽大但仍是单函数题, 同类问题 (val 过拟合/test 不泛化) 大概率重演, 浪费 API。
- ✅ **换更适配、难度稍高的代码基准**: 标准 = (1) 模型未饱和有 headroom (baseline 非零但低于 0.4) (2) 能跑起来 (3) 确定性验证器 (优先 assert 兼容我们的执行器, 否则另写) (4) 衔接 agent 架构叙事 (多步/复杂任务, 让 skill 有作用空间)。
- ✅ **确定数据集后尽快进入方法优化阶段**: 不再在数据集探查上久留, 选定即跑方法对比 (Step1 GRPO+ATE vs SkillOpt val-gate, 见 [[research-direction-skillopt-20260714]] Step1)。

## 候选基准 (待评估, 见 [[dataset-constraint-20260715]])
- LiveCodeBench: 待评估 (格式/验证器/可跑性)。
- BigCodeBench (57): 已排除 (pass 全线负, 基础正确性都没有)。
- APPS_Plus (7413): stdin/stdout, 不兼容 assert 执行器, 要另写执行器。
- 需找: 比 mbpp/humaneval 难, 但 baseline 非零, 且 assert 或可写验证器。

Related: [[dataset-constraint-20260715]], [[skillopt-code-result-20260715]], [[code-benchmark-headroom-20260715]], [[research-direction-skillopt-20260714]], [[api-concurrency-deepseek-20260716]]
