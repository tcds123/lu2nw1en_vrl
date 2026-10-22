---
name: skillopt-code-result-20260715
description: SkillOpt 在代码(MBPP-Pro)上的判别实验结果 - val gate +21pp 但 test 不泛化(过拟合 val); 关键澄清: 本次未用我们的b_agent/池/GRPO, 纯SkillOpt方法
metadata:
  type: project
---

# SkillOpt × MBPP-Pro 判别实验结果 (2026-07-15)

env/实现见 [[skillopt-code-env-20260715]]。4 epochs optimize, train=75/val=38 gate/test=265, DeepSeek-V3.2。

## ⚠️ 实验边界澄清 (2026-07-15, 重要)
本次实验 = **SkillOpt 方法全套** (optimizer + val-gate 编辑 + 单 skill 迭代) + **SkillOpt 自带极简 rollout** (单次 chat_target 生成 + skill 注入) + 我们的 mbpp_pro 数据 + 我们的 assert 单测执行器。
**没有用**: 我们的 b_agent / 402 skill 池 / 检索注入 / ReAct / self-debug / GRPO —— 全部未参与。
⇒ 本实验只回答 "SkillOpt 方法在代码域有没有 headroom", **不回答** "我们的方法问题在哪"。
⇒ 代码域 test 不泛化是 **SkillOpt 方法 + 代码域** 的结果, 与我们的池/注入/GRPO 无关。

## Val gate 轨迹 (selection=38, 严格提升才 ACCEPT)
- baseline (seed v0000): val hard=0.5000
- ep2 step3: ACCEPT new best 0.6579 (v0003)
- ep4 step8: ACCEPT new best 0.7105 (v0008 = best_skill.md)
- ⇒ val gate lift: 0.5000 → 0.7105 (**+21.05pp**), val 上明确有效。

## 干净重测结果 (eval_only, 同 split seed42, test=265, 全 0 限流) — 权威
| skill | test hard | test soft |
|---|---|---|
| noskill (empty) | 0.5283 | 0.5834 |
| seed (initial.md = v0000) | 0.5283 | 0.5805 |
| **best (v0008, val 0.7105)** | **0.5132** | 0.5634 |

(optimize-internal test 数字被 429 污染, 已弃用, 见 [[skillopt-step0-results-20260714]] 同坑。)

## 最终判别结论 (干净, 确定)
- **best test 0.5132 < noskill 0.5283 (−1.5pp)**。val gate +21pp lift **完全没有泛化到 test**, 过拟合 38 题的 val set。
- 代码域 (mbpp_pro 单函数): SkillOpt 优化器 **val 有效但 test 不泛化**, headroom 真实但不稳定/被 val 过拟合吞掉。
- **对照数学** (Step0): best 54% > final 51% > seed 32% > noskill 24%, test 干净泛化。代码域远不如数学稳定。

## 对论文方向的意义
- 代码方向 (mbpp_pro) **坐实弱化**: skill 优化 headroom 小且 val 过拟合严重。继续代码需换更大/更难基准 (BigCodeBench/LCB) + 扩大 val set。
- **数学方向稳**: Step0 +21pp 干净泛化。论文主线数学优先。
- 我们此前 mbpp_pro 全 null (P0/Step1) 现可部分解释: 我们的池+检索注入零收益; 但本次证明即便用 SkillOpt 的单skill编辑+val gate, 代码域 test 也泛化差 ⇒ 代码单函数域本身对 skill 优化不友好 (模型已强, headroom 小, val 过拟合)。

## 控制变量诊断 (下一步, 回答"我们的方法问题在哪")
本次实验固定了优化方法=SkillOpt, 只变了数据域。要诊断**我们的方法**哪个组件是瓶颈, 需在**同一基准**(推荐数学, headroom大)上逐维切换:
- 基线A: SkillOpt 全套 (seed单skill + val-gate编辑 + 单次rollout) = 已证 +21pp test泛化
- 变量1 rollout: 换成 b_agent (单skill注入b_agent ReAct vs 单次生成) → 隔离 b_agent ReAct/self-debug 是否有害
- 变量2 表示: 换成 pool+检索 (单best_skill vs 402池检索注入) → 隔离 池检索是否稀释信号 (P0已暗示有害)
- 变量3 优化信号: 换成 GRPO+ATE (val-gate vs GRPO优势加权) → 隔离 GRPO优化器竞争力 = 论文 Step1 核心
- 变量4 skill来源: 换成 trace蒸馏 (手写seed vs extractor创建) → 隔离 创建阶段 = 论文 Step2
每步只变一个变量, 对位基线A, 看 test lift 是否保持。先做变量3 (Step1, 论文核心)。

Related: [[skillopt-code-env-20260715]], [[skillopt-step0-results-20260714]], [[p0-focused-injection-20260714]], [[research-direction-skillopt-20260714]]
