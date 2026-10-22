---
name: dataset-constraint-20260715
description: 数据集硬约束 - 小论文主数据集必须是代码benchmark, 不限HumanEval/MBPP可换更难的, 数学仅补充
metadata:
  type: project
---

# 数据集选型约束 (2026-07-15, 用户明确, 两次澄清)

## 硬约束
- **小论文主数据集 = 代码 benchmark**。不限于 HumanEval/MBPP, 可更换别的代码 benchmark, 只要代码相关、能衔接论文叙事。
- 数学数据集 (LiveMathematicianBench 等) **仅作补充/验证通用性**, 不是主线。
- 不得表述为"数学优先"。即便数学 headroom 大、test 泛化好, 也不能因此把主线切到数学。
- 背景: 本工作是大论文的相关部分, 小论文锁定代码域。

## 选基准的现实考量
- 理想代码基准: 模型**未饱和**(有 headroom) + **能跑起来**(基础正确性非零) + **确定性验证器**。
- 已知:
  - mbpp_pro (378): 模型偏强, noskill full-pass 0.53, headroom 小, val(38)过拟合严重 (best test 0.513 < noskill 0.528, 见 [[skillopt-code-result-20260715]])。
  - humaneval_pro (164): 同格式, 待测。
  - BigCodeBench (57): 之前跑过, pass 全线负 (基础正确性都没有, skill 帮不上) — 不适合。
  - APPS_Plus (7413): stdin/stdout 比对, **不兼容**我们的 assert 执行器, 要另写执行器。
  - LiveCodeBench: 待评估。
- ⇒ 若 mbpp/humaneval 都 headroom 不足, 换更难但能跑的代码基准是合规且必要的。

## 对实验设计的含义
- 控制变量诊断我们的方法瓶颈, **必须在代码基准上做**。
- 代码域 val 过拟合是必须解决的问题 (扩 val / k-fold / 换 headroom 更大的基准)。
- 数学作"方法流通用性"补充, 放论文后段。

Related: [[skillopt-code-result-20260715]], [[skillopt-code-env-20260715]], [[research-direction-skillopt-20260714]]
