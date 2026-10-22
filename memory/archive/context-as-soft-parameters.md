---
name: context-as-soft-parameters
description: 方向1 — Context作为软参数的等效性理论，建立context变化与参数空间变化的数学对应 (2026-07-10)
metadata:
  type: project
---

# 方向 1: Context 作为"软参数"的等效性理论

## 核心直觉

传统观点：模型能力 = f(参数θ, 输入x)。参数通过梯度下降改变，context 只是"提示"。

但你的实验暗示：**context 可以等价地改变模型行为分布**，且不需要修改 θ。

如果把 context 看作一种"软参数" (\(\theta_{\text{soft}}\))，它与硬参数 θ 的区别是什么？等效的界限在哪里？

## 形式化框架

### 定义: Context-Induced Distribution Shift

对于给定任务集合 T = {t₁, ..., tₙ}，模型 M 的硬参数为 θ。记 M 在输入 x 上的输出分布为 P_θ(·|x)。

一个 context C（system prompt + skills + memory + rules）作用在 M 上产生新的有效行为分布：
P_{(θ, C)}(·|x) = P_θ(·|C ⊕ x)

其中 ⊕ 表示 context 与输入的组合（通常为拼接）。

### 核心理论问题

**问题 1 (等效性存在性)**: 是否存在参数微调 Δθ，使得对所有 t ∈ T：
D(P_{(θ, C)}(·|t) || P_{θ+Δθ}(·|t)) < ε

即：context 的效果能否被一个参数更新等效替代？

**问题 2 (等效成本)**: 如果存在，∥Δθ∥/|C|（参数变化量 vs context 长度之比）是多少？这度量了"用 context 替代微调"的效率。

**问题 3 (互补性)**: context 能表达的分布变换与参数微调能表达的分布变换之间，有多少重叠？（交集 = 可相互替代的部分；差集 = 各自独有的能力）

### 可检验的假设

| 假设 | 描述 | 检验方法 |
|------|------|---------|
| H1: 局部等效 | 存在某些任务，context 效果 ≈ prompt tuning 效果 | 对比 self-debug prompt vs lora fine-tune |
| H2: 压缩效率 | 优质 context 以更少的 token 代价产生更大的行为偏移 | 计算 ∥ΔP∥₂ / |C|_tokens |
| H3: 互补空间 | context 擅长局部约束，参数擅长全局偏移 | 分析 self-debug 在结构性任务 vs 局部 bug 任务上的差异 |

## 已有数据支持

### 来自现有实验的观察

1. **Self-Debug 的 11.6% 提升**: 这是纯粹的 context 级变化（添加了执行反馈和修复指令），没有修改 θ。如果能在某参数微调 baseline 上对比，就能估算等效 Δθ。

2. **Curated Skills 在 BigCodeBench 上首次显著 (p=0.0118)**: 困难任务中 context 差异更明显 → 暗示 context 效应是任务难度的函数。

3. **难度梯度模式**: 
   - 中等难度: SelfDebug >> Curated ≈ Baseline → 流程性 context（执行反馈）更有效
   - 困难: SelfDebug ≈ Curated > Baseline → 知识性 context（skill 指南）+ 流程性 context 都必要
   - 极端: 全部不显著 → context 的能力上限 = 模型原始能力的函数

### 可以分析的指标

- 从 ablation 结果 JSON 中提取：
  - 每条 skill 被触发 vs 未触发时的行为偏移 (ATE)
  - self-debug round 数与行为变化量的关系
  - 不同任务类型上的 context 效应量

## 需要补充的实验

| 实验 | 目的 | 数据集 |
|------|------|--------|
| Prompt Tuning Baseline | 对比 lora/prompt tuning 与 self-debug context 的效果差异 | HumanEval Pro n=50 |
| Context Token Efficiency | 压缩/膨胀 system prompt，测量行为偏移函数 | 已有数据可分析 |
| Task-Type × Context-Type Matrix | 知识的 vs 流程的 context 在不同难度任务上的效应 | 已有数据可分析 |

## 理论价值

如果能证明：
1. Context 诱导的分布偏移可以被一个有限范数的 Δθ 近似 → "Training-Free" 不仅仅是工程术语，而是有严格数学基础的概念
2. Context 和参数微调的互补性 → "上下文空间进化"和"参数空间进化"是正交的两个优化维度

## 相关文件

- [[theory-directions]] — 总览
- [[achievements]] — 实验数据
- [[search-landscape-analysis]] — 方向2 (搜索景观分析依赖这个方向的形式化)
