---
name: search-landscape-analysis
description: 方向2 — Skill组合优化的搜索空间与Reward Landscape分析，分析2^402空间中搜索策略的有效性条件 (2026-07-10)
metadata:
  type: project
---

# 方向 2: Skill 组合优化的搜索空间与 Reward Landscape 分析

## 核心问题

你有 402 条 skill（G0~G401），每条可以被选择或不选择，搜索空间大小为 \( 2^{402} \)。

A 模型（4B, GRPO）试图学习从任务 → 最优 skill 组合的映射。但这个映射的有效性依赖于 reward landscape 的结构特性。

**核心理论问题**: 在什么条件下，搜索 \( 2^{402} \) 空间的启发式策略（如 GRPO 的 rollout 探索）能够收敛到接近最优的组合？

## 形式化框架

### Reward Landscape 定义

对于任务 t，skill 组合 s ⊆ {G0, ..., G401}，定义 reward：
r_t(s) = E_{code ~ B(t, s)}[score(code, t)]

其中 B(t, s) 是以 skill 组合 s 为 context 时 B 模型对任务 t 生成的代码。

### Landscape 特性分析

**特性 1: Smoothness (平滑性)**
- 如果添加/移除单条 skill 对 r 的改变很小 → 平滑 landscape → 梯度信号可靠
- 如果单条 skill 可以剧烈改变 r → 崎岖 landscape → 随机探索困难

**特性 2: Monomodality (单峰性)**
- reward 是否只有一个 global maximum，还是多个 local maxima？
- 多峰意味着不同 skill 组合可能等效地好（但它们应对不同类型的任务）

**特性 3: Skill Interactions (交互效应)**
- Additive: r(s₁ ∪ s₂) ≈ r(s₁) + r(s₂) - r(∅)
- Synergistic: r(s₁ ∪ s₂) > r(s₁) + r(s₂) - r(∅)
- Antagonistic: r(s₁ ∪ s₂) < r(s₁) + r(s₂) - r(∅)
- 当前的 402 条 skill 来自 experience_pool 独立蒸馏，未考虑交互效应

**特性 4: Diminishing Returns (边际效应递减)**
- 加第 1 条 skill 的提升 vs 加第 100 条的提升
- 如果边际效应递减很快 → 小组合就足够，不需要探索全空间

**特性 5: Task Conditioning (任务条件性)**
- 最优 skill 组合 s*(t) 是否高度依赖 t？
- 如果 s* 对所有 t 都差不多 → 一条固定 prompt 就好，不需要"进化"
- 如果 s* 随 t 变化很大 → 需要 task-conditioned selection（这正是 A 模型的价值）

## 已有数据支持

### 可分析的数据

1. **Skill ATE 得分** (score_skills.py 产出): 目前仅 13 个 skill 有非零 score_delta
   - 可以分析：已有 ATE 的正负分布 → 多少 skill 有益 vs 有害
   - 可以分析：高 ATE skill 之间的交互（同时使用时的联合效应 vs 单独效应之和）

2. **消融实验结果**:
   - Curated (使用全部匹配 skill) vs Random (随机选择 skill) → 差异不大 = landscape 可能很平
   - Curated 仅在 BigCodeBench 上显著 → landscape 结构是难度依赖的

3. **BigCodeBench 结果**:
   - Curated p=0.0118 是首次显著 → 困难任务的 landscape 峰值更突出？
   - 或者：简单/中等任务上，多种 skill 组合都能达到上限？

### 可计算的指标

从现有 trace 数据：
- 同任务不同 skill 组合的 reward 方差 → landscape roughness
- Skill 共现矩阵 → 哪些 skill 倾向于一起被使用
- 燃尽分析：添加 skill 数量 vs pass rate 的关系

## 需要补充的实验

| 实验 | 目的 | 方法 |
|------|------|------|
| Additive Check | 验证 skill 效应是否近似可加 | 选 10 个高 ATE skill，穷举 2^10=1024 组合在固定任务集上 |
| Interaction Detection | 检测 skill 间交互 | 2^10 数据中计算交互项 |
| Diminishing Returns | 验证边际效应递减 | 从已有结果中分析 Curated (多 skill) vs Single Best Skill |
| Landscape Visualization | 降维可视化 reward landscape | PCA/t-SNE on skill embedding → reward surface |

## 理论价值

1. **"为什么 GRPO 能（或不能）学好 skill 选择"** — 取决于 landscape 特性
2. **"什么时候 context 进化比参数微调更有优势"** — 当 landscape 有多个 distinct modes 对应不同任务类型时
3. **"多少条 skill 是'够的'"** — 由边际效应递减率决定

## 相关文件

- [[theory-directions]] — 总览
- [[context-as-soft-parameters]] — 方向1 (等效性需要理解 landscape 的全局结构)
- [[distillation-info-bottleneck]] — 方向3 (蒸馏质量决定 skill 在 landscape 上的有效性)
