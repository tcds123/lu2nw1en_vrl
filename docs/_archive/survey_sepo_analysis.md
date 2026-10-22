# SePO: Self-Evolving Prompt Agent for System Prompt Optimization — 论文精读

> **论文信息**
> - arXiv:2606.04465v1, 2026年6月
> - 作者: Wangcheng Tao (NUS), Han Wu (CityU HK), Weng-Fai Wong (NUS)
> - 核心贡献: 自指涉(self-referential)的prompt优化，让prompt agent自身也进化
> - 与本项目关系: ★★★★★ 最直接对应的prompt优化方法

---

## 1. 核心问题

现有prompt优化方法(TextGrad, PromptBreeder, MetaSPO)都有同一个根本缺陷：**optimizer本身的system prompt是手工设计的，从不进化**。

| 方法 | 优化谁 | 优化器自身 |
|------|--------|-----------|
| 普通Prompt Opt | 只优化task agent的prompt | 固定 |
| PromptBreeder | task prompt + mutation prompt | meta-stack顶固定 |
| **SePO (Ours)** | **task agent + prompt agent自身** | **同程序进化** |

## 2. 自指涉设计 (Self-Referential Closure)

SePO把prompt agent自身的system prompt `˜p` 也作为优化变量。定义一个 **prompt task** `˜T = (˜D, ˜S)`:

- `˜D`: prompt agent的输入集合 `˜x = (T, A, E)` —— 任务描述、task agent、评估结果
- `˜S`: 衡量"refined prompt p'是否比p在任务T上提升了task agent准确率"

同一个优化程序同时作用于 `p` (task agent的prompt) 和 `˜p` (prompt agent自身的prompt)。这就是自指涉闭环。

## 3. 两阶段训练

### Stage 1: Pre-training — 进化prompt agent自身

在多任务池上运行开放式进化搜索:

```
输入: seed prompt ˜p⁽⁰⁾, 多任务池 T = {AIME'25, ARC-AGI-1, GPQA, MBPP, Sudoku, ...}
输出: 进化后的最佳prompt agent prompt ˜p⋆

每代:
  1. 从archive中按 score_child_prop 采样parent
  2. LLM生成K=2个child变体
  3. 每个child在多任务子集上比较准确率delta
  4. 如果child的score >= parent → 入archive
```

### Stage 2: Fine-tuning — 优化task agent

固定进化后的 ˜p⋆，对特定target task优化task agent的prompt p。

## 4. 开放式进化搜索 (Open-Ended Evolutionary Search)

这是SePO的核心算法机制。

### Parent Selection: `score_child_prop`

```
Pr(a) ∝ exp((s_a - max_b s_b)/τ) × 1/(1 + c_a)^0.4
τ = max(2/3(max_b s_b - min_b s_b), 0.05)
```

- `s_a`: parent a的score
- `c_a`: a已经产生的child数量
- τ自适应: 分数分散时紧聚焦、集中时宽探索
- child-count penalty: 防止重复探索同一parent

### Scoring: accuracy delta

```
s_child = acc_new - acc_old   (相对parent的准确率变化)
```

### Archive Admission: keep_better

child的score >= parent → 入archive；否则丢弃。

### 关键超参

| 参数 | 值 |
|------|-----|
| 代数 G | 5 |
| 每代child K | 2 |
| batch size | 16 (fail:success ≈ 1:1) |
| 并行worker | 2 |

## 5. 关键设计细节: Fail/Success 1:1 采样

只给failure → 过度矫正边缘case；只给success → 没东西可修。**1:1平衡是最优的**。

## 6. 实验结果

### 主结果 (任务agent: DeepSeek-V3.2, prompt agent: Gemini 3.1 Pro Preview)

| Method | AIME'25 | ARC-AGI-1 | GPQA | MBPP | Sudoku | **Avg** |
|--------|---------|-----------|------|------|--------|---------|
| Manual-CoT | 57.55 | 37.30 | 76.46 | 91.20 | 96.95 | **71.89** |
| TextGrad | 55.99 | 34.75 | 74.44 | 90.15 | 96.60 | **70.39** |
| MetaSPO | 57.71 | 37.27 | 75.51 | 89.30 | 96.80 | **71.32** |
| SePO-Specialist | 60.94 | 37.46 | 76.72 | 95.55 | 99.80 | **74.09** |
| **SePO-Generalist** | **64.22** | **43.39** | **78.18** | **96.20** | **99.90** | **76.38** |

- 相对Manual-CoT: **+4.49 avg**，每个任务上都是最优

### 消融实验

| 变体 | Avg | 变化 |
|------|-----|------|
| Manual-CoT | 71.89 | baseline |
| **SePO-Generalist** | **76.38** | **+4.49** |
| w/o Self-Improvement | 74.94 | **-1.44pp** (prompt agent自身进化的贡献) |
| w/o Open-Ended Evolution | 72.64 | **-3.74pp** (开放式进化的贡献) |

### 模型互换鲁棒性

换 Claude Opus 4.6 做prompt agent + Gemini Flash Lite 做task agent:
- Manual-CoT: 67.95
- SePO-Generalist: 70.08 → **+2.13pp** (跨模型泛化)

### 成本分析

| 方法 | 每任务成本 |
|------|-----------|
| TextGrad | $14.75 - $26.52 |
| SePO-Specialist | $5.72 - $37.63 |
| **SePO-Generalist** | **$7.43均摊** (pre-train $37.14/5 + fine-tune $2.41-$15.51) |

## 7. 与本项目的核心差异

| 维度 | SePO | 本项目 |
|------|------|--------|
| 优化对象 | **prompt文本**(进化搜索) | A模型**参数**(GRPO梯度更新) |
| 搜索方法 | 开放式进化: archive+mutation+selection | GRPO group-relative advantage |
| 奖励信号 | accuracy_delta (child vs parent) | 70%单测 + 30% C模型评判 |
| A模型自身prompt | **也进化**(自指涉) | **固定** |
| 训练阶段 | pre-training → fine-tuning | 单阶段 |
| 需要C模型 | 否(直接看task agent准确率) | 是 |

## 8. 对本项目的启示

### 可直接借鉴的技术

1. **自指涉设计**: A模型自身(4B)的system prompt也可加入优化空间——本项目的A模型system prompt目前固定，SePO证明自指涉可贡献~1.44pp
2. **开放式进化archive**: archive + score_child_prop采样可作为GRPO探索策略的补充
3. **fail/success 1:1采样**: 本项目C模型评判时同此原则
4. **两阶段训练**: 先多任务pre-train再单任务fine-tune

### 与本项目的互补关系

SePO是**文本级进化**(不更新参数)，本项目是**参数级进化**(GRPO真更新)。两者可结合:
- 用SePO的archive作为GRPO候选池初始化
- 训练初期用SePO式快速搜索找好prompt起点，再进入GRPO精细优化
- SePO的成本($7.43/task)可作为GRPO每一步成本的参照基线

---

**参考文献**: Tao et al., "SePO: Self-Evolving Prompt Agent for System Prompt Optimization", arXiv:2606.04465v1, 2026.