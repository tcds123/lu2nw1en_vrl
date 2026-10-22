---
name: causal-attribution-selfdebug
description: 方向4 — Self-Debug的因果归因与反事实推断，用do-calculus和counterfactual给出严格的归因结论 (2026-07-10)
metadata:
  type: project
---

# 方向 4: Self-Debug 的因果归因与反事实推断

## 核心问题

Self-Debug 在 3/4 数据集上显著（p < 0.05），但你计划做的子组件消融（A/B/C/D 四组，控制 detailed_feedback / detect_assertion / explicit_fix_prompt）只是**统计关联分析**，不是因果推断。

**核心理论问题**: 能否用因果推断框架（do-calculus, counterfactual reasoning, mediation analysis）给出更严格的归因——

1. 每个子组件对最终效果的**因果效应**（不是相关）
2. 子组件之间的**中介关系**（A 通过 B 起作用，还是直接起作用？）
3. 在反事实世界中，"如果没有某个组件，效果会差多少？"

## 形式化框架

### 因果图 (Causal DAG)

```
detailed_feedback (F) ──→ assertion_detection_capability (A_detect)
        │                           │
        ▼                           ▼
  model_understanding (U) ──→ fix_quality (Q) ──→ final_pass_rate (Y)
                                   ▲
explicit_fix_prompt (P) ──────────┘

task_difficulty (D) ──→ (moderates all edges)
```

### 可识别的因果量

**1. 平均因果效应 (ATE)**

ATE_F = E[Y | do(F=detailed)] - E[Y | do(F=simple)]

但不同于简单的分组对比，识别 do(F=detailed) 需要**无混杂假设**。在消融实验设计中：
- 随机分配保证 F ⟂ (U_P, U_task) ← 满足 backdoor criterion
- 所以标准的消融实验 ATE = 因果效应 ✓

但这太基础了，不是"理论贡献"。

**2. 中介分析 (Mediation Analysis)**

更深入的问题是：
- detailed_feedback 的效果有多少是**直接**的（F → Y）？
- 有多少是**通过改善 fix_quality 间接**的（F → Q → Y）？

这需要 Natural Direct Effect (NDE) 和 Natural Indirect Effect (NIE) 的分解：
TE = NDE + NIE

识别条件：sequential ignorability (需要 Q 可以实验操作化)

**3. 反事实归因 (Counterfactual Attribution)**

对于每个具体任务 t，计算：
- Y_{t}(F=detailed, P=explicit) — 实际观测到的分数
- Y_{t}(F=simple, P=explicit) — 反事实：如果反馈是简化的会怎样？
- Individual Treatment Effect = 两者之差

这需要：同一个任务上的 paired 实验（对同一个任务分别跑 detailed 和 simple 模式）。

**4. 必要的 vs 充分的因果分析**

- Probability of Necessity: P(Y_would_fail | Y_did_pass, component_removed)
  - 问：在 self-debug 修复成功的任务中，有多大比例是因为 detailed_feedback 的存在才成功的？
- Probability of Sufficiency: P(Y_would_pass | Y_did_fail, component_added)

### 与现有消融的区别

| 维度 | 统计消融 (你计划做的) | 因果归因 (可做的) |
|------|---------------------|-------------------|
| 回答的问题 | "关掉 A 后分数变了多少？" | "A 通过什么路径影响分数？如果关掉 A 但保持 B 通过 A 的中介效应，分数会怎样？" |
| 需要的设计 | between-group 对比 | within-task paired + 中介变量测量 |
| 分析工具 | t-test, ANOVA | mediation analysis, counterfactual prediction |
| 论文价值 | 基本要求 | 亮点贡献 |

## 已有数据支持

### 可直接分析的

1. **Ablation 结果 JSON 中有 per-task 数据**:
   - 对同一个任务，不同组（Baseline/Curated/SelfDebug）的 score
   - 虽然不是 paired（不同随机 seed 可能生成不同代码），但 seed=42 固定后可以近视为 paired
   - 可以直接计算 Individual Treatment Effect (ITE) 分布

2. **Self-Debug trace 中有中间变量**:
   - `n_rounds`: 修复轮数
   - `self_debug_score`: 修复后分数
   - 可以分析 rounds 的中介效应：F/P/Q → rounds → Y

3. **任务级元数据**:
   - 任务难度（从数据集分类）
   - 任务类型（逻辑/算法/数据结构/字符串处理等）
   - 可以分析：哪些任务类型上哪个组件最重要？

### 需要补充的实验

| 实验 | 目的 | 方法 | 数据集 |
|------|------|------|--------|
| Paired Counterfactual | 同一任务上所有变体的 paired 评测 | 对每个任务跑 4 种配置 (A/B/C/D)，固定 seed | Humaneval Pro n=50 |
| Mediation Measurement | 测量 fix_quality 中介变量 | 在 self-debug 每轮后记录 fix_quality (LLM评分) | 同上 |
| Necessity Analysis | 计算"没有 X 就会失败"的概率 | 从反事实预测中统计 | 同上 |
| Task-Type Moderation | 任务特征 × 组件效应的交互 | 标注任务类型标签后做 moderation analysis | Humaneval Pro n=100 |

## 理论价值

1. **"Self-Debug 为什么有效"** — 不仅仅是"效果好"，而是"效果好是因为 X 通过 Y 路径起作用"
2. **"如何设计更好的 Self-Debug"** — 如果中介分析显示大部分效应通过 feedback → understanding → fix 路径 → 优化 feedback 质量比优化 fix prompt 更重要
3. **"在什么任务上 self-debug 最有效"** — moderation analysis 揭示难度/类型的条件因果
4. **"与 Reflexion/Self-Refine 的区别"** — 从因果归因角度，你的方法可能激活了不同的因果路径

## 相关文件

- [[theory-directions]] — 总览
- [[selfdebug-ablation-plan]] — 现有消融设计
- [[sd-ablation-running]] — 消融运行状态
- [[distillation-info-bottleneck]] — 方向3 (counterfactual 是 IB 中缺失的关键信息类型)
