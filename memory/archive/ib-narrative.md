---
name: ib-narrative
description: 方向3论文叙事 — 从Information Bottleneck到具体改进, 论文方法论章节的结构设计 (2026-07-10)
metadata:
  type: project
---

# 论文方向 3 叙事: Information Bottleneck 驱动的 Skill 优化

## 核心故事线

### Hook (吸引点)

"为什么 402 条精心构建的 coding skill 对 B 模型没有帮助？"

答案不是"skill 没用"，而是 **Information Bottleneck 的三个环节出了问题**。

### 三步论证

#### Step 1: 诊断 — 用 IB 统一解释

```
Information Bottleneck (Tishby et al., 1999):
  min I(X;Z) - β·I(Z;Y)

其中:
  X = GRPO rollout trace (原始数据, 37K+ 条)
  Z = 蒸馏后的 skill 文本 (402 条)
  Y = B 模型代码生成 pass rate

IB 视角的诊断:
  ├─ I(X;Z) 太大 → skill 包含了无关信息（噪音）
  ├─ I(Z;Y) 太小 → skill 中的信息对下游任务帮助不大
  └─ 缺失 Z→Z' 反馈 → 坏的 skill 永远保留

对比 12 种方法在 IB 管线上的不同干预位置 → 选择性价比最高的修复
```

#### Step 2: 修复 — 基于诊断的系统性改进

| 修复 | IB 位置 | 做法 | 对应方法 |
|------|---------|------|---------|
| 硬剪枝 | 降低 I(Z;Y) 噪音 | 删除 Δ<0 的 skill, 保留 top 50-100 | SkillOpt 验证门控 |
| Meta-Skill | 提升 I(Z;Y|meta) | 注入"如何用好 skill"的元规则 | SkillLens |
| Counterfactual | 补回 IB 丢弃的 negative 信息 | 给每 skill 加"什么时候不适用" | Bayesian-Agent |
| Correction | 建立 Z→Z' 反馈 | 根据实验 delta 修改 skill 文本 | TextGrad + SkillOpt |

#### Step 3: 验证 — 增量实验

```
E0: Baseline (no skills)     [基线]
E1: Full Curated (402条)     [当前现状, Curated p≈0.14]
E2: E1 + Meta-Skill          [减少负迁移]
E3: E2 + Prune (~100条)      [降低噪音]
E4: E3 + Counterfactual      [减少误触发]
E5: E4 + Correction          [正反馈修复]

预期: E5 > E4 > E3 > E2 > E1 > E0
      E5 vs E0 p < 0.01
```

## 论文呈现

### 在 Method 章节 (3.x)

"3.x Information-Theoretic Framework for Skill Optimization"

1. 将 skill 蒸馏过程形式化为 Information Bottleneck 管线
2. 分析每条 skill 的 Δ = score(with_skill) - score(without_skill) 作为 I(Z;Y) 的代理
3. 提出基于 IB 的优化框架（剪枝 + meta-skill + counterfactual + correction）

### 在 Experiments 章节 (4.x)

"4.x Skill Optimization: Information Bottleneck Analysis"

1. **IB 诊断结果**: 402 skills 的 Δ 分布 (per-skill evaluation 的热力图)
2. **剪枝效果**: Pruning 前后 Curated 组的 pass rate 对比
3. **四阶段提升**: E0→E5 的增量曲线
4. **Case Study**: Best skill (Δ>0.5) vs Worst skill (Δ<-1.0) 的详细分析

### Figure 规划

1. **IB Pipeline Diagram**: 标注 X → Z₁ → Z₂ → Z₃ → Z₄ → Y 的每层压缩，以及 12 种方法各自的干预点
2. **Skill Delta Distribution**: 402 条 skill 的 Δ 直方图（正态分布？偏态？）
3. **Incremental Improvement Waterfall**: E0→E5 逐步提升的柱状图
4. **Skill-Pass Rate Heatmap**: skill × task 的 effect matrix

## 理论贡献表述

**不是**: "我们提出了一种新的 skill 优化方法"
**而是**: "我们建立了 Information Bottleneck 框架来分析 skill 蒸馏的信息损失，基于此设计了四阶段优化策略，实验证明..."

### 差异化的三个论点

1. **统一解释**: 12 种方法（Trace2Skill, SkillOpt, SkillLens, ReSkill 等）本质差异是 IB 管线干预位置不同
2. **定量诊断**: skill Δ 作为 I(Z;Y) 的度量，而非 binary success/failure
3. **可组合性**: 四个修复独立可插拔、独立可验证——剪枝降低噪音、meta-skill 提升利用、counterfactual 减少误触发、correction 闭环反馈

## 相关文件

- [[ib-skill-improvement-plan]] — 改进方案详细
- [[ib-pruning-evaluation]] — 剪枝评估实验
- [[distillation-info-bottleneck]] — IB 理论框架
- [[theory-directions]] — 4 方向总纲
