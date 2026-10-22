---
name: distillation-info-bottleneck
description: 方向3 — 经验蒸馏的信息瓶颈，用Information Bottleneck理论定量分析trace→skill蒸馏的信息损失 (2026-07-10)
metadata:
  type: project
---

# 方向 3: 经验蒸馏的信息瓶颈

## 核心问题

Experience Pool 的构建过程：GRPO rollout trace (完整 rollout 数据) → extractor.py 蒸馏 → experience_pool.json (401条) → enhance_skills.py 结构化 → G*.md (402条 WHET/WHAT/Example/WHY)。

每一层压缩都在丢弃信息。当前 Curated Skills 效果不佳（3/4 数据集不显著），Trace2Skill Gap 分析找到了 7 个实现层面的问题，但问题可能更深：

**核心理论问题**: trace → skill 的信息压缩中，什么信息是"必须保留"的、什么可以"安全丢弃"？能否用 Information Bottleneck (IB) 理论给出定量的保留/丢弃边界？

## 形式化框架

### IB 视角的蒸馏过程

Information Bottleneck (Tishby et al., 1999) 的核心思想：
min I(X; Z) - β·I(Z; Y)

其中 X = 原始数据 (trace)，Z = 压缩表示 (skill)，Y = 下游任务性能 (代码生成评分)。
- I(X; Z) 度量压缩程度（越小越压缩）
- I(Z; Y) 度量表示对下游任务的预测能力（越大越好）
- β 控制 trade-off

### 蒸馏的 IB 分层

```
X: GRPO rollout trace (37K+ 条, 完整 rollout 数据)
  ↓ I(X; Z₁) 损失 ~?
Z₁: 单轨迹总结 (extractor templates.py 中的 single-trace summary)
  ↓ I(Z₁; Z₂) 损失 ~?
Z₂: 跨题经验 (cross-task merge)
  ↓ I(Z₂; Z₃) 损失 ~?
Z₃: experience_pool.json (401 条, 每条 ~100-200 字)
  ↓ I(Z₃; Z₄) 损失 ~?
Z₄: Structured Skill G*.md (WHEN/WHAT/Example/WHY, ~1000-1200 字/条)
  ↓ I(Z₄; Y) ~?
Y: B 模型代码生成 pass rate
```

### 分析维度

**维度 1: 信息内容分类**
trace 中包含的信息类型：
- **Context-Question 关联** (什么时候有效): [WHEN] → 被 IB 压缩了多少？
- **Strategy-Action 映射** (怎么做): [WHAT/HOW] → 被 IB 压缩了多少？
- **Causal Rationale** (为什么有效): [WHY] → 被 IB 压缩了多少？
- **Counterfactual** (在什么情况下无效): 当前 skill 完全缺失 → 可能是最关键的损失！
- **Gradient Signal** (效果的强度/方向): ATE score_delta → 仅 13/402 有非零值说明什么？

**维度 2: 信息损失的可检测标志**
- 语义去重过度（cache.py 的 dedup 阈值过高）→ 丢失多样性信息
- 泛化约束粗暴（G3: 正则过滤替代 LLM 语义判断）→ 丢失任务特异性
- 对比信号阈值过高（G6）→ 丢失渐变差异信息
- 无 counterfactual → 丢失"什么不应该做"的信息

## 已有数据支持

### 可以直接分析的

1. **Skill ATE 分布**: 13/402 有非零 score_delta → 意味着 96.7% 的 skill 在当前评测中与"不使用"无差别
   - 解释 1: skill 确实无效 → 蒸馏时丢失了关键因果信息
   - 解释 2: ATE 数据不足 → 大多数 skill 还没被触发过
   - 解释 3: skill 效应需要特定任务才能显现（如 BigCodeBench 的特殊性）

2. **experience_pool 的语义聚类** (cache.py 中有 clustering 功能):
   - 聚类数与原始 trace 数的比例 → 信息压缩率
   - 类内方差 / 类间方差 → 表示的判别能力

3. **结构化前后的对比**:
   - 原始 experience_pool 条目 (100-200字) vs 增强后 G*.md (1000-1200字)
   - 增强过程中 LLM 添加了什么新信息？这些信息来自 LLM 的先验知识还是从 trace 中恢复的？

### 需要计算的指标

| 指标 | 含义 | 计算方法 |
|------|------|---------|
| Compression Ratio | 从 trace 到 skill 的信息压缩倍率 | raw_trace_tokens / skill_tokens |
| Coverage | skill 覆盖了多少 trace 模式 | trace中的模式 ∩ skill中的模式 |
| Fidelity | skill 精确保持了 trace 中成功模式的程度 | (skill产生的行为) vs (trace中的行为) |
| Counterfactual Coverage | 有多少"失败模式"被记录 | skill 中 negative examples 的比例 |

## 需要补充的实验

| 实验 | 目的 | 方法 |
|------|------|------|
| 信息截断实验 | 测试不同压缩程度下的 skill 有效性 | 从原始 trace 中提取不同粒度的 summary (50字/200字/500字/1000字) |
| Counterfactual Skill 补充 | 为每条 skill 添加 counterfactual (什么情况下不适用) | LLM 从 trace 失败案例中生成 negative examples |
| 跨层 fidelity 评估 | 检查每层压缩丢失了什么类型的信息 | 逐层抽样，人工/LLM评估信息保留率 |

## 理论价值

1. **"为什么当前 Curated Skills 效果不好"** — 不仅仅是 7 个 Gap 的实现问题，而是 structural information loss
2. **"蒸馏的 optimal stopping point"** — 什么时候继续压缩 vs 停止压缩？IB 的 β 提供理论指导
3. **"什么信息类型最关键"** — 如果 counterfactual 和 gradient 是最关键信息 → 重新设计蒸馏流程
4. **"与 SkillOpt/SkillLens/ReSkill 的比较"** — 这些方法的差异可以从 IB 角度统一解释

## 相关文件

- [[theory-directions]] — 总览
- [[trace2skill-gap-analysis]] — 7个Gap分析
- [[search-landscape-analysis]] — 方向2 (蒸馏质量 = landscape 定义的精度)
- [[causal-attribution-selfdebug]] — 方向4 (计数器事实 vs 反事实)
