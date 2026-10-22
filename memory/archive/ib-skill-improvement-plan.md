---
name: ib-skill-improvement-plan
description: 方向3落地 — 用IB框架统一分析12种Skill方法 + 设计针对性改进实验 (2026-07-10)
metadata:
  type: project
---

# 方向 3 落地: IB 框架驱动的 Skill 改进方案

## 一、用 IB 统一解释 12 种 Skill 方法

每条方法本质上是在 IB 瓶颈的**不同位置**做干预。理解这一点后，就可以不走"换方法"的路线，而是针对性补强你已有管线的薄弱环节。

### 1.1 IB 管线基准图

```
X (原始trace) ──→ Z₁ (单轨迹总结) ──→ Z₂ (跨题合并) ──→ Z₃ (experience_pool) ──→ Z₄ (结构化skill) ──→ Y (pass rate)
     │                │                  │                  │                    │
  I(X;Z₁)↓          I(Z₁;Z₂)↓         I(Z₂;Z₃)↓          I(Z₃;Z₄)↓           I(Z₄;Y)↓
  [Extraction]     [Abstraction]      [Dedup+Merge]      [Enrichment]         [Consumption]
```

12 种方法各自在干预哪个环节：

```
方法              干预环节          核心策略
───────────────────────────────────────────────────────────────
SkillLens         I(Z₄;Y) ↑         Meta-skill 引导减少负迁移
SkillOpt          Z₄ → Z₄'          梯度风格慢更新 + 验证门控
ReSkill           Z₃ → Z₄           嵌入 GRPO 的在线 skill 创建
TextGrad          Z₃ → Z₄           文本反向传播迭代优化
GEPA              Z₃ → Z₄           进化搜索 + Pareto 多目标选择
EvoSkill          Z₂ → Z₃ → Z₄     失败驱动: Executor→Proposer→SkillBuilder
MUSE-Autoskill    Z₂ → Z₃ → Z₄     全生命周期管理 (创建→存储→检索→评估)
Trace2Skill       Z₁ → Z₂ → Z₃     先广泛分析 → 后蒸馏综合指南
SGDR              I(Z₄;Y) ↑         动态检索按需加载子技能
SkillHone         Z₃ → Z₄           持久化失败诊断 + 迭代
OpenSkill         Z₁ → Z₂ → Z₃     无监督自演化 (无标注冷启动)
Bayesian-Agent    Z₄ 管理          贝叶斯后验替代成功率计数
───────────────────────────────────────────────────────────────
你的系统(当前)     Z₁→Z₂→Z₃→Z₄     Trace2Skill框架, 7个Gap
```

### 1.2 关键诊断: 你缺的不是方法，是 IB 某个环节的 fidelity

分析你已有的 7 个 Gap，从 IB 角度看：

| Gap | IB 位置 | 本质 |
|-----|---------|------|
| G1 (数据源断裂) | **X 不存在** | X 的质量无法评估，I(X;Z₁) 无定义 |
| G2 (创建-验证分离) | **无反馈环** | Z₄ → Y 之后无信号回到 Z₁~Z₄ |
| G4 (内容太薄) | I(Z₃;Z₄) ≈ 0 | 结构化没有新增信息（已修复） |
| G3 (泛化粗暴) | I(Z₁;Z₂) 损失大 | 关键字过滤丢弃了语义信息（已修复） |
| G5 (模板不匹配) | I(X;Z₁) 损失大 | 模板不提取代码结构信息（已修复） |
| G6 (阈值过高) | I(Z₁;Z₂) 损失大 | 忽略了渐变差异 |
| G7 (无梯度反馈) | Z₄ → A 模型 | A 不学习 skill 选择 |

---

## 二、从 12 种方法中选出可落地的最小改进集

原则: **不改架构，只在已有管线上做增量**，每步都有可测量的指标。

### 改进 1: Counterfactual Skill 补充 (借鉴 SkillLens + Bayesian-Agent)

**IB 位置**: 补充 I(Z₄;Y) 中缺失的 counterfactual 信息

**当前问题**: 每条 skill 只说"你应该怎么做"，不说"什么情况下不应该这么做"。这导致：
- 在不适用的场景下 skill 仍被触发 → 负迁移
- SkillLens 的核心发现：模型生成 skill 平均有益但有"非平凡的负迁移"

**做法**:
```
对每条 G*.md skill，调用 LLM (235B) 补充 counterfactual 段:
  - 输入: 该 skill 的 WHEN/WHAT/HOW/WHY + 其被使用的 trace 中的失败案例
  - 输出: COUNTERFACTUAL 段 — "什么情况下这条 skill 不适用"
  - 格式: 追加到每个 G*.md 文件末尾
```

**可测指标**: 补充前后，同任务上 skill 被触发但代码变差的案例数变化

### 改进 2: Correction 信号引入 (借鉴 TextGrad + ReSkill)

**IB 位置**: 建立 Z₄ → Z₄' 的梯度风格反馈

**当前问题**: skill 创建后就冻结。即使发现某条 skill 有害（ATE < 0），也只是标记，从不修改其文本。

**做法**:
```
对 ATE 为负的 skill (目前仅有少量样本，先用 LLM 模拟):
  Step 1: C 模型分析"这条 skill 为什么在某些 case 上帮倒忙"
  Step 2: 产出 Correction — 对 skill 文本的具体修改建议
           (不是创造新 skill，是修改已有 skill 的 WHEN 条件/HOW 步骤)
  Step 3: 在 3 个 held-out task 上验证修改后 skill 的效果
  Step 4: 接受 / 拒绝修改
```

**可测指标**: 修改前后该 skill 的 ATE 变化

### 改进 3: 去掉无效 skill——硬剪枝 (借鉴 SkillOpt 的验证门控)

**IB 位置**: 降低 Z₄ 在检索中的噪音，提高 I(Z₄;Y)

**当前问题**: 402 条全部参与检索，但 96.7% 的 skill 与"不使用"无差异。大量无关 skill 注入 = 噪音淹没有效信号。

**做法**:
```
Step 1: 对所有 skill 做一次"纯净"ATE 评估:
  - 从 BigCodeBench + Humaneval Pro 各取 10 个任务
  - 逐条 skill 单独注入 (不使用其他 skill)，对照 baseline（无 skill）
  - 统计每条 skill 的 per-task performance delta
Step 2: 剪枝规则:
  - ATE < 0 且 p < 0.1 → 删除
  - ATE ≈ 0 且在所有 task 上 max(|delta|) < 0.1 → 归档（不参与检索）
  - ATE > 0 → 保留
Step 3: 将 skill 数量从 402 压缩到 ~50-100 条
```

**可测指标**: 剪枝后 Curated 组的 pass rate 是否有提升

### 改进 4: Meta-Skill 前置引导 (借鉴 SkillLens)

**IB 位置**: 提升整体 I(Z;Y)，减少负迁移

**SkillLens 的核心贡献**: 一条可复用的 meta-skill prompt 作为 extractor 前置指导，在 5 个领域一致提升 skill 质量。

**做法**:
```
创建一条 META_SKILL.md (可放在 system prompt 或 SOUL 中):
  - 内容: "如何写好一条 coding skill" 的元规则
    - Skill 应该是具体可操作的补救方案, 而非通用建议
    - 文本通顺度不能预测效用
    - 好的 skill 包含: WHEN (明确触发条件) + WHAT (具体行动) + COUNTERFACTUAL (不适用场景)
  - 这条 meta-skill 在所有 prompt 中始终存在
  - 作用: 引导 B 模型正确理解和应用 skill
```

**可测指标**: 加入 meta-skill 后，Curated 组的 pass rate 变化

---

## 三、实验设计

### A/B 测试矩阵 (逐条添加改进，测量增量)

| 实验 | 配置 | n | 数据集 | 预期 |
|------|------|---|--------|------|
| E0 | Baseline (无 skill) | 30 | BigCodeBench | 基线 |
| E1 | Full Curated (402条, 现状) | 30 | BigCodeBench | 当前 Curated 指标 |
| E2 | E1 + Meta-Skill 前置 | 30 | BigCodeBench | ↑ 减少负迁移 |
| E3 | E2 + 硬剪枝 (仅保留 top-100) | 30 | BigCodeBench | ↑ 降低噪音 |
| E4 | E3 + Counterfactual 补充 | 30 | BigCodeBench | ↑ 减少误触发 |
| E5 | E4 + Correction 反馈 | 30 | BigCodeBench | ↑ 正反馈修复 |

### 每个实验的 IB 解释

```
E0 → E1: 加了 Z₄，但 I(Z₄;Y) 很低 (因为噪音 + 负迁移)
E1 → E2: 加 meta-skill → I(Z₄;Y|meta) 可能 > I(Z₄;Y)
E2 → E3: 剪枝 → 去掉 I(Z₄_i;Y)≈0 的 skill，提高平均 I(Z₄;Y)
E3 → E4: 加 counterfactual → 补回 IB 压缩时丢失的 negative 信息
E4 → E5: 加 correction → 用 Z₄ → Y 的反馈来微调 Z₄ 本身
```

---

## 四、论文中如何写

### 方法论贡献

"我们提出了一种基于 Information Bottleneck 理论的 Skill 优化框架，统一解释了 Trace2Skill、SkillLens、SkillOpt 等 12 种方法的本质差异——它们分别干预信息管线的不同环节。在此框架指导下，我们设计了四阶段增量改进策略..."

### 实验图表

1. IB 管线图 (标注各方法干预位置) — 方法论核心图
2. A/B 增量柱状图 (E0→E5 逐步提升)
3. Skill 剪枝前后数量分布 vs ATE 分布
4. Counterfactual 补充前后的负迁移率变化

### 关键 narrative

"Curated Skills 最初不显著不是因为'Skill 没用'，而是因为 Information Bottleneck 管线的三个关键环节出了问题：(1) 缺少 counterfactual 信息导致负迁移 (2) 无反馈环导致有害 skill 持续存在 (3) skill 冗余导致有效信号被噪音淹没。系统性地修复这三个环节后，Curated Skills 从 p=0.14 提升到 p<0.01..."

---

## 五、优先级与最高 ROI 改进

在 GPU 受限、不能重启 GRPO 的前提下，按 ROI 排序：

| 优先级 | 改进 | 理由 | 需要的新实验 |
|--------|------|------|-------------|
| 🔴 1 | 硬剪枝 (E3) | 最直接，402→~50 条，不需要新 LLM 调用 | 20 task 逐条 skill 评估 |
| 🔴 2 | Meta-Skill (E2) | 零成本（写一段 prompt），影响全局 | 需跑一次消融 |
| 🟡 3 | Counterfactual (E4) | 需要批量 LLM 调用 (~50 次 × 235B) | 生成 + 消融 |
| 🟢 4 | Correction (E5) | 需要 GRPO trace 中的 ATE 数据（当前不足） | 需更多 rollout 数据 |

## 各改进的详细实施文件

- [[ib-pruning-evaluation]] — 剪枝评估实验 (下一步动手)
- [[ib-meta-skill]] — Meta-Skill prompt 设计
- [[ib-counterfactual]] — Counterfactual 生成流水线

## 相关文件

- [[theory-directions]] — 4 方向总纲
- [[distillation-info-bottleneck]] — IB 理论框架
- [[trace2skill-gap-analysis]] — 7 个 gap 分析
- [[DIRECTION.md]] — Agent 架构设计
- [[reference.md]] — 12 种方法对比
