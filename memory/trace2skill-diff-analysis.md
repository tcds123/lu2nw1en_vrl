---
name: trace2skill-diff-analysis
description: Trace2Skill (arXiv:2603.25158, Ni et al.) vs lunwen 实现差异对比；含 2026-07-14 arXiv 核验修正与 Step1 判别性结论 (2026-07-13/14)
metadata:
  type: project
---

# Trace2Skill 论文实现 vs lunwen 实现 — 差异分析

## 背景

Trace2Skill 论文 (arXiv:2603.25158) 的方法在多个 benchmark 上有效。
我们在 lunwen 中按照"类似方法"从 GRPO trace 蒸馏的经验产出了 402 条 skill，
但 P1 剪枝评估发现 **70% 有害**，且 Curated Skills 在 3/4 数据集上不显著。

这次深入对比 youtu-agent 中的 Trace2Skill 实现和我们的 experience/extractor.py +
templates.py，找出根本差异。

## ⚠️ 2026-07-14 arXiv 核验修正（在阅读下文 6 条差异前必读）

下文写于 2026-07-13，将 Trace2Skill 归为 "youtu-agent 实现"。2026-07-14 用 arXiv API
直接核验 `id_list=2603.25158`，确认论文真实存在，但其摘要与下文若干关键描述**不符**。
下表的"已核验"列来自 arXiv 摘要原文，"原描述"列是 7-13 的旧表述。

**论文核验信息**: arXiv:2603.25158v5, "Trace2Skill: Distill Trajectory-Local Lessons into
Transferable Agent Skills", Ni et al., 发表 2026-03-26, v5 更新 2026-06-04,
标注 "Work in Progress"。摘要要点（直接引自 arXiv）：

> "...a framework that consolidates broad execution trajectories **in parallel** into a
> unified skill directory through **inductive reasoning** over agent experience. Trace2Skill
> supports both **deepening existing human-written skills** and **creating useful skills from
> weak LLM-generated drafts**... across diverse domains, including **office workflows, math
> reasoning, and vision QA**... Trace2Skill **outperforms sequential skill editing and
> ReasoningBank-style retrieval memories**, compresses recurring failures and workarounds into
> **standard operating procedures (SoPs)**, and yields **portable skills that can be reused
> without parameter updates or test-time retrieval**."

| # | 原描述（7-13，下文） | 核验后修正 |
|---|---|---|
| 归属 | "Trace2Skill (youtu-agent) 实现" | ❌ Trace2Skill 是 Ni et al. 独立论文。下文出现的 `config_dict["agent"]["instructions"]`、SQLite `ExperienceCacheModel`、5 步 `_single_rollout_summary/_group_advantage/_group_update/_batch_update` 等**代码片段无法从 arXiv 摘要核验**，可能来自相关但不同的 codebase（youtu-agent），**不能直接当作 Trace2Skill 的实现**。需查论文正文/代码仓库确认。|
| 差3 注入 | "直接注入**全部** experience，agent 自主判断" | ❌ 摘要明言部署阶段 yields **portable skills reused WITHOUT test-time retrieval**。"全部注入 experience" 恰恰是 **ReasoningBank-style retrieval memories** 的做法，而 Trace2Skill **明确 outperform 它**。故"全量注入"不是 Trace2Skill 的机制，而是它击败的基线。|
| 差4 更新 | "在线闭环，每步 ADD/UPDATE/DELETE/NONE" | ⚠️ 部分修正。摘要说 **in parallel**（批并行），支持 UPDATE（deepen）+ ADD（from drafts）；**DELETE / NONE / 逐步在线** 未出现在摘要中，属过度表述。核心点（他们改进池、我们冻结池）仍成立。|
| 差5 领域 | "web agent / information seeking，多工具" | ❌ 实际领域：**office workflows / math reasoning / vision QA**（含 WikiTableQuestions）。这些都是**多步 agentic 任务**，这点与下文"轨迹有丰富结构化信息"的推断方向一致，但具体领域写错了。|

**核验后仍成立的差异（去伪存真）**：

1. **信息贫瘠（差1）-- 核验成立，且是结构性问题**。摘要 "consolidates **broad execution trajectories**" + "compresses recurring failures and workarounds into SoPs" 直接佐证：他们的 skill 来自多步执行轨迹（含失败与绕行），我们来自贫瘠的 (prompt, code, score) 三元组。**关键新增**：因为我们的任务是**单次代码生成**，本身就没有多步轨迹可蒸馏--信息贫瘠不是实现疏漏，而是**任务结构决定的**上限。
2. **检索差异（差3）-- 核验成立，但机制要改写**。对比不是"他们全量注入 vs 我们 k=5 检索"，而是"他们用 **portable SoP**（自包含、无需检索）vs 我们用 **trigger-word k=5 检索**"。教训不是"全部注入"，而是"让 skill 自包含到不需要检索"。
3. **无质量改进环（差4）-- 核验部分成立**。他们有 UPDATE(deepen)+ADD(from drafts) 的池改进；我们冻结。核心点成立，DELETE/NONE/在线的细节存疑。
4. **任务领域（差5）-- 核验成立，领域已更正**。多步 agentic vs 单次 code-gen。
5. C-model 评判噪声（差6）-- 仍成立（我们用 LLM 打分，他们用可验证答案）。

### 🔴 Step 1 已证伪"注入机制是绑定约束"--核心结论

下文"行动计划#1"建议"改为**全部注入**模式"。但 **Step 1（2026-07-13）已经实现了全量压缩注入 + Meta-Skill**，结果在 MBPP_pro n=100 上 **null**（Baseline 0.685 vs Curated 0.684, p=0.819；详见 [[step1-validation-result-20260713]]）。

这意味着：**注入方式（全量 vs 检索 vs 数量）不是绑定约束**。结合核验后根因#1（信息贫瘠是任务结构决定的上限），绑定约束是 **skill 池质量本身**。

**对 P0 focused-injection 实验的判别性解读**（详见 [[skill-architecture-redesign]] / survey §4.1）：
- 4 组：Baseline / Full-50 / Random-3 / Focused-3（按题语义 top-3）
- 若 **Focused-3 > Full-50 > Baseline** -> GraSP"少量聚焦 > 全量"成立 -> 注入数量是约束 -> 修注入
- 若 **Focused-3 ≈ Full-50 ≈ Baseline（全 null）** -> 池质量是约束 -> 任何注入方案无效 -> 转向 SkillsBench target A（用验证过的好 skill 集测我们的**方法**）+ 修池质量（信息丰富化 / 反馈环）
- **基于 Step 1 + 根因#1 的预测**：focused-injection 大概率也 null，因为单次代码生成的 skill 池质量受任务结构封顶。

---

## 差异 1：Experience 的内容格式 — 从 trace 中提取的信息完全不同

### Trace2Skill 论文实现 (youtu-agent)

**Step 1: 单轨迹总结** — 提取的是 **agent 的工具使用、推理链、执行过程**。

```yaml
SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP 要求 LLM 从 trajectory 中提取:
  - Action Taken: agent 做了什么
  - Tool Called: 调用的工具名 (e.g., search_google_api, web_qa)
  - Tool Parameters: 传给工具的参数
  - Tool Results: 工具的返回数据（要求列出具体值, 不概括）
  - Agent's Reasoning: agent 的思考过程
  - Potential Missed Information: agent 忽略了哪些相关信息
```

**轨迹格式**: JSON-like list of steps, 包含 user inputs, tool calls, tool responses。
例如: `Step 1: search("2024 election results") → Step 2: web_qa("...") → Step 3: ...`

**对比信号**: 有 ground truth answer 的评判 (critique), reward 范围为 [0, 1]。

### 我们的实现 (lunwen)

**Step 1: 单轨迹总结** — 分析的是 **A 生成的 system prompt 和 B 生成的代码**。

```python
SINGLE_ROLLOUT_SUMMARY 要求 LLM 分析:
  - System Prompt (A模型输出): 这段 prompt 写的怎么样
  - Generated Code (B模型输出): 代码有什么 bug
  - Pass Rate: 分数
  - Code Quality Assessment / Prompt Effectiveness / General Principle
```

**轨迹格式**: 一段 prompt 文本 + 一段代码文本 + 一个分数。没有工具调用链，没有中间推理步骤。

**对比信号**: pass_rate（单元测试通过率）+ judge_score（另一 LLM 打分）。

### 🔴 根本差异

| 维度 | Trace2Skill | lunwen |
|------|-------------|--------|
| 被蒸馏的对象 | Agent 的**多步工具调用行为** | **单次生成的代码** + 单次写出的 prompt |
| 轨迹的粒度 | 每一步的工具选择、参数、结果、推理 | 最终的 prompt 文本 + 代码文本 |
| 可分析的信息 | 丰富的中间过程（哪个步骤做对了/错了） | 只有输入和最终输出 |
| 对比维度 | 不同 rollout 的工具选择路径差异 | 不同 prompt 在 pass_rate 上的差异 |

**对 skill 质量的影响**: Trace2Skill 从丰富的多步轨迹中提取关于 "agent 应该怎么思考和行动" 的经验；我们从贫瘠的 (prompt, code, score) 三元组中提取 "prompt 应该怎么写" 的经验。前者的信息含量远大于后者。

---

## 差异 2：Experience 的最终存储格式

### Trace2Skill

每条 experience 的格式是 **极简** 的:

```
Experience name: Brief description.
```

例如:
```
News recency: When users ask for recent news, prioritize recent sources and use time filters for time-sensitive queries.
```

注意: Trace2Skill 使用 SQLite 数据库 (`ExperienceCacheModel`) 存储，而非 JSON 文件。

### 我们的实现

每条 skill 的格式是 **详细但臃肿** 的:

```markdown
---
id: G217
version: 1
score_delta: 0.11
triggers: [...]
---

Explicitly Instruct Edge Case Handling Aligned with Tests

**When to apply**: ...
**What to do**: ...
**Example**: ...
**Why it works**: ...
```

~1000-1200 字/条。

### 🟡 这一差异的潜在影响

Trace2Skill 的短 experience 可能实际上是一种去噪音机制：
- 短 experience 更容易被 agent 正确解析和使用
- 长 experience（1000字+）可能包含太多细节导致 agent 在其中"迷失"
- 不过这个差异的正面/负面影响需要实验验证

---

## 差异 3：Experience/Skill 的注入方式

### Trace2Skill

Experience 被追加到 agent 的 `instructions`（相当于 system prompt）的**末尾**:

```python
experience_text = "\n\nWhen solving problems, you MUST first carefully read and understand "
experience_text += "the helpful instructions and experiences:\n"
experience_text += "\n".join([f"[{i}]. {e}" for i, e in experiences.items()])
config_dict["agent"]["instructions"] = current_instructions + experience_text
```

**关键特性**:
1. 所有 experience 一次性注入（不分条注入）
2. 格式是简单的编号列表 `[0]. content \n [1]. content`
3. 有引导语: "When solving problems, you MUST first carefully read and understand the helpful instructions..."
4. 不放任何 selection/filtering 机制，直接全部注入

### 我们的实现

Skill 通过 `context.py` 中的 ContextBuilder 注入，使用**触发词匹配**检索:

```python
# 每条 skill 有 triggers 列表，通过关键词匹配决定是否注入
# 每次检索 5 条 skill，固定 k=5
# 注入时带有大量元数据 (YAML frontmatter + 结构化正文)
```

**关键特性**:
1. 每次检索 5 条（不是全部注入）
2. 检索基于触发词关键词匹配（容易匹配到不相关的 skill）
3. 每条 skill 自带大量格式信息（YAML + 结构化段）

### 🔴 这一差异导致的致命问题

这是 **70% skill 有害的最直接原因**：在 Trace2Skill 中，所有 experience 直接注入，agent 自己判断哪些适用。在我们的系统中，触发词关键词匹配这种粗粒度的检索机制，会频繁地将 **不相关的 skill 注入到不适用的任务中**。

举例：
- 一条关于"字符串处理"的 skill 有 triggers: `["string manipulation", "text processing"]`
- 如果任务描述中有 "You are given a string..." 这个短语
- 这条 skill 就会被触发，即使它的建议（如"显式处理字符编码"）在当前任务中完全多余

这就是 **负迁移** 的工程原因：**检索精度不足导致不相关 skill 被注入不适用场景**。

---

## 差异 4：经验池的更新/维护流程

### Trace2Skill — 完整的在线闭环

```
训练循环:
  for step in range(num_steps):
    1. rollout (agent 用当前的 experiences 执行任务)
    2. ExperienceUpdater.run(rollouts)  ← 5 步细化管线:
       a. _single_rollout_summary     ← LLM 总结每条轨迹
       b. _group_advantage            ← 同题对比提取新 experience
       c. _group_update               ← 新 experience 与旧池的 ADD/UPDATE/DELETE/NONE
       d. _batch_update               ← 批量调和冲突
       e. assign new IDs + 更新 recorder
    3. ExperienceCache.save_experiences ← 存入 SQLite
    4. 下一轮 rollout 使用更新后的 experiences
```

**关键**: 
- 每步训练后都有 experience 更新
- 有 DELETE 操作可以移除有害 experience
- 有 NONE 操作可以拒绝冗余 experience
- **数据库缓存**可以复用历史结果

### 我们的实现 — 离线批量处理

```
一次性执行:
  python -m experience.extractor
    1. parse_trace (从文本日志解析)
    2. group_by_problem
    3. _extract_from_group (单题蒸馏)
    4. _cross_problem_merge (跨题合并)
    5. 写入 experience_pool.json
  → 然后手动 import 到 skills/
  → skill 一经创建就几乎不再更新
```

**关键差异**:
- 离线的，非迭代的
- 没有 DELETE/NONE 操作（只做 ADD）
- 没有在线的 experience 质量反馈环
- skill 池只增不减 → G2 Gap: 创建-验证分离

---

## 差异 5：任务领域

### Trace2Skill
- 专注于 **web agent / information seeking** 任务
- Agent 有多个工具 (search, web_qa, python_executor)
- 轨迹是真实的 agent-environment 交互

### 我们的实现
- 专注于 **代码生成** 任务
- B 模型单次生成，无工具
- "轨迹" 实际上只有一次生成行为

### 🟡 影响
Trace2Skill 的方法可能天然更适合 tool-use agent 领域，那里的轨迹有丰富的结构化信息（工具名、参数、返回值）。在代码生成中，trace 结构更加单一，蒸馏出有价值的经验更难。

但这不是根本原因——SGDR 和 SkillLens 等后续工作已经在代码/数学领域验证了 skill 方法的有效性。

---

## 差异 6：对比信号的质量

### Trace2Skill
- 使用 **ground truth answer** 作为评判基准（并非 LLM 打分）
- reward 是 binary 或 bounded [0,1]（由环境/单测客观给出）
- 对比信号清晰、无偏

### 我们的实现
- 对比信号来自 C 模型 (另一个 LLM) 的打分
- C 模型评判区分度不足（已知问题）
- 对比信号可能被 C 模型的 bias 污染

---

## 总结：为什么我们的 70% skill 有害？

### 🔴🔴🔴 致命原因（按影响排序）

1. **信息贫瘠**: 我们从 (prompt, code, score) 三元组提取的是"怎么写 prompt"的指导，但信息含量远低于 Trace2Skill 从多步工具调用轨迹中提取的"agent 应该如何推理和行动"的指导。

2. **检索机制导致负迁移**: 我们使用触发词匹配选择性注入 skill (k=5)，但匹配精度不足——不相关的 skill 频繁被注入不适用的任务。Trace2Skill 则直接注入**全部** experience（格式短小），由 agent 自主判断。我们的方式是 "强制使用 + 粗匹配"，他们的方式是 "建议参考 + 自主判断"。

3. **无质量反馈环**: skill 创建后从不被重新审视、修改或删除。有害 skill 永远留在池子里。Trace2Skill 有完整的 ADD/UPDATE/DELETE/NONE 四操作闭环。

4. **离线 vs 在线**: Trace2Skill 在训练循环中迭代优化 experience。我们是一次性离线提取后 skill 冻结。

### 🟡 次要原因

5. C 模型评判区分度不足 → 对比信号可能噪声大
6. skill 文本太长 (1000+字) vs experience 短句 — 虽然结构化格式本身合理，但长度可能在检索场景中造成信息过载

---

## 行动计划修正

基于以上分析，盲目做 P2 剪枝是在治标——去掉 70% 的坏 skill 能降低噪音，但不能从根本上提升 skill 质量。

**正确路径**：

### 1. 立刻做：修复检索机制（最大 ROI，零 API 成本）
- 改为**全部注入**模式（类似 Trace2Skill），不再用触发词做选择性检索
- 但 skill 文本太长 (1000字×402条=402K字)，不能全注入
- 方案：将每条 skill 压缩为 1-2 句的 **短 experience 格式**，然后全量注入 system prompt
- 或方案：增大 k 值（如 k=20），同时加入显式的 "选择性使用" 指令

### 2. 同时做：Meta-Skill 前置注入
- 在 system prompt 中加一段元规则，告诉 B 模型:
  - "这些 skill 是建议而非硬性规则"
  - "如果 skill 的触发条件与当前任务不匹配，忽略它"
  - "优先保证代码正确性，其次才是遵循 skill 建议"

### 3. 之后再考虑：信息丰富化
- 如果 Phase 2 BAgent 启用了 self-debug 和工具调用
- 那么可以像 Trace2Skill 那样从多步工具调用轨迹中蒸馏经验
- 这会比从单次生成中蒸馏更有效

### 4. 长期：建立质量反馈环
- 当 GRPO 重跑后，skill 评分实现真正的 ADD/UPDATE/DELETE 闭环

---

## 相关文件

- [[ib-skill-improvement-plan]] — IB 四阶段改进设计
- [[ib-pruning-evaluation]] — P0/P1 剪枝结果 (70% harmful)
- [[status-review-20260713]] — 项目诚实状态评估
- [[trace2skill-gap-analysis]] — 原有 7 gap 分析 (已过时，本文件提供更准确的诊断)
- [[phase2_issues]] — Phase 2 全部 issue 日志
