---
name: skill-architecture-redesign
description: 对照Trace2Skill重新设计的skill生命周期架构 — 从离线批量到在线闭环进化 (2026-07-13)
metadata:
  type: project
---

# Skill 架构重新设计 — 向 Trace2Skill 靠拢

## 一、当前架构 vs 目标架构

### 当前 (离线批量模式)

```
┌──────────────────────────────────────┐
│         离线，一次性                  │
│                                      │
│  GRPO trace  ──→ extractor.py       │
│  (文本日志)      ──→ templates.py    │
│                    ──→ cache.py      │
│                       │              │
│                  experience_pool.json│
│                       │              │
│              skills.py:import_from   │
│                       │              │
│                  402 个 G*.md         │
│                       │              │
│  BAgent.run()  ──→  skills.retrieve │
│    (触发词匹配, top_k=5)             │
│         │                            │
│    注入 5 条 skill                   │
│         │                            │
│    [END — 无反馈回到 skill 池]        │
└──────────────────────────────────────┘
```

**致命缺陷**:
- skill 创建后**冻结**，从不更新
- 检索精度依赖触发词关键词匹配 → 负迁移
- 无 DELETE/NONE 操作 → 坏 skill 永远留在池子里
- 无反馈环 → 不知道哪些 skill 有用、哪些有害

### 目标 (在线闭环模式，对标 Trace2Skill)

```
┌──────────────────────────────────────────────┐
│              在线迭代闭环                      │
│                                               │
│  ┌──────────┐    ┌──────────┐    ┌─────────┐ │
│  │ A (GRPO) │───→│ B (BAgent)│───→│C (Judge)│ │
│  │ 生成prompt│    │ 生成代码  │    │ 打分    │ │
│  └──────────┘    └──────────┘    └─────────┘ │
│       ↑               │              │        │
│       │          skills_used       scores    │
│       │               │              │        │
│       │          ┌────▼──────────────▼──┐    │
│       │          │   ExperienceUpdater  │    │
│       │          │                      │    │
│       │          │ 1. 单轨迹总结         │    │
│       │          │ 2. 组内对比提取       │    │
│       │          │ 3. ADD/UPDATE/DELETE  │    │
│       │          │ 4. 批量合并调和       │    │
│       │          └──────────┬───────────┘   │
│       │                     │                │
│       │           experience_pool.json       │
│       │           (skill 质量随训练提升)      │
│       │                     │                │
│       └─────────────────────┘                │
│            A 学习选择更好的 skill             │
└──────────────────────────────────────────────┘
```

Trace2Skill 的核心优势不在于蒸馏算法本身，而在于**它是一个在训练循环中迭代运行的闭环系统**。

---

## 二、具体改造方案

### 改造 1：检索机制 — 从"触发词匹配"到"全量注入 + 自主判断"

**现状问题**:
- `skills.retrieve()` 用关键词匹配从 402 条中选 5 条
- 匹配精度不足 → 不相关 skill 被注入不适用的场景 → 负迁移

**Trace2Skill 的做法**:
- 全部 experience 一次性注入 system prompt
- 格式: `[ID]. Experience name: Brief description.`
- 有引导语: "When solving problems, you MUST first carefully read and understand the helpful instructions and experiences"

**改造方案**:

1. **双模式 skill 表示**: 每条 skill 有两个版本
   - `compact`: 1-2 句短描述（用于全量注入 context）
   - `full`: 结构化详细版本（用于 self-debug 参考或按需查阅）

2. **注入策略**:
   - 将全部 skill（或经 score_delta 过滤的 active skills）的 compact 版本注入 system prompt
   - 注入时附带 **Meta-Skill 引导语**（见改造 2）
   - 如果 skill 数量太多（如 >50），按 score_delta 排序取 top-N

3. **compact 版本的生成**:
   - 对每条现有 skill，提取其 `**What to do**` 部分作为 compact 版本
   - 如果 `What to do` 仍然太长，调用一次 LLM 压缩为 1-2 句

**修改文件**:
- `agent/b_agent/skills.py`: 
  - `Skill` 加 `compact` 字段
  - `format_for_context()` 改为 `format_all_active()` — 全量 compact 注入
  - `retrieve()` 保留但改名为 `retrieve_full()` — 仅用于按需查阅完整 skill
- `agent/b_agent/core.py`:
  - `run()` 中注入全部 active skill 而非检索 5 条
  - `generate_code_with_skills()` 同样修改

### 改造 2：Meta-Skill 前置注入 — 教会 B 如何用 skill

**新增内容**（追加到 system prompt 中 skills 部分之前）:

```
## How to Use These Guidelines

The following guidelines are distilled from past coding experiences. They are SUGGESTIONS, not hard rules.

When applying a guideline:
1. First, check if it's actually relevant to your current task. If your task doesn't involve the scenario described, you MUST IGNORE that guideline.
2. A guideline should HELP you write correct code — never follow a guideline if it would make your code incorrect.
3. Prioritize writing correct, runnable code that passes the tests. Guidelines are secondary.

Not every guideline applies to every task. Skip the ones that don't fit.
```

**为什么这样做**:
- SkillLens 实验证明 meta-skill 可以减少非平凡的负迁移
- 相当于给 B 模型一个 "选择性忽略" 的许可
- 零 API 成本，直接写在 `_default_rules()` 或 ContextBuilder 中

**修改文件**:
- `agent/b_agent/context.py`: `_default_rules()` 中加入 meta-skill 引导语
- 或在 `core.py:run()` 中 skill_text 前面加引导语

### 改造 3：Skill 的质量反馈机制 (ExperienceUpdater)

这是最关键的改造——将离线一次性蒸馏变成**在线闭环**。

**当前**: `extractor.py` 离线解析 GRPO trace 日志 → 生成 skill → 结束

**目标**: 每次 BAgent 执行后，根据结果更新对应 skill 的 score_delta

**具体做法**:

在 `custom_reward_with_unittest.py`（或 `b_agent_interface.py`）中，每次 generation 完成后:

```python
def update_skill_scores(trace: dict, skill_lib: SkillLibrary):
    """
    根据本次 generation 的结果更新所用 skill 的评分。
    
    对每条在 skills_used 中的 skill:
      - 如果 pass_rate 高 → 正 delta
      - 如果 pass_rate 低 → 负 delta
      - 用 EMA 平滑更新 score_delta
    """
    skills_used = trace.get("skills_used", [])
    pass_rate = trace.get("pass_rate", 0)
    baseline_pass_rate = ...  # 需要同期 baseline 对比
    
    for skill_id in skills_used:
        # 计算这条 skill 的边际贡献
        # 简化版本: 用本次的 pass_rate 与近期不使用该 skill 的 baseline 对比
        # 完整版本: 需要 per-skill counterfactual（比当前数据更强时）
        delta = ...
        skill_lib.update_score(skill_id, delta)
```

**简化版（立即可做）**:
- 每次批次运行后，对使用了 skill 的 task 和未使用 skill 的 task 做组间对比
- 统计每条 skill 被使用时的平均 pass_rate 与全局 baseline 的差
- 用 EMA 更新

**修改文件**:
- `agent/b_agent/skills.py`: `update_score()` 已存在，不需要改
- `rl/train/verl1/custom_reward_with_unittest.py`: 加入 skill 评分逻辑
- 或新建 `experience/updater.py`（对标 youtu-agent 的 `experience_updater.py`）

### 改造 4：Skill 池的 ADD/UPDATE/DELETE 操作闭环

**借鉴 Trace2Skill 的四操作模型**:

```python
class SkillOperation:
    ADD:    新 skill 加入池子
    UPDATE: 更新已有 skill（合并、深化）
    DELETE: 删除有害/过时 skill
    NONE:   不做任何操作（新 skill 已充分覆盖）
```

**实现**: 在 `experience/updater.py`（新增）中:

```python
async def update_skill_pool(
    new_candidates: List[str],     # 新蒸馏出的候选 skill
    current_pool: Dict[str, Skill], # 当前 skill 池
    llm: SimplifiedAsyncOpenAI,     # 用于合并决策的 LLM
) -> Dict[str, Skill]:
    """
    对照 Trace2Skill 的 group_update + batch_update:
    
    1. 对每个候选 skill，与池中每条已有 skill 对比
    2. 决定 ADD / UPDATE / DELETE / NONE
    3. 批量调和冲突（多条 UPDATE 同一 ID 时合并）
    4. 返回更新后的 skill 池
    """
```

**触发时机**:
- 每 N 步 GRPO 训练后
- 或每次 ablation 实验完成后
- 或手动触发 `python -m experience.updater`

**修改文件**:
- 新建 `experience/updater.py`
- `agent/b_agent/skills.py`: 加入 `add_or_update()` 方法

### 改造 5：数据源修复 — 丰富 trace 信息

**现状**: trace 只有 (prompt, code, score) 三元组
**目标**: 如果 BAgent 启用了 self-debug 和工具调用，记录多步执行轨迹

**做法**:
- `BAgent.run()` 和 `generate_code_with_self_debug()` 已经记录了 steps/debug_trace
- 将这些中间步骤写入 trace log，供后续蒸馏使用
- 类似 Trace2Skill 从工具调用链中提取信息

**修改文件**:
- `rl/train/verl1/custom_reward_with_unittest.py`: trace 中增加 steps/debug_trace 字段
- `agent/b_agent/core.py`: `AgentResult` 的序列化（已有 steps 字段）

---

## 三、改造优先级和顺序

```
Step 1. 检索机制修复 (改造1 + 改造2)      ← 🔴 立即，零API成本，最大ROI
Step 2. Skill 评分闭环 (改造3 简化版)      ← 🔴 需要至少一次消融实验跑通
Step 3. ADD/UPDATE/DELETE 闭环 (改造4)    ← 🟡 需要 LLM 调用，可在 Step2 后进行
Step 4. Trace 信息丰富化 (改造5)           ← 🟢 长期，需要 GRPO 重跑
```

**Step 1 (检索修复) 的具体改动**:

文件修改清单:
1. `agent/b_agent/skills.py`:
   - `Skill` 新增 `compact: str` 字段
   - `format_for_context()` → 改为 `format_all_active_for_context()` 全量注入
   - `retrieve()` 保留但语义改为"获取完整 skill 文本"
   - 新增 `generate_compact()` 方法: 从 strategy 提取 What to do 部分

2. `agent/b_agent/context.py`:
   - `_default_rules()` 加入 Meta-Skill 引导语

3. `agent/b_agent/core.py`:
   - `run()` 中注入全部 active skill 的 compact 版本
   - `generate_code_with_skills()` 中同样修改
   - `generate_code_with_self_debug()` 中同样修改

## 四、预期效果

| 改造 | 预期效果 | 验证方式 |
|------|---------|---------|
| 全量注入 + Meta-Skill | Curated 组效果提升（减少负迁移）| BigCodeBench n=57 对照 |
| Skill 评分闭环 | skill score_delta 开始分化（有的正有的负） | 查看 score_delta 分布 |
| ADD/UPDATE/DELETE | 池子自动去坏留好 | 池子大小和质量的变化 |
| Trace 丰富化 | 蒸馏出的 skill 质量提升 | 新 skill 的剪枝评估 delta |
