# Training-Free GRPO 与 Prompt Optimization 融合优化方案

> 基于对 [Training-Free GRPO](https://arxiv.org/abs/2510.08191) (youtu-agent 实现) 与当前 lunwen 项目 (A→B→C 三模型 GRPO 提示词优化流水线) 的深入分析，讨论融合优化方案。

## 1. 两套系统的架构对比

### 1.1 当前 lunwen 架构

```
训练数据 (OpenCodeInstruct)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  A 模型 (Qwen3-4B, 正在训练)                          │
│  输入: 问题 + 参考解 + few-shot 示例                    │
│  输出: 系统提示词                                       │
│  训练方式: GRPO (参数更新, 8 GPU, verl 框架)            │
└──────────────┬──────────────────────────────────────┘
               │ 系统提示词
               ▼
┌─────────────────────────────────────────────────────┐
│  B 模型 (DeepSeek-V3.2 / Qwen2.5-Coder-7B, 冻结)     │
│  输入: A 的系统提示词 + 用户查询                        │
│  输出: Python 代码                                     │
│  特点: 单次生成，无工具/无迭代                          │
└──────────────┬──────────────────────────────────────┘
               │ 代码
               ▼
┌─────────────────────────────────────────────────────┐
│  C 模型 (DeepSeek-V3.2, 冻结) + 单元测试执行器          │
│  LLM 评判: 提示词流畅度 + 代码质量 + 功能一致性          │
│  单元测试: 实际执行代码，计算通过率                       │
│  反幻觉核实: 4 种策略验证 C 模型的违规声明               │
│  奖励 = 70% 单元测试 + 30% LLM 评判                     │
└──────────────┬──────────────────────────────────────┘
               │ 奖励信号
               ▼
          GRPO 策略更新 → A 模型参数优化
```

### 1.2 Training-Free GRPO 架构 (youtu-agent)

```
训练数据 (DAPO-Math-17k)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Single Agent (DeepSeek-V3.2 + Python Executor Tool) │
│  输入: 数学问题 + 当前经验指导 (G0-Gn)                  │
│  输出: 逐步推理 + 最终答案 (最多 50 轮 tool-calling)    │
│  特点: Agent 可自行调用 Python 验证计算                  │
└──────────────┬──────────────────────────────────────┘
               │ 推理轨迹 + 答案
               ▼
┌─────────────────────────────────────────────────────┐
│  评判 (math_verify / LLM Judge)                       │
│  奖励: 0 或 1 (二元，答案正确/错误)                      │
└──────────────┬──────────────────────────────────────┘
               │ 奖励 + 轨迹
               ▼
┌─────────────────────────────────────────────────────┐
│  Experience Updater (LLM 驱动的经验蒸馏)               │
│  Step 1: 逐轨迹总结 (Single Rollout Summary)           │
│  Step 2: 组内对比 (Group Advantage) — 对比好/坏轨迹      │
│  Step 3: 逐条更新 (Group Update) — ADD/UPDATE/DELETE   │
│  Step 4: 批量合并 (Batch Update) — 去重/调和冲突         │
│  Step 5: ID 分配 — G0, G1, G2, ..., G37               │
│  输出: 文本经验指南注入 agent 的 instructions            │
└──────────────┬──────────────────────────────────────┘
               │ 经验注入
               ▼
    下一个 epoch: agent 带着更新的 G0-Gn 重新 rollout
    (模型参数完全不更新 → "Training-Free")
```

### 1.3 核心差异总结

| 维度 | lunwen | Training-Free GRPO | 融合方向 |
|------|--------|-------------------|----------|
| **优化目标** | A 模型参数 | Agent 的文本指令 | **叠加**：参数 + 指令双优化 |
| **Agent 能力** | 无 (B 是纯代码生成器) | 有 (tool-calling, 多轮推理) | **给 B 模型加 Agent 能力** |
| **奖励粒度** | 连续值 (-15~10) | 二元 (0/1) | lunwen 更精细，保持 |
| **经验形态** | 无 | G0-G37 文本指南 | **新增经验蒸馏层** |
| **训练成本** | GPU 重 (8 GPU GRPO) | API 费用 (LLM 蒸馏) | **Training-Free 预热 → GRPO 精调** |
| **数据隔离** | 训练/测试未分离 | 严格分离 (DAPO→AIME) | **必须引入数据隔离** |
| **模型数量** | 3 个专用模型 | 1 个通用 Agent | 保持多模型，给 B 加工具 |

---

## 2. 融合优化方案 (三个层次)

### 方案一 (推荐优先实施): A 模型侧 — 经验蒸馏 + ICL 增强

**核心思路**: 在 A 模型生成提示词之前，注入从历史成功/失败中蒸馏出的"提示词工程指南"，类似 Training-Free GRPO 的 G0-G37。

```
当前:  问题 + 参考解 + 固定 ICL → A 模型 → 系统提示词

优化:  问题 + 参考解 + 固定 ICL + [E0, E1, ..., En] → A 模型 → 系统提示词
                                          ↑
                                    经验指南 (Experience Guidelines)
                                         │
                              从 B 模型的成功/失败中蒸馏
```

**具体实施步骤**:

```
Step 1 — 基线 Rollout 收集
  对每个训练样本，用当前 A 模型生成 N=5 个系统提示词
  → B 模型生成代码
  → 记录: 哪些提示词导致代码通过所有测试，哪些失败
  → 按 reward 分组: 高分组 (reward > 阈值) vs 低分组

Step 2 — 经验蒸馏 (复用 Training-Free GRPO 的 ExperienceUpdater 逻辑)
  对每个问题:
  - 输入高分/低分提示词对 + 对应的单元测试结果
  - LLM 分析: "为什么提示词 A 比提示词 B 更有效?"
  - 提取可泛化的经验 → 如:
    [E0] 明确指定输入输出类型，避免模糊描述
    [E1] 对边界条件提供具体示例
    [E2] 要求代码包含类型注解和 docstring
    ...

Step 3 — 经验注入
  将蒸馏出的 E0-En 追加到 A 模型的 system prompt 或 ICL 示例之前
  格式参考 youtu-agent:
  "When generating system prompts, you MUST follow these guidelines:
   [E0]. Always specify expected input/output types explicitly.
   [E1]. Include edge case examples in the prompt.
   ..."

Step 4 — 迭代
  用增强后的 A 模型重新生成提示词 → B 模型生成代码 → 评估
  → 新一轮经验提取 → 更新 E0-En
  → 类似 Training-Free GRPO 的 epoch 循环
```

**需要新增的模块**:

| 模块 | 说明 | 参考 |
|------|------|------|
| `experience/extractor.py` | 从 rollout 数据中蒸馏提示词工程经验 | youtu-agent `experience_updater.py` |
| `experience/templates.py` | 经验提取/更新的 prompt 模板 | youtu-agent `prompts/practice/experience.yaml` |
| `experience/cache.py` | 经验的 DB 存储和版本管理 | youtu-agent `experience_cache.py` |
| `experience/injector.py` | 将经验注入 A 模型输入 | youtu-agent `training_free_grpo_processor.py` |

**关键注意点**:

1. **上下文窗口**: A 模型只有 4096 token 上限，经验数量需要控制 (~10-15 条)，需要根据效果动态裁剪低效经验
2. **数据隔离**: 经验提取必须在**独立数据集**上进行（如 APPS_Plus），不能在测试集（HumanEval/MBPP）上提取
3. **经验质量**: 需要引入类似 Training-Free GRPO 的 ADD/UPDATE/DELETE 机制，去除冗余/矛盾经验
4. **蒸馏 LLM 选择**: 经验蒸馏需要强推理能力，建议用 DeepSeek-V3.2 而非本地小模型

---

### 方案二 (中高价值): B 模型侧 — 从代码生成器升级为 Coding Agent

**核心思路**: 当前 B 模型是"一次性代码生成器"——没有工具、没有迭代、没有自我修正。Training-Free GRPO 的核心优势在于 Agent 可以**调用工具 (Python Executor) 来验证自己的推理**。如果给 B 模型加上代码执行工具，它就可以：

- 生成代码 → 执行 → 看到错误 → 修正 → 再执行 → 直到通过

```
当前:  A 提示词 → B (一次生成) → 代码 → 测试 (外部)

优化:  A 提示词 → B Agent (最多 K 轮) → 代码 → 测试 (最终)
                    │
                    ├─ [Tool: python_executor]  执行代码看结果
                    ├─ [Tool: search]           搜索 API 文档
                    └─ [Tool: file_reader]      读取参考代码
```

**具体改动**:

```python
# B 模型从简单生成器升级为 Agent
# 参考 youtu-agent 的 SimpleAgent 模式

class CodingAgent:
    """带工具调用的代码生成 Agent"""
    
    def __init__(self, model, tools):
        self.model = model
        self.tools = [
            ExecuteCodeTool(),   # 执行 Python 代码，返回 stdout/stderr
            SearchAPITool(),     # 搜索 Python 标准库文档
            ReadFileTool(),      # 读取参考代码文件
        ]
        self.max_turns = 10     # 最多 10 轮 self-debug
    
    def generate(self, system_prompt: str, problem: str) -> str:
        """生成代码，支持多轮迭代修正"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
        ]
        
        for turn in range(self.max_turns):
            response = self.model.chat(messages, tools=self.tools)
            
            if response.has_tool_calls():
                # 执行工具调用，将结果加入上下文
                for call in response.tool_calls:
                    result = self._execute_tool(call)
                    messages.append({"role": "tool", "content": result})
            else:
                # 最终代码输出
                return response.extract_code()
        
        return response.extract_code()
```

**对奖励函数的影响**:

当前奖励 = 70% 单元测试 + 30% LLM 评判。Agent 化后还可以增加：

- **自我修正奖励**: B Agent 通过自我修正修复了错误 → 加分
- **效率奖励**: 用更少轮次解决 → 加分
- **工具使用质量**: 合理使用工具而非盲目重试 → LLM 评判中体现

**关键注意点**:

1. **超时控制**: Agent 多轮迭代会增加总耗时，需要严格的轮次和总时间限制 (参考 youtu-agent max_turns=50)
2. **幻觉放大**: Agent 迭代可能产生更多错误代码，需要 C 模型的违规检测覆盖 Agent 轨迹
3. **成本增加**: B 模型的 API 调用次数从 1 次变为最多 K 次
4. **A 模型提示词策略变化**: A 模型需要学会生成"指导 Agent 多轮迭代"的提示词，而非"一次生成完美代码"的提示词

---

### 方案三 (长期): 完整 Training-Free 预热 + GRPO 精调流水线

**核心思路**: 将 Training-Free GRPO 作为 GRPO 训练的前置阶段，用低成本的文本经验蒸馏为昂贵的参数训练提供更好的初始化。

```
Phase 1: Training-Free 经验积累 (纯 API 费用, 不更新参数)
  ┌──────────────────────────────────────────┐
  │ 训练数据 (DAPO-Math-17k 或 OpenCodeInstruct 子集) │
  │   ↓                                       │
  │ Baseline Rollout (A→B→C)                  │
  │   ↓                                       │
  │ Experience Updater (LLM 蒸馏)              │
  │   ↓                                       │
  │ 输出: E0-En 提示词工程指南                   │
  │       F0-Fn few-shot 示例 (最佳提示词)       │
  └──────────────────────────────────────────┘
                    │
                    ▼
Phase 2: 增强的 GRPO 训练 (GPU 训练, 更新 A 模型参数)
  ┌──────────────────────────────────────────┐
  │ A 模型输入: 问题 + 参考解 + E0-En + F0-Fn   │
  │   ↓                                       │
  │ GRPO 训练 (verl, 8 GPU)                   │
  │   ↓                                       │
  │ 输出: 精调后的 A 模型权重                    │
  └──────────────────────────────────────────┘
                    │
                    ▼
Phase 3: 评估 + 持续经验更新
  ┌──────────────────────────────────────────┐
  │ 在测试集上评估 (HumanEval+, MBPP+, LCB)     │
  │   ↓                                       │
  │ 收集新一轮 rollout 数据                     │
  │   ↓                                       │
  │ 更新经验池 E0-En                           │
  │   ↓                                       │
  │ (可选) 继续 GRPO 训练                       │
  └──────────────────────────────────────────┘
```

**关键注意点**:

1. **数据严格隔离** (Training-Free GRPO 论文的核心教训):
   - Phase 1 经验提取: 用 APPS_Plus 或 OpenCodeInstruct 子集 A
   - Phase 2 GRPO 训练: 用 OpenCodeInstruct 子集 B
   - Phase 3 评估: 用 HumanEval+/MBPP+/LCB (完全不同分布)
   - **绝对不能**在测试集上提取经验或训练参数

2. **经验数量控制**:
   - 初始: ~10 条核心经验 + ~5 个高质量 few-shot 示例
   - 随迭代逐步增加，但上限受 A 模型上下文窗口限制 (4096 tokens)
   - 定期 (每个 epoch) 用 UPDATE/DELETE 机制清理冗余经验

3. **奖励信号对齐**:
   - Phase 1 的经验蒸馏用"单元测试通过率"作为分组信号
   - 这天然比 Training-Free GRPO 的二元奖励更精细

---

## 3. 核心技术问题讨论

### 3.1 Agent 部分是否需要补充？

**当前状态**: B 模型是无 Agent 能力的纯生成器，A 模型生成文本提示词。

**建议**: 分阶段引入 Agent 能力：

| 阶段 | 改动 | 风险 | 收益 |
|------|------|------|------|
| **短期** | 不改 B 模型，仅在 A 模型侧加经验注入 | 低 | 中 (提示词质量提升) |
| **中期** | B 模型加 python_executor tool (self-debug) | 中 | 高 (代码质量提升) |
| **长期** | B 模型升级为完整 Coding Agent (多工具) | 高 | 最高 (根本性提升) |

短期优先推方案一，中期试点方案二中仅加 python_executor 一项工具。

### 3.2 如何控制上下文？

| 组件 | 当前上下文 | 增加经验后 | 应对策略 |
|------|-----------|-----------|----------|
| A 模型 | 4096 tokens | 经验 ~500 tokens | 截断策略: 优先保留高分经验 |
| B 模型 | API 模型，上下文充足 | Agent 多轮对话 | 滑动窗口 + 工具输出摘要 |
| C 模型 | ~2000 tokens (评判模板) | 不需要改 | — |
| 经验蒸馏 LLM | 取决于轨迹长度 | 需要完整轨迹 | 先总结再对比 (像 Training-Free GRPO) |

**经验优先级管理**:
```
每个经验维护元数据:
{
  "id": "E0",
  "content": "...",
  "score_delta": +0.12,    # 加入后 reward 提升幅度
  "usage_count": 45,       # 被使用的次数
  "last_used_epoch": 3,    # 最后使用轮次
  "status": "active"       # active / deprecated / replaced
}

优先级排序: score_delta × log(usage_count) → 取 top-N
```

### 3.3 是否需要 Multi-Agent？

**当前 lunwen 已经是"类多智能体"架构** (A→B→C 三个独立模型协作)，但它们是固定的串行流水线。

**Training-Free GRPO 本身是 Single Agent**。

**融合后是否需要 Multi-Agent**:

| 场景 | 建议 | 理由 |
|------|------|------|
| 经验蒸馏阶段 | 不需要 | 沿用现有 A→B→C 流水线即可 |
| B 模型 Agent 化后 | 不需要 | B 作为 single agent + tools 足够 |
| 长期 (终极形态) | 可考虑 | A 和 B 都变成 Agent，通过 orchestrator 协调，但复杂度太高 |

**结论**: 当前阶段不建议引入 Multi-Agent。保持 A→B→C 流水线，仅在 B 模型侧加工具能力即可。Multi-Agent 是过度设计。

### 3.4 经验蒸馏的 LLM 选择

| 任务 | 推荐模型 | 理由 |
|------|---------|------|
| 提示词经验蒸馏 | DeepSeek-V3.2 (API) | 需要强推理能力，本地 4B 模型不够 |
| 代码经验蒸馏 | DeepSeek-V3.2 或 Qwen3-480B | 代码相关分析需要强代码理解 |
| 经验合并 (UPDATE/DELETE) | DeepSeek-V3.2 | 去重/调和冲突需要全局视角 |

**成本估算** (以 100 个训练样本为例):
- 单轨迹总结: 100 × ~500 tokens output = 50K tokens
- 组内对比: 100 × ~1000 tokens = 100K tokens
- 批量合并: ~2000 tokens
- 总计: ~150K output tokens per epoch → 费用可控

---

### 3.5 A 模型能力边界与创新点权衡

> **核心问题**: Qwen3-4B 参数量仅 4B，上下文窗口仅 4096 tokens。能否胜任"理解经验指南 + 阅读问题 + 参考标准解 + 生成高质量系统提示词"的复合任务？如果更换更大模型，是否违背项目"训练微小模型降低成本"的创新初衷？

#### 3.5.1 Qwen3-4B 的能力评估

**当前任务分析**:

```
A 模型输入构成:
┌────────────────────────────────────────┐
│ System: "你是一个系统提示词生成专家..."    │  ~50 tokens
│ ICL 示例: 3-5 个 few-shot 样例           │ ~600 tokens
│ 问题 + 参考解:                            │ ~400-800 tokens
│ 经验指南 (新增): [E0, E1, ..., En]        │ ~300-500 tokens
│ 总计:                                     │ ~1350-1950 tokens
├────────────────────────────────────────┤
│ 输出: 系统提示词                          │ ~200-500 tokens
└────────────────────────────────────────┘
```

在 4096 token 窗口下，输入占用 ~2000 tokens，输出 ~500 tokens，**仍有约 40% 余量**。从上下文容量角度看是可行的。

**参数规模与任务难度的匹配**:

| 能力维度 | 4B 模型能力 | 任务需求 | 评估 |
|----------|:----------:|:--------:|:----:|
| 指令遵循 | ★★★ 良好 | 按格式输出系统提示词 | ✅ 足够 |
| 代码理解 | ★★ 基础 | 理解参考解的逻辑结构 | ⚠️ 边缘 |
| 模式归纳 | ★★ 基础 | 从经验中提取通用规律 | ⚠️ 边缘 |
| 创意生成 | ★★ 基础 | 生成多样化的提示词变体 | ⚠️ 边缘 |
| 多步推理 | ★ 较弱 | 分析为什么某提示词有效 | ❌ 不足 |

**关键发现**: 4B 模型的**弱推理能力**恰恰是 Training-Free 经验注入最有价值的场景。经验指南 (E0-En) 可以**弥补小模型的推理短板**——就像给初级程序员一份最佳实践 checklist，他们虽然不如 senior 工程师，但照着 checklist 做也能达到 80 分的质量。

#### 3.5.2 创新点的正确理解

项目创新点不是"4B 模型有多强"，而是：

```
创新点 = 小模型 (低成本) + 经验蒸馏 (知识补偿) + GRPO 精调 (参数优化)
         ↑                    ↑                        ↑
    降低推理成本          让弱模型变强             让强经验固化
```

这恰恰与 Training-Free GRPO 的动机一致：
- youtu-agent: 不更新参数，通过文本经验让固定模型变强（+3.3~5.4%）
- lunwen: 用小参数模型 + 文本经验 + 参数精调，以小博大

**如果换成大模型反而偏离创新点**——大模型直接生成好提示词不需要这套 pipeline。

#### 3.5.3 渐进式模型升级策略

如果验证后发现 4B 确实不够，推荐的升级路径：

| 阶段 | A 模型 | 参数量 | 上下文 | 适用场景 |
|------|--------|:-----:|:------:|----------|
| **当前** | Qwen3-4B-Instruct | 4B | 4096 | 方案验证、pipeline 调试 |
| **备选 1** | Qwen3-8B-Instruct | 8B | 32768 | 需要更长上下文或更强推理 |
| **备选 2** | Qwen2.5-Coder-7B | 7B | 32768 | 代码领域专门优化 |
| **长期** | Qwen3-4B + 经验蒸馏 | 4B | 4096 | 核心创新路线 |

**建议**: 先不改 A 模型。在 4B 上跑通整个 pipeline，用实验结果说话：
- 如果 4B + 经验 > 4B 无经验 → 证明方法有效，创新点成立
- 如果 4B + 经验 < 目标值但提升显著 → 再考虑升级到 8B
- 如果 4B 完全不够 (提示词质量极差) → 才需要考虑换模型

#### 3.5.4 上下文窗口的实际约束

4096 token 的约束下，建议的创新做法：

```
经验注入预算分配 (共 ~500 tokens):
┌──────────────────────────────────────┐
│ 核心经验 (Top 5, 每条 ~60 tokens)     │  300 tokens  ← 严格控制
│ 动态经验 (Top 5-10, 每条 ~40 tokens)  │  200 tokens  ← 按需轮换
│ 保留余量                              │  ~200 tokens ← 应对长问题
└──────────────────────────────────────┘

超出预算时:
  1. 按 score_delta × log(usage_count) 排序
  2. 裁剪末尾 20%
  3. 对新经验做 A/B test，效果优于旧经验才替换
```

**小模型 + 短上下文的优势**: 推理快 (4B 模型生成 500 token 仅需 ~2-3 秒)，有利于 GRPO 训练中每个 sample 做多次 rollout 的速度。

---

### 3.6 B 模型版本更新策略

> **核心问题**: 当前 B 模型是 Qwen2.5-Coder-7B (2024 Q4)，已有一年多历史。是否需要更新到更新的代码模型？还是先验证算法再换？

#### 3.6.1 当前 B 模型现状

| 属性 | Qwen2.5-Coder-7B (当前本地) | DeepSeek-V3.2 (当前 API) |
|------|---------------------------|-------------------------|
| 发布 | 2024 年 11 月 | 2025 年 |
| HumanEval pass@1 | ~83% | ~95%+ |
| MBPP pass@1 | ~76% | ~90%+ |
| 上下文窗口 | 32768 | 128K+ |
| 速度 | 本地，低成本 | API，按量付费 |
| 角色 | 训练时大规模使用的默认 B 模型 | 高质量对比/最终评估用 |

#### 3.6.2 Qwen2.5-Coder-7B 的适用性

**对于算法验证阶段，Qwen2.5-Coder-7B 完全够用**:

1. **差异敏感性**: B 模型越好，A 模型提示词的提升空间越小 (天花板效应)。Qwen2.5-Coder-7B 的 pass@1 只有 ~83%，有很大的提升空间让 A 模型的提示词发挥作用。

2. **对比鲜明**: 一个好 B 模型 (如 DeepSeek-V3.2) 可能无论提示词好坏都输出高质量代码，导致**无法区分提示词质量**。较弱的 B 模型更能反映提示词的差异。

3. **成本可控**: 本地 7B 模型推理几乎免费，适合大规模 GRPO 训练 (每步需要 N×M 次 B 模型调用)。

4. **已有接口抽象**: `b_interface.py` 已经封装了本地/API 两种后端，切换 B 模型只需改配置。

#### 3.6.3 推荐的 B 模型升级策略

```
验证阶段 (当前):
  B 主模型: Qwen2.5-Coder-7B (本地)
  B 对照: DeepSeek-V3.2 (API)
  目的: 快速迭代，验证算法

优化阶段 (Phase 2 完成后):
  B 主模型: Qwen3-Coder-7B 或 DeepSeek-Coder-V2 (本地)
  目的: 验证算法在新模型上的迁移效果

评估阶段:
  B 模型池: 多个不同版本的模型
  目的: 证明 A 模型生成的提示词具有跨模型泛化能力
```

**关键洞察**: 一个真正好的系统提示词应该**对不同 B 模型都有提升效果**。如果 A 模型的提示词只在 Qwen2.5-Coder-7B 上有效，在 DeepSeek-V3.2 上无效，说明 A 模型学到的不是通用提示词工程技巧，而是针对特定模型的"作弊"。因此在 Qwen2.5-Coder-7B 上验证通过后，必须用 DeepSeek-V3.2 做跨模型验证。

#### 3.6.4 跨模型泛化测试设计

```python
# 评估 A 模型生成的提示词的跨模型泛化性
def evaluate_cross_model_generalization(a_model_prompt: str, test_problems: list):
    b_models = {
        "Qwen2.5-Coder-7B": BModelLocal("Qwen2.5-Coder-7B"),
        "DeepSeek-V3.2": BModelAPI("DeepSeek-V3.2"),
        "Qwen3-Coder-7B": BModelLocal("Qwen3-Coder-7B"),  # 后续加入
    }
    
    results = {}
    for name, b_model in b_models.items():
        pass_rate = evaluate_on_test_set(a_model_prompt, b_model, test_problems)
        results[name] = pass_rate
    
    # 如果所有模型都有提升 → 经验具有泛化性
    # 如果只有特定模型有提升 → 经验过拟合，需要重新蒸馏
    return results
```

---

### 3.7 数据集难度与经验效果的触发条件

> **核心问题**: HumanEval (164 题) 和 MBPP (~500 题) 可能过于简单，不足以触发有意义的经验提取效果。Training-Free GRPO 使用的是 AIME 竞赛数学题 (高难度、低通过率)，简单评测集上经验蒸馏可能效果不明显。

#### 3.7.1 经验蒸馏的"信号条件"

Training-Free GRPO 的经验提取机制依赖于一个关键前提：

```
经验提取的信号条件:
┌─────────────────────────────────────────────────────────┐
│  对同一个问题，需要有 "对比度"                              │
│                                                          │
│  高对比度: reward_high ≈ 1.0, reward_low ≈ 0.0          │
│  → LLM 可以清晰分析 "什么导致了差异"                        │
│                                                          │
│  低对比度: 所有 rollout 都在 0.8-1.0 之间                 │
│  → LLM 无法找到有意义的区分模式                            │
│                                                          │
│  零对比度: 所有 rollout 都是 0 或都是 1                    │
│  → 经验蒸馏无信号，被跳过 (Training-Free GRPO 的默认行为)   │
└─────────────────────────────────────────────────────────┘
```

**这就是为什么 AIME 数学题适合经验蒸馏**: 模型在 AIME 上的 pass@1 只有 ~40-60%，有大量"部分正确"的中间状态，产生了丰富的对比信号。

#### 3.7.2 HumanEval/MBPP 的难度分析

| 数据集 | 题数 | 当前 B 模型预估 pass@1 | 对比信号质量 |
|--------|:---:|:---------------------:|:----------:|
| HumanEval (基础) | 164 | ~83% (Qwen2.5-Coder-7B) | ⚠️ 低 — 大部分题已解决 |
| HumanEval+ (增强测试) | 164 | ~75% (更严格的测试) | ⚠️ 中 — 有 25% 失败率 |
| MBPP (基础) | 500 | ~76% | ⚠️ 中低 — 76% 已解决 |
| MBPP+ (增强测试) | 500 | ~65% | ✅ 中 — 有 35% 失败率 |
| APPS_Plus (竞赛级) | ~1000 | ~30-50% | ✅ 高 — 丰富对比信号 |
| BigCodeBench Lite Pro | ~150 | ~40-55% | ✅ 高 — 真实场景 |
| LiveCodeBench | ~300 | ~35-50% | ✅ 高 — 竞赛级，隐藏测试 |

**问题**: 当 B 模型在 HumanEval 上已经 83% pass@1 时，83% 的问题没有对比信号 (因为都通过了)，只有 17% 的问题能提供信号。

#### 3.7.3 双层验证策略

```
第一层 — 快速验证 (当前数据集, 1 周):
  数据集: HumanEval+ / MBPP+
  目的: 验证经验蒸馏 pipeline 的工程正确性
  预期: 效果可能有限 (+1~3%)，主要验证流程能跑通
  
  具体做法:
  - 只对 pass_rate < 0.8 的难题做经验提取
  - 用 B=Qwen2.5-Coder-7B (较低的通过率提供更多信号)
  - 预期提取出 5-10 条基本经验

第二层 — 效果验证 (Phase 2, 2-3 周):
  数据集: APPS_Plus (竞赛级) + BigCodeBench Lite Pro
  目的: 在高对比度数据上验证经验的真实提升效果
  预期: 提升更显著 (+3~8%)
  
  具体做法:
  - APPS_Plus 有丰富的 pass_rate 分布 (0%~100%)
  - 按难度分层: easy (60-100%) / medium (30-60%) / hard (0-30%)
  - 每层分别提取经验，观察哪层经验最有泛化价值
  - 用 B=DeepSeek-V3.2 做跨模型验证

第三层 — 跨域扩展 (Phase 3, 4-6 周):
  数据集: 加入数学推理 (GSM8K, MATH, AIME)
  目的: 验证经验蒸馏方法在代码之外的泛化能力
  预期: 数学领域的经验提取更类似 Training-Free GRPO 的设置
  
  具体做法:
  - 复用 youtu-agent 已验证的 math pipeline
  - A 模型学习生成"数学解题策略提示词"
  - 与代码域的经验对比，寻找跨域共性
```

#### 3.7.4 如果简单数据集效果确实不明显

这不是失败，而是**有价值的负面结果**，说明：

1. **经验蒸馏需要困难样本**: 简单样本上好的提示词和差的提示词都能让模型做对，无区分度
2. **简单基准的饱和效应**: 当模型能力接近基准上限时，提示词优化的边际收益递减
3. **方法边界清晰**: 这恰好验证了 Training-Free GRPO 论文的核心观察——经验在困难任务上最有效

**应对方案**:

```
如果 HumanEval+ 上经验效果 < 2%:
  → 不气馁, 这是预期内的
  → 立即切换到 APPS_Plus / BigCodeBench Lite Pro
  → 在困难数据上重新评估

如果 APPS_Plus 上经验效果仍 < 2%:
  → 检查经验蒸馏 prompt 模板是否需要针对代码域调整
  → 检查 A 模型生成的提示词是否有足够的多样性
  → 考虑增加 grpo_n (rollout 数量) 以提高对比度

如果所有数据集上经验都无效:
  → 说明代码提示词优化可能不需要复杂的经验蒸馏
  → 或者当前的提示词格式已经接近最优
  → 这也是有价值的发现 (证明 baseline 已经接近上限)
```

#### 3.7.5 数据集难度的渐进路线

```
难度曲线:
  
  HumanEval ──── MBPP ──── HumanEval+ ──── MBPP+ ──── BigCodeBench ──── APPS ──── LiveCodeBench
  (最简单)                                                              (最难)
     83%          76%         75%           65%         40-55%        30-50%        35-50%
     
  验证阶段          ←──────── 从这里开始 ────────→                  效果展示阶段
  (pipeline 调试)  (中等难度, 有对比信号)                           (证明方法有效性)
```

**核心建议**: **不要停留在 HumanEval/MBPP**。用它们快速验证 pipeline 正确性 (1 周内)，然后立即迁移到 APPS_Plus 等更难的数据集上进行真正的效果评估。这与 Training-Free GRPO 论文的做法一致——他们也在 AIME (困难) 而非 GSM8K (简单) 上展示效果。

---

## 4. 实施路线图

```
Phase 1 (1-2 周) — 方案一: A 模型经验蒸馏
  【A 模型: Qwen3-4B | B 模型: Qwen2.5-Coder-7B | 数据集: HumanEval+/MBPP+】
├── 1.1 创建 experience/ 模块 (extractor, templates, cache, injector)
├── 1.2 实现基线 rollout 收集脚本
├── 1.3 实现经验蒸馏 prompt 模板 (参考 youtu-agent experience.yaml)
├── 1.4 实现经验注入: 修改 A 模型输入 pipeline (控制 <= 2000 tokens)
├── 1.5 在 HumanEval+/MBPP+ 上验证 pipeline 正确性 → 预期 +1~3%
├── 1.6 对比: 有无经验注入的 A 模型效果
├── 1.7 迁移到 APPS_Plus (困难数据) → 预期 +3~8%
└── 1.8 决策点: 如果 HumanEval+ 效果 < 2%，立即切到 APPS_Plus (预期内)

Phase 2 (2-4 周) — 方案二: B 模型 Agent 化
  【A 模型: Qwen3-4B | B 模型: Qwen2.5-Coder-7B (Agent) | 数据集: APPS_Plus】
├── 2.1 实现 CodingAgent 基类 (含 python_executor tool)
├── 2.2 接入现有 B 模型接口 (b_interface.py)
├── 2.3 修改奖励函数: 增加自我修正奖励 + 效率奖励
├── 2.4 测试: B=Agent vs B=纯生成器
├── 2.5 跨模型验证: B=Qwen2.5-Coder-7B vs B=DeepSeek-V3.2 (API)
├── 2.6 优化: 超时控制、轮次限制、成本控制
└── 2.7 决策点: 如果 B Agent 效果显著，考虑升级 B 模型版本

Phase 3 (4-6 周) — 方案三: 完整流水线
  【A 模型: Qwen3-4B | B 模型: Qwen2.5-Coder-7B + DeepSeek-V3.2 | 数据集: 全谱系】
├── 3.1 数据严格隔离: 经验提取(APPS_Plus) / 训练(OpenCodeInstruct) / 测试(LCB,BigCodeBench)
├── 3.2 实现 Training-Free 预热 → GRPO 训练 的自动化流程
├── 3.3 持续经验更新: 评估后自动收集+蒸馏
├── 3.4 跨模型泛化测试: 多 B 模型交叉验证
├── 3.5 全面评估: HumanEval+, MBPP+, LiveCodeBench, BigCodeBench, APPS
├── 3.6 跨域扩展: 数学数据集 (GSM8K, MATH, AIME) — 对齐 Training-Free GRPO 设置
└── 3.7 决策点: 是否需要升级 A 模型 (如效果饱和且 4B 确为瓶颈)

Phase 4 (长期) — 持续优化
├── 4.1 经验生命周期管理 (自动淘汰低效经验)
├── 4.2 A/B 模型联合经验蒸馏 (提示词经验 + 代码经验)
├── 4.3 C 模型评判标准自适应优化 (根据单元测试结果校准评判标准)
├── 4.4 其他编程语言扩展 (Java, C++)
└── 4.5 B 模型升级到 Qwen3-Coder / DeepSeek-Coder-V2 等更新版本
```

---

## 5. 预期效果

基于 Training-Free GRPO 论文 (+5.4% on AIME 2025) 和我们的复现验证 (+3.3% on AIME 2024)，以及 A 模型能力、数据集难度的分析，分两层预期：

### 5.1 保守预期 (简单数据集验证阶段)

| 指标 | 当前 (纯 GRPO) | 加入 Training-Free | 预期提升 |
|------|:-------------:|:-------------------:|:-------:|
| HumanEval+ pass@1 (B=Qwen2.5-Coder-7B) | 基线 | +经验注入 | +1~3% |
| MBPP+ pass@1 (B=Qwen2.5-Coder-7B) | 基线 | +经验注入 | +1~3% |
| Pipeline 正确性 | — | 经验提取 + 注入无 bug | ✅ 主要目标 |
| 经验条数 | 0 | 5-10 条 | 足够验证 |

> 简单数据集上效果有限是**预期内的**：高通过率 → 低对比信号 → 经验蒸馏空间小。
> 这一阶段的核心目标是**验证 pipeline 工程正确性**，而非追求性能提升。

### 5.2 乐观预期 (困难数据集效果展示阶段)

| 指标 | 当前 (纯 GRPO) | 加入 Training-Free | 预期提升 |
|------|:-------------:|:-------------------:|:-------:|
| APPS_Plus pass@1 (B=Qwen2.5-Coder-7B) | 基线 | +经验注入 | +3~8% |
| BigCodeBench Lite Pro | 基线 | +经验注入 | +3~8% |
| 跨模型泛化 (B=DeepSeek-V3.2) | 基线 | +经验注入 | +2~5% |
| 训练收敛速度 | 基线 | Training-Free 预热 | 快 20-30% |
| GPU 训练时长 | 基线 | 减少 (更好初始化) | -15~25% |
| B Agent self-debug (Phase 2) | 基线 | +迭代修正 | +10~15% |

### 5.3 探索性预期 (跨域扩展)

| 域 | 数据集 | 参考 Training-Free GRPO | 预期 |
|----|--------|------------------------|------|
| 数学推理 | GSM8K / MATH / AIME | 直接对标 (+3~5%) | 可能最高，数学域经验明确 |
| 代码生成 | HumanEval / APPS | 新领域 (+2~8%) | 取决于数据集难度 |
| 通用 NLP | MMLU / BBH | 未知 | 探索性质 |

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 经验质量低 (LLM 幻觉) | 误导 A 模型 | 反幻觉核实 + 人工抽检 + A/B test |
| 上下文超限 (A 模型 4096 tokens) | 截断关键信息 | 经验优先级排序 + 动态裁剪 + 控制 ICL 示例长度 |
| 数据泄露 (训练→测试) | 虚假提升 | 严格三组隔离 (提取/训练/测试) |
| API 成本过高 | 预算超支 | 批量处理 + 经验去重 + 小规模试点 |
| Agent 化后 B 模型变慢 | 训练吞吐下降 | 限制 max_turns + 异步并发 |
| **A 模型能力不足 (4B)** | 提示词质量差 | 先验证再决定；备选 Qwen3-8B；本质创新点是小模型+经验 |
| **简单数据集效果不明显** | 经验蒸馏似乎无效 | 预期内；快速迁移到 APPS_Plus/BigCodeBench；不是失败是方法边界 |
| **B 模型过时 (Qwen2.5)** | 提示词可能过拟合 | 用 DeepSeek-V3.2 做跨模型泛化验证；B 模型可随时替换 |
| **经验过拟合特定 B 模型** | 泛化性差 | 多 B 模型交叉验证；经验蒸馏时使用多个 B 模型的回滚数据 |

---

## 7. 参考文献

- Training-Free GRPO: https://arxiv.org/abs/2510.08191
- Youtu-Agent 实现: /data/zhuldz/youtu-agent (已复现验证)
- 当前 lunwen 项目: /data/zhuldz/lunwen
- verl (火山引擎 RL 框架): https://github.com/volcengine/verl
