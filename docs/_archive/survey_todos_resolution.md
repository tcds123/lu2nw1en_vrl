# 综述 #TODO 分述解析

> **配套文件**: 本文是对 `survey_agent_skill_evaluation_evolution.md` 中遗留 `#TODO` 的逐条详解。
> **数据来源**: 每条解析均基于 arXiv 原文（HTML 全文 / 摘要页）核对，引用编号见综述第 10 节。
> **目的**: 把综述中"一笔带过"的方法细节展开为可复现的技术说明，并标注对本项目（A→B→C + GRPO）的具体落地建议。
> **两个文件一一对应**: 原综述 `survey_agent_skill_evaluation_evolution.md` 中每处遗留标记 ↔ 本文同名小节。用搜索框搜对应编号（1…8）即可在两文件间唯一定位。

---

#### 🔬 深度解析：RL 驱动 Skill 创建的完整机制

以下基于原论文（SAGE: arXiv:2512.17102; ARISE: arXiv:2603.16060）的完整方法章节，逐阶段拆解 prompt/skill 如何在 RL 训练循环中被生成、选择、使用和进化。

---

##### A. SAGE：Skill Augmented GRPO for self-Evolution

**论文全称**: *Reinforcement Learning for Self-Improving Agent with Skill Library* (Wang et al., 2025/2026, AWS Agentic AI + UW-Madison)

**核心场景**: Tool-using agent（AppWorld 环境，457 APIs，模拟日常 App 操作，如 Spotify、Gmail、Venmo 等）。Agent 通过生成 Python 代码与 API 环境交互。任务按 Scenario 组织，同一 Scenario 下的 3 个任务共享相似的指令结构。

**关键问题**: 现有的 skill library 方法依赖 LLM prompting（在 system prompt 里写"遇到类似任务时生成函数并保存"），这完全受限于基础模型的指令跟随能力，无法保证 skill 质量和复用效果。SAGE 的目标是用 RL 训练模型**内生地学会**何时生成 skill、何时使用 skill。

---

###### A.1 统一的 Skill Library Agent 交互格式

SAGE 在 agent 设计上做了一个关键决策：**将 task solving 和 skill generation 统一为同一种交互格式**。

具体来说，agent 与环境交互时不是直接裸调 API，而是：

```
Step 1: 定义一个 Python 函数 skill → 执行该函数 → 函数 save 到 skill library
Step 2: 后续遇到类似任务 → 从 library 中检索相关 skill 函数 → 直接调用
```

对比传统 baseline agent 与 skill library agent 的交互格式：

```
传统 baseline agent（ReAct）:
  → 直接生成代码: api_call("spotify", "login", user="alice")
  → 每个 step 都是独立的、原子化的 API 调用

Skill library agent（SAGE 统一格式）:
  → Task 1: 定义函数 login_spotify(user), 内部封装多个 api_call
             执行该函数, save 到 library
  → Task 2 (同 scenario): 从 library 中取出 login_spotify, 直接调用
```

Agent 能执行的 4 种核心动作：
1. **Skill Usage**: 从 library 中取出 skill `ai` 并调用执行
2. **Skill Generation**: 定义一个新函数 skill `â`（由多个 API 调用组成），定义后立即调用执行
3. **Skill Update**: 如果调用 skill 执行失败，修改 skill 函数体后重新调用
4. **Skill Save**: 如果 skill 执行成功无错误，将新 skill 加入 library 或更新已有 skill

---

###### A.2 Sequential Rollout（序列化 Rollout）— skill 的"跨任务验证"机制

这是 SAGE 最核心的创新。标准 GRPO 每次只对单个任务做 rollout，skill 的质量无法在训练中被验证——因为你不知道这个 skill 在下一个任务里有没有用。SAGE 的解决方案是：

**将 task chain（任务链）代替单任务作为 rollout 的单位。**

具体做法：
- 将同一个 Scenario 下的 2 个（或更多）任务 `(q₁, q₂)` 组成 task chain
- Sequential Rollout 的过程：agent 先处理 q₁ → 期间生成的 skill 累积到 library 中 → 然后带着这个 library 接着处理 q₂
- q₂ 可以**直接使用** q₁ 生成的 skill
- RL 的梯度反向传播时，q₂ 中使用 q₁ skill 的成功信号会**回传**到 q₁ 的 skill generation 步骤

**为什么这样设计很关键**：

| 维度 | 标准 GRPO (单任务) | SAGE Sequential Rollout (任务链) |
|------|-------------------|----------------------------------|
| Skill 质量评估 | 无法评估：skill 只在一个任务里生成和使用 | 跨任务验证：q₁ 的 skill 在 q₂ 中被检验 |
| 奖励信号 | 只知道 q₁ 本身成功与否 | 知道 q₁ 生成的 skill 是否对后续任务有帮助 |
| 梯度流向 | 只奖励任务成功 | 奖励 task 成功 + skill 被复用后的连锁成功 |
| Skill 复用性 | 无法学习"可复用"这个概念 | 内生学会生成可跨任务复用的 skill |

**实现细节**：
- 任务链长度：实验默认用 2 个任务（论文也尝试了 3 个，发现更长链不一定更好——原因见下文分析）
- 采样方式：先选 Scenario → 再从 Scenario 内采 2 个任务组成链
- 训练规模：24 个 Scenario × 2 任务 × G=8 组 = 384 rollouts/step
- 每个 Scenario 的 library 在链内是共享的（同 Scenario 的任务天然相关），但在不同 Scenario 间是隔离的

---

###### A.3 Skill-integrated Reward（技能整合奖励）— 驱动 skill 生成和使用的奖励信号

这是驱动整个学习过程的"发动机"。SAGE 设计了三种奖励的对比实验，最终 Skill-integrated Reward 效果最优。

**三种奖励设计对比**：

**a) Outcome-based Reward（纯结果奖励）**：
```
R₁ = r₁    （q₁ 的任务完成率）
R₂ = r₂    （q₂ 的任务完成率）
```
→ 只奖励任务完成，不奖励 skill 行为。skill 学习完全依赖 Sequential Rollout 的隐式信号。

**b) Chain-based Reward（链路奖励）**：
```
R₁ = r₁ + 1    [当 q₁ 和 q₂ 都成功时]
R₂ = r₂ + 1    [当 q₁ 和 q₂ 都成功时]
```
→ 当整条链路都成功时额外加 1 分。**问题**：不追踪 skill 的具体使用，可能在早期训练（skill 使用率低时）反而强化了"不使用 skill 也能成功"的模式。

**c) Skill-integrated Reward（技能整合奖励）** ✅ 最终方案：
```
R₁ = r₁ + 1[r₁=1] × 1[r₂=1] × 1_skill(q₂|q₁)
R₂ = r₂ + 1[r₂=1] × 1_skill(q₂|q₁)
```

奖励结构拆解：
- `r₁, r₂`：基础的任务完成奖励，均在 [0,1] 之间（来自 AppWorld 的阶段化评分）
- `1[r₁=1] × 1[r₂=1] × 1_skill(q₂|q₁)`：**Skill Generation Reward**——只有当 q₁ 成功且 q₁ 生成的 skill 被 q₂ 使用且 q₂ 也成功时，才给 q₁ 额外 +1 分
- `1[r₂=1] × 1_skill(q₂|q₁)`：**Skill Usage Reward**——当 q₂ 成功且使用了来自 q₁ 的 skill 时，给 q₂ 额外 +1 分
- `1_skill(q₂|q₁)`：二进制指示器，q₂ 是否使用了 q₁ 生成的 skill

**关键设计原则**：
- ❌ q₁ 成功但 skill 没被复用 → R₁ 只有 r₁（基础分），不额外奖励
- ❌ q₁ 失败（不管 skill 是否被后续使用）→ R₁ 只有 r₁（因为 `1[r₁=1]=0`），不奖励失败的 skill
- ❌ q₂ 使用了 skill 但失败 → R₂ 只有 r₂，不奖励无效的 skill 使用
- ✅ q₁ 成功 + q₂ 使用了 q₁ 的 skill + q₂ 也成功 → 两个任务都获得额外加分

**为什么需要对"skill 使用导致失败"不奖励**：防止模型学会**形式上**使用 skill（调用一个函数）但实质上不用其内容完成任务——这是一种 RL 中的 reward hacking 防御。

此外，对于**完全无代码输出直接终止**的情况，SAGE 施加 -1.0 的惩罚奖励，防止模型走"捷径"直接放弃任务。

---

###### A.4 SAGE 的完整训练流程

**阶段化推进**（三步走）：

```
阶段 0 (Base Model):
  Qwen2.5-32B-Instruct + skill library agent prompt (纯 prompting)
  → TGC: 30.7%, SGC: 19.6%  ← 极差，开源模型无法仅靠 prompt 学会 skill 模式

阶段 1 (SFT):
  用 Claude 3.5 Sonnet V2 作为 expert，在训练集上做 rejection sampling
  - temperature 从 0.05 逐步增加到 1.0（20 步），鼓励多样性
  - 同一 scenario 的 3 个任务顺序执行，保留至少前 2 个全成功的 scenario
  - 最终得到 1,129 个高质量 expert trajectories
  - 用 Llama-Factory 做全参数 SFT（batch=128, lr=1e-6, cosine schedule）
  → TGC: 55.2%, SGC: 41.7%  ← 大幅改善但仍不如无 skill 的 GRPO baseline

阶段 2 (SAGE RL):
  SFT 模型基础上运行 SAGE，G=8，4×H100 8-GPU nodes
  - 每 step 做 384 个 rollouts（24 scenarios × 2 tasks × 8 groups）
  - 不做 KL 惩罚，advantage 用 mean 归一化（不用 std）
  - 每个 task 最多 40 个交互 turn，每 turn 最多 1,500 tokens
  - 选 Dev set 上 TGC+SGC 最优的 checkpoint（step 75）
  → TGC: 72.0%, SGC: 60.7%  ← SOTA，超越 expert (Claude) 和所有 baseline
```

**最终效果（Test Normal）**：

| 指标 | Baseline GRPO (无skill) | SAGE (有skill library) | 提升 |
|------|------------------------|----------------------|------|
| TGC (单任务成功率) | 69.2% | 72.0% | +2.8pp |
| SGC (场景级全成功率) | 51.8% | 60.7% | **+8.9pp** |
| Avg. Steps | 16.4 | 12.1 | **-26%** |
| Avg. Tokens | 3,613 | 1,475 | **-59%** |

SGC 的大幅提升（+8.9pp）尤其说明问题——SGC 衡量的是同一 scenario 下 3 个任务**全部**成功的比例，这个指标的提升直接证明了 skill 的跨任务迁移效果。

---

###### A.5 Skill Library 的行为分析（训练前后对比）

论文用 4 个指标追踪 skill library 的行为变化：

| 指标 | Base Model (prompting) | SFT | SAGE | 结论 |
|------|----------------------|-----|------|------|
| Skill Usage Rate | 高（～80%） | 中 | 中（～65%） | SAGE 选择性使用，不滥用 |
| Success Skill Usage Rate | 低 | 中 | **高（>2× base）** | SAGE 用的 skill 真的有用 |
| Skill Library Size | 大 | 中 | 适中 | SAGE 不滥用 skill 生成 |
| Used Skill Num | 低 | 中 | **高** | SAGE 的 skill 库中更多被实际使用 |

**有趣的发现**：Base model 的 Skill Usage Rate 很高（因为它机械地听从 prompt 指令去生成和使用 skill），但 Success Skill Usage Rate 很低——它生成一堆烂 skill，用了一堆烂 skill，结果任务失败。**SAGE 通过 RL 学习到了"少而精"**：用得更少但用得更对。

---

###### A.6 SAGE 的泛化设计：Skill Retrieval

在测试时，SAGE 默认用**Same Scenario**方法（利用数据集的 scenario 标签直接关联 skill）。但论文也探索了更实际的检索方法：

| 检索方法 | TGC | SGC | 说明 |
|---------|-----|-----|------|
| Same Scenario (理想) | 72.0% | 60.7% | 利用原始 scenario 标签 |
| Query N-gram (Jaccard) | 72.0% | 60.1% | 基于 2-gram 字符重叠匹配 |
| Query Embedding (MiniLM) | 69.6% | 59.5% | 基于语义嵌入匹配 |
| Skill Embedding (Qwen3-Emb) | 66.3% | 56.0% | 将 skill 文档向量化检索 |

**结论**：Query N-gram 几乎能达到理想效果（因为 AppWorld 中同 Scenario 的任务查询结构高度相似），简单的字符匹配就够了。Skill Embedding 最差——因为理解"哪个 skill 函数对当前任务有用"是一个非常难的问题，需要理解的是工具使用能力而不仅是文本相似性。

---

##### B. ARISE：Agent Reasoning via Intrinsic Skill Evolution

**论文全称**: *ARISE: Agent Reasoning with Intrinsic Skill Evolution in Hierarchical Reinforcement Learning* (Li et al., 2026, George Washington University + UT Dallas)

**核心场景**: 数学推理（DeepScaleR 数据集，40K 题-答 pairs，来自 AMC、AIME、MATH、OlympiadBench）。与 SAGE 不同，这里的 skill 不是可执行代码，而是**结构化的推理策略文档**（JSON 格式）。

**关键问题（与 SAGE 互补）**: SAGE 把 skill 的检索/管理交给了外部系统（retriever 或 scenario tag），`skill selection` 不经过 policy gradient；skill library 的更新独立于 RL 目标函数。ARISE 的解决方案是：**让同一个 policy πθ 同时掌管 skill 选择（Manager）和解题（Worker），使 skill selection 也参与梯度更新**。

---

###### B.1 Evolving-Skill MDP（进化技能 MDP）— 形式化建模

ARISE 首先将整个问题形式化为一个 **ES-MDP**:

```
ES-MDP = ⟨Q, M, A, T, R⟩

Q: 问题分布
M: skill library 配置空间（状态的一部分，随 agent 的行动而演化）
A: 动作空间（包含 skill 管理操作 + 解题生成）
T: M × A → M  (library 转移函数，确定性的)
R: Q × M × A → {r₀, r₁, r₂}  (三级层次化奖励，r₂ > r₁ > r₀)
```

核心 insight：skill library **不是外部静态资源**，而是 agent 状态的内生组成部分 `st = (qt, Mt)`。这意味着：
- library 的变化会正式影响 MDP 的状态转移
- policy 的梯度更新可以同时改善"选什么 skill"和"怎么解题"
- skill selection 不是独立检索步骤，而是 policy 的**一个可学习的决策**

---

###### B.2 Manager-Worker 分层架构（同一个 Policy πθ）

这是 ARISE 与 SAGE 及其他 skill-augmented RL 方法最本质的区别：

```
                ┌──────────────────────────────┐
                │      Shared Policy πθ        │
                │                              │
                │  ┌────────┐    ┌──────────┐  │
                │  │Manager │    │  Worker   │  │
                │  │(skill  │    │(解题生成) │  │
                │  │选择+生成)│    │          │  │
                │  └────────┘    └──────────┘  │
                └──────────────────────────────┘
                         ↓
              Download Channel    Upload Channel
              (注入skill到prompt)  (蒸馏轨迹→新skill)
```

联合概率分解为三个因子（**同一套参数 θ，不同的 conditioning context**）：

```
πθ(at, τt | qt, Mt) = π_mgr(z | qt, Mt)       ← 选哪个 skill
                     × π_wkr(τ | qt, m̃z)      ← 条件于 skill 生成解题
                     × π_lib(m_new | qt, τ⁺)  ← 从成功轨迹蒸馏新 skill
```

---

###### B.3 Skill Selection：Policy-Driven（用 Log-Probability 打分）

这是 ARISE 最核心的技术贡献——**用 policy 自己的 log-probability 打分来选择 skill，不依赖外部 retriever**。

**Step 1: 对每个候选 skill 打分**

对于 library cache Mc 中的每个候选 skill `mk`，Manager 计算条件 log-probability 作为相关性分数：

```
sk = Σₗ log πθ(mk⁽ˡ⁾ | qt, mk⁽<ˡ⁾)
```

这衡量的是：在当前 policy 看来，给定问题 `qt`，生成 skill 文本 `mk` 的概率有多高。概率越高 → skill 与问题越相关。

**Step 2: Softmax + ε-greedy 选择**

```
pk = exp(sk/σ) / Σⱼ exp(sⱼ/σ)     （温度 σ=1.0）
zt ∼ (1-ε) × δ_argmax(pk) + ε × Uniform(Mc)   （ε=0.1）
```

- 90% 的概率选得分最高的 skill（exploitation）
- 10% 的概率随机探索（exploration）——探索没被用过的 skill

**Step 3: Confidence Gate（置信度门控）**

```
if max_k pk < δ (=0.35):   → 不注入任何 skill
```

防止注入边缘相关（可能误导）的 skill。这个阈值是精度和召回的关键平衡器：
- δ=0.15 → 90% rollout 有 skill 注入，但精度下降（引入了差 skill）
- δ=0.55 → 仅 42% 有 skill 注入，skill 的好处被浪费
- δ=0.35（默认）→ 73% 注入率 + 最高准确率

**为什么这个设计如此重要**：

与 SAGE 的外部 retriever 相比：
- SAGE 的 skill retrieval 不参与梯度更新（用 scenario tag 或检索器）
- ARISE 的 skill selection **直接参与梯度更新**：如果选了一个好 skill → 解题正确 → 高奖励 → policy gradient 会加强"选这个 skill"的倾向
- 这就是 end-to-end 训练：policy 自己学会什么是好 skill、什么时候该用哪个 skill

**计算开销**：cache 大小 Cc=10，每个 skill 截断到 128 tokens 做打分，总计 10×128=1,280 tokens/query 的额外 forward pass（额外约 0.3s per query on A100，约 8% 的总步时）。

---

###### B.4 Skill Generation：Dedicated Summary Rollout (OG+1)

这是一个与众不同的设计——**在标准的 G 个解题 rollout 之外，ARISE 多做一个专门的 skill generation rollout，称为 OG+1**。

**触发条件**：OG+1 仅在存在正 advantage 的轨迹时执行（即有至少一个 rollout 的解是正确的）。输入是：
- 原始问题 q
- 成功轨迹 `τ⁺ = {τᵢ | Âᵢ > 0}`（最多 2 条，每条截断到 400 字符）

**Prompt 模板（完整）**：
```
You are a skill distiller for math reasoning.
Given one question and trajectories from the same
rollout group, summarize ONE reusable skill.

Question: {question}

Group trajectories:
[SUCCESS #1] {clipped response 1 (≤400 chars)}
[SUCCESS #2] {clipped response 2 (≤400 chars)}

Output MUST be a valid JSON object and nothing else
(no markdown/code fences). Schema:
{"skill name", "problem type", "key insight",
 "method", "check"}

Rules:
- Be generic and transferable; do not copy specific numbers.
- Keep the whole skill within 220 characters.
- key insight is the most important field.
- method must contain 2--3 concise steps.
- Focus on improving correctness, not style.
```

**生成的 Skill JSON 示例**：
```json
{
  "skill name": "exponential_base_matching",
  "problem type": "algebra",
  "key insight": "When both sides can be expressed as powers of the same base, set exponents equal",
  "method": [
    "Rewrite each side with a common base",
    "Set the exponents equal and solve",
    "Verify the solution satisfies the original"
  ],
  "check": "Substitute back into the original equation"
}
```

**400 字符硬上限的设计理由**：如果文档长度不统一，长的 skill 会天然有更低的 log-probability（更负的分数），selection scoring 会被偏向短 skill。硬上限消除了长度偏见，让 selection 基于内容相关性。

**四阶段验证 Pipeline**：
1. **Extraction**: 从原始生成中提取 JSON（去掉 markdown fence、额外文字）
2. **Parsing**: 反序列化，验证 5 个必填字段都存在且类型正确
3. **Truncation**: 截断超长字段，总长超 220 字符的直接丢弃
4. **Insertion**: 通过验证的 skill 进入 cache（ADD 操作）

主 summarization 路径的成功率 >85%（Phase I 之后）。如果失败，有一个 fallback：直接从第一条成功 trace 中取前 220 字符做 `trace abstract` skill。

---

###### B.5 Two-Tier Skill Library（双层缓存-储备库架构）

```
┌──────────────────┐      EVICT (utility最低的被踢出)
│  Cache (Mc)      │ ──────────────────────────────→ ┌──────────────────┐
│  容量: Cc=10     │                                  │ Reservoir (Mr)   │
│  活跃选择池      │ ←────────────────────────────── │  容量: Cr=100    │
│  (scoring+选择)  │      LOAD (utility最高的被召回)   │  归档池           │
└──────────────────┘                                  └──────────────────┘
                                                              │
                                                              ↓ DELETE
                                                   (utility < P10 且 usage=0)
```

每个 skill entry 是 `ek = (mk, uk)`，其中 `uk` 是 utility 的指数移动平均值。

固定顺序的 5 个管理操作（每步执行）：
1. **UPDATE**: `u_zt ← 0.9×u_zt + 0.1×Rt`（基于当前步奖励更新被选中 skill 的 utility）
2. **ADD**: OG+1 产出的新 skill `m_new` 入 cache（初始 u=0）
3. **EVICT**: Cache 超容量 → 最低 utility 的 skill 移到 reservoir
4. **LOAD**: 如果 reservoir 中最高的 utility > cache 中最低的 utility → 交换
5. **DELETE**: reservoir 中 utility<P10 且 usage=0 的永久删除

**双层设计的两个理由**：
1. 小 cache（Cc=10）限制了 scoring 的计算成本，确保只有高 utility skill 参与竞争
2. Reservoir 防止过早丢弃可能在训练后期重新变得有用的 skill

---

###### B.6 Hierarchical Reward（三级层次化奖励）

ARISE 的奖励不是二元的（对/错），而是三级的：

```
R = r_task + r_skill ∈ {0, 1, 2}

配置:
  r₀ = 0  → 解错了（不管是否用了 skill）
  r₁ = 1  → 解对了但没使用 skill（correct, unaided）
  r₂ = 2  → 解对了且使用了 skill（correct, with skill）
```

**一个极其关键的设计细节**：r_skill 只在"用了 skill 且解对了"时才给，"用了 skill 但解错了"不给——防止 policy 被强化"使用不相关 skill 也能得分"的错误模式。

**层次化奖励对 advantage 的影响**（最精妙的部分）：

在 group-relative advantage 下（同一组 G=8 个 rollout 之间比较）：

当一个 group 中同时有 r₂=2 和 r₁=1 的 trajectory 时：
```
组均值 μG = (n₁ + 2n₂)/G

当 μG > 1 时（即 r₂ 比 r₀ 多）:
  Â(r₂=2) = (2-μG)/(σG+ε) > 0    ← 正 advantage
  Â(r₁=1) = (1-μG)/(σG+ε) < 0    ← 负 advantage！虽然解对了但被惩罚！
  Â(r₀=0) = (-μG)/(σG+ε) < 0     ← 负 advantage
```

这意味着：**当大部分 rollout 使用 skill 成功了，那些没用 skill 也能成功的 rollout 会受到负梯度**——RL 在主动惩罚"不使用 skill 的正确解"。

这个机制是 ARISE 的核心推动力：它让"skill utilization"成为 policy 的一个内在激励，而非外部附加目标。

---

###### B.7 Two-Phase Training（两阶段训练）

```
Phase I (Warm-Up, 前 500 steps):
  - skill selection 关闭（不使用 skill 解题）
  - 用标准 GRPO + binary reward r_task ∈ {0,1}
  - OG+1 skill generation 仍在执行（静默填充 library）
  - 结束时 library 约有 50-80 个 skill
  - 目的: 建立基础 policy + 填充初始 library

Phase II (Skill-Augmented GRPO, step 501-2000):
  - 全面激活:
    * Manager 开始 scoring + selection + ε-greedy 选择 skill
    * 奖励从 binary r_task ∈ {0,1} 切换为 hierarchical R ∈ {0,1,2}
    * Solution generation 条件于选中的 skill
  - 前 10-20 步: skill utilization 从 0% 飙升到 40-50%（ε 探索驱动）
  - 100-200 步后: utilization 稳定在 65-75%（confidence gate 过滤）
```

**Phase transition 的两个同时变化**：skill selection 激活 + reward 从二元变三元。这三个变化一起创造了"policy improvement → 更好的 trajectory → 更好的 OG+1 skill → 更好的 selection → 更好的解题"的正反馈循环。

---

###### B.8 技能库的定性进化轨迹

ARISE 论文展示了 library 从 seed → mid-training → late-training 的质变过程：

| 阶段 | Skill 名称 | 类型 | 关键 Insight | 特征 |
|------|-----------|------|-------------|------|
| **Seed** | `equation_setup` | Algebra | 将文字问题转化为变量和方程 | 通用启发式 |
| **Seed** | `case_enumeration` | General | 系统性分裂案例独立验证 | 通用启发式 |
| **Step 1k** | `modular_residue_analysis` | 数论 | 模 n 约化表达式约束整数解 | 领域专业化 |
| **Step 1k** | `generating_function_setup` | 组合 | 编码计数序列为多项式系数 | 领域专业化 |
| **Step 2k** | `vieta_root_reconstruction` | 代数 | 用 Vieta 公式表达根的对称函数 | 高度技巧化 |
| **Step 2k** | `angle_chasing_inscribed` | 几何 | 圆内接四边形→弧关系→角方程 | 高度技巧化 |

**关键观察**：
- 前期的 skill（Step 500-800）多为通用启发式（与 seed skill 类似）
- 中期 skill 开始领域专业化（数论、组合数学）
- 后期 skill 成为高度目标化的技巧（Vieta 公式、圆内接角追逐）
- **Library size 在 Step 1500 左右饱和，但准确率继续提升** → 后期增益来自更优的 selection（选对了已有 skill），而非生成更多新 skill

---

###### B.9 实验结果（Qwen3-4B-Instruct-2507）

| Method | AIME 24 | AIME 25 | Omni-MATH (macro avg) |
|--------|---------|---------|----------------------|
| Baseline (untrained) | 53.6 | 45.6 | 21.2 |
| GRPO | 54.1 | 46.5 | 23.9 |
| DAPO | 55.3 | 47.6 | 25.3 |
| GSPO | 54.9 | 48.5 | 25.5 |
| EvolveR + GRPO | 54.8 | 46.3 | 24.6 |
| SimpleMem + GRPO | 54.5 | 46.9 | 24.5 |
| **ARISE + GRPO** | **56.4** | **48.3** | **26.8** |

**关键发现**：
- ARISE 在所有 benchmark 上**一致**优于所有 GRPO 变体和 memory-augmented baseline
- **OOD (Out-of-Distribution) 上的增益最大**（Omni-MATH: +2.9 vs GRPO vs +2.3 on AIME 24）
  - 因为 evolving library 积累的领域特定推理模式可以迁移到未见过的数学领域
- **token overhead 最小**（+3.0% vs GRPO，而 EvolveR 是 +7.9%）——得益于 skill 文档 220 字符的硬上限

**消融实验（AIME 24）**：

| 变体 | 效果 | 结论 |
|------|------|------|
| ARISE (full) | 56.4% | - |
| (a) Binary reward | 54.7% (-1.7) | **最大的精度损失**，skill utilization 从 73% 暴跌到 31% |
| (b) Random selection | 55.1% (-1.3) | 随机注入保持高 utilization（构造上的）但精度下降（不匹配的 skill） |
| (c) w/o OG+1 | 55.0% (-1.4) | library 冻结在 24 个 seed skill，无法覆盖 Olympiad 级多样性 |
| (d) w/o confidence gate | 56.0% (-0.4) | 影响最小但 utilization 在 91%，门控作为噪声过滤器 |

**结论**：层次化奖励是 skill adoption 的**首要驱动因素**（去掉它 utilization 暴跌）。OG+1 在 OOD 场景最不可缺（静态 seed skill 不覆盖 Olympiad 多样性）。

---

##### C. SAGE vs ARISE 对比总结

| 维度 | SAGE | ARISE |
|------|------|-------|
| **场景** | Tool-using agent (AppWorld API) | 数学推理 (DeepScaleR) |
| **Skill 形式** | 可执行 Python 函数 | 结构化 JSON 推理策略 |
| **Skill Selection** | 外部（scenario tag / retriever） | **Policy-driven**（log-prob 打分 + ε-greedy，参与梯度） |
| **Skill 跨任务验证** | Sequential Rollout（任务链） | Hierarchical Reward（r₂>r₁>r₀） |
| **Skill 生成时机** | 任务执行过程中（inline function definition） | 任务执行后（dedicated OG+1 summary rollout） |
| **Library 管理** | 简单的 save/update | 双层 Cache+Reservoir + 5 operations |
| **奖励设计** | Skill-integrated (R₁, R₂ 各有 bonus 条件) | Hierarchical (r₀=0, r₁=1, r₂=2, 三级 + group-relative advantage) |
| **训练阶段** | SFT → RL（需 expert 数据） | Phase I (warm-up) → Phase II (full hierarchy) |
| **需 External Data** | ✅ 需要 (Claude expert SFT) | ❌ 不需要（self-bootstrapping） |
| **Token Overhead** | -59%（生成效率大幅提升） | +3.0%（prompt overhead 极小） |

---

##### D. 对本项目的具体启示

**启示 1: Sequential Rollout (SAGE) 的思想可用于改进本项目的 reward 设计**

本项目的 `custom_reward.py` 中，每个问题独立评估（B 模型生成代码 → C 模型评判 → 单测执行）。SAGE 的思路启发我们：如果数据集中存在天然相关的题目对（如 HumanEval 中同类型的题），可以考虑：
- 将相关题目组成 task chain
- 对每个 chain 的前一个题目，如果 B 模型生成的 solution pattern（或 BAgent 选用的 skill set）被后一个题目复用并成功，给予**额外的 skill reuse bonus**
- 这个 signal 可以直接反馈到 A 模型的 GRPO 更新中，驱动 A 模型学会生成**更可复用**的系统提示词

**启示 2: ARISE 的 Policy-Driven Selection 可用于改进 BAgent 的 skill 检索**

本项目 BAgent 当前使用关键词/描述匹配检索 skill。ARISE 的方法提示：
- 可以直接用 B 模型（72B API）的 log-probability 来对 candidate skills 打分
- 虽然 72B 无法做梯度更新，但 scoring 不需要梯度——selection 本身是 inference-time 的
- 关键是：conditional log-probability score `log πθ(skill | problem)` 能捕捉 **"模型自己认为这个 skill 对解这个问题有多有用"**，这比 cosine similarity 更本质

**启示 3: ARISE 的 Hierarchy Reward 可直接启发 A 模型训练**

本项目当前的奖励 = 70% 单测通过率 + 30% LLM 评判。ARISE 的三级奖励（r₀=0, r₁=1, r₂=2）可以适配到本项目：
- r₁ = 1：B 模型不用 skill（或不用经验注入）解对了
- r₂ = 2：B 模型用了 A 模型生成的系统提示词/skill 注入解对了
- 这个 distinction 可以驱动 A 模型学会输出**真正有用**的提示词（而非仅仅"存在即有用"的稀疏关系）

**启示 4: 两阶段训练与 warm-up + library 填充**

ARISE 的 Phase I（warm-up + 静默填充 library）→ Phase II（激活 hierarchy）的策略可以直接被本项目采用：
- 本项目当前的训练无此阶段分离
- 可以在前 ~20% 的训练步中只做标准 GRPO（不启用 BAgent skill 注入），但**收集成功轨迹中的经验/skill 模式**
- 之后再激活完整的 BAgent + skill injection 管道

## 目录

> 🔎 **搜索定位**：原综述每处 `#TODO TODO-N`（N=1…8）与本文小节标题一一对应。在任一文件按 `Ctrl+F` 搜 `TODO-N`，每个标记在该文件里只出现一次，直接跳到唯一位置。

| 主题 | 跳转 |
|------|------|
| RL 范式四篇（D2Skill / SkillRL / SkillOS / Skill1） | [跳转 →](#todo-1) |
| SkillForge 结构化失效记录 | [跳转 →](#todo-2) |
| CoEvoSkills 弱基线 / 无真值 / 非标准对话 | [跳转 →](#todo-3) |
| Skills-Coach 直接打分 skill vs 本项目打分产物 | [跳转 →](#todo-4) |
| Ctx2Skill 问卷选择题理解 + 奖励运用 | [跳转 →](#todo-5) |
| AutoSkill vs glm5.2 习惯记录 / 专业性 | [跳转 →](#todo-6) |
| SkillClaw 生成方法 vs 自进化 | [跳转 →](#todo-7) |
| EmbodiSkill 数据类型 + 奖励构造 | [跳转 →](#todo-8) |

- [统一对照表：一句话结论](#对照表)

---

## TODO 总览

综述第 3 节（Skill Evolution 四大范式）共遗留 8 处 `#TODO`，按位置对应：

| # | 位置 | 论文 | 问题焦点 |
|---|------|------|---------|
| 1 | §3 表头 | D2Skill / SkillRL / SkillOS / Skill1 | RL 范式四篇的方法细节缺失 |
| 2 | §3.1 SkillForge | SkillForge (2604.08618) | 结构化失效记录是什么、为何分离、如何保证正确 |
| 3 | §3.1 CoEvoSkills | CoEvoSkills (2604.01687) | 判别器基线是否太弱、无真值、非标准对话可用性 |
| 4 | §3.1 Skills-Coach | Skills-Coach (2604.27488) | 直接打分 skill vs 打分产物，优劣对比 |
| 5 | §3.1 Ctx2Skill | Ctx2Skill (2604.27660) | "规则转问卷选择题"理解是否正确、奖励如何运用 |
| 6 | §3.1 AutoSkill | AutoSkill (2603.01145) | 类似 glm5.2 习惯记录，能否提升专业性 |
| 7 | §3.1 SkillClaw | SkillClaw (2604.08377) | 与其他轨迹提取的区别、生成方法还是自进化 |
| 8 | §3.1 EmbodiSkill | EmbodiSkill (2605.10332) | 数据类型特殊性、奖励构造 |

---

<a id="todo-1"></a>
## TODO-1：RL 范式四篇（D2Skill / SkillRL / SkillOS / Skill1）的方法细节

综述 §3 表头仅给出一句概括，未展开。下面逐一补全（基于 arXiv 摘要 + 方法节标题）。注意这四篇与 SAGE/ARISE 同属 RL 范式，但**信号来源和粒度各不相同**，是本项目 GRPO 设计最直接的对照系。

### 1.1 D2Skill（Tu et al., 2026, arXiv:2603.28716）— 双粒度 + 配对 rollout

- **核心问题**：现有 skill-based RL 只提取 trajectory-level 指导，且缺乏**维护可进化 skill memory 的原则化机制**。
- **双粒度设计（关键创新）**：skill bank 组织为两层粒度：
  - **task skill**：高层指导（"这类任务该怎么做"）
  - **step skill**：细粒度决策支持 + 错误纠正（"这一步出错了该怎么补"）
- **配对 rollout（paired rollout）**：同一 policy 下，对同一任务同时跑两条 rollout——**baseline（无 skill）** vs **skill-injected（注入 skill）**——用两者的**性能差距（performance gap）** 作为 *hindsight utility signal*，同时驱动：
  - skill 更新（哪些 skill 真的有用）
  - policy 优化（policy gradient）
- **skill 维护**：完全从训练时经验构建，通过 reflection 持续扩张，配合 **utility-aware retrieval + pruning**（低效 skill 剪枝）。
- **结论**：在 ALFWorld / WebShop / Search-Augmented QA 上显著优于无 skill 基线；双粒度建模和动态维护两者都关键；学到的新 skill 可跨评测设置迁移，且训练开销适中。
- **对本项目的对应**：D2Skill 的 paired rollout = 综述中反复强调的 **"双 rollout 评估协议（有 skill vs 无 skill）"**，这是验证 skill 贡献、剥离"agent 变强 vs skill 变好"噪声的标准做法。本项目的 `custom_reward.py` 可直接引入此对比作为 reward 的一个维度。

### 1.2 SkillRL（Xia et al., 2026, arXiv:2602.08234）— 递归进化 + 层级 SkillBank

- **核心问题**：memory-based 方法主要存原始轨迹，冗余且噪声大，无法提取可泛化的高层行为模式。
- **三大组件**：
  1. **experience-based distillation**：从原始经验蒸馏出**层级 skill 库 SkillBank**
  2. **adaptive retrieval**：区分通用启发式 vs 任务特定启发式做自适应检索
  3. **recursive evolution**：skill 库与 policy 在 RL 过程中**共同进化**
- **特点**：强调**降低 token 足迹**同时提升推理效用。在 ALFWorld/WebShop/7 个 search-augmented 任务上 SOTA，比强基线高 15.3%，且随任务复杂度上升鲁棒性更好。
- **综述指出的局限（与本节呼应）**：将 skill 检索、利用、进化作为**分离组件**处理，并发更新时有冲突风险——这正是 Skill1 要解决的问题。

### 1.3 SkillOS（Ouyang et al., 2026, arXiv:2605.06614）— 冻结 executor + 可训练 curator

- **核心问题**：agent 是 one-off problem solver，不会从过往交互学习；高质量 **skill curation（技能策展）** 是自进化的关键瓶颈。现有方法要么靠人工策展、要么用启发式 skill 操作、要么只训练短视界的 skill 操作，难以从**间接且延迟的反馈**中学到复杂长期策展策略。
- **架构（关键创新）**：
  - **frozen agent executor**：检索并应用 skill（不可训练）
  - **trainable skill curator**：从累积经验更新外部 **SkillRepo**
- **学习信号设计**：
  - **composite rewards**（复合奖励，含 latency / token cost 等）
  - 在**分组任务流（grouped task streams）**上训练：基于 skill 相关的任务依赖关系组织——**早期轨迹更新 SkillRepo，后期相关任务立即评估这些更新**（这正是 SAGE Sequential Rollout 思想的变体，但作用在 curator 上）
- **结论**：在多轮 agentic 任务和单轮推理任务上，效果和效率都优于 memory-free 和强 memory-based 基线；学到的 curator 可跨 executor backbone 和任务域泛化；SkillRepo 中的 skill 逐渐进化为结构更丰富的 Markdown，编码更高层的 meta-skill。
- **对本项目的对应**：SkillOS 的"evaluation 嵌入训练循环"思路对应本项目 C 模型评判；其 composite reward 设计（已含 latency/token）是本项目"70%单测+30%评判"复合奖励的升级方向。

### 1.4 Skill1（Shi et al., 2026, arXiv:2605.06130）— 单策略统一协同进化

- **核心问题**：维护 skill 库需要三种耦合能力——**selection（选 skill）、utilization（用 skill）、distillation（蒸馏新 skill）**。现有方法孤立优化或用分离奖励源，导致**部分且冲突的进化**。
- **核心创新**：**单个 policy 联合执行**全部操作，朝**共享的任务结果目标**协同进化：
  - 生成 query 搜索 skill 库 → re-rank 选一个 → 条件于该 skill 解题 → 从轨迹蒸馏新 skill
  - **所有学习都来自单一 task-outcome 信号**
- **关键设计（频域分解 credit）**：
  - **低频趋势（low-frequency trend）→ 归功于 selection**（选对 skill 是慢变量）
  - **高频变化（high-frequency variation）→ 归功于 distillation**（蒸馏是快变量）
  - 用频率分离把单一 reward 拆给两个能力，避免多奖励源冲突
- **结论**：ALFWorld/WebShop 上优于 prior skill-based 和 RL 基线；训练动力学证实三能力协同进化；消融显示去掉任一 credit 信号都会退化进化。
- **对本项目的对应**：Skill1 的"单策略统一"正是 ARISE 的同源思想（同一 πθ 管 selection+worker）；频域 credit 分解是对本项目"A 模型同时生成 prompt 和（隐式）决定 skill 使用"的一种归因启发。

### 1.5 四篇对比小结

| 维度 | D2Skill | SkillRL | SkillOS | Skill1 |
|------|---------|---------|---------|--------|
| 训练对象 | policy + skill bank 联合 | policy + SkillBank 共进化 | **只训练 curator**，executor 冻结 | **单 policy** 统一三能力 |
| skill 粒度 | task + step 双粒度 | 层级 SkillBank | Markdown meta-skill | 单一库 |
| 关键信号 | paired rollout gap | 递归蒸馏 + 自适应检索 | composite reward + 分组任务流 | 频域 credit 分解 |
| 解决的痛点 | skill memory 维护无原则 | 原始轨迹冗余噪声 | curation 瓶颈 + 延迟反馈 | 三能力孤立/冲突 |
| 与本项目关系 | ★★★ 双 rollout 可直接用 | ★★ token 足迹启发 | ★★★ composite reward | ★★ 统一 policy 归因 |

---

<a id="todo-2"></a>
## TODO-2：SkillForge 的"结构化失效记录"是什么、为何显式分离、如何保证正确

**论文**: SkillForge (Liu et al., 2026, arXiv:2604.08618, Alibaba Cloud, SIGIR 2026 Industry)

### 2.1 结构化失效记录（Structured Failure Record）具体是什么

SkillForge 的自动诊断流水线是 **三阶段四步**：

```
Phase 1: Failure Analyzer（失效分析器）
  → 对每个 Bad Case 并行做 4 维分析 → 输出 Structured Failure Record
Phase 2: Aggregation（批量聚合）
  → 按类别聚合单条记录 → 识别系统性模式 + 选代表性 case
Phase 3: Skill Diagnostician（诊断器，ReAct）
  → 读聚合报告 + 当前 SKILL.md → 产出 Diagnostic Report + Optimization Plan
Phase 4: Skill Optimizer（优化器）
  → 依 Plan 通过 Virtual File System 做定向编辑 → Skill_v(n+1)
```

**失效记录不是自由文本，而是一个结构化 JSON**，字段包括：

```json
{
  "failure_categories": { ... },   // 4 个维度各自的判定
  "overall_severity":   "high|medium|low",
  "primary_category":   "knowledge|tool|clarification|style",
  "per_dimension_hints": [ ... ]   // 每个维度的诊断提示
}
```

**4 个分析维度（并行、互不替代）**：

| 维度 | 检查内容 |
|------|---------|
| **Knowledge** | 领域知识缺失、错误、自相矛盾 |
| **Tool** | 工具调用缺失、参数错误、结果误读 |
| **Clarification** | 过度询问 / 询问不足 / 问无关问题 |
| **Style** | 语气机械、过度冗长、冷漠 |

**判定规则（确定性，非 LLM 主观）**：
- 任一维度打 high → 判 fail
- 至少两个维度打 medium → 判 fail
- 主类别平局时按**固定优先级** `knowledge > tool > clarification > style` 破解

### 2.2 为什么要显式分离"失效诊断"和"重写生成"

综述原文说 SkillForge "显式分离失效诊断和重写生成"。分离的真正原因有三个：

1. **职责解耦，避免重写器"自圆其说"**。如果把诊断和重写交给同一个 LLM 调用一步完成，模型会倾向于**先归因再按自己的归因改**——归因错了重写必错，且错得"自洽"难以察觉。分离后，诊断器只产出结构化事实（哪里缺、缺什么类型），重写器只能依 Plan 执行，无法自行改变归因。

2. **批量聚合只在"分离"后才可能**。单条记录是无噪声的局部事实；只有把 N 条记录**按类别聚合**（如 `knowledge:missing 出现 8 次`）才能区分**系统性 skill 缺陷** vs **偶发任务噪声**。如果诊断和重写耦合，每条 case 各自重写，系统性模式永远浮现不出来。这是 SkillForge 能"跨任务泛化优于直接操作原始轨迹的系统"的根因。

3. **可审计性**。结构化 JSON 记录 + 确定性判定规则 = 每一步都可回溯。出了问题能定位到"是哪个维度判错了"而非"重写器发挥失常"。

### 2.3 分离之后如何保证正确、从而减少人工验证

SkillForge 用四道闸门保证"分离不等于失准"：

1. **确定性聚合规则**：判定用固定阈值和优先级（见 2.1），不依赖 LLM 主观打分，消除同一 case 多次判定不一致。
2. **批量统计去噪**：聚合后按严重度分布 + issue-type 频率筛 top-k 代表性 case，单条误判被多数票稀释。
3. **诊断器读真实 SKILL.md（Virtual File System）**：ReAct 诊断器不是凭空归因，而是**真的打开当前 skill 文件**，把 issue 映射到具体 SKILL.md 位置，并分类缺陷类型 `missing / insufficient / incorrect`。归因有锚点（文件位置），不是悬空判断。
4. **优化器三原则**：`Minimal Modification`（只改诊断点）+ `Do No Harm`（保留未诊断部分）+ **修改必须有 Plan 指定的 location/content/expected-impact**。重写被 Plan 强约束，无法"自由发挥"。

**减少人工验证的机制**：因为 (a) 记录结构化、(b) 聚合统计去噪、(c) 归因锚定到文件位置、(d) 重写被 Plan 约束，整条链路每一步都可机器校验，所以人工只需抽检"诊断报告是否合理"而非逐条审 skill 改动。论文实验也证明：从 expert-authored / domain-created / generic 三种起点出发，自动进化都能**超越人工策展的专家知识**——说明分离设计在正确性上是站得住的。

### 2.4 对本项目的启示

本项目 `experience/extractor.py` 目前从单条 GRPO trace 蒸馏经验。SkillForge 的分离设计提示：
- 把"经验蒸馏"拆成 **失效分析（结构化 JSON）→ 批量聚合 → 定位经验池条目 → 定向补丁** 四步，而非一次性 LLM 总结；
- 经验条目应带 `primary_category` 和文件锚点，便于去重和归因审计。

---

<a id="todo-3"></a>
## TODO-3：CoEvoSkills 判别器反复纠正 skill，基线是否太弱？无绝对真值、非标准对话可用吗？

**论文**: CoEvoSkills (Zhang et al., 2026, arXiv:2604.01687)

### 3.1 方法核心：Skill Generator + Surrogate Verifier 协同进化

CoEvoSkills 针对的痛点是：Anthropic 的 skill 是**多文件、相互依赖的结构化包**，比单工具复杂得多；现有自进化方法（为 tool 设计）不能直接用；且人工撰写存在**人机认知错位（human–machine cognitive misalignment）**，反而降低 agent 性能（SkillsBench 实测）。

**两个协同进化的组件**：
- **Skill Generator**：迭代精炼 skill（多文件包）
- **Surrogate Verifier（代理判别器）**：**协同进化**，提供 *informative and actionable feedback*，且**不访问 ground-truth 测试内容**（无绝对真值！）

### 3.2 基线对比是否太弱？—— 部分是的，但有合理性

**客观事实**：CoEvoSkills 在 SkillsBench 上对比 **5 个 baseline**，在 Claude Code 和 Codex 两个 executor 上都取得最高 pass rate，并能泛化到另外 6 个 LLM。从"5 baseline + 2 executor + 6 LLM 泛化"看，基线覆盖面并不算单薄。

**但确实存在的"弱基线"隐患**：
1. **Surrogate Verifier 只查输出格式和 pipeline 正确性，不查数值精度**（论文原文明确）。也就是说判别器本身是个"软"标准，它和真正的 Ground Truth Oracle 之间存在 **surrogate-GT gap**。论文案例（系外行星 transit 检测）显示：Version 1–3 都卡在 Oracle 75%，直到 Version 4 才到 100%——说明 surrogate 反馈**不足以单独逼近真值**，需要多轮迭代 + 偶尔的算法级跳变。
2. **无绝对真值时，"自进化优于人工"的结论依赖 Oracle 抽检**。论文的 case study 其实用了 Ground Truth Oracle 做事后验证（"Version 4: Oracle 100%"），说明纯 surrogate 闭环可能收敛到"格式对但答案偏"的局部最优。

**结论**：基线数量尚可，但**判别器本身偏弱**是真实限制。这也正是综述 §3.1 关键发现所说的——执行反馈方法的瓶颈在"执行环境多样性"而非基线强弱。

### 3.3 无绝对真值时方法可用吗？

**可用，但需要约束**。CoEvoSkills 的设计正是为了"无 GT"场景：
- Surrogate Verifier 不看 GT，靠**可执行性 + 格式 + pipeline 正确性**给反馈；
- 通过 **co-evolution**：Generator 进化 skill，Verifier 进化判别能力，互相施压避免 Verifier 停在"全通过"的退化态；
- **可执行 skill（带 Python 函数）是关键**：自进化产出的 skill 把 1096 行人工文档压缩成 64 行文档 + 142 行可测函数，**函数可直接跑、可直接验证**，这就把"无 GT 的软判别"部分转化为"有执行的硬判别"。

**对本项目的直接对应**：本项目的 C 模型评判也是"无绝对真值"（除了单测通过率），CoEvoSkills 的"可执行部分硬化软判别"思路 = 本项目"70% 单测（硬）+ 30% LLM 评判（软）"的复合奖励正是同一哲学。

### 3.4 非标准（开放域）对话时方法可用吗？

**部分受限**。CoEvoSkills 的强项是**有可执行 pipeline 的专业任务**（科学分析、软件修复、数据流水线），反馈信号来自执行结果。对于**纯开放对话、无客观执行结果**的场景：
- Surrogate Verifier 退化为纯 LLM 主观判别，surrogate-GT gap 放大；
- 协同进化可能演变成 Generator 和 Verifier 互相"讨好"的合谋（adversarial collapse 风险，Ctx2Skill 专门用 Cross-time Replay 防这个问题）。
- 因此**非标准对话场景下，CoEvoSkills 需要额外引入外部锚点**（人工抽检 / 规则校验 / 另一模型交叉验证）才稳健。

---

<a id="todo-4"></a>
## TODO-4：Skills-Coach 直接对 skill 打分 vs 本项目对 skill 产物打分——两种方法优劣对比

**论文**: Skills-Coach (Tian et al., 2026, arXiv:2604.27488)

### 4.1 Skills-Coach 的 Training-Free GRPO 到底怎么运作

这是综述标记"⭐与本项目最相关"的方法。其 **Lightweight Optimization Module** 的运作机制（基于论文 §2.3 全文）：

> "Grounded in Training-Free GRPO ... departing from conventional gradient-based parameter optimization, this module leverages the introspective capabilities of LLMs to refine skill instruction and code via a multi-driven mechanism."

**核心：GRPO 的"群组相对"思想被搬到 prompt/代码层面，但不做梯度更新**。每个优化 epoch 有两条并行通路：

**通路 1 — Instruction Optimization（指令优化）**：
1. 对当前 skill 指令生成 **多个变体（variants）**（类似 GRPO 的 G 个 rollout）
2. 在合成测试集上**比较执行得分**
3. 选**得分最高的变体**作为下一轮 baseline（类似 GRPO 的 group-relative advantage：只保留组内相对最优，而非绝对分数）

**通路 2 — Code Optimization（代码优化）**：对 skill 的可执行代码做同样的事。

**四个模块支撑**：
- **Diverse Task Generation**：从 skill 规格自动生成 standard / advanced / boundary 三级测试任务，配自动化校验准则（文件存在、关键词、正则）；8 维度 51 个离散指标。
- **Lightweight Optimization**：上述 Training-Free GRPO。
- **Comparative Execution**：原始 skill 和优化后 skill **同时**跑测试集，得客观可比结果。
- **Traceable Evaluation**：系统评估 + 详细分析报告，驱动下一轮迭代。

**Virtual / Real 双模式**：virtual 模式在沙箱跑，real 模式接真实环境。

**关键收益（论文数据）**：训练时间从 hours → minutes，数据需求从数千样本 → 数十条，降低过拟合并提升跨域迁移。

### 4.2 两种打分对象的根本区别

| 维度 | Skills-Coach（直接打分 skill） | 本项目（打分 skill 的产物） |
|------|------------------------------|---------------------------|
| **打分对象** | skill 本身（指令变体 / 代码变体）在合成测试集上的执行表现 | skill（A 模型生成的系统提示词）经 B 模型产出代码后，由 C 模型 + 单测评代码 |
| **被优化的"模型"** | skill 指令/代码（无参数更新，纯文本进化） | A 模型参数（GRPO 真梯度更新） |
| **奖励信号路径** | skill → 合成测试执行 → 分数（短路径，skill 直接暴露在 reward 下） | A 输出 prompt → B 生成代码 → C 评判+单测 → reward 回传 A（长路径，经过两个冻结模型） |
| **信号噪声** | 低：合成测试是确定性的，skill 与分数直接耦合 | 高：reward 混杂了 B 模型能力、C 模型评判偏差、单测覆盖度 |
| **是否需要 C 模型** | **不需要**（合成测试自给自足，这正是"Training-Free"的含义之一） | 需要（C 模型是奖励来源） |
| **优化粒度** | skill 文本局部变体 | A 模型权重（影响所有 prompt 生成） |
| **可复现性** | 高（合成测试确定性） | 中（B/C 是 API，有随机性） |
| **泛化机制** | skill 文本跨任务复用 | A 模型学到"生成好 prompt"的通用能力 |

### 4.3 两种方法优劣对比

**Skills-Coach（直接打分 skill）的优势**：
1. **信号干净**：skill 与 reward 直接耦合，没有 B/C 模型噪声，credit assignment 简单。
2. **无需 C 模型**：降本降延迟，不依赖外部评判 API。
3. **收敛快**：分钟级，数据需求几十条。
4. **可解释**：每个 skill 变体的得分可追溯。

**Skills-Coach 的劣势**：
1. **依赖合成测试质量**：测试集是 LLM 生成的，**测试本身可能错或偏**——优化出的 skill 是"对合成测试过拟合"的好，未必对真实任务好。这是其最大风险。
2. **只优化已有 skill，不学生成能力**：它精炼 skill 文本，但不会让模型学会"如何生成好 skill"——每次新 skill 仍需重跑优化。
3. **无跨任务泛化的参数学习**：不像本项目 A 模型那样学到可迁移的 prompt 生成策略。
4. **boundary 探索受限于生成器想象**：合成 boundary 任务的天花板 = 任务生成 LLM 的想象天花板。

**本项目（打分产物）的优势**：
1. **端到端学能力**：A 模型学到的是"生成好 prompt"的泛化能力，可迁移到未见任务。
2. **真实任务反馈**：reward 来自真实代码生成 + 单测，比合成测试更贴近部署。
3. **可叠加复合奖励**：latency / token / 代码质量都可纳入。

**本项目的劣势**：
1. **信号噪声大**：长路径 reward 经过 B、C 两个模型，credit assignment 难。
2. **成本高**：每步要调 B（72B）+ C（235B）API，训练慢且贵。
3. **依赖 C 模型质量**：C 评判偏差直接污染 reward（综述 Phase 2 issue #4 已记录"C 模型区分度不足"）。

### 4.4 落地建议：混合而非二选一

两种方法**互补**，本项目可采用 **Skills-Coach 作为轻量前置/降级模式**：
- **前置**：先用 Skills-Coach 的 Training-Free GRPO 在合成测试上快速精炼 BAgent 的 402 个 skill（分钟级，零 C 模型开销），得到高质量 skill 文本，再进入 A→B→C 主训练。
- **降级**：C 模型不可用 / 延迟过高时，用合成测试通过率作为 reward 的降级信号。
- **互补验证**：用 Skills-Coach 的合成测试作为 skill 质量的"快测"，用本项目的真实 C 评判作为"慢测"，两者交叉验证 skill 改进是否真实。

---

<a id="todo-5"></a>
## TODO-5：Ctx2Skill 把规则转换为问卷选择题的理解是否正确？奖励如何运用？

**论文**: Ctx2Skill (Si et al., 2026, arXiv:2604.27660, 清华 + DeepLang)

### 5.1 "规则转问卷选择题"的理解——部分正确，但本质是自博弈

综述原文说 Ctx2Skill "从参考文档中生成合成诊断问题，从 agent 对这些问题的回答反馈中学习"。把它理解成"把规则转成格式问卷选择题"**抓住了"生成 probing 任务"这一面，但漏了核心的对抗自博弈结构**。

**真实机制（基于论文 §3.3 全文）**：Ctx2Skill 是一个 **5 角色多智能体自博弈循环**，目标是从未见过的长上下文 C 中自主发现、精炼、选择 context-specific skill，**无需人工标注、无需外部反馈**。

**5 个 frozen-LM 角色**：

| 角色 | 职责 |
|------|------|
| **Challenger** | 基于 C 和自己的 skill 集 S^C 生成一批**探测任务 + rubric**（rubric 设计成"必须归纳 C 的规则而非复述表面片段"才能答对） |
| **Reasoner** | 基于 C + 任务 + 自己的 skill 集 S^R 作答 |
| **Judge**（中立） | 对每个任务给**二元判定**（对/错），把批次分成 failed / solved |
| **Proposer** | 诊断失败 case 缺了哪部分上下文知识 |
| **Generator** | 据诊断做**针对性 skill 更新** |

**关键：两组 skill 集协同进化**：
- **S^R（Reasoner 侧）**：failed case 流入 → Proposer 诊断缺什么 → Generator 更新 S^R，让 Reasoner 逐步积累 C 的知识
- **S^C（Challenger 侧）**：solved case 流入 → 对应 Proposer-Generator 收紧 S^C，让 Challenger 下一轮维持对抗压力
- **严格对抗**：任何 prompt 都不以对方 skill 集为条件，两边严格 adversarial

**所以"问卷选择题"的比喻**：Challenger 出题（确实是 probing task），Reasoner 答题，Judge 判对错——形似问卷。但**本质区别**在于：
1. 题目不是静态预设的，是 Challenger **带 skill 进化着出的**，越来越难；
2. 答题者也在进化（S^R 累积知识）；
3. 出题者和答题者的 skill 集各自独立进化、互相施压——这是 **GAN 式对抗**，不是单向测验。

### 5.2 奖励具体如何运用？

**这是关键澄清：Ctx2Skill 没有标量 reward，用的是"二元判定驱动的文本编辑"**。

- Judge 只给 **binary verdict（对/错）**，不给分数。
- "奖励"的运用方式是**路由 + 文本更新**，不是 reward gradient：
  - **failed case → 流向 Reasoner 侧**：触发 Proposer 诊断 + Generator 把缺失知识写进 S^R（正反馈：补能力）
  - **solved case → 流向 Challenger 侧**：触发收紧 S^C（让对手更强，倒逼 Reasoner 继续进化）
- 也就是说，"答对"的奖励不是给 Reasoner 加分，而是**让 Challenger 升级出更难的题**；"答错"的信号是**给 Reasoner 补 skill**。这是用**任务难度和 skill 内容**做隐式奖励，而非数值 reward。

### 5.3 防退化机制：Cross-time Replay

纯对抗自博弈会 **adversarial collapse**：Challenger 出题越来越极端、Reasoner 的 skill 过度专门化。Ctx2Skill 引入 **Cross-time Replay**：在代表 case 上找出**平衡最好的 skill 集**，确保 skill 鲁棒可泛化。

### 5.4 对本项目的启示

- Ctx2Skill 的"自博弈出题"思路可用于**自动构造 BAgent skill 的边界测试**（对应本项目 Phase 2 issue #1/#2：触发词检索精度=随机）。让一个 agent 出"探测题"测 skill 检索是否命中，比手工设计测试更系统。
- "二元判定驱动文本编辑"而非标量 reward 的范式，可作为本项目 C 模型评判的补充：当 C 模型给分区分度不足时（issue #4），改用"对/错二元 + 失败原因路由到 skill 更新"。

---

<a id="todo-6"></a>
## TODO-6：AutoSkill 类似 glm5.2 用户习惯记录，能否提升专业性？

**论文**: AutoSkill (Yang et al., 2026, arXiv:2603.01145)

### 6.1 AutoSkill 与 glm5.2 习惯记录的异同

用户观察：glm5.2 会根据用户使用习惯生成对应记录文档——这与 AutoSkill 的"从交互偏好中提取 skill"形似。**但两者定位不同**：

| 维度 | glm5.2 习惯记录 | AutoSkill |
|------|----------------|-----------|
| **抽象层次** | 偏好级（语气、术语） | skill 级（可复用、可组合的能力包） |
| **形式化** | 记录文档 | 标准化 skill 表示，可跨 agent/用户/任务共享迁移 |
| **目的** | 个性化适配 | 终身学习 + 个性化（含专业性） |
| **注入方式** | 隐式影响生成 | 动态注入未来请求，**无需重训底层模型**（model-agnostic plugin） |

AutoSkill 的明确目标：把**短暂的交互经验**（如"减少幻觉、遵循机构写作规范、避免过度技术化措辞"）转化为**显式、可复用、可组合的能力**。它抽象出 skill、支持持续自进化、动态注入——比单纯的"习惯记录文档"结构化得多。

### 6.2 能否提升专业性？—— 有条件地能，但有边界

**能提升专业性的场景**：
- 当用户的"稳定偏好"本质是**领域专业规范**（如医学写作规范、法律术语、金融报告格式）时，AutoSkill 把这些固化为可复用 skill，确实提升专业性。
- AutoSkill 强调"model-agnostic plugin + 标准化 skill 表示"，意味着专业 skill 可**跨 agent 共享**，组织级沉淀专业能力。

**专业性提升的边界（glm5.2 式记录同样受限）**：
1. **偏好 ≠ 知识**。用户偏好（语气、措辞）大多是风格层，**不直接增加领域知识深度**。AutoSkill 论文自己定位为"personalized capabilities"，专业性提升依赖**用户交互中是否真的暴露了专业知识**——如果用户只是闲聊式偏好，沉淀不出专业 skill。
2. **个性化可能损害通用专业性**。过度拟合某用户习惯，可能让模型在该用户场景外表现下降（过个性化 → 通用专业性退化）。
3. **无真值验证**。AutoSkill 从对话 trace 抽 skill，没有客观正确性校验，可能把用户的**错误习惯**也固化成 skill（这是 AutoSkill 未解决的问题，需配合 CoEvoSkills 式 verifier 或 SkillForge 式失效分析）。
4. **"有待商榷"的本质原因**：习惯记录提升的是**一致性**和**契合度**，而非**正确性**。专业性 = 正确性 × 契合度，AutoSkill 主要提升后者。

### 6.3 对本项目的启示

本项目 BAgent 的 `memory.py`（AgentMemory + history.jsonl）与 AutoSkill 同构。要提升专业性而非仅个性化：
- memory 不应只记"用户偏好"，应**分级**：偏好层（风格）/ 规范层（机构约定）/ 知识层（领域事实）；
- 知识层 skill 必须经过验证（单测 / C 模型 / 交叉模型）才能入库，避免把错误习惯固化；
- AutoSkill 的"标准化 skill 表示可跨 agent 共享"启发：BAgent 的 402 个 skill 应有统一 schema，便于组织级复用。

---

<a id="todo-7"></a>
## TODO-7：SkillClaw 与其他轨迹提取方法的区别——是 skill 生成方法还是自进化方法？

**论文**: SkillClaw (Ma et al., 2026, arXiv:2604.08377, OpenClaw 团队)

### 7.1 SkillClaw 的定位：是自进化方法，且核心在"多用户集体进化"

基于摘要和综述定位，**SkillClaw 是自进化方法，不只是 skill 生成方法**。区别的关键在信号源和更新机制：

**SkillClaw 的核心**：在**多用户 agent 生态系统**中，把**跨用户、随时间累积的交互**作为 skill 改进的主信号。

```
多用户各自产生轨迹 → 持续聚合 → 自主 evolver 识别 recurring behavioral patterns
→ 转化为 skill 更新（精炼已有 skill 或扩展新能力）→ 共享仓库同步给所有用户
→ 一个用户上下文里发现的改进，系统级传播，无需用户额外努力
```

### 7.2 与其他轨迹提取方法的本质区别

| 方法 | 轨迹来源 | 进化机制 | 单/多用户 |
|------|---------|---------|----------|
| **Trace2Skill** | 单类型多轨迹 | 多轨迹补丁 + 合并去冲突 | 单 agent |
| **SPARK** | 在线轨迹 | 任务-环境证据验证后才蒸馏 | 单 agent |
| **Memento-Skills** | 读写反射循环 | Router 检索 → 执行 → 反馈更新 | 单 agent |
| **XSkill** | 视觉轨迹 | 视觉摘要 + 跨 rollout 批判 | 单 agent |
| **SkillClaw** | **多用户异构交互轨迹** | **集体 evolver 聚合跨用户重复模式** | **多用户共享仓库** |

**三个本质区别**：

1. **信号是"跨用户互补"的**。SkillClaw 论文强调：不同用户的交互提供**关于 skill 何时工作、何时失败的互补信号**——单用户系统看不到的失败模式，在多用户聚合后才浮现。其他方法都是单 agent 视角。
2. **更新是"系统级传播"的**。一个用户上下文发现的改进，同步到共享仓库后所有用户立即受益。这是**集体进化（collective evolution）**，不是单 agent 的局部进化。
3. **目标是"累积能力提升 + 跨用户知识转移"**，而非仅生成新 skill。它解决的是"相似工作流、工具用法、失败模式跨用户被重复发现"的浪费问题。

### 7.3 是 skill 生成方法还是自进化方法？

**两者都是，但侧重自进化**：
- 它**包含** skill 生成（evolver 扩展新能力）
- 但核心贡献是**自进化机制**：持续聚合 + 模式识别 + 系统级同步
- 综述把它归在 §3.1 执行反馈范式，是因为它的信号来自**实际使用中的执行轨迹反馈**，而非离线蒸馏

### 7.4 对本项目的启示

本项目当前是单 agent（A→B→C）视角。SkillClaw 的多用户集体进化提示：
- 如果本项目部署到多用户/多任务场景，**跨任务的失败模式聚合**（类似 SkillClaw 的跨用户聚合）可暴露单任务视角看不到的 skill 缺陷；
- `experience/cache.py` 的去重聚类可升级为"跨任务 recurring pattern 识别"，把多次出现的失败模式优先沉淀为经验。

---

<a id="todo-8"></a>
## TODO-8：EmbodiSkill 的数据类型特殊性与奖励构造

**论文**: EmbodiSkill (Ju et al., 2026, arXiv:2605.10332, NJU + MSRA + 清华 AIR)

### 8.1 数据类型特殊性——具身轨迹 + skill 双结构

这是 8 个 TODO 中数据类型最特殊的。EmbodiSkill 处理的是**具身环境**（ALFWorld / EmbodiedBench），其数据类型与文本/代码任务根本不同：

**具身轨迹形式化（论文式 1）**：
```
τ = (I, o₁, a₁, …, o_T, a_T, r)
```
- `I`：自然语言任务指令
- `o_t`：**环境观察**（视觉状态、空间布局、物体配置——非文本）
- `a_t`：**物理动作**（导航、抓取、放置——非 API 调用）
- `T`：有限视界
- `r ∈ {0,1}`：**最终任务成功信号（二元）**

**特殊性带来的核心难题**：**失败归因歧义**。一次失败的任务执行可能反映：
- (a) **skill 内容错误**（skill 本身有缺陷）—— 应改 skill
- (b) **执行失误（execution lapse）**（skill 正确，但 agent 没照做）—— **不应改 skill，应强调让 agent 照做**

**直接套用数字环境的"轨迹→粗粒度 skill 更新"范式是错的**：会把 execution lapse 误判为 skill defect，**改坏原本正确的 skill**（论文 Figure 1(c) 的 motivating example）。这是 EmbodiSkill 要解决的核心问题。

**skill 的双结构（论文式 2）**：
```
S⁽ⁿ⁾ = (S_body⁽ⁿ⁾, S_app⁽ⁿ⁾)
```
- `S_body`：主 prescription 内容（可修订）
- `S_app`（appendix）：**不引入新规则，只强调 body 中应被严格遵守的已有内容**——专门用来对付 execution lapse

### 8.2 奖励构造——不是数值 reward，是"skill-aware 反射 + 结构化指令"

**关键澄清：EmbodiSkill 是 training-free 的，executor 参数 θ 固定，没有 reward gradient**。它的"奖励"是**任务成功信号 r 驱动的结构化反射记录**。

**目标函数（论文式 5）**：
```
J(S; πθ) = E_{I,ℰ}[ r(τ) ],  τ = Execute(πθ, I, ℰ, S)
```
即：构造 skill 序列 S⁽⁰⁾→…→S⁽ᴺ⁾，使**同一 executor** 用后期 skill 的任务成功率更高。θ 不动，所有改进外化到 skill。

**Skill-Aware Reflection（核心机制）**：每个轨迹后，skill 进化模型 F 产出最多 K 条反射记录（或 0 条，当轨迹无可靠证据时）：

```
ρᵢ = (reflection type cᵢ, trajectory evidence eᵢ, update directive dᵢ, [target skill content bᵢ])
```

**反射类型按轨迹结果二分（这是"奖励"的真正运用方式）**：

| 轨迹结果 | 反射类型 | 含义 | 指令作用 |
|---------|---------|------|---------|
| **成功 r=1** | `Discovery` | 轨迹揭示了 body 未覆盖的有用内容 | 提议**新增** skill 内容 |
| **成功 r=1** | `Optimization` | 现有内容有效，但有更优做法 | 修订 target 内容 |
| **失败 r=0** | `SkillDefect` | 现有内容错误/不完整/欠规范 | 纠正 target 内容 |
| **失败 r=0** | `ExecutionLapse` | 现有内容**有效**，但 executor 没照做 | **不改 body，只更新 appendix 强调** |

**这就是奖励构造的精髓**：
- `r=1`（成功）→ 正向：Discovery / Optimization（巩固和优化）
- `r=0`（失败）→ **二分归因**：是 skill 的错（SkillDefect）还是 executor 的错（ExecutionLapse）。**只有 SkillDefect 才动 body，ExecutionLapse 只动 appendix**。这避免了把 execution lapse 误改成 skill defect。

### 8.3 为什么这个设计对具身场景不可或缺

数字环境（代码/对话）里，失败基本可归因到 skill/plan；具身环境多了"物理执行失败"这一维度（动作 precondition 不满足、视角不对、物体状态变化）。EmbodiSkill 的 skill-aware reflection 把**轨迹解释相对于当前 skill** 而非孤立总结，用 **skill-changing evidence** 更新 body、用 **execution-lapse evidence** 保留并强调有效指导。

**结果**：在 ALFWorld 上，冻结的 Qwen3.5-27B executor 达 93.28% 成功率，比 GPT-5.2 直接当 agent（无 skill）高 31.58pp。

### 8.4 对本项目的启示

本项目是代码生成（数字环境），失败归因歧义小于具身，但**execution lapse 的类比依然存在**：
- B 模型生成错误代码，可能是因为 (a) A 模型的 prompt/skill 不好，也可能是因为 (b) B 模型自身没照着好 prompt 做。
- 当前 `custom_reward.py` 把所有失败都归到 A 模型（reward 回传 A），相当于把 execution lapse 也算作 skill defect，会污染 A 的训练信号。
- **EmbodiSkill 启发**：在 reward 计算前加一层**归因判定**——B 失败时先用 C 模型判"是 prompt 的问题还是 B 没执行好 prompt"，只把前者回传给 A。这与本项目 Phase 2 的"反幻觉核实"（c_interface.py）方向一致，可显式化。

---

<a id="对照表"></a>
## 统一对照表：八条 TODO 的一句话结论

| TODO | 一句话结论 |
|------|-----------|
| 1 | RL 四篇按训练对象分：D2Skill（双粒度+配对rollout）、SkillRL（递归SkillBank）、SkillOS（冻结executor训curator）、Skill1（单policy统一+频域credit）；本项目最该借的是 D2Skill 双 rollout 和 SkillOS 复合奖励 |
| 2 | 结构化失效记录是 4 维 JSON（knowledge/tool/clarification/style）；分离是为了批量聚合去噪+归因锚定文件位置+重写被Plan约束，从而可机器校验、减少人工 |
| 3 | 基线数量尚可但 surrogate verifier 偏弱（只查格式/pipeline不查精度）；无GT时靠"可执行部分硬化软判别"可用；纯开放对话需补外部锚点防合谋 |
| 4 | 直接打分skill信号干净、无需C模型、收敛快，但依赖合成测试质量、不学生成能力；本项目打分产物端到端学能力但信号噪声大成本高；建议Skills-Coach作前置/降级模式 |
| 5 | "问卷选择题"只抓到出题面，本质是5角色对抗自博弈；没有标量reward，用二元判定驱动文本更新（failed补Reasoner skill，solved收紧Challenger）；Cross-time Replay防退化 |
| 6 | AutoSkill比glm5.2习惯记录更结构化（标准化skill表示、可跨agent共享）；能提升专业性但边界在"偏好≠知识"，主要提升契合度而非正确性，需配验证机制防固化错误习惯 |
| 7 | SkillClaw是自进化方法（非仅生成），核心在多用户集体进化：跨用户互补信号+共享仓库系统级同步，区别于单agent的Trace2Skill/SPARK/Memento |
| 8 | 数据特殊性=具身轨迹(o,a,r)+失败归因歧义；奖励不是数值而是r驱动的skill-aware反射（成功→Discovery/Optimization，失败→二分SkillDefect vs ExecutionLapse），execution lapse只动appendix不动body |

---

## 附录：本文引用的 arXiv 原文

| 编号 | 论文 | arXiv |
|------|------|-------|
| [11] | SkillForge | 2604.08618 |
| [4] | CoEvoSkills | 2604.01687 |
| [12] | Skills-Coach | 2604.27488 |
| [13] | Ctx2Skill | 2604.27660 |
| [14] | AutoSkill | 2603.01145 |
| [15] | SkillClaw | 2604.08377 |
| [16] | EmbodiSkill | 2605.10332 |
| [23] | D2Skill | 2603.28716 |
| [24] | SkillRL | 2602.08234 |
| [25] | SkillOS | 2605.06614 |
| [26] | Skill1 | 2605.06130 |
