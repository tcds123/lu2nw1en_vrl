# SAGE & ARISE: RL驱动Skill创建的两条路径 — 深度解析

> **论文信息**
> - SAGE: "Reinforcement Learning for Self-Improving Agent with Skill Library", Wang et al., arXiv:2512.17102 (AWS + UW-Madison)
> - ARISE: "Agent Reasoning with Intrinsic Skill Evolution in Hierarchical RL", Li et al., arXiv:2603.16060 (GWU + UT Dallas)
> - 原始深度解析见 [`survey_todos_resolution.md`](./survey_todos_resolution.md) 的「🔬 深度解析：RL驱动Skill创建的完整机制」节
> - 本文档提炼关键方法对比和落地建议

---

## 1. SAGE vs ARISE 全景对比

| 维度 | SAGE | ARISE |
|------|------|-------|
| 场景 | Tool-using agent (AppWorld, 457 APIs) | 数学推理 (DeepScaleR, 40K 题) |
| Skill形式 | 可执行Python函数 | 结构化JSON推理策略 |
| **Skill Selection** | 外部 (scenario tag / retriever) | **Policy-driven** (log-prob打分+ε-greedy, 参与梯度) |
| Skill跨任务验证 | Sequential Rollout (task chain) | Hierarchical Reward (r₂>r₁>r₀) |
| Skill生成时机 | 任务执行过程中(inline function) | 任务执行后(dedicated OG+1 summary) |
| Library管理 | 简单的save/update | 双层Cache+Reservoir + 5 operations |
| 奖励设计 | Skill-integrated (R₁,R₂各有bonus条件) | Hierarchical (r₀=0, r₁=1, r₂=2 + group-relative advantage) |
| 训练阶段 | SFT → RL (需expert数据) | Phase I (warm-up) → Phase II (full) |
| Token Overhead | **-59%** (生成效率提升) | +3.0% (prompt overhead极小) |

---

## 2. SAGE: Sequential Rollout (最核心创新)

### 思想

标准GRPO每次只对单个任务做rollout，skill的质量无法在训练中被验证。SAGE的解决方案: **将task chain作为rollout单位**。

### 做法

同一Scenario下2个任务(q₁, q₂)组成链:
- Agent先处理q₁ → 生成的skill累积到library
- 带着library接着处理q₂
- q₂可以直接使用q₁的skill
- RL梯度反向传播: q₂中使用q₁ skill的成功信号**回传**到q₁

### 奖励设计: Skill-Integrated Reward

```
R₁ = r₁ + 1[r₁=1] × 1[r₂=1] × 1_skill(q₂|q₁)
R₂ = r₂ + 1[r₂=1] × 1_skill(q₂|q₁)
```

关键原则:
- ❌ q₁成功但skill没被复用 → R₁只有r₁
- ❌ q₁失败 → R₁只有r₁
- ❌ q₂用了skill但失败 → R₂只有r₂
- ✅ q₁成功 + skill被q₂用 + q₂也成功 → 两个都加分

### 效果

| 指标 | GRPO(无skill) | SAGE | 提升 |
|------|-------------|------|------|
| TGC(单任务) | 69.2% | 72.0% | +2.8pp |
| SGC(场景全成功) | 51.8% | 60.7% | **+8.9pp** |
| Avg Steps | 16.4 | 12.1 | -26% |
| Avg Tokens | 3,613 | 1,475 | **-59%** |

**SGC +8.9pp 证明skill的跨任务迁移效果**。平均token从3613降到1475(-59%)，说明学会的skill让agent高效很多。

---

## 3. ARISE: 最丰富的技术组件

ARISE是方法最丰富的工作，6个关键组件都直接可借鉴。

### 3.1 Policy-Driven Skill Selection (★★★★★ 最值得借鉴)

**不要外部retriever，让policy自己学选skill**。

对library cache McC中每个候选skill mk，计算条件log-probability:

```
s_k = Σ_l log πθ(mk^(l) | q_t, mk^(<l))
```

衡量: **给定问题q_t，policy认为生成这个skill的概率有多高**。概率高 = skill与问题相关。

然后用ε-greedy选择 + confidence gate:
```
p_k = softmax(s_k/σ)
z_t ~ (1-0.1) × argmax(p_k) + 0.1 × Uniform(Mc)

if max p_k < 0.35 → 不注入任何skill
```

**计算开销**: cache=10, 每skill截断128 tokens, 10×128=1,280 tokens/query (约0.3s on A100, 总步时~8%)

### 3.2 Hierarchical Reward (★★★★★ 最易落地)

```
R = r_task + r_skill ∈ {0, 1, 2}
r₀=0: 解错了
r₁=1: 解对了但没使用skill
r₂=2: 解对了且使用了skill
```

**group-relative advantage下的自惩罚**: 当group中大部分rollout用skill且成功时:
```
μ_G = (n₁ + 2n₂)/G
当 μ_G > 1:
  Â(r₂=2) > 0  ✓ (正advantage)
  Â(r₁=1) < 0  ❌ (解对了但没skill → 被惩罚!)
```

**消融证据**: 去掉层级化奖励 → skill utilization从73%暴跌到31%，是skill adoption的首要驱动因素。

### 3.3 OG+1 Dedicated Summary Rollout (★★★★)

在G个解题rollout之外，多做一个专门的skill generation rollout。

**触发条件**: 有至少一个rollout做对(+ advantage)
**输入**: 成功轨迹τ⁺ (最多2条, 每条≤400字符)
**输出**: 结构化JSON skill (总长≤220字符)

```json
{
  "skill name": "exponential_base_matching",
  "problem type": "algebra",
  "key insight": "When both sides can use the same base, set exponents equal",
  "method": ["Rewrite each side with common base", "Set exponents equal", "Verify"],
  "check": "Substitute back"
}
```

**400字符硬上限的原因**: 防止长skill天然有更低log-probability → selection时被偏见。

### 3.4 Two-Tier Skill Library (★★★)

| Layer | Capacity | 用途 |
|-------|----------|------|
| Cache (Mc) | Cc=10 | 活跃选择池，scoring+selection |
| Reservoir (Mr) | Cr=100 | 归档池，防过早丢弃 |

5个管理操作: UPDATE → ADD → EVICT → LOAD → DELETE (每步执行)

### 3.5 Two-Phase Training (★★★★)

| Phase | Steps | Skill Selection | Reward | Library |
|-------|-------|----------------|--------|---------|
| I (Warm-Up) | 0-500 | 关闭 | binary {0,1} | OG+1静默填充 (50-80 skills) |
| II (Full) | 501-2000 | 激活 | hierarchical {0,1,2} | 正反馈循环 |

### 3.6 结果 (Qwen3-4B-Instruct-2507)

| Method | AIME 24 | AIME 25 | Omni-MATH |
|--------|---------|---------|-----------|
| Baseline | 53.6 | 45.6 | 21.2 |
| GRPO | 54.1 | 46.5 | 23.9 |
| DAPO | 55.3 | 47.6 | 25.3 |
| **ARISE+GRPO** | **56.4** | **48.3** | **26.8** |

**OOD增益最大**(Omni-MATH +2.9 over GRPO) → evolving library的领域特定模式能跨域迁移。

**消融**: Binary reward → 54.7(-1.7, utilization从73%→31%)；Random selection → 55.1(-1.3)；w/o OG+1 → 55.0(-1.4)

---

## 4. 对本项目的落地建议 (按优先级)

| 优先级 | 技术 | 本项目位置 | 直接解决的问题 |
|--------|------|-----------|-------------|
| **P1** | ARISE三层reward | custom_reward.py | C模型区分度不足(#4) |
| **P1** | Policy-driven selection | BAgent context.py | 触发词精度=随机(#1/#2) |
| **P2** | OG+1多轨迹对比蒸馏 | experience/extractor.py | 单条蒸馏信息不足 |
| **P2** | SAGE task chain reward | GRPO训练 | reward信号单一 |
| **P3** | Two-tier library | experience/cache.py | skill池管理无原则 |
| **P3** | Two-phase training | GRPO训练流程 | 训练无预热阶段 |

### 具体到 custom_reward.py 的改造

当前: reward = 0.7 × pass_rate + 0.3 × C_score

改造为:
```python
# 检测当前rollout是否注入了经验
injected = trace.get("skills_used", []) != []

if pass_rate == 0:
    reward = 0  # r₀: 解错了
elif not injected:
    reward = 1  # r₁: 解对了但无注入
else:
    reward = 2  # r₂: 解对了且有注入
    
# GRPO的group-relative advantage会自动处理：
# 当组内多数r₂时，r₁的rollout受负梯度 → 驱动A模型学会产出有用的skill
```

---

**参考文献**: 
- Wang et al., "Reinforcement Learning for Self-Improving Agent with Skill Library", arXiv:2512.17102
- Li et al., "ARISE: Agent Reasoning with Intrinsic Skill Evolution in Hierarchical RL", arXiv:2603.16060