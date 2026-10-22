# D2Skill / SkillRL / SkillOS / Skill1 — RL驱动Skill进化四篇对比精读

> **对应**: survey_todos_resolution.md §TODO-1
> **论文列表**:
> - D2Skill (Tu et al., 2026, arXiv:2603.28716)
> - SkillRL (Xia et al., 2026, arXiv:2602.08234)
> - SkillOS (Ouyang et al., 2026, arXiv:2605.06614)
> - Skill1 (Shi et al., 2026, arXiv:2605.06130)

---

## 1. D2Skill: Dynamic Dual-Granularity Skill Bank

### 核心创新

**双粒度skill bank + paired rollout**:

- **Task skill**: 高层指导("这类任务该怎么做")
- **Step skill**: 细粒度决策支持+错误纠正("这一步出错了怎么补")

### Paired Rollout

同一policy下，对同一任务同时跑两条rollout——**baseline(无skill)** vs **skill-injected(有skill)**。用两者的性能差距作为 *hindsight utility signal*，同时驱动:
- skill更新(哪些skill真的有用)
- policy优化(policy gradient)

### 对本项目的价值 ★★★

D2Skill的paired rollout = 综述中反复强调的"**双rollout评估协议(有skill vs 无skill)**"。是本项目custom_reward.py验证skill贡献、剥离"agent变强vs skill变好"噪声的标准做法。

---

## 2. SkillRL: Evolving Agents via Recursive Skill-Augmented RL

### 三大组件

1. **Experience-based distillation**: 从原始经验蒸馏出层级skill库(SkillBank)
2. **Adaptive retrieval**: 区分通用启发式vs任务特定启发式
3. **Recursive evolution**: skill库与policy在RL过程中共同进化

### 特点
- 强调**降低token足迹**同时提升推理效用
- ALFWorld/WebShop/7个search-augmented任务上SOTA (+15.3%)
- **局限**: 检索/利用/进化作为分离组件，并发冲突——正好是Skill1要解决的问题

---

## 3. SkillOS: Experience-Driven Joint Training ★★★

### 核心创新: Evaluation嵌入训练循环

```
frozen agent executor (检索+应用skill，不可训练)
trainable skill curator (更新外部SkillRepo)
```

关键信号设计:
- **Composite rewards**: 含latency/token cost等
- **Grouped task streams**: 早期轨迹更新SkillRepo，后期相关任务立即评估

### 对本项目的价值

1. **"evaluation嵌入训练循环"** = 本项目C模型评判
2. **Composite reward设计**(含latency/token) = 本项目"70%单测+30%评判"复合奖励的升级方向
3. **学到的curator可跨executor backbone和任务域泛化**

---

## 4. Skill1: Unified Co-Evolution

### 核心创新: 单策略统一三能力

**单个policy联合执行** skill搜索、选择、任务求解、skill进化——所有学习来自单一task-outcome信号:

- 生成query搜索skill库 → re-rank选一个 → 条件于该skill解题 → 从轨迹蒸馏新skill

### 频域分解Credit (最精妙设计)

- **低频趋势 → 归功于selection**(选对skill是慢变量)
- **高频变化 → 归功于distillation**(蒸馏是快变量)
- 用频率分离把单一reward拆给两个能力，避免多奖励源冲突

---

## 5. 四篇对比总结

| 维度 | D2Skill | SkillRL | SkillOS | Skill1 |
|------|---------|---------|---------|--------|
| 训练对象 | policy + skill bank联合 | policy + SkillBank共进化 | **只训练curator**，executor冻结 | **单policy**统一三能力 |
| skill粒度 | task + step双粒度 | 层级SkillBank | Markdown meta-skill | 单一库 |
| 关键信号 | paired rollout gap | 递归蒸馏+自适应检索 | composite reward + 分组任务流 | 频域credit分解 |
| 解决的痛点 | skill memory维护无原则 | 原始轨迹冗余噪声 | curation瓶颈+延迟反馈 | 三能力孤立/冲突 |
| **本项目关系** | ★★★ 双rollout可直接用 | ★★ token足迹启发 | ★★★ composite reward | ★★ 统一policy归因 |

---

**参考文献**:
- Tu et al., "D2Skill: Dynamic Dual-Granularity Skill Bank for Agentic RL", arXiv:2603.28716
- Xia et al., "SkillRL: Evolving Agents via Recursive Skill-Augmented RL", arXiv:2602.08234
- Ouyang et al., "SkillOS: Learning Skill Curation for Self-Evolving Agents", arXiv:2605.06614
- Shi et al., "Skill1: Unified Evolution of Skill-Augmented Agents via RL", arXiv:2605.06130