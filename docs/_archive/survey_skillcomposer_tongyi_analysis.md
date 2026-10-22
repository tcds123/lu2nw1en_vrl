# SkillComposer (Tongyi Lab): Learning to Evolve Agent Skills for Specification and Generalization — 论文精读

> **论文信息**
> - arXiv:2606.06079v1, 2026年6月
> - 作者: Qi Zhang, Zhaopeng Feng, Xiaonan Shi 等 (Zhejiang University, Tongyi Lab, NUS)
> - 核心贡献: 将skill构建分解为三个可训练操作: create / improve / merge，用delta pass rate rejection sampling训练
> - 与本项目关系: ★★★★ 经验提取和skill库管理方法

---

## 1. 核心问题

Skill构建面临一个根本矛盾: **specification vs generalization**：
- skill太具体 → 不能跨任务迁移
- skill太抽象 → 指导不足

现有方法都是"一次性提取"，没有系统机制来平衡这两个维度。

## 2. 三个可训练操作 (核心贡献)

### Skill Create: 从原始执行轨迹中提取skill
- 输入: 任务x + 原始轨迹τ
- 输出: 可复用的skill文本
- 提取过程性知识(procedural knowledge)，非事实性知识

### Skill Improve: 基于新经验精炼skill (驱动specification)
- 输入: 任务x + 当前skill s₀ + 新轨迹τ'
- 输出: 改进后的skill s
- 迭代精炼，让skill更贴合具体任务模式

### Skill Merge: 合并语义相似skill (驱动generalization)
- 输入: 两个语义相似的skill s₁, s₂
- 输出: 合并后的通用skill s
- 相似度 > δ(=0.8)时才合并
- 消除冗余，产生更宽泛可迁移的skill

## 3. Delta Pass Rate Rejection Sampling (训练方法)

**核心思想**: 只用**真正提升了任务完成率**的skill做训练。

```python
# Skill Create
p_base = pass@1(LLM(x), n)       # 无skill时的通过率
p_skill = pass@1(LLM(x, s), n)   # 有skill时的通过率
if p_skill - p_base >= ϵ_c(=0.4):
    D_create += (x, τ) → s       # 加入训练集

# Skill Merge
# 用合并后的skill在原始两个任务上计算平均pass rate提升
Δ = ½ Σᵢ[pass@1(LLM(xᵢ, s), n) - pass@1(LLM(xᵢ, sᵢ), n)]
if Δ >= ϵ_m(=0.4):
    D_merge += (s₁, s₂) → s

# Skill Improve
# 情境类似，验证新skill是否优于旧skill
Δ = ½ Σ[pass@1(LLM(xᵢ, s), n) - pass@1(LLM(xᵢ, sₒ), n)]
if Δ >= ϵᵢ(=0.4):
    D_improve += (x', sₒ, τ') → s
```

**三个操作共享同一个阈值ϵ=0.4**。这个阈值保证只有足够"有用"的skill才进入训练数据。

## 4. 三种部署模式

### Offline模式
1. 对训练集所有任务生成skill (Skill Create)
2. 计算skill间相似度 → 迭代合并(merge)直到无相似pair > δ
3. 推理时检索top-k skill + self-select

### Online模式
1. 空库开始
2. 每个新任务: 生成raw trajectory → Skill Create → 用Skill Improve迭代精炼(T次)
3. 适合流式/无数据场景

### Hybrid模式
1. offline库初始化
2. 在线推理时用Skill Improve进一步specification
3. **平衡覆盖+精炼**

**表现**: Online > Hybrid > Offline (在线精炼带来最大提升)

## 5. 实验设置

- Base模型: Qwen3.5-4B (训练SkillComposer) + Qwen3.5-4B/27B (skill executor)
- 嵌入模型: Qwen3-Embedding-8B (计算skill相似度)
- 训练: LlamaFactory全参SFT
- 超参: lr=1e-6, batch=64, 3epoch, cosine schedule, max_len=65536

### Benchmark
1. **τ²-Bench** (多轮agent): Retail(训练)/Airline/Telecom(跨域)
2. **LiveCodeBench v6** (代码生成): Easy/Medium/Hard分层
3. **AppWorld** (跨任务泛化, 未见过的benchmark)

## 6. 实验结果

### 主结果 (Qwen3.5-4B executor)

| Method | τ²-Bench Overall | LCB v6 Overall |
|--------|-----------------|---------------|
| No Skill | 79.5 | 56.6 |
| offline | 81.3 | 53.7 |
| **online** | **84.8** | **59.1** |
| hybrid | 85.7 | 59.0 |

### 27B executor上提升
- τ²-Bench online: **88.0** (+4.5 over No Skill 83.5)
- LCB v6 online: **83.1** (+3.4 over No Skill 79.7)

### 跨域泛化 (τ²-Bench Airline = 未见过的域)
- 4B executor: No Skill 68.8 → hybrid **82.5** (+13.7!)
- 27B executor: No Skill 77.5 → online **83.5** (+6.0)

### 跨任务泛化 (AppWorld = 从未见过)
| Method | Accuracy | Pass Rate |
|--------|----------|-----------|
| Vanilla | 55.2 | 88.1 |
| Untrained composer | 57.1 | **86.0** ↓ (skill添乱) |
| **SkillComposer-4B** | **57.7** | **88.3** |

### 操作消融 (ablations)

| Skill Operations | τ²-Bench | LCB v6 |
|-----------------|----------|--------|
| Create only | 73.8 | 58.5 |
| Create/Improve | 75.6 | 57.3 |
| **Create/Merge** | **77.5** (+3.7) | **59.0** (+0.5) |
| Create/Merge/Improve | 76.9 | 57.9 |

**关键发现**: **Merge是最重要的操作**(+3.7)。在offline模式下Improve反而拖后腿(因为offline侧重泛化而非专精)。

### 迭代进化 vs 重复采样

SkillComposer的迭代进化持续提升(迭代1→5逐步上升)，而untrained composer在迭代3后开始下降(过拟合噪声)。**说明学到了"持续改善skill"的能力**。

## 7. 对本项目的启示

### 与 experience/ 模块的对应

| SkillComposer | 本项目 | 当前状态 |
|-------------|--------|---------|
| **Skill Create** | `experience/extractor.py` | ✅ 已有 |
| **Skill Improve** | 缺失 | 🔴 待实现 |
| **Skill Merge** | `experience/cache.py`去重 | 🟡 部分，需增强 |
| **Delta Pass Rate过滤** | 无质量门槛 | 🔴 待实现 |

### 具体落地建议

1. **Delta Pass Rate作为入库门槛**: 当前extractor从所有trace提取经验 → 改为只提取p_skill - p_base ≥ 0.4的
2. **增加Merge操作**: 402个skill中语义相似的应该自动合并(降低token开销+提升检索精度)
3. **增加Improve操作**: 当skill被多次使用后，根据C模型反馈自动精炼
4. **Online模式**: 在大规模训练前，先用online模式快速积累有效skill

### 对TF路线的印证

本项目的TF per-task结果(+8.4pp hard on LCB)正对应SkillComposer的"online specification > offline library"发现。全局skill无效的原因也在此——强模型不需要通用启发式，需要的是针对具体失败模式的精炼skill。

---

**参考文献**: Zhang et al., "SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization", arXiv:2606.06079v1, 2026.