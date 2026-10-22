# SkillComposer (UNC-Honda): Generative Skill Composition for LLM Agents — 论文精读

> **论文信息**
> - arXiv:2606.32025v1, 2026年6月
> - 作者: Xinyu Zhao (UNC), Zhen Tan (ASU), Vaishnav Tadiparthi 等 (Honda Research Institute USA)
> - 核心贡献: 用轻量解码器(~3.9M参数)做skill序列预测，联合解决选哪些/选几个/什么顺序
> - 与本项目关系: ★★★★ BAgent skill路由改进方案
> - ⚠️ 注意: 同名但不同方法，与 Tongyi Lab 的 SkillComposer(2606.06079)是不同的论文

---

## 1. 核心问题

当skill库大到196个时，agent面临**结构化skill组合问题**:
1. **选哪些skill** (subset)
2. **要几个skill** (cardinality)
3. **按什么顺序执行** (order)

现有方法:
- **扁平检索**(embedding/LLM judge): 只返回无序子集，忽略依赖关系
- **全部暴露给agent**: token开销大，agent自己推理组合质量不稳定

## 2. 方法: 任务条件的skill序列预测

### 架构概览

```
任务x + 环境c + skill库S={s₁,...,s₁₉₆}
    ↓
冻结Qwen3-Embedding编码器 → task vector hx
    ↓
轻量自回归解码器(3层256dim Transformer) → skill ID序列
```

### 三头联合预测

1. **自回归头**(主): 逐token生成skill ID序列
   - 跨注意力到skill元数据嵌入
   - Beam search解码, 最大N=8

2. **基数头**: 预测需要几个skill (∈[1,8])
   - 线性分类器 on hx
   - p_ψ(bn | x, c) = softmax(W_n·h)
   - 提供独立于AR头的长度信号

3. **集合头**: order-agnostic的skill相关性判断
   - 每个skill si独立匹配task vector hx
   - σᵢ = MLP([h; eᵢ; h⊙eᵢ; |h-eᵢ|])
   - 四路拼接: 身份+交互+距离

### 推理时Logit融合

```
fused_logit = contextual_logit + α·retrieval_prior + β·set_membership
```

三个信号一起决定最终输出:
- `ℓ_t(i)`: AR头的上下文logit(依赖前缀)
- `rᵢ`: TF-IDF检索得分(位置无关)
- `σᵢ`: 集合头membership得分(任务感知)
- α=1.0, β=0.5 (val上调优)

## 3. 训练数据合成

从SkillBench(真实human-curated skill库)构建9,872条训练记录:

| 数据类型 | 数量 | 说明 |
|---------|------|------|
| 真实anchor | 65 | 人工标注的skill序列+执行顺序 |
| 单skill合成 | 2,880 | Gemini 2.5 Flash生成，覆盖全部196个skill |
| 多skill合成 | 6,927 | Gemini 2.5 Pro生成，含依赖关系 |

### Skill依赖图

构建了924条边的skill依赖图:
- **Dependency edge**(I/O overlap): 658条 — 上游skill输出类型覆盖下游输入类型
- **Workflow edge**(共现): 266条 — 真实anchor中经常一起出现的skill对

多skill合成中65%来自依赖边，35%来自workflow边。

## 4. 实验结果

### Skill预测质量

| Method | Synthetic Set F1 | **Real-task Holdout Set F1** |
|--------|-----------------|---------------------------|
| BM25(k=2) | 33.0 | 47.0 |
| TF-IDF(k=2) | 52.5 | 60.6 |
| Qwen3-Emb(k=3) | 43.9 | 58.5 |
| LLM-judge(Gemini) | 61.0 | 59.9 |
| SFT 600M | 71.1 | **43.6** ↓ (过拟合合成分布!) |
| **SkillComposer** | **73.9** | **62.9** ✅ |

**关键发现**: 分布外迁移时，SFT退化27.5pp，SkillComposer只退化11pp。冻结的检索编码器+轻量解码器结构提供了天然的分布外鲁棒性。

### 下游任务表现 (SkillsBench)

| Method | Codex Pass Rate | Gemini Pass Rate |
|--------|----------------|-----------------|
| No Skills | 22.2 | 25.7 |
| All Skills(196个全注入) | 14.3 | 15.7 |
| Top-k retrieval | 34.2-40.0 | 28.6-36.4 |
| Gold Skills(oracle上限) | 44.5 | 44.6 |
| **SkillComposer** | **45.3** | **43.9** |
| SkillComposer vs Top-k | **+23.1pp vs No Skills** | **+18.2pp vs No Skills** |

### Case Study: 少skill > 多skill

**exoplanet detection**:
- SkillComposer(3个): minimal preprocess → Lomb-Scargle → TLS → **reward=1.00**
- Gold Skills(6个): + redundant box-least-squares + heavy exoplanet-workflows → **reward=0.00** ❌
- → 少skill精练集 > 多skill全集

## 5. 消融实验

| 变体 | Set F1 | 变化 |
|------|--------|------|
| AR-only (无辅助头) | 69.3 | baseline |
| + set head | 71.8 | +2.5 |
| **SkillComposer(完整)** | **73.9** | **+4.6** |
| - decode set-fusion(β=0) | 65.0 | **-8.9** |
| - decode retrieval prior(α=0) | 67.5 | -6.4 |

## 6. 对本项目的启示

### BAgent skill路由的直接改进方案

本项目的BAgent当前使用关键词/描述匹配从402个skill中检索——正在遭遇的正是这篇论文要解决的问题(触发词精度=随机，Phase 2 issue #1/#2)。

**可借鉴的技术**:
1. **基数预测**: 需要注入多少条skill到system prompt是最优？当前hardcode top_k=5，应动态预测
2. **依赖图**: 402个skill之间的I/O依赖关系可构建，帮助决定执行顺序
3. **轻量解码器**: 3.9M参数即可做到好的组合预测——比全量暴露给B模型(72B)或LLM judge更高效
4. **冻结编码器+训练解码器**: 复用Qwen3-Embedding等已有组件，只训练小参数量

### 与综述TODO-7(SkillClaw)的对比

SkillClaw是多用户集体进化(跨用户聚合模式)，这篇是单库组合预测(静态库+推理时路由)。两者互补——BAgent需要先有好的路由(这篇)，再考虑多用户聚合(SkillClaw)。

---

**参考文献**: Zhao et al., "Generative Skill Composition for LLM Agents", arXiv:2606.32025v1, 2026.