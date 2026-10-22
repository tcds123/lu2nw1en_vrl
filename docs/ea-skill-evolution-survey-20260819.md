# 调研：skill 进化中是否使用进化算法/遗传算法 (2026-08-19)

> **动机**: Phase C 结构自进化 REFUTED 后 (test 0.3703/0.3687 远低于 SkillOpt 0.5464), 用户提出新思路——调研 skill 进化过程中是否使用进化算法/遗传算法相关方法。
> **结论先行**: 是, 而且是 2023→2026 持续活跃的方向。但「用 EA」分三档: 真 GA(种群+交叉+选择)、skill 库增长/精炼(有选择无交叉)、单 lineage hill-climbing 套壳。用户现有 `skill_evolver_math.py` 处于**最弱一档**(连 SePO 的 archive 阶石都没有, 纯 keep-best 单 lineage)。
> **对泛化痛点的核心启示**: 直接对标用户 val→test 脱节的是 **DIVE**(多种群+2-parent 重组+三分防过拟合) 和 **DarwinX**(种群吸收 in-loop proxy 偏差)、**GSE**(cluster consolidation 抽象通用能力)。

---

## 一、判定标准

经典 EA/GA 三要素缺一不可:
1. **种群** — 多个体并存, 非单 lineage
2. **选择压力** — 按 fitness 有偏好保留 (top-k/锦标赛/轮盘), 非随机
3. **重组/交叉** — 两个或多个父本重组生成 offspring

---

## 二、奠基论文 (Line 1: prompt/harness 进化)

### 1. EvoPrompt (Guo et al. 2023, arXiv 2309.08532) — ✅ 教科书式真 GA+DE
- **真 EA**: 同时实现 GA 与差分进化 DE 两种实例化, 全文围绕 EA 算子设计。
- **种群**: N=4~12 (消融测 4/6/8/10/12)。
- **选择**: GA 用**轮盘赌** (p_i=s_i/Σs_j), 消融对比 tournament/random, 轮盘赌最优。DE 按序遍历每个体做 base。
- **crossover**: ✅ 核心。GA: LLM 对两父本 prompt 重组生成 offspring; DE: LLM 将变异结果与 base 做组件替换交叉。
- **mutation**: LLM 执行; DE 取两 prompt 差异部分用 LLM 模拟 F(b−c)。
- **fitness**: development set (大小 200) 上任务准确率。
- **多样性**: 初始种群混手工 prompt + LLM 生成; 无显式 archive/岛/小生境。
- **泛化**: dev/test 分离, dev 上选最优在 test 上报分。无早停。
- **vs 用户 evolver**: 本质区别在**有种群+交叉**。每轮从 N 个体按轮盘赌选 2 父本 crossover, 非单 lineage 变异当前最优。GA 更新 = (N旧+N新) top-N 精英保留。

> **论文简介**(通俗版): EvoPrompt 是这条路的开山之作——第一次系统地证明, 把 prompt 当成可进化的"个体", 用真正的遗传算法(GA)和差分进化(DE)两种经典进化算法, 就能自动找出好 prompt, 而不靠人一遍遍手写试错。
> **要解决的问题**: 写 prompt 是人工反复试错的过程, 作者想把它自动化。
> **基本设定**: prompt = 个体; 用这个 prompt 解开发集上的题所得的准确率 = 适应度(fitness)。
> **GA 流程**(有性繁殖): ① 初始种群 4~12 个 prompt(手写好 prompt + 随机生成的混在一起, 保起点多样)→ ② 算每个的准确率作为 fitness → ③ 轮盘赌选两个父母(分高者占的抽中概率大, 但弱者也有少量概率)→ ④ 交叉: LLM 把两个父母"重组"成一个孩子(如取 A 的"分步推理" + B 的"先列公式")→ ⑤ 变异: LLM 顺手改一句、加个词, 引入随机扰动 → ⑥ 新旧混合 2N 取 top-N 精英保留, 进入下一代 → 循环。
> **DE 流程**(借差异微调): 取两个 prompt 让 LLM 找出它们"哪里不同"(差异向量 F(b−c)), 把这个差异叠加到第三个 prompt(base)上得到变异体, 再做组件替换交叉(保留 base 的某几句、替换成变异体), 好的留下。
> **几个关键名词**:
> - **轮盘赌**: 分数越高抽中概率越大, 但故意不总选第一名, 给弱者留一点被选中的机会——因为弱者可能藏着一个独特的好处, 能结合出新东西。
> - **交叉**: 全文灵魂。两个不同好 prompt 融合, 能产生"各自单独改"得不到的更好结果(1+1>2)。这是它区别于"只盯着当前最优改"的关键。
> - **变异**: 给后代加随机扰动, 防止种群慢慢变得一模一样、失去探索新方向的能力。
> - **精英保留(2N→top-N)**: 每代不是全换新孩子, 而是新旧混合取前 N, 保证上一代的最好 prompt 不会被差孩子顶掉。
> **边界/没做的事**: ① 无多样性维持机制(无 fitness sharing / archive / 岛模型), 存在过早收敛到局部最优的风险; ② 无早停, 只跑满预设代数; ③ 泛化处理简单, 仅 dev/test 分离, 未针对泛化痛点做更多设计——这些留给后面的论文补。

### 2. PromptBreeder (Fernando/DeepMind 2023, arXiv 2309.16797) — ✅ 真 GA, 多样性最全
- **真 EA**: binary tournament GA (引 Harvey 2011), 种群+锦标赛+交叉齐全。
- **种群**: **N=50**, 演化 20-30 代 (1-2k fitness 评估)。
- **选择**: **binary tournament** (随机配对两两竞争, 赢家变异覆盖输家); crossover 选父用 fitness proportionate 轮盘赌。
- **crossover**: ✅ 10% 概率用种群内另一成员(按 fitness 选)的 task-prompt 替换当前。另有 Context Shuffling。
- **mutation**: **9 算子 5 类** — Direct/EDA/EDA-Rank&Index/Lineage/Hyper-mutation(进化 mutation-prompt 本身)/Lamarckian。
- **自指**: 进化单元 = (2 task-prompt)+(1 mutation-prompt)+few-shot; **mutation-prompt 本身也进化**(meta 优化), 这是相对 EvoPrompt 的核心差异。
- **多样性**: **最完善** — (1) prompt 前随机字符串; (2) **fitness sharing**(BERT embedding 相似度, 引 Shir&Bäck 2005); (3) 温度 1.0-2.0 随机化。EDA 算子内 BERT 相似度>0.95 去重。
- **泛化**: train/test 分离; 无官方 split 的手动对半分; 跑到 training fitness plateau 取全程最优在 test 上报分。随机批评估降单批过拟合。
- **vs 用户 evolver**: 远超。种群(50)+锦标赛+9 变异+10% crossover+fitness sharing+自指。**fitness sharing 是经典 GA 防收敛到局部最优的标配, 用户 evolver 完全没有**。

### 3. SePO (Tao et al. 2026, arXiv 2606.04465) — ❌ 非 GA, archive-based 单 lineage hill-climbing
- **判定**: 无 crossover、无种群竞争。自称"open-ended evolutionary search"(引 Darwin Gödel Machine)但无 GA 算子。
- **机制**: 维护 archive (只增不减, keep_better 准入), 每代 G=5、每代 K=2 子代各自从 archive **softmax 采样 1 个父本独立变异**, 无重组。
- **编码**: prompt agent 的 system prompt 文本(自指: 优化 prompt agent 自身 system prompt)。
- **泛化**: **两阶段专门解泛化** — pre-train 在多任务池进化 prompt agent 自身 system prompt → fine-tune 在目标任务优化 task agent prompt。消融去掉 open-ended evolution, AIME'25 掉 6.98 分。
- **vs 用户 evolver**: **最接近用户形态**。消融"w/o open-ended evolution"= linear search always picks latest candidate。即 SePO 相比贪心只多了 archive 阶石 + softmax 父本采样。**用户 evolver 比 SePO 还弱**(连 archive 和 softmax 采样都没有, 纯 keep-best)。

### 4. DarwinX (Zhang et al. 2026, arXiv 2608.07545) — ✅ 真 EA, 泛化处理最严谨
- **真 EA**: 有种群(archive 树含输家)+多父本 merge 重组+contract 选择。标题"Natural Selection"字面义非比喻。
- **编码**: 整个 agent harness (prompts+tools+skills+control flow)。merge 时按 Δ=Δ_code⊕Δ_skill⊕Δ_prompt⊕Δ_tool 组合。
- **crossover**: ✅ **核心创新**。merge operator: 多个解互补任务的变体从共同祖先 H0 合并 additive edits → H=H0⊕Δ。子代须覆盖所有父本 wins 并集才保留。可合并>2 父本(真多父本重组)。
- **选择**: preserve-and-extend contract — 子代须保留父代已解任务集且扩展 S(c)⊇S(p), 否则降为 stepping stone。父本按累积 lineage gain G 排序采样。avg@k 重复确认。
- **多样性**: **保留所有变体含输家** ("a variant worse on average may be the only one that cracks a particular task")。archive 树含 improvers/neutral/stepping stones/archived specialists。
- **泛化**: **4 级信号-测试分离梯度最严谨** — RQ1 同集 / RQ2 held-out 任务 / RQ3 synthetic→real / RQ4 跨 benchmark 迁移。**§5.1「in-loop proxy 过拟合, 种群吸收偏差」几乎是对用户 val→test 脱节的直接诊断** — single-lineage winner-takes-all 让 in-loop 评估偏差固化; 种群/archive 保留多 lineage(含输家)吸收偏差, 整体仍泛化。明确批评"single-lineage search is path-dependent, local wins often regress other tasks"。

---

## 三、Skill 库进化 (Line 2: skill/代码生成)

### 5. EvoAgent (Yuan et al. 2024, arXiv 2406.14228) — ✅ 真 EA, 但进化 agent 配置非 skill
- **真 EA**: 标准 EA 结构, 种群+crossover+selection+代际迭代, 记 EvoAgent(N,T)。
- **crossover**: ✅ Algorithm 1 STEP2 — 让父 agent 各自生成结果, LLM 整合父本信息更新子 agent 设置(信息融合)。
- **selection**: quality-check module (LLM 判定) 筛不合格 child, 每轮留 N 个。
- **被进化**: agent 配置(role/profile/memory/planning/tool), 非 skill 文本本身。
- **泛化**: ❌ 未明确 train/test 分离, 评估即直接在 benchmark 上跑。

### 6. DIVE (Xiong et al. 2026, arXiv 2608.12486) — ✅ 真 EA + 与用户最对位 + 泛化最系统
- **判定**: 多种群并行 + 2-parent Recombination + UCB 选择。**5 篇 Line2 里唯一对 skill 文本做真交叉的**。
- **被进化**: 自然语言 skill 文本(reasoning procedures/verification strategies/failure modes/output constraints)。
- **种群**: **核心设计** — "independently evolving multiple skill populations from bootstrapped experience", 多个独立种群从 bootstrap 子集独立演化, 避免单 trajectory 过拟合。
- **crossover**: ✅ **Recombination (2 parents)** — "Synthesizes complementary strengths from two parent skills into a coherent child skill while resolving redundancy or conflicting guidance"。
- **mutation**: 4 类算子 — Reflective Repair(局部修)/Exploratory Revision(大幅换策略)/Compression(压缩去冗余)/Recombination(交叉)。UCB 自适应分配算子预算(exploit 高收益+explore 少用)。
- **fitness**: val set 上 verifier 评判; test split **严格 hold-out 仅终评**。
- **多样性**: 核心卖点 — bootstrap 不同子集→独立种群→异构算子→joint 互补选择。有 Coverage/Selection Regret 分析。
- **泛化**: **明确处理且最系统**。摘要直指"optimizing a single skill trajectory can overfit to sampled experience or converge to a suboptimal solution"; 三分 evolution/validation/test; Table 3 验证 skill 跨模型族(Qwen3.5→DeepSeek-v4)迁移有效。
- **vs 用户 evolver**: **最直接学术对照**。DIVE 明确批评 greedy single-candidate refinement brittle(line 320), 用多种群并行+2-parent recombination 替代单 lineage。**这是对用户 skill_evolver_math.py 的直接批评与升级路径**。

### 7. GSE (Yang et al. 2026, arXiv 2608.06153) — ❌ skill 库增长非 GA, 但泛化处理针对代码场景
- **判定**: 聚类去重 + replay 门控入库, 无种群无 crossover, 靠 Skill Relation Graph 防冲突。
- **被进化**: skill 文本 + Skill Relation Graph (dependency/co-usage/conflict 边)。
- **泛化**: **三层防线针对过拟合** — (1) cluster-based consolidation 把 trace-specific proposal 聚成通用抽象; (2) replay-driven validation 在历史 case 上验回归; (3) one-project-held-out 评测协议。line 1290 给具体过拟合案例: baseline 学了项目特定冲突 skill, GSE 抽象成高阶能力。
- **vs 用户 evolver**: **与用户"K 候选变异 + gate consolidate"最像**(consolidate≈cluster consolidation, gate≈replay validation)。区别: GSE 显式建模 skill 关系图防冲突, 用户是平铺 skill 池。GSE 过拟合处理更系统。

### 8. EvoSkill (Alzubi et al. 2026, arXiv 2603.02766) — ❌ 带精英池的贪心非 GA
- **判定**: fixed-capacity Pareto frontier 竞争选择 + 单 parent mutation, **无 crossover**, 是"带精英池的贪心搜索"。
- **机制**: 候选 val 分超 frontier 最弱成员则替换入队, 否则丢弃。round-robin 从 frontier 选 1 个 parent。存 git branch 含 lineage。
- **泛化**: 三分 train/validation/held-out test。OfficeQA 15% 数据反低于 10% 提示 mild overfitting(作者承认); 零样本迁移 SealQA skill→BrowseComp +5.3% 作"非过拟合"证据。
- **vs 用户 evolver**: **几乎就是用户"K 候选变异+winner 选择"范式**, 区别是维持 frontier 池而非单 winner, round-robin 探索多 lineage。可视为用户方法的"带 frontier 增强版"。

### 9. Verilog-Evolve (Pei et al. 2026, arXiv 2605.26498) — ❌ skill 库增长, 与用户 A→B→C 流水线结构最同构
- **判定**: create/improve/skip + validation gate, task 内多候选选优, 无种群繁殖无交叉。
- **被进化**: RTL 代码(major/minor 版本化) + modular skill 文件。
- **fitness**: 可执行反馈 — iverilog/vvp 功能仿真 + Yosys 综合 + ABC timing proxy + 可选 GEMM downstream。多目标。
- **泛化**: 部分处理。validation-gated skill evolution 在 GEMM held-out pass rate 上最优; "downstream-friendly RTL"跨目标泛化。无系统项目级 train/test 分离。
- **vs 用户 evolver**: task 内"多候选→选最优→promote"(用户 K 候选变体的硬件版) + skill 库 create/improve/skip+gate(用户 consolidate+gate 的硬件版)。**结构高度同构, 是用户方法在 HDL 领域的镜像**。

---

## 四、综合分类 (9 篇)

| 论文 | 真 EA | 种群 | crossover | 选择压力 | 泛化处理 | 对用户位置 |
|------|:----:|:----:|:--------:|:--------:|---------|-----------|
| EvoPrompt | ✅ | N=4-12 | ✅ 轮盘赌选父 | 轮盘赌+精英 | dev/test 分离 | prompt 优化奠基 |
| PromptBreeder | ✅ | N=50 | ✅ 10% | 锦标赛+轮盘 | train/test+fitness sharing+随机批 | 多样性最全 |
| DarwinX | ✅ | archive树含输家 | ✅ 多父本merge | preserve-extend契约 | **4级梯度最严谨** | **泛化诊断直接对标** |
| EvoAgent | ✅ | N×T代 | ✅ 父本融合 | quality-check | ❌ 未处理 | agent 配置非 skill |
| **DIVE** | ✅ | **多种群bootstrap** | ✅ **2-parent skill重组** | UCB | **三分+跨模型迁移** | **最直接对位/升级路径** |
| SePO | ❌ | archive | ❌ | softmax弱 | 两阶段pre-train→fine-tune | 用户形态最接近 |
| GSE | ❌ | 无 | ❌ | replay门控 | **cluster consolidation+held-out** | 代码场景防过拟合 |
| EvoSkill | ❌ | frontier池 | ❌ | Pareto竞争 | held-out+零样本迁移 | 用户方法增强版 |
| Verilog-Evolve | ❌ | 无 | ❌ | validation gate | 部分held-out | 用户 A→B→C 镜像 |

---

## 五、对用户的核心启示

### 1. 用户当前形态定位
用户 `skill_evolver_math.py` = **单 lineage + K 候选变异 + keep-best winner 选择 + gate consolidate**, 无种群、无 crossover、无多样性维持、无 fitness sharing。

文献定位: 处于 **EvoSkill/Verilog-Evolve 一档(精英池贪心, 无 crossover)**, 但比它们更弱(连 frontier 池/archive 都没有, 纯单 winner)。比 SePO(archive 阶石+softmax 采样)还弱一档。

**这就是 val→test 脱节的直接原因之一**: 单 lineage winner-takes-all 让 in-loop 评估偏差固化(DarwinX §5.1), 候选只在 val 局部最优附近搜索, 无多样性压力打破 val 过拟合(PromptBreeder fitness sharing 正是防此)。

### 2. 升级为真 EA 的两条已验证路径
- **DIVE 路径**(skill 级 crossover, 最直接): 多种群 bootstrap 子集并行演化 + 2-parent Recombination 重组互补 skill + UCB 算子分配 + 三分 evolution/val/test。**明确批评用户这种 greedy single-candidate refinement brittle**。
- **DarwinX 路径**(harness 级, 泛化最严谨): archive 树保留所有变体含输家 + 多父本 merge 重组互补能力 + preserve-and-extend contract 保证不回归。主张"种群吸收 in-loop proxy 偏差"。

### 3. 解决 val→test 泛化的三个已验证范式
| 范式 | 来源 | 机制 |
|------|------|------|
| 种群/多样性吸收偏差 | DarwinX | archive 保留多 lineage 含输家, in-loop proxy 过拟合被种群吸收 |
| fitness sharing 防局部最优 | PromptBreeder | BERT 相似度惩罚种群内相似个体 |
| 多种群 bootstrap 防单 trajectory 过拟合 | DIVE | 不同子集独立演化, 三分严格 hold-out |

用户三次尝试(Task3 内容/Task4 输入/PhaseC 结构)都卡 val→test, 且越来越弱, 说明瓶颈不在「优化哪个维度」而在「单 lineage 搜索本身不泛化」。**DIVE/DarwinX 的诊断正中此点**: 换维度救不了, 需换搜索结构(单 lineage → 种群+交叉+多样性)。

---

## 六、其它相关线索 (未深读, 备查)
- FunSearch (DeepMind 2023) — LLM+进化岛模型进化程序, 与代码生成相关。
- OPRO (Yang/Google 2023, 2309.03409) — LLM 当 optimizer, 非严格 EA 但相邻。
- Voyager (NVIDIA 2023) — skill 库增长+课程, 非严格 GA。
- Self-Evolving Coding Agents (2608.03392) — 综述视角。
- MAGE (2605.10064) — 多 agent co-evolutionary knowledge graph。

---

## 七、用户新思路对位建议 (待用户确认方向)
报告完成后, 待用户明确新思路偏向哪类再深入:
- (A) 给现有单 lineage evolver 加真 GA(种群+crossover 重组多父本 skill) → 借 DIVE 2-parent Recombination + PromptBreeder fitness sharing。
- (B) 用 EA 进化别的对象(B 模型 harness/工具/记忆而非 system prompt 文本) → 借 DarwinX harness-level natural selection。
- (C) 用 EA 解决泛化本身(val/test 分离+种群多样性维持防 val 过拟合) → 借 DarwinX 种群吸收偏差 + DIVE 三分 + GSE cluster consolidation。
- (D) 其它/先看报告再定。

---

**深读 agent 核实声明**: 9 篇全文(除 FunSearch/OPRO/Voyager 仅抓摘要)均成功获取并逐篇核实方法章节与算子定义(含附录 prompt 模板), 非摘要层面臆测。crossover 算子经全文检索确认存在性。
