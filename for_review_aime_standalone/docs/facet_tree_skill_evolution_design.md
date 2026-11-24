# 设计文档:Facet-Tree 真遗传算法驱动的 Skill 进化

> **状态**: 设计草案 v4 (2026-08-25 修订:新增「涌现 vs 预设现状」诚实标注 +「S₀ 多样性初始化」设计点; 其余 v3 定稿项不变)。Stage 0 已 GO (test 0.4953 > 0.4654 破天花板线)。
>
> **⚠️ v4 现状诚实标注 (2026-08-25, 回应「涌现 vs 预设」提问)**:
> 本文档**设计目标**声称「facet 轴自动涌现而非预设」, 但 **Stage 0 实际跑的是 `--no_emergence`**——即 facet 轴 = **4 个固定 `## ` 段人工预设**(Option Comparison / Theorem-Level Precision / Hypotheses / Final Answer), 分支自由度由 `facet_edit_guard` 锁死(每分支只许动自己那段)。**涌现机制 (§2.2(b)) 尚未兑现**——它只在终局 C 组(多代、开涌现、K=3 门控能触发)才启动。**所以当前是「预设」, 「自动涌现」仍是设计目标, 非已实现。** 详见 §2.2 修订与 §十一。这是 novelty 的最大风险点: 若终局 C 组无法兑现涌现, novelty 从「创新」塌回「又调 prompt 模板」。
> **v3 定稿要点**: ① 进化对象 = **Line A 全局 math skill**(非 BAgent 402 条),math skill 已是 slotted template 故 §四-① skill 表示问题已消解;② 起点**钉死真 S₀(4 段 `initial.md`)**,非 V2 默认 5 段版——`Common Failure Modes` 是后期涌现的第5段,从4段起步让其靠涌现从零长出(因果归因+与SkillOpt同点可比,双效,§九-7c);③ 两个设计点已定(§九-7):slotted template(方案 a)+ probe 沿用 probe-split 29 题 seed=42 锁死;④ C-judge 走 `deepseek-v4-pro` 独立账户(§九-8),物理隔离 B 的 V3.2 池,保留 LLM 评判维度;⑤ 成本模型改实测(§九-4):28RPM 共享下限 + "零新评估器≠零新调用"口径修正。Stage 0 = group C 固定轴先验 go/no-go,~3.5–4h 纯 API 下限。
> **动机**: Phase C 结构自进化 REFUTED (test 0.3703), Task3/Task4 连续三次 val→test 脱节且越来越弱。EA 调研 ([docs/ea-skill-evolution-survey-20260819.md](ea-skill-evolution-survey-20260819.md)) 诊断瓶颈不在「优化哪个维度」而在「单 lineage 搜索结构本身不泛化」。用户提新思路:**让 skill 按不同侧重点/baseline 树状进化,分支间交叉**;v2 进一步要求 **facet 轴自动涌现而非人工预设**(类比多头注意力的 specialization 是训练涌现的,不是预设 head)。
> **结论先行**: 站在 DIVE / DarwinX / EvoAgent 三者肩上换轴 + 借 QD(MAP-Elites)的 archive-illumination 框架,落**真 GA 档**。核心增量(经文献核实,见 §一·2)= **"自动涌现的 facet 轴 + facet 组合交叉" 用在 skill 文本进化**——这个交叉缝隙无人占据,但必须窄讲,不能宽泛说"自动分支/MAP-Elites"。

---

## 一·定位

### 一·1  三件已有 + 一个增量

| 组件 | 来自 | 本设计的版本 |
|------|------|------------|
| 2-parent 重组 | DIVE (2608.12486) | 不同 facet 分支交叉,合成兼顾多侧面的子代 |
| archive 树(保留输家/specialist) | DarwinX (2608.07545) / MAP-Elites | 每个侧重点长一条分支,弱变体不删 |
| 多基因配置 crossover | EvoAgent (2406.14228) | facet = 基因,组合式交叉 |
| **自动涌现的 facet 轴(非预设) + facet 组合交叉 用在 skill 文本进化** | ❌ 文献无(见 §一·2 核实) | **本设计增量** |

9 篇里 specialization 轴的取法:

- **DIVE**: specialization 轴 = *task/experience 覆盖*。两个父本各解不同任务,重组合并任务覆盖面。
- **DarwinX**: specialization 轴 = *任务覆盖*(preserve-and-extend 契约判 S(c)⊇S(p))。Δ 分解是 harness 级粗粒度(code/skill/prompt/tool 四个文件),merge 是**加性**的、按任务覆盖选。
- **EvoAgent**: 多基因(role/profile/memory/planning/tool)配置 crossover——最接近 facet 思想,但所有基因在**同一 fitness** 下演化,无 specialist 分支,无 train/test 分离。

**本设计**: facet 是 **skill 文本内部自动涌现的语义段**,每个 facet **独立长成 specialist 分支**,分支间 **facet 组合式交叉**(取父本1的 A 段 + 父本3的 C 段合成子代)。

### 一·2  QD/MAP-Elites 边界核实(2026-08-19 二轮核实,防重蹈漏 DIVE 覆辙)

用户提"像多头注意力自己训练出不同语义层级"后,核验 QD/MAP-Elites 是否已占此空间:

| 论文 | 进化空间 | b(behavior 轴) | crossover | 与本设计关系 |
|------|---------|---------------|-----------|------------|
| **Diverse Prompts** (2504.14367) | prompt ✅ | **手工**(#shots / reasoning depth) | 无 | MAP-Elites 用到 prompt,但**轴是人工预设**——正是要超越的点 |
| **RainbowPlus** (2504.15047) | 对抗 prompt | 多元素 archive | QD 式 | 红队场景,轴非语义 facet |
| **AutoQD** (2506.05634) | 机器人轨迹 ❌ | **自动/无监督** ✅ | 无 | 自动轴已成熟,但**不在文本/skill 域** |
| **VQ-Elites** (2504.08057) | 机器人轨迹 ❌ | **无监督 VQ** ✅ | 无 | 同上,无 crossover |
| **Diversity from Human Feedback** (2310.06648) | 生成文本 | 学习的 descriptor | 无 | 轴可学,但无 crossover 无 skill 分支 |

**核实结论(诚实边界)**:
- "自动涌现 behavior 轴"在 QD 领域**已成熟**(AutoQD/VQ-Elites),不能宣称全新;
- "MAP-Elites 用在 prompt"**已有**(Diverse Prompts),但轴是**手工**的;
- "skill 文本进化 + 自动涌现 facet 轴 + facet 组合交叉"三者交叉**无人占据**——这才是真实增量,且必须窄讲。

这是 **incremental-but-real novelty**: 站在 DIVE/DarwinX/EvoAgent 肩上把 specialization 从「任务覆盖」换成「自动涌现的内容 facet」,crossover 从「整 skill 重组 / 加性 merge」换成「facet 模块化组合」,并借 QD 的 archive-illumination 框架让 facet 轴自适应涌现。**不是吹新范式,是换轴 + 跨域移植**。

---

## 二、核心结构:树状 archive + facet-constrained 分支 + 2-parent facet 组合交叉

### 2.1 进化对象

自然语言 skill 文本,与现有 `agent/b_agent/skills/G*.md` 同形态(YAML frontmatter + markdown body)。新增:body 按 facet slot 分解。

### 2.2 facet 分解:从"预设 4 个"到"自动涌现 + seed"(核心升级)

**⚠️ 现状 vs 目标 (2026-08-25 诚实标注)**: v2 的「自动涌现」是**设计目标**; **Stage 0 实际跑的是 `--no_emergence`, facet 轴 = 人工预设的 4 个 `## ` 段**(§一 v4 标注)。下面 (a)(b) 是完整设计; Stage 0 只实现了 (a) 的冷启动 seed (= 4 段预设), (b) 涌现机制尚**未实现/未验证**, 兑现点在终局 C 组。

**v1(已弃)**: 人工预设 4 个固定 facet。问题——预设轴是人工先验,会限制涌现空间,且和 Diverse Prompts(手工 #shots/depth 轴)同病。

**v2(设计目标)**: facet 轴**自动涌现**,仅用少量 seed 引导冷启动,类比多头注意力的 specialization 是训练涌现而非预设 head。

**两阶段机制**:

**(a) 冷启动 — seed facet(非强制)**: 用进化对象自身的天然段作为初始 seed(仅引导,非固定全集)。

> **进化对象已定为 Line A(全局 math skill),非 BAgent 的 402 条代码 skill**(决策见 §九-7a 及 §四-①)。理由:所有成本/天花板数据(0.4654、probe-29 seed=42、`eval_on`、CI-gate)都在 Line A 上;且 math skill **已是 slotted template**(按 `## ` 切段即可,§四-① 的 skill 表示问题在 Line A 上**已消解**,无需迁移、无需 PoC)。⚠️**起点段数订正(2026-08-21 实读)**:真 S₀(`initial.md`)= **4 段**(Option Comparison / Theorem-Level Precision / Hypotheses / Final Answer);`## Common Failure Modes` 是**进化中后期涌现的第 5 段**,不是 S₀ 固有。Stage 0 起点**钉死=真 S₀ 4 段版**(见 §九-7c),`failure_modes` 作为「待涌现 facet」从零长出而非预装。故下表 seed facet = **math skill 自身 4 冷启动段 + `failure_modes` 作涌现检验目标 + 该段内的 `theorem_type` 子簇**,而非 §一原本参照 BAgent 的 `tool_guidance`/`process`/`self_debug`(那是 Line B 概念,与 math skill 形态不符,已弃)。

| seed facet (Line A math skill 段) | 内容 | 作用 |
|-----------|------|------|
| `option_comparison` | `## Option Comparison` 段:选项对比、量词追踪、最强陈述判定 | seed |
| `theorem_precision` | `## Theorem-Level Precision` 段:定理结论强度、正则性/尺度边界 | seed |
| `hypotheses` | `## Hypotheses` 段:假设与定义域校验、等号/极端情形 | seed |
| `failure_modes` | `## Common Failure Modes` 段:Stage 0 起点为**空**(真 S₀ 不含此段),靠 §2.2(b) 涌现从零长出;段内按 `theorem_type`(真实 13 类值域,见 §九-7b)分簇 | ⚠️ **非冷启动 seed**——是「待涌现 facet」(Decision-C:Stage 0 空占位,终局 C 组兑现) |
| `final_answer` | `## Final Answer` 段:输出格式约束 | seed(冻结,不参与交叉,见 §2.5) |

seed 只在第一代注入,之后系统可**分裂出新 facet / 淘汰无效 facet**(见 (b))。`theorem_type` 是 split 数据自带的结构化元字段(`livemathematicianbench_1-3-6_seed42` 的 `test/val/train/items.json` 每题含 `theorem_type` 字符串列表),evolver 现有 `classify_failure_type`(行 130)已隐式按此分簇——本设计把该隐式分簇**显式化**为 facet 分支信号。

> ⚠️**`theorem_type` 真实值域(2026-08-21 实读,权威值域见 §九-7b)**:值是**字符串 list**(一题可多类,非单值),实值如 `['Existential–Universal', 'Inequality or Bound']`;test(n=106) 前四主类 Existence 35 / Implication 34 / `Inequality or Bound` 31 / Universal 30,值域共 13 类(含 `unknown` 脏标签 test=3、val=5)。**对 `failure_modes` 涌现分簇的含义**:字段名对、evolver 已用(`it.get('theorem_type','?')` 行 357/411),但分簇不能直接按字面值做——需先处理 `unknown`(剔除或归入回退簇)+ 复合类(如 `Universal–Existential`↔`Existential–Universal` 是否存在模态顺序差,是否合并)才能得稳定簇信号。此列为 §七 格式核对必验项。

**(b) 涌现 — 从失败模式自动聚类出新 facet 轴**(类比多头涌现 specialization):
- 每代结束后,收集该分支变异-评估轨迹,对**失败模式**做语义聚类(复用你 `skill_evolver_math.py` 现有的 `classify_failure_type` 行 130 + `compute_failure_distribution` 行 169)
- 聚出的失败簇 = 潜在新 facet 候选:若某簇跨多题复现且现有 facet 未覆盖 → **自动开新分支**(adaptive 分支数,非固定)
- 类似地,长期无 delta 贡献的 facet 分支 → **自动合并/淘汰**(分支数自适应收缩)
- 这是 QD 里 "learned behavior descriptor" 思想(AutoQD/VQ-Elites)向 skill 文本域的移植——**轴是涌现的,不是预设的**

**表示方案**(二选一,见 §四-①):
- (a) slotted template: skill body 用具名段(段名**可由涌现机制新增/重命名**),交叉时按 slot 取段——**可靠**,推荐
- (b) LLM 语义抽取 facet——**有噪声**,但不用改 skill 格式

### 2.3 树结构(借 DarwinX archive 树 + MAP-Elites illumination)

- 根 = 初始 skill S₀ (或 v2 fair-best)
- 每个分支 = 一个 facet specialist:该分支的变异**只允许动对应 facet slot**(facet-edit 约束),其余 slot 继承父代不动
- **分支数 adaptive**: 3 个 seed 起步,运行中按 §2.2(b) 涌现机制自动开新分支/合并无效分支(非固定 4 个)。这是相对 Diverse Prompts 固定网格的关键差异
- **保留所有变体含输家**(DarwinX 原话:"a variant worse on average may be the only one that cracks a particular task")——弱变体可能是某侧面的 specialist,不删。archive = MAP-Elites 式 illumination(每个 facet 维度的高 quality 个体都留一格)
- 节点存:skill 文本 + facet 标签 + lineage(父链)+ val/test delta + 已解侧面集

### 2.4 变异(借现有 evolver 的 stance 旋转)

你的 `skill_evolver_math.py` 已经有 facet stance 雏形:
- `generate_candidates` 的 `angle_hints`(行 462-467,4 个角度)
- `STRUCTURE_ANGLES`(行 537-544,6 个 stance 旋转)

本设计把 stance 旋转**绑定到 facet 分支**:某分支的变异始终只动对应 facet slot,用该 facet 相关的 stance。这比现状(每轮随机轮换角度)更结构化,且**物理上防止分支塌缩**(见 §五)。**涌现新 facet 时**(§2.2(b)),为其生成专属 stance 池。

### 2.5 交叉(借 DIVE 2-parent Recombination + DarwinX 多父本 merge)

- **2-parent facet 组合交叉**:从树里按 fitness 选两个父本,取父本 A 的强 facet slot + 父本 B 的强 facet slot,LLM 合成一个连贯子代(消解冗余/冲突,这正是 DIVE 的 "synthesize complementary strengths ... while resolving redundancy or conflicting guidance")
- 与 DIVE 区别:DIVE 重组整 skill 文本;本设计重组**facet slot**,粒度更细
- 与 DarwinX 区别:DarwinX 是加性 merge(Δ 叠加);本设计是**替换式组合**(slot 级替换)

### 2.6 选择(全局 fitness + preserve-and-extend)

- **within-branch fitness**: 所有分支**共用同一全局 fitness**(现有 `skill_evolver_math.py` 链路,**零新评估器**)。⚠️**订正(2026-08-21 实读 eval_on 回路)**: 该链路 fitness 实际 = **EM pass_rate(k 次采样投票均值)+ bootstrap CI 下界**, **不含 C 模型 LLM 评判**——`eval_on`(行289)→`run_one`(行142)→`math_evaluate`(`evaluator.py:41`)= 纯确定性 EM 判卷(`is_correct = float(predicted_label == correct_label)`)。70/30 的 LLM 评判真实存在于 `c_interface.py` 的 `compute_reward`(GRPO 训练链路 A 模型奖励),**非 skill_evolver 演化分支 fitness**——两条链路 fitness 来源不同,见 `docs/phase2_issues.md` Facet-Tree 问题4。分支差异只靠 facet-constrained mutation + facet-emphasis 种子,不造 facet-specific 评估器
- **preserve-and-extend 契约**(借 DarwinX): 子代须保留父代已解侧面集且扩展 S(c)⊇S(p),否则降级为 stepping stone。防局部提升牺牲其他侧面(直击 Phase C REFUTED 的同类机制)

---

## 三、对实验痛点的直接对应

你三次卡 val→test(Task3 内容 / Task4 输入 / PhaseC 结构)且越来越弱。EA 调研诊断:**换维度救不了,瓶颈在单 lineage 搜索结构本身不泛化**。

本设计同时换两样东西:

1. **换搜索结构**(单 lineage → 树):archive 树保留多 lineage 含输家 → 借 DarwinX「种群吸收 in-loop proxy 偏差」,直接解 val→test 脱节
2. **换 fitness 维度结构**(单一 → facet 分支):不同分支专攻不同 facet → 让树天然长出 specialists,而非逼出单一局部最优

---

## 四、两个决定成败的设计点(含推荐)

### ① skill 表示 — Line A 上已消解,工程前置绕开

> **决策(2026-08-21 定稿)**: 进化对象 = **Line A 全局 math skill**(S₀/v2 根 `tf_global_skill_v2_math.md`),非 BAgent 的 402 条代码 skill。

现状(Line A): S₀ 根技能**已是 slotted template** —— 真 S₀(`initial.md`)含 4 具名段 `## Option Comparison` / `## Theorem-Level Precision` / `## Hypotheses` / `## Final Answer`(1018 chars);`## Common Failure Modes` 是**进化中后期涌现的第 5 段**(晚期 `run_20260728_131934` 5 段版才有,3655 chars;evolved 181981e9 保留五段仅扩 `Failure Modes`)。Stage 0 起点钉死真 S₀ 4 段(§九-7c),`failure_modes` 待涌现。facet 边界**是结构的不是语义的**,`facet_decompose` 按 `^## ` 切段即可,零迁移成本。故原 §四-① 的"绕不开"前提在 Line A 上**不成立**:

- **(a) slotted template(已采纳)**: 即 math skill 现状。交叉按 slot 取段,可靠。slot = §2.2(a) 五段。
- **(b) LLM 语义抽取**: 不采纳。会引入噪声淹灭交叉信号(交叉无增益时无法归因"是交叉本身无效还是抽取噪声"),Line A 无需它。

> **Line B(BAgent 402 条)的 slotted template 迁移成本确实存在**(§九-1),但 Line B 推迟,Stage 0 不触及。

### ② within-branch fitness — 用现有的,别造新评估器

- **不要**搞 facet-specific 评估器(测"工具选得对不对"要单独造,太贵)
- **推荐**: 所有分支共用全局 fitness = C 模型 code-pass(现有 A→B→C)。分支间差异只靠 facet-constrained mutation + facet-emphasis 种子。**直接复用现有 infra,零新评估器**

---

## 五、⚠️ 必须挂的护栏(防 REFUTED 同类机制)

**塌缩风险**: 若 4 个分支的 fitness 都是同一全局 code-pass,且无 diversity 护栏,分支会**塌缩回单目标**——每个分支都往"同一最优 skill"靠拢,facet 分叉失效,退化成 4 个并行单 lineage。这正是 Phase C 结构自进化被 REFUTED(test 0.37)的同类机制。

**至少挂一个**(三选一,推荐第一个,最便宜):

| 护栏 | 来源 | 挂在哪 | 代价 |
|------|------|--------|------|
| **facet-edit 约束强制差异化** | 本设计 | 分支①硬性只能动 `tool_guidance` slot → 物理上无法塌缩 | 最低(变异 prompt 约束) |
| fitness sharing | PromptBreeder | 同 facet 分支内 BERT 相似度惩罚,逼探索 | 中(需 embedding) |
| preserve-and-extend 契约 | DarwinX | 子代须保留父代已解侧面 → 防局部提升牺牲其他 | 中(需侧面集追踪) |

**泛化护栏**(与 EA 调研三范式对齐,至少挂一个):
- **三分 evolution/validation/test**(DIVE):test 严格只终评,绝不入循环
- **种群/多样性吸收偏差**(DarwinX):archive 保留多 lineage 含输家
- **cluster consolidation 抽象通用能力**(GSE):把 trace-specific 聚成通用抽象

---

## 六、「泛化性/总结性」不是分支,是全局选择标准(v2 已消解歧义)

**v1 歧义来源**: 固定 4 个 facet 时,「泛化性/总结性」既是潜在的内容 facet(读法 A:skill 文本写强调抽象),又像 fitness 目标(读法 B:专挑泛化好的)——两者机制不同,EA 上分别管交叉结构和选择压力,故先并列待定。

**v2 消解**: facet 轴改为自动涌现 + 自适应分支数(§2.2)后,「该不该有第 4 个分支」这个伪问题消失了——分支是涌现出来的,不是数到第 4 个。但**底下的概念区分仍然成立且必须立住**:

| 角色 | 机制 | 位置 | 说明 |
|------|------|------|------|
| **内容 facet**(如 `abstraction`) | skill 文本写成强调抽象总结,fitness 仍是 code-pass | 一个**可能涌现**的分支 | 只有当失败模式聚类(§2.2(b))显示「缺抽象」跨题复现时才开;否则不预设 |
| **全局选择标准「泛化性」** | held-out/跨题源 probe 通过率作为**选择压力** | 挂在**所有**分支之上 | 不属于任何分支,是 cross-cutting 的选择目标 |

**结论(立住两点)**:
1. 「泛化性」作为**全局选择标准**挂到所有分支上(借 DarwinX preserve-and-extend + DIVE 三分 hold-out + Ctx2Skill Cross-Time Replay 静态 probe 回放——见 [[coevol-ctx2skill-anti-collapse-20260819]]),而非某个分支的专属目标。**理由**: 把泛化塞进单一分支 = 让一个分支同时背「交叉结构」和「选择压力」两个 EA 职责,会塌缩(= Phase C REFUTED 同类机制)。
2. 「abstraction 段」作为内容 facet **不预设**——只有涌现机制(§2.2(b))判定需要时才开分支。这与 v2「轴涌现而非预设」一致,也回应了用户「分支可更多可自动生成」的要求:abstraction 只是**候选**涌现 facet 之一,不是写死的第 4 头。

> 即:多头注意力里「泛化」不是某个 head,是整个 attention 要优化的目标;这里同理。

---

## 七、对现有代码的落地映射

现有 `rl/train/verl1/skill_evolver_math.py` 已有可复用的部件:

| 现有部件 | 行号 | 复用方式 |
|---------|------|---------|
| `generate_candidates` + `angle_hints` | 420, 462-467 | stance 旋转绑定到(涌现出的)facet 分支 |
| `STRUCTURE_ANGLES` | 537-544 | 已是 facet 级 stance 雏形,扩展为 facet-constrained 变异 |
| `generate_structure_candidates` | 561 | 结构重组逻辑可作 facet 抽取参考 |
| `consolidate` | 618 | 现是单 skill 精炼;需扩为 **2-parent facet 组合交叉** |
| `clean_skill` | 108 | 清洗 LLM 输出,直接复用 |
| `eval_on` + `bootstrap_ci_lower` | (评测) | 全局 fitness,直接复用,**零新评估器** |
| `gate_consolidate` | 656 | 防 facet 交叉损质,直接复用 |
| C 模型 A→B→C 流水线 | `custom_reward.py` | 全局 fitness 来源,直接复用 |

**新增核心部件**:
1. `facet_decompose(skill) -> dict[facet, text]`: slotted template 下按 slot 分解(方案 a)或 LLM 抽取(方案 b)
2. `facet_compose(parent_a_slots, parent_b_slots) -> skill`: 2-parent facet 组合交叉,LLM 合成连贯子代
3. `archive_tree`: 节点结构 {skill, facet_label, lineage, val_delta, test_delta, solved_facets}
4. `preserve_extend_check(child, parents)`: 契约校验 S(c)⊇∪S(p)
5. main 循环改造: 单 lineage → 树遍历(每代对每分支变异 + 跨分支交叉)

**预估工作量**: slotted template 迁移(1天)+ archive 树结构(2天)+ facet crossover(2天)+ main 循环改造(2天)+ 消融实验(3天)≈ 10 天。

---

## 八、消融实验设计(验证 contribution)

要证明"facet 分支+交叉"有效,需消融隔离每个新部件:

| 消融组 | 配置 | 假设 |
|--------|------|------|
| **baseline** | 现有 evolver(单 lineage keep-best) | 当前 fair-best 0.462 |
| **A: 仅树** | archive 树保留多 lineage,但分支无 facet 约束(随机分叉),无交叉 | 测"树"本身的贡献 |
| **B: 树+facet约束** | 分支 facet-edit 约束,但无交叉 | 测"facet 分支"的贡献 |
| **C: 树+facet+交叉(完整)** | 本设计完整版 | 主结果 |
| **D: 交叉消融** | 完整版但去掉交叉(每分支独立 hill-climbing) | 测"交叉"的贡献——facet crossover novelty |
| **E: 固定轴消融** | 完整版但 facet 轴**人工预设固定 4 个**(回退到 v1),关闭 §2.2(b) 涌现机制,分支数锁死 | 测"自动涌现 vs 人工预设"的贡献——**v2 核心增量(多头注意力式)** |

**成功判据**:
- C > baseline(fair-best 0.462)闭合 val→test gap,且 test 4-seed **≥ 判据线 0.5464**。
  ⚠️**判据口径钉对(2026-08-24 A2 power analysis, 见 memory/a2-power-analysis-20260824.md)**:
  判据线 **0.5464** 是 design 本条"显著超"的历史目标值(evolver-probe-split 基线), 与 Stage 0 4-seed
  重测出的 skillopt **0.5629** 是**两个不同源**, 不可互换。真实判据 gap = evolver(0.4953) → 0.5464 =
  **0.0511 = 5.88 SE**(σ=0.0174 pooled within-skill df=9, SE_4seed=0.0087), 达标极稳。0.0676
  (=0.4953 vs skillopt 实测 0.5629)只作"仍需闭合到 SkillOpt 上界"的旁证, **不进判据门槛**。涌现相对
  阈值(C>E)的效应量待终局 C 组跑完 E 组后据 2×SE≈0.017(方向)/3×SE≈0.026(显著)再定, 不作
  Δ=0.02 默认阈值; +0.34pp 级小尾巴 4-seed 测不动(需 ~70 seed), 只作方向记录。
- C > D(交叉贡献显著)→ 证明 facet crossover 是核心 novelty 之一
- C > B(facet 约束贡献显著)→ 证明 facet 分支化有效
- C > A(树结构贡献)→ 证明树状 archive 有效
- **C > E(涌现贡献显著)→ 证明「自动涌现 facet 轴」相对「人工预设轴」有效,即多头注意力式 specialization 涌现成立**。这是 v2 区别于 Diverse Prompts(手工轴)的关键判据;若 C ≯ E,说明自动涌现未带来增益,该退回固定轴以降工程复杂度(诚实留退路)

---

## 九、风险与未决

1. **slotted template 迁移成本**(仅 Line B 适用,Stage 0 不触及): 现有 402 个 BAgent 代码 skill 迁移到具名段,可能丢信息。需一次性 LLM 转写 + 人工抽样校验。**Line A 已免**:math skill 自带具名段结构(`## ` slotted,真 S₀ 4 段 + 涌现的第 5 段,见 §四-①/§九-7c)。
2. **facet 边界模糊**: 某些 skill 内容跨 facet(如一段既是 process 又是 self_debug)。方案 a 下需明确定义 slot 归属规则
3. **交叉的 LLM 噪声**: facet 组合靠 LLM 合成,可能引入不一致。需 `clean_skill` + `gate_consolidate` 双重护栏
4. **计算成本**(2026-08-21 修订,以实测数据替换原空估):
   - **绑定约束 = 28 RPM 共享上限**(≤30 硬约束,`skillopt/model/qwen_backend.py` token-bucket + 429 指数退避,见 memory/api-concurrency-deepseek-20260716)。**API 时间 ≠ wall-clock**:40 并发(`skill_evolver_math.py:216` ThreadPoolExecutor)被限速器削回稳态 ≈28 调用/分钟;多代持续压力下 1.0–1.2x 系数,夜间不争用时才逼近纯 API 下限。每晚次额外 eval_on(分支/基线/gate-reval/s0 补算)都抢同一 28RPM 池,与 B-gen/C-judge/他实验争用会互相拖慢——**排程按下限估,别按乐观系数估**。
   - **"零新评估器"≠"零新调用"**(重要口径修正):复用 `eval_on` 省的是**写评估代码的钱**,不是**跑评估的 API 钱**。调用量随分支数线性翻倍:
     | 搜索结构 | 每代评估 skill 数 | 每 gen B 调用(probe 29×k3=87) |
     |---|---|---|
     | 单 lineage(现状) | K=4 | 348 |
     | Facet-tree Stage 0(固定 5 轴) | B=5×C=3 + ~4 交叉 = **~19** | **~1650** |
   - **Stage 0 实测下限**(group C, 固定 5 轴, 3 代): ~1650 B-调用/gen × 3 + ~45min test(4-seed k3) ≈ **3.5–4h 纯 API 下限**。**⚠️ 实测速率锚定(2026-08-24 A5, 见 memory/a2-power-analysis-20260824 + phase2_issues#851)**:admission smoke 2 代跑 `evolver_log.json` 实测 `secs` 新 run=2933s(≈49min)/旧 run=3414s(≈57min), 均值 **~53min/gen**; 每代 ~1650 B-调用 ÷ 28 RPM ≈ **59min/gen 纯 API 下限** —— 实测 49~57min 与纯 API 下限吻合, 说明 **28 RPM 确为唯一瓶颈(加 GPU 不提速), ~53min/gen 是硬数非估算**。据此重算五组+终评: 每组(3~6 代)× 53min + test ~45min ≈ **每代 ~53min 是基本单元**, 完整消融 5×(~3 代×53min + test 45min) ≈ **~17h**(非原估 25h, 因固定 5 轴里只有 3 轴实跑, 调用数 1650→~990, 见下口径修正)。**口径注意**: L251 的 ~1650 B-调用/gen 是 group C「固定 5 轴」估算; 当前 smoke 实际只跑 **3 轴**(Option Comparison / Theorem-Level Precision / Hypotheses, Final Answer 冻结不产分支)→ 每 gen B 调用 ≈ 990(=3轴×~330), 故 ~53min/gen 对应 3 轴, 终局 group C 开满 5 轴实际 ~1650 → **~88min/gen**。**算力账必须按实跑轴数折算, 不能照抄 5 轴基数。**
   - **scope 杠杆**: B=3×C=2 → ~8 skill/gen → ~2.5h。
   - **完整消融(§8 五组)**: 5 × (演化 ~3.5h + test ~1.5h 含 s0 复评) ≈ **~25h 纯 API 下限**,摊 3–4 夜可行,但仅无人争用时逼近。
5. **涌现轴不稳定风险**(v2 新增): 自动开/并分支若依赖每代失败模式聚类,可能**震荡**——某失败簇偶现就开分支、下一代消失又合并,分支数与 facet 集抖动。需加最小持续代数门控(某簇连续 K 代复现才开分支、连续 K 代无 delta 才合并),K 初设 3。这与用户「多头注意力自己训练出不同语义层级」的意图一致——但真实 attention 的头数是训完才稳的,运行中需先容忍抖动再加门控,否则退化为随机开合
6. **slotted template 段名随涌现增删**: v2 下 facet slot 名可由 §2.2(b) 涌现机制新增/重命名,交叉(§2.5 取 slot)要能处理两父本 slot 集不交的情形(slot 名对齐/缺 slot 退化策略)。这比 v1 固定 slot 集复杂,是 (a) 方案相比 (b) LLM 抽取的新增成本
7. **两个设计点(2026-08-21 定稿)**:
   - **(a) skill 表示 = 方案 a slotted template**:**已定**。Line A 上是现状(真 S₀ 已 4 段 slotted),无需迁移/无需 PoC;方案 b(LLM 抽取)弃——噪声会淹灭交叉信号、破坏因果归因(交叉无增益时分不清"交叉本身无效"还是"抽取噪声")。代价 §九-1/6(slot 涌现增删时两父本 slot 集不交的对齐/退化策略)仍需实现,但比噪声可控。
   - **(b) 泛化 probe 集配比 = 沿用 probe-split 现有 29 题 seed=42,锁死不再动**:**已定**。理由:Facet-Tree 唯一要证的是**破 0.4654 天花板**,0.4654 只在 probe-29 seed=42 这杆秤上可读;换秤则基线不可比。配比细节(原"跨题源/跨难度"歧义已消解):
     - split 数据是 `livemathmaticianbench_1-3-6_seed42`(train18/val53/test106),**单题源**(LiveMathematicianBench),故"跨题源"不适用——只有"跨月批次"(数据 `month` 字段:202511–202602 四批,test 全覆盖,train/val 已跨四批,无需额外分层)。
     - split 自带 `theorem_type` 结构化元字段——⚠️**字段值是字符串 list(一题可多类,非单值)**,实读 `livemathematicianbench_1-3-6_seed42/test/items.json`(n=106)值域 13 类,按主成分计数:Existence 35 / Implication 34 / `Inequality or Bound` 31 / Universal 30 / `Existential–Universal` 18 / `Biconditional or Equivalence` 14 / `Classification or Bijection` 14 / `Asymptotic or Limit` 8 / `Universal–Existential` 7 / Nonexistence 4 / unknown 3 / Uniqueness 2 / Equivalence 2。evolver `skill_evolver_math.py:357/411` 已 `it.get('theorem_type','?')` 读用。probe=29 取自 select=71 的 seed=42 不相交划分,**保留原 `theorem_type` 分布**(与 test 同源同分布),不额外分层——分层会偏离 0.4654 基线条件。`theorem_type` 不做 probe 分层,而是**喂 §2.2(b) 涌现机制**做 facet 分簇信号(现 `classify_failure_type` 已隐式用之,本设计显式化)。⚠️**开工清单 step1 格式核对须顺带验字段名+值域+list 结构**(避免 `facet_decompose` 按错字段名取 `theorem_type` 致涌现分簇静默失效)。
     - probe 即"全局选择标准「泛化性」"的载体(§六):挂在所有分支之上作 selection pressure,不进任何单一分支。
   - **(c) Stage 0 起点 = 真 S₀(4 段 `initial.md`),非 V2 默认 5 段版**:**已定**(2026-08-21)。实读核实:`skill_evolver_math.py:56` `V2_SKILL_DEFAULT`(5 段,含 `Common Failure Modes`)是 `--init_skill` 默认,但真 S₀ `initial.md` 仅 **4 段**(Option Comparison/Theorem-Level Precision/Hypotheses/Final Answer);`Common Failure Modes` 是进化中后期涌现的第 5 段。起点钉死 4 段,理由三层(不只是「公平」):
     - **涌现 facet 的内置验证(分阶段,问题1定稿选项C)**:从 4 段起步,`Common Failure Modes` 必须靠 §2.2(b) 涌现机制**从零长出**。⚠️**兑现点 = 终局 C 组(多代、开涌现、K=3 门控能触发),不是 Stage 0**——Stage 0 只跑 2 代 < K=3 门控且用 `--no_emergence`,涌现机制物理上不触发,故 Stage 0 判据**只一条**(test 4-seed > 0.4654,只答树+交叉破不破天花板),`failure_modes` 在 Stage 0 是空 slot 占位(既不预装也不涌现)。4 段起点的内证价值在终局 C 组兑现:那里多代能触发 K=3,`failure_modes` 是否从零长出 = 第二条判据(「涌现 facet」从无到有即 novelty 不辩自明),同时也是 C>E 消融的对照判据。见 [docs/facet_tree_stage0_gonogo.md](facet_tree_stage0_gonogo.md) §四。
     - **与 SkillOpt 0.5464 的可比性(决定性论据)**:SkillOpt-best 是 `S0 起(SkillOpt自有)`、标公平(memory/baseline-fairness-audit-20260817);我们主链 `S0→v2→evolver` 全从 S0 起亦公平;只有显式种 `skill_v0002.md`(=SkillOpt-best)的 evo_seed/Run F 标 `❌ 嵌套已排除`。**若从 5 段 V2 版起步 = 从 v2 继承预装 `failure_modes`,而 SkillOpt 是从同一 4 段 S₀ 自力长出 0.5464——起点不再对齐,对比失公平,重蹈 Run F 嵌套病。** 故「真 S₀ 起步」同时满足因果归因 + 可比性,双效。
     - **多元性归一(2026-08-25 修订, 回应「钉死单 S₀ = 自断双臂」质疑)**: 原 v3 以「因果可归因」为由主张钉死单 S₀。这一理由**过时**——因果归因可靠**统计显著性**(4-seed 均值 ± SE,见 [[a2-power-analysis-20260824]])判定, 不必靠「单 lineage 逐行可追溯」, 那个代价花得不值。~~原「因果可归因」条~~ 已被本条取代。真正的公平约束只管「起点是不是 S₀」, 管不到「S₀ 能否多样初始化」——**见 §八·2 / §十一 新增的「S₀ 多样性初始化」设计点**: 从单 S₀ 起步是单 lineage 时代遗产, 进入真 GA 后应改为「S₀ → 多样性扰动 → 异质初始种群 → 进化」, 这不违反公平性(每个个体祖先都是同一 S₀), 反而补上 GA 的基因多样性短板(文献支持: EvoPrompt 手工+LLM 混合初始种群 / PromptBreeder fitness sharing / DIVE bootstrap 子集异质起步)。
     - **落地**:进 plan mode 拆 Stage 0 时显式传 `--init_skill=<真S0 initial.md 路径>`,不用默认 `V2_SKILL_DEFAULT`;5 段 V2 版保留作 §八消融的一个自然对照臂(量化「预装第 5 段」值多少 delta)。
     - split 数据是 `livemathematicianbench_1-3-6_seed42`(train18/val53/test106),**单题源**(LiveMathematicianBench),故"跨题源"不适用——只有"跨月批次"(数据 `month` 字段:202511–202602 四批,test 全覆盖,train/val 已跨四批,无需额外分层)。
     - split 自带 `theorem_type` 结构化元字段——⚠️**字段值是字符串 list(一题可多类,非单值)**,实读 `livemathematicianbench_1-3-6_seed42/test/items.json`(n=106)值域 13 类,按主成分计数:Existence 35 / Implication 34 / `Inequality or Bound` 31 / Universal 30 / `Existential–Universal` 18 / `Biconditional or Equivalence` 14 / `Classification or Bijection` 14 / `Asymptotic or Limit` 8 / `Universal–Existential` 7 / Nonexistence 4 / unknown 3 / Uniqueness 2 / Equivalence 2。evolver `skill_evolver_math.py:357/411` 已 `it.get('theorem_type','?')` 读用。probe=29 取自 select=71 的 seed=42 不相交划分,**保留原 `theorem_type` 分布**(与 test 同源同分布),不额外分层——分层会偏离 0.4654 基线条件。`theorem_type` 不做 probe 分层,而是**喂 §2.2(b) 涌现机制**做 facet 分簇信号(现 `classify_failure_type` 已隐式用之,本设计显式化)。⚠️**开工清单 step1 格式核对须顺带验字段名+值域+list 结构**(避免 `facet_decompose` 按错字段名取 `theorem_type` 致涌现分簇静默失效)。
     - probe 即"全局选择标准「泛化性」"的载体(§六):挂在所有分支之上作 selection pressure,不进任何单一分支。
8. **C-judge 走独立模型账户——已定(2026-08-21)**:**C-model 换 `deepseek-v4-pro`(pro 专用 key 账户),物理隔离于 B 的 V3.2 池**。背景:原 `judge/qwen_235B.py` 的 C-model 直打 `aimp-deepseek` 端点 + `/model/DeepSeek-V3.2`(行 260/262),与 B(evolver 的 `QWEN_CHAT`)**同一端点同一模型同一 28-RPM 账户池**——B+C 互不感知合并打同一池,叠加流量触 30RPM 硬上限 → 429 污染指标(同 memory/api-concurrency-deepseek-20260716 事故)。`/data/zhuldz/model-launch/` 提供三个**账户隔离**的可用模型(glm-5.2 / deepseek-v4-flash / deepseek-v4-pro,各有独立 key,ccr 账户层隔离、消耗/配额/日志分开)。**决策 B=选 pro**:C 调用改指向 pro 端点+pro key+`deepseek-v4-pro`,B 仍走 V3.2,二者物理隔离零限流冲突。落地:`judge/qwen_235B.py` 的 `API_URL`/`AUTH_TOKEN`/`MODEL_C` 三参数改指向 pro(`/data/zhuldz/model-launch/start-pro.sh` 的配置)。**备选(保留不用)**:① C-model 套同一 `_TokenBucket`(共享额度,仍抢池);② C-judge 退纯本地单测 pass_rate(削掉 C 调用,仅终评用)——若 pro 账户额度不足或判分稳定性差再退此二选一。

   > ⚠️**适用边界订正(2026-08-21 实读,见 Facet-Tree 问题4)**:C-judge=pro 的「保留 LLM 评判维度」语义**仅在 C 被调用时成立**, 即 **GRPO 训练链路**(`c_interface.py:compute_reward` 的 70/30 合成奖励)。**Stage 0 跑的是 `skill_evolver_math.py` 演化链路**, 其 fitness = EM pass_rate(`eval_on`→`run_one`→`math_evaluate`, **不调 `qwen_235B.py`**)——故**改 `judge/qwen_235B.py` 三参数对 Stage 0 的 fitness 是空操作**(Stage 0 判据 test 4-seed > 0.4654 走 EM, 不受 pro 账户影响)。C-judge=pro 真正生效 = Stage 0 产出 skill 后**回灌 GRPO 训练**(A 模型奖励走 `c_interface`),那是 Stage 0 之后的前置, **非 Stage 0 本身阻塞项**。gonogo §七 step2 据此**降级**:Stage 0 不依赖 pro 账户就绪。

---

## 十一、v4 两个新增设计点 (2026-08-25, 回应两条核心质疑)

两条质疑其实指向同一处(当前实现还带着「单 lineage + 人工模板」残余, 没真正进入「种群随机化 + 轴涌现」的 GA 形态)。分别落成设计点如下。

### 11.1 设计点一: S₀ 多样性初始化(解「钉死单 S₀ = 自断双臂」)

**质疑**: 钉死单 S₀ 让基因池太浅, 违背「多样性是 GA 有效性源泉」; 且 skillopt 也从 S₀ 起步, 故把 S₀ 做多样性处理后再进 GA 完全不违反公平。

**结论**: 质疑成立。钉死单 S₀ 是单 lineage 时代的遗产, 进入真 GA 后应改为「S₀ → 多样性扰动 → 异质初始种群 → 进化」。

**公平性边界(关键)**: 只要**起点都是 S₀**, 就公平。约束管的是「起点是不是 S₀」, 不是「起点那一刻的种群规模」。故允许把 S₀ 复制 N 份, 各自做无结构破坏的随机扰动, 得 N 个**同源且多样**的初始个体一起进种群——每个个体祖先都是同一 S₀, 无人比 skillopt 提前占便宜。

**文献支持(§一/调研):**
- **EvoPrompt**: 初始种群混「手工 prompt + LLM 生成」, 非单模板起步。
- **PromptBreeder**: fitness sharing(BERT 嵌入相似度 >阈值去重/惩罚)防止收敛局部最优——多样性维持是经典 GA 标配, 当前 evolver 完全没有。
- **DIVE**: 多种群从 bootstrap 子集独立演化, 避免单 trajectory 过拟合。

**落地候选(待选定, 按公平+便宜排):**
| 扰动算子 | 做法 | 是否破坏 facet 结构 | 公平性 |
|---|---|---|---|
| 温度改述 | 同段内 LLM 改述措辞(温度 1.0–2.0) | 否(仍 4 段) | ✅ |
| 段落顺序微调 | 非冻结段调换(不在 `facet_decompose` 内重排则语义风险) | 轻微 | ✅ |
| 增删非结构子弹点 | 段内增删不影响 `## ` 头的要点 | 否 | ✅ |
| PromptBreeder fitness sharing | BERT 相似度惩罚同质个体 | — | ✅ |

> ⚠️ 是否纳入终局 C 组、初始种群大小 N 取多少(文献 N=4~50), 是下一步要定的, 本条只钉「该做」不锁参数。

### 11.2 设计点二: 涌现 vs 预设的判据清单(解「现在不就是人工预设吗」)

**质疑**: 当前 facet 轴是人按 `## ` 段切好、护栏锁死自由度 → 是预设不是涌现。

**结论**: 对, **现在就是预设**。Stage 0 用 `--no_emergence` 跑, 4 轴人工预设, 涌现未兑现。这不是 bug, 是阶段性问题, 但它是 novelty 的最大风险点。

**现状态三处预设(一层比一层实):**
1. facet 段结构(4 个 `## ` 标题)是人定的(继承 S₀)。
2. 每段语义角色(选项比对/定理精度/假设/最终答案)是预设的。
3. 每段可编辑范围由 `facet_edit_guard` 护栏硬编码(自由度结构人手指定)。

**「真涌现」的判定标准(须同时满足):**
- 轴**不是预设**的(不是人先说「要有这 3 个轴」, 而是算法自己发现「分这几个关注点更有效」);
- 轴的数量/命名/边界**是长出来的**(可能长出预设之外的轴、合并/淘汰轴)。

**可验证的涌现观测点(现状已有其一, 补足成清单):**
| 观测点 | 什么算涌现发生 | 状态 |
|---|---|---|
| (1) `Common Failure Modes` 第 5 段从零长出且有因果贡献 | 起点 4 段不预装, 靠 §2.2(b) 聚类长出第 5 段 | 终局 C 组兑现(Stage 0 已留空 slot 占位) |
| (2) 弱 facet 被合并/淘汰 | Theorem-Level Precision / Hypotheses(+0.026 弱主轴)在涌现阶段被自动合并 | 待 C 组观察 |
| (3) 长出预设之外的轴 | 新轴名不在 4 预设段内, 由 `theorem_type`/失败模式聚类逼出 | 待 C 组观察 |
| (4) C>E(涌现轴 > 固定轴) | 涌现版显著优于固定 4 轴消融臂(§八消融 E 组) | 待 C 组 |

> **诚实留退路(§八已设): 若 C ≯ E, 说明自动涌现无增益, 应退回固定轴降复杂度。** 本清单把这扇门显式写出来——「涌现」是待验证的**主张**, 不是已兑现的**结果**。审稿人问「facet 轴不是预设的吗」, 正解是: 起点是预设(公平+可归因), 涌现是终局 C 组要去验证的增量, 目前未兑现——诚实比自夸更扛问。

---

## 十二、参考文献(均经 EA 调研全文核实)

- **DIVE** (Xiong et al. 2026, arXiv:2608.12486) — 多种群 bootstrap + 2-parent skill 重组 + UCB + 三分。明确批评 "greedy single-candidate refinement brittle"。**最直接对位**。
- **DarwinX** (Zhang et al. 2026, arXiv:2608.07545) — archive 树保留输家 + 多父本 merge + preserve-and-extend 契约。§5.1「种群吸收 in-loop proxy 偏差」直诊 val→test。
- **EvoAgent** (Yuan et al. 2024, arXiv:2406.14228) — 标准 EA,多基因配置 crossover。最接近 facet 思想,但无 specialist 分支无 train/test 分离。
- **PromptBreeder** (Fernando/DeepMind 2023, arXiv:2309.16797) — fitness sharing 防局部最优,9 变异算子。
- **EvoPrompt** (Guo et al. 2023, arXiv:2309.08532) — GA+DE 教科书式真 GA。
- **GSE** (Yang et al. 2026, arXiv:2608.06153) — cluster consolidation 抽象通用能力,代码场景防过拟合。
- **SePO** (Tao et al. 2026, arXiv:2606.04465) — archive-based 单 lineage,用户形态最接近(但本设计要超越它)。

完整 9 篇逐篇细节见 [docs/ea-skill-evolution-survey-20260819.md](ea-skill-evolution-survey-20260819.md)。

---

**核实声明**: 本文档引用的 9 篇论文方法均经 EA 调研全文核实(含算子定义、附录 prompt 模板、泛化协议),非摘要层面臆测。现有代码行号引用经 `skill_evolver_math.py` 实读确认。
