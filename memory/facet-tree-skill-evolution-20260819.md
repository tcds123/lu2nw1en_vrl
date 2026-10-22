---
name: facet-tree-skill-evolution-20260819
description: 新实验方向——树状+facet交叉的真GA skill进化设计v3。站在DIVE/DarwinX/EvoAgent肩上换轴(自动涌现的facet轴,specialization训练涌现类比多头注意力),用户思路"按不同侧重点树状进化+分支交叉"的落地方案
metadata:
  type: project
---

## Facet-Tree 真遗传算法 skill 进化方向 (2026-08-19, v3 定稿 2026-08-21: 进化对象锁 Line A / 两设计点已定 / 成本模型改实测)

### 🏆 Stage 0 go/no-go 终局 = GO (2026-08-21)

2 代树进化 test=106 4-seed mean = **0.4953 > 判据 0.4654** (Δvs s0=+0.2075), 破天花板 → 机制值得开多代。
skillopt 0.5629 仍高 0.0676 = val→test gap 未闭合, 归终局 C 组(开涌现+多代 K=3 门控)兑现。
详见 [[facet-tree-stage0-result-20260821]]。

**起因**: 用户在 EA 调研([[ea-in-skill-evolution-survey]] 已删重复版, 权威版见 `docs/ea-skill-evolution-survey-20260819.md`) 后提新思路:"让 skill 通过不同 baseline 或不同侧重点要求,像树状一样进化"。追问确认:(1) 分支间**要交叉**(落真 GA 档);(2) 侧重点举例(tool调用选择/顺序/结果 / 过程步骤详细程度 / 步骤结果与重复机制 / 泛化总结性)。**v2 升级**: 用户进一步要求 facet 轴**自动涌现而非人工预设**——"分支只是举例,可设计更多或自动生成,就和多头注意力一样,自己就能训练出不同的语义层级"。

**设计文档**: `docs/facet_tree_skill_evolution_design.md`(已落盘, v3 定稿 2026-08-21: 进化对象锁 Line A / 起点钉真 S₀ 4 段 / C-judge 换 pro 独立账户)。

### 核心结构

站在三件已有 + 一个增量(经二轮文献核实):
- 2-parent 重组 ← DIVE(2608.12486)
- archive 树保留输家/specialist + MAP-Elites illumination ← DarwinX(2608.07545)
- 多基因配置 crossover ← EvoAgent(2406.14228)
- **增量(9篇 + QD/MAP-Elites 5篇均无,见下)**: **自动涌现的 facet 轴**(类比多头注意力的 specialization 是训练涌现非预设 head) + **facet 组合式交叉**(取父本1的A段+父本3的C段合成子代)用在 skill 文本进化

### 增量 novelty 的诚实边界(二轮核实,防重蹈漏 DIVE 覆辙)

用户提"像多头注意力自己训练出不同语义层级"后,核验 QD/MAP-Elites 是否已占此空间:
- **Diverse Prompts**(2504.14367): MAP-Elites 用到 prompt,但轴**手工**(#shots/reasoning depth)——正是要超越的点
- **AutoQD**(2506.05634)/ **VQ-Elites**(2504.08057): 自动/无监督 behavior descriptor 已成熟,但**在机器人轨迹域,无 crossover**
- "skill 文本进化 + 自动涌现 facet 轴 + facet 组合交叉"三者交叉**无人占据**——这才是真实增量,**必须窄讲**,不能宽泛说"自动分支/MAP-Elites"
- 定位: incremental-but-real。站在 DIVE/DarwinX/EvoAgent 肩上把 specialization 从「任务覆盖」换成「自动涌现的内容 facet」,crossover 从「整 skill 重组/加性 merge」换成「facet 模块化组合」,借 QD 的 archive-illumination 让 facet 轴自适应涌现。**不是吹新范式,是换轴 + 跨域移植**

### facet 涌现机制(§2.2 核心)

- **(a) 冷启动 seed facet(非强制)**: tool_guidance / process / self_debug 三 seed 仅第一代注入,之后可分裂/淘汰
- **(b) 涌现**: 每代失败模式语义聚类(复用 `skill_evolver_math.py` `classify_failure_type` 行130 + `compute_failure_distribution` 行169)→ 跨题复现且未覆盖簇自动开新分支(分支数 adaptive,非固定)→ 长期无 delta 贡献分支自动合并。QD "learned behavior descriptor" 向 skill 文本域移植
- 护栏(§九新增): 最小持续代数门控 K=3(连续 K 代复现才开/合并),防震荡退化为随机开合

### §六「泛化性」歧义 v2 已消解

v1 固定4 facet 时「泛化性/总结性」有读法A(内容facet)/读法B(fitness目标)歧义。v2 下分支涌现故「第4个分支」伪问题消失。立两点:(1)「泛化性」作**全局选择标准**挂所有分支(借 DarwinX preserve-and-extend + DIVE 三分 + Ctx2Skill Cross-Time Replay 静态 probe 回放,见 [[coevol-ctx2skill-anti-collapse-20260819]]),不塞进单一分支(否则塌缩=Phase C REFUTED 同类);(2) abstraction 段**不预设**,仅涌现机制判定需要时才开。类比:多头注意力里「泛化」不是某个 head,是整个 attention 要优化的目标。

### 与现有代码的契合(已实读 `skill_evolver_math.py` 确认)

- `generate_candidates` 的 `angle_hints`(行462-467,4角度)和 `STRUCTURE_ANGLES`(行537-544,6 stance)**已是 facet stance 雏形**——但现是平铺变异选单winner,无树无交叉
- 选择 = keep-best 单 lineage(main循环argmax delta);`consolidate`(行618)是单skill精炼非多skill合并
- 复用件: `clean_skill`/`eval_on`+`bootstrap_ci_lower`/`gate_consolidate`/C模型A→B→C 全局fitness(零新评估器)
- 新增: `facet_decompose`(slotted template按slot分解)+`facet_compose`(2-parent facet组合交叉)+`archive_tree`+`preserve_extend_check`+main循环改单lineage→树遍历。预估~10天

### 两个设计点(2026-08-21 已定稿)

1. **skill表示 = 方案 a slotted template (已锁)**: Line A 全局 math skill 是 slotted template, `facet_decompose` = 按 `^## ` 切段即可, **零迁移成本**。方案 b(LLM 语义抽取)弃用——噪声会淹灭交叉信号。v1 设计的 BAgent seed facet (`tool_guidance`/`process`/`self_debug`) **已弃**——那是 BAgent 代码 skill 概念, 与 math skill 自身段不对位; 用之 facet_decompose 会直接失败。**起点段数已订正(§九-7c, 决策A)**: 真 S₀(`initial.md`)只含 **4 冷启动段**(`## Option Comparison` / `## Theorem-Level Precision` / `## Hypotheses` / `## Final Answer`); `## Common Failure Modes` 是进化中后期才涌现的第 5 段(晚期 5 段版=V2 默认才补), 不是 S₀ 固有。**Stage 0 起点钉真 S₀ 4 段**, `failure_modes` 作「应涌现的检验目标」从零长出(=涌现 facet 的内置验证, 非预装)——详见 [[facet-tree-s0-startpoint-20260821]]; 晚期 5 段 V2 版保留作 §八消融对照臂。
2. **probe 集配比 = 沿用 probe-split 29 题 seed=42 锁死 (已锁)**: 这是唯一可读 0.4654 基线的秤, 换秤则基线不可比。歧义消解: LiveMathematicianBench 单题源故 "跨题源" 不适用; 有 `month`(源批次, 202511–202602 四批, test 全覆盖) 与 `theorem_type`(结构化 list) 两元数据 → `theorem_type` 喂 §2.2(b) 涌现机制做 facet 分簇信号(**不做 probe 分层**, 免偏离 0.4654 基线); test 全覆盖四 month 批次已天然跨源, 无需额外配比。

### ⚠️ 必挂护栏(防 Phase C REFUTED 同类塌缩)

若分支同fitness无diversity护栏→塌缩回单目标→退化成并行单lineage(=Phase C test 0.37 REFUTED同类机制)。至少挂一:
- **facet-edit约束**(分支只许动对应facet slot,物理防塌缩,最便宜,推荐)
- fitness sharing(PromptBreeder,BERT相似度惩罚)
- preserve-and-extend契约(DarwinX,子代须保留父代已解侧面)

泛化护栏至少挂一: 三分evolution/val/test(DIVE,test只终评) / 种群吸收偏差(DarwinX,archive保留多lineage含输家) / cluster consolidation(GSE)。

### 消融设计(验证contribution,含 v2 新增 E 组)

baseline(现evolver) / A仅树无facet约束 / B树+facet约束无交叉 / C完整版 / D去交叉 / **E固定轴消融(回退v1人工预设4 facet+锁死分支数,关涌现)**。核心判据:
- C超fair-best 0.462且test 4-seed显著超SkillOpt 0.5464闭合val→test gap
- **C>E(涌现贡献显著)→ 证明「自动涌现facet轴」相对「人工预设轴」有效,即多头注意力式specialization涌现成立**。这是 v2 区别于 Diverse Prompts(手工轴)的关键判据;若 C≯E 该退回固定轴降复杂度(诚实留退路)
- C>D(交叉贡献)+C>B(facet分支有效)+C>A(树有效)

**Why**: Phase C REFUTED后三次卡val→test(Task3内容0.4654/Task4输入0.3632/PhaseC结构0.3703越来越弱),EA调研诊断瓶颈在单lineage搜索结构本身不泛化而非优化维度。本设计同时换搜索结构(单lineage→树)+换fitness维度结构(单一→涌现facet分支),直击痛点。v2 涌现升级直接回应用户「像多头注意力自己训练出不同语义层级」的要求。

**How to apply**: **五项决策全定稿 (2026-08-21)**——① skill 表示=方案 a slotted template, `facet_decompose`=按 `^## ` 切段;原 v1 的 `tool_guidance`/`process`/`self_debug` seed facet 是 BAgent 概念,与 math skill 段不对位,**已弃**,Stage 0 facet 轴用 math skill 自身冷启动段。② 泛化 probe=沿用 probe-split 29 题 seed=42 锁死(破 0.4654 天花板唯一可读的秤,换秤则基线不可比);split 自带 `theorem_type`+`month` 元字段,前者喂涌现机制做 facet 分簇信号(不做 probe 分层,免偏离基线),test 全覆盖四 month 批次已天然跨源。③ **起点=真 S₀ 4 段 `initial.md`(决策A, §九-7c)**: 非 V2 默认 5 段版; `## Common Failure Modes` 是涌现的第 5 段, 起点空、靠 §2.2(b) 从零长出作涌现 facet 内置验证; 显式传 `--init_skill=<真S0 initial.md 路径>`, 不用默认 `V2_SKILL_DEFAULT`(5 段); 晚期 5 段 V2 版保留作 §八消融对照臂。详见 [[facet-tree-s0-startpoint-20260821]]。④ **C-judge 走 deepseek-v4-pro 独立账户(决策B, §九-8)**: C-model 换 pro(`/data/zhuldz/model-launch/start-pro.sh` 配置: msk-2a75… 账户, /data/ccr-deepseek:3456, 按 model 名路由账户隔离), 物理隔离于 B 的 V3.2 池(28RPM token-bucket), 解 B+C 同端点打 30RPM→429 污染风险(原①限流器/②退纯单测 pass_rate 降级为 pro 额度不足时备选)。⑤ **涌现分阶段(决策C, 问题1, §九-7c/§四)**: Stage 0 **只一条判据**(test 4-seed > 0.4654, 关涌现, 只答「树+交叉破不破天花板」); `failure_modes` 在 Stage 0 是**空 slot 占位**(既不预装也不涌现)。**理由**: §2.2(b) 涌现机制有 K=3 最小持续代数门控(连续 K 代复现才开/合并分支), 而 Stage 0 只跑 **2 代**, 2 < K=3 门控 → 涌现**物理上不触发**, 故「failure_modes 是否长出」在 Stage 0 是恒空的伪测试, 不可作判据。涌现验证(failure_modes 从零长出)的**兑现点挪到终局 C 组**(多代、开涌现、K=3 门控能触发): 判据二 = failure_modes 是否从零长出(novelty 不辩自明), 判据三 = C > E(涌现轴 > 固定轴)。**Stage 0 = 终局 C 组的前置闸**(过则证明机制值得开多代, 不过则单 lineage 已证伪树方向)。详见 [[facet-tree-s0-startpoint-20260821]] 与 [docs/facet_tree_stage0_gonogo.md](../docs/facet_tree_stage0_gonogo.md) §四。**进化对象锁定 Line A 全局 math skill**(非 BAgent 402 条,那条线 skill 形态不同、无现成 eval_on/probe 对位,推迟)。Stage 0=group C 固定轴先验 go/no-go(~3h 纯 API 下限,判据 test 4-seed>0.4654),详见 [docs/facet_tree_stage0_gonogo.md](../docs/facet_tree_stage0_gonogo.md) 与设计 v3 [docs/facet_tree_skill_evolution_design.md](../docs/facet_tree_skill_evolution_design.md)。**⚠️ Stage 0 工程前置**:C-judge 走 pro 独立账户(决策B 已定, §九-8); 优先实现 facet-edit 约束护栏(最便宜防塌缩), 实验前必须挂至少一个 diversity 护栏 + 一个泛化护栏, 否则重蹈 Phase C 覆辙。与 [[phase-c-structure-evolution-result-20260819]] 同病(单lineage val→test脱节),本设计是其后继方向。
