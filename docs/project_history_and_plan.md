# lunwen 项目历史与未来规划（汇报版）

> 更新日期：2026-08-05 ｜
> 一句话定位：**Training-Free GRPO + Prompt/Skill 优化融合方案**，核心流水线 A(生成 prompt/skill) -> B(生成代码/解答) -> C(评判打分) -> 奖励驱动 A 更新。研究方向已从「参数空间优化」转向「上下文空间进化」。
>
> **本文档读法**：第 0–6 节为**汇报用的现状与设计**（自顶向下读完即可汇报）；附录 A 为**按时间线保留的详细过程记录**（含每次反转的根因与 bug 修复，供溯源）；附录 B 为关键文件索引。

---

## 0. 项目骨架

```
A (Qwen3-4B, GRPO训练) -> 系统提示词/Skill -> B (DeepSeek-V3.2 API, 冻结) -> 代码/解答 -> C (DeepSeek-V4-Flash, 冻结) -> 评分+单测 -> GRPO更新A
```

- A 模型：Qwen3-4B-Instruct-2507，verl 框架，8 GPU；输入 问题+参考解+ICL -> 输出系统提示词/Skill
- B 模型：DeepSeek-V3.2 API 单次生成（Phase 2 升级为 BAgent 自进化框架；当前主线数学域用单次生成 + skill 注入）
- C 模型：DeepSeek-V4-Flash 评判 + 单测执行；奖励 = 70% 单测/答案通过率 + 30% LLM 评判

---

## 1. 汇报速览（2026-08-05）

**一句话现状**：在免训练设定下，我们提出「失败模式蒸馏 + 追加式合成 + evolver 自进化迭代」的 skill 优化方法，在 LiveMathematicianBench 上相对初始 skill S₀ 取得 **+16.1pp**（4-seed，t=9.02，高度显著）的主结果；并通过机制消融定位了主导机制（种群选择 + 失败驱动 patch）。AIME 跨形态泛化为负，作为方法的泛化边界诚实报告。

| 维度 | 结果 | 状态 |
|------|------|------|
| **主结果** evolver 2-iter vs S₀（livemath test=106） | **+16.1pp**，t=9.02，95%CI[+0.126,+0.196]，4-seed 全正 | ✅ 坐实 |
| evolver vs v2（净增量） | +9.8pp，t=5.84，CI[+0.065,+0.130] | ✅ 坐实 |
| v2 vs S₀ | +6.4pp（4-seed，t=4.58）/ +6.6pp（2-seed） | ✅ 坐实 |
| 离线 trace2skill-math 基础池 | +2.4pp | ✅ modest 正 |
| **机制消融**（Task1） | M1 种群选择 +10.5pp(p=1.9e-7)、M2 失败patch +7.8pp(p=9.2e-5) 高度显著；M3/M4 不显著 | ✅ 故事干净 |
| **AIME 两条负证据** | ①跨形态(livemath skill->AIME24) +1.9pp(t=0.70)不显著；②同域从零(AIME24训)门控拒全部候选,best==S₀ | ✅ 核心创新验证(诚实拒非泛化)+算法改进切入点 |
| evo_seed vs 强基线 SkillOpt | 16-seed +0.59pp(p=0.42) 不显著 | 🟡 parity（未超） |
| 10-iter vs 2-iter | -2.4pp 不显著，2-iter 是甜点 | ✅ 迭代深度消融 |
| **跨 agent（Task3）** single-call vs ReAct | 冒烟通过，路径 bug 已修，全量 eval 待启动 | ⏳ 待跑 |

**三句话结论**：
1. 方法有效且稳健：主结果 +16.1pp 经 4-seed + 多次 test 复评坐实，机制归因清晰（M1+M2 主导）。
2. 核心创新被验证 + 边界诚实：自进化方法的门控能**诚实识别“找不到可泛化 global skill”而不自欺放行**（AIME24 同分布 train->val 拒掉全部候选、best==S₀；对比 07-29 正是门控失效放行了未改动 skill）。这条负证据把“方法有效性决定因素 = 失败模式跨题可泛化性”钉死、直接给出算法改进方向；强基线 SkillOpt 上仅 parity 是另一条诚实边界。
3. 下一步：跑完跨 agent（ReAct 自调试）验证 skill 跨 agent 架构的泛化性；据此定稿论文叙事。

---

## 2. 当前方法设计与依据（汇报重点：为什么这么做）

### 2.0 通俗版：我们在做什么（一分钟看懂）

**要解决的问题**：一个强模型（B，DeepSeek）解题好坏，很依赖开头给它的那段“提示词”——我们叫它 **skill**，本质就是一份“解题须知”。我们想**自动把这份须知改得更好**，但不重新训练模型本身（所以叫“免训练”）。

**A-B-C 流水线**：
- **A**（小模型）负责“写/改须知”；**B**（强模型）拿着须知去解题；**C**（裁判）判 B 解得对不对、打分；分数反过来指导 A 怎么改。
- 免训练版里 A 不做梯度更新，它就是个 LLM；我们用“打分”来筛它的改写哪个好。

**三段式：S₀ -> v2 -> evolver**：
1. **S₀**：一份手写的初始“解题须知”（读两遍题、分清求的是个数还是和、case 列全、最后核对答案形式）。**这是被改的对象，不是改写器**。
2. **v2**：拿训练题让 B 用 S₀ 解，看 B 在哪错；把错因总结成一张“常见错误清单”；**把清单追加到 S₀ 末尾**（不动 S₀ 原文，只加一节“## 常见错误”）；然后在验证集上测——**只有统计上确定比 S₀ 好（不是噪声）才接受**。
3. **evolver**：从 v2 产物出发迭代多轮。每轮生成好几个改写候选，专攻“到现在还错的题”；测、挑最好的（过了置信门控才要）；外加防退化、定期合并。产出最终 skill。

**四个关键设计（和别人的区别，详见 2.4）**：
- **追加，不改写**：好基线 S₀ 原样保留，只追加错误清单。改写好 skill 往往越改越差。
- **失败驱动**：看 B 真实错在哪，针对性写补丁；不是泛泛的“仔细点”。
- **统计门控（核心创新）**：只有 95% 置信下界 > 0 才接受改进。后果是——**测不出改进就老实退回 S₀，不自欺**。
- **固定基线快照**：永远和同一个 S₀ 比对，公平。

**核心创新点（一句话）**：方法不只优化，还能**诚实识别“优化不动”**——AIME 上它试了、产了候选，但没一个过门控，就正确退回 S₀，没假装有进步。这就是“核心创新被验证”。

**为什么 LiveMath 涨、AIME 不涨（一句话）**：LiveMath 全是选择题、结构共享，一条“避错提醒”能帮很多题 → 清单可泛化 → +16.1pp；AIME 每题不同话题（几何/数论/组合…），一条提醒只帮一窄块 → 清单不泛化 → 门控正确拒绝。


### 2.1 方法全貌：S₀ -> v2 -> evolver 三段式流水线

```
[S₀ 初始 skill] --v2 在线闭环--> [v2 skill = S₀ + 追加失败模式节] --evolver 迭代--> [evolved skill]
      |
      +-- 每一步都用 delta pass-rate 作信号, bootstrap CI 门控, k 投票降噪评测
```

- **S₀**：手写的初始 skill（`data/skill_*.md`，一份通用系统提示词/程序性指令）。**被优化的对象，不是优化器**。
- **v2（在线闭环优化器，`tf_global_v2_math.py`）**：rewriter（DeepSeek-V3.2）见 S₀ 仍错题的真实失败（B 的 pred vs 正解 + 推理节选 + 定理原文）-> 蒸馏**可泛化数学失败模式** -> **追加式**合成（S₀ 原样 + `## Common Failure Modes` 节，不重写 S₀）-> bootstrap CI 门控选优。
- **evolver（自进化迭代器，`skill_evolver_math.py`）**：以 v2 产物为起点，融合 SkillOpt + GEPA + ReSkill 的免训练迭代（四机制 M1–M4，见 2.3），test-free 选择 + 最终 test 验证。
- **A 模型 GRPO 训练线**：创新点原型（A 模型梯度训改写 S₀），1 卡 0.6B 端到端冒烟已打通；**正式 8 卡 4B 训练暂缓**（用户决策：先在免训练设定下把方法做扎实，免训练线同时作强 baseline 对标 SePO）。

### 2.1.1 算法逻辑逐步详解（技术细节）

> 本节给 v2 + evolver 的逐步流程（伪代码级），供汇报讲清"算法每一步在做什么、为什么"。实现：`rl/train/verl1/tf_global_v2_math.py`、`skill_evolver_math.py`。

**v2 在线闭环（每轮 iter）**：
1. **S₀ 固定快照**：在 train+val 上对 S₀ 做 k=5 投票评测（EVAL_TEMP=0.3，pass_rate=通过次数/k），存 `s0_cache`。baseline delta≡0（不现场重跑 S₀，消除采样偏差）。
2. **逐题生成 g=4 变体**：rewriter（DeepSeek-V3.2，GEN_TEMP=0.8 真采样）输入 = S₀ + 题干/选项 + **定理原文** + 当前经验池 + 角度提示；指令"保留 S₀、追加针对本题型的**数学诊断程序**（量词翻转/等号情形/存在 vs 唯一/必要 vs 充分）、不解题、不提答案标签"。每题 g 个变体并发生成。
3. **评测变体**：每个变体在该题 run_one（k=1）得 pass_rate、delta=pass−s0_pass，并**保留 B 的 response/pred**用于诊断（修 v1 bug#3：rewriter 从未见真实错误）。
4. **蒸馏失败模式**：同题 g 个 rollout（skill_excerpt + B 推理节选 + pred vs correct + 定理）按 delta 排序喂 `FAILURE_MODE_TEMPLATE`，**反转泛化规则**--拒绝 prompt 措辞技巧、拒绝单题答案，只留跨题可复用的数学诊断程序（修 v1 bug#2）。
5. **增量合并经验池**：`merge_failure_modes` 显式去重+增量，hash 校验防池冻结（修 v1 bug#1），池上限 20 条。
6. **追加式合成**：S₀ 原样 + `## Common Failure Modes` 节（**不重写 S₀**，修 v1 bug#5）；退化保护：合成结果 <0.9×S₀ 长度则回退 S₀。
7. **val 门控**：S_new 在 val 上 k=5 投票，delta vs `s0_cache`；bootstrap 1000 次取 95% CI 下界；**接受 ⟺ CI 下界>0 且 delta>历史 best**（修 v1 bug#6：裸 delta 接受 1 题翻转噪声）。否则 KEEP PREV。

**evolver 自进化（每轮 iter）**，以 v2 产物为起点（不回 S₀）：
1. **失败收集（M2）**：当前 best 在 select(train+val) 上 k 投票，收集 pass_rate==0 的仍错题（含 pred/correct/response/theorem），优先 S₀ 也错的（真 headroom），取 n_diag=8。
2. **focus 轮换（M4）**：按 `FAILURE_FOCI[(t−1)%8]` 选本轮失败类型侧重（蕴含 vs 双向 / 存在 vs 唯一 / 量词序 / 正则性 / 等号情形 / meta-option / 定义域 / 分类完备性）。
3. **生成 K=4 候选 patch（M1）**：`PATCH_TEMPLATE` 指令"小外科手术式编辑、加**具体可执行决策程序**（非通用 checklist）、≤5000 字、锚定诊断失败、避免 rejected patch"；K 候选各带不同角度提示增多样性。
4. **评测+选择（M1）**：每候选在 select 上 k 投票得 delta+CI；**winner 仅从 novel（hash≠current）且 CI>0 的候选里选 delta 最高**--产物身份校验，防原地踏步被误判改进（07-29 bug 教训）。
5. **接受判定（M4）**：improved = winner.delta>best 且 CI>0 且 hash≠current；**LOST 监控**：winner.pass<base−0.06 则拒（防退化）。接受时 assert hash 已改。
6. **慢更新（M3）**：每 3 轮且长度>4000 时 `consolidate`（合并冗余、保留所有决策程序、≤4500）。
7. **早停**：连续 patience×2 轮无提升则停。最终校验 hash(init)≠hash(final)。

**关键技术机制与目的**（每个设计解决什么问题）：

| 机制 | 实现 | 目的（解决的失败模式） |
|---|---|---|
| delta pass-rate 信号 | reward=skill_pass−s0_pass（对 s0_cache 固定快照） | 绝对 pass 被题目难度污染；delta 隔离 skill 贡献 |
| bootstrap CI 门控 | 1000 次重采样取 95% 下界，>0 且>best 才接受 | 裸 delta 在小样本上接受 1 题翻转噪声（v1 bug#6）；CI 正确拒掉 train-overfit 不泛化候选（**AIME 负结果即此机制生效**） |
| 追加式合成 | S₀ 原样 + 失败模式节，不重写 | 强模型+好 S₀ 时重写=干扰（v1 反转教训）；追加保留 S₀ 优势只加增量 |
| k 投票 + 固定基线 | 评测 k=5 投票（EVAL_TEMP=0.3），S₀ 也 k 投票对称，baseline 用快照 | temp=0 在 vLLM 网关非确定（±2~3 题噪声）；k 投票降噪、对称消除 baseline 采样偏差（v2 +15.3pp 假阳性根因） |
| 产物身份校验 | md5(output)≠md5(input) 才算改进，winner 仅选 novel | 07-29 evolver 8 轮 best==init 被误判 +15.7pp；身份校验是第一道关 |
| 失败模式蒸馏（反转泛化） | 模板拒绝 prompt 措辞/单题答案，留跨题数学诊断程序 | v1 bug#2：泛化检查只留通用写法→pool=S₀ 陈词滥调→换皮→无效 |
| 失败驱动 patch | patch 锚定当前仍错题的 pred vs correct + 推理 | 通用 checklist 对强模型无效；具体决策程序才撬动 headroom |


### 2.2 核心设计决策与"为什么"

| # | 设计决策 | 为什么（依据 / 教训） |
|---|---------|---------------------|
| 1 | **优化上下文空间（skill），不做参数梯度**（免训练线为主） | 研究方向从参数空间转向上下文空间；免训练零 GPU、可作强 baseline 对标 SePO；GRPO 训练线冒烟已通但暂缓，待免训练方法坐实后再对位 |
| 2 | **从 S₀ 优化，不从零生成** | 从零生成时 A/rewriter 易走捷径输出通用废话 prompt 拿基础分；给 S₀ 作起点强制做"改进" |
| 3 | **奖励 = delta pass-rate**（skill_pass − s0_pass） | 绝对 pass rate 被题目难度污染；delta 隔离 skill 贡献（呼应 D2Skill 双 rollout、SkillComposer 也用 delta pass rate） |
| 4 | **蒸馏失败模式，而非 prompt 措辞技巧** | v1 的 6 bug 根因之一：模板的 Generalization Check 拒绝问题特定内容只留通用写法 -> pool 全是 S₀ 已有陈词滥调 -> global skill = S₀ 换皮 -> 对强模型纯干扰（test 持平 S₀）。失败模式是 S₀ 没有的、跨题可复用的**数学诊断程序** |
| 5 | **追加式合成，不重写 S₀** | 强模型 + 好初始 skill 时，重写 S₀ = 干扰（TF-global v1 反转教训）；追加一节保留 S₀ 优势、只加增量 |
| 6 | **bootstrap CI 门控（95% 下界>0 才接受），不用裸 delta** | v1 用裸 delta，val=18 题上 +0.056 = 1 题翻转噪声被接受 -> test 持平。CI 门控让 v2 iter1/2 被正确拒绝、仅 iter3 接受 -> test 泛化成立 |
| 7 | **k 投票评测 + 固定基线快照 + 真采样温度** | v2 +15.3pp 假阳性根因：① temp=0 在 vLLM 网关上非确定（batch/并发/浮点累加序不同）；② baseline 单次采样撞偏低值、candidate 多次取平均噪声减半 -> delta 虚高。修后真实增益 +6~7pp |
| 8 | **global 形态，非 per-task** | per-task 每题一份 skill 不泛化（SkillComposer 仅 per-task online 有效）；真问题是 TF-global 要超 S₀，靠失败模式 + 追加式解决 |
| 9 | **evolver 2-iter 是甜点，不多迭代** | 10-iter 全量长跑 test +13.7pp 坐实但**未超 2-iter**（-2.4pp 不显著）；更多迭代在 select set 上刷分更高（val +0.291 > +0.199）却过拟合、test 反低。val delta 不预测 test 泛化 |
| 10 | **最小定向 patch，不做整段 splice 合并** | 朴素 splice（SkillOpt 全文 + evolver 失败模式节）test -4.7pp 稀释——整段加入让模型注意力从简练启发式偏移。evolver 的非 meta-option 优势来自整体结构，非可移植 bullet |
| 11 | **产物身份校验**（hash(output)≠hash(input) 才算改进） | 07-29 evolver "+15.7pp" 推翻教训：8 轮 best==init（md5 全同），"增益"实为两版 v2 对比。优化器产物必须先验身份 |

### 2.3 evolver 四机制与消融归因（Task1，2026-08-04）

evolver 由四个机制组成（`skill_evolver_math.py` 源码定义），通过 **leave-one-out 消融**定位贡献（test=106，4-seed，per-qid paired t, df=105）：

**通俗版（一分钟看懂 M1-M4）**：evolver 每轮做四件事，消融实验（去掉某件看掉多少分）证明哪件真有用：

- **M1 赛马选优（最关键，去掉掉 10.5pp）**：每轮不只让 A 写一个改写，而是写好几个（K=4，从不同角度），都测一遍，挑最好的留下。-> "多出几版方案，赛马选优"。
- **M2 哪错补哪（第二关键，去掉掉 7.8pp）**：专盯"到现在还错的题"，针对这些题的真实错因写小补丁，不泛泛改。-> "哪错补哪"。
- **M3 定期合并（没啥用，去掉只掉 0.08pp）**：每隔几轮把零散补丁整理合并去冗余。-> "定期收拾房间"，测出来对泛化没贡献。
- **M4 安全护栏 + 换口味（边际，去掉掉 1.9pp）**：防越改越差（低于基线就拒）+ 轮换关注不同失败类型、别老盯一种。-> "护栏 + 换口味"，不显著。

**一句话结论**：增益主要来自 M1（赛马选优）+ M2（哪错补哪），两者高度显著；M3/M4 对泛化无显著贡献。故事干净，论文机制归因清晰。

| 机制 | 定义 | 去掉后 Δ(FULL−abl) | p | 结论 |
|------|------|------|------|------|
| **M1** 候选种群 + 帕累托选择（GEPA） | 每轮 K 候选，val bootstrap CI 选前沿 | **+10.5pp** | 1.9e-7 | *** 主导机制 |
| **M2** 失败驱动局部 patch（SkillOpt 贪婪编辑 + ReSkill 失败归因） | 只做 add/replace 小改，锚定当前仍错题的具体失败，产出具体决策程序非通用 checklist | **+7.8pp** | 9.2e-5 | *** 第二关键 |
| M3 慢更新 consolidate（SkillOpt epoch-level） | 快周期局部 patch + 每 N 轮蒸馏整合去冗余 | +0.08pp | 0.958 | ns（select 上 +0.251 但不迁移 test = 过拟合） |
| M4 防退化 + LOST 监控 + 多样性 | 轮换失败类型侧重，防退化护栏 | +1.9pp | 0.267 | ns（边际） |

**核心发现**：test 增益主要来自 M1（种群搜索 + 选择）和 M2（失败驱动 patch），二者高度显著；M3/M4 对 test=106 泛化无显著贡献。M1+M2 = +18.3pp > FULL−S₀ +15.9pp（非加性、有重叠，符合预期）。消融故事干净，给论文提供了机制归因。

### 2.4 与已有方法的差异（创新点所在）

| 方法 | 优化器 | 信号 | 起点 |
|---|---|---|---|
| SkillOpt（Microsoft） | LLM 启发式 rollout-reflect | 绝对 pass | S₀ |
| SkillComposer | rejection sampling（非 RL） | delta pass | per-task |
| SePO | 进化搜索（非 RL） | LLM-judge 相似度 | S₀ |
| **我们** | **evolver 免训练迭代（+ GRPO 训练线原型）** | **delta pass rate** | **S₀（追加式不重写）** |

差异化：① 别人用启发式/进化搜索/rejection sampling，我们用 GRPO 梯度信号训 A 改写 skill（训练线）+ 免训练 evolver（对标 SePO 的强 baseline）；② 追加式合成不干扰强模型的 S₀；③ 失败模式蒸馏看到 B 的真实错误 + 定理原文，而非 prompt 措辞。

**机制级技术差异（每条可指认的算法分歧）**：

- **vs SkillOpt**（最强基线，Microsoft）：SkillOpt 用独立 optimizer LLM 对单 skill 文档做 add/delete/replace 改写 + 文本学习率，held-out 严格提升门控，**绝对 pass** 信号，epoch 级慢更新；我们用失败蒸馏驱动**追加式** patch、**delta pass** 信号、**bootstrap CI** 门控。SkillOpt 反思 rollout 文本，我们喂 B 的 pred vs correct + 定理原文。结果对照：免训练 evolver 在弱基线(v2)上 +16.1pp 高度显著；种到 SkillOpt 上达 parity（不强于，但其失败驱动**最小 patch 不稀释** vs 朴素 splice −4.7pp）。
- **vs SkillComposer**（通义）：把 Create/Improve/Merge 三操作**训练进模型**（SFT 内化），delta≥0.4 作 SFT 数据拒绝阈值，**per-task online** 有效（LCB +23pp）但 **global offline 失效**（MemP 47.4<noskill 56.6）；我们**免训练**、delta 作**在线 CI 门控**、**global + 失败模式分段**。关键：SkillComposer 证明 per-task 路径有效，我们证明 global + 失败模式分段也能超 S₀（推翻"global 对强模型无效"误判）。
- **vs SePO**：自指涉进化搜索（连 optimizer 的 prompt 一起优化）+ archive keep-better，delta 是 **child-vs-parent**（基线漂移）；我们 population + **CI 门控 max-delta**，delta 是 **skill-vs-固定 S₀**（隔离 skill 贡献）。SePO 突变改写 prompt，我们追加。
- **vs GEPA**（M1 灵感源，ICLR 2026 Oral）：自然语言反思 + **Pareto 前沿**（精度+简洁多目标），绝对精度；我们 M1 借鉴种群+选择，但用 **delta + CI 单目标**，反思是**结构化失败蒸馏**而非自然语言泛泛反思。
- **vs ReSkill**：把 skill 创建嵌进 GRPO 循环，组内 skill 版本对比 + Thompson 采样，"一次采样三用"；我们**免训练**（不进梯度循环），skill-vs-S₀ 固定基线 + CI 门控。

**三条最关键的算法分歧**（论文创新点所在）：
1. **失败信号结构化**：别人反思 rollout 文本 / 准确率 delta（SkillOpt/SePO/GEPA）；我们直接喂 B 的 **pred vs correct + 定理原文**，蒸馏出可复用数学诊断程序（M2 失败驱动 patch 核心）。
2. **追加式 vs 改写式合成**：所有参考方法都**改写** S₀/prompt（SkillOpt 文档编辑、SkillComposer Improve、SePO 突变、GEPA 反思突变）；我们**追加**失败模式节、S₀ 原样保留。强模型+好 S₀ 时改写=干扰（v1 反转教训），追加只加增量。
3. **统计门控严格性**：SkillOpt held-out 严格提升（无置信区间）、SePO keep-better、SkillComposer 固定阈值 ε=0.4；我们 **bootstrap 95% CI 下界>0**。后者在小样本上正确拒掉 train-overfit 噪声（v1 bug#6、**AIME 负结果即此机制生效**--门控诚实拒掉不泛化候选），前者易接受 1 题翻转假阳性。



---

### 2.5 基线与 S₀ 出处（可复现性说明）

**LiveMath S₀（主结果基线）= SkillOpt 官方 seed，非本项目撰写**：
- 文件 `skillopt/envs/livemathematicianbench/skills/initial.md`（1018 字节），来自 SkillOpt 官方仓库（arXiv:2605.23904，作者 Yifan Yang，随仓库分发，git PR #133）——即 SkillOpt 为 livemathematicianbench env 提供的 `initial.md` seed skill。
- 我们方法的 v2/evolver 均以该官方 seed 为 S₀、追加 failure-mode 节。故“evolver +16.1pp vs S₀”的 S₀ = 官方 seed，对位干净。

**所有基线数值 = 我们复现，非引用论文表格**：
- SkillOpt README 报跨 env 平均 +23.5pp（direct chat）/ +24.8pp（agent），为论文数。
- 我们在自己的后端（Midea DeepSeek-V3.2 via qwen_chat）+ 自己的 split（2:1:7 seed42，test=124/106，greedy / k-投票）上复现：noskill 29.0% / S₀ 33.1% / SkillOpt-best 51.6%（seed→best +18.6pp），与论文量级一致（+18.6~21.8pp vs 论文 +23.5pp）。
- 即 **seed skill 文件是官方的，但表中所有 acc 数值为我们复现测得**，非引用 SkillOpt 论文表格。
- “SkillOpt-best” = 我们跑 SkillOpt 官方 optimize loop 复现的其方法产出（`skill_v0002.md`），作强基线对照。

**AIME S₀（泛化边界实验基线）= 项目内手写（已知局限）**：
- AIME **不是 SkillOpt 自带 env**（自带 9 个 env：livemathematicianbench / appsplus / lcb / mbpppro / alfworld / docvqa / officeqa / searchqa / spreadsheetbench，无 aime），无官方 seed。
- 故 AIME S₀ = `data/skill_aime_initial.md`（2239 字节），2026-08-04 建链时**项目内手写**（照 LiveMath seed 结构撰写竞赛推理 skill）。
- ⚠️ **已知局限（confound）**：手写 AIME S₀ 把形态级元策略写得较满（count-vs-value / case 互斥完备 / 答案形式 / 量级校验），比官方 LiveMath seed（1018B）详细 2.2×（2239B）。因此 §4.1 ⑤ 中“AIME 形态级无 headroom、CI 拒”的归因**部分受 S₀ 作者选择影响，不完全是任务属性**。
- **处理（用户决策，暂不跑）**：不刻意降低 baseline；后续将 AIME S₀ 工程化/合理化（对齐官方 seed 饱和度或采用最小化 seed）以去除该 confound。当前报告中 AIME 结论措辞限定为“优化器在给定 S₀ 上诚实检测到无可泛化改进”，S₀ 出处作 limitation 注明。负结果本身（门控诚实拒绝、未自欺）不依赖 S₀ 出处。


## 3. 验证结果总表（2026-08-05，同条件对位）

**主试验场**：LiveMathematicianBench（数学选择题，确定性 `<answer>` 验证，单轮，无 Docker，本机零 GPU）。test=106（1-3-6 split, seed42），eval_k=3 投票，EVAL_TEMP=0.3。

| skill | mean pass | Δ vs S₀ | t | 95%CI | 显著? |
|-------|-----------|---------|---|-------|-------|
| S₀（baseline） | 0.301（fresh 8-04: 0.291） | - | - | - | - |
| v2 | 0.365 | +6.4pp | 4.58 | [+0.037,+0.091] | ✅ |
| **evolver 2-iter（主结果）** | **0.462**（fresh 8-04: 0.450） | **+16.1pp**（fresh +15.9pp） | 9.02 | [+0.126,+0.196] | ✅✅ |
| evolver 10-iter（迭代消融） | 0.438 | +13.7pp | 8.10 | [+0.104,+0.171] | ✅（未超 2-iter） |
| SkillOpt-best（强基线，别人方法） | 0.546 | +24.5pp | - | - | - |
| evo_seed（evolver 种在 SkillOpt 上） | 0.559 | +25.8pp | vs SkillOpt +0.59pp (p=0.42) | [-0.008,+0.020] | 🟡 parity |
| trace2skill-math（离线基础池） | 0.355 | +2.4pp | - | - | ✅ modest |

**机制消融**（fresh 8-04，4-seed）：见 2.3 表。fresh FULL=0.4497（+15.9pp vs S₀ 0.2909）；FULL 低于已发表 0.4623 约 1.3pp = API 跨日漂移（cached 7-30 系统性高 4-6pp），消融 delta 全 fresh-vs-fresh 公平有效。

**跨任务泛化**（AIME，数值题，0-999 整数）：
- AIME24（3-seed, k=3, 30 题）：evolver +1.9pp（t=0.70, CI[-0.034,+0.071]）❌ 不显著
- AIME 从头链（Task2，train18/val12 + AIME25 held-out test）：v2+evolver 全 KEEP PREV，best skill == S₀（未改动），test 三组同噪声带（s0 0.497 / v2 0.500 / evolver 0.533，同 hash）❌ 负

---

## 4. 诚实边界（负结论，不是失败）

> **定位（08-05 修正）**：AIME 同域负结果不是“方法失败”，而是**核心创新点的正面验证**。我们的自进化要找“从 baseline 持续进化的 global skill”，门控（bootstrap CI on held-out val）能在**找不到可泛化改进时诚实拒掉 train-overfit 候选、退回 S₀**，而非自欺放行（07-29 的 bug 正是放行了未改动 skill）。AIME24 同分布 train->val 全候选被拒 = 这套诚实门控在工作。它把“方法有效性的决定因素 = 失败模式跨题可泛化性”钉死，给出算法改进的明确切入点（见 §4.1 ④）。**这是好消息。**

1. **AIME 跨形态泛化边界**（跨任务，07-31）：livemath 选择题上 +16.1pp 高度显著，AIME24 数值题上 +1.9pp（t=0.70）不显著。skill 学到的是选择题形态特有的诊断程序（选项措辞检查、量化词判断、选项对比排除），不迁移到无选项的数值竞赛题。用户原话"不一定就是必须要泛化"--作边界证据补入。
2. **AIME 同域优化失败 = 核心创新验证**（08-04/05，详见 §4.1）：AIME24 拆 train18/val12，v2+evolver 全 KEEP PREV、best skill == S₀（未改动）。关键在**失败发生在 AIME24 自身同分布 train->val 上**（连同分布都不泛化，更谈不上 24->25 跨数据集）--**已确认非 bug，是门控正确拒掉 train-overfit 候选的算法正确运行结果**（对比 07-29 门控失效放行垃圾）。根因 = AIME 失败模式逐题特异、组不成可泛化 global skill。
3. **强基线 parity**：evolver 种在 SkillOpt 上（seeded-evolver）16-seed 复评 +0.59pp（p=0.42）不显著，8-seed +1.65pp 是向上波动。方差分解显示 92.6% 方差是 k=3 投票测量噪声，split-half 逐题符号一致率 43%<随机--无稳定逐题信号，加 seed 无益。**方法的泛化性随基线增强而递减**（弱基线 v2 上 +16.1pp 高度显著；强基线 SkillOpt 上 parity），诚实边界。
4. **代码域 negative**：LCB/mbpp/APPS_Plus 上 skill 优化均不泛化（test -0.7~-2.1pp）。根因非"skill 对代码无用"（SkillComposer +23pp / SePO MBPP +4.49 同类正先例成立），而是我们用通用 prompt 文本 skill + stdin 竞赛题 + 全量注入。代码线降级为对照，作"模型饱和 -> 需 agent 架构"的过渡证据。
5. **🔑 关键洞察（方法有效性的决定因素 = 算法改进的杠杆）**：综合 4 条负结论，方法的有效性取决于**任务失败模式的跨题可泛化性**（任务结构属性），而非数据域本身。选择题（LiveMath）有共享的元结构（选项措辞/强弱结论判断），一条失败模式帮很多题 -> global skill 泛化 -> +16.1pp；AIME 自由作答异质（每题独特话题），失败模式逐题特异 -> 组不成泛化 global skill。**这既是方法的核心边界条件（论文“何时有效/何时无效”判据），也是算法改进的直接切入点**：让 evolver 的失败模式蒸馏在异质任务上也产出跨题可复用内容（见 §5）。

### 4.1 AIME 同域实验深度分析（2026-08-05，bug 核验 + 根因 + 算法改进方向）

**实验**：S0 -> v2_aime(iters=2,g=4) -> evolver_aime(iters=4,K=4) -> eval AIME25(4-seed,k=3)。AIME24(30) 拆 train18/val12(seed42)，AIME25(30) held-out test。

**① 非 bug 核验（三条证据链）**：
- **候选真生成了**：v2 产出 5KB/4.8KB 候选（S0 2.2KB + 15 条 AIME 失败模式诊断程序）；evolver 4 轮 x 4 = 16 个 novel 候选（hash 全不同：bbd652d064/6d040bc184/be92f9610b/57d6d09caa）。失败模式蒸馏质量高（experience_iter1.md 含 Power of a Point、maximal vs maximum、collinearity、conditional probability sample space 等 10+ 条 symptom/root-cause/diagnostic 结构真 AIME 内容，非 v1 的 prompt 措辞陈词滥调）。
- **训练信号是真的**：v2 train_mean_best_delta +0.148(iter1)/+0.259(iter2)--rewriter 确实在 train 上改对了 S0 错的题。
- **CI 门控正确拒绝（与 07-29 bug 相反）**：07-29 bug = 门控失效、接受未改动 skill（假阳性 +15.7pp）；本次 = 门控正常、拒绝所有候选（真阴性，best 退回 S0）。候选点 delta 正（+0.067/+0.056）但 CI_lower <= 0（不显著）-> 拒；iter4 候选 pass 0.611 < base 0.678 -> 反退化护栏拒。产物身份校验也正确识别 best==init 并 flag。**算法在正确工作，只是没找到统计显著的改进。**

**② AIME24 vs AIME25 差距（真实但非主因）**：

| | S0 pass | 主题分布 |
|---|---|---|
| AIME24 select(30) | 0.567（mean_vote）/ 0.678（evolver baseline） | geometry 9 / NT 9 / algebra 4 / comb 3 |
| AIME25 test(30) | 0.497（4-seed mean） | geometry 12 / NT 9 / algebra 2 / comb 4 |

- 差距 ~7-18pp，AIME25 更难（几何更多，S0 几何弱）。但**主题分布相近、同属 AIME 竞赛数学，无域迁移**。
- **非主因**：优化失败发生在 AIME24 train->val（同分布）上，24/25 差距只拖累 test 表现，不是优化失败的瓶颈。

**③ 真正根因 = train->val 泛化失败（失败模式跨题可泛化性不足）**：
- v2 iter1：train +0.148 -> val +0.111（点估正）但 CI_lower -0.028 -> 拒。**train 改进没可靠迁移到 val**。
- **AIME 题目异质**：每题独特话题组合（几何/组合/数论/代数），"Power of a Point"只帮几何题；15 条失败模式各管一窄块，组不成泛化 global skill。
- **对比 LiveMath（选择题）**：失败模式（选项措辞检查、"更强结论能否证明"）利用所有题共享的选择结构，一条帮很多题 -> global skill 泛化 -> +16.1pp。
- **叠加低统计功效**：val=12、k=3，1 题翻转 = +0.083，候选 val 点 delta +0.111（~1.3 题）落在噪声带内，CI 无法确认。

**④ 结论：核心创新验证 + 算法改进切入点（无需 AIME25 验证）**：AIME24 自身门控（同分布 train->val）过不了 = 算法未产出可泛化 skill 的直接判决，**无需再拿 AIME25 验证**--再验反而是在怀疑我们自己的门控（且 eval 已证 s0/v2/evolver 三者同 hash、AIME25 表现 == S₀±噪声，与门控退回 S₀ 完全一致）。这正是核心创新点的体现：自进化方法能**诚实识别“当前找不到可泛化 global skill”并退回 S₀**，而非自欺放行 train-overfit 候选（07-29 bug 正是放行了）。**好消息**：这把“方法有效性决定因素 = 失败模式跨题可泛化性”钉死。**算法改进方向**：诊断 LiveMath（共享 MCQ 元结构->失败模式跨题可复用->+16.1pp）vs AIME（异质自由作答->失败模式逐题特异->组不成泛化 global skill）的差异，让 evolver 的失败模式蒸馏/合成在异质任务上也产出跨题可复用内容（如按题型聚类蒸馏、跨题失败模式去重/抽象、形态无关的诊断程序）。

**⑤ 内容级证据：为什么 LiveMath 泛化、AIME 不泛化（算法改进依据，2026-08-05）**：
对比两条 skill 的 `Common Failure Modes` 节（`run_20260730_111306/best_skill_evolved.md` vs `tf_skill_aime_v2/run_20260804_223830/skill_iter1.md`）：
- **LiveMath（泛化，+16.1pp）**：11 条失败模式全部是**形态级**（option-vs-theorem 结构关系）--等价双向检验、量词域核对、强度排序、等号 vs 包含、meta-statement 决策程序。每条适用于任意含该选项结构的定理级 MCQ -> 跨题高复用 -> global 泛化；且这些结构程序 **S₀ 未全覆盖**（S₀ 只说"对比选项、跟踪量词"）-> 蒸馏真增了可泛化内容。
- **AIME 被拒候选（不泛化，CI 拒）**：15 条失败模式多为**内容级/主题特异战术**--Power of a Point（仅圆几何）、Maximizing Sinusoidal Sums（仅三角）、Median for Even-Length Lists（仅统计）。每条只管一窄块 -> 跨题低复用 -> 组不成泛化 global skill；且 ~4 条形态级的（maximal vs maximum / 完备性 / at-least vs exactly）**已在 S₀ 的 Approach 节** -> 蒸馏没产出 S₀ 还没有的可泛化内容。
- **深层结论**：方法能在 S₀ 之上改进 ⟺ (a) 任务有共享元结构产出可复用结构失败模式，且 (b) S₀ 尚未覆盖。AIME 两者皆不满足（自由作答异质 + S₀ 已吃掉形态级 headroom）-> 无可泛化改进可得 -> CI 门控正确拒绝。
- **算法改进方向**（已建 `experience/templates_math_v3.py`）：① 蒸馏分层 FORM/CONTENT，只把形态级进 global（防内容级稀释）；② 加 S₀-coverage 检查（避免产出 S₀ 已有内容）；③ **per-topic skill 库**（按主题聚类，每簇同质如 LiveMath -> 簇内可泛化，是 AIME 取得增益的真正路径，tradeoff=每簇统计功效）。


**⑥ S₀ 出处局限（2026-08-05）**：上述“AIME S₀ 已吃掉形态级 headroom”的归因存在 confound——AIME S₀ 是项目内手写（非官方 seed，见 §2.5），其形态级饱和度由撰写者决定。故“AIME 是泛化边界”的结论强度依 S₀ 饱和度而定：在当前（较满的）S₀ 上成立，在更弱 S₀ 上待证。后续工程化 AIME S₀ 后复测可去混淆；负结果本身（门控诚实拒绝、未自欺）不依赖该归因。


详见 memory/aime-fromscratch-deepanalysis-20260805.md。
---

## 5. 下一步规划

### 短期（立即推进）
1. **诊断 LiveMath vs AIME 差异 -> 改进算法**（用户 08-05 指定核心方向）：AIME24 同域门控拒全部候选揭示了“失败模式跨题可泛化性”是方法有效性的决定因素。诊断为什么 LiveMath（共享 MCQ 结构）失败模式跨题可复用、AIME（异质自由作答）逐题特异，据此改进 evolver 的失败模式蒸馏/合成机制（按题型聚类蒸馏 / 跨题失败模式去重抽象 / 形态无关诊断程序），目标让方法在异质任务上也找到可泛化 global skill。
2. **跑完跨 agent（Task3）**：同 skill（evolver, hash 12633201）/ 同模型 / 同 28RPM，唯一差异 = single-call vs ReAct（+ execute_code 自调试工具循环）。冒烟已通过，路径 bug 已修，baseline 改 8-04 fresh 避免漂移。启动：`bash rl/train/verl1/run_cross_agent_react.sh`（≈2-4.5h）。验证 skill 跨 agent 架构的泛化性。
3. **据此定稿论文叙事**（见 5 长期）。

### 中期（1 个月内）
4. **控制变量诊断**（数学 bench 上，一次一变量 vs SkillOpt baseline）：① rollout（ReAct vs single-shot，= Task3）；② 表示（池+检索 vs 单 best_skill）；③ **优化信号（GRPO+ATE vs val-gate）= 论文核心**；④ skill 来源（trace 蒸馏 vs 手写 seed）。
5. **8 卡 4B 正式 GRPO 训练**（托管平台）：1-iter smoke 确认无 OOM + reward 流通，再 max_iter=5；`check_delta_health.py` 监控 agent_ok% / degen% / mean_delta 衰减。训练线 vs 免训练线同表对位。
6. **多 seed 复测**关键正结果坐实统计显著性（主结果已 4-seed 坐实）。

### 长期 / 论文级
7. **论文叙事**：从"参数空间 -> 上下文空间"。创新点 = GRPO 训 A 改写 S₀（delta pass-rate 信号）+ 失败模式 trace 蒸馏创建 + 池化检索；数学为主试验场，代码 null 作过渡证据，AIME 作泛化边界。
8. **泛化验证**（P2，单 agent 有效后）：smolagents/LangGraph 多 agent 适配器，跨 agent ATE 排序一致性。
9. **工程化**（P3）：ACP 协议标准化 A/B/C 通信；API key 集中管理。

### 算法未来优化方向（汇总，2026-08-05）

**来自 AIME 诊断（§4.1 ⑤，已建 `experience/templates_math_v3.py`）**：
1. **蒸馏分层 FORM/CONTENT**：只把形态级（跨题型结构）失败模式进 global，内容级（主题特异）尝试抽象否则拒。防内容级稀释。
2. **S₀-coverage 检查**：蒸馏/合并时对照 S₀，拒绝已被 S₀ 覆盖的模式（避免产出 S₀ 已有内容）。
3. **per-topic skill 库**：按主题聚类，每簇同质（如 LiveMath）-> 簇内可泛化；AIME 取得增益的真正路径，tradeoff=每簇统计功效。

**用户 08-05 新提出**：
4. **允许修改/删除 S₀，不只追加**：当前 v2 合成是纯追加（S₀ 原样 + 失败模式节），理由是"改写好 S₀ 越改越差"（v1 反转教训）。但 S₀ 可能含拖后腿的条目（过时/冗余/与任务冲突）。改进：在统计门控下做**定向删除/改写**（如消融某条 S₀ bullet，CI 涨才删），而非整段重写--兼顾"去有害内容"与"不破坏好基线"；与 evolver M2 的 add/replace 小补丁互补（M2 目前只加/换，不删）。
5. **从正确题中学习，不只从错误**：当前蒸馏只看 B 的错题（失败模式）。但 B 解对的题里也有可推广的"正确做法"（成功模式/正信号）。改进：对正确题蒸馏"这条通用解法值得固化"的正向指令，与失败模式互补（正负双信号）；注意"对 ≠ 优"（简单题对了无信息），应聚焦难题/干净解法。参考 GEPA 的正向反思（我们目前只做了负向）。


### 三种可发表结果形态（来自双线方案）
- GRPO-4B ≫ TF -> "训练值得"
- TF ≈ GRPO-4B -> "免训练足够"
- TF > GRPO-4B -> "小模型 RL 优化 prompt 的诚实负结果"

---

## 6. 关键经验教训（汇报可强调）

1. **同条件对位是命门**：两次大反转（07-18 APPS_Plus、07-23 TF-global）都因拿历史 baseline 比；绝对值必须同 key/同模型/同采样重测，delta 信号必须基于同条件 baseline。
2. **小样本假阳性**：test=200 的 +6pp 在 test=2765 翻盘；v2 +15.3pp 是评测噪声假阳性（修后 +7pp）。结论必须多 seed 全量复测。
3. **产物身份校验**：evolver "+15.7pp" 因 best==init（md5 全同）被推翻。优化器产物必须先验 hash(output)≠hash(input)。
4. **val delta 不预测 test**：10-iter val +0.291 > 2-iter +0.199 但 test 反低 2.4pp；select set 信号会过拟合。test-free 选择必须最终 test 验证。
5. **temp=0 ≠ 确定性**：vLLM 网关级 batch/并发/浮点累加序致输出不可复现；baseline 与 candidate 必须同 pass 数同条件。
6. **静默失败**：reward 路径 try/except 吞掉 TypeError/auth 403/429；日志"看着干净"≠真干净，要 smoke 副作用。
7. **API 是瓶颈不是 GPU**：reward 占 79% step 时间，GPU 利用率仅 2.6%；4 卡只比 8 卡慢 ~15%。28RPM 全局硬限约束实验串行。

---

# 附录 A：详细时间线（过程记录，供溯源）

> 以下保留完整的按时间线过程记录（含每次反转的根因与 bug 修复）。结论以第 1–6 节为准；本附录中个别早期"乐观判断"已被后续复测推翻（保留作过程记录，已标注）。

## A.0 时间线总览

| 阶段 | 时间 | 主题 | 一句话结论 |
|---|---|---|---|
| 起点 | 2025-11~12 | GRPO 基础训练链路搭建 | 8 卡 verl 训练跑通，rollout/评测/监控成型 |
| 设计奠基 | 2026-06 | 融合设计 + BAgent 架构 | 确立 A->B->C 三层融合，BAgent 自进化 Agent 落地 |
| 早期正结果 | 2026-07-01~03 | 消融实验 + Self-Debug | MBPP/Humaneval SD p=0.0009/0.0048 显著，自调试参与率 35-43% |
| 诚实体检 | 2026-07-13 | 全面审计 + 导师 4 建议 | 402 skill 全 ATE=0、70% 有害；GRPO 停在 step 637；Step1 双阴性 |
| 方向重定位 | 2026-07-14 | SkillOpt 试验场 + 因果链 | 选 SkillOpt 为 testbed+最强基线；确立"先单 agent 有效再谈泛化" |
| 代码域判别 | 2026-07-15 | SkillOpt × 代码 | val +21pp 但 test 不泛化(-1.5pp)；代码域存疑 |
| 数据集博弈 | 2026-07-16~18 | APPS_Plus 主基准 | test=200 +6pp 假阳性 -> 全量 test 反转为 -2.1pp |
| 污染排除 | 2026-07-20~21 | LCB 分层 + optimize | 排除数据污染；hard 题真实能力仅 0.18，headroom 巨大 |
| 创新点确立 | 2026-07-21~22 | GRPO 优化 Skill 方法 | 创新点=A 模型 GRPO 改写 S₀，reward=delta pass-rate；端到端冒烟通过 |
| 双线并行 | 2026-07-23 | 训练 vs 免训练 | TF-global 反转(更差)、TF per-task +8.4pp(不泛化)；方向纠回 global |
| 当前转向 | 2026-07-24 | 暂试数学 bench | 数学 bench 干净正结果 +18.6pp，代码线降级为对照 |
| 阴性诊断+v2修复 | 2026-07-27 | TF-global 数学阴性根因 + v2 修复 | 诊断 v1 6 bug(优化器没产出新信息)+数学域从未建 trace2skill；v2 修 6 bug test 0.4839(+15.3pp 强正, global)，trace2skill-math +2.4pp；推翻"global 对强模型无效"结论 |
| 复测坐实+evolver | 2026-07-27 | 多 seed 复测 + skill 自进化 | +15.3pp 单次需复测；启动 skill_evolver_math(iter1 val+0.302 强于 v2, 目标超 0.5161)占 API 配额，复测暂停让其先跑、watcher 自动接力 |
| v2 假阳性修复 | 2026-07-28 | 评测 bug 诊断 + 修复 | +15.3pp 是评测噪声假阳性(temp=0 非确定+baseline 不对称)；修 k 投票+固定基线+真采样+扩 val，test=106 +7.0pp 泛化成立 |
| v2 坐实+evolver 推翻 | 2026-07-29 | 多 seed 复测 + 纠偏 | v2 +6.6pp 2-seed 坐实(t=2.78 p≈0.006 CI[+0.020,+0.113])；evolver +15.7pp 推翻(8 轮 best==init 未改 skill) |
| evolver 修复+冒烟 | 2026-07-30 | 产物身份校验 + 冒烟通过 | 修 2 bug(产物身份校验+k 投票+generate_candidates 签名)，冒烟 2 轮全 ACCEPT，产物 hash 真改(b98a9abd->12633201) |
| evolver 主结果坐实 | 2026-07-31 | 4-seed test 复评 + AIME24 | evolver 2-iter +16.1pp 4-seed 坐实(t=9.02 CI[+0.126,+0.196])定主结果；AIME24 跨任务泛化失败(+1.9pp t=0.70 不显著)为泛化边界负结论 |
| 10 轮迭代消融 | 2026-08-03 | 全量 10 轮长跑 + test 复评 | 10 轮 +13.7pp 坐实(t=8.10)但未超 2 轮(-2.4pp 不显著)；val 过拟合 select set；2 轮是甜点，主结果不变 |
| 超 SkillOpt 尝试 | 2026-08-03 | seeded-evolver + 16-seed 复评 | evo_seed 8-seed +1.65pp(p=0.053)曾判趋势级，16-seed 收窄至 +0.59pp(p=0.42) parity；推翻"超 SkillOpt"乐观判断 |
| 机制消融 + AIME 从头链 | 2026-08-04~05 | Task1 消融 + Task2 AIME + Task3 跨agent | Task1 干净(M1+M2 显著)；Task2 AIME 负(优化器卡 S₀)；Task3 冒烟通过待全量 |

## A.1 起点与设计奠基（2025-11 ~ 2026-06）

- **2025-11~12**：8 卡 verl GRPO 基础链路搭建，rollout 日志、评测脚本、训练监控成型
- **融合设计文档**（`docs/training_free_grpo_fusion_design.md`）：复现 Training-Free GRPO（论文 +5.4%，自测 AIME +3.3%），提出三层融合方案——①A 模型经验蒸馏+ICL 注入 ②B 升级为带 python_executor 的 Coding Agent ③Training-Free 预热 -> GRPO 微调。明确警示：HumanEval/MBPP 信号太弱，不要停留。
- **BAgent 架构落地**（`agent/b_agent/ARCHITECTURE.md` / `DIRECTION.md`）：自建薄层 Agent（~500 行/模块），核心理念是**上下文组件（SOUL/Skill/Memory/Tools/Rules）独立进化、独立版本、独立评分**。设计三层反馈：标量 reward（高频->A 的 GRPO）/ 因果 ATE（中频->SkillLibrary 权重）/ 结构化归因（低频->Skill 文本创建改写）。推荐 Direction C（Curator+Generator 双环混合架构）。

## A.2 早期正结果（2026-07-01 ~ 07-03）

- **消融矩阵**（n=100）：MBPP Pro SD **p=0.0009**、Humaneval Pro SD **p=0.0048** 均显著；APPS_Plus Hard SD p=0.39（-18.6% 回退）。
- **Task #1/#5 改造落地**（`docs/improvement_plan_task1_task5.md`）：修复 YAML 触发词序列化 bug；为 BAgent 加 `python_executor` 自调试工具。Self-Debug 修复参与率 35-43%（预期 15-25%），pass@k 提升 +11.6%(MBPP)/+11.4%(Humaneval)。
- **论文大纲 v1**：标题定为 "From Parameter Space to Context Space"，主打 self-debug + context curation。

### A.2.1 名词解释：p 值怎么算、消融四组 skill 怎么生成

（统计方法实现于 `agent/b_agent/ablation_test.py`。）

**① p 值怎么算出来的（配对样本 t 检验，`scipy.stats.ttest_rel`）**

- 消融跑 **n=100 道题**，每道题在**完全相同条件下**跑全部组——同一道题每组都有一个分数，构成"配对"。
- 每题每组的分数 = composite = 0.7×pass_rate + 0.3×C-judge。100 题 -> 两个长度 100 数组，如 `baseline_scores` vs `selfdebug_scores`。
- 对每题算差值 `d_i = s_i − b_i`，检验**原假设 H₀：差值总体均值=0**（两组无差别，observed 差异纯随机波动）。统计量 `t = d均值 / (d标准差/√n)`，查 t 分布（自由度 n−1）得 p 值 = "H₀ 成立时出现这么大差异的概率"。
  - **p=0.0009**（MBPP Pro）：随机波动出这么大差异的概率仅 0.09% -> 拒绝 H₀，SelfDebug 真有效（p<0.001 高度显著）。
  - **p=0.39**（APPS_Plus Hard）：39% 概率纯靠波动出现 -> 无法拒绝 H₀，且均分 0.48 < Baseline 0.59（-18.6% 方向反了）-> 无效/回退。
- **配对设计控制了题目难度混杂**：每题自己和自己比，难题对两组都难，不污染结果。这就是同样 100 题能检出 +0.85 分微小稳定效应的原因。经验法则：p<0.05 显著、<0.01 高度、<0.001 极高度。
- ⚠️ **诚实补充**：这组 07-01 正向 p 值后来在 07-13 体检中被质疑——APPS Hard 实际只有 1/100 题真正修复（96 题触发循环但几乎全失败，p=0.39 真实）；MBPP/Humaneval 的 p=0.0009/0.0048 当时成立但**缺多 seed 复现**，且 Step1 在 BigCodeBench/MBPP_pro 重测得**双阴性**（p=0.349/0.819）。属早期正向信号，论文能否站住看后续复测。

**② 四组 skill 怎么生成（都来自同一个 402 条 SkillLibrary 池）**

402 条 skill = 从 GRPO trace 日志蒸馏出的经验（`experience_pool.json` -> `agent/b_agent/skills/G1~G401.md`），每条是一段带 YAML frontmatter 的程序性指令文本。四组区别只在**怎么挑、挑几条、怎么塞进 prompt**：

| 组 | 怎么挑 skill | 挑几条 | 谁挑 | 目的 |
|---|---|---|---|---|
| Baseline | 不注入 | 0 | - | 纯模型基线 |
| Random | `random.sample` 等概率随机抽 | 5 | 随机 | 噪声对照（涨分非随机功劳） |
| Curated（早期） | 关键词 triggers 匹配检索 | 5 | 规则检索 | 验证"精选有用" |
| Curated（后期） | 全量 compact 注入 + Meta-Skill 引导 | ~50 | 按 score 排序全塞 | 与 GRPO 训练路径对齐 |
| Focused-3（07-14） | LLM 当选择器看题挑 | 3 | B 模型选 | 验证 GraSP 聚焦假设 |

- **Random**（`ablation_test.py:638`）：`random.sample(skill_lib.active, 5)` 从 402 条等概率抽 5 条，不看题目——纯噪声基线。
- **Curated（早期）**：用 skill 的 `triggers` 触发词做关键词检索命中 top-5（<100 条规模不上 embedding）。**这版得到 p=0.0118 显著**。
- **Curated（后期，Step1）**（`_build_full_injection_system:478`）：换成全量 compact 注入——每条 skill 压成 ≤150 字符短描述，按 score_delta 排序取 top-50，前置 Meta-Skill 引导语（"建议非硬规则、不相关就忽略"）。**这版显著性消失（p=0.349）**——retrieve-k=5->full-injection-50 是 07-13 反转的代码级根因。
- **Focused-3**（`_select_focused_skills_llm:505`）：把题目+全部 skill 目录丢给 LLM，让它 JSON 输出 top-3 相关 ID。结果 vs Baseline **p=0.828 仍 null** -> 证明问题不在"挑几条/怎么挑"，而在**池质量本身**（绑定约束）。

**③ 一个反直觉现象的解读**：APPS Hard 里 Random(0.69) 竟 > Baseline(0.59)。因为 402 条都是通用编程 heuristics，随便抽几条都"有点用又有点害"，净效应是噪声——这恰恰说明池子里 skill **不专一、不针对具体任务**，为后来"池质量是绑定约束"结论埋下伏笔。

## A.3 诚实体检与方向重定位（2026-07-13 ~ 07-14）⚠️ 关键转折

- **全面审计**（`status-review-20260713`）：工程 ~50%、训练 ~5%、实验 ~20%、理论 ~20%。**GRPO 停在 step 637（2026-06-29）两周未动，无 A checkpoint**；402 条 skill **全部 ATE=0、usage_count=0**；P1 剪枝显示 21/30 有害（70%）。
- **Step 1 双阴性**（`step1-validation-result`）：BigCodeBench n=57 **p=0.349**；MBPP_pro 5 组 n=100 **全部 p>0.19**。**skill 注入对 pass-rate 零收益**；Meta-Skill 趋势有害；旧 retrieve 路径的显著性（p=0.0118）在切到全量注入后消失。
- **P0 focused-injection**（`p0-focused-injection-20260714`）：Focused-3 vs Baseline **p=0.828** 全 null。**判别结论：池质量是绑定约束，不是注入数量/选择**。注入侧已穷尽。
- **导师 4 建议**（`advisor-feedback-20260713`）：①SkillBench 调研 ②多 agent 验证 ③ACP 协议 ④skill 优化综述。07-14 review 重排优先级：**"先单 agent 有效再谈泛化"**，多 agent 冻结、ACP 远期。
- **SkillOpt 选型**（`research-direction-skillopt-20260714`）：选 SkillOpt（Microsoft, arXiv:2605.23904）为 testbed + 最强基线——pip 可装、无需 Docker、6 benchmark×7 模型×3 harness 全 best-or-tied、GPT-5.5 +23.5pp。论文定位：SkillOpt 证明"skill 可优化且有 headroom"，我们补上"创建阶段（trace 蒸馏）+ 池化检索 + GRPO 信号"。

## A.4 代码域判别实验（2026-07-15）

- **SkillOpt Step 0（数学）**（`skillopt-step0-results`）：LiveMathematicianBench test=124，noskill 24.19% -> seed 32.26% -> **best 54.03%（+21.77pp）**，干净复现，test 泛化成立。
- **SkillOpt × 代码**（`skillopt-code-result`）：mbpp_pro val +21pp 但 **test 0.513 < noskill 0.528（-1.5pp）**，过拟合 38 题 val。扩大 val（38->75）仍治不住（best test 0.4604 < baseline 0.4679）。**代码单函数域对 skill 优化不友好**。
- **数据集硬约束**（`dataset-constraint-20260715`）：小论文主数据集锁定为**代码**，数学仅补充（后被 07-24 方向切换推翻）。

## A.5 数据集博弈（2026-07-16 ~ 07-18）

- **基础设施**：DeepSeek-V3.2 API 硬约束并发 ≤30 RPM，实现进程级 token bucket + 429 退避重试（防污染指标）。
- **APPS_Plus 主基准**（interview 3952 题，stdin/stdout）：07-17 test=200 得 best 0.635 vs noskill 0.575（**+6pp，代码域首个正结果**）；07-18 全量 test=2765 **反转为 -2.1pp**（best 0.6341 < noskill 0.6564）。**+6pp 系小样本假阳性**。代码方向再次存疑。

## A.6 污染排除与代码域判死（2026-07-20 ~ 07-21）

- **LCB 污染假说**（`lcb-contamination-task`）：LCB noskill baseline 0.43（零 429）远低于 APPS_Plus 0.66，怀疑是v3.2已经训练过这些代码数据集。
- **LCB 分层 baseline**（`lcb-stratified-baseline`）：OLD(seen) 0.708 vs NEW(unseen) 0.427；信号全在 hard（0.54 vs 0.18，+0.355）；easy 天花板无信息。**无污染 hard 真实能力仅 0.18，headroom 巨大**。
- **LCB NEW optimize**（`lcb-new-optimize-20260721`）：val +15pp 但 **test -0.7~-1.4pp 过拟合，hard 零收益（0.193->0.193->0.181）**。无污染条件下 skill 优化仍不泛化，排除最后混淆变量。**代码域判死**。
- **负结果根因**（`code-skill-negative-rootcause`）："skill 对代码无用"是错的——综述 SWE 类 + 5 篇代码域论文（SkillComposer +23pp / SePO MBPP +4.49）有效；我们负根因 = 通用 prompt 文本 skill + stdin 竞赛题 + 全量注入。
- **核实先例**（`skillcomposer-sepo-verified`）：PDF 原文核实 SkillComposer（2606.06079）LCB v6 per-task online 有效，SePO（2606.04465）MBPP 91.2->96.2。代码 skill 同类正先例成立。

## A.7 创新点确立与端到端打通（2026-07-21 ~ 07-22）🔥

- **创新点定型**（`our-grpo-skill-method-design`）：**A 模型 GRPO 改写初始 skill S₀，reward = delta pass-rate**（skill_pass − s0_pass，隔离 skill 贡献）。别人方法都是 baseline，我们训 A 让 skill 更好。
- **端到端冒烟通过**（`grpo-skill-smoke-passed`）：本机 1 卡 Qwen3-0.6B verl 1-step COMPLETE——A 改写 skill -> B 生成代码 -> LCB 单测 -> delta reward -> GRPO 更新全链路打通，hard 题 delta>0（abc380_g：S₀=0 -> 改写后 B 全对）。修 5 个阻塞 bug（S₀ backend / reward key / prompt 爆长 / **test_code 截断保 delta 可比** / return_dict）。
- **GPU 策略**：本机 1 卡 L40（0.6B 冒烟），正式训练申请托管 8 卡 L40（4B）；reward 占 79% step 时间（API 是瓶颈非 GPU）。

### A.7.1 名词解释：S₀ 是什么、谁在"优化 skill"

- **S₀ = 初始 skill 文本**：就是 `skillopt/envs/lcb/skills/initial.md`（LCB 版 1796 字符），一份手写的通用系统提示词/skill，内容是竞赛题编程 heuristics（边界处理、正确性纪律、常见陷阱等程序性指令）。它**是 B 模型 system prompt 的一段**——B 生成代码时 context 里会注入它。
- **S₀ 本身不优化任何东西，它是被优化的对象（起点）**。**A 模型才是优化器**，通过 GRPO 学会怎么改写 S₀。
- **完整流水线**：
  ```
  A 输入: S₀ + 问题 (+失败模式反馈)  ->  A 输出: 改写后的 skill
  B 输入: 改写后的 skill + 问题     ->  B 输出: 代码
  C/单测: 评判代码                  ->  reward = pass(改写skill) − pass(S₀)
  GRPO: 用这个 delta 当奖励更新 A
  ```
- **为什么从 S₀ 优化、而非从零生成**：从零生成时 A 容易走捷径输出通用废话 prompt 拿基础分；给 S₀ 作起点强制它做"改进"。**delta 奖励隔离 skill 贡献**——绝对 pass rate 会被题目难度污染，delta 只看 skill 带来的增量（呼应 D2Skill 双 rollout、SkillComposer 也用 delta pass rate）。
- **退化检测**：A 最稳的捷径是原样复制 S₀（delta=0 无负奖励），所以加 `similarity(改写, S₀) > 0.95 -> 直接 floor`，逼 A 真改。
- **与别人的区别（创新点所在）**：见正文 2.4 表。

## A.8 双线并行与方向纠偏（2026-07-23）

- **双线方案**（`training_vs_training_free_parallel_plan`）：训练线（8 卡 GRPO，**Qwen3-4B A 模型改写 skill**）vs 免训练线（TF-GRPO，零 GPU，**DeepSeek-V3.2 当 rewriter 改写 skill**，冻结不更新参数，借用 GRPO 群组相对对比思想而非 RL 梯度）。两线共用 B/C/S₀ 同表对位；免训练线作强 baseline（对标 SePO），训练线作创新点。
  - **免训练线的优化信号**：rewriter 生成多个 skill 变体 -> 每个变体喂给 B 生成代码 -> 跑单元测试得客观 pass-rate -> delta = skill_pass − s0_pass -> 群组内高/低 delta 对比蒸馏经验 -> 注入下一轮 rewriter。**反馈来自执行结果（+C 模型评判）这种外部信号，不是模型自我反思错误**。
- **TF-global 反转**（`tf-global-reversal-samecond`）：同条件重测 S₀=0.6005/hard 0.3012，TF-global 0.5828 反而更差；之前 +6pp 是旧 key baseline 偏低假象。
- **TF per-task 正结果**（`tf-pertask-positive`）：per-task 实时改写 + 经验池，LCB hard 0.3012->0.3855（**+8.4pp**），印证 SkillComposer per-task 有效，但非泛化目标。
- **方向纠正**（`direction-global-not-pertask`）：per-task 不泛化（每题一份不通用）；真问题 = **TF-global 比 S₀ 差**，借鉴 SkillComposer hybrid（offline 领域 library + 轻量特化）。

## A.9 当前转向：暂试数学 bench（2026-07-24）

- **数学 bench 全量基线坐实**（`math-baseline-sameond`）：test=124 同条件 noskill 29.0% / S₀ 33.1% / **best 51.6%（+18.6pp）**。Step 0 正结果复现非旧 key 假象。数学是干净试验场，headroom +23pp。
- **导师建议跳出代码**：暂试 LiveMathematicianBench 作主试验场，代码线不放弃、降级为对照。推翻 dataset-constraint 的"数学仅补充"。

### 2026-07-24：数学 bench (LiveMathematicianBench) 切入 + TF-global 结果

**背景**：导师建议跳出纯代码 bench。切数学作主试验场（代码线降级为对照，不放弃）。数学 env 已就绪于 `/data/zhuldz/skillopt`（确定性 `<answer>` 验证、单轮、无 Docker、本机零 GPU）。

**同条件全量基线**（test=124，新 key msk-4601b27，greedy，0 个 429）：

| 方法 | test acc | Δ vs S_0 |
|------|---------|----------|
| noskill | 29.0% | -3.9pp |
| **S_0（seed `initial.md`）** | **33.1%** | 基线 |
| SkillOpt best（per-task optimize，别人方法） | 51.6% | +18.6pp |

Step 0 旧正结果（seed->best +21.8pp）在同条件下复现（+18.6pp）——数学是干净试验场，headroom +23pp，远胜代码 LCB hard 死区。

**TF-global 数学版（我们，免训练，零 GPU）**：`tf_skill_optimizer_math.py`，T=3 g=4 train=35，rewriter=DeepSeek-V3.2，delta 信号基于同条件干净 cache（53 题）。

- 训练侧：train mean best_delta 持续正（+0.23~+0.29），rewriter 确实在 train 上把 S_0 错的题改对（best_delta=+1.0 真撬动）。
- val 门控：iter2 ACCEPT（val delta +0.056 vs baseline -0.111），iter1/3 回归被拒。
- **test=124 = 0.3306（33.06%）≈ S_0，持平未超（-0.04pp）**。

**结论**：TF-global 在数学上也未能超过 S_0，与代码方向（TF-global<S_0）一致——**global 改写对强模型 + 好初始 skill 收益甚微**。但数学≠代码：数学 train 侧 rewriter 真撬动 S_0 错题（代码 LCB hard 零收益），有 headroom，只是 global 形态不泛化。**per-task 才是路径**（SkillOpt per-task 51.6% 已证；我们的 TF-per-task 待测）。⚠️ 此结论后被 07-27 v2 修复推翻——不是"global 对强模型无效"，是 v1 优化器有 6 bug。

## A.10 阴性诊断 + v2 修复 + 数学域 trace2skill（2026-07-27）🔥🔥🔥

### 起因与诊断

7-24 的 TF-global 数学版 test=0.3306≈S₀（持平）被判定阴性。用户要求诊断"是 trace2skill 提取基础不好，还是优化不好"，并放弃 8 卡 4B 训练、坚持免训练 + global 形态做出效果。

**根因诊断（代码级证据）**：两层问题，都需要修。

1. **优化机制坏（v1 的 6 个 bug）** - 数学 TF 线 `tf_skill_optimizer_math.py` (v1) 有 6 个结构性 bug：
   | # | Bug | 证据 |
   |---|-----|------|
   | 1 | 经验池完全卡死 | `experience_iter{1,2,3}.md` 三份 md5 相同，merge 输出=输入，3 轮积累=0 |
   | 2 | 蒸馏信号是 prompt 措辞技巧而非失败模式 | 模板 Generalization Check 拒绝问题特定内容只留通用写法 -> pool 全是 S₀ 已有陈词滥调 |
   | 3 | rewriter 从未见真实错误 | 蒸馏只给 skill 文本+delta，不给 B 实际选了什么 |
   | 4 | theorem/sketch 字段未用 | 每题有定理+证明，rewriter/B 只看 question+choices |
   | 5 | compose 只是 S₀ 换皮 | 把 S₀+通用原则重新措辞=S₀ 已有内容，对强模型纯干扰 |
   | 6 | val 门控太弱 | val=18，+0.056=1 题翻转被接受 -> test 持平 |
   一句话：优化器没产出任何 S₀ 还没有的信息 -> global skill=S₀ 换皮 -> 强模型严格≤S₀。

2. **基础从未在数学域建立（用户的二次纠正）** - 现有 `experience_pool.json`（402 条）是**代码域** trace2skill 产出（源=GRPO `readable_trace_*.log`，A->prompt->B->**代码**->单测），每条代码 prompt 技巧（"写可执行代码""处理空输入"），对数学 MCQ 完全无用。数学域从未跑过 trace2skill，根本没有数学域经验基础。v1/v2 都是在线从零蒸馏，没有可复用的数学域经验池。

### 方法（两条互补的免训练 global 路径，均与代码域 pool 物理隔离）

**公共设计原则**（修 v1 bug #2/#5 的核心）：从"蒸馏通用 prompt 写作技巧"转向"蒸馏可泛化数学失败模式 -> 追加注入 S₀"。global skill 要超 S₀，必须往里加 S₀ 没有的、跨题可复用的**数学诊断程序**，而非 prompt 措辞。合成采用**追加式**（S₀ 原样 + `## Common Failure Modes` 节，不重写 S₀ 避免干扰强模型）。失败模式蒸馏模板（新建 `experience/templates_math.py`：`FAILURE_MODE_TEMPLATE`/`FAILURE_MERGE_TEMPLATE`/`COMPOSE_APPEND_TEMPLATE`）**反转**泛化规则：拒绝 prompt 措辞技巧、拒绝单题答案、保留可泛化数学诊断程序（量词翻转/等号情形/存在 vs 唯一/正则性强度/子集 vs 相等）。

**路径 A：离线数学域 trace2skill**（新建 `experience/extractor_math.py`，对齐代码域 `extractor.py` 的 4 阶段理念但数学域适配）：
- Stage1 生成数学 trace（持久化 jsonl：每 rollout 含 id/theorem/skill/pred/correct/pass/delta/response_excerpt）-> Stage2 按题蒸馏失败模式 -> Stage3 跨题合并 -> Stage4 追加式 global 合成。
- 产出 `experience_pool_math.json`（version=math_trace2skill_v1，isolated_from=experience_pool.json code domain）+ `skill_math_trace2skill.md`。
- 与 v2 区别：离线（生成一批 trace -> 蒸馏一次 pool -> 合成一次），trace 持久化可审计/可重蒸馏，给基础池。

**路径 B：在线闭环优化器 TF-global v2**（新建 `rl/train/verl1/tf_global_v2_math.py`，v1 保留作对照）：修 v1 全部 6 bug - rewriter 见 theorem + 保留 B 的 pred/response 喂蒸馏（#3/#4）、失败模式模板反转泛化（#2）、merge 显式增量 + hash 校验防池卡死（#1）、追加式合成不重写 S₀（#5）、bootstrap CI 门控 + multi-pass 一致性（#6，95% CI 下界>0 才接受，不再接受 1 题翻转噪声）。每轮 pool 演化驱动下一轮 rewriter。

### 实验结果（test=124，同条件，新 key msk-4601b27，greedy，0 个 429）

| 方法 | test acc | Δ vs S₀ | 性质 |
|------|---------|---------|------|
| noskill | 0.2903 | −3.9pp | 基线 |
| **S₀（seed initial.md）** | **0.3306** | 0 | 基线 |
| **trace2skill-math**（路径 A，离线，35题×g4） | **0.3548** | **+2.4pp** | ✅ 数学域基础池 modest 正结果 |
| **TF-global v2**（路径 B，在线闭环，CI门控 iter3） | **0.4839** | **+15.3pp** | ✅✅ 强正结果（⚠️ 后被 07-28 诊断为评测噪声假阳性，真实 +7pp） |
| SkillOpt-best（别人方法，per-task） | 0.5161 | +18.6pp | 上界对照 |

**关键里程碑**：
- **7-24 的"TF-global 对强模型无效"结论被推翻**：修复优化器 6 bug 后，TF-global v2 大幅超 S₀，且 global 形态（非 per-task）。失败原因不是"global 对强模型无效"，而是 v1 优化器没产出新信息 + 没有数学域基础。
- **数学域 trace2skill 基础池首次建立**：路径 A 产出与代码域隔离的 `experience_pool_math.json`，test +2.4pp 证实数学域基础池本身就有 modest 增益。
- v2 门控诚实：iter1/2 均被正确拒绝，仅 iter3 被接受。对比 v1 接受 1 题翻转噪声导致 test 持平。

### 教训
- "global 改写对强模型无效"是错误结论，根因是优化器没产出新信息（v1 6 bug）+ 没有数学域基础。修对优化信号（失败模式 vs prompt 措辞）+ 追加式合成（不干扰 S₀）后 global 能大幅超 S₀。
- 失败模式蒸馏必须看到 B 的真实错误（pred vs correct + 推理节选）+ 定理原文，否则只能猜出泛泛之论。
- val 门控必须用 bootstrap CI 而非裸 delta，否则 1 题翻转噪声会被接受、test 不泛化。
- 域隔离：代码域经验池对数学无用，数学域必须独立建 trace2skill 基础。

### 产出文件
- `experience/templates_math.py`（失败模式蒸馏模板）
- `experience/extractor_math.py`（数学域 trace2skill，4 阶段）
- `rl/train/verl1/tf_global_v2_math.py`（TF-global v2，修 6 bug）
- `experience/outputs/math_trace2skill/run_20260727_134528/`（trace jsonl 140 rollouts + experience_pool_math.json + skill_math_trace2skill.md）
- `rl/train/verl1/outputs/tf_skill_math_v2/run_20260727_100047/`（v2 run，含 tf_global_skill_v2_math.md + tf_run_log.json）
- test 评测：`/data/zhuldz/skillopt/outputs/eval_trace2skill_math_0727`、`eval_v2_global_math_0727`

## A.11 多 seed 复测启动 + skill 自进化迭代器（2026-07-27 下午）

- **+15.3pp 单次需复测**：v2 test=0.4839 是单次优化采样结果，项目史多次小样本假阳性->复测反转（APPS_Plus 07-17 +6pp->07-18 -2.1pp），必须多 seed 坐实。
- **随机性定位**：评测端 skillopt `eval_only.py` 是 greedy(temp=0.0)+固定 split seed42，对同一 skill 文件 test acc 可重现、**无评测噪声**；随机性全在优化端（v2 rewriter temps 0.7~0.9、distill/merge temp=0.3）。故复测 = 多次重跑全量 v2->各产出不同 skill->各自 greedy 评测 test=124，看 +15.3pp 是否稳定重现。无需改代码加 seed 参数（每次 API LLM 采样自带随机性即一次独立 seed）。⚠️ 此"评测无噪声"判断后被 07-28 推翻：temp=0 在 vLLM 网关上非确定。
- **复测方案**：3 seed 串行（28 RPM 全局硬限，并行只抢配额致 429 污染），复用 s0_cache（53 题）省预计算。
- **冲突与让位**：启动复测时发现 `skill_evolver_math.py --iters 8 --K 3`（16:38 启动，pid 66951）正在跑——一个以 v2 当前 skill 为起点、融合 SkillOpt+GEPA+ReSkill 的免训练自进化迭代器，目标超 SkillOpt-best 0.5161，iter1 已达 val delta=+0.302（CI+0.170，**强于 v2 iter3 的 +0.222**）。两实验共用 28 RPM 全局配额，并行必互相 429 污染。**用户决策：暂停复测让 evolver 先跑**（kill 我的 seed1，evolver 存活）。
- **自动接力**：watcher 脚本(`/tmp/wait_and_relay.sh`, pid 24085)后台每 60s 检测 evolver 是否结束->等 5min 让 RPM 恢复->自动串行跑 3 seed 全量 v2。
- **skill_evolver_math 设计**：局部编辑+多候选(K=3)+帕累托选择(bootstrap CI)+失败驱动 patch（锚定当前仍错题产出具体决策程序，非通用 checklist）+拒绝缓冲区+慢更新去冗余；选择集用 train+val(53 题)替代 val(18 题)降门控噪声。

### 2026-07-28：v2 +15.3pp 假阳性诊断 + 评测 bug 修复 -> 真实 +7.0pp 🔥🔥🔥

**起因**：07-27 续章的 3-seed 复测在 07-28 02:53 完成，3 个 seed 全部 CI 门控拒绝，best skill 退回 S₀（1017 bytes）。用户怀疑"不是方法假阳性，是筛选/优化机制设计问题"，暂停记录先诊断。

**根因 = 2 个评测设计 bug（非方法失效）**：
- **Bug 1（temp=0 非确定）**：DeepSeek-V3.2 网关 vLLM 后端在 temp=0 下，batch 大小 + 并发 + 浮点累加顺序不同 -> 输出不完全可复现。v2 用 `ThreadPoolExecutor(max_workers=8)` 并发 val，每次 batch 组合不同 -> 同一 S₀ 跑出 4/18、6/18、7/18 不同结果，delta 混入 ±2~3 题纯评测噪声（18 题上 +1 题翻转=+0.056，正是门控抖动量级）。
- **Bug 2（baseline/candidate 不对称）**：第 337 行 baseline `eval_on(passes=1)` 单次采样撞到 4/18 偏低 -> base_delta=-0.111 虚假低基线；第 371 行 candidate `eval_on(passes=2)` 两次取平均噪声减半 -> 看似 delta 更高。**+15.3pp 大概率是 baseline 采样偏差，不是真增益**。
- **附带：llm() temperature 形参死参**：chat_target 全链路不接受 temperature，靠模块级单例 `TARGET_CONFIG.temperature`；v2 llm() 的 temperature 形参从未传入 -> rewriter/distill/merge 实际全在 temp=0.0 跑（无采样多样性）。

**修复（`tf_global_v2_math.py`）**：
1. k 次投票：`run_one` 加 k 参数，每题采样 k 次（EVAL_TEMP=0.3），pass_rate=通过次数/k（连续值降噪声）。
2. S₀ 对称：`precompute_s0` 也用 k 投票。
3. 固定基线：baseline 用 `s0_cache` 固定快照不现场重跑，base_delta=0。
4. 真采样：`set_temp()` 写 TARGET_CONFIG.temperature，GEN_TEMP=0.8（rewriter 真采样）、EVAL_TEMP=0.3（评测）。
5. 移除 val_passes，改 `--eval_k`（默认 5）。
6. 扩 val：split 1:3:6（train=18/val=53/test=106），seed42，bootstrap CI 功效 ~3x。

**结果**：
- val 3-iter 全 ACCEPT 单调递增：+0.079（CI+0.042）-> +0.083（CI+0.045）-> +0.098（CI+0.053），best skill = S₀ + 追加数学诊断节（3655 chars，非换皮）。
- **test=106：S₀ 0.2906 -> v2 0.3604（+7.0pp）**，泛化成立（val +9.8pp CI+5.3pp -> test +7.0pp）。
- **最终定论**：+15.3pp 是评测噪声假阳性（baseline 采样偏差）；修复后真实增益 = val +9.8pp / test +7.0pp，统计显著可复现。方法（TF-global v2 失败模式蒸馏 + 追加式合成 + CI 门控）有效。
- run：`rl/train/verl1/outputs/tf_skill_math_v2/run_20260728_131934`

**教训**：temp=0 ≠ 确定性（vLLM 网关级）；baseline 与 candidate 必须同 pass 数同条件；chat_target 无 temperature 参数，采样温度靠 TARGET_CONFIG 全局单例，多阶段流程要在阶段入口显式 set_temp。

## A.12 test=106 多 seed 复测坐实 v2 +6.6pp + evolver 主结果推翻（2026-07-29）🔴🔴🔴

**task1 - v2 +7.0pp 多 seed 复测（坐实）**：
- 配置：k=3 投票，2 seed，test=106，S₀+v2，`eval_test_multiseed.py`（行缓冲 + 断点续跑）。
- 结果（k=3 同条件）：

| seed | S₀ | v2 | Δ |
|------|------|------|------|
| 1 | 0.2673 | 0.3585 | +0.0912 |
| 2 | 0.3082 | 0.3491 | +0.0409 |
| **mean** | **0.2877** | **0.3538** | **+0.0660 (+6.6pp)** |

- 两 seed 均为正，per-qid 配对 t=2.78（df=105, p≈0.0064），bootstrap 95% CI=[+0.0204, +0.1132] **下界>0**。正向 40/负向 19/持平 47，增益非少数题驱动。
- **v2 +6.6pp 统计显著可复现，坐实 07-28 +7.0pp；之前 +15.3pp 确为噪声假阳性，真实增益 +6~7pp。**

**task2 - evolver +15.7pp 推翻（关键纠偏）**：
- 复审 evolver `run_20260727_163814` 发现：`best_skill_evolved.md` 与 8 个 `skill_iter{1..8}.md` 的 md5 全部相同（`911597f7df...`），且 = evolver 的 **init_skill** -> **evolver 8 轮迭代从未修改 skill，best == init**。
- 根因：`val_passes=1` 单次采样噪声 + `cands.append(best_skill)` 保底回退进候选池 -> winner 常是 current 的重采样，`improved=True` 但 `best_skill` 重赋值为同一字符串 -> 8 轮原地踏步被误判改进。LOST/CI 门控被单次采样噪声绕过。
- 那 "+15.7pp" 实为 **v2_skill_0727 vs v2_skill_0728**（两版 v2）的对比，与 evolver 机制无关。evolver 主结果候选**撤回**。
- **v2 +6.6pp 不受影响，独立成立。**

**evolver 修复（07-29，代码改完）**：复用 v2 的 k 投票评测机制（删本地单次采样版）；删保底回退竞争，winner 仅在 `md5!=current_hash 且 ci>0` 的 novel 候选里选（产物身份校验），accept 时 `assert md5(winner)!=current_hash`；CLI `--val_passes`->`--eval_k`；split->1-3-6；起点->run_20260728_131934。

**教训**：优化器产物必须先验证 `hash(output)!=hash(input)`，否则"增益"是噪声——产物身份校验是第一道关；val_passes=1 单次 0/1 评测在 temp>0 下 delta 抖动 ±0.06~0.13 足以伪造"improved"。

## A.13 evolver 冒烟暴露并修复第二个 bug -> 冒烟通过（2026-07-30）✅✅✅

**起因**：启动修复后 evolver 的冒烟测试（iters=2/eval_k=3/train_n=5，DeepSeek-V3.2 API，28RPM）。

**第二个 bug - generate_candidates 签名错误致 0 novel**：
- 冒烟 Iter 1 报 `generated 4 candidates, 0 novel` -- 4 个候选全部 hash == current_skill（b98a9abd），patch 生成全部失败回退。
- 关键观察：`cand0: delta=+0.067 CI=+0.006 novel=False`（current 重评测噪声），**07-29 修复的 novel_positive 过滤正确将其排除出 winner**（即使 CI>0 也不接受），证明产物身份校验逻辑生效。
- 根因（代码级）：`generate_candidates` 把 `dict(max_tokens=4096, stage="evolver_patch")` 作为**第三个位置参数**传给 `llm()`，但 `llm(system, user, *, max_tokens, stage)` 的 max_tokens/stage 是 **keyword-only** -> `TypeError` -> 被静默 `except Exception` 吞掉 -> 4 候选全 fallback `current_skill` -> 永远 0 novel。
- **修复**：第 198 行改为 `ex.submit(llm, PATCHER_SYSTEM, user, max_tokens=4096, stage="evolver_patch")`（dict 解包为 kwargs）；except 改为打印异常（不再静默吞，防"0 novel 假象"无法诊断）。验证：修复后 generate_candidates 4/4 novel。

**冒烟最终结果（run_20260730_111306）**：

| 阶段 | val delta | CI_lower | skill_len | hash |
|------|-----------|----------|-----------|------|
| init(v2) | +0.038 | -0.018 | 3655 | b98a9abd10 |
| Iter1 ACCEPT | +0.176 | +0.109 | 3906 | 1053fda28a |
| Iter2 ACCEPT | +0.199 | +0.124 | 4670 | 1263320145 |

- 两轮共 8 候选全部 novel（4/4 + 4/4），07-30 签名 bug 修复确认有效。
- **产物身份校验通过**：best_skill_evolved.md hash（1263320145）≠ init（b98a9abd10）-- evolver 第一次真正进化出新 skill。
- 增益单调递增 +0.038->+0.176->+0.199，CI 下界全正且递增；select set pass 0.276->0.437（+16.1pp）。

**evolver 加了什么（skill 内容对比）**：
- 保留 v2 全部结构原样不变。
- 改动聚焦 2 处（对应 Iter 1 focus=implication/biconditional、Iter 2 focus=existence/uniqueness）：
  1. 第 26 行 **Meta-Statement Evaluation** 改写为更具体的判定程序：比较定理的*已证*内容与其他选项，若定理只建立单向蕴含/弱 containment/非 sharp 结果，则 meta-statement 常为真。
  2. 新增第 29 行 **Meta-Statement Decision Procedure**（~700 chars 增量）：针对 "a stronger result can be proven" 类 meta 选项，给出 3 步具体测试（找最强显式结果 -> 查定理结论开放缺口：单向蕴含/非最 sharp 正则类/exists vs for all -> 缺口未闭合则 meta 真）。
- 符合 evolver 设计目标："具体 decision procedure 而非通用 checklist"，是追加式聚焦 patch 非换皮。

**⚠️ 关键提醒**：+0.199 是 select set（train+val）上的训练信号，evolver 是 test-free 选择。**能否泛化到 test=106 未知** -- 正是 v2 +15.3pp 假阳性教训。必须跑 test 复评才能定论。（后由 07-31 4-seed test 复评坐实 +16.1pp。）

**执行层教训**：28RPM + 429 退避使 evolver 极慢（2 轮冒烟 ~4h）。全量 10 轮将 ~20h 不可接受，正式跑须降 eval_k / 缩 select set / 题内 k 采样改并发（后由 max_workers 8->40 解决，2.2x 加速）。

## A.14 evolver 4-seed 坐实 +16.1pp + AIME24 跨任务泛化失败（2026-07-31）

### task1 - evolver 2-iter test=106 扩到 4-seed 坐实（论文主结果定型）

07-30 修复后的 evolver 2-iter 产物（hash 12633201）从 2-seed 扩到 4-seed，CI 大幅收窄，同时同 run 补测 v2 拆净增量：

| 对比 | mean Δ | 95%CI | t (pooled n=424) | per-seed Δ |
|------|--------|-------|------------------|------------|
| **evolver vs S₀** | **+0.1612** | [+0.126, +0.196] | 9.02 | +0.195/+0.151/+0.148/+0.151 |
| v2 vs S₀ | +0.0637 | [+0.037, +0.091] | 4.58 | +0.063/+0.091/+0.057/+0.044 |
| **evolver vs v2**（净增量） | **+0.0975** | [+0.065, +0.130] | 5.84 | +0.132/+0.060/+0.091/+0.107 |

- 四 seed 全正且高度显著，evolver ΔvsS₀ CI 从 2-seed [+0.124,+0.222] 收窄到 [+0.126,+0.196]。
- **evolver 相对 v2 仍有 +9.8pp 净增量**（CI 不跨 0）-> evolver 迭代本身有独立增益，不只是继承 v2。
- **evolver 2-iter（+16.1pp, t=9.02）定为论文主结果**。

### task2 - AIME24 跨任务泛化：负结论（泛化边界）

evolver/v2 在 LiveMathematicianBench（选择题）上的强正增益能否泛化到 AIME24（数值题，不同题源，完全 held-out）？这是终极跨任务泛化判据。

**评测配置**：AIME24（HuggingFaceH4/aime_2024，30 题，答案 0-999 整数），k=3 投票，3 seed，同 v2 条件；数值抽取（\boxed{}-><answer>->末整数）比对 groundtruth。

**结果**（3-seed, k=3, AIME24=30）：

| skill | seed1 | seed2 | seed3 | mean |
|-------|-------|-------|-------|------|
| S₀ | 0.633 | 0.611 | 0.578 | 0.607 |
| v2 | 0.600 | 0.622 | 0.578 | 0.600 |
| evolver | 0.611 | 0.600 | 0.667 | 0.626 |

| 对比 | mean Δ | t | 95%CI | 显著? |
|------|--------|---|-------|-------|
| evolver vs S₀ | +0.0185 | 0.70 | [-0.034, +0.071] | ❌ 不显著 |
| v2 vs S₀ | -0.0074 | -0.32 | [-0.054, +0.039] | ❌ 不显著 |
| evolver vs v2 | +0.0259 | 0.96 | [-0.027, +0.079] | ❌ 不显著 |

- livemath test=106 上 evolver +16.1pp（t=9.02）高度显著；AIME24 上 evolver +1.9pp（t=0.70, CI 跨 0）**不显著**。增益从 +16.1pp 暴跌到 +1.9pp 且不显著 -> **skill 学到的是选择题形态特有的诊断程序，不迁移到数值题**。
- **根因**：skill 内容针对 livemath 选择题（选项措辞检查、量化词判断、选项对比排除），在 AIME 数值题（无选项、需算出具体整数）上无用。
- **泛化边界明确**：同形态任务内泛化强，跨形态不泛化。

### 论文叙事影响

这是**有价值的负结论**，不是失败：
1. 诚实界定方法泛化边界：skill 优化在选择题基准内有效（+16.1pp），但不跨任务形态泛化。
2. 与文献对照：SkillComposer/SePO 也在同基准内有效，跨任务泛化是 open problem。
3. 主结果仍成立：evolver 在 livemath test=106 的 +16.1pp（4-seed 坐实）不受影响，AIME24 作为泛化边界证据补入。
4. 故事线修订：evolver livemath +16.1pp（主结果）-> AIME24 跨任务泛化失败（泛化边界，诚实报告）-> 未来工作：形态无关的失败模式蒸馏。

## A.15 evolver 全量 10 轮长跑 + test 复评（迭代深度消融，2026-08-01 ~ 08-03）

### 长跑

07-31 18:19 启动 `skill_evolver_math.py --iters 10 --eval_k 3 --K 4 --train_n 35`（起点 v2 skill，select set train+val=71 题），08-01 07:14 完成（~13h）。

- **产物身份校验通过**：best_skill_evolved.md md5 = `3888edefc1...` ≠ init `b98a9abd10`（evolver 真改了 skill，非原地踏步，07-29 教训已规避）。
- 进化曲线（val delta, select set）：init +0.042 -> iter9 ACCEPT +0.291 (CI+0.229) -> iter10 KEEP_PREV，收敛于 iter9 附近。skill 长度 3655 -> 4539 chars。
- val +0.291 明显高于 2-iter 的 +0.199 -> **是否 -> test 更好？需 test 复评定论**。

### test=106 复评（08-03，4-seed k=3，s0/v2 走缓存仅新算 evolver 4 seed）

| skill | mean pass | ΔvsS₀ | t (n=424) | 95%CI | 显著? |
|-------|-----------|-------|-----------|-------|-------|
| S₀ | 0.3011 | - | - | - | - |
| v2 | 0.3648 | +6.4pp | 4.58 | [+0.036, +0.091] | ✅ |
| **evolver 10-iter** | **0.4379** | **+13.7pp** | **8.10** | **[+0.104, +0.171]** | ✅ |
| evolver 2-iter（对照） | 0.4623 | +16.1pp | 9.02 | [+0.127, +0.196] | ✅ |

**关键对比**：evolver 10-iter vs 2-iter = **-2.4pp，t=-1.68，p=0.093，95%CI[-0.054, +0.004] 跨 0 -> 不显著**。

### 结论

1. **10 轮 evolver 坐实为强正结果**：+13.7pp，4 seed 全正，t=8.10 高度显著。方法稳健，非假阳性。
2. **但未超过 2 轮产物**：10-iter +13.7pp < 2-iter +16.1pp，差 -2.4pp（不显著，数值略低）。
3. **val 信号过拟合 select set**：10-iter val +0.291 > 2-iter +0.199，但 test 反而低 2.4pp -> 更多迭代在 train+val 上刷分更高，却没泛化到 test。val delta 不是 test 泛化的可靠预测器。
4. **2-iter 是甜点**：evolver 收敛快，~2 轮即达峰值，10 轮无收益甚至微退化。**2-iter（+16.1pp, t=9.02）保持为论文主结果**，10-iter 作"迭代深度消融"证据。

## A.16 超越强基线 SkillOpt 尝试（2026-08-03）——最终 parity

### 背景

用户要求：GRPO 训练线暂弃，**目标是在免训练设定下超过 SkillOpt**（Microsoft，rollout-reflect skill 优化），并据此评估论文可写性。SkillOpt-best = `/data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md`（test=124 = 0.5161）。

### Step 1：SkillOpt 对位 test=106（4-seed k=3）

| skill | mean pass | ΔvsS₀ |
|-------|-----------|-------|
| S₀ | 0.3011 | - |
| evolver 2-iter（原主结果） | 0.4623 | +16.1pp |
| **SkillOpt-best** | **0.5464** | **+24.5pp** |

**诚实负结论**：SkillOpt 比 evolver 2-iter 高 **+8.4pp**（paired t p=0.0002，显著）。原 evolver 主结果未超 SkillOpt。

### Step 2：Badcase 诊断（差距根因）

逐题对比 SkillOpt-wins vs evolver-wins：
- SkillOpt 赢的题 **77% 是 meta-option**（题干含"更强的结论能被证明/以下哪个能被加强"）
- evolver 赢的题仅 **7% 是 meta-option**
- 即 **SkillOpt 强 meta-option / 弱非 meta-option；evolver 正好相反 -> 互补**

根因：SkillOpt skill（9821 chars）重 meta-option 决策流程；evolver skill（4670 chars）重 Common Failure Modes，无系统 meta-option 处理。

### Step 3：朴素 splice 合并（失败）

构造 `merge_skillopt_evolver_manual.md` = SkillOpt 全文 + evolver 8 条非 meta-option failure-mode 检查。

| 子集(seed1) | merge | skillopt | evolver |
|------|-------|----------|---------|
| meta(87) | 0.563 | 0.602 | 0.513 |
| non(19) | 0.316 | 0.404 | 0.456 |
| all(106) | 0.519 | 0.566 | 0.503 |

**负结论**：merge 在**两个子集都输给 SkillOpt**（-4.7pp overall）。加 ~2500 字整段 failure-mode -> 模型注意力从 SkillOpt 简练启发式偏移 -> **稀释效应**，全局劣化。evolver 的非 meta-option 优势来自其整体结构，非可移植的 bullet。**朴素 splice 不可行。**

### Step 4：seeded-evolver（种在 SkillOpt 上，2-iter 最小 patch）

策略：把 evolver **种在 SkillOpt 上**（`--init_skill skill_v0002.md`），继承 SkillOpt 的 meta-option 优势，用 evolver 的**失败驱动最小 patch**（add/replace 小改，非整段）补 SkillOpt 的非 meta-option 短板。最小 patch 避免稀释。

工程加速：`max_workers` 8->16（API latency-bound ~7 calls/min 远低于 28RPM cap），实测 **2.2× 加速**（11.8 calls/min）。

evolver 轨迹（select 集 train+val=71）：
- init（SkillOpt）：pass=0.549, delta=+0.324
- Iter 1 ACCEPT：delta=+0.380（cand3, +5.6pp, 最小 patch +631 chars, hash cfae156d->f14e9a03）
- Iter 2 ACCEPT：delta=+0.390（cand1, +1.0pp, hash f14e9a03->72a95d13）
- 失败数 21->16（iter1 patch 有效）；按 v2 甜点经验取 **2-iter 产物**（skill_iter2.md, hash 72a95d13）。

### Step 5：test=106 复评 —— 8-seed 趋势级，16-seed 回 parity

**8-seed**（k=3，n=106，每题 24 votes）：
- evo_seed 0.5688 vs skillopt 0.5523 = **+1.65pp**；paired t two-sided **p=0.1065**，one-sided **p=0.0533**（临界）；赢 5/8 seed。
- ⚠️ 曾据此判断"趋势级胜出"——**后被 16-seed 推翻**。

**16-seed**（k=3，n=106，48 votes/q，seeds 1-8 cached + 9-16 fresh）：
```
skillopt: 0.5531   evo_seed: 0.5590   diff = +0.59pp（从 8-seed +1.65pp 收窄）
paired t=0.812, two-sided p=0.4189, one-sided p=0.2094  <- 不显著
bootstrap CI [-0.0083, +0.0204]（宽含 0）；seed wins 9/16；win/loss 47/41
```
- **seeds1-8: +1.65pp；seeds9-16: −0.47pp -> 符号翻转，效应不稳健**。

**split-half 诊断（决定性）**：batch1(seeds1-8) vs batch2(seeds9-16) 逐题 diff：Pearson r=0.146（p=0.135 ns）；逐题符号一致率 43%（< 50% 随机基线）。**无稳定逐题信号** -> 加 seed 无法放大 -> evo_seed 与 SkillOpt 统计上不可区分（parity）。

**为何先前投影（16-seed p~0.013）失败**：方差分解显示 92.6% 方差是 k=3 投票测量噪声、仅 7.4% 真实题间异质性，投影把 8-seed 均值（+1.65pp）当真效应固定只缩 SE。实际加 seed 时均值回归（8-seed +1.65pp 是向上波动，SE~1.0pp；16-seed +0.59pp 落其 1 SE 内）。**教训**：测量噪声占 92.6% 意味着 8-seed 点估计高度不确定，不可作真效应外推。

### 修正结论（推翻先前"趋势级胜出"乐观判断）

- **evo_seed 未显著超过 SkillOpt**：16-seed +0.59pp，p=0.42，符号不稳健，无稳定逐题信号。8-seed +1.65pp（p=0.053）是向上波动，不可作"胜出"证据。真效应若存在 ≤ ~+0.6pp，实际不可确认。
- **论文定位修正**："training-free 失败驱动最小 patching **匹配**强基线 SkillOpt（parity，+0.59pp ns）" + 机制（splice −4.7pp 稀释 vs 最小 patch 不稀释）+ 弱基线 v2 +16.1pp（t=9.02 高度显著）仍为主结果。**非"超过 SkillOpt"**。

### 产出
- evo_seed skill：`rl/train/verl1/outputs/skill_evolver_math/run_20260803_184657/skill_iter2.md`（11133 chars, hash 72a95d13）
- merge skill（负对照）：`rl/train/verl1/outputs/skill_evolver_math/merge_skillopt_evolver_manual.md`
- eval cache：`multiseed_test/per_evo_seed_72a95d13_seed{1-16}.json`

## A.17 隔夜三任务流水线（2026-08-04 ~ 08-05）

### Task 1：evolver 机制消融（完成，故事干净）✅

详见正文 2.3。leave-one-out 关闭一个机制重跑 evolver（iters=2, K=4, train_n=5），再 4-seed 评测。fresh 8-04 API 全部同日重算，避免跨期漂移。

机制贡献（Δ = FULL − ablation，正=机制有用）：

| 机制 | Δ | t | p | 显著性 |
|------|------|------|------|--------|
| M1 GEPA种群+选择 | **+0.1046** | 5.21 | 1.9e-7 | *** |
| M2 失败驱动patch | **+0.0778** | 3.91 | 9.2e-5 | *** |
| M3 慢更新consolidate | +0.0008 | 0.053 | 0.958 | ns |
| M4 防退化+多样性 | +0.0189 | 1.11 | 0.267 | ns |

结果文件：`rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test_ablation/ablation_matrix.json`

### Task 2：AIME 从头全链（完成，负结果 = 泛化边界）❌

完整方法 S0 -> v2_aime (iters=2, g=4) -> evolver_aime (iters=4, K=4) -> eval AIME25，proper train/val/test split（AIME24 train18/val12 seed42 + AIME25 held-out test）。

**结果：优化器卡在 S0，无 skill 改动**
- **S0 本身有效**：select pass=0.678, delta=+0.111 vs no-skill（CI +0.033 显著）。
- **v2_aime - KEEP PREV**（两 iter 均拒）：iter1 composed skill val Δ=+0.111 但 CI_lower=−0.028；iter2 val Δ=−0.028。best_val_delta=+0.000, v2_best == S0。
- **evolver_aime - KEEP PREV**（4 iter 全拒）：每 iter 生成 4 novel 候选（生成器正常），但所有候选 CI_lower<0；iter4 winner pass 0.611 << base 0.678 触发反退化 REJECT。skill 2238->2238 chars hash 6f0e7669 未改动。
- **根因（非 bug，是 CI 门控正确工作）**：per-task/select 上有正 delta，但组成的 global skill 不泛化到 held-out val（CI_lower 一律<0）。与 07-29 evolver bug 不同——那次门控失效让未改动 skill 通过；这次门控工作正常，只是没找到可泛化改进。

**test eval 最终结果**（AIME25, 4-seed, k=3）：

| skill | mean | per-seed |
|-------|------|----------|
| s0 | 0.4972 | 0.489, 0.467, 0.511, 0.522 |
| v2 | 0.5000 | 0.500, 0.511, 0.500, 0.489 |
| evolver | 0.5333 | 0.500, 0.567, 0.567, 0.500 |

三个 skill 字节相同（同 hash 6f0e7669），差异纯为 k=3 投票 temp=0.3 采样噪声。**确认负结果**：同一 skill 的噪声带 ~±3pp，val CI 门控正确拒绝了过拟合 patch。

**对论文影响**：主结果不变（livemath +15.9pp）。AIME 负结果与 AIME24 负结果共同构成"跨形态泛化边界"证据。与 Task 1 消融呼应：M1(GEPA)+M2(failure-patch) 在 livemath 显著有效，但在 AIME 上 failure-patch 组成的 global 不泛化——M2 的 patch 是形态特异的。

### Task 3：跨 agent（single-call vs ReAct）——冒烟通过，全量待跑 ⏳

**实验问题**：在已优化的 evolver skill（hash 12633201, 主结果 test=0.4623）之上，把 B 模型从「单次生成」升级为「ReAct 循环 + python_executor 自调试」是否进一步提升 LiveMath test=106 通过率？

**公平性设计（关键）**：
- 同一 skill（s0 68571cb2 / evolver 12633201），同一模型（DeepSeek-V3.2），同一 28RPM 限速。
- **baseline = single-call**：用 8-04 fresh `multiseed_test_ablation/`（mean 0.4497，同 hash 12633201），避免 7-30 cached seed1 drift +5.7pp 虚高偏置 delta。
- **treatment = ReAct**：system = skill + `REACT_SUFFIX`（追加工具说明）；多轮循环，`<tool_call>` 调 execute_code，无工具回复=最终答案。eval_k=3, 4-seed。
- **唯一差异 = REACT_SUFFIX + 工具循环**。skill/模型/限速/评测完全一致。

**关键技术决策**：
1. 后端用 skillopt `chat_target_messages`（DeepSeek-V3.2, 28RPM token bucket），不用 BAgent._call_llm（无限速会打爆 API）。
2. execute_code 复制自 `agent/b_agent/core.py`（沙箱 subprocess, 10s 超时），复制而非导入避免触发 BAgent 的 SOUL/402-skill 机制，保证 system prompt 干净 = evolver skill 独立。
3. 不重用 BAgent.run()：它注入 402 skills + SOUL 会混淆变量。
4. max_steps=6，用尽后强制 "FINAL ANSWER" nudge 保证有答案。

**当前状态（2026-08-05 02:50）**：
- ✅ 冒烟通过：`SMOKE_RESULT: OK pass_rate=1.0`（1题 max_steps=2, n_tools=0=模型直接答未用工具，难题会显真实工具使用率）。
- ✅ 路径 bug 修复：`run_overnight_pipeline.sh:52` 硬编码 `livemathmaticianbench`（漏 e）-> 已改对。
- ✅ baseline drift 修正：默认 single_dir（7-30 cached 虚高）-> 改用 8-04 fresh。
- ⏳ **全量 eval 待启动**：启动命令 `bash rl/train/verl1/run_cross_agent_react.sh`（≈2-4.5h）。产物 `outputs/tf_skill_math_v2/multiseed_test_react/react_eval.json`（含 react+single 对比 + 配对 t）。

详见 `memory/cross-agent-build-20260804.md`、`memory/aime-from-scratch-result-20260804.md`、`memory/evolver-ablation-result-20260804.md`。

---

# 附录 B：关键文件索引

- 设计/方向：`docs/training_free_grpo_fusion_design.md`、`agent/b_agent/DIRECTION.md`、`ARCHITECTURE.md`
- 创新方法：`memory/our-grpo-skill-method-design-20260721.md`、`docs/training_vs_training_free_parallel_plan.md`
- 方法实现：`rl/train/verl1/tf_global_v2_math.py`（v2 在线闭环）、`rl/train/verl1/skill_evolver_math.py`（evolver 自进化）、`experience/extractor_math.py` + `templates_math.py`（数学域 trace2skill）
- 评测：`rl/train/verl1/eval_test_multiseed.py`（多 seed test）、`rl/train/verl1/eval_aime24_multiseed.py`、`rl/train/verl1/eval_bagent_react_multiseed.py`（跨 agent ReAct）
- 主结果 skill：`rl/train/verl1/outputs/skill_evolver_math/run_20260730_111306/best_skill_evolved.md`（evolver 2-iter, hash 12633201, 4672 chars）
- 消融矩阵：`rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test_ablation/ablation_matrix.json`
- 问题日志：`docs/phase2_issues.md`（全部 bug/根因/修复按日期）
- 训练指南：`docs/lcb_8card_training_guide.md`
- 综述：`agent/b_agent/skill_optimization_survey_2026.md`、`reference.md`
- 索引：`MEMORY.md`
