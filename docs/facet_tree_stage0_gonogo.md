# Facet-Tree Stage 0 go/no-go 协议 (2026-08-21)

> **状态**: 定稿前最后一次收敛。前接 [[ea-skill-evolution-survey-20260819.md]]（文献诊断）+ [[facet_tree_skill_evolution_design.md]]（设计 v2）+ [[facet-tree-skill-evolution-20260819]]（记忆）。
> **本文解决的根本问题**: 设计文档说"能落地，约 10 天"，但没回答"开工前必须先钉死的两个约束"。本文把实测成本数据压进 go/no-go 协议，并把**唯一会让所有因果归因失效的变量——skill 表示——提到 Stage 0 的前置位**。
> **三层分工（已锁定 2026-08-21）**: **进化对象 = 真 S₀ `initial.md` 的文本**（4 具名段，`facet_decompose` 切的就是这份 `.md` 的段落）；**进化引擎 = `skill_evolver_math.py`**（变异/交叉/选择/树遍历的代码，Stage 0 要加 `facet_decompose`+`facet_compose`+`archive_tree`）；**评估量表 = `tf_global_v2_math.py`**（定义 `eval_on`/`bootstrap_ci_lower`，被引擎 import，是 fitness 来源，不是被进化对象）。不是 BAgent 402 条代码 skill。

---

## 零、一个必须钉死的判断（本文立论）

GA/EA **能做**，绑定约束是**工程量（~10 天编码）而非 API 算力**——但前提是 scope 控制在"固定轴 + 2 代 + probe-split 保留"的最小子集。这个判断已经把两个变量排除在外：

- **E 组（涌现 facet 轴）最贵，Stage 0 不碰**。v2 核心增量（自动涌现 facet 轴）是 §九·5 自认风险最高、最贵（失败模式聚合 + 分支自适应开合 + K=3 门控）的部分。先验证 C 组（固定轴）的"树+交叉"基础假设，若 test > 0.4654 再看要不要投资涌现（消融 E）。
- **BAgent 402 条代码 skill 这条线是另一条演化路径，Stage 0 不碰**。它 skill 形态不同（YAML frontmatter + 自由 markdown，非 §2.2 的 seed facet），无现成 eval_on/probe 对位。对象已锁定 math。

---

## 一、实测成本数据（2026-08-21 从实际产出物收集，替换设计 §七的"~10 天"空估）

| 项 | 值 | 来源 |
|---|---|---|
| 每迭代 patch 评估 | 4 候选 × probe(29 题 × k=3 = 87 调用) = 348 调用 | 实测 |
| 28 RPM 下纯 API 时间 | ~12 min | 实测 |
| 观察实际开销 | ~2.5×（含 C 模型判断） | 实测 |
| 完整 3-skill × 4-seed 终评 | 164 min / 3816 调用 | 实测 |
| 1 skill × 4-seed 终评 | 1272 调用 / 28 RPM ≈ 45 min | 实测 |
| API 上限 | 28 RPM（≤30 硬限制），token-bucket + 429 指数退避 | 单元测试已覆盖 |
| 评估基础设施 | 完全可复用（eval_on / bootstrap_ci_lower / gate_consolidate / C 模型 A→B→C），零新评估器 | 设计 §七，已实读核对 |
| 新增代码 | 5 函数（facet_decompose / facet_compose / archive_tree / preserve_extend_check / 树遍历主循环） | 设计 §七 |

---

## 二、三处算错/说轻的账（go/no-go 判定的修正）

### 1. API 时间 ≠ wall-clock（最重要的一条修正）

设计文档和初稿都用"65 min/代"当排程数，但这是 API 时间，不是 wall-clock：

- 当前 evolver 用 `ThreadPoolExecutor(max_workers=40)` 并发（`skill_evolver_math.py:216`）。**40 并发对 28 RPM token-bucket，并发只会被限速器削回 28 RPM**，稳态 ≈ 28 调用/分钟，不是 40 真并发。
- 初稿的"聚合稳态 1.2× 开销"来自**一次性吞吐观测**；facet 树是**持续 1400 调用/代**，这个系数在多代持续压力下会退化到 ~1.0。
- **每次额外 eval_on 调用（分支评估 / baseline / gate re-eval / s0 补算）都抢同一个 28 RPM 池**——和 B 模型代码生成、C 模型评判、其它并发实验互相拖慢。见 [[api-concurrency-deepseek-20260716]]。

结论：65 min/代仍可作量级参考，但**别当"可精确排程"的数**，它是被共享限速池左右的下限。夜间无人争用时才逼近。

### 2. "零新评估器"是**写**评估代码的钱省了，**跑**评估的 API 钱没省

设计 §七"zero new evaluator"准确（fitness 机器复用 eval_on），但无可避免地会**让评估调用量线性翻倍**：单 lineage 每代评 K=4 候选，facet 树每代评 B×C=12 + 交叉 4 = 16 个 skill。"复用评估器"省的是*写评估代码*的钱，省不了*跑评估*的 API 钱——两笔账。

### 3. 真正的 no-go 风险不在算力，在三个点

1. **机制能否破 0.4654 天花板——未知**。这是研究风险，不是工程风险。probe-split 已证"同源选择过拟合"不是绑定瓶颈（[[eval-signal-debiasing-diagnosis-20260820]]），Facet-Tree 赌的是"单 lineage 搜索结构不泛化"这个诊断对。赌错就是又一个 Phase C REFUTED（0.3703）。
2. ~~两个未定设计点挡在实现前~~ → **已定（2026-08-21）**：skill 表示=slotted(§三·1)、probe=29 seed=42 锁死(§三·2)、外加起点=真 S₀ 4 段(§九-7c)、C-judge=pro 独立账户(§九-8)——共 4 项全定，无设计阻塞。
3. **slotted 迁移是被低估的前置成本**。见 §三·1。

---

## 三、两个设计点——已定（2026-08-21），Stage 0 前置项全清

### 1. skill 表示 = 方案 a（slotted template）—— Line A 上已是现状，POC 降级为格式核对

**为什么必须先定**：`facet_compose` 交叉按 slot 取段、`preserve_extend_check` 按 slot 比侧面集，都依赖表示方案。**若用方案 b（LLM 抽取）跑出"交叉无增益"，无法归因**——分不清是"交叉本身无效"还是"每次 LLM 抽取的噪声淹没了交叉信号"（设计 §九·2 已承认 b 有噪声）。

**定的结果**：方案 a（slotted template）。**Line A 实读发现 S₀ 根 skill 已是 slotted**——`# Live Mathematical MCQ Heuristics` 下真 S₀(`initial.md`)含 `## Option Comparison` / `## Theorem-Level Precision` / `## Hypotheses` / `## Final Answer` **4 具名段**（1018 chars）。故 `facet_decompose` = 按 `^## ` 切段，**零迁移成本**，原 §三·1 的"402 条全迁 ~1 天 / POC 转写"在 Line A 上**不适用**（那是 Line B BAgent skill 的事，Stage 0 不碰）。

⚠️**起点段数订正（2026-08-21 实读）**：`## Common Failure Modes` **不是** S₀ 固有段——它是**进化中后期涌现的第 5 段**。两份 `.md` 长度对照（别混）：**真 S₀ `initial.md` 4 段 = 1018 chars**（Stage 0 起点，见上）；**晚期 `run_20260728_131934` 的 V2 默认 5 段版 = 3655 chars**（才补 `Failure Modes`，evolved 181981e9 保留五段仅扩 `Failure Modes`）。多数 run 的 `.md` 是 4 段。v3 文档原写「五段 3655 chars」**不精确**，已订正。

**起点钉死 = 真 S₀（4 段 `initial.md`），非 V2 默认 5 段版**（设计 §九-7c，已定）。三层理由：① **因果可归因**——从 5 段版起步 = 预装 `failure_modes` facet，该分支后续长得好时分不清「涌现发现」还是「起点自带」，正是 §九-7b/E 组要避免的归因病；② **涌现 facet 内置验证**——从 4 段起步，`Common Failure Modes` 必须靠 §2.2(b) 涌现机制从零长出，跑出来即「涌现 facet」novelty 不辩自明；③ **与 SkillOpt 0.5464 可比（决定性）**——SkillOpt-best 是 `S0 起(SkillOpt自有)` 标公平，我们主链全从 S0 起亦公平，只有种 SkillOpt-best 的 Run F 标 `❌ 嵌套已排除`；若从 5 段 V2 版起步 = 继承 v2 预装 `failure_modes` 而 SkillOpt 从同 4 段 S₀ 自力长出 → 起点不对齐、重蹈 Run F 嵌套病。落地：显式传 `--init_skill=<真S0 initial.md>`，不用默认 `V2_SKILL_DEFAULT`。

**POC 降级**：从"转写 + 信息损失率"降为 ~5 min 格式核对——确认 4 段切分稳定（无跨段内容、`## Final Answer` 单行可冻结不参与交叉）。

**Stage 0 facet 轴 = math skill 自身 4 冷启动段 + `failure_modes` 作涌现检验目标**（非 v1 的 `tool_guidance`/`process`/`self_debug`——那是 BAgent 代码 skill 概念，与 math skill 段不对位，用之 `facet_decompose` 会直接失败）。冷启动 seed = option_comparison / theorem_precision / hypotheses / final_answer（`final_answer` 冻结不参与交叉），`failure_modes` 起点为空靠涌现长出。详见设计 v3 §2.2(a) seed facet 表 + §九-7c。

### 2. 泛化 probe 集配比 = 沿用 probe-split 现有 29 题 seed=42，锁死不再动

**为什么**：Facet-Tree 唯一要证明的就是"破了 0.4654 天花板"，秤不换才能比出这个结论。probe-split 的 29 题 seed=42 划分是已有可比基线（`skill_evolver_math.py` 行 741–749），沿用可和 0.462 主结果直接对齐。

**结论**：锁死。"锁死"是这个设计点唯一要做的决策——不要在新探针集上花时间，那会毁掉可比性。

---

## 四、Stage 0 最小可跑版（go/no-go 判据）

```
对象: 全局 math skill 文本 (Line A, 真 S₀ 4 段 initial.md)
facet 轴: math skill 自身 4 冷启动段 (option_comparison / theorem_precision / hypotheses / final_answer[冻结]) + failure_modes 作空 slot 占位(Stage 0 既不预装也不涌现), 关涌现, 锁死分支数
代数: 2 代
probe-split: 保留（诊断 diag / 选择 probe 已知, 29 题 seed=42）
护栏: 只挂最便宜的 facet-edit 约束（分支只许动对应 slot，物理防塌缩）
前置: C-judge 走 deepseek-v4-pro 独立账户 (§九-8, 已定): 物理隔离 B 的 V3.2 池, 保留 LLM 评判
起点参数: --init_skill=<真S0 initial.md 路径>, 不用默认 V2_SKILL_DEFAULT
```

> **涌现验证分阶段（问题1定稿，选项 C）**：Stage 0 判据**只一条** = test 4-seed > 0.4654，干净地只答「树+交叉+facet-edit 护栏破不破天花板」。`failure_modes` 在 Stage 0 是**空 slot 占位**——既不预装也不涌现。理由：§2.2(b) 涌现需 K=3 连续代门控才开分支，而 Stage 0 只有 2 代，机制**物理上不触发**；把「failure_modes 是否自己长出」写进 Stage 0 判据 = 一个永远为空的检验项，反而不诚实。**「4 段起点让 failure_modes 靠涌现从零长出」的内证价值在终局 C 组兑现**（多代、开涌现、K=3 门控能触发），不是 Stage 0 的事。故 §四表里 `failure_modes` 标注从「作涌现检验目标」订正为「空 slot 占位(Stage 0 不涌现)」。

| 项 | Stage 0 值 |
|---|---|
| 编码 | facet-edit 约束 + archive 树 + 2-parent 交叉最小可跑版，跳过涌现机制。Line A 已 slotted 故**免迁移/免转写 PoC**（但需传 `--init_skill=真S0` + C 走 pro），~2–3 天 |
| 算力 | 2 代 × ~65min(下限) ≈ 2.2h 演化 + 45min test ≈ 3h 纯 API 下限 (wall-clock 只会更长, 夜间逼近) |
| 判据 | **test 4-seed 是否 > 0.4654**。是 → 涌现(终局 C 组, 多代开 K=3 门控让 failure_modes 从零长出) + 完整消融值得投；否 → 单 lineage→树没破天花板，Facet-Tree 方向证伪，省下 25h + 6 天 |

这个分法对应设计 §八的核心判据，但先只跑 C（固定轴），把「涌现验证 + E 组涌现消融」留到 Stage 0 通过后的终局。

**Stage 0 被否决会怎样**：若固定轴的"树+交叉"都破不了 0.4654，则"单 lineage → 树"的换结构假设清盘，Facet-Tree 方向证伪。这是一个干净的、可证伪的 go/no-go，而非开放式探索。

---

## 五、完整消融量级（Stage 0 通过后再考虑）

完整消融（设计 §八 A/B/C/D/E 五组，baseline 已有）：5 ×（演化 ~3.5h + test ~1.5h 含 s0 复评）≈ 25h 纯 API 时间，摊到 3–4 天夜间跑可行。

注意：演化期不必用 4-seed/k=3。CI 门可用 k=3 但只在 select(29 题) 上跑，终选才上 4-seed test。当前 evolver 已是这个模式（演化期 select 上 k=3，test 期才 4-seed），所以 §二·1 的 3.3h 已经是削过的，不是全量。

---

## 六、与既有诊断链的衔接（不要孤立看这个方向）

Facet-Tree 不是新起炉灶，是站在三处既有诊断上的后继：

- **Phase C REFUTED（0.3703）** → 单 lineage 结构自进化的同类塌缩机制，Facet-Tree 用 facet-edit 约束 + archive 树 + preserve-and-extend 契约直接防它。见 [[facet-tree-skill-evolution-20260819]] §五护栏。
- **probe-split 诊断（0.462 = val 内容过拟合，非失败类型偏差假天花板）** → probe 集 29 题 seed=42 是 Stage 0 的选择信号源。见 [[eval-signal-debiasing-diagnosis-20260820]]。
- **Ctx2Skill Cross-Time Replay** → 泛化性作为全局选择标准挂所有分支（loop 外静态 probe 回放，跳出"自己评自己"闭环）。见 [[coevol-ctx2skill-anti-collapse-20260819]]。

**三者方向收敛**：诊断-选择解耦（probe-split）、固定 probe 回放校准（Ctx2Skill）、泛化性挂全局（Facet-Tree §六）——都是同一枚硬币的三面，即"跳出单 lineage 的 val 自评闭环"。

---

## 七、开工清单（当用户确认后执行）

1. **格式核对**（~5 min）→ (a) 确认**真 S₀（4 段 `initial.md`）**按 `^## ` 切分稳定（`## Final Answer` 冻结不参与交叉）。Line A 已 slotted, 无需 402 条转写 PoC。(b) ⚠️顺带验 **`theorem_type` 字段**:实读 `livemathematicianbench_1-3-6_seed42/test/items.json` 确认字段名=`theorem_type`、值是**字符串 list(一题多类)**、值域 13 类(Existence 35/Implication 34/`Inequality or Bound` 31/Universal 30/…,详见设计 §九-7)。这是 `failure_modes` 涌现分簇的信号源,字段名/结构错→分簇静默失效,~1 min。
2. **C-judge 走 pro 独立账户**（设计 §九-8, 已定）→ `judge/qwen_235B.py` 的 `API_URL`/`AUTH_TOKEN`/`MODEL_C` 改指向 `deepseek-v4-pro`（pro 专用 key, `/data/zhuldz/model-launch/start-pro.sh` 配置）, 物理隔离 B 的 V3.2 池, 保留 LLM 评判维度。备选 ①套_TokenBucket/②退纯客观分 仅在 pro 额度不足时启用。⚠️**Stage 0 非阻塞订正(2026-08-21, 问题4)**: Stage 0 fitness 走 `eval_on`→纯 EM pass_rate, **不调 `qwen_235B`/`c_interface`**(`skill_evolver_math.py` 实读确认), 故改三参数对 Stage 0 判据 test>0.4654 是空操作; C-judge=pro **仅在 Stage 0 产出 skill 后回灌 GRPO 训练时生效** → 移出 Stage 0 关键路径, 标后置。详见 [facet_tree_stage0_plan.md](facet_tree_stage0_plan.md) Step 2。
3. **写 5 函数最小版**：facet_decompose（按 `^## ` slot 分解）/ facet_compose（2-parent facet 组合交叉）/ archive_tree（节点存 skill + facet_label + lineage + delta）/ preserve_extend_check（S(c)⊇S(p) 契约）/ 主循环改树遍历。
4. **挂 facet-edit 约束护栏**（最便宜，物理防塌缩）。
5. **跑 Stage 0**（2 代 + math skill 4 冷启动段固定轴 + `failure_modes` 空 slot 占位[关涌现] + probe-split），**显式双传 `--init_skill=真S0 initial.md`（进化起点, L633）+ `--s0=真S0 initial.md`（delta 基线对象, L634；不可只传 `--init_skill`，否则 delta 基线隐式退到 `S0_DEFAULT` 碰巧一致）**，判据 test 4-seed > 0.4654。
6. 结果写入 [[facet-tree-skill-evolution-20260819]] + MEMORY.md；no-go 也写（证伪也是结论，见硬性规则）。

---

**下一步待用户确认**：4 项设计决策(①slotted ②probe-29seed42 ③起点真S₀4段 ④C-judge→pro)+ 问题1分阶段(选项C, Stage0只一条判据 test>0.4654, 涌现验证挪终局C组)已全定,无设计阻塞。确认开工后进入 plan mode,把 Stage 0(§七 1-5 步)拆成具体实现步骤。