# LLM Agent 技能优化：2025-2026 新进展综述

> 检索日期：2026-07-13 | 检索源：arXiv API（原 agent 报告全部核验；2026-07-14 独立复核状态见文末"独立复核状态"）
> 当前月份：2026 年 7 月

## 与 reference.md 的关系

本文档**补充而非重复** `reference.md`。`reference.md`（更新于 6 月 21 日）以 12 个具体方法为主线，做横向对比。本文档在以下方面增量：

- **新增方法**：聚焦 reference.md 截止后（2026 年 2 月–7 月）出现的工作，共 20+ 篇，按主题归类，不再重复已有 12 法。
- **综述类文献**：补充 reference.md 缺失的 survey 级文献，给出分类法与关键贡献。
- **趋势综合**：跨文献提炼 2025-2026 新兴趋势。
- **对 5 项改造的启示**：将新想法显式映射到我们当前的 5 项 redesign。

reference.md 已覆盖的 12 法（不再列出）：SkillOpt、Trace2Skill、TextGrad、GEPA、EvoSkill、MUSE-Autoskill、OpenSkill、SGDR、SkillHone、SkillLens(2605.23899)、ReSkill、Bayesian-Agent。

---

## 一、新增方法/论文（post reference.md，2025-2026）

### 1.1 技能验证与门控（Validation & Gating）

这一簇工作直接回应我们 reform #2（EMA 成功率打分）和 #3（ADD/UPDATE/DELETE/NONE 循环）的核心风险：**没有可靠验证，技能演化会静默失败**。

#### 1. HDSO — Hypothesis-Driven Skill Optimization（2026-06）
- **机构**：Fangxin Shang, Yehui Yang
- **arXiv**：[2606.22330](https://arxiv.org/abs/2606.22330)
- **核心思想**：把技能更新建模为"可证伪假设"。curator 先提出带显式验证计划的假设，实例化为候选 skill 包，通过**配对 control/treatment 执行**验证，仅把被支持的候选并入已批准仓库。curator 与 executor 均为冻结推理端点（train-free）。
- **与 12 法的差异**：Bayesian-Agent 用后验做版本选择，SkillOpt 用验证集门控；HDSO 首次引入"假设-验证"科学范式 + 配对对照实验，且**在 20% 随机翻转的成功/失败反馈下仍保持 +7.1 分增益**——这正是我们 C 模型评判有噪声时的痛点。
- **对我们的启示**：reform #2 的 EMA 成功率是观测性的，HDSO 的 control/treatment 配对可作为更鲁棒的替代或交叉验证手段；reform #4 的 debug-trace 蒸馏可借用其"显式验证计划"模板。

#### 2. The Blind Curator — 偏置评判如何静默瘫痪技能退役（2026-07）
- **机构**：Xing Zhang, Yanwei Cui, Guanghui Wang 等
- **arXiv**：[2607.07436](https://arxiv.org/abs/2607.07436)
- **核心思想**：自进化 agent 靠"看坏技能失败"来退役它。但当 judge 看不到失败时会发生什么？论文证明：**false-pass 偏置（失败被误判为通过）会在一个尖锐阈值上彻底瘫痪 contribution-based 退役**，且**没有任何数据量能跨过该阈值**。对称噪声不影响退役，但 false-pass 会。更危险的是：下游整体指标可能**保持平稳**，使瘫痪"静默"——不暴露在任何聚合指标里。
- **与 12 法的差异**：12 法多假设奖励无偏；本文首次给出"偏置 judge → 退役机制失效"的因果安全分析，并提供廉价的 defect-injection 审计方法。
- **对我们的启示**：**直接命中我们的已知 issue #4（C 模型区分度不足）**。我们的 reform #2（EMA 成功率）和 #3 的 DELETE 操作都依赖 C 模型奖励；若 C 模型有 false-pass 偏置，DELETE 会静默失效。部署前应先跑 defect-injection 审计判断 C 模型在阈值哪一侧。

#### 3. GRASP — Gated Regression-Aware Skill Proposer（2026-05）
- **机构**：Johannes Moll, Jean-Philippe Corbeil, Daniel Rueckert 等（TU Munich 等）
- **arXiv**：[2605.29668](https://arxiv.org/abs/2605.29668)
- **核心思想**：把 agent 改进视为对有界 skill 库的一连串编辑，**仅当候选在平衡 held-out probe 上产生净改进且不突破硬回归预算时才接纳**。关键发现：未经验证的 skill 写入**不比无 skill 更好**。
- **与 12 法的差异**：SkillOpt 有验证门控但无"硬回归预算"概念；GRASP 显式约束"新 skill 不能让旧正确行为回归"，且证明冻结库可跨模型迁移（强模型写的 skill 能提升弱 executor）。
- **对我们的启示**：reform #3 的 UPDATE/DELETE 应加"硬回归预算"——新 skill 不仅要比 baseline 好，还不能让任何已通过题目回归。这把 NONE 操作从"被动不更新"升级为"主动防回归"。

#### 4. RSEA — Recursive Self-Evolving Agents via Held-Out Selection（2026-06）
- **机构**：Michael Nguyen, Quoc Nguyen, Paul Vuong
- **arXiv**：[2606.28374](https://arxiv.org/abs/2606.28374)
- **核心思想**：三层自然语言状态（imperative strategy / reusable skills / procedural playbook），跨代重写全部三层，**仅当候选在不相交 held-out split 上不回归时才提交**（strict keep-better gate）。关键发现：无门控的在线 context 演化是"高方差且不安全"的（Dynamic Cheatsheet 在 ALFWorld 近最佳，但在 WebShop 崩溃到 0.14）。
- **与 12 法的差异**：相比 EvoSkill/OpenSkill 的变异-筛选，RSEA 强调"monotone-safe"——**永不显著劣于 base agent**，最差回退到 vanilla ReAct。
- **对我们的启示**：reform #3 的 NONE 分支应有"最差情况回退到无 skill 基线"的保证；reform #5 的 injection-mode toggle 可借鉴其三层状态分离。

### 1.2 技能图与组合（Skill Graph & Composition）

#### 5. GraSP — Graph-Structured Skill Compositions（2026-04）
- **机构**：Tianle Xia, Lingxiang Hu, Ming Xu 等（Zhejiang Univ. 等）
- **arXiv**：[2604.17870](https://arxiv.org/abs/2604.17870)
- **核心思想**：首个可执行 skill graph 架构。在检索与执行间引入编译层，把扁平 skill 集转为带 precondition-effect 边的类型化 DAG，节点级验证 + 5 种类型化修复算子，把重规划从 O(N) 降到 O(d^h)。**关键反直觉发现：2-3 个聚焦 skill 优于完整文档；过多 skill 反而有害**。
- **与 12 法的差异**：SGDR 做动态检索，SkillLens 做 utility 评估；GraSP 首次把 skill 组织成可执行 DAG 并量化"少即是多"。
- **对我们的启示**：**直接验证 reform #1（Meta-Skill "selective ignore"）**。我们的 402 条 skill 全注入可能正是性能瓶颈；GraSP 的"2-3 skill 优于全量"是强证据，支持我们向"紧凑注入 + 选择性忽略"迁移。

#### 6. SkillComposer — Generative Skill Composition（2026-06）
- **机构**：Xinyu Zhao, Zhen Tan, Kwonjoon Lee 等
- **arXiv**：[2606.32025](https://arxiv.org/abs/2606.32025)
- **核心思想**：把 skill 组合形式化为联合决策——选哪些子集、选几个、什么顺序，三者不可解耦。用受约束自回归解码器在 skill 标识符上单次解码，子集/数量/顺序联合涌现。在 GPT-5.2-Codex 上 pass@1 比 no-skill baseline 高 +23.1pp，接近 gold-skill 上界且 prompt-token 成本更低。
- **与 12 法的差异**：现有检索多为 embedding top-k；SkillComposer 用"结构化序列预测"替代检索，且**面向代码生成 agent**（与我们任务一致）。
- **对我们的启示**：reform #1 的"紧凑注入"可从"top-k 检索"升级为"联合子集-顺序预测"；reform #5 的 toggle 可增加"组合预测"模式。

> ⚠️ **同名论文澄清 (2026-07-21 核实)**：存在**两篇同名 SkillComposer 论文**，勿混淆：
> - **本条 2606.32025** (UNC Zhao/Tan/Lee, "Generative Skill Composition for LLM Agents") — 评估 SkillsBench, **无 MBPP/LCB 实验**。+23.1pp 是 SkillsBench agent pass rate, 非 MBPP 代码 pass@1。此前曾误引为"MBPP 代码先例", 已纠正。
> - **另一篇 2606.06079** (浙大+通义+NUS Qi Zhang 等, "SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization") — 评估 **τ²-Bench + LiveCodeBench v6 + App-World**, **在 LCB v6 上 skill 有效** (No Skill 56.6→online 59.1 +2.5, Hard hybrid 67.2)。**这篇才是与我们同数据集的代码 skill 正先例**。详见 [[skillcomposer-sepo-verified-20260721]]。PDF: lunwen/docs/survey_agent_skill_evaluation_evolution/paper/2606.06079v1.pdf

#### 7. SkillDAG — Self-Evolving Typed Skill Graphs（2026-06）
- **机构**：Tong Bai, Zhenglin Wan, Yang You 等（NUS 等）
- **arXiv**：[2606.03056](https://arxiv.org/abs/2606.03056)
- **核心思想**：把 skill 间关系（依赖/冲突/特化/重复）建模为类型化有向图，作为**agent 可调用的结构化检索接口**在推理时查询与演化。propose-then-commit 协议让 agent 注册执行支撑的边。优势随 pool 增长 10x 仍稳健（固定 pipeline 退化）。
- **与 12 法的差异**：相比 SkillHone 的决策历史，SkillDAG 暴露**显式冲突信号**给 agent，且图在执行中积累。
- **对我们的启示**：reform #1 中 402 skill 的"重复触发词"问题（known bug #2）可被 SkillDAG 的冲突检测替代——不是修 YAML，而是建类型化图暴露重复。

#### 8. SkillGraph — Skill-Augmented RL via Evolving Skill Graphs（2026-05）
- **机构**：Xiaoyuan Li, Keqin Bao, Yubo Ma 等
- **arXiv**：[2605.12039](https://arxiv.org/abs/2605.12039)
- **核心思想**：skill 作为有向图节点，typed edge 编码 prerequisite/enhancement/co-occurrence。检索返回有序 skill 子图而非孤立条目；图从轨迹与 RL 反馈持续更新，skill 库与策略共同进化。
- **与 12 法的差异**：ReSkill 把 skill 创建嵌入 GRPO；SkillGraph 把 skill **关系结构**嵌入 RL，解决"何时合并/拆分/删除"的结构线索问题。
- **对我们的启示**：reform #3 的 MERGE/SPLIT 操作可借助 prerequisite/co-occurrence 边做结构化判断。

### 1.3 技能库策展与生命周期（Curation & Lifecycle）

#### 9. SkillBrew — Multi-Objective Curation of Skill Banks（2026-05）
- **机构**：Wentao Hu, Zhendong Chu, Junda Wu, Ming Jin 等
- **arXiv**：[2605.29440](https://arxiv.org/abs/2605.29440)
- **核心思想**：把 skill bank 策展形式化为**约束多目标问题**——bank 必须 useful（对 agent 有用）、diverse（内容多样）、coverage（覆盖查询分布）。用 Pareto-aware 优化 + bi-level propose-then-verify 循环求解。明确反对 append-only 无限增长。
- **与 12 法的差异**：MUSE-Autoskill 管全生命周期但无显式多目标；SkillBrew 首次把"留什么/删什么"变成多目标 Pareto 问题。
- **对我们的启示**：**reform #3 的 ADD/UPDATE/DELETE/NONE 可直接借鉴其三目标框架**——DELETE 不只看成功率，还要看 diversity 和 coverage，避免删掉唯一覆盖某类题的 skill。

#### 10. SLIM — Dynamic Skill Lifecycle Management for Agentic RL（2026-05）
- **机构**：Junhao Shen, Teng Zhang, Hong Cheng（HKU）
- **arXiv**：[2605.10923](https://arxiv.org/abs/2605.10923)
- **核心思想**：把活跃 skill 集视为与策略共同更新的动态优化变量。用 **leave-one-skill-out 验证**估计每个 skill 的边际外部贡献，执行 retain/retire/expand 三操作。关键发现：策略学习与外部 skill 保留不互斥——部分 skill 被吸收进策略，部分持续提供外部价值。
- **与 12 法的差异**：相比 Bayesian-Agent 的 5 动作，SLIM 用 leave-one-out 做因果式边际贡献估计（仍属观测性但比朴素计数强），且区分"内化"与"外部保留"。
- **对我们的启示**：reform #2 的 EMA 成功率可补充 leave-one-skill-out 交叉验证；reform #3 的 retire 判据可用边际贡献而非绝对成功率。

#### 11. AutoSkill — Experience-Driven Lifelong Learning（2026-03）
- **机构**：Yutao Yang, Junsong Li, Bihao Zhan 等
- **arXiv**：[2603.01145](https://arxiv.org/abs/2603.01145)
- **核心思想**：从对话与交互轨迹自动导出、维护、复用 skill 的终身学习框架。model-agnostic 插件层，标准化 skill 表示支持跨 agent/用户/任务迁移。把瞬时交互经验转为显式可复用能力。
- **与 12 法的差异**：面向"用户偏好"类长期交互（非代码任务），强调个性化与跨会话累积。
- **对我们的启示**：reform #4 的 debug-trace 蒸馏可参考其"标准化 skill 表示"设计，确保蒸馏出的 skill 可跨题迁移。

### 1.4 上下文工程与提示优化（Context Engineering & Prompt Opt）

#### 12. SAGE/SPO — Stochastic Prompt Optimization（2026-06）
- **机构**：Ziyi Zhu, Luka Smyth, Jinghong Chen 等
- **arXiv**：[2606.18902](https://arxiv.org/abs/2606.18902)
- **核心思想**：明确指出"context engineering 已成为无参数更新的主杠杆"，且**textual gradient 不是真正的梯度**——因此应把自动提示优化（APO）当黑盒随机搜索。提出三种递增策略：error-informed 随机搜索、遗传算法、SAGE（多 agent + 诊断代码执行）。关键发现：无单一策略通吃，效果取决于 landscape 结构与错误类型的交互。
- **与 12 法的差异**：直接 critique 我们已有的 TextGrad（reference.md #3），论证"文本梯度"不可靠。比 GEPA 的进化搜索多了"诊断+验证"耦合论证。
- **对我们的启示**：我们若用 TextGrad 风格优化 A 模型 prompt，应警惕"文本梯度"假设；SAGE 的"定性诊断 + 定量验证"耦合更适合我们的 A->B->C 流水线。

#### 13. SePO — Self-Evolving Prompt Agent（2026-06）
- **机构**：Wangcheng Tao, Han Wu, Weng-Fai Wong
- **arXiv**：[2606.04465](https://arxiv.org/abs/2606.04465)
- **核心思想**：**自指设计**——prompt agent 不仅优化 task agent 的 system prompt，还优化**自己的** system prompt。两阶段：预训练在多任务池上演化，微调迁移到目标任务。在 MBPP（代码生成）等 5 benchmark 上优于 Manual-CoT/TextGrad/MetaSPO +4.49 分。预训练获得的"prompt 优化技能"可泛化到预训练混合之外的任务。
- **与 12 法的差异**：现有方法把 prompt agent 的自身 prompt 固定；SePO 让优化器自我进化，且**显式覆盖代码生成（MBPP）**。
- **对我们的启示**：我们的 A 模型本质是"prompt agent"，其自身的系统 prompt（指导它如何生成 B 的 prompt）也可作为优化目标——这正是 reform #1 的 Meta-Skill 思想在 A 模型侧的体现。

#### 14. Microskill Architecture — 模块化技能驱动的 AI 原生代码生成（2026-06）
- **机构**：Mohammad Zare, Omid Abdolrahmani
- **arXiv**：[2606.05720](https://arxiv.org/abs/2606.05720)
- **核心思想**：受微服务启发，把知识封装为原子化、窄域的 skill capsule，动态路由器只选语义相关 capsule。把上下文分配形式化为**token 预算下的语义相关性约束优化**。案例：token 消耗降 90%，首次编译成功率近翻倍，架构违规归零，自学习机制自动提取注册 7 个新 capsule。
- **与 12 法的差异**：直接面向代码生成，且把"注入多少"形式化为约束优化（而非启发式）。
- **对我们的启示**：**reform #1 + #5 的理论支撑**。我们的"全量紧凑注入"可形式化为"token 预算下的相关性约束优化"；reform #5 的 toggle 对应其"动态路由"。

#### 15. Adaptive Auto-Harness — 持续自改进（2026-06）
- **机构**：Zewen Liu, Zhan Shi, Yisi Sang 等
- **arXiv**：[2606.01770](https://arxiv.org/abs/2606.01770)
- **核心思想**：指出 GEPA/A-Evolve/Meta-Harness 等在固定离线 benchmark 上评估，但真实部署是开放任务流。**单一反复密集更新的 harness 会变脆：准确率早期达峰后下降**。提出按任务适配的持续 harness 构建。
- **与 12 法的差异**：直接 critique GEPA 的评估范式，揭示"peak-then-decline"退化模式。
- **对我们的启示**：reform #2 的 EMA 评分若在固定题集上更新 skill，可能过拟合后退化；应考虑任务流适配或滚动验证集。

### 1.5 技能表示与参数化（Representation & Parametrization）

#### 16. Parametric Skills — 文本技能转 LoRA 参数（2026-06）
- **机构**：Xuan Zhao, Haonan He 等
- **arXiv**：[2606.30015](https://arxiv.org/abs/2606.30015)
- **核心思想**：**把自由文本 skill 在测试时转换为 LoRA adapter 参数**，实现"无上下文"的技能利用。先建大规模高质量 skill 库，用 OpenCode 合成单/多轮利用轨迹，训练 hypernetwork 接收文本 skill 输出 LoRA adapter。在 6 个 SWE 子任务上比 in-context learning 平均高 6.44 分，且为测试时持续学习提供途径。
- **与 12 法的差异**：12 法全部是"上下文空间"优化；Parametric Skills 把 skill 变成**软参数**——这是上下文空间与参数空间的桥梁。
- **对我们的启示**：reform #5 的 injection-mode toggle 可增加"参数化注入"模式——高频 skill 转 LoRA，低频 skill 留上下文。这与我们"从参数空间转向上下文空间"的方向形成有趣张力：也许最优解是混合。

#### 17. From Skill Text to Skill Structure（2026-04）
- **机构**：Qiliang Liang, Hansi Wang 等
- **arXiv**：[2604.24026](https://arxiv.org/abs/2604.24026)
- **核心思想**：当前 skill 多为 SKILL.md 式文本文档，机器可用证据埋在自然语言里。提出 Scheduling-Structural-Logical（SSL）表示，把调用接口、执行结构、具体副作用从单一文本表面解耦。
- **与 12 法的差异**：不优化 skill 内容，而优化 skill**表示格式**。
- **对我们的启示**：我们的 known bug #1（YAML triggers 序列化错误）本质是表示问题；SSL 思路可指导我们重构 skill 文件格式，把 trigger/前置条件/副作用结构化。

#### 18. RESOURCE2SKILL — 多模态资源蒸馏为可执行 skill（2026-06）
- **机构**：Yijia Fan, Zonglin Di, Mingxi Cheng 等
- **arXiv**：[2606.29538](https://arxiv.org/abs/2606.29538)
- **核心思想**：从教程视频、代码仓库、文章等多模态人类资源蒸馏为可执行 skill，组织为分层 multimodal Skill Wiki。视频捕获时序操作、代码捕获工具模式、文章提供概念基础。覆盖不足时同一构造算子可在线获取新 skill。
- **与 12 法的差异**：Trace2Skill 从 agent trace 蒸馏；RESOURCE2SKILL 从**人类多模态资源**蒸馏，信息源更丰富。
- **对我们的启示**：reform #4 的 debug-trace 蒸馏目前只从 GRPO trace 提取；可扩展到从参考解、教程等外部资源蒸馏。

### 1.6 其他相关方法

#### 19. SkillCoach — 自进化评分量规（2026-07）
- **arXiv**：[2607.01874](https://arxiv.org/abs/2607.01874)
- **核心思想**：从真实 rollout 导出 skill-grounded 过程评分量规（rubric），沿 4 维评估：skill selection / skill following / skill composition / skill-grounded reflection。把外部 verifier 作为独立 outcome 信号，**区分过程质量与偶然成功**。演化出的 rubric 还可作过程监督筛选训练轨迹。
- **对我们的启示**：reform #2 的 EMA 成功率是 outcome 信号；SkillCoach 的 4 维过程评估可补足"为何成功/失败"的诊断维度，直接缓解 issue #4（C 模型区分度不足）。

#### 20. EvoLib — 测试时演化库（2026-05）
- **机构**：Weijia Xu, Alessandro Sordoni, Michel Galley 等（Microsoft Research）
- **arXiv**：[2605.14477](https://arxiv.org/abs/2605.14477)
- **核心思想**：测试时学习框架，维护共享库（modular skills + reflective insights），从模型自身推理轨迹自动提取。引入**加权与整合机制**联合优化即时效用与长期价值，让实例特定的抽象演化为更通用可复用的。在数学推理、**代码生成**、多轮 agent 环境上显著优于 test-time scaling 方法。
- **对我们的启示**：reform #4 的 debug-trace 蒸馏可借鉴其"加权整合"——不是每条 trace 都蒸馏，而是按即时效用+长期价值加权后整合为通用 skill。

#### 21. SkillCenter — 大规模源落地 skill 库（2026-07）
- **arXiv**：[2607.07676](https://arxiv.org/abs/2607.07676)
- **核心思想**：据称为最大开源 agent skill 库，强调 source-grounded（每条 skill 可追溯到源），让 agent 输出不仅可执行而且正确、安全、可维护。
- **对我们的启示**：我们的 skill 缺少 provenance 追溯；落地源标注可提升可审计性。

---

## 二、综述类文献

### S1. A Comprehensive Survey on Agent Skills（2026-05）⭐ 最相关
- **标题**：*A Comprehensive Survey on Agent Skills: Taxonomy, Techniques, and Applications*
- **作者**：Yingli Zhou, Wang Shu, Yaodong Su, Yixiang Fang 等
- **arXiv**：[2605.07358](https://arxiv.org/abs/2605.07358) | 资源库：[github.com/JayLZhou/Awesome-Agent-Skills](https://github.com/JayLZhou/Awesome-Agent-Skills)
- **范围**：以 agent skill 为核心的全面综述，定义 skill 为"协调工具/记忆/运行时上下文的可复用过程产物"。
- **分类法/贡献**：按 skill 生命周期 4 阶段组织——**representation / acquisition / retrieval / evolution**。讨论质量控制、互操作性、安全更新、长期能力管理等开放挑战。是我们领域最直接的 survey，其 4 阶段框架可与 reference.md 的 12 法交叉索引。

### S2. From Question Answering to Task Completion（2026-06）
- **标题**：*From Question Answering to Task Completion: A Survey on Agent System and Harness Design*
- **作者**：Jianyuan Guo, Zhiwei Hao 等 | **arXiv**：[2606.20683](https://arxiv.org/abs/2606.20683) | [github.com/ggjy/Awesome-Agent-Engineering](https://github.com/ggjy/Awesome-Agent-Engineering)
- **范围**：以 model-harness 视角审视 LLM agent。
- **分类法/贡献**：把执行 harness 分解为 6 项耦合运行时职责——observation / context / control / action / state / verification。梳理 4 种 agent 工程范式（prompt → workflow → context engineering → harness engineering → agent-native training with co-evolution）。对我们 A->B->C 流水线的 harness 设计有直接指导。

### S3. Experience Compression Spectrum（2026-04）⭐ 概念性突破
- **标题**：*Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents*
- **作者**：Xing Zhang, Guanghui Wang, Yanwei Cui 等
- **arXiv**：[2604.15877](https://arxiv.org/abs/2604.15877)
- **范围**：统一 memory / skill / rule 的框架性论文（含 1136 篇引用的引用分析）。
- **分类法/贡献**：把三者定位在**单一压缩轴**上——episodic memory（5-20x）、procedural skill（50-500x）、declarative rule（1000x+）。揭示"missing diagonal"：**没有任何系统支持跨压缩级别的自适应压缩**。指出 memory 社区与 skill 社区引用交叉率 <1%，各自重复解决相同子问题。
- **对我们的启示**：我们的 experience_pool（skill）与 agent memory 是分开的；可探索"自适应压缩"——高频 skill 压缩为 rule，低频保留为 episodic。

### S4. From Storage to Experience（2026-05）
- **标题**：*From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms*
- **作者**：Jinghao Luo, Yuchen Tian 等 | **arXiv**：[2605.06716](https://arxiv.org/abs/2605.06716)
- **分类法/贡献**：3 阶段进化框架——Storage（轨迹保存）/ Reflection（轨迹精炼）/ Experience（轨迹抽象）。前沿 Experience 阶段的两大变革机制：proactive exploration 与 cross-trajectory abstraction。

### S5. Agent Skills: Architecture, Acquisition, Security（2026-02）
- **标题**：*Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward*
- **作者**：Renjun Xu, Yang Yan | **arXiv**：[2602.12430](https://arxiv.org/abs/2602.12430)
- **分类法/贡献**：survey 级，强调 progressive disclosure、portable skill 定义、与 Model Context Protocol (MCP) 集成。覆盖安全维度。

### S6. Automatic Prompt Optimization 综述三篇（2025-02）
- *A Systematic Survey of Automatic Prompt Optimization Techniques* — [2502.16923](https://arxiv.org/abs/2502.16923)
- *A Survey of Automatic Prompt Engineering: An Optimization Perspective* — [2502.11560](https://arxiv.org/abs/2502.11560)
- *A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm* — [2502.18746](https://arxiv.org/abs/2502.18746)
- 这三篇从优化视角系统梳理 APO，是 TextGrad/GEPA 类工作的背景综述。

### S7. Agent Memory 综述（2026 上半年，补充）
- *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers* — [2603.07670](https://arxiv.org/abs/2603.07670)
- *Rethinking Memory Mechanisms of Foundation Agents in the Second Half: A Survey* — [2602.06052](https://arxiv.org/abs/2602.06052)
- *Anatomy of Agentic Memory: Taxonomy and Empirical Analysis* — [2602.19320](https://arxiv.org/abs/2602.19320)
- *Graph-based Agent Memory: Taxonomy, Techniques, and Applications* — [2602.05665](https://arxiv.org/abs/2602.05665)
- *From Agent Traces to Trust: A Survey of Evidence Tracing and Execution Provenance* — [2606.04990](https://arxiv.org/abs/2606.04990)（与 trace 蒸馏相关）

---

## 三、趋势综合（2025-2026 跨文献提炼）

### 趋势 1：从"更多 skill"转向"更少但更结构化"
GraSP（2-3 skill 优于全量）、SkillComposer（联合子集预测）、Microskill（token 预算约束）一致表明：**skill 注入存在最优点，过载反而有害**。这与 reference.md 中 SkillLens 的"负迁移真实存在"结论呼应，但 2026 工作给出了量化边界与结构化解决方案（DAG/图/序列预测）。我们 402 条全注入的做法已属被验证的次优范式。

### 趋势 2：验证门控从"可选"变"必需"，且暴露 judge 偏置风险
HDSO（control/treatment）、GRASP（硬回归预算）、RSEA（held-out keep-better）、The Blind Curator（false-pass 阈值）形成共识：**无门控的自演化是高方差且不安全的**。更关键的是 The Blind Curator 证明偏置 judge 会"静默"瘫痪退役——不暴露在任何聚合指标。这与我们 issue #4（C 模型区分度不足）高度相关，是 2026 最重要的安全发现。

### 趋势 3：Memory-Skill-Rule 统一为"压缩谱"
Experience Compression Spectrum 提出 memory/skill/rule 是同一压缩轴上的不同点，且"missing diagonal"（无系统支持跨级别自适应压缩）是开放问题。AutoSkill、EvoLib 也在向"经验→skill→rule"的自动升级靠拢。这预示 skill 与 memory 的工程边界正在消融。

### 趋势 4：Skill-as-Soft-Parameters——上下文与参数空间的再融合
Parametric Skills（文本 skill → LoRA）打破了"上下文空间 vs 参数空间"的二元对立。我们 CLAUDE.md 写明研究方向"从参数空间转向上下文空间"，但 2026 出现了**回流**：把高频上下文 skill 参数化为轻量 adapter。这可能意味着最优解是混合而非纯上下文。

### 趋势 5：Skill 图结构化与类型化关系
SkillDAG、SkillGraph、GraSP 都把 skill 间关系（依赖/冲突/特化/重复）建模为类型化图。这超越 reference.md 中扁平 skill 库 + 语义检索的范式，让"何时合并/拆分/删除"有结构线索。

### 趋势 6：Harness 工程成为独立学科
"From QA to Task Completion" survey 把 harness（运行时脚手架）从"辅助工具"提升为与基础模型对等的一等公民，分解为 6 项耦合职责。Adaptive Auto-Harness 进一步揭示 harness 在开放任务流上的"peak-then-decline"退化。我们的 A->B->C 流水线本质是 harness，应按此框架审计。

### 趋势 7：过程评估 vs 结果评估的分离
SkillCoach（4 维过程 rubric）、The Blind Curator（区分机制失效与结果失效）都强调：**最终成功太粗，无法区分"靠试错通过"与"正确使用 skill"**。这与我们 C 模型仅给 outcome 分数的现状形成对照。

### 趋势 8：TextGrad 式"文本梯度"受到质疑
SAGE 明确论证 textual gradient 不是真梯度，应回归黑盒搜索。SePO 在 MBPP 上超越 TextGrad。我们若考虑 TextGrad 路线需警惕此假设。

---

## 四、对我们 5 项改造的启示

| 我们的改造 | 相关新工作 | 具体启示 |
|-----------|-----------|---------|
| **(1) 全量 skill 紧凑注入 + Meta-Skill "selective ignore"** | GraSP, SkillComposer, Microskill, SkillDAG | GraSP 证明"2-3 聚焦 skill 优于全量"——**直接验证我们的方向**。建议把"全量注入"改为"token 预算下相关性约束优化"（Microskill 形式化），并用 SkillComposer 的联合子集-顺序预测替代 top-k 检索。Meta-Skill 的"selective ignore"可借鉴 SkillDAG 的冲突信号。 |
| **(2) online EMA success-rate scoring（观测性，非因果 ATE）** | The Blind Curator, HDSO, SkillCoach, SLIM | **最大风险**：The Blind Curator 证明 false-pass 偏置会静默瘫痪基于贡献的退役。部署 EMA 评分前**必须先对 C 模型跑 defect-injection 审计**。可补充 HDSO 的 control/treatment 配对做交叉验证，以及 SkillCoach 的 4 维过程评估补足 outcome 信号的粗糙。SLIM 的 leave-one-skill-out 比朴素 EMA 更接近因果。 |
| **(3) ADD/UPDATE/DELETE/NONE skill-pool loop via LLM** | SkillBrew, GRASP, RSEA, SLIM, AutoSkill | SkillBrew 的三目标（useful/diverse/coverage）Pareto 框架可直接套用——DELETE 不能只看成功率，还要保 coverage。GRASP 的"硬回归预算"让 NONE 从被动变主动防回归。RSEA 的 held-out keep-better gate 保证"monotone-safe"。SLIM 的 retain/retire/expand 三操作与我们四动作对齐。 |
| **(4) debug-trace 蒸馏** | EvoLib, RESOURCE2SKILL, HDSO, ExpGraph | EvoLib 的"加权整合"（即时效用+长期价值）比逐条蒸馏更优。HDSO 的"显式验证计划"模板可约束蒸馏质量。RESOURCE2SKILL 启发我们扩展信息源到 agent trace 之外（参考解、教程）。ExpGraph 的图结构记忆可组织蒸馏产物。 |
| **(5) injection-mode toggle** | Parametric Skills, From Skill Text to Structure, SkillDAG, Microskill | Parametric Skills 提供"参数化注入"新模式——高频 skill 转 LoRA，低频留上下文（混合最优）。From Skill Text to Structure 的 SSL 表示可指导 toggle 的格式切换。Microskill 的动态路由是 toggle 的工程实现。SkillDAG 的 agent 可调用结构化接口让 toggle 在推理时动态决策。 |

### 4.1 可落地首实验：focused-injection 消融（对应 Step 1 负结果）

**动机**：Step 1 验证（`memory/step1-validation-result-20260713.md`）显示全量注入 50 条 active skill 在 MBPP_pro/BigCodeBench 上零收益，而 GraSP（arXiv:2604.17870，待复核）给出"2-3 个聚焦 skill 优于完整文档；过多 skill 有害"的反直觉证据。两者共同指向假说：**全量注入稀释了信号，少量 per-task 聚焦 skill 才可能产生正效应**。

**实验设计**（本轮仅就绪方案，不执行）：

| 组 | 注入方式 | skill 数 | 选取方式 | 目的 |
|----|---------|---------|---------|------|
| Baseline | 无 | 0 | - | 基准 |
| Full | 全量 compact | 50 | active 全集（Step 1 现状） | 复现零效应 |
| Random-3 | 随机 | 3 | 从 active 池随机 | 控制"少注入"本身 |
| Focused-3 | per-task 聚焦 | 3 | 按题目语义检索 top-3（或 LLM 选 3） | 测 GraSP 假说 |

- 数据集：MBPP_pro n=100（确定性单元测试，已知 baseline 0.685）+ SkillsBench 子集（skill-specific，预期效应更可观测，见 `data/skill_benchmarks/README.md` §3.0 目标 A/B）。
- 判读：Focused-3 > Full 且 Focused-3 > Random-3 → 支持"聚焦优于全量"，应把注入模式从全量改为 per-task 聚焦检索（呼应 5 项改造 #1 的 retrieve 路径回归，但用更小的 k=3）。
- 与综述的联结：Focused-3 直接验证趋势 1（GraSP）；若 Focused-3 仍 null，则问题在 skill 池质量（70% 有害），转向 §3.0 目标 A（用 SkillsBench 自带 skill 测我们的优化方法本身）。

**定位**：本实验属 P0（先行），是多 agent 验证（导师建议 #2）的前置之一（见 `docs/multi_agent_validation_plan.md` §0 前置条件）。

---

## 五、参考资料（arXiv 主源）

### 新增方法
- HDSO: https://arxiv.org/abs/2606.22330
- The Blind Curator: https://arxiv.org/abs/2607.07436
- GRASP: https://arxiv.org/abs/2605.29668
- RSEA: https://arxiv.org/abs/2606.28374
- GraSP: https://arxiv.org/abs/2604.17870
- SkillComposer: https://arxiv.org/abs/2606.32025
- SkillDAG: https://arxiv.org/abs/2606.03056
- SkillGraph: https://arxiv.org/abs/2605.12039
- SkillBrew: https://arxiv.org/abs/2605.29440
- SLIM: https://arxiv.org/abs/2605.10923
- AutoSkill: https://arxiv.org/abs/2603.01145
- SAGE/SPO: https://arxiv.org/abs/2606.18902
- SePO: https://arxiv.org/abs/2606.04465
- Microskill Architecture: https://arxiv.org/abs/2606.05720
- Adaptive Auto-Harness: https://arxiv.org/abs/2606.01770
- Parametric Skills: https://arxiv.org/abs/2606.30015
- From Skill Text to Skill Structure: https://arxiv.org/abs/2604.24026
- RESOURCE2SKILL: https://arxiv.org/abs/2606.29538
- SkillCoach: https://arxiv.org/abs/2607.01874
- EvoLib: https://arxiv.org/abs/2605.14477
- SkillCenter: https://arxiv.org/abs/2607.07676

### 综述类
- A Comprehensive Survey on Agent Skills: https://arxiv.org/abs/2605.07358 | https://github.com/JayLZhou/Awesome-Agent-Skills
- From QA to Task Completion (Harness Design): https://arxiv.org/abs/2606.20683 | https://github.com/ggjy/Awesome-Agent-Engineering
- Experience Compression Spectrum: https://arxiv.org/abs/2604.15877
- From Storage to Experience (Memory): https://arxiv.org/abs/2605.06716
- Agent Skills: Architecture, Acquisition, Security: https://arxiv.org/abs/2602.12430
- Systematic Survey of Automatic Prompt Optimization: https://arxiv.org/abs/2502.16923
- Survey of Automatic Prompt Engineering (Optimization Perspective): https://arxiv.org/abs/2502.11560
- Survey of APO with Heuristic Search: https://arxiv.org/abs/2502.18746
- Memory for Autonomous LLM Agents: https://arxiv.org/abs/2603.07670
- Rethinking Memory Mechanisms of Foundation Agents: https://arxiv.org/abs/2602.06052
- Anatomy of Agentic Memory: https://arxiv.org/abs/2602.19320
- Graph-based Agent Memory: https://arxiv.org/abs/2602.05665
- From Agent Traces to Trust (Evidence Tracing): https://arxiv.org/abs/2606.04990

---

## 附：检索说明

本文档全部论文均通过 arXiv API (`export.arxiv.org/api/query`) 在 2026-07-13 检索并验证存在，摘要取自 arxiv 原文。未发现无法验证的条目。领域在 2026 上半年产出密集（仅 "skill + agent" 查询即返回数十篇 2026 年论文），说明该方向处于活跃期。

> **独立复核状态（2026-07-14）**：原 agent 2026-07-13 报告"全部核验、未发现无法验证条目"。2026-07-14 独立用 arXiv API 复核时遭遇限流（429），仅确认 SkillOpt(2605.23904) 真实（标题完全吻合）；SkillsBench(2602.12670) 与 Blind Curator(2607.07436) 此前已由 agent curl 核实；GraSP(2604.17870) / SkillLens(2605.23899) / HDSO(2606.22330) / SkillComposer(2606.32025) / SLIM(2605.10923) 等 5 个决策性 ID 因限流待复核。**引用这些待复核条目作决策性结论时需谨慎**，待网络恢复后逐条复核并更新本标注。
