# Skill Self-Evolution for AIME Math Reasoning — For Review (2026-10-15)

> 本包是 **AIME 数值题上 skill 自进化**的完整代码/数据/产物归档，供重新 review。
> 文档围绕一个**诚实的负面结果**展开：3 个月、10+ 次尝试，机制全链路验证通过，
> 但在 AIME 上始终无法稳定超越初始 skill S₀。下方把每一次尝试、精确断点、困难与疑惑点写清楚。

---

## 0. TL;DR（给 reviewer 的一页摘要）

| 维度 | 结论 |
|------|------|
| **任务** | AIME 竞赛数学（数值答案 0–999，无选择题），B 模型 = DeepSeek-V3.2（冻结 API），C 模型 = DeepSeek-V4-Flash/Qwen3-235B（冻结）|
| **方法** | Skill self-evolution：不训练 B 参数，进化 B 的 system prompt（技能 markdown）；EFL facet-tree + expand/overlay/strengthen 算子 + 准入门控 |
| **AIME 全程** | 2026-07~10，10+ 次尝试，**无一稳定超越 S₀**（详见 §3 时间线）|
| **最新 3 次（EFL 线）** | run#1 flat=0 admit；run#2 facet-tree=0 admit；run#3 facet-tree+group-admit gate=0 admit |
| **门控正确性** | ✅ 门控**不是瓶颈**。run#3 放宽门控（per-domain 隔离）后仍 0 admit，证明 0 admit 不是「门太严」而是 patch 方向问题 |
| **精确断点** | 🔴 expand 算子 **10/10 轮 target_type 完全相同** = 泛化桶 `Equation / setup errors`，生成的 10 个 section 全是方程验证变体，从未针对几何 |
| **根因链** | focus 选「最高频未覆盖 failure type」→ C 模型把大量几何失败归入泛化方程桶（每轮 Computation&Verification facet 内 3 次）→ 几何专属类型从未成为 target → 永远生成泛化方程验证补丁 |
| **诚实结论** | 机制全链路跑通（taxonomy 涌现 ✅、facet 路由 ✅、门控判据 ✅），但 patch 生成方向被 C 模型分类偏差 + focus 频率选择结构性锁定，无法产生 domain-specific 增益 |

---

## 1. 项目背景与约束

**A→B→C 流水线**（Training-Free GRPO + Prompt Optimization）：
```
A (Qwen3-4B, GRPO 训练) → system prompt → B (DeepSeek-V3.2 API, 冻结) → 数学解 → C (DeepSeek-V4-Flash, 冻结) → 评分 + 单测 → 奖励回传更新 A
```
本包聚焦**技能自进化**子问题：在不更新 B 模型参数的前提下，通过进化 B 的 system prompt（技能）来提升 AIME 解题 pass 率。

**三条硬约束（在效）：**
1. 小样本可行，**不扩数据集**（训练 n=18，准入 n=30）。
2. 必须是 **skill 自进化**优化，不切换到 scaffold/agent 框架。
3. 算法**自动发现/涌现** facet，不人工注入分类法。（注：per-domain 分组用的 `classify_item_group` 是零 API 确定性关键词分类器，只用于切分准入信号，是测量仪器，不是注入 taxonomy。）

**授权后端**：仅 DeepSeek-V3.2（B）与 DeepSeek-V4-Flash / Qwen3-235B（C）。

**数据集**：AIME-2024（30 题）、AIME-2025（30 题），数值答案 0–999 整数，无选项。AIME 是「最干净的数学推理试金石」——无 MCQ 选项捷径，必须算出具体整数。

---

## 2. 包结构

```
for_review/
├── README.md                  ← 本文件
├── code/                      ← AIME 线源码（9 个 .py，API key 已脱敏）
│   ├── skill_evolver_aime_v2.py   (4717行) 进化器主程序：facet-tree + 5算子 + 门控
│   ├── tf_global_v2_aime.py       (412行)  评测引擎：run_one / eval_on / bootstrap_ci / precompute_s0
│   ├── tf_global_v2_math.py       (773行)  评测后端基类（被 _aime 继承，含 chat_target 调用 + AIME 数值抽取）
│   ├── facet_library.py          (566行)  EFL：持久化/可生长/LLM分类/聚类的 facet taxonomy
│   ├── templates_aime.py          (138行)  AIME 失败模式蒸馏 prompt 模板（纯常量，无依赖）
│   ├── templates_math.py          (142行)  数学题失败模式蒸馏 prompt 模板（纯常量，无依赖）
│   ├── eval_test_multiseed.py     (186行)  多 seed 复测脚本
│   ├── eval_aime24_multiseed.py        AIME-2024 多 seed 评测
│   └── eval_aime25_multiseed.py        AIME-2025 多 seed 评测
├── data/                      ← AIME 数据 + 初始 skill
│   ├── aime24.jsonl           (12KB) AIME-2024 (30 题)
│   ├── aime25.jsonl           (16KB) AIME-2025 (30 题)
│   └── skill_aime_initial_v2.md  (600B) 初始技能（content hash=c11d7dc0c8）
├── results/                   ← 三次 EFL 实验全量产物
│   ├── run1_flat_20261013/              run#1 flat 进化（59K）
│   ├── run2_facet_tree_20261013/         run#2 facet-tree（444K）
│   └── run3_group_admit_20261014/        run#3 facet-tree + group-admit gate（440K）
├── docs/                      ← 设计文档（参考）
│   ├── facet_tree_skill_evolution_design.md
│   ├── training_free_grpo_fusion_design.md
│   ├── ea-skill-evolution-survey-20260819.md
│   └── 进展汇报_2026-09-14.md
└── archive/                   ← 旧版本归档（LiveMath 主线 + 旧 review 文档 + 旧 zip）
    ├── previous_zips/         4 个历史 zip（2026-09-17/24/28 + code）
    ├── review_docs/           7 个旧 review/response 文档
    └── live_math_line/        旧 LiveMath 线的 code/data/results
```

### 2.1 单个 run 产物（以 run#3 为例）
```
run3_group_admit_20261014/
├── evolver_log.json      (40KB) 核心：10 轮迭代日志，每轮含 facet_tree/expand/overlay/crossover/branch_failure_types
├── best_skill_evolved.md (599B) = 初始 skill（未进化）
├── best_skill_iter{1..10}.md   每轮 best（全是 599B = init）
├── archive_iter{1..10}.json    每轮候选归档
├── archive_final.json    (32KB) 终局归档
├── tree_iter{1..10}.json       每轮 facet tree 快照
├── facet_library.json    (7.9KB) 涌现的 facet taxonomy
└── s0_cache.json         (1.4KB) S0 基线缓存（30 题，pass@k 基线）
```

### 2.2 ⚠️ 代码依赖与完整性说明（必读）

本包 `code/` 的 9 个 .py **除一个外部依赖外自洽**：

**外部依赖（未打包）**：`skillopt` 包（`/data/zhuldz/skillopt`，4.9GB，是独立的 BASELINE 项目，按用户硬性规则「不得修改」，不纳入本 review 包）。`tf_global_v2_math.py`、`skill_evolver_aime_v2.py`、`eval_aime24_multiseed.py` 均从 `skillopt` 引入：
- `skillopt.model.chat_target` — 统一的 LLM 调用接口（DeepSeek-V3.2 后端）
- `skillopt.envs.livemathematicianbench.rollout` — `_build_system` / `_build_user`（system/user prompt 构造）
- `skillopt.envs.livemathematicianbench.evaluator` — `evaluate`（答案判定）
- `skillopt.model.qwen_backend.TARGET_CONFIG` — 后端配置

**已打包的内部依赖闭包**：`skill_evolver_aime_v2 → {tf_global_v2_aime, tf_global_v2_math} → {eval_aime24_multiseed, templates_aime, templates_math}`，除上述 skillopt 外全部在 `code/` 内。

**硬编码绝对路径**（需在 reviewer 本地对应位置存在）：
- `tf_global_v2_aime.py:31` `_LUNWEN="/data/zhuldz/lunwen"`
- `tf_global_v2_aime.py:47-48` `S0_DEFAULT`/`AIME24_DATA` 指向 `/data/zhuldz/lunwen/data/`
- `skill_evolver_aime_v2.py:25` `_SKILLOPT="/data/zhuldz/skillopt"`
- `eval_test_multiseed.py:25-31` 旧 LiveMath skill 路径（仅复测脚本用，非主链路）

**运行前需**：`export QWEN_CHAT_API_KEY=<授权 key>`（唯一硬编码 secret 已脱敏为 env 占位）。reviewer 主要用于**代码阅读与追踪**，完整复现需 skillopt 环境 + API 授权。

---

## 3. AIME 数据集全部尝试时间线（核心章节）

这是 reviewer 最需要看的一节：**我们在 AIME 上做过的所有尝试及其精确结果**。按时间顺序，无一省略。

### 3.1 跨任务泛化验证（2026-07-31）— 负结论
**问题**：LiveMath 选择题上 evolver +16.1pp（高度显著），能否泛化到 AIME24 数值题（不同题源、完全 held-out）？

| skill | seed1 | seed2 | seed3 | mean | Δvs S0 | t | p | 显著? |
|-------|-------|-------|-------|------|--------|---|---|-------|
| S0 | 0.633 | 0.611 | 0.578 | 0.607 | — | — | — | — |
| v2 | 0.600 | 0.622 | 0.578 | 0.600 | −0.7pp | −0.32 | 0.75 | ❌ |
| evolver | 0.611 | 0.600 | 0.667 | 0.626 | +1.9pp | 0.70 | 0.49 | ❌ |

- **配置**：AIME24 30 题，k=3 投票，eval_temp=0.3，3 seed。
- **结论**：增益从 LiveMath +16.1pp 暴跌到 +1.9pp 且不显著 → **skill 学到的是选择题形态特有诊断程序，不跨任务形态泛化**。这是第一次明确泛化边界。

### 3.2 AIME 完整重跑链（2026-08-07）— 负结论，确认泛化边界
**问题**：干净复现 08-05 阴性结果（fresh，无缓存复用）。

| skill | mean_pass | passes(4 seed) | Δvs S0 | t | p |
|-------|-----------|----------------|--------|---|---|
| S0 | 0.5083 | [0.467,0.522,0.511,0.533] | — | — | — |
| v2 | 0.5083 | [0.533,0.467,0.544,0.489] | +0.00pp | −0.00 | 1.00 |
| evolver | 0.4917 | [0.511,0.444,0.500,0.511] | −1.67pp | −0.62 | 0.54 |

- **配置**：AIME25 30 题，k=3，4 seed，fresh 全新缓存。
- **机制验证**：链诚实运行非崩溃——v2 跑了 2 iter（pool 真进化），iter2 val 一度 +0.111 但 CI_lower=0.000 → CI 门控诚实 KEEP PREV；evolver 4 iter 全 delta 小且 CI 跨 0 → 全 KEEP PREV。
- **结论**：确认 AIME 是泛化边界，Run F 在 live 数学题的 +26.9pp 增益不跨题源。CI 门控「宁可不涨也别过拟合」原则有效。

### 3.3 数据集换型决策（2026-09-23）— 方向转折点
**背景**：code 域（mbpp_pro/HumanEval Pro/LCB）三条路全判死（test 不泛化），LiveMath 数学没超 skillopt（other_60 −0.017）。

**关键诊断**（`aime-vs-livemath-content-diagnosis`）：
- skillopt 在 LiveMath 有效 ⟺ (a) 任务有共享元结构（MCQ「选项 vs 定理」）产出可复用结构失败模式 + (b) S₀ 尚未覆盖。
- **AIME 两者皆不满足**：自由作答数值题无 MCQ 元结构 + S₀ 已覆盖形态级。
- 决策：复跑 AIME，但用**最新 on/off 算法**（`skill_evolver_math.py` 的 pool_filter/facet_tree/expand），非复跑旧简版。

### 3.4 AIME v2 evolver 构建 + MCQ 耦合发现（2026-09-30）
- 移植 math evolver 后端到 AIME（`skill_evolver_aime_v2.py`）：item schema 改 {problem,answer,year}，prompt 改输出 `\boxed{N}`，shuffle_seed 设 no-op。
- **关键发现（`aime-port-mcq-coupling`）**：AIME 无选项→shuffle_seed 不适用，退化 pool_filter ON/OFF 单变量。早期 evolver 的 FAILURE_FOCI 8 维**无几何靶点**，6/11 失败题是几何 → patcher 反复生成同一非几何段 → 撞天花板 +2/30。

### 3.5 AIME25 三方公平对比（2026-10-08）— 负结论，根因定位
**问题**：S₀ / evolver / skillopt 三方公平对比，谁超 S₀？

| skill | mean_pass | per-seed | Δvs S0 | 显著? |
|-------|-----------|----------|--------|-------|
| **S0** | **0.4963** | 0.467,0.511,0.511 | — | — |
| evolver | 0.5000 | 0.489,0.533,0.478 | +0.004 | ❌ (噪声) |
| skillopt | 0.5074 | 0.444,0.544,0.533 | +0.011 | ❌ (噪声) |

- **配置**：AIME25 30 题，k=3 投票@0.3，3 seed，统一协议。
- **核心发现**：**两方法均未显著超越 S0**。evolver Δ=+0.004 纯采样噪声（GA 终局 best_skill == S0，0 promoted）；skillopt Δ=+0.011 跨 0 噪声。
- **S₀ 是强基线**，两方法都无法可靠改进它——本轮 AIME 根本发现。
- **三层归因**：(1) 搜索空间受限（FAILURE_FOCI 无几何靶点，几何题 patch 撞天花板）；(2) 严 CI 门防假阳性（n=30 CI 宽 ±0.1，+2/30 的 CI 下界 −0.033<0→拒收）；(3) S₀ 非过强（baseline 0.496，理论空间存在但被 (1) 锁死）。

### 3.6 失败机制根因（2026-10-13）— evolver/skillopt 镜像互补
**问题**：为何 evolver 和 skillopt 都打不过 S₀？失败在哪一环？

| | 发现 (Discovery) | 接受 (Gate) |
|---|---|---|
| **evolver** | ❌ 关键词路由器看不见几何 | ✅ Beta-Binomial LCB，统计诚实 |
| **skillopt** | ✅ LLM 语义分类，正确抓到几何 | ❌ n=6 严格 `>` 单点比较，噪声杀信号 |

- **evolver 失败**：`classify_failure_type` 纯关键词匹配题面，几何题（triangle/hexagon/angle）不命中 FAILURE_FOCI[7]→落 foci[5]（answer-form）兜底→给几何题写答案格式补丁。**发现环节误路由，不是缺几何轴**。gate 是好的。
- **skillopt 失败**：reflect LLM 正确标 geometry（step6/15/22 诊断），但 gate=`cand>current` 严格 `>`，无 CI/配对/bootstrap，n=6 selection 粒度 16.7pp。step6 ACCEPT 5/6→test 3/6=−0.167 负迁移（winner's curse）。23 步全 REJECT。
- **约束#3**：两者都用静态人写类目表，**都无「累积/聚类/涌现新 facet」模块**。
- **方向**：把 evolver 的发现从关键词换成语义（LLM 读轨迹分类），保留其 LCB gate → **EFL（Emergent Facet Library）应运而生**。

### 3.7 Agent 范式对照实验（2026-10-10/12）— 负/正对照
**问题**：加 agent 能力（ReAct / CodeAgent PoT）能否提升 AIME？（注意：这超出约束#2「不切换到 scaffold/agent」，但作为对照实验跑，非主线）

| skill | single-call | ReAct | CodeAgent(PoT) |
|-------|-------------|-------|----------------|
| S0 | 0.4741 | 0.4000 (Δ−7.4pp, p=0.046 **SIG负**) | 0.6222 (Δ+14.8pp, p=0.0007 **SIG正**) |
| evolver | 0.5185 | 0.3889 (Δ−13.0pp, p=0.0003 **SIG负**) | 0.5815 (Δ+6.3pp, p=0.147 NS) |

- **ReAct 负**：两 skill 均显著有害。机制：注意力分散（Think→Act→Observe 拉向元决策）+ token 预算挤占 + 过度验证。
- **PoT 正**：S₀ 显著 +14.8pp。竞赛数学靠「写代码算」远胜「交错调工具」。
- **跨范式稳健发现**：单次 evolver 胜 S₀(+4.4pp)，但两种 agent 范式下排名都翻转（S₀ 胜）→ **强 skill 压缩 agent headroom，非正交可加**。
- **对主线意义**：纯 prompt 优化(+4.4pp) > 加 ReAct agent 能力(−13pp) → skill-only 优化的正当性反面证据。但 PoT 正向也提示：上下文空间进化与 agent 能力不是对立，**范式选择决定方向**。

### 3.8 EFL 三次实验（2026-10-13/14）— 最新主线，3 次 0-admit
**EFL 设计**：承接 §3.6，把 evolver 发现从关键词换成 LLM 语义分类（自动涌现 facet），保留 LCB gate，并新增 per-facet-group 准入门控。

| Run | 配置 | iters | best_n_sec | best_val_delta | admits | best_skill | 耗时 |
|-----|------|-------|-----------|---------------|--------|------------|------|
| #1 (10-13) | flat, 无 facet-tree | 6(早停) | 4 | 0.0 | 0 | == init | 4.82h |
| #2 (10-13) | facet-tree + subset_admit | 10 | 4 | 0.0 | 0 | == init (c11d7dc0c8) | 7.26h |
| #3 (10-14) | facet-tree + group_admit(ON) | 10 | 4 | 0.0 | 0 | == init (c11d7dc0c8) | 7.4h |

- **run#1**（`efl-flat-run-summary`）：6 轮早停，0 ACCEPT。LCB gate 6 次正确拒绝。信号强度递增（+0.011→+0.033→+0.044）但 CI 始终负。几何 label 每轮涌现但被 merge 过吸收进 F7（Equation/setup）→ 永不 promote。**坐实 run#2 facet-tree 必要性**。
- **run#2**（`efl-tree-run-launched`）：10 轮，0 admitted，best=init。expand 5 不同轴轮换（4 轮选几何），F14 Torus 独立涌现。但 iter9/10 末期 merge 又把几何 candidate 回吞（F7 cnt→79）。**过吸收仅延迟未根治**。
- **run#3**（`group-admit-gate-implemented`）：放宽门控（per-domain 隔离），10 轮 0 admit，best=init。**门控 0 假阳性**（Geometry CI 全负，正确拒绝）。详见 §4。

**run#3 详细（evolver_log.json）**：
- `baseline_val` (S0): pass=0.6, ci=−0.1222
- 10 轮 `best_val_delta=0.0`, `best_val_ci=−0.1222`, `best_n_sec=4`（始终未脱离初始 4-section 技能）
- `rescue_by_source={}`（subset + group 两条救援路径都没触发）
- `operator_counts`：全算子 promoted=0（facet_edit 30/overlay 10/expand 10 attempted，promoted 0）

---

## 4. 门控正确性结论（run#3 的核心价值）

run#2 → run#3 唯一变量是**开启 group-admit 门控放宽**。结果同样是 0 admit、best_skill 同 hash。但 run#3 提供了 **per-domain 可见性**：

```
iter1 group_signal (per-domain CI):
  Geometry: -0.1667   Combinatorics: +0.0333
  NumberTheory: -0.1667   Algebra: -0.3333
```

每轮 Geometry 的 CI 都是负数 → group-admit 判据「无组 CI>0」正确拒绝（10 轮 0 误判）。这证明 **0 admit 不是因为门太严**——门已放宽到 per-domain 粒度，几何信号本身在 k=3 下确实达不到 CI>0。**瓶颈在 patch 生成方向，不在门控。**

---

## 5. 🔴 精确断点：expand 的 target_type 结构性锁死

这是本次 review 最关键的发现。从 `evolver_log.json` 逐轮提取 expand 的 `target_type`：

| iter | target_type | 生成的 section |
|------|-------------|---------------|
| 1 | `Equation / setup errors (wrong variable, wrong formula, misread constraint)` | Equation & Constraint Verification |
| 2 | 同上 | Equation/Setup Verification via Dimensional/Unit Consistency & Symmetry Check |
| 3 | 同上 | Equation & Constraint Verification |
| 4 | 同上 | Equation/Constraint Verification |
| 5 | 同上 | Equation & Constraint Verification |
| 6 | 同上 | Equation Cross-Check |
| 7 | 同上 | Equation/Constraint Reinterpretation Check |
| 8 | 同上 | Equation & Constraint Verification |
| 9 | 同上 | Equation Setup Verification via Dimensional/Unit Consistency |
| 10 | 同上 | Equation Setup Verification |

**10/10 轮 target_type 完全相同，10 个生成的 section 全是方程验证变体，无一针对几何。**

### 5.1 根因链（三层）

1. **focus 选择层**：expand 算子的 coverage 选择逻辑选「最高频未覆盖的 failure type」作为 `target_type`。
2. **C 模型分类偏差**：C 模型把大量几何失败归入泛化桶 `Equation / setup errors`。证据（`branch_failure_types`，每轮 Computation & Verification facet 内）：
   - `Equation / setup errors (…)`：每轮出现 1–3 次（最高频）
   - `Geometry Structure Misidentification`：1 次
   - `Geometry Chain Setup`：1 次
   - `Geometry Parallel Extension`：1 次
   - `Geometry Chain Tangent Setup`：1 次（仅 iter10）

   几何失败被**拆散**成多个低频专属类型，而泛化桶**聚合**成单一高频类型 → focus 永远选泛化桶。
3. **补丁生成层**：target_type 一旦是泛化方程桶，SECTION_EXPAND_PROMPT 即使要求 domain-specific，LLM 也只能生成泛化「方程/约束验证」section（因为 target 本身就不含几何语义）。

### 5.2 后果
- 所有 patch 都是「验证你的方程对不对」这类对几何题无差别泛化指令
- 几何题（61/73/81，baseline 零分）需要的「先做辅助线/识别结构/建系」这类 domain-specific 启发从未被生成
- patch 对几何题无增益 → Geometry group CI 持续为负 → 门控正确拒绝 → 0 promote → best_skill 不变

**这不是门控问题，是 patch 生成方向被结构性锁死。** 门控放宽（run#3）只能让「已有真信号」通过，无法凭空创造信号。

### 5.3 与历史根因的呼应
§3.5（10-08）的「FAILURE_FOCI 无几何靶点」→ §3.6（10-13）的「关键词路由器看不见几何」→ EFL（10-13）改用 LLM 语义分类**确实让几何 facet 涌现了**（run#1/2 每轮新 geometry label 涌现 ✅）。但 §5 表明：**涌现不等于被选中**——facet 涌现了，C 模型却把几何失败过分类进泛化桶，focus 的频率选择仍锁死在方程桶。这是比 §3.6 更深一层的断点。

---

## 6. 方法：Skill 自进化机制

### 6.1 Skill 表示
技能是一段 markdown，由若干 `## ` section 组成。进化 = 在技能上施加算子（add/edit/overlay section），由门控决定是否晋升为新 best。

### 6.2 EFL（Emergent Facet Library）
`facet_library.py` 维护一个**自动涌现**的失败类型 taxonomy：
- C 模型对每次失败标注 failure type（自由文本）
- EFL 用 LLM 分类 + 聚类把自由文本归并成 facet 树
- 进化器从 facet 树里选「最高频未覆盖」的 failure type 作为 expand 的 target

### 6.3 算子（5 种）
| 算子 | 作用 | run#3 计数 |
|------|------|-----------|
| `facet_edit` | 编辑现有 section | attempted=30, admitted=30, **promoted=0** |
| `expand` | 新增一个 section 针对 target failure | attempted=10, admitted=0, **promoted=0** |
| `overlay` | 覆盖/强化 | attempted=10, admitted=0, promoted=0 |
| `crossover` | 组合两个技能 | fired=0 |
| `strengthen` | 强化 best | attempted=0 |

### 6.4 准入门控（关键设计）
- **主门控（LCB）**：候选整体 delta 的 bootstrap 置信下界 > 0 才晋升。EVAL_TEMP=0.3 → S₀ 重算有噪声，用 k=3 投票降低但不消除。
- **subset_admit（2-way，默认 ON）**：整体未过门时，按「答案形式 vs 非」切 2 子集，任一子集 CI>0 仍可救。
- **facet_group_admit（N-way，run#3 新增，默认 OFF）**：把 per-item delta 按数学域（Geometry/Combinatorics/NumberTheory/Algebra）切 N 子集，**任一组 CI>0 且无组 CI<-eps** 仍可晋升。零 API 确定性关键词分类器 `classify_item_group`。
- **设计动机**：信号全集中在 Geometry（12/30 题，10 题 baseline 不及格含 3 个零分题 61/73/81）。几何补丁修这 3 题 → 整体 ~+0.10/题 CI<0（被 30 题稀释），但 Geometry 子集 ~+0.25/题 CI 可>0。group-admit 旨在隔离被淹没的几何信号。

---

## 7. 🟡 未来的疑惑点 / 待 GPT 指点（共 6 问）

下列是我们目前卡住、想请 reviewer 评估的方向。每条附现状数据、初步判断与顾虑。**Q1–Q3 是机制级硬问题（直接卡 0 admit），Q4–Q6 是策略级方向问题（决定下一阶段往哪走）。**

### Q1. 为什么 skillopt 在 LiveMath 上有效，在 AIME 上却失效？根因是数据集还是机制？
**现状（两数据集对比）**：

| | LiveMath（选择题 MCQ）| AIME（数值题 0–999）|
|---|---|---|
| 任务形态 | 四选一，有「选项 vs 定理」共享元结构 | 自由作答数值，无选项捷径 |
| skillopt 效果 | mean **0.5629**（4-seed），是当时最强 baseline 之一 | 0.5074，held-out **−0.167 过拟合**（val+0.166→test−0.167）|
| evolver 效果 | evo_seed 9/16 seed 超 skillopt，seed12/3 达 0.5912 | Δ=+0.004，best_skill==S0（0 promoted）|
| S₀ 强度 | 已饱和（other_60 四 skill 持平，路由增益≈0）| 强基线但非饱和（baseline 0.496，理论上限 +0.504）|

- 我们已诊断出 skillopt 在 LiveMath 有效 ⟺ 两个条件同时成立：(a) 任务有共享元结构（MCQ「选项 vs 定理」）→ 产出可复用的结构化失败模式；(b) S₀ 尚未覆盖该元结构。
- AIME **两个条件都不满足**：自由作答无 MCQ 元结构 + S₀ 已覆盖形态级。
- **疑惑**：这个「MCQ 元结构是 skillopt/evolver 生效的必要条件」判断是否成立？如果是，是否意味着**纯 skill 文本进化在无共享元结构的自由作答任务上本质受限**？这会把问题从「工程 bug」上升到「方法适用边界」。reviewer 是否认同这个边界判断？还是说 AIME 上仍存在某种「隐含元结构」只是我们没找到（如「几何辅助线范式」「数论模算术套路」），算法应能自动发现它？

### Q2. S₀ 的设置问题：3 个变体怎么来的？哪个有效？
**现状**：项目中存在 3 个 AIME S₀ 变体，效果差异显著但来源与选择逻辑未系统梳理：

| 变体 | 文件 | 大小 | hash(strip) | 来源 | baseline pass@k (AIME24, k=3) |
|------|------|------|-------------|------|------------------------------|
| S₀(v1) | `skill_aime_initial.md` | 2239B | d26b84bdbf | 08-04 从头构建，4 段通用策略（read-twice/重述量/选表示/枚举验证），**刻意不含 failure-mode** | 0.678（train+val 30 题，+11.1pp vs no-skill，CI+0.033 **显著**）|
| S₀(v2) | `skill_aime_initial_v2.md` | 600B | **c11d7dc0c8** | 09-30 精简版，4 段压缩为 bullet | run#1=0.544 / run#2=0.589 / run#3=0.600（**同 hash 跨 3 run 差 5.6pp!**）|
| S₀(v2_strong) | `skill_aime_initial_v2_strong.md` | 2409B | f968273257 | v1 的 detail-rich 版，含「几何建系/数论模算术/组合计数」per-domain 具体策略 | 未独立评测 |

- **S₀ 本身有效**：v1 vs no-skill +11.1pp（CI_lower=+0.033，显著）。初始 AIME skill 已比无 skill 好。
- **但 3 次实验全用 v2（600B 精简版）**，而 v1/v2_strong 这两个**更详细的变体从未在 evolver 主线跑过**。
- **硬噪声证据**：**同一 S₀（v2, hash c11d7dc0c8）在 run#1/2/3 的 baseline pass 分别是 0.544/0.589/0.600，跨 3 run 差 5.6pp**。这说明 k=3@0.3 评测下，S₀ baseline 本身有 ±0.03 量级的重测噪声——而 evolver 候选的 delta 也在 ±0.03 量级（§3.5）。**信号和噪声同量级**。
- **疑惑**：
  - (a) 为何 3 次实验都选了 600B 精简 v2 而非 2239B 的 v1 或更详细的 v2_strong？S₀ 的选择是否本身就限制了进化上限（精简 S₀ 缺几何 per-domain 策略 → evolver 从缺几何的起点进化 → §5 几何断点）？
  - (b) v2_strong 已含「几何建系/对称识别」per-domain 策略，**如果用 v2_strong 作 S₀ 跑 evolver，几何失败是否还会被锁死在方程桶？** 这是未验证但高价值的假设——S₀ 是否应预置 domain skeleton 让 evolver 去 refine 而非从零涌现？
  - (c) 同 S₀ 跨 run 差 5.6pp，是否说明 **k=3 评测根本无法支撑 ±0.03 量级的增量判别**？baseline 噪声就吃掉了真实信号。是否必须先固定 S₀ cache（run#3 复用 run#2 的 s0_cache，但 baseline 仍因 probe 抽样不同而 0.589→0.600）？

### Q3. S₀ baseline 噪声（±0.03）与候选 delta（±0.03）同量级，门控如何还能区分信号？
**现状**：EVAL_TEMP=0.3 下 S₀ 重算非确定。k=3 投票降低但不消除噪声。run#3 的 baseline_val=0.6 但 ci=−0.1222（CI 下界远低于点估计）。
- evolver 候选整体 delta 典型 +0.01~+0.04（§3.5 三方对比），而 S₀ baseline 重测抖动 ±0.03。**候选增量可能完全落在 baseline 噪声带内**。
- 当前用 `s0_cache` 冻结 baseline（precompute_s0 一次，后续复用），冻结了 baseline 的不确定性，但候选侧仍 k=3 有噪声。门控 `lcb_score_from_deltas` 用 per-item 配对（候选 pass_rate − S₀ cache pass_rate）部分抵消，但 S₀ cache 本身是单次 precompute 的点估计。
- **疑惑**：这种「冻结 baseline 单点 + 候选 k=3 投票」的协议，统计上是否自洽？更严格的做法是否该是**候选和 baseline 都用相同 k 投票、相同 seed、配对 per-item bootstrap**？当前 0 admit 里有几个是「真实负信号被拒」（正确），有几个是「正信号被 baseline 噪声淹没」（假阴性）？能否从 evolver_log 的 per-item delta 里反推？这直接关系到「0 admit 是真负面还是测量上限」。

### Q4. 后续数据集怎么办？
**现状**：AIME24+25 共 60 题，已被 8 轮实验反复跑过（含 ReAct/CodeAgent/PoT/3 方对比/EFL×3）。约束#1「不扩数据集」在效。
- **疑惑**：
  - (a) AIME 是否已「跑死」——同一 30 题反复跑 + S₀ baseline 噪声 ±0.03，继续在 AIME 上跑是否只是「测量噪声内的随机游走」？但约束#1 禁止扩到新题。
  - (b) 若不扩题，是否换**评测协议**（如 greedy@0.0 vs k-voting@0.3）或换**任务形态切片**（AIME 内按 domain 切，只评几何子集 n=12）来提统计功效？这算「换数据集」吗？
  - (c) 如果 AIME 确实是 skill 进化的不适区（Q1），是否应在约束内找一个「有共享元结构 + S₀ 未饱和」的数据集继续验证方法？现有 LiveMath 已饱和，AIME 无元结构——是否还有第三选择（如带选项的竞赛题、或 CodeForces 题面有结构但答案数值的混合形态）？

### Q5. 算法该怎么优化？针对 §5 断点的具体修法
**现状**：§5 已定位精确断点 = expand 的 focus 选「最高频 failure type」→ 永远选泛化方程桶 → patch 永远泛化。这是可修复的工程问题，非方法本质缺陷。
- **疑惑**：以下 3 种修法，reviewer 认为哪种最可能突破且不违反「不注入」约束#3？
  - **修法 A（focus 重构）**：focus 不选「全局最高频」，改选「某 facet 内最高频未覆盖 + 该 facet 有零分题」。即把频率选择从 facet 间下沉到 facet 内，并耦合零分题信号。风险：仍受 C 分类影响。
  - **修法 B（zero-shot 域诊断）**：对零分题（baseline pass=0）独立跑一次 LLM 细粒度失败分析（不进 EFL 聚类，绕过过吸收），直接作为 patch 的 target_type。零分题是增益空间最大的子集。风险：是否算「用题目结果（零分）注入先验」？我们认为零分是数据标签非分类法注入。
  - **修法 C（patch 后验证）**：expand 生成后，对每个候选 patch 单独在零分题子集 eval（而非全局），用零分题通过率作准入信号。即 group-admit 的更激进版——直接以「零分题集合」为单一 admission group。风险：n=3 零分题统计更弱。
  - **关键判据**：修法 B/C 都用到「零分题」这个数据标签。约束#3 禁的是「人工注入 facet 分类法」，而「零分」是评测产出的客观标签。reviewer 是否认同这条边界——**用客观评测标签（零分/通过）做加权算「数据驱动」，用人工分类法做加权算「注入」**？

### Q6. 这套 B/C 模型组合下，skill 进化的天花板在哪？是否到了换「评测信号」的时机？
**现状**：B=DeepSeek-V3.2（72B 级），C=DeepSeek-V4-Flash。C 做失败分类时把几何失败过分类进泛化桶（§5.1）——这是 C 模型能力的上限。
- **对照**：CodeAgent/PoT（§3.7）显示「写代码算」+14.8pp 显著正向。这暗示对 AIME 几何题，**算力（执行验证）比 prompt 策略（推理步骤）更有效**——但 PoT 超出约束#2（skill-only）。
- **疑惑**：
  - (a) C 模型分类精度（把几何失败归入方程桶）是否是当前 B/C 组合的硬上限？换更强的 C（如 DeepSeek-V4 正式版而非 Flash）能否改善分类、从而解锁 §5 断点？还是说失败分类本身就是开放问题、换模型也难根治？
  - (b) skill 进化的信号源目前是「C 评 pass + C 评失败类型」。如果 C 的失败分类不可靠（§5），是否应把 skill 进化的**奖励信号从「C 失败分类」切到「单测通过率」**（已有 70% 客观权重，但失败诊断仍依赖 C）？即：准入只看 pass 率 delta（已有），失败诊断只用于指导 patch 生成（可降权或弃用 C 分类）。
  - (c) PoT 正向 +14.8pp 是否提示：对数值竞赛题，**上下文空间的有效进化方向不是「优化推理 prompt」而是「优化何时调用代码执行」**？这仍在上下文空间（不训练参数），但把进化目标从「推理策略」转向「执行策略」。这是否算仍满足约束#2（skill 进化）的边界扩展？

---

## 8. 复现

> ⚠️ 所有 `.py` 中硬编码 API key 已替换为 `os.environ.get("QWEN_CHAT_API_KEY", "")` 占位。运行前需 `export QWEN_CHAT_API_KEY=<授权 key>`（仅 DeepSeek/Qwen3-235B 授权）。完整复现需 skillopt 环境（§2.2）。

```bash
# 环境
source lunwen/bin/activate
export QWEN_CHAT_API_KEY=<your-authorized-key>

# run#3 复现（facet-tree + group-admit）
cd rl/train/verl1/skill_evolution
python skill_evolver_aime_v2.py \
  --emergent_facets --facet_tree --expand \
  --facet_group_admit --subset_admit \
  --iters 10 --eval_k 3 --K 4 --train_n 18 \
  --output_dir results/run_GROUPADMIT_<ts> \
  --s0_cache <reuse run#2 s0_cache.json>   # 复用基线保证 byte-identical

# 多 seed 复测
python eval_test_multiseed.py
python eval_aime24_multiseed.py
python eval_aime25_multiseed.py
```

run#3 的 `s0_cache.json` 复用了 run#2 的基线（30 题，相同 split），保证 S₀ byte-identical。

---

## 9. 与旧版本（archive/）的差异

旧版（2026-09 系列，见 `archive/`）是 **LiveMath** 主线，聚焦 meta-shortcut（规则猜测 +9pp）与 facet 库早期验证。本包是全新 **AIME** 线：
- 数据集从 LiveMath 换成 AIME-2024/2025
- 新增 facet-tree 进化路径 + group-admit 门控 + EFL 涌现 taxonomy
- 新增 3 次完整 run 产物（run#1/#2/#3）+ 完整 AIME 尝试时间线（§3）
- 聚焦负面结果的精确断点定位（§5）

旧版所有 zip、review 文档、LiveMath 代码已归档到 `archive/`，互不干扰。

---

## 10. 给 reviewer 的建议阅读顺序

1. 本 README §0（TL;DR）+ §3（AIME 全时间线）+ §5（精确断点）+ §7（6 疑惑点）
2. `code/skill_evolver_aime_v2.py` — 进化器（重点看 §2.2 依赖说明 + `classify_item_group`/`group_signal`/`group_promote_decision`/`_combined_rescue`/ expand focus 选择逻辑）
3. `results/run3_group_admit_20261014/evolver_log.json` — 逐轮证据（target_type / group_signal / branch_failure_types）
4. `docs/facet_tree_skill_evolution_design.md` — facet-tree 设计原意
5. `archive/` — 旧 LiveMath 线对照（可选）

**最想请 reviewer 回答的 6 个问题**（§7）：
- **Q1**（机制根因）skillopt 在 LiveMath 有效但 AIME 失效，根因是数据集（无 MCQ 元结构）还是机制？是否意味 skill 进化在自由作答任务上本质受限？
- **Q2**（S₀ 设置）3 个 S₀ 变体怎么来的、效果如何？为何 3 次实验都用 600B 精简 v2 而非含几何策略的 v1/v2_strong？同 hash S₀ 跨 run 差 5.6pp 的 baseline 噪声怎么处理？
- **Q3**（统计功效）baseline 噪声 ±0.03 与候选 delta ±0.03 同量级，门控如何区分信号？0 admit 是真负面还是测量上限？
- **Q4**（数据集）AIME 是否已跑死？不扩题的前提下换协议/切 domain 子集是否可行？
- **Q5**（算法优化）针对 §5 断点的 3 种修法（focus 重构 / zero-shot 域诊断 / patch 后验证），哪种最可能突破？用「零分题」客观标签做加权算数据驱动还是注入？
- **Q6**（天花板）C 模型分类精度是否硬上限？是否该把奖励信号从「C 失败分类」切到「单测通过率」？PoT +14.8pp 是否暗示进化方向应转向「执行策略」？
