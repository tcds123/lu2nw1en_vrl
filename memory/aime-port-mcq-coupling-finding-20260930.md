---
name: aime-port-mcq-coupling-finding-20260930
description: 移植 math evolver 到 AIME 的关键障碍——新机制(facet_tree/expand/pool_filter/strengthen/subset_admit)深度耦合 LiveMath MCQ 结构(Step6已证=捷径),非换评测后端可解。移植需重设计 taxonomy/classifier/peaks/axes。旧AIME evolver(M1-M4,AIME-native taxonomy)已存在且08-07已阴性。2026-09-30
metadata:
  type: project
---

# AIME 移植关键发现：新机制深度耦合 MCQ (2026-09-30)

承接 [[dataset-pivot-aime-onoff-20260923]]（计划"以 skill_evolver_math.py 为基底换 AIME 评测后端，~2-3h"）与 [[step6-answer-form-verdict-20260929]]（ON 增益=MCQ 标签捷径）。本记录推翻"换后端即可"假设——实测发现新机制本身就是 MCQ 耦合的。

## 核心发现：新机制围绕 MCQ 结构工程化

`for_review/code/skill_evolver_math.py` (4017 行) 的**区分性新机制**（facet_tree/expand/pool_filter/strengthen/subset_admit）几乎每一处都锚定 LiveMath 的 theorem-level-MCQ 结构——正是 Step6 已证为捷径的那个结构：

| 耦合点 | 行 | MCQ 依赖 |
|---|---|---|
| `classify_failure_type` | L162 | `_META_Q_SIG`=("strongest","can be proved",...) / `_META_OPT_SIG`=("one of the remaining",) = LiveMath 定理级 MCQ 签名；93% 难题回退 `FAILURE_FOCI[5]`(meta) |
| `classify_item_meta` | L204 | 读 `item["choices"]`+`item["theorem"]` → AIME 无此字段 |
| `classify_item_dim` | L235 | 同上，尖峰关专用 |
| `FAILURE_FOCI` | L94-97 | 8 foci 全定理级 MCQ 内容：iff/existence/quantifier/regularity/sharp-constant/domain/divisor — **非 AIME 数值失败类型** |
| `_FIVE_PEAKS` | L255 | pool_filter 尖峰关 5 峰 = implication/existence/quantifier/sparse/meta — 全 MCQ 维度 |
| `--subset_admit` (break-③) | L3482 | `has_meta` 切 meta/not-meta 子集 — AIME 无 meta 概念 |
| `--data_driven_focus` | L3358 | 根因"93% meta-option 失败" — LiveMath 特性 |
| `--facet_axes` 默认 | L3394 | "Option Comparison,Theorem-Level Precision,Hypotheses,Final Answer" — Option Comparison/Theorem-Level = MCQ 概念 |
| expand "skip meta" | L488 | `find_uncovered_type_for_expand` 显式跳 `FAILURE_FOCI[5]`(meta 由 facet_edit/Option 轴负责) |
| `strengthen` 算子H | L3439+ | 后移 meta 检查让 existence/uniqueness 可见 — 全程在 MCQ 失败类型体系内 |

## 含义（对论文叙事 + 实验设计）

1. **"最新算法"的区分性特征恰恰是 MCQ 耦合的**。facet_tree/expand/pool_filter 不是领域无关的通用优化器——它们是针对 LiveMath meta-option 结构（捷径源）特化的。Step6 证了**产物**（skill）利用 MCQ 标签；本发现补证**过程**（优化器机制）也锚定 MCQ 结构。二证收敛：捷径不是偶然副产品，是机制设计目标。
2. **移植到 AIME ≠ 换评测后端**。换 `extract_int_answer`/`evaluate_aime` 只动最底层；上层 taxonomy/classifier/peaks/axes/subset 全需重设计为 AIME 数值版。原计划"~2-3h"严重低估。
3. **唯一真正未测的实验**：facet_tree/expand/pool_filter **叠加在 AIME-native taxonomy 上**。旧 `skill_evolver_aime.py`(410 行) 已有 M1-M4 + AIME-native `FAILURE_FOCI_AIME`(8 数值类型：misread-quantity/case-enumeration/off-by-one/setup/answer-form/inclusion-exclusion/number-theory/arithmetic)，且 08-07 已测 = 阴性（v2 Δ=0.00pp, evolver Δ=−1.67pp，见 [[aime-fresh-rerun-result-20260807]]）。未测的仅是"新机制 × AIME taxonomy"组合。

## 移植设计岔路（待用户决策）

| Fork | 做法 | 测什么 | 预测 | 工作量 |
|---|---|---|---|---|
| **A 忠实移植** | facet_tree/expand/pool_filter 照搬，MCQ 专属部分降级 no-op（Option Comparison 轴删/has_meta→False/5峰塌缩为 AIME 数值峰用 FAILURE_FOCI_AIME） | 剥掉 MCQ 机制后算法能否超 s0 | **阴性**（与 Step6 一致：增益=捷径，剥捷径即塌） | 中（stub 化 MCQ 部分） |
| **B 重设计移植** | facet 轴+taxonomy 重设计为 AIME-native（8 数值 foci 作轴/峰），机制照搬 | 算法在 taxonomy 匹配领域时是否有效 | 可能阳性（但混淆"算法有效"与"手工好 taxonomy"） | 大（重设计 taxonomy/classifier/peaks） |
| **C 两者** | A 主+B 副 | 同时测捷径假设 + 领域匹配假设 | — | 最大 |

**Why:** Fork A 直接检验 Step6 的过程级推论（捷径不仅在产物，在机制）；Fork B 测算法通用性但需手工设计 taxonomy（公平性存疑：阳性可能归因 taxonomy 而非算法）。

## skillopt 对比的同构问题

skillopt 同样 MCQ 耦合：唯一数学 env = livemathmaticianbench（MCQ label 匹配 `parse_choice_label`），蒸馏模板 theorem-level-MCQ 风味，**无 AIME env**。故 skillopt-on-AIME 亦需写 AIME env adapter（Path A，重）让其 trainer 在 AIME-native 后端上跑；直接评测 skillopt 现有 LiveMath skill（Path B，轻）不公平——skillopt 从未在 AIME 训练，仅测迁移失败，非能力对比。两算法要公平对比，**都需 AIME-native 评测后端 + AIME-native 失败 taxonomy**。

## 修正/细化（两 Explore agent 返回后，2026-09-30）

上表强调"耦合点"但**未给反例**，易误读为"整机制须重写"。两 agent 返回后细化：**机制结构 schema-agnostic 可复用，只有喂给门控的"内容"MCQ 耦合需替换**。

### math evolver agent 的关键修正
- 进化器只是**编排层**：无模型调用/答案抽取/判分（全在导入的 `tf_global_v2_math` + `skillopt.envs.livemathematicianbench`）。
- **可直接复用（schema-agnostic）**：facet-tree GA(`_run_facet_tree`/`build_archive_tree`/`partition_diag`)、facet 分解组合(`facet_decompose`/`facet_compose`)、expand/strengthen 算子(`find_uncovered_type`/`_run_strengthen`/`section_expansion_guard`)、pool_filter 门控**结构**(`admission_filter`/`subset_signal`/`subset_promote_decision`/`spike_peak_hit`/`git_diff_similarity`)、蒸馏循环(`collect_failures`/`generate_candidates`/`consolidate`)、交叉(`recombine_facets`/`_overlay_assemble`)、选择/CI。
- **必须替换（MCQ-coupled 内容）**：`classify_failure_type`/`classify_item_meta`/`classify_item_dim`、`FAILURE_FOCI` taxonomy、`_FIVE_PEAKS` 峰定义、`--facet_axes` 默认、`has_meta` 子集切分、诊断渲染("Choices:/Theorem:")、`PATCHER_SYSTEM`/`PATCH_TEMPLATE` prompt、导入的评测后端。
- ⚠️ 选项位置 shuffle **不存在**于 math evolver（先前计划提到的 `--shuffle_seed` 无此 flag；实际是 `--probe_seed`，shuffle 的是 diag-items/交叉池，非选项位置）。AIME 本无选项，on/off 在 AIME 退化为 pool_filter/facet_tree/expand 的 ON/OFF。

### skillopt agent 的 GREEN 判决
- skillopt trainer(`ReflACTTrainer`)+reflect 引擎+通用 prompts **零 MCQ 偏置**，全 MCQ 逻辑锁在 `livemathematicianbench/` 目录。
- trainer 只读 rollout 结果的 `hard`/`soft`(+可选 `task_type`/`fail_reason`/`id`)——**原始 item schema 完全由 env 自定义**。
- 加 AIME adapter = ~9 文件新建 + 1 行注册，复用 `eval_aime24_multiseed.py` 的 `extract_int_answer`/`evaluate_aime`。**skillopt 会公平蒸馏非 MCQ 数值 skill**。
- 风险(操作非架构)：30 题太小(split 6/3/21，val=3 门控方差大)→用 7:1:2 或预制 split；`conversation.json`+prompt 文件**必须**写否则 reflect 返 None→skill 不进化(最易静默失败)；trainer 内单采样、最终 skill 用 `eval_aime24_multiseed.py` k-vote 复测。

## 决策（据用户"试我们算法+AIME多数据集+skillopt对比"指令）

- **Fork B（重设计移植）**：以现有 `skill_evolver_aime.py`(410行,AIME-native,含 `FAILURE_FOCI_AIME` 8 数值类型)为基底，叠加 math evolver 的**机制结构**(facet_tree/expand/pool_filter GA)，把 MCQ-coupled 内容替换为 AIME-native（8 数值 foci 作 facet 轴/峰、classify_* 改读 `item["problem"]`/`item["answer"]`、诊断渲染改 "Answer:/pred N/correct N"）。Fork A(stub MCQ→no-op)仅复确 Step6 阴性、无需 skillopt 对比——与用户"试算法表现+对比"意图不符，弃。
- **skillopt Path A（写 AIME env adapter + 跑其 trainer）**：唯一公平对比。Path B(直接评测 skillopt 现有 LiveMath skill on AIME)=迁移失败测试非能力对比，弃。
- **数据集**：AIME24 + AIME25（用户"不同 AIME 数据集"=复数）。

## 构建图（待执行）

| 产物 | 基底 | 新/改 |
|---|---|---|
| `skill_evolver_aime_v2.py` | `skill_evolver_aime.py` | 加 facet_tree/expand/pool_filter 结构；taxonomy/axes/peaks/classify/diagnostics AIME-native |
| `tf_global_v2_aime.py` | 已存在(AIME-native) | 复用 run_one/eval_on/precompute_s0 |
| skillopt `envs/aime/` (adapter/dataloader/rollout/evaluator/prompts/skills) | 复制 livemath 结构 | MCQ→数值 |
| `scripts/train.py` | +1 行注册 `aime` | |
| `configs/aime/default.yaml` | 复制 livemath | 去 choices/theorem，改数据路径 |

## 下一步

构建(无 API 成本) → smoke(低成本) → smoke 结果作自然检查点 → 才启动昂贵全量跑。迁移不可在决策前启动——Fork 决定产物语义。

关联：[[dataset-pivot-aime-onoff-20260923]]、[[step6-answer-form-verdict-20260929]]、[[aime-fresh-rerun-result-20260807]]、[[aime-vs-livemath-content-diagnosis-20260805]]、[[onoff-shuffled-final-verdict-20260923]]。
