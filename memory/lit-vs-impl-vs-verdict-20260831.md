---
name: lit-vs-impl-vs-verdict-20260831
description: 文献机制 vs 已实现 vs 实验终局对照(用户8-31方法论修正). 核心区分: "接口接通"≠"机制成功". DIVE bootstrap=接口接通但异构环境经验上从未形成(每分支每代75-100%失败仍是meta-option, 弱轴吃不到轴专有失败); fitness sharing(PromptBreeder)/多父本merge+Pareto(DarwinX/EvoSkill)=根本未实现. expand算子=触发且val成best但test显著退步0.0700. 三档深度: 仅接口/触发未泛化/未实现. 用户判据: 每个机制须深入尝试探索, 不可接口接通即判证伪. 后续: 方向A(条件化单skill)优先但须per-seed验证, 各机制留待深探.
metadata:
  type: project
---

# 文献机制 vs 已实现 vs 实验终局对照 (2026-08-31)

## 触发: 用户方法论修正
用户指出先前对 facet-tree 机制"已实现且已验证不奏效"的判断**无效**——已实现只是浅尝辄止, 机制根本未真正发生就判了证伪. 原话要点: "目前实现的这些机制根本只是浅尝辄止，没有真正深入，比如 DIVE bootstrap 未真发生——失败 87-93% 全 meta-option，弱轴也吃不到轴专有失败……根本就没有实现和成功，我们每使用一个方法都应该深入尝试和探索".

**核心区分写入记忆**: `接口接通 ≠ 机制成功`. 接口存在 + 被调用 + 记一条 fired 计数, 不等于文献机制描述的现象真的发生了. 本文件逐机制核对"接口是否存在 / 机制是否真正发生 / 实验终局".

## 对照表 (9 篇文献 → 3 档实现深度)

| 文献机制 | 文献来源 | facet-tree 接口 | 机制是否**真正发生** | 实验终局 |
|---|---|---|---|---|
| **DIVE 多种群 bootstrap 异构环境** | DIVE 2608.12486 | ✅ 接口在 (`partition_diag` L1028 切 diag 成不相交子集, L1141-1144 每分支喂自己子集, 还有 L1201-1213 专门验证探针) | ❌ **未真发生** — 见下"铁证" | 异构环境从未形成 → bootstrap 帮/害的前置不成立, 无法判 DIVE 机制本身的终局 |
| PromptBreeder **fitness sharing** (防局部最优) | PromptBreeder 2309.16797 | ❌ 未实现 | ❌ n/a | 从未尝试, val→test 脱节的"无多样性压力打破 val 过拟合"假说从未被机制级检验 |
| DarwinX **多父本 merge 重组 + keep-losers archive 吸收偏差** | DarwinX 2608.07545 | 🟡 部分: archive(keep-losers `_ArchiveNode` L985) ✅; 但**只 2-parent crossover** (`recombine_facets` L913 + binary-tournament L1304), 无多父本 merge | 🟡 archive 真存在但"吸收偏差"未发生 — 见下 | crossover 在 826 run fired+admitted(delta 0.310) 但 **sub-additive 未超单轴 best(0.339)**; best_ratio 恒 1.0 |
| EvoSkill **Pareto frontier 竞争选择** | EvoSkill 2603.02766 | ❌ 未实现 | ❌ n/a | 从未尝试 |
| DIVE **2-parent skill 重组** (crossover) | DIVE 2608.12486 | ✅ 接口在 (L1302-1354) | ✅ 真触发 (`crossover_fired=1` 826 run, iter1 admitted delta 0.310) | sub-additive: 叠加/重组 delta ≤ 单轴 best, became_best 恒 false (见 [[facet-tree-capability-ceiling-20260827]] 已修正) |
| **section 展开算子** (新增 section, 非文献直接对应, 破 facet_edit 单 slot 边界) | — (自研, 对位 skillopt 多段) | ✅ 接口在 (`--expand` L1402+, `section_expansion_guard` L831) | ✅ 真触发 (2 场 run, iter1 长出第5段 admitted+一度成best) | 🟡 val 验收#1达成(机制可工作) 但 **test 显著退步 -0.0700**(CI[-0.1077,-0.0330]), 验收#2/#3 未达 — 触发但未泛化 |
| EvoPrompt 真 GA (种群+交叉+DE) | EvoPrompt 2309.08532 | 🟡 部分: 有变异无真种群 | — | 单 lineage 起步, 未达种群规模 |
| SePO archive+softmax 父本采样 | SePO 2606.04465 | 🟡 archive 有, softmax 采样无 | — | archive 已超 SePO 阶石, 但采样策略未上 |
| Verilog-Evolve validation gate | Verilog-Evolve 2605.26498 | ✅ ci>0 gate (`bootstrap_ci_lower` 准入) | ✅ 真工作 | gate 真过滤候选, 但未破 test 天花板 |

## 三档实现深度 (本记忆的核心归类)

1. **仅接口接通, 机制未真正发生** — DIVE bootstrap 异构环境. 接口切了不相交子集, 但"异构"从未经验上成立. **不能据此判 DIVE 机制终局**, 这是先前错误的方法论.
2. **触发但未泛化** — expand 算子 / crossover. 算子真的跑了且有时 admitted, 但 test 上 sub-additive 或显著退步. 机制层面"发生"了, 终局是 val→test 脱节.
3. **根本未实现** — fitness sharing / 多父本 merge / Pareto frontier. 连接口都没有, 谈不上判终局.

## 铁证: DIVE bootstrap 异构环境经验上从未形成
数据源: 8238b5e8 血统两场 run 的 `evolver_log.json` → `branch_failure_types` (evolver 内置的 DIVE 验证探针 L1201-1213 落盘, 零新 API, 2026-08-31 复核):

```
run_20260826_153302 (→1dbbb821, bootstrap ON):
  iter1/iter2 每分支每代: Option 100% meta, Theorem 100% meta, Hypotheses 100% meta
run_20260826_162059 (→8238b5e8, bootstrap ON + overlay):
  iter1: Option 6/7=86% meta, Theorem 2/2=100% meta, Hypotheses 3/4=75% meta
  iter2: Option 6/7=86% meta, Theorem 2/2=100% meta, Hypotheses 2/3=67% meta
```

**判决**: `partition_diag` 把 diag_items 切成了不相交子集(接口成功), 但各分支诊断出的失败**全部收敛到同一个 `Meta-options ('a stronger result can be proven')` 主导模式** — 弱轴(Theorem/Hypotheses)喂进的子集诊断出来仍是 meta-option, **轴专有失败 ≈ 0**. evolver 代码自己的注释(L1202-1203)就预言了这一死法: "若弱轴喂进的 12 题诊断出来 8/8 仍是 meta-option, 则 bootstrap 对该轴是'无效环境'……bootstrap 帮/害的前置不成立". 实测正是如此.

**这不是"DIVE 被证伪", 而是"DIVE 的前置条件(异构失败环境)在本数据集上未满足, 该机制从未真正跑起来"**. 要真正试 DIVE, 须先让弱轴吃到轴专有失败(可能需重平衡 diag 子集 / 按 theorem_type 分层切分而非随机切分 / 扩大 diag 池), 再谈机制终局.

## 用户后续思路 (写回记忆, 须遵守)
- **每个机制都须深入尝试和探索, 禁止"接口接通→跑一次→判证伪"**: 先前把 DIVE bootstrap 当"已实现且证伪"是错的, 它只是"接口接通, 机制未发生".
- **方向 A (条件化单 skill, 内化 has_meta router 的 not-meta fallback 进 8238b5e8) 仍为优先项** — 机制真实(not-meta 子集 expand 跨 4 seed 全正), 但**必须 per-seed 层级验证**(非 per-qid bootstrap), 否则重蹈 val→test 脱节. 见 [[has-meta-router-recompute-20260831]].
- **未实现机制(fitness sharing / 多父本 merge / Pareto)不应急着全上, 但须各自深探一次**: fitness sharing 直接对位 DarwinX §5.1 诊断的"单 lineage winner-takes-all 让 in-loop 评估偏差固化" — 这是 val→test 脱节最直接的文献级假说, 至今未机制级检验过, 值得单独深探.
- **val→test 脱节是贯穿 Task3/Task4/PhaseC/expand 的同一病根**: 不是某一个算子的问题, 是"在 val 局部最优附近搜索 + 无多样性压力 + 无 held-out gate"的结构病. 单点修算子治标.

## 待办判据 (深探 DIVE 的前提条件, 不立即烧 API)
让 DIVE 异构环境真发生, 须先满足: 弱轴子集诊断出的轴专有失败占比 > 某 threshold (如 30%), 而非当前的 0%. 可零成本先在现有 `branch_failure_types` 落盘数据上试: (a) 按 theorem_type 分层切分 diag 子集 (非随机); (b) 扩大 diag 池(n_diag 8→16+) 看轴专有失败是否浮现. 满足前置后, DIVE 机制终局才有意义.

## 关联
[[facet-tree-capability-ceiling-20260827]] (能力边界诊断, 算子触发记录已 8-31 修正) / [[has-meta-router-recompute-20260831]] (方向 A 唯一正向信号) / [[expand-operator-api-step-result-20260828]] (expand 触发但未泛化) / [[eval-signal-debiasing-diagnosis-20260820]] (93% meta-option 是数据结构属性) / [[probe-split-result-20260820]] (多机制全收敛~0.36-0.47 天花板) / docs/ea-skill-evolution-survey-20260819.md (9 篇文献机制源) / docs/facet_tree_empirical_verdicts_20260831.md (8-31 两判决合成)
