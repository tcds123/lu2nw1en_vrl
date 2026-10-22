---
name: facet-tree-capability-ceiling-20260827
description: facet-tree算子能力边界诊断(代码级铁证). 8-seed把gap坐实(-0.0291显著落后)后,诊断门合法打开. 实查skill_evolver_math.py: facet-tree线核心算子facet_edit(FACET_EDIT_PROMPT L759-773)明令"ONLY edit ONE named section"+"extend/replace at most one bullet"+facet_edit_guard物理强制非目标slot md5不变→结构上只能单slot内精炼,禁止新增section. 8238b5e8的826 run(run_20260826_162059)实测crossover_fired=1且overlay两iter均admitted(delta0.3268/0.3239),但became_best=false、best_val_delta全程冻结0.3394(Option Comparison单轴best)、cumulative_diag.best_ratio=1.0→算子触发但sub-additive未超单轴best; 另一场824 run(出1dbbb821)crossover_fired=0、overlay未启用(args无overlay key). 代码天花板: overlay=叠加已有best段不新增section; crossover=2-parent段互换不新增section. 故skillopt多出的6个section(Meta-Option Decision Procedure/Strategy/Domain-Specific Heuristics/Strength Discrimination等,含5步决策程序+6领域特化)落在facet-tree算子能力边界外. 这是正向代码推导非gap跨0倒推.
metadata:
  type: project
---

# facet-tree 算子能力边界诊断 (2026-08-27)

## 触发与合法性
- 8-seed 同窗口重测把 gap 坐实: evolver 显著落后 skillopt -0.0291 (两判据CI上界<0). 见 [[h2h-8seed-final-result-20260827]].
- gap 坐实后, "为什么输"的诊断门**合法打开** (此前 gap 跨0时下诊断=倒推, 已撤, 见 [[h2h-gap-diagnosis-20260827]] 三条校准).
- 本诊断从代码能力边界正向推导, 非从统计不成立的 gap 反推.

## 代码级铁证: facet-tree 算子只能单 slot 精炼, 结构上禁止展开
实查 `rl/train/verl1/skill_evolver_math.py`:

### 1. 核心算子 facet_edit (L1141-1145 调用, L759-773 prompt)
- `FACET_EDIT_PROMPT` (L759-773) 明令:
  - "ONLY edit the section named `{facet}`" — 只编辑一个命名段
  - "keep every other section byte-for-byte identical" — 其余段逐字不变
  - "SMALL and SURGICAL (extend/replace **at most one bullet**)" — 最多改一个 bullet
- `facet_edit_guard` (L776): 物理校验非目标 slot md5 不变, 否则拒候选 (护栏)
- → **每次变异 = 单 slot 内最多改一个 bullet, 物理禁止跨 slot、禁止新增 section**

### 2. 8238b5e8 血统两场 run 的实际算子记录 (evolver_log.json, 2026-08-31 复核修正)
**校正前错误记录**: 曾写"两场 run 全程 crossover=None/overlay=None 未触发"——经逐字段复核 evolver_log.json 为**误记**, 两场实况不同:
- run_20260824_153302 (出 1dbbb821c3): `crossover_fired=0`, **overlay 未启用**(args 无 overlay key); iter0 `crossover=None`; iter1 crossover 选了 parents(Hypotheses, Option Comparison) 但 `admitted=false, reason=guard/cohere-fail`
- run_20260826_162059 (出 8238b5e8): `crossover_fired=1`, **overlay 启用并两 iter 均 admitted**:
  - iter0: overlay `admitted=true, delta=0.3268, ci=0.2535, became_best=false, n_axes_combined=3`
  - iter1: crossover `admitted=true, delta=0.3099, ci=0.2352` **且** overlay `admitted=true, delta=0.3239, became_best=false`
  - **best_val_delta 全程冻结 0.3394 (=Option Comparison 单轴 best), cumulative_diag.best_ratio=1.0**
- → **校正后结论更强**: 8238b5e8 的算子**真实触发并 admitted** (非"没触发"), 但**sub-additive**——叠加/重组后的 delta(0.327/0.310/0.324) 始终 ≤ 单轴 best(0.3394), `became_best` 恒 false, best_ratio 恒 1.0. 即"算子有能力叠加, 但叠加不超越单轴最强". 另一场 1dbbb821 才是 overlay 没启用、crossover guard 失败.

### 3. 即便触发的天花板: 仍不新增 section
- overlay (L1273): 把每轴历史 best 段叠加成一个 skill, **叠加已有段, 不生成新 section**
  - 历史: facet-tree-overlay-fix-20260825 live 跑过, best_ratio恒1.0/n_axes_combined恒1, 未超越单轴best
- crossover (L1217): 2-parent facet 段互换, **在 4 段内重组, 不新增 section**
- → **整个 facet-tree 线没有"新增 section + 领域特化"的展开算子**

## 结构缺口对照 (skillopt 多出的内容落在能力边界外)
- evolver 8238b5e8 = 4 段 / 1986 字 (S0 4段 1018字 + 各slot精炼一bullet, 净增~970字, 全在原4段内)
- skillopt cfae156d = 10 段 / 9822 字, 比 S0 多 6 个 section:
  1. Meta-Option Decision Procedure (完整5步: identify→validity→maximality→precision→doubt)
  2. Meta-Option Strategy for 'Strongest Provable'
  3. Strength Discrimination (s_j equality/inequality, stably unbounded 等数学示例)
  4. Domain-Specific Heuristics (6个: PDE正则性/模型论/群作用/图论/集合论/capacity)
  5. Domain and Range Precision
  6. Final Decision Rule
- 这 6 段含"展开型内容"(决策程序步骤 + 领域特化映射 + 数学示例), 恰是 facet_edit "at most one bullet / 禁新增section" 能力边界**之外**的东西

## 诚实标注的反驳点 (2026-08-31 复核修正)
- **旧前提已废**: 原写"overlay/crossover 没触发"作让步——经复核 826 run 算子**确实触发且 admitted**, 该让步不成立, 已改.
- **真正稳健的论据有两条 (都不依赖"没触发")**:
  1. 即便触发, 826 run 实测叠加/重组 delta(≤0.327) **未超单轴 best(0.3394)**, became_best 恒 false, best_ratio 恒 1.0 → 算子有叠加能力但 **sub-additive**, 不产生净增益
  2. 代码天花板: overlay 叠加已有 best 段 / crossover 4 段内重组, **两者都不新增 section**
- 故结论稳健且更强: 算子真的跑了仍未超单轴 best, 且天花板不新增 section → 真正的展开能力(新增section+领域特化)整个 facet-tree 线无对应算子

## 优化方向 (待选, 不启动)
- **方向1: 加展开型 mutation 算子** — 给定诊断失败模式, 让 LLM 生成新 section (决策程序步骤/领域示例), 突破"单slot单bullet"约束. 改 evolver 算子代码 + facet_edit_guard 放宽新增section路径. 需 API 跑 Step 验证.
- **方向2: 起点/算子对齐对照** — 实验一(旧evolver,V2起-8.4pp) vs 实验三(facet-tree,真S0起-2.9pp)混了算法代际+起点段数两变量. 同V2起跑一臂facet-tree, 或同真S0起跑一臂旧evolver, 干净归因facet-tree算法本身贡献. 论文层面补测.
- ⚠️ 不重跑判别实验: "种skillopt重跑判别"=实验二已做过(循环parity), 见 [[evolver-lineage-three-experiments-20260827]], 勿重跑白烧API.

## 关联
[[h2h-8seed-final-result-20260827]] (8-seed gap坐实, 本诊断合法性前提) / [[h2h-gap-diagnosis-20260827]] (诊断+三条校准, 其「信息密度根因」现被代码级坐实为能力边界) / [[evolver-lineage-three-experiments-20260827]] (三场血统) / [[facet-tree-overlay-fix-20260825]] (overlay live 跑过未超越) / [[facet-tree-next-experiments-20260826]] (下一步实验双线)
