---
name: design-drift-audit-20261127
description: 原始设计一致性审计 — 原始设计(2026-08-25)要求emergence自动涌现+结构生长, 当前实现emergence默认OFF(stage0)+expand跳过meta致结构锁死; 6个机制偏移逐个核实(部分待fork agent补全)
metadata:
  type: project
---

# 原始设计一致性审计 (2026-11-27, 零API)

用户判断: "Run E 不是新研究方向, 而是恢复原始算法设计意图. 实现偏移来自多个局部优化叠加." 本审计核实此判断.

## 原始设计意图 (docs/facet_tree_skill_evolution_design.md, 2026-08-25)

设计文档§2.2(b) 明确: **facet 轴自动涌现** — "聚出的失败簇 = 潜在新 facet 候选: 若某簇跨多题复现且现有 facet 未覆盖 → **自动开新分支**(adaptive 分支数,非固定)" + "长期无 delta 贡献的 facet 分支 → **自动合并/淘汰**".

设计文档§五: `facet_edit_guard` 护栏的初衷是**防塌缩回单目标**(4分支同fitness会塌缩), 是**搜索差异化护栏**, 非结构约束.

**设计作者自己的诚实标注**(L5-6, L61): "Stage 0 实际跑的是 `--no_emergence`, 涌现机制 (§2.2(b)) 尚未兑现, 当前是预设, 自动涌现仍是设计目标, 非已实现."

## emergence 偏移 (已核实, 核心偏离)

| 项 | 内容 |
|----|------|
| 原始设计 | §2.2(b) facet 轴自动涌现, 从失败模式聚类开新 section 分支, 分支数自适应 |
| 当前实现 | `--no_emergence` 默认 False, 但 emergence 仍 "OFF(stage0)" (L2970: `emergence={'OFF(disabled)' if args.no_emergence else 'OFF(stage0)'}`). **即使不开 --no_emergence, emergence 也关**. facet_axes 默认硬编码4段 (L3961). |
| 偏离时点 | 2026-08-31 `docs/facet_tree_empirical_verdicts_20260831.md` 判决1: **theorem_type 聚类 = 结构性空结果**(91%题多type, 12 hard-fail=11不重复combo, K=3门控永不触发). 用户拍板写成"实证证伪证据". |
| 偏离原因 | **theorem_type 这一种聚类键被证伪** → 退回预设4段 + emergence OFF. 但证伪的是聚类方法, **不是"从失败模式自动开新section"这个目标**. expand/overlay/crossover 在9月仍实现了结构生长(非emergence路径), 说明目标可达, 只是聚类键不对. |
| 是否仍在代码 | 是 (emergence OFF(stage0) 是默认, facet_axes 硬编码4段) |

**这是最核心的偏移**: 为修 emergence 本体不可行, 退回预设4段, 但**把"结构生长"目标也一并放弃了**. 然而9月 expand-v2 证明结构生长可通过覆盖驱动 expand 实现(非emergence), 11月却把这条路径也限死了(见下).

## expand 跳过 meta 偏移 (已核实, 一次负向禁整类)

| 项 | 内容 |
|----|------|
| 原始设计 | 无"expand 跳过 meta" 限制. §2.2(b) 要求从失败模式自动开分支, meta 是主失败模式(87-98%)应优先. |
| 当前实现 | `find_uncovered_type_for_expand` (L747-768) L763 显式 `if lbl==meta_lbl: continue` — expand 永不攻 meta. |
| 偏离时点 | 2026-09-01 `memory/expand-v2-design-20260901.md`: v1 expand 攻 meta 致 test −0.07 退步 → v2 修复 = expand 显式跳过 meta, 只攻 not-meta. |
| 偏离原因 | v1 expand 走裸 hardest-8(~100% meta) 与已饱和 Option 轴内容重叠 → −0.07. v2 改覆盖驱动 + meta-skip. |
| 是否仍在代码 | 是 (L763) |

**关键矛盾(用户指出的"一次负向禁整类")**: v2 meta-skip 在9月合理(当时 expand 攻 not-meta Quantifier 能 +0.211 进 best, L79-89 iter3 突破). 但**9月成功依赖 not-meta 失败有足够占比**. 11月 LiveMath 失败 98% meta + S₀ not_meta pass 33%→46%(not-meta 失败变少) → expand 跳过 meta = expand 无 target = 饿死. **代码没有在新失败分布下重新审查这个限制**. 已有 paired rescore + CI + veto 保护, 早期的硬编码排除是否仍必要本应重新审查(用户原话).

**文档自身记录的未修复局限** (expand-v2-design L100-103): "per-axis coverage targeting 未接入 facet_edit, 根轴 facet_edit 仍全 meta-saturated". 这个局限从未修复 → facet_edit 攻 meta 但只能改固定段, expand 不攻 meta → **meta 这个最大 headroom 无任何结构扩展路径**.

## overlay 偏移 (待 fork agent 补全 parent/组合逻辑细节)

| 项 | 内容 |
|----|------|
| 原始设计 | §2.5 2-parent facet 组合交叉, overlay 跨轴叠加 best (§五算子E). 9月 run_20260904 overlay 是结构增长主力(delta +0.32). |
| 当前实现 | `_overlay_assemble` (L2567) + overlay_slots=dict(root_slots)(L2585). Run A overlay 6次 0晋升. |
| 偏离原因 | 待 fork agent 核实 overlay parent 来源 + paired_rescore 对 overlay 竞争力的影响. 9月 vs 11月: 11月加 paired_rescore (overlay vs parent best_skill 配对), 可能改变 overlay 相对竞争力. |

## leaf 与 global best 状态管理 (叶隔离, 待补全)

| 项 | 内容 |
|----|------|
| 原始设计 | §2.3 树状 archive, "保留输家/specialist", 节点存 skill+facet标签+lineage+delta+已解侧面集. |
| 当前实现 | 每 leaf 独立 slots, leaf.slots 初始化全=S₀ (L2900 dict(root_slots)). A-axis 晋升不传播到 B-axis leaf → 跨轴回退. |
| 偏离原因 | 待 fork agent 核实 leaf.slots 何时初始化为全S₀ (而非继承best). Run D global_inherit 部分缓解, 但不解决根本. |

## 新增 section 生命周期 (待补全)

| 项 | 内容 |
|----|------|
| 原始设计 | 新 section 应能被继承、继续优化、评估后删除 (动态 slot registry). |
| 当前实现 | expand 饿死(meta-skip) → 新 section 几乎不产生. 即使产生, leaf 隔离使其他 leaf 不继承. overlay carry-forward 仅取 best_node.slots 非root段(若 best_node 4段则无段可携). |

## K 与 n_parallel 演变 (待补全)

| 项 | 内容 |
|----|------|
| 原始设计 | §2.4 stance 旋转绑定 facet, `angle_hints`(4角度)/`STRUCTURE_ANGLES`(6 stance). |
| 当前实现 | n_parallel 控制 facet_edit 候选数, K 在 facet_tree 模式被忽略(L4262). _FACET_EDIT_ANGLES 7个角度, n_par=4用angle[0..3]. |

## 9月 vs 11月 配置差异 (已核实)

| 参数 | 9月(0904, 成功) | 11月(Run A, 结构锁死) |
|------|---------------|---------------------|
| no_emergence | False (但仍OFF stage0) | False (仍OFF stage0) |
| expand | True | True |
| expand_every | **2** | **1** |
| expand_candidates | 1 | 1 |
| n_parallel | 1 | 1 |
| paired_rescore | **无** | **True** |
| pool_filter | **无** | **True** |

**emergence 两边都关** → 9月 expand 成功靠的是覆盖驱动(非emergence). 关键差异 = **paired_rescore + pool_filter (11月移植的新机制)** + **失败分布变化**(9月 not-meta 有占比, 11月 98% meta). 这不是单一原因, 是"新机制移植 + 失败分布变化 + v2 meta-skip"三者叠加.

## 修正后的实验定位 (用户拍板)

| 实验 | 之前定位 | 修正后定位 |
|------|---------|-----------|
| Run A | 新机制完整实验 | **当前受限实现的参考组** |
| Run B | 搜索广度优化 | 恢复多候选探索能力 |
| Run C/D | 新增全局累积机制 | 恢复跨策略经验继承能力 |
| Run E | 新一代自适应结构算法 | **原始完整设计的恢复与重构** (非新增) |

## 审计核心结论 (部分, 待 fork agent 补全后汇总)

1. **emergence 偏移是设计性回归, 非代码bug**: 8/31 证伪 theorem_type 聚类 → 退回预设4段 + 放弃结构生长目标. 但9月 expand-v2 证明结构生长可通过覆盖驱动实现(非emergence路径), 该能力被后续 meta-skip + 叶隔离叠加限死.
2. **expand meta-skip 是"一次负向禁整类"的典型**: v2 为修 v1 -0.07 引入, 9月合理(依赖 not-meta 占比), 11月失败分布变98%meta后变成禁整类探索, 未重新审查. 已有 paired/CI/veto 保护, 硬编码排除冗余.
3. **"Run E = 恢复"判断成立**: 原始设计要求结构生长 + 自主发现, 当前实现限死两者. 恢复 = 解除 meta-skip + 让结构生长能力回来(非emergence的覆盖驱动路径已被9月验证有效) + 动态 slot registry.
4. **保留新机制优势**: paired_rescore + bootstrap CI + veto + pool_filter 是有效测量/控制, 应保留. 恢复的是进化能力, 不是退回旧测量.

## 待补全 (fork agent 进行中)
- overlay parent/组合逻辑的历史差异
- leaf.slots 初始化为全S₀ 的时点
- K 被 n_parallel 覆盖的时点
- emergence OFF(stage0) 之外的 EFL(--emergent_facets, 11-24 移植自AIME)是否是替代涌现路径

**关联**: [[facet_tree_skill_evolution_design]] (原始设计) / [[facet_tree_empirical_verdicts_20260831]] (emergence证伪) / [[expand-v2-design-20260901]] (meta-skip引入) / [[bottleneck3-strategy-capacity-20261127]] (第三瓶颈机制) / [[three-skill-strategy-compare-20261127]] (三方对比)
