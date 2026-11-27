---
name: design-drift-audit-final-20261127
description: 原始设计一致性审计终稿 — 6机制偏移核实 + 4个关键区分(叶隔离后续引入/overlay上游饿死/emergence从未完成/expand长期默认) + 原始设计-实现-偏移-恢复-验收对应表 + 8项算法级验收测试 + Run E完整定义
metadata:
  type: project
---

# 原始设计一致性审计 · 终稿 (2026-11-27, 零API)

用户判断(已证实): "我们面对的不是原始设计不够好, 而是原始设计在长期局部修复/机制移植/防退化约束中逐渐发生实现偏移. Run E = 原始完整设计的恢复与未完成能力的补全, 非新增."

## 两类不同性质的问题 (用户区分, 核实)

| 能力 | 性质 | 历史 | 当前 |
|------|------|------|------|
| 多候选探索 | 已实现后被削弱 | 已有 | Run B 恢复中 |
| 自动发现失败模式 | 部分实现 | 已有(预设分类) | 仍依赖预设 |
| **Facet 轴自主涌现** | **设计但从未充分实现** | Stage 0 未兑现 | EFL flag 注册但未适配MCQ |
| Skill 新模块生长 | **已实现后被限制** | 9月覆盖驱动 expand 成功 | meta-skip 限死 |
| 跨模块经验累积 | **已实现后被削弱** | 历史 overlay 路径 | 叶隔离, D 恢复中 |
| **分支自动合并淘汰** | **设计但从未实现** | 原始设计§2.2(b) | 无实现 |
| 统计评估防退化 | 持续完善 | — | 新机制优势(保留) |

→ Run E = 恢复(已实现被削弱的: 结构生长/组合/累积) + 补全(从未实现的: 自主涌现/合并淘汰). 非回滚9月版本.

## 4 个关键区分 (决定恢复/重实现/替换)

### 区分1: 叶隔离 — 后续实现引入, 非原始设计
- **原始设计§2.3 L98**: "该分支的变异只允许动对应 facet slot, 其余 slot **继承父代不动**". 这里的"父代" = **全局 archive 节点**(含已晋升内容), 非 S₀.
- **当前实现**: 每 leaf 独立 slots, `leaf.slots` 初始化全 = `dict(root_slots)` (S₀) (L2900). A-axis 晋升进 archive+best_node, 但 B-axis leaf.slots[A] **仍是 S₀** → 跨轴回退.
- **结论**: 叶隔离(slots全S₀+不跨leaf同步) = **后续实现引入的限制**. 原始设计的"继承父代"指继承当前 best(含晋升), 不是继承 S₀.
- **恢复措施**: 不需恢复旧代码, 需**重新实现** global slot registry — 候选 parent 取自全局 best_node.slots(含所有已晋升段), 非 leaf.slots. Run D global_inherit 是部分实现, 需扩展为动态 registry.

### 区分2: overlay — 上游候选材料不足, 非自身回归
- **9月 run_20260904 overlay**: 8候选, 5-8sec(多轴组合), delta +0.12~+0.23, **多轴 ci>0** → overlay 多轴组合有效.
- **Run B iter1 overlay**: n_axes=1, delta=−0.009, admitted=N. **只有 OC ci>0(+0.0751)** → overlay 只能从 OC 取 donor = n_axes=1, 无法多轴组合.
- **结论**: overlay 组合算法**没回归**, 是**饿死的** — 上游 facet_edit 只有1轴 ci>0. 与 expand 同病(都是上游材料不足).
- **恢复措施**: 不需改 overlay 本身. 需让上游多轴产出 ci>0 候选(解 expand meta-skip + 放宽 facet_edit 容量) → overlay 自然恢复多轴组合.

### 区分3: 自主涌现 — 从未完成, 非曾完成后被关闭
- **代码核实**: 无 `def.*emergent_cluster` / `自动开新分支` 实现. emergence 永远 OFF(stage0) (L2970).
- **EFL (--emergent_facets)**: 2026-11-24 移植自 AIME, 但 help 明确 "LiveMath MCQ taxonomy 未适配 EFL prompt, ON 路径需先建 facet_library 的 MCQ 版, 默认 off, ON 会 no-op 回退". Run A/B 均未启用.
- **结论**: 自主涌现**从未完成**. 8/31 证伪 theorem_type 聚类键后, 退回预设4段, 未寻找其他涌现路径. EFL 是移植的替代路径但未适配 LiveMath.
- **恢复措施**: **需补全实现**, 非恢复旧代码. 但不必复刻 theorem_type 聚类(已证伪). 可走"覆盖驱动 expand 涌现" — 9月证明 expand 自主发现未覆盖失败类型(Quantifier)并开新段 = 涌现的可行替代路径. 关键: 涌现 = 自主发现结构缺口并创建模块, 不限于聚类键方法.

### 区分4: expand meta-skip — 长期默认, 非临时配置
- **代码核实**: L763 `if lbl==meta_lbl: continue` **无条件**, 无 flag 控制. `--expand` 默认行为 = 跳过 meta. expand_every/expand_candidates 有 flag 但 meta-skip 无.
- **结论**: expand 限制是**长期默认算法行为**, 非临时实验配置. 已进入算法骨架.
- **恢复措施**: 需**代码修改**解除 L763. 保留质量门控(paired/CI/veto)替代硬编码排除 — 用户原则: "性能问题优先通过反馈/竞争/选择解决, 非硬编码永久排除合法策略".

## 原始设计—实际实现—偏移原因—恢复措施—验收证据 对应表

| # | 原始设计能力 | 实际实现 | 偏移原因(时点) | 偏移性质 | 恢复措施 | 验收证据 |
|---|------------|---------|--------------|---------|---------|---------|
| 1 | 多候选探索(§2.4 stance旋转) | n_parallel=1(K被忽略L4262) | 机制移植 | 已实现被削弱 | Run B n_parallel=4(进行中) | iter1 OC 4差异化候选 |
| 2 | 自动发现失败模式(§2.2b) | 预设 classify_failure_type | 8/31聚类证伪 | 部分实现 | 补全: 覆盖驱动发现+涌现 | 验收#1自主发现 |
| 3 | **Facet轴自主涌现**(§2.2b) | emergence OFF(stage0), EFL未适配 | 8/31证伪+退回预设 | **从未完成** | 补全: 非聚类路径涌现 | 验收#1+#2 |
| 4 | **Skill新模块生长** | expand跳过meta(L763) | 9/1修v1 -0.07 | **已实现被限制** | 解除L763, 保留质量门控 | 验收#2候选生成 |
| 5 | 跨模块经验累积(§2.5) | 叶隔离(leaf.slots全S₀) | 后续实现引入 | 已实现被削弱 | D global_inherit→动态registry | 验收#5#6 |
| 6 | 策略组合(§2.5 overlay/xover) | overlay饿死(上游材料不足) | 间接(被#4#2拖累) | **未回归(饿死)** | 解#4#2后自然恢复 | 验收#7组合再优化 |
| 7 | **分支自动合并淘汰**(§2.2b) | 无实现 | 从未实现 | **从未完成** | 补全: 冗余轴合并候选 | 验收#8合并淘汰 |
| 8 | 统计评估防退化 | paired/CI/veto/pool_filter | 持续完善 | **优势(保留)** | 保留, 不回退 | — |

## 三处研究逻辑错误 (用户记入研究档案)

1. **将实现路径失败扩大为目标失败**: 8/31证伪 theorem_type 聚类键 ≠ 证伪"从失败模式发现优化维度/创建模块/动态调整结构". 9月覆盖驱动 expand 成功证明目标可通过其他路径实现. **错误在放弃目标, 非换技术路径**.
2. **将候选级失败转化为类别级禁止**: v1 expand攻meta −7pp → v2 永久禁止expand攻meta. 应: 拒绝该候选+保留反馈+继续探索. **通用原则: 性能问题优先通过反馈/竞争/选择解决, 非硬编码永久排除合法策略**. 结构合法性检查可保留, 但防退化护栏不能替代搜索机制.
3. **工程零回归 ≠ 算法能力零回归**: 11月移植验 paired/CI/veto/缓存/端到端, 未验"新版本是否仍具完整进化能力". 后续须加算法级验收(下方8项).

## Run E 完整定义 (用户设计, 核实可行)

**核心**: Facet Tree = 动态组织搜索的工具; Global Skill = 最终持续进化对象. 两者不绑定为固定一一对应.

进化过程:
```
Failure-driven Discovery (从失败样本自主发现共享问题与策略缺口)
  → Adaptive Facet Organization (已有轴优化 + 新轴涌现 + 冗余轴合并淘汰)
    → Refine (改进已有策略) / Expand (发现新模块) / Compose (组合有效策略)
      → Paired Global Competition (统一全局竞争门控, 保留新机制优势)
        → Dynamic Global Skill (内容/模块/结构共同进化)
          → 下一轮失败诊断与持续进化
```

关键设计: Evolver 可发现新策略模块**不必立即为它创建独立搜索轴**; 也可发现两 facet 解决相同失败机制 → 合并. 符合原始 §2.2(b) adaptive 分支数.

## 8 项算法级验收测试 (用户定义)

| # | 能力 | Mock/确定性测试 | 真实API实验 |
|---|------|----------------|------------|
| 1 | 自主发现 | 从未预设名称的共享失败模式识别策略缺口 | 模型自主发现有效模块 |
| 2 | 候选生成 | 针对缺口生成新section(非仅改已有段落) | 模型生成实质新模块 |
| 3 | 多策略竞争 | 同一失败模式生成实质不同方案 | 候选策略真实差异 |
| 4 | 正确晋升 | 新模块过统一全局竞争门控 | 晋升模块有实际收益 |
| 5 | 动态注册 | 晋升后进可持续继承的Skill结构 | 结构跨轮存活 |
| 6 | 跨轴继承 | 改其他facet不丢失已晋升新模块 | 累积不回退 |
| 7 | 组合再优化 | 新模块后续被编辑/组合/优化 | 模块被组合产生增益 |
| 8 | 合并淘汰 | 提出删除/合并低效模块候选并评估 | 冗余模块被移除不损性能 |

**两层验收**: Mock证明系统允许结构进化(状态/血缘/动态注册/继承/门控逻辑正确); 真实API证明Evolver具备实际自主发现能力. 两者都需要.

## 修正后实验优先级 (用户拍板, 效果优先)

| 实验 | 修正后角色 | 安排 |
|------|-----------|------|
| A | 受限实现参考组 | ✅ 完成 |
| B | 多候选搜索能力恢复 | 🚀 运行中, 不干预 |
| C | 累积机制消融对照 | 保留(B/D/E后补) |
| D | 多候选+跨轴累积 | B后执行 |
| **E** | **原始完整方法恢复与补全** | **立即进入设计与开发** |
| SkillOpt | 强方法基线 | 独立对照(D/E后) |

**执行顺序**: B → D → **E(真实性能验证, 优先)** → C + 机制消融. 效果优先: 先让完整算法发挥能力, 再系统解释各部分贡献.

## 论文定位修正 (用户判断)
- A/B/C/D = **部分实现或机制对照**, 非完整方法
- E = **完整方法的候选实现**, 非"后来新发明的机制"
- 若E能力本属原始设计, 论文如实标注, 不宣称新发明

## fork agent 补全的历史细节 (2026-11-27, 已合并)

**机制4 叶隔离精确时点**: 2026-08-25 `memory/facet-tree-no-cumulative-20260825.md` 已诊断叶隔离为"结构性设计缺陷"——9叶除本轴段外其余全=S₀. 但当时定位为**防塌缩护栏的代价**, 靠 overlay/crossover 补救. 即叶隔离是 Stage 0 实现(8月)就有的有意设计(防塌缩), 非后期意外引入, 但其补救通道(overlay/crossover)在11月链式失效后彻底失声. global_inherit(11-27)仍只继承leaf拥有的轴、不引expand第5段(L3032-3034注释自承"当前机制不继承跨结构段, 记为限制").

**机制3 overlay parent 来源差异**: 原始设计§2.5 = **2-parent facet crossover**(LLM合成两父本强slot). 当前实现 overlay = **构造性按段拼接**(root_slots底座+best_node carry-forward+donor池按段名取各段top-K历史最佳), **已取代设计中的crossover成主力结构增长机制**. 9月能长段(因expand先长段→overlay carry-forward带过来); 11月overlay 6次0晋升(因expand饿死→无新段可carry-forward). overlay自身逻辑未变, 被机制2间接锁死.

**机制5 section lifecycle 完成度**: "被继承" = break-①修复(donor池含archive-only段,L2468-2487)+carry-forward 部分实现; "继续优化" = 受叶隔离限(leaf只持4段,新段不进leaf.slots); **"评估后删除" = 完全未实现**(archive keep-losers只增不删, 无合并/淘汰). 设计§2.2(b)的"自动合并/淘汰"从未实现.

**机制6 K/stance 时点**: n_parallel 2026-09-09前后随方向E pop_mutation引入(`phase2_issues.md` L2262). stance绑定facet在facet_edit_guard物理约束后已无意义(变异被guard锁在ax段). K被忽略是facet_tree路径重构副产物. 专属stance池因emergence放弃永无兑现.

**链式锁死总判(fork agent)**: 6机制中emergence(机制1)8/31实证证伪主动放弃; 其余5机制从"涌现式自适应"退化为"结构化硬约束". 9月失败分散时仍能长段(theo_iter6 9段/0.5542); 11月meta主导(98%)下链式锁死: **expand跳meta(机制2)→expand饿死→overlay无新段carry-forward(机制3)→best_n_sec始终4(机制5)→叶隔离跨轴不累积(机制4)→K被n_parallel取代stance不绑facet(机制6)**.

## 原始设计能力当前状态 (fork agent 总结表)

| # | 机制 | 原始设计能力 | 当前代码状态 | 性质 |
|---|------|------------|------------|------|
| 1 | emergence 自动涌现 | 失败聚类开/合并facet | --no_emergence默认OFF; EFL移植但_EFL恒None | **禁用**(8/31实证证伪) |
| 2 | expand攻meta | 对任何失败类型长新段 | L763显式跳过meta | **削弱(有意禁用)** |
| 3 | overlay跨轴组合 | 2-parent LLM合成 | 构造性按段拼接; 依赖expand长段 | **削弱**(被机制2间接锁死) |
| 4 | leaf/global状态 | 树状archive共享lineage | 叶隔离slots全S₀; global_inherit部分补救 | **削弱**(8/25诊断结构性缺陷) |
| 5 | section生命周期 | 继承+优化+删除 | 继承部分实现; 优化受限; **删除未实现** | **削弱** |
| 6 | K/stance绑定facet | stance绑定facet轴 | K被忽略; stance绑定修补角度 | **削弱** |

## 核心结论
1. **叶隔离 + overlay饿死 = 后续实现引入/间接拖累, 可恢复**(不需恢复旧代码, 需重实现global slot registry + 解上游材料不足)
2. **自主涌现 + 分支合并淘汰 = 从未完成, 需补全**(非恢复, 但可走9月验证的覆盖驱动expand路径, 不必复刻已证伪的聚类)
3. **expand meta-skip = 长期默认, 需代码修改解除**(保留paired/CI/veto替代硬编码)
4. **保留新机制测量优势(paired/CI/veto/pool_filter), 恢复的是进化能力**
5. **Run E = 恢复(被削弱的) + 补全(未实现的), 非回滚9月, 非新增**

**关联**: [[design-drift-audit-20261127]] (初稿) / [[facet_tree_skill_evolution_design]] (原始设计) / [[facet_tree_empirical_verdicts_20260831]] (emergence证伪) / [[expand-v2-design-20260901]] (meta-skip引入) / [[bottleneck3-strategy-capacity-20261127]] (第三瓶颈) / [[three-skill-strategy-compare-20261127]] (三方对比) / [[history-corrections-20261127]] (历史修正)
