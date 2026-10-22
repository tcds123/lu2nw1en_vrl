---
name: evolver-future-goals-20260902
description: evolver 算法未来优化方向与用户方法论锚点 (2026-09-02). 核心方法论: 允许涌现更多 section + 跨轴交叉变异遗传, 坚持探索不半途而废, 遇问题研究"为什么没接受"而非判证伪. 三 break 已实现待 API 验证, 后续方向记录于此.
metadata:
  type: project
---

# evolver 算法未来目标 (2026-09-02)

## 用户方法论锚点 (verbatim, 最高优先级)

> "工作重心应该放在**允许涌现更多的 section** 上，或者让**不同轴之间进行交叉变异遗传**，这些算法思路上，并且要**坚持探索不能半途而废**，遇到问题不是直接判算法证伪，而是要**研究为什么没有接受，如何进行下一步优化，怎么才能接受**。"

> "在整个树状结构中也是允许**回退和复现路径的变化**的，**历史中的组合可能会更好**。"

> 禁令: **不得手 craft skill 文本**, 只能优化算法本身。

## 当前状态 (三 break, 2026-09-02)

三处结构性断口已修复, 零 API 单测全绿, API 实验运行中 (见 [[three-breaks-facet-tree-20260902]])。
- ① `select_section_donors` 放开复合节点段进遗传池 (旧 `select_axis_donors` 排除 expand/xo/overlay)。
- ② coherence 允许协同 (merge/reorder/bridge) + recombine_facets guard 放宽。
- ③ per-subset 诊断回路 + 子集信号准入 (has_meta 分层, not-meta 增益救援)。
- 附带修 `facet_compose` 前缘粘连 bug (接口接通 ≠ 机制成功 的直接命中)。

## 北星目标: 闭合与 skillopt 的 gap

- evolver fair-best (8238b5e8) test 4-seed = 0.5354; skillopt = 0.5629; gap = −0.0275。
- expand-v2 (2f3c0c1f) 5 段含 Quantifier, not-meta 17 题 +0.157 (p=0.0235 显著), 但整体 −0.021 被淹没。
- 三 break 正是让"子集增益不再被整体单目标淹没"+"新段可被跨轴继承"的机制层修复。

## 未来优化方向 (按用户方法论排序)

### A. 涌现更多 section (用户首要)
- 当前 expand 只长 1 段 (section_expansion_guard add-only)。未来: 允许 expand 在"无专属段且非 meta"的最高频失败类型处多代累积长多段, 段数上限可调。
- emergence 判定细化 (零 API): 段名从未在 S₀ 出现 + val_ci>0 = 真涌现。
- 终局 C 组 (多代、开涌现、K=3 门控): `Common Failure Modes` 类 facet 从零长出 = 涌现 facet 内置验证 (见 [[facet-tree-s0-startpoint-20260821]])。

### B. 跨轴交叉变异遗传 (用户次要但同等重要)
- break ② 已让 coherence 不再反协同。未来: crossover 不只 2-parent 段嫁接, 可做多 parent 段组装 (3+ donor 段同台), 或基于失败分布定向选 donor (哪段改了哪类失败)。
- overlay 已能堆每轴最佳段。未来: overlay 不只单轴 top-1, 可做"每轴 top-k + 组合搜索"找最优段组合 (组合爆炸用 CI 门控剪枝)。

### C. 树状回退 + 历史组合复现 (用户"历史组合可能更好")
- `--donor_topk` 已实现偶数代取历史第二优 (回退路径)。未来: archive 历史节点不只 top-1/top-2, 可做"祖先链回溯" — 找到某代 best 的祖先组合重试。
- 跨代 donor: 让当前代 best 从 N 代前的某失败节点 (后来被淘汰但含未被继承的好段) 取 donor 段。

### D. 准入判据从单目标到多子集
- break ③ 已实现 not-meta 子集救援。未来: 子集不只 meta/not-meta 二分, 可按失败类型分多子集 (coverage_diagnosis 的全失败分布作基底), 每子集独立 CI 门控。
- multi-objective Pareto: 不再单 best, 而是 Pareto front (各子集 delta 不被支配), 最终从 front 选均衡解。

### E. 坚持 + 诊断回路 (用户方法论)
- 每次候选被拒, evolver_log.json 已记 admit/reason + per-subset 信号 (break ③ 诊断回路)。未来: 拒因聚类 → 若同一拒绝模式反复出现 (如 guard-reject), 升级算子/放宽 guard, 而非判算法证伪。
- facet_compose 前缘 bug 是范式: donor 段进了池却被格式细节静默丢弃 → 研究到根因才修。未来同类"接口通机制不通"需同样深挖。

## 验证纪律 (硬性规则)

- API 实验信号正向 → 4-seed test=106 复测坐实 (CI 纪律, 只判 CI 方向非点估计)。
- 任何 bug/异常 → 立即记 `docs/phase2_issues.md`。milestone → 记 `memory/` + 更新 MEMORY.md 索引。
- 不得只存会话上下文 (下次会话丢失)。

## 关联
[[three-breaks-facet-tree-20260902]] (三 break 实现细节 + API 实验) / [[expand-v2-4seed-result-20260902]] (③ 信号源) / [[lit-vs-impl-vs-verdict-20260831]] (断口诊断) / [[facet-tree-s0-startpoint-20260821]] (起点决策) / [[algorithm-future-optimization-directions-20260805]] (早期算法方向)
