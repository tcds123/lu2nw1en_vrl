# 方向 E (备选): archive 驱动 GA / 跨代遗传 (2026-09-09 用户提出)

## 背景 (用户提议)
方向 B (S₀ FAIR 起步) 北星未越后, 用户提出备选方向: 树状进化每次往下走时, 记录进化过程的变化, 针对 这些内容尝试跨代优化和保存。算法思想 = **不只 keep-best, 将历史中好的表现通过交叉/遗传/变异保存下来**。

## 命中的根因 (非 floor-regression, 更深一层)
当前架构 = **单 lineage keep-best**: 每代只从当前 `best_node` 突变, 历史峰值一旦被 floor 拉低就丢失。
- MEMORY.md 第 55 行早标注: "用户 skill_evolver_math.py 处最弱一档 (连 SePO 的 archive 都没有, 纯 keep-best 单 lineage) → val→test 脱节直接原因"。
- archive 其实已在存历史好基因 (archive_iter*.json 节点含 skill 全文 + val_delta + val_ci), 但只用于 crossover donor, **没用于突变起点和输出环节**。
- 方向 B 的多 parent crossover 是此思想雏形 (从 archive 选 3 donor 嫁接), 但只改"交叉来源", 没改"突变来源"+"输出" → floor 仍能拉低产物。

## 与现有方案的关系 (不同层级)
| 方案 | 治什么 | 层级 |
|------|--------|------|
| 修 floor (=0 / 输出 archive max) | best_node 被拉低 | 治标立竿见影 |
| 方向 C (ancestor-chain backtrack) | 回看祖先链找 donor | 中层局部回看 |
| **方向 E (archive 驱动 GA)** | 单 lineage 过拟合 val → val→test 不迁移 | 治本架构升级 |

## 三个可操作落点 (待读代码确认 best_node 更新逻辑后细化)
1. **输出环节**: `best_skill_evolved.md` = archive 里 `val_delta` 最高节点, 非 `best_node` (最小改动, 直接治 floor-regression)。
2. **突变来源**: 每代从 archive 按 `val_ci` 加权采样精英种群作突变起点, 非只从 best 突变 (population-based, 类 PromptBreeder/DIVE)。
3. **交叉池**: archive 历史好基因全进 donor 池跨代重组, 非只匹配当前 iter 失败类型。

## 风险 ⚠️
train set 仅 18 题。archive 驱动 GA 在小 train 上**更易过拟合 val**, 多样性可能放大 val→test 不迁移。不应裸跑, 须配防过拟合机制:
- DIVE 多种群 bootstrap (MEMORY.md 第 55 行调研)
- PromptBreeder fitness sharing
- 严格 select→test 迁移监测 (现有 4-seed CI 口径)

## 决策时机 (等峰值诊断 PID 104276)
- 峰值 skill (cb901c5c, val +0.319) test **明显 > 0.5393** → floor-regression 主因 → 先修 floor (治标快), 看能否凭此越线。
- 峰值 test **≈ 0.5393** → 单 lineage 过拟合 val 是真根因 → 方向 E (治本) 上, 带 DIVE bootstrap 防过拟合。

## 文献支撑 (已调研, MEMORY.md 第 55 行)
- DIVE: 唯一 skill 级 2-parent 重组 + **多种群 bootstrap** + 三分防过拟合 ← 防过拟合范本
- PromptBreeder: fitness sharing ← 多样性维持
- SePO: archive 概念来源 (单 lineage 套壳, 最弱)

## 教训 (待验证)
1. 方向 B 已证"多 parent 交叉机制可行但 floor 拖累" → 方向 E 须先修输出环节 (落点1) 再上种群突变 (落点2), 否则同样被 floor 拖累。
2. 小 train + GA = 过拟合放大器, 防过拟合机制是方向 E 的必要组件非可选。

**关联**: [[northstar-direction-b-verdict-20260909]] (方向B未越→用户提E) / [[direction-b-run-trajectory-20260909]] (floor-regression发现) / MEMORY.md 第55行 (单lineage根因+文献调研) / `docs/phase2_issues.md` (floor-regression系统性bug)
