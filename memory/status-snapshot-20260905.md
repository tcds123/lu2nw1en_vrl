---
name: status-snapshot-20260905
description: 实验进展状态快照 (2026-09-05). 公平条件下 evolver 逼近 skillopt (−0.87pp, theo_iter6 0.5542), 未 CI-gated 追平. 系统瓶颈 = val→test 不迁移 + 数据集 meta 占比过高. 下一步三件事已启动.
metadata:
  type: project
---

# 实验进展状态快照 (2026-09-05)

## 项目定位与北星

- **试验场**: LiveMathematicianBench 数学选择题 (177 题, 1:3:6 → train=18/val=53/test=106)。
- **流水线**: training-free evolver (`skill_evolver_math.py`) 做 facet-tree 真 GA skill 进化 (facet_edit 单轴精炼 + overlay 跨轴叠加 + crossover 交叉 + expand 长新段 + strengthen 加固段)。
- **北星**: 公平条件 (真 S₀ 起跑、零 skillopt 嵌套) 下追平/超越 **SkillOpt = 0.5629**。
- **硬约束**: 不得手 craft skill 文本, 只能优化算法; 遇问题研究"为什么没接受"非判证伪。

## 关键数据全景 (test=106, eval_k=3, 多 seed)

| 版本 | test=106 | vs skillopt(0.5629) | 公平性 |
|---|---|---|---|
| S₀ | 0.3050 | −25.8pp | — |
| 8238b5e8 (旧 fair-best) | 0.5354 | −2.75pp | ✅ 干净 |
| iter5 crossover | 0.5440 | −1.89pp (未 CI-gated 越) | ✅ |
| **theo_iter6 (当前最近)** | **0.5542** | **−0.87pp** | ✅ |
| strengthen iter8 facet | 0.5409 | −2.20pp | ✅ |
| strengthen iter6 composite | 0.5189 | −4.40pp | ✅ |
| evo_seed (种 SkillOpt) | 0.5590 | −0.39pp | ❌ 嵌套非公平 |
| skillopt (基线) | 0.5629 | — | — |

**结论**: 公平条件 evolver 已从落后 8.4pp 收窄到 **−0.87pp** (theo_iter6), 未 CI-gated 越过 skillopt; seed4=0.5723 单点越过, 统计无定论 (非 defeat)。

## 最新进展 (9 月)

### 算子H strengthen run (9/5 02:42 完成) — 未达标, 产出方法学发现
- 新增 `--strengthen` flag, 目标关闭 theo_iter6(0.5542) vs evo_seed(0.5590) 第2双向轴(existence↔uniqueness) ~0.5pp gap。
- test=106 双 eval: iter8 facet **0.5409** (−1.33pp) / iter6 composite **0.5189** (−3.53pp), 均未越 theo_iter6; 但 vs S₀(0.3050) 均显著正 (Δ+0.23)。
- **🚨 证伪"best-selection bug"假说**: EVAL#1 后我误判选了单 facet 丢了含双向轴的复合, EVAL#2 实测 iter6 composite 反而比 iter8 facet **更差 −2.2pp** → 选择逻辑**正确**。expand 长出的双向轴段是 val/probe 过拟合产物, **不迁移 test 且主动有害**。
- **strengthen 全程未触发**: 设计前提 (expand 饱和→target=None) 在 8 iter 内难达 — expand 对"已覆盖但仍有失败"的类型持续生成更强变体 (Implication 段 iter4 +0.127 → iter6 +0.258), 抢先于 strengthen。**设计缺口待修**。

### 跨路径收敛 (9/4) — 算法自洽性证据
- 干净 theo_iter6(0.5542) 与 seeded evo_seed(0.5590) **独立收敛到同一"双向测试区分强弱命题"启发式**。干净路径只停 1 轴 (implication↔equivalence), evo_seed 多第 2 轴 (existence↔uniqueness) → ~0.5pp gap ≈ 缺第 2 对。算法不靠注入也长出双向测试 = 设计自洽。

### 三 break 修复 + theo_iter6 坐实 (9/3)
- break① (donor 池放开复合段) / ② (coherence 允许协同) / ③ (per-subset 诊断回路) 修复后, theo_iter6=0.5542, gap 收窄至 −0.0087 (从 −0.0275 共收窄 +0.0188, seed4=0.5723 越过 skillopt)。

## 核心瓶颈 (系统性, 非单次 bug)

1. **val→test 不迁移 (最致命)**: val 最难 15 题 ~93% meta-option, patcher/expand 过拟合 val 特定失败分布不匹配 test。Task3(0.4654)/Task4(0.3632)/Phase C(0.3703)/strengthen 全同病。**"内容存在" ≠ "test 迁移"** — 同名启发式在不同演化路径下迁移性不同。
2. **数据集组成限制**: select set 86% meta / test=106 同结构 (89 meta/17 not-meta)。非 meta 题太少 → expand 可涌现失败类型穷尽 (仅 Quantifier/Implication 2 类) → 涌现饱和是数据本体造成, 非算法缺陷。
3. **expand "已覆盖但更强变体"行为**: 让 strengthen 永不触发 (需在 select_strengthen_target 收紧饱和判定)。

## 下一步计划 (2026-09-05 启动)

### 立即可做 (低/零 API) — ✅ 三件齐 (2026-09-07 全完成)
1. ✅ **strengthen 饱和判定收紧 (零 API 代码层)**: 已实现+自测全绿。详见 [[strengthen-saturation-fix-20260907]]。
2. ✅ **同窗口 skillopt 臂跑 test=106**: theo_iter6 vs skillopt 同窗口 4-seed paired CI95[−3.07,+4.32]含0 → 统计无定论 (非 defeat 非 越)。gap 从点估计 −0.87pp 收窄到同条件 −0.63pp。详见 [[same-window-skillopt-paired-ci-20260907]]。
3. ✅ **best_skill_iter5.md 4-seed 复测**: mean=0.5480, ΔvsS0=+0.2469 CI95[+0.1769,+0.3176] PASS。⚠️4-seed 推翻旧"bit-identical→eval 确定性"结论。详见 [[iter5-crossover-test4seed-20260904]]。

### 算法侧 (消耗 API, 按用户方法论排序)
4. **跨轴交叉变异遗传 (方向B)**: 当前 crossover 2-parent 段嫁接, guard/cohere 频繁失败 (iter5 两双向段尝试 2 次全结构失败)。做多 parent 段组装 (3+ donor 同台) 或基于失败分布定向选 donor。
5. **树状回退 + 历史组合复现 (方向C)**: `--donor_topk` 已实现偶数代取历史第二优。可做祖先链回溯 — 从某代 best 祖先组合重试, 或从 N 代前被淘汰但含未被继承好段的失败节点取 donor。
6. **多子集 Pareto 准入 (方向D)**: 按失败类型分子集独立 CI 门控, 从 Pareto front 选均衡解 — 直接对抗"not-meta 增益被 meta 89 题淹没"。

### 需用户定夺 (有 tradeoff)
7. **突破数据天花板**: (a) rebalance train:val; (b) 放开 expand 只攻非 meta 限制 (曾 v1 退步 −0.07, 高风险); (c) 引入其他数学证明题源 (偏离 benchmark)。

## 关联
[[strengthen-operator-run-20260904]] (算子H 完整数据) / [[bidirectional-test-convergence-20260904]] (双向测试 gap 叙事, 注意 test 迁移性) / [[evolver-future-goals-20260902]] (用户方法论锚点) / [[three-breaks-facet-tree-20260902]] (三 break + theo_iter6) / [[iter5-crossover-test4seed-20260904]] (iter5 0.5440) / [[lit-vs-impl-vs-verdict-20260831]] (接口接通≠机制成功)
