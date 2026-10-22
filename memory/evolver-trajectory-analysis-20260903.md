---
name: evolver-trajectory-analysis-20260903
description: evolver_log.json 逐代轨迹深度分析 (run_20260902_205708, PID 115988, 8 iters). 进化路径/交叉机制/涌现section + 6 个算法侧优化靶点 (非数据层). 核心发现: crossover 4/7 拒因=cohere-LLM 结构守卫单次失败非价值拒; expand 只攻非 meta 稀疏类型 (2 段后饱和); branch_failure_types 8 代不变=诊断陈旧; expand "已覆盖"判据=存在性非 efficacy.
metadata:
  type: project
---

# evolver 逐代轨迹分析 (2026-09-03)

回应用户: "算法肯定还是有优化空间...分析进化路径, 怎么交叉的, 涌现了哪些section"。本分析**只找算法侧优化空间**, 不下"转数据层"结论。数据源: `run_20260902_205708/evolver_log.json` 全 8 代 + 4 个 candidate .md + `skill_evolver_math.py` guard 代码。

## 一、进化路径 (baseline → iter8)

| iter | best_Δ | best_CI | n_sec | best 来源 | pass | 关键事件 |
|------|--------|---------|-------|-----------|------|---------|
| base | +0.019 | -0.028 | 4 | — | 0.244 | S₀ 自比≈0 (CI 跨 0) |
| 1 | +0.268 | +0.193 | 4 | overlay | 0.493 | overlay 叠 Theo+Hypo 两轴 |
| 2 | +0.282 | +0.217 | 5 | expand:Quantifier | 0.507 | **涌现① Quantifier** (n4→5) + crossover 拒 |
| 3 | +0.305 | +0.226 | 6 | overlay | 0.531 | facet Theo +0.390 PROMOTE + **crossover 准入** (pass0.596 最高) |
| 4 | +0.380 | +0.305 | 6 | overlay | 0.606 | **PEAK** + **涌现② Implication** (n6→7) + crossover 拒 |
| 5 | +0.268 | +0.186 | 9 | overlay | — | **Bug C 回退** + crossover 准入 (Δ+0.164 低) |
| 6 | +0.319 | +0.232 | 9 | overlay | — | facet Theo +0.394 (真 best) + **expand 饱和** + crossover 拒 |
| 7 | +0.315 | +0.235 | 9 | overlay | — | crossover 拒 |
| 8 | +0.324 | +0.238 | 10 | overlay | — | crossover 准入 (Δ+0.362) + **expand 饱和** (max-sections) |

- **iter4 是真峰** (+0.380/pass0.606, n6 干净)。iter5-8 全部 Bug C 污染 (best 退 0.380→0.268, 膨胀 n6→n10)。**~0.57 天花板结论是从 iter4/iter6 干净候选抽的, Bug C 修复后须 rerun 才能定真天花板**。
- n_sec 单调生长 S₀=4 → I2 5 → I3 6 → I4 7 → (Bug C 膨胀) I5 9 → I8 10。涌现机制跨代累积有效。

## 二、交叉机制 (怎么交叉的)

**7 次尝试: 3 准入 (I3/I5/I8), 4 拒绝 (I2/I4/I6/I7 全标 "guard/cohere-fail")。**

### 4 次拒绝的真相: 结构守卫失败, 非价值拒绝
`recombine_facets` (L1116-1166) 的 cohere-LLM 单次产出过 6 道结构守卫之一即返 `ok=False` → 主循环 L1667 统一记 "guard/cohere-fail", **从未进入 eval**。6 道守卫: LLM 异常 / clean_skill 空 / decompose 空 / **段数减少** (L1154 `len(child)<len(parent_a)`) / freeze 段丢 / freeze md5。
- **关键**: 4 次拒绝 = cohere-LLM 单次产出不合结构, **不是"重组价值不够"**。它们根本没被评测过。
- **段数减少守卫 (L1154) 与 cohere-prompt 矛盾**: prompt (L1100) 明确允许 "MAY merge two sections", 但 merge 合法降段数时 L1154 直接拒。这是设计内冲突 — 允许合并的 prompt 撞上禁降段数的守卫。

### 准入门槛: 纯 `bootstrap_ci_lower > 0` (L1625)
- crossover **无 LOST/pass-drop 守卫** (overlay L1695 / expand L1813 都有 `pass>=base-0.06`, crossover 没有)。故 I5 crossover Δ+0.164 (低) 也准入 — 只要 CI>0。
- 3 准入: I3 (Theo×Quantifier +0.371, pass0.596 最高) / I5 (Hypo&Param×Implication +0.164, 低但 CI>0) / I8 (Theo×Option +0.362)。I3/I8 都以 Theo 为 parent A。

### 父本选择: 随机洗牌非适应度
`select_section_donors` (L1169) 取每段 top-K(=1) by val_ci → 展平 pool → **seeded shuffle** (L1599 `_xr.Random(probe_seed+t)`) → 取 pool[0]×pool[1]。无锦标赛/无适应度比例。每代**仅尝试 1 对** (pool[0]×pool[1])。

## 三、涌现了哪些 section (2 个)

1. **I2 "Quantifier Order & Uniformity"** — target=`Quantifier order and parameter dependence (∀∃ vs ∃∀, uniform vs dependent bounds)`。驱动=Hypotheses 分支 1 个 quantifier 失败。became_best=Y, not_meta_ci=+0.067 (子集救援准入)。
2. **I4 "Implication Direction & Biconditional Check"** — target=`Implication vs biconditional (one-way vs iff)`。驱动=Option 分支 1 个 implication 失败。admitted 但非 best (n6→7)。

**I6 expand 饱和**: "all-covered-no-gap" n_fail=17, failure_counts={meta-options:16, implication:1}。两个非 meta 类型已被 I2/I4 涌现段"覆盖" → 无新 gap → 停。I8 二次饱和 "max-sections-reached"。

## 四、🔥 6 个算法侧优化靶点 (非数据层)

1. **crossover 结构拒因 = cohere-LLM 单次失败** (4/7)。修: guard-fail 时重试/re-prompt (当前单次 L1612), 或放宽 L1154 段数守卫让合法 merge 过。**直接对应"研究为什么没接受"** — 答案: 不是重组不好, 是 LLM 一次性产不出合规结构。
2. **诊断陈旧**: `branch_failure_types` 8 代几乎不变 (meta-options 每分支 6-8, 全程)。诊断大概率对着固定 S₀ 失败集算, 不随 evolved best 刷新 → 看不到"进化后失败模式变了"。修: 诊断对当前 best 重跑。
3. **expand "已覆盖"判据=存在性非 efficacy**: meta-options 有 bullet (Theo 段 "a stronger result can be proven") → 判"已覆盖"→ 永不 expand/refine。但 meta-options 占 92% 失败仍持续。**"有段"≠"解决了"**。修: efficacy-based gap (段存在但该类型失败率仍高 → 列为 refine 靶)。
4. **无剪枝无定向父本**: archive=40 全留, leaves_ratio_mean=1.0 (L 无叶子被剪); best_ratio=0.0 (iter3+ best 恒为复合非叶)。父本随机洗牌非适应度。修: 失败/低 CI 节点剪枝释放算力; 锦标赛/定向选两强轴交叉。
5. **每代仅 1 对 crossover**: "历史组合可能更好" → 当前 pool[0]×pool[1] 单发。修: 每代尝试多对, 或跨代回溯祖先链 donor。
6. **Bug C 修复待 rerun**: iter5-8 best 退 0.380→0.268 + 膨胀 n6→n10 是已知 bug (已修磁盘待生效)。真天花板须 Bug C 修后干净 rerun 才能定。**不得在 Bug C 污染的轨迹上下"算法饱和"结论**。

## 五、对照用户方法论

- "允许涌现更多 section" → 靶点 3 (efficacy-gap 解锁 meta expand) + 靶点 2 (诊断刷新暴露新 gap) 直接扩涌现空间, 非数据层。
- "让不同轴交叉变异遗传" → 靶点 1 (降结构拒率) + 靶点 5 (多对尝试) + 靶点 4 (定向父本) 提升交叉产出。
- "研究为什么没接受" → 4 拡拒因已定位: cohere-LLM 单次结构失败 (非价值) + L1154 段数守卫与合并 prompt 冲突。
- "不得手 craft skill" → 全部靶点是改 `skill_evolver_math.py` 算法本身。

## 关联
[[three-breaks-facet-tree-20260902]] (主结果) / [[three-breaks-115988-iter1-6-20260903]] (逐代日志) / [[bug-c-subset-rescue-fix-20260903]] (Bug C) / [[evolver-future-goals-20260902]] (用户方法论锚点+方向A-E)
