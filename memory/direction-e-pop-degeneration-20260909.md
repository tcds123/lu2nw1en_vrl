---
name: direction-e-pop-degeneration-20260909
description: 方向E 落点② pop_mutation 在 n_parallel=1 + ci 加权下退化为单 lineage 的根因与解法方向
metadata:
  type: project
---

方向E Phase 1 (落点②, 突变起点=同 facet archive 种群采样) 在 `--n_parallel 1 --pop_mutation --pop_eps 0.3` 下, **前两代 (iter1+iter2) 退化为选 incumbent leaf, 但 iter3 起 pop-sample 激活** (2 条 `🧬 facet[...] pop-sample` 日志)。run = `run_20260909_171309_dirE_pop`。

**⚠️ 更正 (2026-09-09 21:05)**: 早期诊断"全程 0 条 / 字节近等价 direction-B" **错误**, 仅对 iter1+iter2 成立。iter3 实测 pop-sample 激活, run ≠ direction-B。

**根因 (按代际分层)**: `sample_mutation_parent` (skill_evolver_math.py L2080-2089) 权重=`max(0,val_ci)+1e-6`, ε-greedy eps=0.3 (30% 直返 incumbent, 70% 加权采样)。
- **iter1 (gen1)**: pool≤1 (仅 root leaf) → `len(pool)<=1` 早退返 leaf。无打印。✅ 设计内回退。
- **iter2 (gen2)**: pool={root_leaf(ci=-0.020→权重~1e-6), iter1_child(=incumbent, ci=+0.020→权重~0.02)}。加权采样 ~99.99% 落 incumbent → `sampled_parent is leaf` → 不打印。seed=42 iter2 draw=0.8015 复现确认。**此处退化成立**。
- **iter3 (gen3+)**: pool 增至 3 节点 {root, iter1_elite(ci=+0.020), iter2_elite(=incumbent, ci=+0.055)}。iter1_elite 非 incumbent 且 ci 正 → 权重 0.020, 占非-incumbent 质量 ~27%, 整体 P(采到 iter1_elite)≈0.70×0.27≈19%。**实测激活**: Option Comparison 采样 parent=iter1 elite(δ=+0.080), Theorem-Level Precision 采样 parent=iter1 elite(δ=+0.122)。

**iter3 pop-sample 结果分叉 (非字节等价 direction-B)**:
- Option Comparison: 从 iter1 elite(δ=+0.080) 突变 → cand +0.019 CI=-0.025 → **KEEP-PARENT** (劣于 incumbent +0.108)
- Theorem-Level Precision: 从 iter1 elite(δ=+0.122) 突变 → cand **+0.235 CI=+0.162** → **PROMOTE** 超 incumbent +0.136 (大跳, 可能 pop-sample 驱动 — 待与 direction-B iter3 对比确认)

**Why**: ci 加权让"最高 ci"垄断, 但只在 pool={root, incumbent} (2节点, root 权重~0) 时退化为单 lineage。pool≥3 (iter3+) 后历史 elite 有正 ci → 可被采到。**退化是 gen1-gen2 warm-up 问题, 非全程**。

**How to apply**:
- 验证 pop 路径是否真激活 = 看 `🧬 pop-sample` 日志出现, **非**只看 flag ON。
- 解法 (待跑): (a) uniform 采样取代 ci 加权; (c) 采样显式排除 incumbent (sampled_parent ≠ leaf)。组合 (a)+(c) 最直接。
- RNG seed 缺 facet 索引 (L2086 `Random(seed*1000+t)`): 同 iter 三 facet 共享 draw, 应加 `+bi*7` 独立化。
- n_parallel=1 是单 lineage 温床; 种群方法价值依赖 n_parallel>1 的 pool 丰富度。

**当前 run 处置**: 让其跑完取 4-seed test=106 CI。预期 ≈ direction-B 0.5393 (字节近等价)。若确如此 → 证伪 pop_mutation 在 n_parallel=1 有增量, 转 (a)+(c) 重跑。这是一个干净对照数据点 (证明退化, 为解法 motivation)。

**关联**: [[direction-e-impl-design-20260909]] (落点②设计, 其 "gen≥2 种群生效" 假设被本发现推翻) / [[direction-b-run-trajectory-20260909]] (0.5393 baseline) / `docs/phase2_issues.md` (完整根因+解法)
