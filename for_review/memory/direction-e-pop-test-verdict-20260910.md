---
name: direction-e-pop-test-verdict-20260910
description: 方向E 落点② pop_mutation 4-seed test 判定 — val+0.239 未迁移, test 显著负于 direction-B (负结果)
metadata:
  type: project
---

方向E 落点② pop_mutation run (`run_20260909_171309_dirE_pop`) 的 4-seed test 评测完成, **结论: pop_mutation 是净负且显著 — 方向E 落点② 证伪**。

## 结果 (test=106, eval_k=3, seeds=4)

| skill | mean_pass | per-seed | ΔvsS0 |
|---|---|---|---|
| s0 (initial.md) | 0.3011 | [0.3082, 0.3019, 0.2987, 0.2956] | — |
| **evolver 方向E pop (d2d46581)** | **0.4693** | [0.4843, 0.4906, 0.4434, 0.4591] | +0.1682 |
| direction-B evolver (07bc48c6, 基线) | 0.5393 | [0.547, 0.544, 0.538, 0.528] | — |
| skillopt (skill_v0002.md) | 0.5605 | [0.563, 0.531, 0.591, 0.557] | — |

## 统计判定 (unpaired Welch t, n=4 each)

- DirE vs direction-B: **gap=−6.99pp** t=−5.941 **p=0.0045** → 显著落后 ❌
- DirE vs skillopt: gap=−9.12pp t=−5.522 **p=0.0015** → 显著落后 ❌
- 4 seed 全部低于 direction-B 对应 seed (0.484<0.547, 0.491<0.544, 0.443<0.538, 0.459<0.528), 无一例超。

## 关键结论

1. **val +0.239 完全未迁移到 test, 反而净负**。方向E pop_mutation 在 val 侧跑出比 direction-B 更高的 val Δ (+0.239 vs 方向B 峰值 +0.319 以下), 但 test 侧 0.4693 比 direction-B 0.5393 低 7pp。这是本项目**最极端的一次 val→test 反向迁移** (通常 val 改善迁移 test 时损耗 ~0.85 ratio, 这里直接反向)。

2. **上条记忆 [[direction-e-pop-degeneration-20260909]] 的两层预测全部落空, 但方向相反**:
   - 预测"全程退化为单 lineage → 产物≈direction-B" → 错, pop-sample 实际激活 (iter3 采到历史 elite)。
   - 预测的 hidden 前提"若 pop 真有增量, 应 ≥ direction-B" → 也错, pop 确有增量但增量方向是**负的** (val 过度拟合 → test 崩)。

3. **根因 (已更正, 见 [[livemath-meta-option-trap-20260910]])**: 真正有害的不是 pop_mutation 采样机制本身, 而是被采样到的历史 elite 里藏着的 **Theorem-Level Precision 第 9 行毒段**——把"出现 meta-option 签名→倾向选 meta"写成启发式。livemath 数据实证: meta-option ("One of the remaining options is correct, but a stronger result can be proven") **出现必为正确答案 (100%)**, 但 meta 题占比 **train 66.7% > val 50.9% > test 43.4% 单调递减**。毒段在 meta 占比更高的 val 上"赢", 在 test 较低的 43.4% 上**过度触发**→val+0.239 转化为 test 净负。pop_mutation 只是**加速**毒段遗传 (ci 加权反复采样它为突变起点), 非毒源本身。

4. **与 direction-B 对比定位罪魁 (需修正归因)**: direction-B 与 DirE 的唯一变量确为"突变起点是否 pop-sample 历史 elite"，但罚的不是采样机制本身，而是**采到的那条 meta-偏向毒段被遗传放大**。direction-B (纯 incumbent leaf, 07bc48c6) 未采到毒段 → 0.5393; DirE (pop 历史 elite, d2d46581) iter3 采到 Theorem-Level Precision 毒段 δ+0.122→+0.235 PROMOTE → 0.4693。→ 真凶 = 毒段内容 (meta 偏向启发式)，pop_mutation 是放大器。

> **后续关系更正 (2026-09-11)**：本文件产物的毒段 (meta 倾向启发式)，已被 [[meta-rule-pathB-test-verdict-20260911]] 针对性修复——那个 run **关掉 `pop_mutation`（本文件定罪的放大器）、上 `meta_contrast`（纯对/错对照，让模型自主归纳不 ci 加权捞）**，把同一条 meta 规则用干净机制重学成确定性决策段 (0.4701)。两数值 p=0.96 无差不是"两条独立路径"，而是**同一 meta 规则在同一物理上限 (test +~0.17) 的汇合**。此前的「Path B 与方向E 殊途同归」旧读法已被证伪。

## 下一步 (建议)

- **放弃方向E 落点② pop_mutation (当前形式)**: 直接证伪, 不应再调 eps/uniform 修修补补 — 问题在"用 val 信号选突变起点"这个设计本身。
- **回归 direction-B (07bc48c6, 0.5393) 仍是当前最强公平 artifact**, 但北星 (追平 skillopt 0.5605) 未越。
- 转项候选: (a) 方向C ancestor-chain backtrack + select_donor_rescue (针对 floor-regression); (b) 方向D 多子集 Pareto 准入对抗 val→test 不迁移; (c) 修复 floor-regression (subset-promote floor=0.05 把 best_node 拉低, 输出 archive 峰值)。
- **更正后的大教训**: 根因是**毒段内容本身** ("meta 出现⇒倾向选" 启发式) 在 meta 占比 val>test 的分布漂移上被放大，而非采样机制。详见 [[livemath-meta-option-trap-20260910]]。

**关联**: [[direction-e-pop-degeneration-20260909]] (退化预测, 双层落空) / [[direction-e-pop-test-eval-launch-20260910]] (启动快照+基线) / [[northstar-direction-b-verdict-20260909]] (direction-B 0.5393 / skillopt 0.5605) / [[peak-diagnosis-val-to-test-negative-20260909]] (val→test 反向系列) / docs/phase2_issues.md