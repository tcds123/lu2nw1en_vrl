---
name: facet-tree-no-cumulative-20260825
description: Facet-Tree 累积进化从未发生 — 跨轴改进永不叠加, best 永远是"单轴最优+S0其余3段"; archive 实证 9 叶除本轴段外其余全=S0; 根因=_run_facet_tree 只做单轴内累积, 交叉形同虚设; 用户从"三段没变"反推算法缺陷正确
metadata:
  type: project
---

# Facet-Tree 累积进化从未发生 (结构性设计缺陷) — 2026-08-25

## 触发
用户质疑: "其他三个段落完全相同也有很大的问题, 进化算法一点没变化, 说明算法设计中有更多 bug, 没真正将理论执行下去"。核查后**确认正确**。

## 实证 (153302 archive_final.json, 9 叶)
每个叶除自己轴段外, 其余 3 段**全部 = S0 原文逐字未动**, 无一例外:
| facet | parent | Opt-Comp | Theo | Hypo | Final | val_delta |
|-------|--------|----------|------|------|-------|-----------|
| Opt-Comp (gen1) 368e63 | Opt-Comp | 改 | S0 | S0 | S0 | +0.156 |
| Theorem (gen1) e6e751 | Theorem | S0 | 改 | S0 | S0 | +0.200 |
| Hypotheses (gen1) 79ccd2 | Hypotheses | S0 | S0 | 改 | S0 | +0.194 |
| Opt-Comp (gen2) 1dbbb8 | Opt-Comp | 改 | S0 | S0 | S0 | +0.344 |
| Theorem (gen2) e25d6c | Theorem | S0 | 改 | S0 | S0 | +0.106 |
| Hypotheses (gen2) 4bef60 | Hypotheses | S0 | S0 | 改 | S0 | +0.172 |

- 3 轴各自进化出改进 (Opt-Comp +0.344, Theorem +0.200, Hypotheses +0.194), **互不叠加**。
- 全局 best = S0 全文只换一个轴段 (1dbbb821 = S0 + Opt-Comp 改)。**best 永远是"单轴最优+S0其余3段"**。

## 根因 (skill_evolver_math.py `_run_facet_tree` L941-1043)
1. **L942**: 4 分支叶初始化都 = `init_skill`(S0), 各持 S0 的 `root_slots` 副本。
2. **L1006 `facet_edit_guard(leaf.slots, ax, child_text)`**: child 只改 `ax` 段, 其余段继承当前 `leaf.slots`。
3. **L1031 精英晋级**: child 成新活跃叶, 其非 ax 段 = 父叶非 ax 段 = ... 一路追到 root = S0。
- **算法只做"单轴内累积" (Opt-Comp 叶每代只动 Opt-Comp 段), 不做"跨轴累积"**。Opt-Comp 叶的 Theorem 段永远 S0; Theorem 轴独立进化出更好 Theorem 段 (+0.200) 但**永远无法流回 Opt-Comp 叶**。

## 交叉算子本该补救但 Stage 0 形同虚设
- 交叉 (L1066 `recombine_facets`) 理论是累积载体 (A 轴改进段 + B 轴改进段拼接)。
- 但: ① 每代最多 1 次 (L1047); ② 只取每轴 CI 最高 donor 随机拼 2 轴; ③ 拼出 child 不超 best 就不成 best (L1082)。
- 153302 交叉被 guard 拒; 183940 交叉 child 0.189 < best 0.194 未胜出。**累积通道 Stage 0 基本没生效**。

## 影响 (推翻上轮"选择方差"归因)
- 上轮"退步归因选择方差"仍不准确。深层根因 = **累积进化机制 (交叉) 形同虚设, 4 轴独立改进永不叠加, best 必然"单轴最优+S0其余"**。
- 后果: ① val 提升被天花板压住 (单轴改进有上限, 叠加才突破); ② test 波动大 (单轴变体迁移性不稳, 缺其他轴支撑)。
- ⚠️ "其余3段=S0"不是巧合是**结构必然**。判据线 0.5464 难突破的真因可能在此: 单轴改进天花板有限, 4 轴叠加才可能逼近 skillopt; 当前架构 best 永远单轴, 或被结构卡住而非搜索不够。

## 解决方案候选 (未实施)
1. **跨轴叠加 best (核心)**: 每代末把全局 best 各段换成"该段历史最佳叶的段" (Opt-Comp 段取 Opt-Comp 轴 delta 最高叶...), 组装"全轴最优叠加 skill"重新 eval, 超则替换。直接打通跨轴累积。
2. **活跃叶携带全局 best 的非本轴段**: 精英晋级时 child 非 ax 段不继承父叶(=S0), 而继承当前全局 best 对应段。Theorem 轴改进流进 Opt-Comp 叶。需评估是否破坏 facet_edit_guard 非目标段不变约束。
3. **放开交叉触发频次**: Stage 0 每代 1 次太少, 提到每代每轴对 1 次; 准入放宽 (不必超 best, ci>0 即成 best 候选)。
4. **验证判据**: 修复后重跑 2 代, 检查 best 各段是否含 ≥2 轴改进 (而非全 S0)。

## 教训
- **"独立进化≠协同进化"**: 4 轴并行各改各段 ≠ 累积进化。必须显式打通跨轴改进流, 否则 multi-facet 退化成 4 个互不交互单轴搜索。
- **看结果先看结构**: "其余三段没变"看似细节, 实为结构性 bug 指纹。用户从"结果没变化"反推"算法没执行理论"是对的。
- 关联 [[facet-tree-retest-result-20260825]] (上轮"选择方差"归因需向下修正为"累积通道失效"), [[facet-tree-crossover-fix-20260824]] (交叉 bug 修对了但没解决累积失效)。
