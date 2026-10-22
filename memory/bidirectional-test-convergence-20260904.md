---
name: bidirectional-test-convergence-20260904
description: 跨路径收敛发现 — 干净 S₀ 路径(theo_iter6)与 seeded 路径(evo_seed 72a95d13)独立长出同一"双向测试(bidirectional verification)区分强弱命题"启发式。evo_seed 16-seed mean=0.5590 vs skillopt 0.5531 同窗口 paired Δ+0.0059 CI95[−0.0104,+0.2218] 跨0(9/16 seed 超, seed12/seed3 单0.5912)=用户记得的"某 seed 超过"。evo_seed 只改 Theorem-Level Precision 段(+2 bullet: implication↔equivalence, existence↔uniqueness)。干净路径缺第二对→~0.5pp gap。可借鉴: facet_edit 靶此段 + merge-into-facet 算子 + 双向测试模板化推广。
metadata:
  type: project
---

# 跨路径收敛: 双向测试启发式 (2026-09-04)

## 触发
用户: "a中的skillopt 应该已经跑过很多遍了，记得之前有个方法是有一个seed超过了，再看看那个方法有什么特殊点或者优点可以强化借鉴的"。核查 159 个 per_*_seed*.json 找出 >0.5629 的 artifact。

## 数值全表 (16-seed 同窗口, skillopt cfae156d 是同条件基准)

| artifact | 起点 | 段数 | bytes | n-seed mean | vs skillopt(0.5531) |
|----------|------|------|-------|------|------|
| s0 | 干净 S₀ | 4 | 1018 | 0.3011 | — |
| **evo_seed (72a95d13)** | **SkillOpt+2iter** | 10 | 11241 | **0.5590** (16s) | **+0.0059 CI95[−0.0104,+0.0218] 跨0** |
| **evolver (15f21311)** | SkillOpt+v3 resume | 10 | 12350 | **0.5597** (4s) | +0.0066 |
| skillopt (cfae156d) | — | 10 | 9926 | 0.5531 (16s) | — |
| theo_iter6 (干净路径) | 干净 S₀ | 9 | 7608 | 0.5542 (4s) | +0.0011 |
| iter5 crossover (本次复测) | 干净 S₀ | 9 | 3876 | 0.5440 (2s) | −0.0091 |

- evo_seed **9/16 seed 超 skillopt**, seed12=0.5912 seed3=0.5912 seed6/4/2=0.5755+ — 用户记得的"某 seed 超过"即此。
- **种在 SkillOpt 上**非干净起点 → 论文只能作"seeded-evolver 互补"非"training-free 从零超越"。

## 核心: 两条路径独立收敛到同一启发式

**evo_seed vs SkillOpt diff = 只改 Theorem-Level Precision 段 (+1363B, 2 bullet)**:
1. **Distinguish implication from equivalence** — 对"若P则Q"/"P iff Q"双向测试: 定理证了 P⇒Q 吗? 也证了 Q⇒P 吗? 只单向→等价声称太强→选项 false, 选更弱蕴含或 meta-option。
2. **Distinguish existence from uniqueness** — 对"恰有N个"/"唯一解"测试: 定理证了存在吗? 也排除了其他? 只证存在未证唯一→唯一性声称太强。

v3_15f21311 = 同思路增强版 (Theorem-Level Precision +1796, Option Comparison +654)。

**干净路径 theo_iter6 独立涌现的 `Implication Direction & Biconditional Check` 段是同一个东西** — `bidirectional verification` (Forward 充分 / Backward 必要), 措辞几乎一致 ("An option is correct only if it passes both checks")。

→ **算法自洽性强证据**: 不靠 SkillOpt 注入, 干净 S₀ 也长出双向测试程序, 只是停在 1 轴(implication), evo_seed 多发展了第 2 轴(existence/uniqueness)。

## 可借鉴/强化 (不含手 craft, 只调算法)

差距 (干净 theo_iter6 0.5542 vs evo_seed 0.5590 的 ~0.5pp) ≈ "existence/uniqueness 这第 2 对双向测试 bullet"。
1. **facet_edit/expand 靶 Theorem-Level Precision 段** — 高价值进化段(两路径都改它)。当前 evolver 把涌现段长成独立新段, evo_seed 把同类启发式合并进现有段。可加 **"merge-into-existing-facet" 算子** (与 expand 长新段互补)。
2. **双向测试范式模板化推广** — implication↔equivalence, existence↔uniqueness 同模式 ("声称强度>定理证明强度")。让 diagnostic 主动发现第 3/4 对 (充分↔必要, 构造↔存在, 整体↔局部), 驱动 expand 长新双向测试 bullet 而非随机攻 meta-options。
3. **诊断刷新(靶点2)直接受益** — 诊断对当前 evolved best 重跑会看到 "implication 失败已被缓解但 existence/uniqueness 仍在失败" → 自然引向第 2 对。

## 诚实边界
evo_seed +0.0059 CI 跨0 **未统计显著超过 skillopt**, 只是 mean 平 + 个别 seed 超。干净定论仍需 theo_iter6 或更强干净 artifact 跑同窗口 skillopt arm。

## 关联
[[iter5-crossover-test4seed-20260904]] (本次复测, 干净 iter5=0.5440 未越) / [[three-breaks-facet-tree-20260902]] (干净 theo_iter6=0.5542 gap−0.0087, 涌现 Implication 段) / [[evolver-trajectory-analysis-20260903]] (6 靶点, 靶点2=诊断刷新) / [[facet-tree-capability-ceiling-20260827]] (skillopt 优势=6 展开型 section)
