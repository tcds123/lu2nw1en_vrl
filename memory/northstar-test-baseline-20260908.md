# 北星基线: test=106 4-seed theo_iter6 vs skillopt (2026-09-08)

## 确认数据 (4-seed 均值, test=106, k=3)

| 臂 | test pass@3 | vs s0 |
|----|------|------|
| s0 (initial.md) | 0.2956 | — |
| skillopt (skill_v0002.md) | 0.5605 | +0.2649 |
| evolver = theo_iter6 (warm-start init) | 0.5542 | +0.2586 |

- **evolver − skillopt = −0.0063**, bootstrap CI95 = [−0.0417, +0.0322], **下界 < 0 → 含 0, 未越过北星**。
- 胜/负/平 = 32/41/33 (evolver 略负 skillopt)。

## 关键结论

1. **theo_iter6 (当前 run 的 init_skill) 在 test 上本就低于 skillopt (−0.63pp)**。当前 facet-tree run 从 theo_iter6 起步, select 口径所有候选 (iter1 expand +0.380, iter2 xover +0.366, iter2 strengthen +0.357, iter3 全 LOST/KEEP) 全低于 init select +0.413 → 产物大概率 ≤ theo_iter6。
2. **skill 优化天花板信号**: s0→skillopt 与 s0→theo_iter6 的 test 增益几乎相同 (+0.265 vs +0.258)。两种独立优化路径都收敛到 ~+0.26, 暗示这是当前 skill 空间的增益上限, 非算法问题。
3. **evolver vs s0 在 test 是强正 (+0.2586, 仅 12/106 题<0)** → init 在 test 上对 s0 强。问题不在 init 弱, 而在 init 未能超越 skillopt 的 +0.265 (差 0.63pp)。

## evolver<skillopt 最差题 (test seed 平均)

- 202512:39: ev=0.250 sk=0.833 (−0.583)
- 202602:3: ev=0.333 sk=0.917 (−0.583)
- 202602:48: ev=0.500 sk=0.917 (−0.417)
- 202512:1: ev=0.500 sk=0.833 (−0.333)
- 202511:25/202512:29/202602:19: ev=0.75 sk=1.0 (−0.25)

这些题 skillopt 满分而 theo_iter6 大幅丢分 → skillopt 有 theo_iter6 缺的覆盖。xover 产物 (含 ## Coordinating Quantifiers 第2双向轴) 若恰好补上这些题可能反超。

## 缓存可复用

`outputs/skill_evolver_math_3break/run_20260902_205708/4seed_test_theo_iter6/`:
- per_s0_68571cb2_seed{1..4}.json (s0 hash 匹配, 可复用)
- per_skillopt_cfae156d_seed{1..4}.json (skillopt hash 匹配, 可复用)
- per_evolver_671135ff_seed{1..4}.json (theo_iter6 旧 hash, xover 49a1f865 不匹配需重跑)

## 决策影响

北星判据 = evolver vs skillopt CI95 下界 > 0。当前 init 起点已 −0.0063, run 产物若 ≤ init 则必然失败。两条路:
- **路 A (当前)**: 等 run 完成, 若 xover 产物 (subset-promote 救援挑的泛化候选) test 反超 skillopt → 方案1 有效; 否则记录。
- **路 B**: skill 空间天花板, 需换更弱的 init (如 s0 +0.019 给进化空间) 或结构升级 (方向 B 多 parent crossover 注入 skillopt 缺的覆盖)。

**关联**: [[pertype-strengthen-run-20260908]] (run 监控) / [[swift-booping-balloon plan]] (方向 B/C 设计) / `docs/phase2_issues.md`
