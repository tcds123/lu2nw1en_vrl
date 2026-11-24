# 北星判定: 方案1 (theo_iter6) 未越过 (2026-09-09)

## 判定结果 (test=106, 4-seed, k=3, paired bootstrap B=10000)

| 臂 | mean | ΔvsS0 | vs skillopt |
|----|------|------|------|
| s0 (initial.md) | 0.2956 | — | −0.2649 |
| evolver (iter3 overlay, 3a8c2343) | **0.5440** | +0.2484 | **−0.0165** |
| skillopt (skill_v0002.md) | 0.5605 | +0.2649 | — |

- **evolver − skillopt = −0.0165**, paired bootstrap per-qid CI95 = **[−0.0597, +0.0259]**, **下界 < 0 → 含 0 → 未越过北星 ❌**
- 胜/平/负 (per-qid 4-seed均值) = 32/32/42
- evolver 对 s0 强: +0.2484, CI95=[+0.1832,+0.3153] 全正 ✅ (进化产物本身有效, 只是没超 skillopt)

## 关键: evolver 比 init theo_iter6 还低

- theo_iter6 test = 0.5542 (4-seed, [[northstar-test-baseline-20260908]])
- evolver (iter3 overlay) test = 0.5440 → **比 init 低 1.02pp**
- 即 overlay 产物的 select delta 降 (init +0.4131 → best +0.3850, −0.0281) **确实迁移到了 test** (−0.0102)
- subset-promote 救援 (iter2) + overlay 回升 (iter3) 产出的 2 新双向轴 (Existence↔Uniqueness, Quantifier coordinating) **没有补上 skillopt 满分而 evolver 丢分的题**。

## 根因: init 天花板 (theo_iter6 已在 skill 空间上限)

1. **init 过强**: theo_iter6 test=0.5542 已达 skill 空间 ~+0.26 天花板 (s0→skillopt +0.2649, s0→theo_iter6 +0.2586 收敛)。从 theo_iter6 起步, 候选 select delta 全 < init +0.4131, 进化空间被压在 init 之下。
2. **select→test 迁移坐实**: select delta 降 → test 也降。subset-promote floor 让"略低于 init 但子集正向"候选夺权, 但这些候选在 test 上也弱于 init (不是泛化更好, 是真的更弱)。
3. **新轴未命中目标题**: evolver 丢分最重题 (202602:50 ev=0.417 sk=1.0, 202511:28 ev=0.083 sk=0.667, 202512:22 ev=0.333 sk=0.917) 与 theo_iter6 丢分题 (202512:39, 202602:3) 部分重叠, 新增的 Existence/Quantifier 段没解决它们 → directed donor (按失败类型选) 是对的方向, 但 theo_iter6 archive 内无覆盖这些题的 donor。

## 决策: 转方向 B + 改用 s0 起步 (FAIR 条件)

**核心洞察**: 北星判据要求 FAIR 条件 = true S₀ start, zero SkillOpt nesting。theo_iter6 是 warm-start (非 S₀), 已消费大部分 headroom, 留给进化超过 skillopt 的空间≈0。**s0 起步才有完整 +0.265 headroom (skillopt 已证明可达)。**

方向 B (plan swift-booping-balloon):
- **多 parent crossover** (`recombine_facets_multi`): 3+ donor 段同时嫁接进 base, 一次 coherence pass。比 2-parent 覆盖更多失败类型。
- **定向 donor 选择** (`select_donors_directed`): 按 section body 的 `_TYPE_COVERAGE_SIG` 与当前 `counts` (失败分布) 匹配度排序, 而非纯 val_ci。让 donor 对准当前失败类型。
- **从 s0 起步**: 给进化完整 headroom, 这是北星 FAIR 条件本就要求的。

## 缓存可复用
- s0_68571cb2 + skillopt_cfae156d (4seed, 已在本 run out_dir) — 方向 B 从 s0 起步, s0 是 init 不是对照, 但 skillopt 对照仍复用。
- evolver 3a8c2343 (本 run 产物, 作 theo_iter6 warm-start 基线存档, 不再用于方向 B)。

## 教训
1. **warm-start 不是 FAIR**: theo_iter6 消费了 headroom, 从它起步测不出"进化能否超 skillopt"。北星须从 s0 起步。
2. **select delta 降 = test 降** (本 run 坐实): subset-promote floor 救援的"略低 delta 子集正向"候选, 在 test 上也弱, 不是泛化优势。floor=0.05 在 init 过强场景是净负。
3. **per-type strengthen 机制成功但未夺 best**: Task #7 修复后 strengthen 真 fire (iter3), 但 init 过强下 overlay 更强, strengthen 未超。其价值需在 s0 起步 (有 headroom) 场景验证。

**关联**: [[pertype-run-complete-20260909]] (run 完成链路) / [[northstar-test-baseline-20260908]] (theo_iter6 基线) / [[swift-booping-balloon plan]] (方向 B/C 设计) / `docs/phase2_issues.md`
