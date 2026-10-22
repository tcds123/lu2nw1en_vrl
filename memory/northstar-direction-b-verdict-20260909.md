# 北星判定: 方向 B (S₀ FAIR 起步) 未越过 (2026-09-09)

## 判定结果 (test=106, 4-seed, k=3, paired bootstrap B=20000)

| 臂 | mean | per-seed | ΔvsS0 |
|----|------|----------|-------|
| s0 (initial.md) | 0.2956 | [0.277,0.318,0.293,0.296] | — |
| evolver (iter4, 07bc48c6) | **0.5393** | [0.547,0.544,0.538,0.528] | +0.2437 |
| skillopt (skill_v0002.md) | 0.5605 | [0.563,0.531,0.591,0.557] | +0.2649 |

- **evolver − skillopt = −0.0212**
- 判据1 seed-level paired t (n=4): CI95 = [−0.0649, +0.0225], 正/负=1/3 → **跨0, 不可区分 🟡**
- 判据2 per-qid paired bootstrap (主, B=20000): CI95 = **[−0.0700, +0.0275]**, 正/负/零=36/41/29 → **跨0, 不可区分 🟡**
- **北星判据 (CI下界>0) = 未越过 ❌** (下界 −0.0700 < 0)
- **非 CI-gated defeat** (上界 +0.0275 > 0, 不能判 evolver 显著落后)
- evolver vs s0 = +0.2437, 4 seed 全正 [0.27,0.23,0.25,0.23] → 进化产物本身强且显著 (延续历史 ~+0.24)

## 与方案1 对比 (关键: FAIR 起步也没救)

| 实验 | 起步 | evolver test | Δvs skillopt | bootstrap CI95 | 判定 |
|------|------|-------------|-------------|----------------|------|
| 方案1 (3a8c2343) | theo_iter6 warm-start | 0.5440 | −0.0165 | [−0.0597,+0.0259] | 未越 ❌ |
| 方向B (07bc48c6) | **S₀ FAIR (68571cb2)** | 0.5393 | −0.0212 | [−0.0700,+0.0275] | 未越 ❌ |

- 方向 B 是真·FAIR (init hash 68571cb2=S₀, baseline delta≈0, 完整 +0.265 headroom), 但结果**略差于**方案1 的 warm-start。
- 推翻"theo_iter6 warm-start 消费 headroom 是主因"的假设 — 给了完整 headroom (S₀ 起步), 仍没超 skillopt。

## 根因: floor-regression (而非 warm-start)

1. **subset-promote floor=0.05 又把 best_node 拉低**: iter2/3 峰值 val_delta=+0.3192, iter4 MP xover +0.2864 (<峰值0.033 但 ≥ floor=+0.269) → admit+promote → 最终产物=iter4 (+0.286), 非峰值 (+0.319)。
2. **−0.033 select gap 迁移到 test**: 方案1 同病 (select 降 → test 降)。方向 B 的 −0.0212 vs 方案1 −0.0165, 差 ~0.005, 与 iter4 比 iter3 多降的 select 量级一致。
3. **MP xover 机制通但未夺峰值**: directed donor + 3-parent 嫁接真 fire 且 admitted (iter3/4), 但 delta (+0.305/+0.286) 没超 iter2 单 expand (+0.319)。当前失败分布下, 单新轴 (Existence) 覆盖价值 > 多段嫁接组合。

## 决策: 先验峰值 skill (floor-regression 嫌疑), 再定方向 C

**下一步 (已启动)**: 评测峰值 skill `best_skill_iter3.md` (eval_hash cb901c5c, val +0.3192, 3361 chars) — 直接验证 floor-regression 假设。
- 若峰值 test 明显 > iter4 (07bc48c6 的 0.5393) → floor-regression 是主因 → 修 floor (=0 或输出 archive 峰值) 后重跑。
- 若峰值 test ≈ iter4 → 峰值 select 优势不迁移, 转**方向 C** (ancestor-chain backtrack + select_donor_rescue, 已设计)。
- 预估: 峰值 select +0.319 vs iter4 +0.286, 按 transfer ratio ~0.85, 峰值 test ≈ +0.271(vs s0) → vs skillopt +0.265 → Δ≈+0.006 (近 parity, CI 仍跨0 但更居中)。

## 教训
1. **warm-start 非主因被推翻**: S₀ FAIR 起步给完整 headroom 仍没超, 真瓶颈是 floor-regression + select→test 迁移损耗, 非 headroom 不足。
2. **floor-regression 是系统性病根 (第3次出现)**: 8/28 expand + 9/9 方案1 + 9/9 方向B, 三次都被 subset-promote floor 把 best_node 从峰值拉低。必须修: floor=0 (只允许严格超 best 才能 promote) 或直接输出 archive 峰值而非 best_node。
3. **守 CI 纪律**: 两判据都跨0 = 统计无定论, 不报"evolver 落后" (上界>0), 也不报"追平" (下界<0)。点估计 −0.0212 是单次采样, 非真值。

**关联**: [[direction-b-run-trajectory-20260909]] (run 轨迹+floor-regression 发现) / [[direction-b-impl-complete-20260908]] (代码) / [[northstar-pertype-verdict-20260909]] (方案1 未越) / `docs/phase2_issues.md` (floor-regression 系统性 bug)
