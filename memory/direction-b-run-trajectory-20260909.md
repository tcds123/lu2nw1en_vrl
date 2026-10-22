# 方向 B run 轨迹 + 产物溯源 (2026-09-09)

## run 概况
- **run 目录**: `rl/train/verl1/outputs/skill_evolver_math/run_20260909_024419/`
- **init = S₀ (initial.md, hash 68571cb2a6)** — 真·FAIR 起步, 非 theo_iter6 warm-start。baseline_val: pass=0.2300, delta=+0.0047, ci=−0.0451。
- **参数** (方向 B): `--facet_tree --expand ... --strengthen --strengthen_per_type --xo_multiparent 3 --xo_donor_directed --xo_allow_merge --donor_topk 2 --subset_admit ...` (4 iters)
- **产物**: `best_skill_evolved.md`, 5658 chars, 6 段, eval口径 hash=**07bc48c6**

## best 演化链 (select 口径, probe=train+val=71)
| iter | best_delta | best_ci | n_sec | became_best 事件 |
|------|-----------|---------|-------|------------------|
| init | +0.005 | −0.045 | 4 | baseline (S₀) |
| 1 | +0.005 | −0.045 | 4 | overlay/expand 均 rejected (ci≤0) |
| 2 | **+0.319** | +0.247 | 5 | expand Existence admitted, became_best=True ← 高水位 |
| 3 | +0.319 | +0.253 | 5 | 🔀 MP xover(n=3,directed) admitted(+0.305); overlay admitted(+0.286,7axes); expand strengthen became_best=True(平 +0.319) |
| 4 | +0.286 | +0.215 | 5 | 🔀 MP xover(n=3,directed) admitted(+0.286); overlay/expand 均 became_best=False |

## 🔀 多 parent crossover 机制验证成功
- iter3 + iter4 均 fire: `n_parents=3, directed=True, admitted=True`。
- 非 dead code: 定向 donor (按 failure-type coverage 匹配) 选了 3 个不同 section 段同时嫁接。
- 但两次 MP xover 的 delta (+0.305 / +0.286) 均 **未超 iter2 expand 的高水位 +0.319**。

## ⚠️ 关键发现: floor-regression 再现 (同方案1 病根)
- iter2/3 峰值 best_val_delta = **+0.3192** (hash 539300b5 iter2 / cb901c5c iter3)。
- iter4 MP xover delta=+0.2864, **低于峰值 0.033**, 但 ≥ floor (best−0.05=+0.269) → admit + promote。
- **最终 best_skill_evolved.md = iter4 产物 (07bc48c6, delta +0.286), 非 iter2/3 峰值 (+0.319)**。
- 即 subset-promote floor 又一次把 global best_node 从峰值拉低到 iter4 的 floor-rescue 产物。
- **若 07bc48c6 test 弱**, 首要嫌疑 = floor-regression 选了 iter4 而非峰值 iter2/3。
  - 补救: 直接评测峰值 skill (iter3 cb901c5c 或 iter2 539300b5), 看峰值 test 是否更强。

## 产物结构 (6 段)
- Theorem-Level Precision / Hypotheses / Existence vs Uniqueness Distinction / Option Comparison / Final Answer / (Coordinating Quantifiers?)
- 从 S₀ 的 4 段 (1017 chars) 增到 6 段 (5658 chars), 新增 Existence vs Uniqueness (iter2 expand, 双向轴)。

## 产物溯源
| 文件 | eval_hash | chars | val_delta |
|------|----------|-------|-----------|
| best_skill_iter1.md | 68571cb2 | 1017 | +0.005 (=S₀) |
| best_skill_iter2.md | 539300b5 | 2916 | +0.319 (峰值) |
| best_skill_iter3.md | cb901c5c | 3361 | +0.319 (峰值, 同值紧ci) |
| best_skill_iter4.md | 07bc48c6 | 5658 | +0.286 (最终产物, floor-rescue) |
| best_skill_evolved.md | 07bc48c6 | 5658 | +0.286 (= iter4) |

## 待跑/进行中
test=106 4-seed CI eval (PID 42007, ~52min): evolver(07bc48c6) vs skillopt(cfae156d), s0+skillopt 缓存复用。
- 北星判据: paired bootstrap per-qid CI95 下界 > 0 = 成功。
- 若 ≤0: 先验峰值 skill (cb901c5c/539300b5) test; 若峰值仍弱 → 方向 C (ancestor-chain backtrack)。

## 教训
1. **floor-regression 是系统性病根**: subset-promote floor=0.05 在 S₀ 起步也复现了 — iter4 的 MP xover (+0.286) 被 promote 覆盖了 iter2/3 峰值 (+0.319)。方案1 和方向 B 同病。下一步应考虑 floor=0 (只允许严格超 best 才能 promote) 或干脆输出 archive 峰值而非 best_node。
2. **MP xover 机制通但未夺峰值**: directed donor + 3-parent 嫁接真 fire 且 admitted, 但 delta 没超 iter2 单 expand。说明在当前失败分布下, 单新轴 (Existence) 的覆盖价值 > 多段嫁接组合。
3. **FAIR 起步确认**: init hash 68571cb2a6 = S₀, baseline delta=+0.005 ≈ 0, headroom 完整 (skillopt 已证 +0.265 可达)。

**关联**: [[direction-b-impl-complete-20260908]] (代码实现) / [[northstar-pertype-verdict-20260909]] (方案1 未越→转B) / [[pertype-run-complete-20260909]] (方案1 同病 floor-regression) / `docs/phase2_issues.md`
