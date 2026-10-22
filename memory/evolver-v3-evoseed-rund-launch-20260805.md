---
name: evolver-v3-evoseed-rund-launch-20260805
description: Run D (evo_seed + M3-gate) 启动, 种 SkillOpt-best 0.546 上 evolve, 冲击超 0.559
metadata:
  type: project
---

# Run D: evo_seed + M3-gate (2026-08-05 05:24 启动)

## 动机
Run A/B/C (从 S0/v2 起) 均未破 ~0.48 test 天花板, Run C 的 M3-gate 已验证机制 (iter3 GATE-ACCEPT, 见 [[evolver-v3-m3gate-launch-20260805]]).
转向高 floor 路线: 种 SkillOpt-best (test 0.546) 上 evolve, 目标超 evo_seed 0.559.

## evo_seed 历史 (0.559) 轨迹分析 (run_20260803_184657)
| iter | val delta | CI | skill_len |
|------|-----------|----|-----------|
| baseline | +0.324 | +0.236 | 9926 |
| 1 | +0.380 | +0.293 | 10452 |
| 2 | +0.390 | +0.292 | 11133 |

- **evo_seed 只跑 2 iter, consolidate 从未触发** (consolidate_every=3). Run D (6-iter) 是该 seed 上首次 consolidate.
- iter1 增 +0.056, iter2 增 +0.010 -> 快速 plateau.
- val-test 传递率低: (0.559-0.546)/(0.390-0.324) = 0.013/0.066 = **0.20** (vs S0 evolver 0.65). 即使 val 推高, test 增益小.
- SkillOpt-best skill 9926 chars (远超 4000), 膨胀. **M3-gate 的 consolidate 可压缩 bloat 不损质** -> 潜在增益点.

## 启动配置 (Run D)
```
python rl/train/verl1/skill_evolver_math.py \
  --init_skill /data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md \
  --iters 6 --eval_k 3 --train_n 35 --K 4 --patience 3 \
  --gate_consolidate --cache_failures \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_v3_evoseed_m3gate
```
- seed: SkillOpt-best (9821 chars after clean_skill, test 0.546, val delta +0.324)
- flags: M3-gate ON + D1 cache ON, 其余关
- s0_cache 复用 (delta 仍 vs S0 固定快照, 与 init_skill 无关)
- early-stop: patience*2=6 (防 val-overfit, 如 10-iter S0 退化 0.478->0.438)

## 预期与判定
- **乐观**: iter3 consolidate 压缩 9821-char bloat + 更多 iter 攀升 -> val +0.40~0.42 -> test 0.56~0.57 (超 0.559, 新最佳)
- **悲观**: plateau iter2 (+0.390) 如 evo_seed, low transfer -> test ~0.555 (无提升)
- 判定: test > 0.559 = 有效 (新最佳); test ≤ 0.559 = evo_seed 已近天花板, 需 Run E (per-topic 库)

## Run D 进展 (2026-08-05)

### iter1 STALL - k=3 eval 噪声问题暴露 (06:25)
- baseline: pass=0.610, **delta=+0.385** (vs evo_seed 历史 +0.324, **同 skill 噪声高 +0.061!**)
- 4 候选: cand0 +0.347, cand1 +0.362, cand2 +0.329, cand3 +0.347 (全 < +0.385 bar)
- **KEEP PREV (no_improve=1)**: 无候选过 bar.
- **根因分析**: k=3 eval 噪声 ±0.06. baseline 噪声高 (+0.385) -> 等质量候选 (~+0.36, 真值≈+0.385) 被噪声压低, 无法过 bar. evo_seed 当时 baseline 噪声低 (+0.324) -> 轻松爬到 +0.390. **同 skill, 不同噪声, 一个爬一个 stall.**
- **方法论发现**: evolver 能否攀升部分由 baseline 噪声运气决定, 非纯算法质量. s0_cache 是 k=5 但候选 eval_k=3, delta 有 k 错配 + k=3 采样噪声.
- iter3 consolidate+gate 仍会触发 (skill 未变, 9821>4000) - 这是 Run D 剩余关键数据: 压缩 bloated SkillOpt-best 是否帮 test?

### 待观察
- iter2-6: 大概率继续 stall (噪声高 bar). 除非某轮候选碰巧 eval 高.
- iter3 consolidate: 压缩 9821-char skill, gate 验证. 若 GATE-ACCEPT -> final skill = 压缩版, test 可能 ≠ 0.546.
- 最终 test: 若 stall 全程, final skill ≈ SkillOpt-best (或压缩版), test ~0.546±. 大概率 ≤ 0.559.

## 噪声问题的可能修复 (未实现, 待 Run D 后决定)
1. **eval_k=5**: 候选 eval 提到 k=5 (匹配 s0_cache), 噪声 ±0.03. 干净 delta. 代价: 每 iter ~55min (vs 30). 最原则性修复.
2. **baseline refresh**: 每轮 re-eval 当前 best 作 fresh bar (同轮噪声可比). 代价: +1 eval/轮. 但可能 thrash.
3. **accept margin**: 接受 delta > best - margin (容忍噪声) + CI>0. 风险: 接受等质量候选致 drift.

## Run D 终止 (2026-08-05 06:29, iter2 处)
iter1 STALL (no_improve=1, 噪声高 bar), iter2 在跑时终止. 原因: 噪声高 baseline (+0.385) 致 stall, 剩余 iter2-6 + test 价值低 (大概率全 stall, test ~0.546). iter3 consolidate 数据非必要 (Run C 已验证 gate). **终止, API 让给 Run F (eval_k=5 测噪声假说).**

## Follow-up: Run F (eval_k=5 evo_seed + M3-gate, 2026-08-05 06:29 启动)
**假说**: evo_seed plateau (+0.390) 是 k=3 噪声阻塞 (候选真值≈+0.39 但 eval +0.37 被 bar 拒), 还是真天花板?
**设计**: 同 Run D seed (SkillOpt-best) 但 `--eval_k 5` (匹配 s0_cache k=5, 干净 delta, 噪声 ±0.046 vs ±0.06). 4-iter + M3-gate.
```
python rl/train/verl1/skill_evolver_math.py \
  --init_skill /data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md \
  --iters 4 --eval_k 5 --train_n 35 --K 4 --patience 3 \
  --gate_consolidate --cache_failures \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_v3_evoseed_k5
```
- 预期 baseline: ~+0.324 (稳定, 非 +0.385 噪声高)
- 判定: 若攀升超 +0.390 (evo_seed plateau) -> plateau 是噪声, eval_k=5 解锁; 若仍 +0.390 -> 真天花板, 需 Run E (per-topic)
- watcher PID 126710, evolver 126728, log `skill_evolver_math_v3_evoseed_k5.log`
- 时间线: 4 iter @ eval_k=5 ~4.5h -> evolver ~11:00, test ~11:45

## Run D/F 对照表 (evo_seed 路线)
| run | eval_k | baseline | iter1 | ... | test |
|-----|--------|----------|-------|-----|------|
| evo_seed (历史) | 3 | +0.324 | +0.380 | +0.390(plateau) | 0.559 |
| Run D | 3 | +0.385(噪声高) | STALL | (终止) | - |
| Run F | 5 | **+0.341(稳定!)** | ? | ? | ? |

### Run F baseline 验证 (06:47): eval_k=5 降噪成立 ✅
- Run F baseline (k=5): delta=+0.341, CI=+0.251, pass=0.566
- vs evo_seed (k=3): +0.324 -> 差 +0.017 (k=5 稳定, 接近真值)
- vs Run D (k=3): +0.385 -> 差 -0.044 (Run D 是噪声高 outlier)
- **结论**: k=3 噪声 ±0.06 致 baseline 在 +0.324~+0.385 抖动; k=5 降到 ±0.046, baseline 稳定在真值附近 (+0.341). 候选现可公平比较, 不被噪声高 bar 阻塞.
### Run F iter1 ACCEPT (08:05): k=5 攀升成立 ✅
- iter1 winner cand1: delta=+0.375 (CI=+0.293, pass=0.600, len=10424), baseline +0.341 -> **+0.375 CLIMB**
- cand2 +0.375 (并列), cand0 +0.332, cand3 +0.335 (均 < winner)
- vs evo_seed 历史 iter1 +0.380 -> 几乎一致 (k=5 复现了 evo_seed 的攀升轨迹)
- vs Run D k=3 iter1 STALL (baseline +0.385 噪声高, 无法超) -> **k=5 降噪解除了 Run D 的阻塞**
- iter2 启动 (focus=Existence vs uniqueness), ~09:25 出结果.
- **关键问题**: iter2/3 是否突破 +0.390 (evo_seed plateau) 还是持平?

## guarded chained watcher (PID 116905, /tmp/chain_rund_evoseed.sh)
evolver 完成 -> 自动跑 test=106 (s0+evolver, seeds=2, k=3) -> 打印结果.
log: `rl/train/verl1/outputs/skill_evolver_math_v3_evoseed_m3gate.log`

## 时间线
- 05:24 启动, baseline ~7min, 6 iter ~3-4h -> evolver ~09:00 完成
- test ~45min -> ~09:45 出结果

## 对照
- S0: 0.305 | 主结果: 0.478 | RunA: 0.481 | RunB: 0.4591 | RunC: 终止(iter3 gate验证, ~0.45估)
- SkillOpt-best: 0.546 | evo_seed (2-iter): 0.559 | **Run D 目标: > 0.559**

详见 [[evolver-v3-m3gate-launch-20260805]] (Run C + M3-gate 验证) 和 [[algorithm-future-optimization-directions-20260805]]。
