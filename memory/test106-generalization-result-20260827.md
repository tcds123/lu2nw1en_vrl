---
name: test106-generalization-result-20260827
description: test=106 held-out 泛化评测完成 (8-27). Step1 best 8238b5e8f2 vs s0, eval_k=3, seeds=4, 同口径(s0 vs evolver 同 test=106). evolver 0.5354 vs s0 0.3011, Δ=+0.2343. per-qid paired bootstrap CI=[+0.1635,+0.3066] 下界>0→泛化增益方向成立(正). 4 seed delta 全正且稳[0.220-0.258]. CI带纪律: 只下方向, 真值量级(0.16-0.31区间内)分不出
metadata:
  type: project
---

# test=106 held-out 泛化评测结果 (2026-08-27 完成)

## 结论 (CI 带方向判据)
- **泛化增益方向成立 (正)**: per-qid paired bootstrap 95% CI = **[+0.1635, +0.3066]**, CI_lower > 0
- 这是本次评测唯一可下的结论: Step1 best skill (8238b5e8f2) 在 held-out test=106 上**方向性地优于 s0**
- ⚠️ 真值量级分不出: 真增益 95% 落 [+0.16, +0.31], 是 +0.17 还是 +0.30 无法分辨, 都兼容. mean_diff +0.2343 是单次采样实现值非"去噪真值" (守 [[ci-band-not-point-est-20260826]])

## 数字 (同口径, k=3, test=106, seeds=4)
| skill | mean_pass | per-seed passes |
|---|---|---|
| s0 (68571cb2) | 0.3011 | [0.3082, 0.3019, 0.2987, 0.2956] |
| evolver (8238b5e8) | 0.5354 | [0.5377, 0.5597, 0.5189, 0.5252] |
| skillopt (cfae156d) | 0.5464 | [0.5660, 0.5409, 0.5283, 0.5503] |
| ΔvsS0 (evolver) | **+0.2343** | [0.2296, 0.2579, 0.2201, 0.2296] |
| evolver vs skillopt | **-0.0110** | 跨0, 不可区分 (见下) |

## 🆕 evolver vs skillopt 上界 — 已追平到不可区分 (8-27 增补, 用户问"还没超skillopt吗")
- **evolver 0.5354 vs skillopt 0.5464, 点估计 evolver 微低 -0.0110, 但 per-qid paired bootstrap 95% CI=[-0.0417, +0.0189] 跨0 → 统计上无法区分**
- per-qid 配对: 106 题 **33 正 / 36 负 / 37 零** → 几乎对半, 两 skill 题层面表现极接近
- 对照 (验证 skillopt 是有效上界): skillopt vs s0 = +0.2453, CI=[+0.1682,+0.3223], 下界>0 ✅ (与 evolver vs s0 的 +0.2343 几乎重合 → evolver 已达 skillopt 量级)
- **判据 gap 质变**: Stage0 那次 evolver=0.4953 → 判据 gap 0.0511(5.88 SE, 超稳落后, 见 [[a2-power-analysis-20260824]]); 本次 evolver=0.5354 → gap=0.5464−0.5354=0.0110(≈1.3 SE, 跨0). 从"超稳落后"到"追平不可区分"
- ⚠️ 口径注 (诚实): skillopt 缓存是**单独 head-to-head 跑的** (`eval_skillopt_headtohead.log`), seed 组与 evolver/s0 不同 (本次 skillopt mean 0.5464 ≠ 历史 Stage0 0.5629, 两套独立采样). → **seed 级 paired-t 对 evolver vs skillopt 无效 (不报)**; 只能用 per-qid bootstrap (每题各自 4-seed 均值, 题目配对消除难度差, 但 seed 采样噪声未完全消除 → CI 偏宽保守). 若真差异仅 0.011, 即便同 seed 配对 4-seed 仍分不出 (需 ~70 seed, 见 a2 小尾巴)
- 守 CI 纪律: 不可称"已超过 skillopt", 也不可称"仍落后"; 唯一结论 = **点估计微低但统计不可区分, 已追平到噪声带内**. 要确认"不亚于/超过"需同 seed 配对 + 多 seed

- 4 seed delta 全正且范围窄 [0.220, 0.258] → 稳, 非单 seed 偶然
- per-qid 配对 (avg over 4 seeds each): 106 题, **63 正 / 26 负 / 17 零**, 方向强正

## 两个 CI 互相印证 (主/辅)
- **主判据** (per-qid bootstrap over questions, n_boot=2000): CI=[+0.1635, +0.3066], 下界>0 ✅
- 辅判据 (seed-level t-CI, df=3, n=4, 仅参考): CI=[+0.2083, +0.2603], 下界>0 ✅
- 两个 CI 均正区且重叠, 互相印证方向. 但辅判据 n=4 噪声大, 不作主判据; bootstrap over questions 是主.

## 口径自洽 (核心价值达成)
- s0 与 evolver **同一 test=106 集, 同 eval_k=3** → delta 不跨口径, 是干净泛化信号
- 避开了 Step1 内部 select(71题)/probe(36题) 口径错位 (best pass≈0.565 vs 判据线 0.5464 跨口径不可比)
- s0 基线 4-seed 全缓存 (inferred_k=[1,3] 核过 k=3 干净, key 不含 k 但缓存本身即 k=3 产物), 零 API
- evolver 8238b5e8 新 hash 重算, 无旧缓存污染

## 与 Step1 / 历史的关系 (禁跨口径比量级, 只判方向)
- Step1 select 口径 (71题) val_delta = +0.339 / CI +0.263 (见 [[step1-result-20260827]])
- 本次 test 口径 (106题) Δ = +0.2343 / CI +0.164
- ⚠️ **不同集不可比量级**: 0.339 vs 0.234 的差 (+0.105) 是两点估计差非 paired CI, 不可称"test 缩水"或"泛化损失". 唯一可说: **两口径方向均正** (select +0.263 CI>0, test +0.164 CI>0)
- 旧废弃 evolver (12633201, run_20260727 bug run) 同 test k=3 seeds2 曾得 0.4591 (Δ+0.154) — 数值被采样污染已废, 本次 8238b5e8 取代: Δ +0.2343 > 旧 +0.154, 但旧值 seeds=2 非同条件, 仍不可直接比, 仅参考

## 方法学备忘
- paired bootstrap unit = question: 每题 evolver 4-seed 均值 − s0 4-seed 均值 = 该题配对增益, bootstrap over 106 题
- 消除了 seed 采样噪声 (每题内部平均 4 次), 比 seed-level (n=4) 稳
- s0 pass_rate 粒度 {0,1/3,2/3,1} = k=3 确认; evolver 同 k=3, 口径一致

## 关联
[[test106-generalization-launch-20260827]] (启动配置) / [[step1-result-20260827]] (Step1 best 来源) / [[step1-caliber-mismatch-risk-20260826]] (口径 mess, 本次避开) / [[ci-band-not-point-est-20260826]] (CI 纪律) / [[test106-fairness-audit-20260827]] (本结果公平性核查 — evolver vs skillopt 对比能否作论文指标)
