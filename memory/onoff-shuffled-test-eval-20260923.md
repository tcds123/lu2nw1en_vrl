---
name: onoff-shuffled-test-eval-20260923
description: 干净 skill (shuffle_seed=0, 8/8) 的 test=106 评测结果 — OFF 54.1% / ON 52.2% / s0 33.7%, OFF-ON 仅 +1.9pp(t=2.88), 提升几乎全来自 meta_46(+0.493), other_60 实际为负(-0.017), 即无真实非meta算法增益. 60.2% lucky-end 不可复现.
metadata:
  type: project
---

# 干净 skill test=106 评测 (2026-09-23, shuffle_seed=0)

## 一句话结论

干净 skill (ON/OFF 均 shuffle_seed=0, 8/8) 的 **test=106 pass@1: OFF 54.1% / ON 52.2% / s0 33.7%**. OFF−ON 仅 **+1.9pp** (paired t=2.88, n=3, 边际) — val 侧的 −9.4pp 佣缩在 test 侧进一步收缩到噪声带. **提升几乎全部来自 meta_46 (meta-option 题, +0.493), other_60 实际为 −0.017 (略降)**: 即 skill 学到的本质是"识别 meta 题并选 meta 选项", 非位置偏置, 但也非"更会做数学"; 旧污染数据 60.2% 是 lucky-end, 不可复现.

## 权威数据 (multiseed_test_eval.json, k=3, seeds=3, test 已打乱选项)

| 臂 | mean pass@1 | meta_46 | other_60 | 3 seed passes | Δvs s0 |
|---|---|---|---|---|---|
| **OFF** (+0.3803 val) | **0.5409** | 0.7053 | 0.4148 | [0.525, 0.538, 0.560] | **+0.2044** |
| **ON** (+0.2864 val) | **0.5220** | 0.6498 | 0.4241 | [0.519, 0.516, 0.531] | +0.1855 |
| s0 基线 | 0.3365 | 0.2126 | 0.4315 | [0.321, 0.355, 0.333] | — |
| rule_meta_guess (零API) | 0.5220 | 1.0000 | 0.1556 | [0.519, 0.519, 0.528] | +0.1855 |

- **OFF−ON paired**: diffs=[+0.006, +0.022, +0.028], mean **+0.0189 (+1.9pp)**, sd=0.0113, **t=2.882** (n=3, 边际显著, p≈0.10 单侧).
- **rule_meta_guess 自检✓**: meta_46=1.000, other_60=0.156 (≈0.2 预期) — 打乱逻辑无 bug.

## 核心发现: 提升的来源分解 (OFF vs s0)

| 子集 | OFF | s0 | 差 | 对总分贡献 |
|---|---|---|---|---|
| meta_46 (46题) | 0.705 | 0.213 | **+0.493** | **+0.2138** |
| other_60 (60题) | 0.415 | 0.432 | **−0.017** | −0.0094 |
| **合计** | | | | **+0.2044** (= 实际Δ) ✓ |

→ **提升 100% 来自 meta-option 题的识别**. 在非 meta 的 60 道真数学题上, skill 不但没提升, 反而略降 (−1.7pp). rule_meta_guess 的 other_60=0.156 < OFF 的 0.415, 说明非 meta 题 OFF 仍有真实解题能力 (> 随机猜), 但相对 s0 基线没有净增益 — skill 提供的是 meta 检测启发式, 不是通用数学推理增强.

## 与污染数据对照 (回答"60.2% 能否复现")

| skill | shuffle_seed | test pass@1 | meta_46 | other_60 |
|---|---|---|---|---|
| 污染 OFF multirun2 (eea6e9ad) | None | **60.2%** | 88.4% | 38.6% |
| 污染 OFF multirun1 (e629d3f5) | None | 41.0% | 43.5% | 39.2% |
| **干净 OFF (b175e50dbd, 8/8)** | **0** | **54.1%** | 70.5% | 41.5% |

- **60.2% 不可复现**: 同配置两次污染 run 差 19pp (41 vs 60), 干净 run 落在中间 (54%).
- 干净 OFF 的 meta_46=70.5% 介于污染两次 (43.5% / 88.4%) 之间 — meta 检测能力本身随 evolver 收敛运气波动.
- **干净 skill 跨 seed 方差极低**: ON 3 seed=[0.519,0.516,0.531] (σ≈0.0066), OFF=[0.525,0.538,0.560] (σ≈0.0146) — 远低于污染数据跨 run 的 19pp 抖动. shuffle_seed=0 不仅治 val 污染, 也显著降方差.

## val→test 一致性

- val delta: ON +0.2864 vs OFF +0.3803 → ON−OFF=−0.094 (OFF 高 9.4pp val).
- test: ON 0.522 vs OFF 0.541 → OFF−ON=+0.019 (OFF 高 1.9pp test).
- **方向一致** (OFF 均≥ON), 但 val 侧 9.4pp 差在 test 侧佣缩到 1.9pp 且不显著 (t=2.88, n=3). val delta 的绝对量级与 test 不直接可比 (val 是相对 s0_cache 的 delta, test 是绝对 pass), 但排序方向可信.

## 产物路径

- 评测: `rl/train/verl1/outputs/onoff_shuffled_test_eval_20260923/multiseed_test_eval.json` (47KB)
- launcher: `rl/train/verl1/outputs/onoff_shuffled_20260922/launch_clean_test_eval.py`
- 干净 skill: ON=`onoff_shuffled_20260922/on_arm/run_20260922_162135/best_skill_evolved.md` (07707e1b15), OFF=`.../off_arm/.../run_20260923_095250/best_skill_evolved.md` (b175e50dbd)

## 关联

[[onoff-shuffled-final-verdict-20260923]] (val 侧 8/8 判定) / [[phaseA-shuffle-verdict-20260917]] (污染 test 侧佣缩) / [[pool-filter-on-off-verdict-20260917]] (+15.6pp 出处, 已推翻)
