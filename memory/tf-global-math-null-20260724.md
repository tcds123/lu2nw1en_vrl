---
name: tf-global-math-null-20260724
description: 🚨TF-global数学版test=124=0.3306≈S_0(33.1%), -0.04pp持平未超; train best_delta+0.23~0.29真撬动但不泛化; per-task才是路径
metadata: type: project
---

# 免训练线 TF-global 数学版结果 (2026-07-24) 🚨

## 核心结果: LiveMathematicianBench test=124 (同条件, 新key, greedy, 干净cache)

| 方法 | test acc | Δ vs S_0 |
|------|---------|----------|
| noskill | 0.2903 (29.0%) | -3.9pp |
| **S_0 (seed initial.md)** | **0.3306 (33.1%)** | 基线 |
| **TF-global (我们, 免训练)** | **0.3306 (33.06%)** | **-0.04pp (持平, 未超)** |
| best (SkillOpt per-task optimize) | 0.5161 (51.6%) | +18.6pp |

**TF-global test = S_0, 持平未超过。** val 门控 best_val_delta=+0.056 看似正向, 但 val 仅 18 题, 门控弱, 未泛化到 test。

## 训练侧信号 (delta 干净, 同条件 53 题 cache)
| iter | train mean best delta | val new delta | 决策 |
|------|----------------------|---------------|------|
| baseline | — | -0.111 | — |
| 1 | +0.257 | -0.167 | KEEP |
| 2 | +0.286 | +0.056 | ACCEPT (产出) |
| 3 | +0.229 | -0.056 | KEEP |

**train best_delta 持续正 (+0.23~+0.29)**: rewriter 确实能在 train 上把 S_0 错的题改对 (前2题 best_delta=+1.0, deltas=[0,1,1,0] 真撬动)。但全局合成的 skill 没泛化到 test。

## 关键结论
1. **TF-global 在数学上也未能超过 S_0** — 与代码方向 ([[tf-global-reversal-samecond-20260723]]) 结论一致: global 改写对强模型(DeepSeek-V3.2) + 好初始 skill 收益甚微。
2. **但数学≠代码**: 数学 train 侧 rewriter 真撬动 S_0 错题 (best_delta 正), 代码 LCB hard 死区零收益。数学有 headroom, 只是 global 形态不泛化。
3. **per-task 才是路径**: best(SkillOpt per-task optimize) 51.6% >> S_0 33.1%, 印证 SkillComposer per-task 才对 (与代码方向 [[tf-pertask-positive-20260723]] per-task +8.4pp 一致)。
4. **val 18题门控弱**: +0.056 未泛化到 test。需更大 val 或交叉验证。

## 真问题的新答案
原问"数学上 global 能否超 S_0?" → **不能** (持平)。
新问"数学上 per-task 能否超 S_0?" → **能** (SkillOpt per-task 51.6% 已证, 但那是别人方法; 我们的 TF-per-task 待测)。

## 产出文件
- TF-global skill: rl/train/verl1/outputs/tf_skill_math/run_20260724_141709/tf_global_skill_math.md (1432 chars, iter2)
- 训练 log: 同目录 tf_run_log.json, experience_pool_tf_math.json
- test 评测: /data/zhuldz/skillopt/outputs/eval_tf_global_math_0724
- s0_cache (干净53题): rl/train/verl1/outputs/tf_skill_math/s0_cache_math_full.json
- 脚本: rl/train/verl1/tf_skill_optimizer_math.py, precompute_s0_math.py

## 下一步
1. ⏳ TF-per-task 数学版 (每题现场 rewriter + 经验池, 对标代码 per-task +8.4pp): 验证我们的方法 per-task 是否也能超 S_0
2. ⏳ 若 TF-per-task 数学也超 S_0 → 与代码一致, 论文故事=per-task 通用有效 + global 通用无效
3. ⏳ 8卡4B GRPO 训练线 vs TF 同表对位
4. 代码线结果继续作对照保留

Related: [[math-baseline-samecond-20260724]], [[math-trial-direction-20260724]], [[tf-global-reversal-samecond-20260723]], [[tf-pertask-positive-20260723]], [[direction-global-not-pertask-20260723]]
