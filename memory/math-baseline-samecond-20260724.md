---
name: math-baseline-samecond-20260724
description: 🔥数学bench全量基线同条件坐实 - noskill 29.0%/seed(S_0) 33.1%/best 51.6%(+18.6pp); Step0正结果复现, 数学是干净试验场
metadata: type: project
---

# 数学 bench 全量基线同条件坐实 (2026-07-24) 🔥🔥🔥

## 核心结果: LiveMathematicianBench test=124 (同条件, 新key msk-4601b27, greedy temp=0, qwen_chat backend, 0 个 429)

| 条件 | acc (n=124) | Step0 旧值(旧key) | Δ |
|------|-------------|-------------------|---|
| **noskill** (empty) | 0.2903 (29.0%) | 24.19% | +4.9pp (小差异, 模型/采样) |
| **seed** (initial.md = S_0) | 0.3306 (33.1%) | 32.26% | 基本一致 ✅ |
| **best** (skill_v0002, Step0 best-on-val) | 0.5161 (51.6%) | 54.03% | 基本一致 ✅ |

**seed→best = +18.6pp (33.1%→51.6%)**, 与 Step0 旧值 +21.8pp 量级一致。

## 关键结论
1. **Step0 正结果在同条件下复现**: skill 优化在数学上真实有效, 不是旧key假象。与代码方向(07-23 同条件重测反转, TF-global<S_0)形成鲜明对比 — **数学是干净试验场**。
2. **5题smoke的0分是小样本偶然+格式坑**: 全量124题下 noskill 29% (非0)。格式坑(<answer>标签)在全量下不致命, DeepSeek多数情况守格式。
3. **headroom巨大**: noskill 29% → best 52%, skill优化空间 +23pp, 远比代码方向(LCB hard 死区)可观。

## S_0 定义确认
- 数学 bench 的 **S_0 = seed skill = `skillopt/envs/livemathematicianbench/skills/initial.md`** (33.1%)
- best = `outputs/optimize_ep4/skills/skill_v0002.md` (51.6%, Step0 的 best-on-val, 即 SkillOpt 方法产出)

## 产出文件
- noskill: /data/zhuldz/skillopt/outputs/full_noskill_0724
- seed (S_0): /data/zhuldz/skillopt/outputs/full_seed_0724
- best: /data/zhuldz/skillopt/outputs/full_best_0724
- 日志: /tmp/math_full_baseline.log
- 脚本: /tmp/math_full_baseline.sh

## 对位表 (待填 TF/训练线)
| 方法 | acc | Δ vs S_0 |
|------|-----|----------|
| noskill | 0.2903 | -3.9pp |
| S_0 (seed) | 0.3306 | 0 |
| SkillOpt-best (别人方法, 旧跑) | 0.5161 | +18.6pp |
| TF-global (我们, 免训练) | ⏳ 待跑 | |
| TF-per-task (我们, 免训练) | ⏳ 待跑 | |
| GRPO-4B (我们, 训练线) | ⏳ 待跑 | |

## 下一步
1. ⏳ 把 tf_skill_optimizer.py 切到数学 env, 测 TF-global / TF-per-task vs S_0 (真问题: 数学上 global 能否超 S_0? 代码上不行)
2. ⏳ 注意: SkillOpt-best 51.6% 是 per-task optimize 产出 (val 门控选 best), 与 TF-global(全局改写) 不同形态。对位时需明确
3. ⏳ 8卡4B GRPO 训练线 vs TF 同表对位
4. 代码线结果继续作为对照保留, 不删除

Related: [[math-trial-direction-20260724]], [[skillopt-step0-results-20260714]], [[tf-global-reversal-samecond-20260723]], [[direction-global-not-pertask-20260723]]
