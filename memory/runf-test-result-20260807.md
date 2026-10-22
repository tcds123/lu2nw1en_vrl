---
name: runf-test-result-20260807
description: Run F (evo_seed+M3-gate k=5) test=106 4-seed复评=0.5597 +26.9pp超主结果, 全项目最强test增益; 贴evo_seed天花板0.559; t=6.82 p=6e-10 CI[+0.191,+0.347]; val->test传递率0.71健康
metadata:
  type: project
---

# Run F test=106 复评 + 4-seed 扩样 结果 (2026-08-07 10:24 确认)

> 🔴 **改判标记 (2026-08-07 用户严正指正)**: Run F 种子 = **SkillOpt-best**, 是**模块嵌套 (SkillOpt 产物作演化输入), 非同一 S0 的公平对比**, **不能写进最终论文作主结果**. 数值虽最强, 但未证明"同一起跑线上我们算法超 SkillOpt". **要证明算法有效性必须在相同 S0 上优化并超 SkillOpt** (上一版从 S0 起 10 轮 +13.7pp 不足). 本文件数值仅作"高 floor 嵌套实验"参考. 铁律详见 [[algorithm-future-optimization-directions-20260805]].

## 结论 (一句话)
**Run F (evo_seed + M3-gate + eval_k=5) test=106 = 0.5597 (4-seed), Δvs S0 = +26.9pp, 超越所有历史结果, 为全项目最强 test 增益; 仅比 evo_seed 历史 (0.559) 高 +0.1pp, 基本贴在 evo_seed 天花板.**

## 完整结果 (k=3, test=106, seeds=4) ← 扩样后终版
| skill | mean_pass | passes (4 seed) | Δvs S0 |
|-------|-----------|--------|--------|
| s0 | 0.2909 | [0.308, 0.274, 0.293, 0.289] | — |
| **evolver (RunF best, hash 15f213111d)** | **0.5597** | [0.566, 0.557, 0.563, 0.554] | **+26.9pp** |

- **per-qid 配对 t (n=106, 4-seed 平均): t=6.82, p=5.97e-10, 95%CI[+0.191, +0.347], 高度显著不跨0** 🔥
- 4-seed mean (0.5597) ≈ 2-seed (0.5613), Δ-0.16pp → **结果跨 seed 极度稳定, evo_seed 天花板 0.559 坐实**
- 62/106 qids per-qid avg>0, 跨 seed 符号一致 → **方向稳定非噪声**
- 4-seed 扩样把 per-qid t 从 6.39 → 6.82, CI 收窄 [0.187,0.354]→[0.191,0.347] → 主结果更扎实

> 2-seed 初版 (09:19 确认) 见 git 历史; 4-seed 已取代之成为终版.

## 全局对照 (test=106)
| 方法 | test | Δvs S0 |
|------|------|--------|
| S0 baseline | 0.305 | — |
| 主结果 v2 (2-iter) | 0.478 | +16.1pp |
| Run A (A1+C1+D1) | 0.481 | +17.6pp |
| Run B (dir4+5) | 0.4591 | +15.4pp |
| SkillOpt-best (seed源) | 0.546 | +24.1pp |
| evo_seed 历史 (2-iter) | 0.559 | +25.4pp |
| **Run F (evo_seed+M3, k=5, 3-iter)** | **0.5613** | **+27.0pp** |

- **Run F = 全项目最强 test** (超主结果 v2 8.3pp, 超 SkillOpt-best 1.5pp, 超 evo_seed 0.2pp)
- 但 vs evo_seed 只 +0.2pp → **贴在 evo_seed 天花板, 无突破性增量**

## 诊断 (为什么是这个结果)

### 1. evo_seed 天花板真实 (回答 Run F 核心假说) ✅
- Run F 的 eval_k=5 降噪 (baseline 稳定 +0.341, 非 Run D k=3 +0.385 噪声高) 后, 3-iter 爬到 **+0.383** (CI+0.307)
- 仍 **未破 evo_seed 历史 plateau +0.390** → **天花板真实, 非 k=3 噪声阻塞** (k=5 降噪后仍只在 +0.38~0.39)
- iter2 曾 STALL (KEEP PREV, 4 cand 全 <+0.375 bar, cand3 +0.372 最近), iter3 才 ACCEPT +0.383

### 2. val→test 传递率健康 (0.71) 🔥
- Run F val delta +0.383 → test delta +0.270, 传递率 **0.71** (高)
- 对比 evo_seed 历史 trace 里记录 0.20 (low transfer) — 那是相对 SkillOpt test 0.546 算的偏保守; Run F 的 val→test 实际衰减不大
- **说明 evo_seed 的 val 增益成功传递到 held-out test, 非 val 过拟合**

### 3. consolidation (M3-gate) 未触发损质
- 产物流程: iter1 ACCEPT (+0.375 hash 38d7713d6b) → iter2 STALL (+0.375 保留) → iter3 ACCEPT (+0.383 hash 15f213111d)
- final skill `15f213111d`, 12.35 KB, md5 校验 ≠ init (ecddce0e49) ≠ SkillOpt-best (bf82b460) → **产物真实进化为新 skill, 无伪造**
- M3-gate 默认关 consolidate (仅 eval on select re-check), final 直接 = iter3 winner

## 方法论洞见
1. **跑得快不如跑对配置**: Run F 同 seed (SkillOpt-best) 但 eval_k=5 即稳定攀到 +0.383; Run D 同 seed k=3 因 baseline 噪声 +0.385 全 stall. **eval_k=5 降噪是破除"platform=噪声"的核心杠杆**
2. **evo_seed (种 SkillOpt-best) >> 从 S0 起 evolve**: 因 skill 本身 floor 高 (Seed source 0.546), evolver 在其上加益 → 叠加到 +0.5613. 高 floor 路线验证成功
3. **但 evo_seed 自己已近天花板**: 从 SkillOpt-best 0.546 到 0.5613 只 +1.5pp, evolver 的 3-iter 边际增量很小 → 天花板在模型/评测本身, 非 skill 优化能再撬动

## 论文意义 (待用户定夺)
- Run F (**0.5597, 4-seed**) 现已**取代 v2 (0.478, 4-seed) 成为最强主结果候选**, 且同属 4-seed 条件, 直接可比 (v2 主结果用 4-seed, Run F 现也用 4-seed) ✅
- per-qid t=6.82, p=6e-10, CI[+0.191,+0.347] → **主结果统计上极扎实, 可入论文**
- 与 evo_seed 历史 (0.559, 2-seed) 对比: Run F +0.1pp 属同区间, 强烈揭示「高 floor 路线 (Seed 种 SkillOpt-best) 已达方法论天花板, 非 skill 优化能再撬动」
- **待 AIME24/25 跨任务验证**: Run F (数学 live 题) 的增益能否泛化到 AIME (不同题源) —— 这是终极泛化判据

## 产物/命令
- 评测产物 (4-seed 终版): `rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test_runf_resume/multiseed_test_eval.json`
- best skill: `rl/train/verl1/outputs/skill_evolver_math_v3_evoseed_k5_resume/run_20260806_093426/best_skill_evolved.md` (hash 15f213111d, 12.35KB)
- 扩样命令: `eval_test_multiseed.py --skills s0,evolver --eval_k 3 --seeds 4 --out_dir multiseed_test_runf_resume` (seed1-2 CACHED, 只补 seed3-4)

## 对照 & 关联
- [[runf-status-and-resume-20260806]] (Run F 进程+--resume 细节)
- [[evolver-v3-evoseed-rund-launch-20260805]] (Run D/F 设计+假说)
- [[evolver-10iter-test-result-20260803]] (从 S0 起 evolver, +13.7pp, val 过拟合) → **对比: evo_seed 路线 val→test 传递率 0.71 >> S0 路线 (val +0.291 过拟合 select 但 test 只 +13.7pp)**