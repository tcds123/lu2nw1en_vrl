---
name: evolver-data-driven-focus-result-20260817
description: Task3结果 — data-driven focus(data_driven_focus flag)治标不治本: 机制证实生效(val+0.268,3.8×超round-robin,focus正确选meta-option 8/8两iter),但val增益不迁test: test=106 4-seed=0.4654仅+0.3pp超旧fair-best0.462,仍落后SkillOpt 0.5464 8.1pp(p=0.0004显著);根因=val最难15题93%meta-option→patcher过拟合val特定失败分布不匹配test
metadata:
  type: project
---

# Task3 data-driven focus 实验结果 (2026-08-17 终局)

## 结论 (一句话)
**机制证实生效但 val→test 迁移失败**: `--data_driven_focus` 让 evolver 正确诊断到 meta-option 失败并写 meta-option 决策程序 (val delta +0.268, 3.8× 超 round-robin 典型 +0.07, iter1/iter2 focus 均 8/8 meta-option), **但 val 增益未泛化到 test** — test=106 4-seed = 0.4654, 仅 +0.3pp 超旧 fair-best 0.462 (噪声内), 仍落后 SkillOpt 0.5464 **8.1pp** (per-seed t=-17.2 p=0.0004; per-qid t=-3.39 p=9.8e-4). **8.4pp gap 未闭合**.

## 实验设置 (公平, 单变量)
- 代码改动: `skill_evolver_math.py` 加 `classify_failure_type` helper + `--data_driven_focus` flag (默认关, 保主结果复现), focus 选择改为已诊断失败的多数类型 (非 round-robin). 见 [[evolver-meta-focus-diagnosis-20260817]].
- Treatment run: `--init_skill v2(b98a9abd) --iters 2 --K 4 --eval_k 3 --n_diag 8 --train_n 5 --data_driven_focus`, run_20260817_095953.
- 产物: `best_skill_evolved.md` md5=**cde0bb9275** (≠v2 b98a9abd ≠SkillOpt bf82b460/cfae156d — 公平无嵌套 ✅), 4416 chars, meta-option grep 计数 4(>2 ✅).
- Eval: `eval_test_multiseed --skills s0,evolver,skillopt --eval_k 3 --seeds 4`, multiseed_test_meta/. s0+skillopt 复用旧 multiseed_test 缓存 (同 hash+seed=同内容, byte-等价), 仅 evolver 4 seed fresh.

## 运行轨迹 (treatment)
- **iter1**: ACCEPT cand3, delta=+0.268 CI=+0.205 pass=0.506 (val select 0.31→0.506, 已达 SkillOpt 级), hash b98a9abd→cde0bb92. focus tally: Meta-options 8/8 → 选 foci[5] ✅.
- **iter2**: 4 候选全正 (cand0+0.159/cand1+0.216/cand2+0.164/cand3+0.228, CI 全>0), 但 **KEEP PREV** (no_improve=1) — M3/M4 gate 拒绝进一步 val 拟合 (诚实防过拟合, 同 AIME CI-gate 行为 [[aime-fresh-rerun-result-20260807]]). 最终产物=iter1 winner cde0bb92.

## test=106 4-seed 结果 (k=3, EVAL_TEMP=0.3, split 1-3-6 seed42)
| 方法 | mean | per-seed | Δvs S0 |
|------|------|----------|--------|
| s0 | 0.3011 | [0.3082,0.3019,0.2987,0.2956] | baseline |
| **evolver(data_driven)** | **0.4654** | [0.4969,0.4623,0.4371,0.4654] | +0.1643 |
| skillopt | 0.5464 | [0.566,0.5409,0.5283,0.5503] | +0.2453 |

统计:
- evolver vs skillopt: per-seed mean diff=-0.0810 sd=0.0094 **t=-17.247 p=0.0004** (df=3); per-qid paired t=-3.393 **p=9.77e-04** (n=106); evo>sk qids 30/106.
- evolver vs s0: per-qid paired t=6.398 **p=4.47e-09**, evo>s0 qids 58/106.
- **vs 旧 fair-best**: 0.4654 vs 0.462 = +0.3pp (噪声内, 未显著提升).

## 根因诊断 (为何 val→test 迁移失败)
1. **根因诊断本身正确**: val 最难 15 题 14/15=93% meta-option 属实; 旧 round-robin 早 iter 用 "Implication vs biconditional" 错方向 patch 属实.
2. **fix 在 val 上生效**: focus 正确选 foci[5], 候选 delta +0.268 远超 round-robin. 机制假设验证通过.
3. **但 val 分布 ≠ test 分布**: patcher 针对 **val 的 8 个最难失败** 写决策程序; 这 8 题是 val 特定样本, 其失败模式 (具体定理/选项) 不完全代表 test 的 meta-option 失败. val select 0.506 是 **val 过拟合信号**, 非 test 泛化信号. iter2 的 KEEP PREV gate 已嗅到此 (4 正候选全拒) — 是诚实防过拟合, 但也意味着 evolver 已无更多 val 信号可榨.
4. **本质**: gap 8.1pp 是 **SkillOpt 的 meta-option 决策程序内容更强** (结构化 2-section+SLOW_UPDATE+领域启发式, 9926 chars), 而非 "evolver 没看到 meta 失败". 单靠 focus 引导 patcher 从零写决策程序, 2 iter 不足以逼近 SkillOpt 的成熟结构. focus 是必要非充分条件.

## 判据对照 (plan)
- ❌ 未达成 "treatment test 4-seed > 0.5464 且配对 t vs SkillOpt p<0.05 → gap 闭合".
- 🟡 部分进展: 机制证实 (focus 正确+val 强增益), 但未达 test 目标.
- fallback 方向 (plan step 3): 若 ≤0.462 则 refuted → 本结果 0.4654 略 >0.462 但噪声内, 接近 refuted. 建议结合 opt-flags 或直接 meta-procedure 注入 (自家内容) 作下一实验.

## 公平性 (硬性铁律 [[baseline-fairness-audit-20260817]])
- ✅ init=v2(b98a9abd) 从 S0 起, 无 SkillOpt 嵌套 (cde0bb92≠bf82b460).
- ✅ eval 同 harness (eval_test_multiseed, test=106, k=3, EVAL_TEMP=0.3, 同 split), SkillOpt 0.5464 同条件复用.
- ✅ 默认 flag 关 → 主结果 (0.462) 可复现性保持.

## 产物
- Treatment skill: `rl/train/verl1/outputs/skill_evolver_math_meta/run_20260817_095953/best_skill_evolved.md` (md5 cde0bb9275)
- Treatment log: `.../skill_evolver_math_meta/treatment_run.log`
- Eval: `rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test_meta/multiseed_test_eval.json`

## 下一步 (未闭合, 需新方案)
gap 8.1pp 未闭合. data-driven focus 证实"看到了但用错框架"诊断对, 但单变量不足以追平 SkillOpt 成熟结构. 候选:
- **(A) 更长 iter / 更大 K**: 给 patcher 更多机会逼近 meta-option 结构 (但 iter2 已 stall, 边际收益存疑).
- **(B) 直接 meta-procedure 注入 (自家内容)**: 手写 meta-option 决策程序作 init, fair (自家内容非抄 SkillOpt), 但非纯 evolver 改进.
- **(C) v3 分层 (form/content 分层 + S0-coverage 检查)** [[algorithm-future-optimization-directions-20260805]]: 从结构层而非 focus 层破.
- **(D) 跨题诊断**: 用 train+val 全部 meta-option 失败 (非仅 8 最难) 作诊断输入, 扩大失败模式覆盖, 减 val 过拟合.

Related: [[evolver-meta-focus-diagnosis-20260817]] (诊断+plan), [[baseline-fairness-audit-20260817]] (公平现状表), [[aime-fresh-rerun-result-20260807]] (CI-gate 诚实拒先例), [[algorithm-future-optimization-directions-20260805]] (铁律+优化方向)
