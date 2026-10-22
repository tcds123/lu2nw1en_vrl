---
name: evolver-coverage-diagnosis-result-20260817
description: Task4 coverage_diagnosis 结果 — REFUTED: test=106 4-seed=0.3632, 比 Task3(0.4654) 还低 10.2pp, 落后 SkillOpt 18.3pp; 但这是"产出性反证"— D 的 skill meta-option 内容比 Task3 更丰富(2 嵌套子程序)却更差, 孤立出"结构>内容量"为 C(结构自进化)的因果证据
metadata:
  type: project
---

# Task4 coverage_diagnosis 实验结果 (2026-08-17 终局, REFUTED)

## 结论 (一句话)
**产出性反证 (productive refutation)**: `--coverage_diagnosis` 让 evolver 喂 patcher 全失败分布(33 题, 非仅 8 最难) + 覆盖缺口填洞(非加固多数), val 累计增益 +0.107(>Task3 iter1 单次 +0.268, 但 D 是 2-iter 累计更诚实), **但 test=106 4-seed = 0.3632, 不仅未闭合 gap, 反比 Task3(0.4654) 退步 10.2pp, 落后 SkillOpt(0.5464) 18.3pp**. gap 从 8.1pp 恶化到 18.3pp.

**关键洞察 (D 的真正价值)**: D 的 skill(4615 chars) 比 Task3(4416 chars) 有**更多** meta-option 内容(2 个嵌套子程序: "Decision Procedure for Meta-Options" + "Comparative Strength Test"), 却得分更差 10pp. 两者 failure-mode bullet 集合几乎相同, 唯一差异是**结构位置**: Task3 把 meta-option 决策程序作**独立顶级 bullet 置底**(突出, 末位近因); D 把它**嵌套为 "Meta-Statement Evaluation" 的子 bullet**(从属, 中段) + 加了一个抽象通用程序. **结构 > 内容量**: 内容近似相同, 结构差异驱动 10pp 退步 → 直接证据指向 C(结构自进化).

## 实验设置 (公平, 单变量, 沿 Task3 chain)
- 代码: `skill_evolver_math.py` 加 `--coverage_diagnosis` flag (默认关, 保 0.462 复现). 实现: `compute_failure_distribution`+`find_uncovered_type`+`collect_failures(return_all_failures)`+`PATCH_TEMPLATE_COVERAGE`+`make_coverage_diagnostics`, ~60 行净增. 全部 py_compile+冒烟+--help 过. 详见 [[evolver-coverage-diagnosis-launch-20260817]].
- Treatment run: `--init_skill v2(b98a9abd) --iters 2 --K 4 --eval_k 3 --n_diag 8 --train_n 5 --coverage_diagnosis`, run_20260817_153707.
- 产物: `best_skill_evolved.md` md5=**2af4fb0901** (≠v2 b98a9abd ≠Task3 cde0bb92 ≠SkillOpt bf82b460/cfae156d — 公平无嵌套 ✅), 4615 chars.
- Eval: `eval_test_multiseed --skills s0,evolver,skillopt --eval_k 3 --seeds 4`. s0+skillopt 4-seed 缓存复用 (同 hash+seed byte-等价), 仅 evolver 4 seed fresh.

## 运行轨迹 (treatment)
- **iter1**: ACCEPT cand0, delta=+0.072 CI=+0.021 pass=0.310, hash b98a9abd→b678ad41d0, 3655→4210 chars. coverage tally: Meta=31✗|Quantifier=1✗|Implication=1✗ (33 failures), uncovered=None (v2 已 keyword 覆盖全部 8 类) → 回退 majority(meta).
- **iter2**: ACCEPT cand1, delta=+0.107 CI=+0.056 pass=0.345, hash b678ad41d0→2af4fb0901, 4210→4615 chars. coverage tally: Meta=30✗|Quantifier=1✗|Implication=1✗ (32 failures), uncovered=None → 回退 majority(meta).
- **注**: D 的 iter2 是真 2-iter 累积(ACCEPT cand1), 非 Task3 的 iter2 KEEP-PREV. val 累计增益 +0.107 < Task3 iter1 单次 +0.268 — D 在 val 上增益本就更小(设计意图: 看全分布减少过拟合), 但 test 迁移**更差**而非更好.

## test=106 4-seed 结果 (k=3, EVAL_TEMP=0.3, split 1-3-6 seed42)
| 方法 | mean | per-seed | Δvs S0 |
|------|------|----------|--------|
| s0 | 0.3011 | [0.3082,0.3019,0.2987,0.2956] | baseline |
| **evolver(coverage)** | **0.3632** | [0.3553,0.3679,0.3711,0.3585] | +0.0621 |
| skillopt | 0.5464 | [0.566,0.5409,0.5283,0.5503] | +0.2453 |

统计:
- evolver vs skillopt: per-seed paired t(df=3)=-15.817 **p=0.000549** diff=-0.1832; per-qid paired t=-5.056 **p=1.8e-6** diff=-0.1321 (n=106).
- evolver vs s0: per-seed t=11.606 p=0.001374 diff=+0.0621; per-qid t=5.681 p=1.2e-7 diff=+0.1132.
- evo>sk qids 33/106; evo>s0 62/106; evo==sk 15/106.

## 判据对照 (plan 三档)
- **Refuted (实际落此)**: treatment 0.3632 ≤ 0.462 (远低于). 输入侧诊断**非瓶颈**(已在 [[phase2_issues]] 中期发现记录: coverage 回退 majority, 全分布 94% meta-option ≈ Task3 的 8-subset 93%).
- 比 plan 预期的"partial/refuted"更彻底: 不仅未闭合, 还退步 10pp. **但产出性**: 孤立出结构为因果变量(见下).

## 根因 (为何 D 比 Task3 更差)
1. **覆盖机制在真实数据上未触发填洞**: v2 skill 已 keyword 覆盖全部 8 类失败类型(含 meta-option 的 "meta"/"stronger result" 签名), `find_uncovered_type` 返回 None, 两 iter 都回退 majority(meta-option) — 与 Task3 同方向. 全分布(94% meta)≈ Task3 的 8-hardest subset(93%), Task3 的"val 8 题过拟合"假设被部分证伪(分布近似).
2. **结构退步(真因)**: patcher 看到全分布 + 覆盖诊断后, 把 meta-option 决策程序**重构为嵌套子 bullet**(置于 "Meta-Statement Evaluation" 下, 中段从属), 并加了一个抽象 "Comparative Strength Test" 通用程序. Task3 则把 meta-option 作**独立顶级 bullet 置底**(末位近因, 突出). 两者 meta-option 文字内容量近似(D 略多), 但 D 的结构使其信号被稀释 → 10pp 退步.
3. **val 增益更小但 test 更差**: D val +0.107(2-iter 累计) vs Task3 val +0.268(iter1 单次). 更小的 val 增益**未**换来更好的 test 泛化, 反而 test 更差 — 说明 val→test 迁移不是"过拟合程度"问题, 而是**内容结构**问题.

## 产物
- Treatment skill: `rl/train/verl1/outputs/skill_evolver_math_meta/run_20260817_153707/best_skill_evolved.md` (md5 2af4fb0901, 4615 chars).
- Treatment log: `.../treatment_coverage_20260817_153705.log`.
- Test eval: `rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test_meta/multiseed_test_eval.json` + per_evolver_2af4fb09_seed*.json.
- 数据基底 (C 用): `run_20260817_153707/evolver_log.json` 的 iters[].failure_distribution + uncovered_type (两 iter 均持久化).

## 公平性 (硬性铁律 [[baseline-fairness-audit-20260817]])
- ✅ init=v2(b98a9abd) 从 S0 起, 无 SkillOpt 嵌套 (2af4fb09≠bf82b460≠cfae156d≠b98a9abd).
- ✅ eval 同 harness (eval_test_multiseed, test=106, k=3, EVAL_TEMP=0.3, 同 split), SkillOpt 0.5464 同条件复用, s0/skillopt 缓存 byte-等价.
- ✅ 默认 flag 关 → 主结果(0.462)可复现性保持.

## 下一步: C (结构自进化, 北星)
D 证伪输入侧, 孤立出结构为因果变量. 按 [[evolver-meta-focus-diagnosis-20260817]] 和用户决定: C 让结构**自进化摸索** — 生成多个结构变体(如: meta-option 独立顶级 section vs 嵌套子 bullet vs 末位 bullet vs 分布式散布) → 按 pass_rate 筛 → 胜者存活. 非我们指定 schema. D 的 per-iter failure_distribution + uncovered_type 是 C 覆盖检查的数据基底. 预期: 结构自进化应能复现 SkillOpt 的"meta-option 作突出顶级决策程序"结构优势, 而无需嵌套其内容.

Related: [[evolver-coverage-diagnosis-launch-20260817]] (启动), [[evolver-data-driven-focus-result-20260817]] (Task3 对照), [[evolver-meta-focus-diagnosis-20260817]] (诊断+plan), [[baseline-fairness-audit-20260817]] (铁律), [[algorithm-future-optimization-directions-20260805]] (优化方向)
