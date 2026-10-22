---
name: evolver-coverage-diagnosis-launch-20260817
description: Task4 coverage_diagnosis 实验启动记录 — --coverage_diagnosis flag 实现+验证完成, treatment run PID47879 已启动 (15:37), 单变量 vs Task3 0.4654/0.462, 攻 INPUT 侧 (val 过拟合); C (结构自进化) 随后用其 failure_distribution 数据基底
metadata:
  type: project
---

# Task4 coverage_diagnosis 实验启动 (2026-08-17 15:37)

## 背景 (为何做 Task4)
Task3 ([[evolver-data-driven-focus-result-20260817]]) 证实机制但 val→test 迁移失败 (0.4654, +0.3pp 噪声, 落后 SkillOpt 0.5464 8.1pp). 根因精化 ([[evolver-meta-focus-diagnosis-20260817]]): gap 不是"没看到 meta 失败"而是两层耦合缺陷:
- **INPUT 侧 (D 攻此)**: `collect_failures` 跑完全部 58 select 题但只返回最难 8 → patcher 过拟合 val 的 8 个特定失败样本, 不匹配 test 分布.
- **OUTPUT 侧 (C 攻此, 待做)**: PATCH_TEMPLATE "keep <5000 chars" + CONSOLIDATE "merge/keep <4500" 结构性阻塞多 section 富内容.

用户决定: **先 D 后 C**; C 的结构让其**自进化摸索** (生成多结构变体→按 pass_rate 筛→胜者存活), 非我们指定 schema. 北星 = skill 逐迭代自进化, 非我们定算法规则优化它.

## 实现 (单 flag `--coverage_diagnosis`, 默认关 = 保 0.462 复现)
全部在 `skill_evolver_math.py`, ~60 行净增. 已 py_compile 过 + helper 冒烟过 + --help 过:
1. `compute_failure_distribution(failures)` (~L169): 全失败按 classify_failure_type 分桶 tally + 每类 up-to-3 最难 reps (复用已算 run_one, 零额外 API).
2. `find_uncovered_type(counts, skill_content)` (~L198): `_TYPE_COVERAGE_SIG` 签名词检测当前 skill 已覆盖哪些类型, 返回最高频**未覆盖**类型 (填洞非加固多数), 全覆盖→None (回退多数).
3. `collect_failures(..., return_all_failures=False)` (~L209): True 时额外返回全失败 list (零额外 API, 仅多返 `list(failures)`).
4. `PATCH_TEMPLATE_COVERAGE` (~L282): 喂全分布 tally 表 (每类 COVERED/UNCOVERED 标注) + 未覆盖类分层样本, rule 5 "Focus area this round (an UNCOVERED gap, not the majority)".
5. `make_coverage_diagnostics(reps, counts, uncovered, skill)` (~L367): 渲染 tally + 分层样本, **丢 600-char response_excerpt** (bloat 源), 字段截断 ~200 char.
6. main loop (~L738-766): coverage 模式独占 focus (supersede data_driven), `cov_counts/cov_reps/cov_uncovered` 选未覆盖类型, majority 回退; `coverage_bundle` 传 generate_candidates.
7. `evolver_log.json` iters 持久化 `failure_distribution` + `uncovered_type` (**C 阶段的数据基底**).
- 顺修: `--data_driven_focus` help 的 `93%` 转义为 `93%%` (argparse `%` 格式 bug, 原 Task3 遗留, 致 --help 崩); failure_cache 序列化/反序列化改 `list(v)`/`tuple(v)` 保 4 元 (原硬切 3 元致 resume 丢 all_failures).

## 公平性 (铁律 [[baseline-fairness-audit-20260817]])
- init=v2 (md5 **b98a9abd**) 从 S0 起, 非 SkillOpt-best (无嵌套).
- 同 Task3 harness: `--iters 2 --K 4 --eval_k 3 --n_diag 8 --train_n 5`, split 1-3-6 seed42, select=58.
- 默认关 → 主结果 0.462 复现性保持; flag 仅 treatment run 开.

## 启动状态
- **PID 47879**, log `rl/train/verl1/outputs/skill_evolver_math_meta/treatment_coverage_20260817_153705.log`
- 启动确认: `ablation=FULL | K_eff=4 | failure_driven=True | antideg=True` + `opt-flags ON: Task4:coverage_diag` + `select(train+val): 58 | S₀ cache: 71 | eval_k=3` (与 Task3 fair 同条件 ✅).
- 预期: baseline eval ~6min → iter1 (collect_failures 全失败 + coverage focus + K4 候选 + eval) ~20min → iter2 ~20min. 总 ~45min.

## 判据 (诚实三档)
- **Closed**: treatment 4-seed > 0.5464 且配对 t vs SkillOpt p<0.05 → gap 闭合.
- **Partial (likely)**: 0.462 < treatment ≤ 0.5464 → D 闭合部分 gap, 验证 val 过拟合是真损失分量, 暴露全分布作 C 基底.
- **Refuted**: treatment ≤ 0.462 (噪声内) → input 侧诊断非瓶颈, 损失在 output 侧 (结构), 确认 C 是必要下一步 (带 D 分布数据).

## 下一步
跑完 → eval_test_multiseed (s0+skillopt 4-seed 缓存复用, 仅 evolver fresh) → 结果入 `memory/evolver-coverage-diagnosis-result-20260817.md` + MEMORY.md + 异常入 `docs/phase2_issues.md`. 然后启动 C (结构自进化).

Related: [[evolver-data-driven-focus-result-20260817]], [[evolver-meta-focus-diagnosis-20260817]], [[baseline-fairness-audit-20260817]], [[algorithm-future-optimization-directions-20260805]]
