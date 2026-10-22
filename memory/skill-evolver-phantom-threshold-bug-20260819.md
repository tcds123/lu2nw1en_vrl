---
name: skill-evolver-phantom-threshold-bug-20260819
description: skill_evolver_math.py no_improve 闸"幻影门槛"bug — consolidate 接受分支不同步 best_val_delta, 后续候选须超历史峰值才 ACCEPT; 已确认未实质影响 Phase C 结论 (峰值版 b6a80db9 对照 test≈巩固版 p=0.910); 完整根因+修复方向在 docs/phase2_issues.md
metadata:
  type: reference
---

# 幻影门槛 bug (consolidate 不更新 best_val_delta)

**完整分析位置**: `docs/phase2_issues.md` 2026-08-19 条 (行 982-1015). 本文仅作快速召回指针 + 结论影响判定.

## 一句话
`skill_evolver_math.py` 的 `no_improve` 闸对比基准是 `best_val_delta` (历史峰值), consolidate 接受分支 (gate 路径 L1004 / 非 gate L1013) **从不更新 `best_val_delta`** → 巩固后 `best_skill` 实际 delta 与记录的 `best_val_delta` 分裂 → 后续候选须超巩固前峰值才 ACCEPT ("幻影门槛").

## 关键行
- 闸判断 (L965): `improved = winner["delta"] > best_val_delta and winner["ci"]>0 and winner["hash"]!=current_hash`.
- `best_val_delta` 仅 3 处赋值: L757(resume) / L766(fresh=init) / **L976(仅 improved 分支)**. consolidate 接受分支无赋值.
- gate 路径 L1006-1007 甚至 print `delta {best_val_delta}->{gd}` (记录峰值→巩固后实测), **只打印不赋值**.

## 对 Phase C 结论的影响 = 无 (已排除)
- Phase C 最终产物 = 巩固后 2f68a346 (val +0.108), 而非峰值 b6a80db9 (val +0.131, iter2 ACCEPT 巩固前, 从未 test).
- **峰值版对照已测** (PID 47142): b6a80db9 test=0.3687 vs 巩固版 2f68a346 test=0.3703, per-seed p=0.910 / per-qid p=0.910, peak>consol 仅 42/106 → 两版统计不可区分, 都落 0.37 失败区.
- **结论**: bug 未实质选错 skill. Phase C "未闭合 gap" 是方法本身 test 不迁移, 非 bug 选错产物. 详见 [[phase-c-structure-evolution-result-20260819]].
- winner_kind 诊断 (structure 2/patch 0 accepted) 仍有效: ACCEPT 发生在幻影门槛建立前 (iter1/2).

## 修复方向 (待决, 未改)
consolidate 接受分支同步 `best_val_delta` 到实测值:
- gate 路径 L1004-1005 后补: `best_val_delta = gd; best_val_ci = gcl`.
- 非 gate 路径 L1013 后同理.
- 这是代码正确性修复, 非任何已结实验结论的影响因素.

Related: [[phase-c-structure-evolution-result-20260819]] (结论), [[phase-c-structure-evolution-20260818]] (启动)
