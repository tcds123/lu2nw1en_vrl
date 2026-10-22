---
name: probe-split-result-20260820
description: Task5 诊断-选择解耦 probe-set 实验终局=机制工作但未破天花板. probe-split evolver test=106 4-seed=0.4654, 与 Task3(0.4654)逐题配对 t=-0.000 p=1.00 完全持平, pass判据(>0.4654)未达. 机制按设计正确过滤了 diag-overfit 候选(iter1 拒3个正delta负CI候选, 仅采纳正probe-CI), 但切断"诊断题=选择题"同源链并未提升 test → 同源选择过拟合非绑定瓶颈, 0.4654 是对 2-iter-from-S0 evolver 的稳健天花板.
metadata:
  type: project
---

# Task5 诊断-选择解耦 Probe-Set 实验终局 (2026-08-20)

## 一句话结论
**机制工作但未破天花板**: probe-split evolver test=106 4-seed=**0.4654**, 与 Task3(0.4654)**逐题配对 t=-0.000 p=1.00 完全持平**; 切断"诊断题=选择题"同源过拟合链**未**提升 test → 同源选择过拟合**非**绑定瓶颈, 0.4654 是 2-iter-from-S0 evolver 的稳健天花板.

## 实验设计 (隔离对照, 遵 baseline 公平铁律)
- **Treatment**: FULL control + 仅 `--probe_split` (单变量隔离, 不叠 `--data_driven_focus`/`--select_by_ci`).
- **切分**: select=58 (5 train + 53 val) → 固定 seed=42 shuffle → 不相交 **probe=29 (选择信号) + diag=29 (诊断)**. flag 关时 diag=probe=select (字节复现 0.462 路径).
- **路由**: baseline/candidate/gate 的 `eval_on` → probe_items; `collect_failures` → diag_items. winner 选择逻辑(novel_positive/ci>0/argmax)不动, 仅 delta/ci 来源从 select 改为 probe.
- **零额外 S0 算**: diag∪probe=select ⊆ s0_cache(71 keys 超集).
- 种子=S0(initial.md), init_skill=v2(b98a9abd), 与 Task3/Task4 同起跑, 公平无嵌套✅.
- 产物: `run_20260820_140332/best_skill_evolved.md`, hash b98a9abd10→**181981e9** (≠v2≠Task3≠SkillOpt, 公平✅).

## 机制行为证据 (机制按设计工作)
evolver_log.json: probe=29 diag=29 n_select=58, baseline_val(probe) pass=0.184 delta=-0.002 CI=-0.092.
- **iter1** (focus=Implication vs biconditional): 4 候选, **3 个正 delta 但负 CI 被 CI 门拒** (cand0 +0.032/CI-0.057, cand1 +0.067/CI-0.021, cand2 +0.021/CI-0.037) → 这些正是"diag 过拟合程序": 在诊断题上 delta 正, 但在 probe(诊断时未见内容)上 CI 负. 仅 cand3 (+0.101/CI+0.037) 被采纳. **解耦机制正确过滤了同源过拟合候选.**
- **iter2** (focus=Existence vs uniqueness): cand2 (+0.297/CI+0.186) 大幅胜出被采纳.
- 最终 val(probe) delta: -0.002 → **+0.297 (CI+0.186)**, skill 3655→4861 chars.

## test=106 4-seed 结果 (eval_k=3, seeds=4)

| skill | mean | per-seed | ΔvsS0 |
|---|---|---|---|
| s0 | 0.2909 | [0.2830, 0.2862, 0.2830, 0.3113] | — |
| **evolver (probe-split)** | **0.4654** | [0.4843, 0.4654, 0.4748, 0.4371] | +0.1745 |
| skillopt | 0.5566 | [0.5409, 0.5660, 0.5440, 0.5755] | +0.2657 |

产物: `multiseed_test_meta_probe/multiseed_test_eval.json` (独立 out_dir, 未覆盖 Task3/4 旧 JSON).

> ⚠️ **复现性 caveat (2026-08-21 独立复核确认)**: `multiseed_test_meta/multiseed_test_eval.json` 汇总 JSON 是 Task4 coverage(0.3632) 的, 非 Task3(0.4654) — 被 coverage run 覆盖回写了. 复现 Task3 的 0.4654 须读 `multiseed_test_meta/per_evolver_cde0bb92_seed{1..4}.json` 平均, 勿读该目录汇总. probe-split 配对检验用的是 per-seed 文件 (独立复现 t=-0.000 p=1.00 一致✅), 结论不受影响. 详见 docs/phase2_issues.md 末条.

## 逐题配对检验 (probe-split 181981e9 vs Task3 cde0bb92, n=106)
- mean diff = **−0.0000**, sd=0.1560, **t=−0.000, p=1.0000** (df=105).
- 两 skill 逐题 grand mean 均 = 0.4654.
- probe-split 更好 38 题, Task3 更好 33 题, 持平 35 题 → 逐题有 churn 但净差为 0.
- **结论: probe-split 与 Task3 在 test 上统计不可区分.**

⚠️ 注意: s0 跨 run 不同 (本 run 0.2909 vs Task3 dir 0.3011), 因 EVAL_TEMP=0.3 每次 re-sample; 故绝对均值精确持平部分属巧合, 逐题配对检验(p=1.00)才是干净对比, 已证无信号.

## 判定 (回答 pass criterion)
1. **test > 0.4654 (Task3 best)? 否, = 0.4654 持平, 逐题 p=1.00.** pass 判据未达.
2. **机制工作吗? 是.** iter1 正确拒了 3 个 diag-overfit 候选(正delta负CI), 仅采纳正 probe-CI 候选. 解耦链按设计切断.
3. **同源选择过拟合是绑定瓶颈吗? 否.** 切断该链 → test 不变 → 同源选择过拟合**非**val→test 脱节的根因.

## 释义 — 这对研究假设意味着什么
- **原假设** ([[eval-signal-debiasing-diagnosis-20260820]]): val→test 脱节由"诊断题=选择题"同源过拟合造成 → 切断应提升 test.
- **实验证伪了该假设的"切断即提升"预测**: 解耦后 val(probe) 升到 +0.297(高于 Task3 的 +0.268, 但集不同不可直比), **test 持平 0.4654**. val 信号甚至更乐观, test 仍不动 → val→test 脱节**不在选择信号层**.
- **真瓶颈定位收窄**: 0.4654 是 2-iter-from-S0 evolver 的稳健天花板 — Phase C 结构自进化(0.37)、Task4 coverage(0.3632)、Task3 data_driven(0.4654)、本 probe-split(0.4654) 多机制**全部**收敛到 ~0.36-0.47, 单一机制改动无法突破. 瓶颈可能在 (a) skill 内容本身过拟合 val 特定定理/选项内容(与选择解耦无关), 或 (b) 2-iter 深度不足 / S0 起点限制.
- **与 survey 收敛** ([[ea-skill-evolution-survey-20260819]]): 用户 evolver 是"纯 keep-best 单 lineage 连 SePO archive 都没有"最弱一档; 多机制无法破天花板 ⇒ 支持转向**种群+archive**范式(Facet-Tree GA / DIVE 多种群 bootstrap / DarwinX 种群吸收 in-loop 偏差), 而非继续在单 lineage 上调机制 flag.

## 下一步候选 (破天花板方向)
- **不再**: 在单 lineage 上增机制 flag (probe_split/data_driven/coverage/structure 均已证不破 0.4654).
- **转向**: 种群+archive 的真 GA (Facet-Tree v2, [[facet-tree-skill-evolution-20260819]]); 或更长 iter(10-iter 曾 +0.038 但不显著); 或换 init(S0 起点限制假设).
- probe-split flag 保留 (机制正确, 零额外算, 默认关不破坏复现), 可作未来种群方法的子组件 (winner 选择仍可用 probe 解耦).

Related: [[eval-signal-debiasing-diagnosis-20260820]], [[evolver-data-driven-focus-result-20260817]], [[evolver-coverage-diagnosis-result-20260817]], [[phase-c-structure-evolution-result-20260819]], [[ea-skill-evolution-survey-20260819]], [[facet-tree-skill-evolution-20260819]], [[baseline-fairness-audit-20260817]]
