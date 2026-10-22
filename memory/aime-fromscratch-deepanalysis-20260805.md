---
name: aime-fromscratch-deepanalysis-20260805
description: AIME同域从头链负结果深度分析——非bug、是核心创新(诚实门控)的正面验证、失败模式跨题可泛化性是算法改进杠杆
metadata:
  type: project
---

# AIME 同域从头链负结果深度分析（2026-08-05）

## 背景

AIME 从头全链实验：S0 -> v2_aime(iters=2,g=4) -> evolver_aime(iters=4,K=4) -> eval AIME25。AIME24(30) 拆 train18/val12(seed42)，AIME25(30) held-out test。结果：优化器全 KEEP PREV，best skill == S0（未改动）。用户原预判"换数据集该泛化"，结果完全相反，认为此实验很有代表性。

## 用户 08-05 重新定位（关键）

> "根本不需要在 aim25 上验证，你 aim24 自己的门控都过不了完全说明了我们的算法没有能够优化 skills，这就是我们的核心创新点。我们的自进化就是要从 baseline 上找到能够持续进化的 global skills，这个实验说明了我们目前的算法没有跨数据集泛化的能力，这是个好消息，关键是要通过这个部分优化算法，看看为什么在 livemathbench 上有效果，在 aim24 上却不行。"

**结论转向**：AIME24 自身门控（同分布 train->val）过不了 = 算法未产出可泛化 skill 的直接判决，**无需再拿 AIME25 验证**（再验 = 怀疑自己的门控）。这是**核心创新点的正面验证**，不是方法失败。

## ① 非 bug 核验（三条证据链）

| 证据链 | 07-29 真 bug（对照） | 本次 08-04/05（正确） |
|---|---|---|
| 候选生成 | 失败：产不出新 skill | ✅ v2 产 5KB/4.8KB 候选；evolver 4轮×4=16 novel 候选（hash 全不同） |
| 训练信号 | - | ✅ v2 train_mean_best_delta +0.148(iter1)/+0.259(iter2)，train 上真改对了 |
| 门控判定 | bug：放行未改动 skill（假阳性+15.7pp） | ✅ iter1 val CI_lower -0.028<0 正确拒；evolver 4轮全 CI≤0 拒；iter4 pass 0.611<base 0.678 反退化拒 |

失败模式蒸馏质量高（experience_iter1.md 含 Power of a Point、maximal vs maximum、collinearity、conditional probability sample space 等 10+ 条 symptom/root-cause/diagnostic 真AIME内容，非 v1 prompt措辞陈词滥调）。**算法在正确工作，只是没找到统计显著的改进。**

证据文件：
- `rl/train/verl1/outputs/tf_skill_aime_v2/run_20260804_223830/tf_run_log.json`
- `rl/train/verl1/outputs/skill_evolver_aime/run_20260804_234339/evolver_log.json`

## ② AIME24 vs AIME25 差距（真实但非主因）

| | S0 pass | 主题分布 |
|---|---|---|
| AIME24 select(30) | 0.567(mean_vote)/0.678(evolver base) | geometry 9 / NT 9 / algebra 4 / comb 3 |
| AIME25 test(30) | 0.497(4-seed mean) | geometry 12 / NT 9 / algebra 2 / comb 4 |

差距 ~7-18pp，AIME25 更难（几何更多）。**非主因**：优化失败发生在 AIME24 train->val（同分布）上，连同分布都不泛化，24/25 差距只拖累 test 表现，不是优化失败瓶颈。

## ③ 真正根因 = 失败模式跨题可泛化性不足

- v2 iter1：train +0.148 -> val +0.111（点估正）但 CI_lower -0.028 -> 拒。train 改进没可靠迁移到 val。
- AIME 题目异质：每题独特话题组合，"Power of a Point"只帮几何题；15条失败模式各管一窄块，组不成泛化 global skill。
- 对比 LiveMath（选择题）：失败模式（选项措辞检查、"更强结论能否证明"）利用所有题共享的选择结构，一条帮很多题 -> global skill 泛化 -> +16.1pp。
- 叠加低统计功效：val=12、k=3，1题翻转=+0.083，候选 val 点 delta +0.111（~1.3题）落噪声带内。

> ⚠️ **S₀ 出处 confound（2026-08-05 补）**：上述"AIME 失败模式逐题特异、组不成泛化 global skill"的根因仍成立（任务结构属性，见 [[aime-vs-livemath-content-diagnosis-20260805]]）。但"AIME S₀ 已吃掉形态级 headroom"这一条有 confound--AIME S₀ = `data/skill_aime_initial.md` 是**项目内手写**（AIME 非 SkillOpt 自带 env，无官方 seed），其形态级饱和度由撰写者决定（2239B，比官方 LiveMath seed 1018B 详细 2.2×）。故"AIME 是泛化边界"的结论强度依 S₀ 饱和度而定：当前（较满的）S₀ 上成立，更弱 S₀ 上待证。用户决策：不刻意降低 baseline，后续工程化 AIME S₀ 去混淆，暂不跑。负结果本身（门控诚实拒绝、未自欺）不依赖该归因。详见 `docs/phase2_issues.md` 同日条目 + 汇报文档 §2.5/§4.1⑥。

## ④ 核心结论（用户 08-05 定调）

1. **这是核心创新点的正面验证**：自进化方法的门控（bootstrap CI on held-out val）能在找不到可泛化改进时**诚实拒掉 train-overfit 候选、退回 S0**，而非自欺放行。对比 07-29 bug 正是门控失效放行了未改动 skill。AIME24 同分布 train->val 全候选被拒 = 诚实门控在工作。
2. **好消息**：把"方法有效性决定因素 = 失败模式跨题可泛化性"钉死。
3. **算法改进方向**：诊断 LiveMath（共享 MCQ 元结构->失败模式跨题可复用->+16.1pp）vs AIME（异质自由作答->失败模式逐题特异->组不成泛化 global skill）的差异，让 evolver 的失败模式蒸馏/合成在异质任务上也产出跨题可复用内容（按题型聚类蒸馏 / 跨题失败模式去重抽象 / 形态无关诊断程序）。

## 已撤销的待补实验

原计划测被拒的 v2 iter1 候选（run_20260804_223830/skill_iter1.md, hash 87af48f4, train+0.148/val CI-0.028）在 AIME25 上以区分"CI过保守 vs 失败模式不泛化"。**用户 08-05 决定不跑**：AIME24 自身门控过不了已是判决，再验等于怀疑自己的门控。曾短暂启动（PID 51967）后立即 kill，无结果。aborted run 残片在 `outputs/tf_skill_aime_v2/aime25_test_rejected_iter1/`（仅 .progress，已可清理）。

相关：[[aime-from-scratch-result-20260804]]、[[aime24-generalize-result-20260731]]、[[evolver-unchanged-skill-bug-20260729]]、[[evolver-ablation-result-20260804]]
