---
name: baseline-fairness-audit-20260817
description: 基线公平性审计 - 代码+实验双重核实"用 SkillOpt-best 当基线/种子"的不公问题, 固化铁律与公平对比现状表(含统计)
metadata:
  type: project
---

# 基线公平性审计 (2026-08-17)

## 起因
用户记忆: "有个基线是用了 skillopt 的 best 又作为基线进行的, 这是不对的, 要用我们自己的算法和 skillopt 用相同的基线对比".
核实结论: **记忆正确**。问题在 2026-08-07 已被用户亲自指正并立"方法论铁律" ([[algorithm-future-optimization-directions-20260805]] 顶部), 项目记录已改判。本次审计复核代码+实验链, 固化结论。

## 不公来源: 两类 SkillOpt 嵌套实验

均指把 evolver 种在 `skill_v0002.md` (= SkillOpt-best, md5 bf82b4608513) 上再做优化, 非同一起跑线:

1. **evo_seed 系列 (2026-08-03)**: `skill_evolver_math.py --init_skill .../skill_v0002.md`. 4-seed test=106=0.5692 宣称"超 SkillOpt +2.3pp"; 但 16-seed 复评 +0.59pp p=0.42, split-half 逐题符号一致率 43%(<随机), **已自我证伪回 parity**。见 [[skillopt-headtohead-start-20260803]]:246。
2. **Run F (2026-08-07, 数值最强)**: `evo_seed + M3-gate + k=5`, test=106=0.5597, Δvs S0=+26.9pp, t=6.82 p=6e-10。种子=SkillOpt-best = 模块嵌套 → **已改判, 不能进主结果/论文**。见 [[runf-test-result-20260807]]:10。

## 方法论铁律 (现行最高优先级, [[algorithm-future-optimization-directions-20260805]]:10)
- (a) **主结果必须用 S0 (initial.md) 当种子/基线**, 公平同起点;
- (b) SkillOpt-best 只能当"高 floor 种子"做嵌套实验, 产物单独标注, **不竞逐主结果**;
- (c) 最终论文主结果 = 我们算法 vs 其它最优算法(含 SkillOpt)的 **S0 公平对比**。

## 公平对比现状表 (test=106, EVAL_TEMP=0.3, k 投票, split 1-3-6 seed42)

| 方法 | seed来源 | test=106 | Δvs S0 | 统计 | 公平? |
|------|---------|---------|--------|------|------|
| S0 (initial.md, 共享基线) | — | 0.301–0.305 | — | 基线 | ✅ 基线 |
| SkillOpt-best (别人方法) | S0 起(SkillOpt自有) | **0.5464** | +24.5pp | 4-seed, 对比标的 | ✅ 公平 |
| v2 (我们, global追加合成) | **S0 起** | 0.354 | **+6.6pp** | 4-seed t=2.78 p≈0.006 CI[+0.020,+0.113] | ✅ 公平坐实 |
| evolver 2-iter (我们,主结果) | **S0→v2 起** | 0.462 | **+16.1pp** | 4-seed t=9.02 CI[+0.126,+0.196] | ✅ 公平坐实 |
| evo_seed / Run F (种 SkillOpt-best) | SkillOpt-best | 0.559–0.569 | +26.9pp | — | ❌ 嵌套, 已排除 |

**核心差距**: 公平起跑最强=evolver 2-iter 0.462, 落后 SkillOpt **8.4pp** (p=0.0002 显著)。即"对标最优 SkillOpt 尚不足"。AIME 跨任务亦阴性 ([[aime-fresh-rerun-result-20260807]])。

## 代码层核实 (eval_test_multiseed.py + tf_global_v2_math.py + skill_evolver_math.py)

1. **S0 定义一致**: `tf_global_v2_math.py:71` `S0_DEFAULT = .../livemathematicianbench/skills/initial.md` (md5 9af736315893, 1018 chars). SkillOpt 与我们同源共用。✅
2. **split 一致**: 评测均用 `SPLIT_136 = livemathematicianbench_1-3-6_seed42` (test=106)。SkillOpt 评测同 split。✅
3. **评测路径公平**: 所有 skill 走同一 `run_one`/`eval_skill_on` (同 backend=DeepSeek-V3.2, 同 EVAL_TEMP=0.3, 同 k, 同 math_evaluate 确定性验证器)。S0 也 k 投票对称 (`precompute_s0`)。✅
4. **evolver 主结果链路 = S0→v2→evolver, 全自家算法**: `skill_evolver_math.py:352` 默认 `--init_skill = V2_SKILL_DEFAULT` (run_20260728_131934 的 v2 skill, md5 b98a9abd)。v2 本身从 S0 起。故 evolver 主结果(12633201,+16.1pp)**无 SkillOpt 嵌套, 公平**。✅
5. **嵌套仅发生在显式传参**: 仅 evo_seed(08-03)/Run F(08-07) 传 `--init_skill skill_v0002.md` 时嵌套。✅

## ⚠️ 脚本默认值陷阱 (防未来误用)

`eval_test_multiseed.py:27` 的 `EVOLVER_BEST` 默认指向 `run_20260727_163814/best_skill_evolved.md` —— **这是未改 skill 的 bug run, 已作废** (见 [[evolver-unchanged-skill-bug-20260729]], best==init md5 911597f7df)。
evolver 真实主结果产物在 `run_20260730_111306/best_skill_evolved.md` (md5 12633201455d)。
**跑公平对比时必须显式 `--evolver_best .../run_20260730_111306/best_skill_evolved.md`, 勿用脚本默认值**, 否则会评测到未进化的 init skill。

## 下一步 (对应铁律"对应行动")
回到同一 S0 起跑改进 evolver 能力, 目标公平条件下追平/超过 SkillOpt 0.5464。当前公平最优 0.462, gap=8.4pp。
改进线索 (来自 08-03 badcase 诊断): SkillOpt 强 meta-option 题(赢题77%是meta), evolver 弱 meta-option; 二者互补但朴素 splice 稀释 -4.7pp。需最小定向 patch 放大 meta-option 优势, 而非整段合并。详见 Task3 规划。

Related: [[algorithm-future-optimization-directions-20260805]], [[runf-test-result-20260807]], [[skillopt-headtohead-start-20260803]], [[evolver-test-generalize-20260730]], [[evolver-unchanged-skill-bug-20260729]], [[aime-fresh-rerun-result-20260807]]
