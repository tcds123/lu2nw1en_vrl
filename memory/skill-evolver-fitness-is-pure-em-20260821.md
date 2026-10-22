---
name: skill-evolver-fitness-is-pure-em-20260821
description: skill_evolver_math.py 演化链路 fitness = 纯 EM pass_rate(非 70/30 LLM 评判); C-judge=pro 对 Stage 0 fitness 是空操作, 仅回灌 GRPO 时生效。Facet-Tree 问题4。
metadata:
  type: reference
---

# skill_evolver 演化链路 fitness = 纯 EM (非 70/30)

## 核心事实 (2026-08-21 实读)

Facet-Tree Stage 0 进化对象 = `skill_evolver_math.py` 演化的 Line A math skill。该链路 fitness **是纯确定性 EM pass_rate**, **不含 C 模型 LLM 评判**:

- `eval_on`(`tf_global_v2_math.py:289`) → `run_one`(`:142`) → `math_evaluate` = `skillopt.envs.livemathematicianbench.evaluator.evaluate` (`/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/evaluator.py:41-62`)。
- `evaluate` 纯 EM: `is_correct = float(predicted_label == correct_label)` (行52), 仅 `re` 抽 `<answer>` + 字符串比对, **零 LLM 调用**。
- `eval_on` fitness 合成: `mean_pass = Σ pass_rate/n`, `pass_rate = n_pass/k` (k 次采样投票), `delta = pass_rate − s0_cache[qid].pass_rate`。**无加权、无 LLM 评判、无 CModelJudge**。
- `skill_evolver_math.py` + `tf_global_v2_math.py` **均不 import** `judge/qwen_235B.py` / `CModelJudge` / `c_interface` (grep 确认空)。

## 70/30 真实存在于另一条链路

70%单测+30%LLM 评判真实存在于 `c_interface.py`: `load_c_model`(`:362`) + `compute_reward`(`:373`) → `calculate_final_reward_with_verification`(`:442`)。但那是 **GRPO 训练链路 (A 模型奖励)**, 走 `c_interface` import `qwen_235B.CModelJudge`(`:17`)。**两条链路 fitness 来源不同**。

## 对 Facet-Tree Stage 0 的影响 (Facet-Tree 问题4)

| | skill_evolver 演化链路 (Stage 0 实际跑) | c_interface GRPO 训练链路 |
|---|---|---|
| fitness | EM pass_rate (纯确定性) | 70/30 合成 (含 LLM) |
| 调 qwen_235B? | ❌ 否 | ✅ 是 |
| C-judge=pro 决策 | **空操作** (改了 fitness 不变) | **真前置** (隔离限流) |

**结论**: gonogo §七 step2「改 `judge/qwen_235B.py` 三参数」**对 Stage 0 的 fitness 是空操作**。Stage 0 判据 (test 4-seed > 0.4654) 走 EM, 不经 C, 不受 pro 账户影响。C-judge=pro 真正生效 = Stage 0 产出 skill 后**回灌 GRPO 训练**, 是 Stage 0 之后的前置, 非 Stage 0 阻塞项。plan step2 据此降级。

## 订正

- 设计 §2.6 within-branch fitness 已订正为「EM pass_rate + CI, 非 70/30」(70/30 仅 GRPO 训练阶段)。
- 设计 §九-8 已加适用边界: C-judge=pro 仅 GRPO 回灌时生效, Stage 0 不阻塞。
- 行1100 已有 issue (C 换 pro 解 429) 的「保留 LLM 评判维度」对 skill_evolver 链路是误归因——该链路无 LLM 评判维度可保留。其限流隔离价值仅在 C 被调用 (GRPO 回灌) 时成立。

## 教训

CLAUDE.md 项目概述的 70/30 reward 是**项目级**描述, 不等于每个子链路都用它。skill_evolver 子链路用更简化的 EM。读子链路代码优于信项目概述; 「改 X 三参数」类指令写进 plan 前必须 `grep import` 确认目标链路真调 X。

Related: [[facet-tree-skill-evolution-20260819]], [[facet-tree-s0-startpoint-20260821]], [[theorem-type-field-facts-20260821]]
