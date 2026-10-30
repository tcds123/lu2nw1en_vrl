---
name: skill-evolver-fitness-is-pure-em-20260821
description: skill 演化线 fitness = 纯 EM pass_rate，不走 C 模型 / 70-30 LLM 评判（详细 C/GRPO reward 状态见 c-model-and-grpo-reward-status-20261030）
metadata:
  type: reference
---

# skill 演化线 fitness = 纯 EM（不走 C 模型）

`eval_on` → `run_one` → `math_evaluate`（`skillopt/envs/.../evaluator.py:41`）= `is_correct = predicted_label == correct_label`，**零 LLM 调用**，fitness = mean pass_rate + bootstrap CI。`skill_evolver_math.py` / `tf_global_v2_math.py` 均不 import `c_interface` / `CModelJudge`（grep 已核实）。

→ 对演化线而言，改 C 模型的任何参数都是空操作；C 只与早期的 GRPO 训练 reward 有关，详见 [[c-model-and-grpo-reward-status-20261030]]。

**教训**：CLAUDE.md 的 70/30 是项目级早期描述，不等于每个子链路都用它。读子链路代码优于信项目概述。

Related: [[facet-tree-skill-evolution-20260819]], [[c-model-and-grpo-reward-status-20261030]]
