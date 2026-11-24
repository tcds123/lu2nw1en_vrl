---
name: math-trial-direction-20260724
description: 🚨方向临时切换 - 暂试数学bench(LiveMathematicianBench)作主试验场; 代码线不放弃, 降级为对照/上界; 导师建议跳出纯代码
metadata: type: project
---

# 方向临时切换: 暂试数学 bench (2026-07-24) 🚨

## 决策来源
导师建议: 不做代码领域的 bench, 看综述里其他领域 (skillbench/数学/其他)。
用户决策 (2026-07-24): **暂试数学 bench**, 但**不放弃代码**, 代码线降级保留。

## 为什么选数学 (而非 SkillsBench / SWE / 交互式 agent)
综述 `docs/survey_agent_skill_evaluation_evolution/` §4-5 列了 6 类 skill bench + 通用 bench。逐类筛选:
- **数学推理 (AIME/Omni-MATH/DeepScaleR/LiveMathematicianBench)** ✅ 首选: 确定性 `<answer>` 验证器、单轮、无 Docker/容器、本机零 GPU 能跑、headroom 大
- Utility (SkillsBench 86任务/11领域) ⚠️: 配对评测最对口但需 BenchFlow task.md 包+多领域, 工程量大
- Generation (SkillLearnBench) ⚠️: 测 skill 自动生成质量, 小
- Retrieval/Routing (SRA-Bench/SkillRouter) ❌: 测检索精度非内容优化
- Safety / SWE / Real-world / 交互式 ❌: 容器/浏览器/仓库, 工程量大本机难复现

## 三个硬证据支撑数学
1. **已在数学拿到干净正结果** ([[skillopt-step0-results-20260714]]): LiveMathematicianBench no-skill 24.19% → seed 32.26% → SkillOpt-best **54.03% (+21.77pp)**, greedy, 0 失败, 可复现
2. **代码方向已判死**: [[lcb-new-optimize-20260721]] 无污染条件下 skill 优化 LCB hard 零收益; [[appsplus-full-test-reversal-20260718]] -2.1pp
3. **与 07-23 发现自洽**: [[direction-global-not-pertask-20260723]] 真问题=TF-global 比 S_0 差。数学题共性更强, global skill 更可能超 S_0 — 正好解这个真问题

## 代码线定位 (不放弃)
- 降级为**对照线 / 上界对照**, 不作主试验场
- [[tf-pertask-positive-20260723]] (LCB hard +8.4pp) 和 [[tf-global-reversal-samecond-20260723]] 结果存档保留
- 代码 env (LCB/APPS_Plus) 保留, 必要时回切或作消融

## 数学 bench 执行环境 (已就绪)
- 路径: `/data/zhuldz/skillopt`
- env: `skillopt/envs/livemathematicianbench/` (adapter/dataloader/evaluator/rollout/skills)
- 启动: `bash run_midea.sh {smoke|noskill|optimize|evalskill}`
- backend: qwen_chat (DeepSeek-V3.2 via aimpapi), max_turns=1, 确定性 `<answer>` 验证器
- split: ratio 2:1:7 → train=35, val=18, test=124 (idhash c2f36ea25493)
- ⚠️ `run_midea.sh` 内 key 是旧 msk-bf2324 (已 401), 需换新 key msk-4601b27 (见 [[tf-grpo-skill-result-20260723]])

## 进展 (2026-07-24)
- ✅ 新 key msk-4601b27 在 skillopt env (qwen_chat backend) 验证可用 (curl 通)
- ✅ 5 题 smoke 链路打通: no-skill acc=0.000 (格式坑, 见 docs/phase2_issues.md), seed skill acc=0.200 → skill 价值立刻可测, headroom 显著
- ✅ 验证 aimpapi 多模型 tool_call: Qwen3-235B / DeepSeek-V3.2 均支持原生 tool_call, 推翻 core.py:509 旧注释 (已更新注释)
- 🔄 全量基线 (test=124, noskill/seed/best=skill_v0002) 同条件重测中, 后台跑 /tmp/math_full_baseline.sh
- ⏳ 把 tf_skill_optimizer.py 切到数学 env, 测 TF-global / TF-per-task vs S_0
- ⏳ 8卡4B GRPO 训练线 vs TF 数学同表对位
- 代码线结果继续作为对照保留, 不删除

Related: [[direction-global-not-pertask-20260723]], [[skillopt-step0-results-20260714]], [[tf-global-reversal-samecond-20260723]], [[dataset-constraint-20260715]], [[training-vs-tf-parallel-plan-20260723]]
