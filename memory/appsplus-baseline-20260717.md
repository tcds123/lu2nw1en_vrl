---
name: appsplus-baseline-20260717
description: APPS_Plus interview noskill baseline 跑通 - n=50 hard=0.50/soft=0.54 零429; optimize 4ep 后台跑中验证 test 泛化
metadata:
  type: project
---

# APPS_Plus interview baseline + optimize 启动 (2026-07-17)

承接 [[appsplus-env-20260716]]。本次补完待办：run_code.sh 加 appsplus 模式 + 冒烟 + noskill baseline + 启动 optimize。

## 实现补完 (2026-07-17)
- `run_code.sh` 加 `BENCH` 环境变量 (mbpppro|appsplus)，appsplus 自动切 CFG/configs/appsplus/default.yaml、DATA=APPS_Plus/data/v1、DATASET_GLOB=data_with_id.json、追加 `env.difficulty_filter=interview`。eval_only.py + train.py 注册此前已存在 (见 [[appsplus-env-20260716]])。
- 用法：`BENCH=appsplus bash run_code.sh {smoke|noskill|evalskill|optimize}`

## 冒烟 (smoke n=5, 限流 28RPM)
- ✅ full-pass=0.6000, avg_soft=0.6000, agent_ok=1.0, 零 429/RolloutException。管道打通。

## noskill baseline (test=50, 同 split seed42, 限流 28RPM)
- **hard=0.5000, soft=0.5400**
- agent_ok=50/50, 零 exception/429。token bucket 限流生效，无污染。
- split: train=790 / val=395 / test=2765 (interview 3952 题 2:1:7)。

## 判读 (vs 旧基准)
- appsplus interview baseline hard 0.50 ≈ mbpp_pro 0.5283，比 mbpp 略低但**没有明显更难**。soft 0.54 → 46% headroom 仍在。
- baseline 偏高（理想 0.2-0.4 未达），但用户决策：不纠结数据集探查，直接跑 optimize 看 test 能否泛化（判死 mbpp 的核心问题）。
- val=395 充足（vs mbpp val=38/75），val 过拟合风险应大幅降低。

## optimize 4 epochs (后台跑中, PID 94361)
- 启动: `BENCH=appsplus TAG=appsplus_optimize_ep4 bash run_code.sh optimize 4`
- log: `/data/zhuldz/skillopt/outputs/appsplus_optimize_ep4.log`
- out: `/data/zhuldz/skillopt/outputs/appsplus_optimize_ep4_optimize_ep4/`
- config: epochs=4 steps/epoch=20 (auto) batch=40, gate metric=hard, val=395 selection
- baseline-on-selection 跑中（395 题）。预计 wall ~1-2h (限流 28RPM)。

## optimize 进度与节奏 (2026-07-17 14:05 更新)
- 4/80 steps 完成 (~2h)。history.json gate 轨迹 (selection=val=395, metric=hard):
  - step0 baseline: val=0.5975
  - step1: 0.5848 → reject
  - step2: 0.6025 → **accept_new_best** (best)
  - step3: 0.5873 → reject
  - step4: 0.5823 → reject
  - ⇒ 4 步仅 +0.5pp lift, 3/4 reject。模式与 mbpp (val 微涨) 相似, 待观察是否 test 不泛化。
- **节奏问题**: 每 step ~25-40min, evaluate_s (395题selection eval) 是大头 (1234-2059s)。80 steps 全程 ~30h。
  - 每 step 含: 6 rollout (40题) + reflect + aggregate + 1 次 395 题 selection eval。selection eval 频率太高是瓶颈。
- **决策 (用户 14:05)**: 先暂存。若 18:00 前能出结果就试, 否则跑全量 80 steps。
  - 计划: 让进程后台跑, ~17:30 评估。val lift 明确(>+3pp 多accept) → 用 best_skill 提前 eval_only test 验泛化; val 仍微弱(+0.5pp 大量reject) → 直接 best_skill eval test 确认泛化负结论。
- best_skill 路径: `/data/zhuldz/skillopt/outputs/appsplus_optimize_ep4_optimize_ep4/best_skill.md` (当前 = step2 candidate)
- noskill test baseline (n=50): hard=0.50。泛化判别阈值 = best test vs 0.50。

## 关键判别 (待 optimize 完成)
- best-on-val test > noskill 0.50 → appsplus headroom 真实且可泛化, 代码方向活, 进 Step1 方法对比 (GRPO+ATE vs SkillOpt)
- best-on-val test ≤ noskill 0.50 → interview 单函数竞赛题同 mbpp 命运, 换 competition 子集或更难基准

Related: [[appsplus-env-20260716]], [[dataset-transition-narrative-20260716]], [[api-concurrency-deepseek-20260716]], [[skillopt-code-result-20260715]], [[code-benchmark-headroom-20260715]]
