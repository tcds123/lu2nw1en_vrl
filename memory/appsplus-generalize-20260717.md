---
name: appsplus-generalize-20260717
description: APPS_Plus interview 代码域首次 test 泛化 - best test 0.635 vs noskill 0.575 (+6pp), 三层单调 noskill<seed<best; 代码方向复活
metadata:
  type: project
---

# APPS_Plus interview 首次 test 泛化 (2026-07-17) 🔥

承接 [[appsplus-baseline-20260717]]。optimize 跑 5/80 steps 后趋势明确(val +0.5pp 停滞, 4/5 reject), kill 提前验泛化。

## optimize val gate 轨迹 (kill 前, selection=val=395, metric=hard)
- step0 baseline: 0.5975
- step1: 0.5848 → reject
- step2: 0.6025 → **accept_new_best** (best_skill.md = 此步 candidate, 1797→4857 chars)
- step3: 0.5873 → reject
- step4: 0.5823 → reject
- step5: 0.6000 → reject
- ⇒ val gate 仅 +0.5pp, candidate 在 0.58-0.60 抖动。表面与 mbpp 同型 (val 微涨停滞)。

## 干净 test 对位 (eval_only, test=200, 同 split seed42, temp=0, 零 429, agent_ok=200/200)
| skill | test hard | test soft | vs noskill (hard) |
|---|---|---|---|
| noskill (empty) | 0.5750 | 0.6025 | — |
| seed (initial.md, 手写) | 0.6000 | 0.6375 | +2.5pp |
| **best (step2 优化后)** | **0.6350** | **0.6758** | **+6.0pp** |

## 核心结论 (代码域突破)
- **best test 0.6350 > noskill 0.5750 (+6.0pp hard, +7.3pp soft)** —— **代码域首次 test 泛化正向结果**。
- 三层单调递增 noskill < seed < best:
  - 手写 seed 已 +2.5pp (skill 内容本身有用, 与 mbpp 上 seed≈noskill 零提升不同)
  - SkillOpt 优化再 +3.5pp (优化器在 seed 基础上有增量收益)
- **反直觉**: val gate 仅 +0.5pp 却泛化到 test +6pp。val=395 充足 (vs mbpp val=38/75), val 信号更稳但 lift 小; test=200 上 lift 反而大。可能 val 子集偏难/偏易, 但 monotone 排序保持 → 优化方向正确。
- 对比 mbpp_pro: best test 0.513 < noskill 0.528 (−1.5pp, 不泛化); appsplus +6pp (泛化)。**appsplus interview 是比 mbpp/humaneval 更适合 skill 优化的代码基准**。

## 对论文方向的意义 (重大)
- 代码方向**复活**: 此前 mbpp/humaneval 双 null + APPS 未测, 代码域 skill 优化可行性存疑。appsplus 干净泛化证明代码域**有 headroom 且可优化**。
- [[dataset-transition-narrative-20260716]] 的"换更难基准"决策正确: appsplus interview (竞赛题, 多步推理, stdin/stdout) 让 skill 有作用空间, 单函数 mbpp/humaneval 没有。
- 可进 Step1 方法对比 (GRPO+ATE vs SkillOpt val-gate, [[research-direction-skillopt-20260714]] Step1): appsplus 作主基准, 数学(LiveMath)作通用性补充。
- 控制变量诊断 ([[skillopt-code-result-20260715]] 变量1-4) 现可在 appsplus 上跑 (headroom 足, 泛化成立, 能区分各组件贡献)。

## 待办 / 下一步
- ⏳ (可选) 跑完 optimize 80 steps 看 best 是否还能涨 (当前 best 来自 step2, 后续 78 步可能找到更好 skill)。但节奏 ~30h, 优先级低 —— test 泛化结论已成立。
- ⏳ 进入 Step1: 在 appsplus 上对位 SkillOpt val-gate (基线A) vs GRPO+ATE (变量3, 论文核心)。
- ⏳ 扩 test 到 500+ 拿更稳的泛化数字 (当前 test=200, +6pp 已显著但可加固)。

## 复现命令
```bash
cd /data/zhuldz/skillopt
export QWEN_CHAT_API_KEY="msk-..."
# 三组对位 (同 split seed42, temp=0, 28RPM 限流)
BENCH=appsplus TAG=appsplus_noskill_200 bash run_code.sh noskill 200
BENCH=appsplus TAG=appsplus_seed_200 bash run_code.sh evalskill skillopt/envs/appsplus/skills/initial.md 200
BENCH=appsplus TAG=appsplus_best_200 bash run_code.sh evalskill outputs/appsplus_optimize_ep4_optimize_ep4/best_skill.md 200
```

Related: [[appsplus-baseline-20260717]], [[appsplus-env-20260716]], [[dataset-transition-narrative-20260716]], [[skillopt-code-result-20260715]], [[code-benchmark-headroom-20260715]], [[research-direction-skillopt-20260714]]
