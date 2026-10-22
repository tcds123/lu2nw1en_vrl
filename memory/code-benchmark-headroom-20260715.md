---
name: code-benchmark-headroom-20260715
description: 代码基准 headroom 探查 - HumanEval Pro noskill 0.41 比 mbpp_pro 0.53 更未饱和; mbpp扩val实验进行中
metadata:
  type: project
---

# 代码基准 headroom 探查 (2026-07-15)

背景: [[skillopt-code-result-20260715]] 发现 mbpp_pro 上 SkillOpt val+21pp 但 test 不泛化 (过拟合38题val)。两个修复方向并行: (1) mbpp_pro 扩 val 看是否数据太少; (2) 换 headroom 更大的代码基准。约束见 [[dataset-constraint-20260715]]。

## 实现改动 (已落地)
mbpppro env dataloader/adapter 增加 `dataset_glob` 参数 (默认 mbpp_pro*, 可切 humaneval_pro* 等), 通过 `env.dataset_glob` cfg 或环境变量 `DATASET_GLOB` 切换。run_code.sh 增加 `DATASET_GLOB` / `RATIO` / `TAG` 环境变量。已端到端验证 HumanEval 切换+加载+rollout+evaluator 全通。

## HumanEval Pro noskill baseline (2:1:7, test=115, 干净)
- **full-pass=0.4087, avg_soft=0.4501**
- 对比 mbpp_pro noskill 0.5283: HumanEval 低 12pp ⇒ **模型更未饱和, headroom 更大**。
- HumanEval Pro 164 题, 2:1:7 → train33/val16/test115。val 16 偏小 (后续若用 HumanEval 主线需调 ratio 扩 val)。
- ⇒ HumanEval Pro 是比 mbpp_pro 更有潜力的代码主基准候选。

## mbpp_pro 扩 val 实验 (RATIO=1:2:7, ✅ 已完成 2026-07-15 18:47)
- 1:2:7 → train38/val75/test265 (val 从 38 扩到 75)。4 epochs optimize, wall=3125s, 2.82M tokens。
- val gate 仍有效: baseline selection(val=75)=0.48 → ep2 0.56 → ep3 best 0.60 (+12pp val lift)。
- **干净 test 结果 (run-internal, 同 split seed42, test=265)**:
  | skill | test hard | test soft |
  |---|---|---|
  | baseline (initial.md S_0) | 0.4679 | 0.5190 |
  | best-on-val (step3) | 0.4604 | 0.5105 |
  | final/last | 0.4377 | 0.4868 |
- **best-on-val test 0.4604 < baseline 0.4679 (−0.75pp), final −3.0pp**。
- ⇒ **扩 val(38→75) 没有治住过拟合**, best test 仍 ≤ baseline。"val 太小→过拟合" 假设**被证伪**。
- 代码域 (mbpp_pro 单函数) skill 优化 val 有效但 test 不泛化 = 域本身 headroom 不足, 与 SkillOpt 方法无关 (见 [[skillopt-code-result-20260715]] 边界澄清)。

### ⚠️ 429 污染警告 (跨 run 不可比)
- 本次 run test 有 ~24% RolloutException (63/265 agent_ok=0.76, 429 限流), 绝对分被压低。
- **跨 run 不可比**: 本次 baseline 0.4679 ≪ 旧 2-1-7 干净 noskill 0.5283 (同 test=265), 差 6pp 全是 429。
- **run-内仍可比**: baseline/best/final 三组 429 数相近(63/63/69), best-vs-baseline 对位有效 → 泛化负结论可信。
- 教训: 后续代码 eval 必须并发降到 4-8 + 失败重试, 否则绝对分被 429 吞掉, 跨实验对位失效。

## 结论 (mbpp_pro 已穷尽)
- mbpp_pro (2-1-7 与 1-2-7) 两种 split 均 best test ≤ baseline → **mbpp_pro 代码单函数域 skill 优化无 test 泛化**, 主基准候选排除。

## 待办
- ⏳ HumanEval Pro 跑 seed + optimize (需先调 ratio 扩 val, 当前 2:1:7→val=16 太小; headroom 大 noskill 0.41)。
- ⏳ 据结果定主基准: HumanEval(headroom大但需调val+降并发) / 更难代码基准(LCB等, 需评估可跑性)。

Related: [[skillopt-code-result-20260715]], [[dataset-constraint-20260715]], [[skillopt-code-env-20260715]]
