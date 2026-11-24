---
name: appsplus-full-test-reversal-20260718
description: APPS_Plus 全量 test(n=2765) 复测反转 07-17 结论 - best 0.6351 < noskill 0.6564/seed 0.6582, +6pp 泛化消失变 -2pp; 小样本(n=200)假阳性
metadata:
  type: project
---

# APPS_Plus 全量 test 复测 → 反转 (2026-07-18) 🚨

承接 [[appsplus-generalize-20260717]]。07-17 在 test=200 上拿到 best 0.6350 vs noskill 0.5750 (+6pp) 的代码域首次泛化正结果。07-18 凌晨跑**全量 test=2765** 复测，**结论反转**。

## 全量复测结果 (eval_only, test=2765, 同 split seed42, 28RPM 限流) ✅ 2026-07-20 补全
| skill | n_eval | hard | soft |
|---|---|---|---|
| noskill (empty) | 2765 | 0.6564 | 0.6856 |
| seed (initial.md 手写) | 2765 | 0.6499 | 0.6853 |
| best (step2 优化后, 9079 chars) | 2755 ⚠️ | 0.6341 | 0.6667 |

- **best eval 原中断于 2217/2765**, 2026-07-20 用 `scripts/appsplus_complete_best.py` 补全缺失 546 题 (复用确定性 split + 同 best_skill + DeepSeek-V3.2 temp=0, 限流 28RPM)。补全后 best eval 跑到 2755/2765 (10 题 process_one 异常未写 conversation.json: ids 5,13,296,804,1277,1645,1957,2810,6147,6266)。
- 这 10 题的 noskill hard=6/10=0.60, 即使按最优假设给 best 全过, best 上界=0.6340 仍 < noskill 0.6564。10 题缺失不影响结论。
- **apples-to-apples (2755 共同题, 全部三组都评测完)**: noskill 0.6566 / seed 0.6505 / best 0.6341。逐题 best vs noskill: pos=150 neg=212 tie=2393, mean_delta=-0.0225。两比例 z 检验 z=-1.746 (|z|<1.96, 不显著, 但方向为负)。

## 反转解读 (重要)
1. **07-17 的 +6pp 是小样本假阳性**: test=200 上 best 0.6350 vs noskill 0.5750 的 +6pp, 在 test=2765 上消失并翻转为 best 0.6351 vs noskill 0.6564 (**-2.1pp**)。
2. **三层单调性也破了**: 全量上 noskill(0.6564) ≈ seed(0.6499) > best(0.6351), seed 已不再优于 noskill (07-17 seed +2.5pp 也是小样本噪声)。
3. **baseline 比预期高很多**: 07-17 test=200 noskill=0.5750, 全量 noskill=0.6564。小样本严重低估了 baseline → 把噪声当成了 lift。这正是 [[status-review-20260713]] §3.4 担忧的"无重复/无方差"风险的实锤。
4. **val gate 仍微弱**: 21 步 optimize, baseline-on-val=0.5848, best(step2)=0.6177 (+3.3pp), 后 19 步全 reject。val +3.3pp 未泛化到 test。

## 对论文方向的影响
- **代码方向再次存疑**: APPS_Plus interview 上 skill 优化 val 有效(+3.3pp) 但 test 不泛化(-2.1pp), 与 mbpp_pro 同型 (val lift 不泛化)。07-17 的"代码复活"结论**撤回**。
- 单函数题(mbpp/humaneval)和竞赛题(appsplus interview)在 B=DeepSeek-V3.2 + skill 注入下都未见 test 泛化。可能根因: (a) 大模型 baseline 已饱和, headroom 小; (b) val set 395 题仍不足以稳定选出泛化 skill; (c) 单 skill 编辑 + val-gate 这套方法本身过拟合 val。
- **数学方向仍是唯一干净正结果**: [[skillopt-step0-results-20260714]] LiveMath +21pp 泛化。但 [[dataset-constraint-20260715]] 锁定小论文主数据集=代码。

## 下一步选项 (待用户决策)
1. 补全 best 剩余 556 题 (确认反转数字, ~1.5h API), 但 apples-to-apples 已足够下结论。
2. 跑控制变量诊断 ([[skillopt-code-result-20260715]] 变量1-4): 在 appsplus 上隔离 b_agent/池/GRPO 各组件贡献——但前提是单 skill 基线本身有泛化, 当前看没有。
3. 重新考虑数据集: interview 子集 baseline 0.66 偏高, 换 APPS_Plus competition 子集 (更难) 或重新评估代码域是否可行。
4. 接受代码域无泛化, 论文主线转数学 + 把代码 null 作为 [[dataset-transition-narrative-20260716]] 的过渡证据。

## 复现命令
```bash
cd /data/zhuldz/skillopt
export QWEN_CHAT_API_KEY="msk-..."
# 全量 test 复测 (output dirs: appsplus_full_{noskill,seed,best})
BENCH=appsplus TAG=appsplus_full_noskill bash run_code.sh noskill 2765
BENCH=appsplus TAG=appsplus_full_seed bash run_code.sh evalskill skillopt/envs/appsplus/skills/initial.md 2765
BENCH=appsplus TAG=appsplus_full_best bash run_code.sh evalskill outputs/appsplus_full_optimize/best_skill.md 2765
# optimize 21步 (best=step2): outputs/appsplus_full_optimize/{history.json,best_skill.md}
BENCH=appsplus TAG=appsplus_full_optimize bash run_code.sh optimize 4
```

Related: [[appsplus-generalize-20260717]] (被本文反转), [[appsplus-baseline-20260717]], [[skillopt-code-result-20260715]], [[dataset-transition-narrative-20260716]], [[status-review-20260713]] §3.4
