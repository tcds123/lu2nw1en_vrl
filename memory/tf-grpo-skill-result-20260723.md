---
name: tf-grpo-skill-result-20260723
description: 🔥免训练线(TF-GRPO)首结果 - LCB test=143 hard 0.193→0.253(+6pp), overall 0.4406→0.5828(+14pp); 首次撬动hard死区
metadata: type: project
---

# 免训练线 TF-GRPO 首结果 (2026-07-23) 🔥🔥🔥

## 核心结果: LCB test=143 (无污染 NEW, 对位 baseline)

| 难度 | n | pass_rate | hard(pass@1) |
|------|---|-----------|--------------|
| easy | 34 | 0.9706 | 0.9706 |
| medium | 26 | 0.6859 | 0.6154 |
| **hard** | **83** | **0.3916** | **0.2530** |
| **TOTAL** | **143** | **0.5828** | **0.4895** |

对位:
| 方法 | overall | hard | Δ vs S_0(hard) |
|------|---------|------|------|
| noskill | 0.4266 | 0.181 | -0.012 |
| SkillOpt(别人) | 0.4336 | 0.193 | 0 |
| S_0(initial) | 0.4406 | 0.193 | 0 |
| **TF-GRPO(我们,免训练)** | **0.5828** | **0.253** | **+0.060** |

**关键**: hard 题 0.193→0.253 (+6.0pp, +31%相对) — 这是 SkillOpt/S_0/noskill 全部零收益的死区, 免训练线首次撬动。与 train/val delta 信号一致 (train +0.224, val +0.125), 非偶然。

## ⚠️ 可比性标注 (诚实)
baseline 0.4406 是历史数字; 本次 TF 评测用新 key + 当前 DeepSeek-V3.2, easy 已 0.97 天花板, 整体偏高可能有模型/采样差异。**hard +6pp 趋势可信 (train/val/test 三者一致), 但绝对值需重测 S_0 baseline 同条件对位坐实**。

## 运行细节 (T=3, g=4, train_n=41 hard优先)
| iter | train mean best delta | val new delta | 接受? |
|------|----------------------|---------------|-------|
| baseline | - | -0.025 | - |
| 1 | +0.224 | **+0.125** | ✅ ACCEPT |
| 2 | +0.248 | +0.056 | ❌ KEEP(回归) |
| 3 | +0.251 | +0.063 | ❌ KEEP(回归) |

train delta 持续升但 val iter1 见顶后回落 = 过拟合 (与 SkillOpt 同模式)。**val 门控生效**: 保留 iter1 skill 作最终产出, 拒绝 iter2/3。最终 tf_global_skill.md = iter1。

## 方法 (零 GPU)
rewriter=DeepSeek-V3.2 API → 每题采样4变体 → B(DeepSeek)生成代码 → LCB stdin/stdout单测 → delta=pass-s0_pass → 组内对比蒸馏经验(GROUP_ADVANTAGE_TEMPLATE) → 跨题合并(CROSS_PROBLEM_MERGE_TEMPLATE) → 合成全局skill → **val门控防回归**。
复用: experience/extractor 模板 + skillopt chat_target/evaluator + construct_training_prompt(SePO风格)。

## 产出文件
- skill: `rl/train/verl1/outputs/tf_skill/run_20260723_104604/tf_global_skill.md`
- 经验池: `experience_pool_tf.json` + `experience_iter{1,2,3}.md`
- 评测: `outputs/eval_lcb_tf_global_20260723.json`
- 日志: `/tmp/tf_full_run.log`, `/tmp/tf_eval_test.log`

## API key 更换 (2026-07-23)
旧 key msk-bf2324.. 401失效, 换新 key msk-4601b27.. (DeepSeek-V3.2, aimpapi)。新 key skillopt 非流式解析正常。tf_skill_optimizer.py 默认 key 已更新。

## 下一步
1. ⏳ 同条件重测 S_0 baseline (test=143) 坐实 hard +6pp (排除模型/采样差异)
2. ⏳ 跑 TF online per-task 模式 (SkillComposer 发现 per-task 才对代码题有效, 可能更强)
3. ⏳ 8卡4B GRPO 训练线 (等资源) → 与 TF 同表对位
4. ⏳ 填 docs/training_vs_training_free_parallel_plan.md §5 对位表
5. 三种结局: 目前 TF 强于历史 baseline → 若 GRPO-4B 再超 TF 则"训练值得"; 若持平/低于 TF 则"免训练已足够"

Related: [[training-vs-tf-parallel-plan-20260723]], [[sepo-mbpp-reference-20260721]], [[skillcomposer-sepo-verified-20260721]], [[lcb-new-optimize-20260721]], [[code-skill-negative-rootcause-20260721]]
