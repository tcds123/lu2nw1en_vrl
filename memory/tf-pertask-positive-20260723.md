---
name: tf-pertask-positive-20260723
description: 🔥TF per-task模式成功 - LCB test=143 hard 0.3012→0.3855(+8.4pp同条件), overall 0.6005→0.6698; 印证SkillComposer per-task才有效
metadata: type: project
---

# 免训练线 TF per-task 模式成功 (2026-07-23) 🔥🔥🔥

## 核心结果: LCB test=143 (同条件对位, 新key)

| 方法(同条件) | overall | easy | medium | hard |
|------|---------|------|--------|------|
| S_0 | 0.6005 | 0.9706 | 0.7644 | 0.3012 |
| TF global | 0.5828 | 0.9706 | 0.6859 | 0.253 (比S_0差) |
| **TF per-task** | **0.6698** | 0.9706 | 0.7436 | **0.3855** |

**TF per-task vs 同条件 S_0**: hard +8.4pp (+28%相对), overall +6.9pp。easy持平(天花板), medium略降-2.1pp, hard大幅领先。

## 关键洞见: 印证 SkillComposer 发现
**per-task online specification 才对代码题有效, 全局 library 无效** ([[skillcomposer-sepo-verified-20260721]]):
- TF global (全局改写skill) 比S_0差 — rewriter加的通用启发式干扰强模型DeepSeek-V3.2 (medium/hard都掉)
- TF per-task (每题现场rewriter针对失败模式) 比S_0好 — 针对具体题改写真正撬动hard题
- 与 SkillComposer 在 LCB v6 结论一致: online per-task +2.5, offline library 差

## 方法 (零GPU)
每道test题: rewriter(DeepSeek-V3.2) + 经验池(从train 41题蒸馏) + S_0基底 → 现场生成该题专属skill → B(DeepSeek)生成代码 → LCB stdin/stdout单测。
test无答案, greedy生成(g=1, temp=0.3), 取最长(最具体)变体。
经验池 = TF训练循环iter1产出 (GROUP_ADVANTAGE蒸馏 + CROSS_PROBLEM_MERGE合并)。

## 产出文件
- 评测: `outputs/eval_lcb_tf_pertask_20260723.json`
- 经验池: `outputs/tf_skill/run_20260723_104604/experience_pool_tf.json`
- 日志: `/tmp/tf_pertask_eval.log`

## 论文价值
**免训练 per-task skill 改写在 LCB hard 题 +8.4pp (同条件对位)** — 可写论文。
- vs 历史baseline (SkillOpt 0.4336/S_0 0.4406): +14.7pp overall, +19.2pp hard (但需标注历史baseline旧key偏低)
- vs 同条件S_0: +6.9pp overall, +8.4pp hard (这个是干净的)
- 故事: 免训练(零GPU) + per-task specification + 群组对比蒸馏经验 > 全局改写 > S_0

## 下一步
1. ⏳ 稳固性验证: per-task g>1 (多采样取最优变体, 但test无答案需用启发式选), 或不同temp
2. ⏳ 消融: 无经验池的per-task (纯rewriter+S_0, 不蒸馏) vs 有经验池 — 验证经验池贡献
3. ⏳ 修TF训练循环: s0_cache同条件重算 (当前delta信号污染, 但per-task eval不依赖s0_cache故结果可信)
4. ⏳ 8卡4B GRPO训练线 vs TF per-task同表对位 — 回答"训练是否值得"
5. ⏳ 统计显著性: 多seed/run, n=143已够但需多次确认

Related: [[tf-global-reversal-samecond-20260723]], [[tf-grpo-skill-result-20260723]], [[training-vs-tf-parallel-plan-20260723]], [[skillcomposer-sepo-verified-20260721]], [[sepo-mbpp-reference-20260721]]
