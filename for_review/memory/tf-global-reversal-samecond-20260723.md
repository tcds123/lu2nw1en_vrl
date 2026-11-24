---
name: tf-global-reversal-samecond-20260723
description: 🚨同条件重测反转 - S_0同条件0.6005/hard0.3012, TF global 0.5828/hard0.253反而更差; 之前+6pp是旧key baseline偏低假象
metadata: type: project
---

# 免训练线 TF-global 同条件重测反转 (2026-07-23) 🚨

## 反转结论
同条件(新key msk-4601b27 + 当前 DeepSeek-V3.2)重测 S_0 baseline, 推翻了 [[tf-grpo-skill-result-20260723]] 的 +6pp 结论:

| 方法(同条件) | overall | easy | medium | hard |
|------|---------|------|--------|------|
| **S_0(同条件)** | **0.6005** | 0.9706 | **0.7644** | **0.3012** |
| TF global | 0.5828 | 0.9706 | 0.6859 | 0.253 |
| S_0(历史旧key) | 0.4406 | - | - | 0.193 |

**TF global 实际比同条件 S_0 差**: overall -1.8pp, medium -7.8pp, hard -4.8pp。之前看到的 "hard +6pp" 是假象, 因为对比的历史 baseline 0.4406 是旧key/旧模型跑的偏低值。

## 根因
1. **s0_cache 偏低**: TF 整个 delta 信号建立在 s0_cache (旧key预计算, 平均0.64) 上。新key下 S_0 同条件实测 overall 0.60/hard 0.30, 远高于 s0_cache。delta = skill_pass - s0_cache_pass 被系统性高估。
2. **val 门控不可靠**: val baseline S_0 delta=-0.025, iter1 +0.125 都是基于偏低 s0_cache 算的。val(20题)上的 +0.125 没泛化到 test。
3. **rewriter 干扰强模型**: S_0 本身对 DeepSeek-V3.2 已很好(0.60)。rewriter 加的全局通用启发式(Mandatory Tracing/Anti-looping 等)对强模型是干扰, medium/hard 都掉。SePO 在 MBPP baseline 91 天花板 +5pp 是基础弱; 我们 B=强模型 + S_0 已 0.60, headroom 结构不同。

## 教训 (重要)
- **绝对值必须同条件对位**: 历史 baseline 0.4406 与新key结果不可直接比。任何 "+Xpp" 结论必须同 key/同模型/同采样重测坐实。[[tf-grpo-skill-result-20260723]] 的 +6pp 结论作废。
- **delta 信号要基于同条件 baseline**: s0_cache 用旧key预计算, 与新key skill_pass 不同源, delta 不可比。TF 训练循环的 delta 信号全部受污染。
- **val 集太小(20题)门控不够**: val +0.125 没泛化到 test。需更大 val 或交叉验证。
- **强模型 + 好初始 skill 下, 全局通用改写可能有害**: rewriter 加通用启发式干扰强模型。per-task 针对性改写可能不同 (待测)。

## 产出文件
- S_0 同条件: `outputs/eval_lcb_s0_samecond_20260723.json`
- TF global: `outputs/eval_lcb_tf_global_20260723.json` (已存在)

## 下一步
1. ⏳ 跑 TF per-task 模式 (test=143, 每题现场 rewriter + 经验池, 针对性更强可能优于 global)
2. ⏳ 若 per-task 也不如 S_0 → 免训练 global 改写这条路死, 考虑: (a) per-task 是唯一可能; (b) 转训练线; (c) 重新审视 S_0 已很强这个事实
3. ⏳ 修 TF 训练循环: s0_cache 必须同条件重算, 否则 delta 信号不可信

Related: [[tf-grpo-skill-result-20260723]], [[training-vs-tf-parallel-plan-20260723]], [[lcb-new-optimize-20260721]], [[sepo-mbpp-reference-20260721]]
