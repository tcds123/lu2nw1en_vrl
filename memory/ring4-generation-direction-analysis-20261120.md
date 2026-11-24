---
name: ring4-generation-direction-analysis-20261120
description: 环4不是"方向全错" — iter3 EarlyCommit 纯增益+0.052 有效, iter4 才反生产力; 瓶颈是"加段让模型做更多工作→触顶"
metadata:
  type: project
---

# 环 4 生成方向精细分析 (2026-11-20)

**背景**: 用户反思"两种 LLM 都走错方向 (口号型/阈值型, 非压缩型), 环 4 是唯一要修的"。逐 candidate 拆解后**推翻"方向全错"的笼统判断**: 4 个 budget section 质量参差, iter3 其实有效。

## 决定性数据 (delta 扣除"答对题被搞坏"损失)

| 候选 | delta | 搞坏答对题 | 搞坏损失 | **纯增益** |
|------|-------|-----------|---------|-----------|
| iter1 Rapid-Exit | +0.011 | 1题(85) | +0.004 | **+0.007** |
| iter2 ForcedFinalAns | +0.011 | 0题 | 0 | **+0.011** |
| iter3 EarlyCommit | +0.056 | 1题(78) | +0.004 | **+0.052** ✅ |
| iter4 EscapeHatch | -0.022 | 3题(78,66,85) | +0.011 | **-0.033** ❌ |

**iter3 EarlyCommit 纯增益 +0.052** — budget section 确实救回了题 (不只不搞坏), 只是 n=30 CI 跨 0。iter4 的负 delta 主要来自搞坏 3 个答对题。

## 为什么"输出更短=更好"假设是错的

mean tokens vs delta (无负相关):
- S₀ baseline: 3035 tok, tr=5.6%
- iter1 Rapid-Exit: **2842 tok** (最短), delta **+0.011** (最低正)
- iter3 EarlyCommit: **3365 tok** (最长), delta **+0.056** (最高!)
- iter4 EscapeHatch: 3375 tok, tr=**15.6%** (截断翻倍), delta -0.022

→ iter3 输出最长但 delta 最高。**不是"写更短"救了它, 是"验证型内容"提升了准确率**, 同时只触发 6.7% 截断 (没失控)。iter4 输出和 iter3 差不多长, 但截断 15.6% — 因为 EscapeHatch 内容是"reduce to finite grid + cell count × periods", **更复杂的计算程序**让模型在 aime24:73 上 3/3 seed 全写满 8192。

## 修正后的环 4 诊断

不是"方向全错", 是**两个机制叠加**:
1. **增益机制** (iter3 验证型): 加"小 case 交叉检查"段 → 准确率升 → delta 正。这是对的, 该保留。
2. **破坏机制** (iter4 重算型): 加"reduce to grid × periods"段 → 模型做更多结构化计算 → token 涨 → 触顶截断 → 答对题变 0 分 → delta 负。

**真正的瓶颈**: expand LLM 分不清"会增 token 的策略"和"会降 token 的策略"。EarlyCommit/EscapeHatch 这种"换方法重算"型天然增 token; 而"在小 case 上验证现有答案"型不增 (甚至能早停)。

## 对"压缩型 vs 时间型"框架的修正

用户提出"正确方向是 writing style 压缩, 不是时间维度 commit"。数据**部分支持但更精细**:
- iter2 ForcedFinalAns (纯时间型"立即发答案"): 纯增益 +0.011, 不搞坏题, 但增益低 (没救回题)
- iter3 EarlyCommit (验证型, 非纯压缩): 纯增益 +0.052, 最高 — 因为它让模型在小 case 上验证, **既能纠错又能早停**
- iter4 EscapeHatch (重算型): 负 — 让模型换方法重做, 增 token, 搞坏题

→ 最优解可能不是"压缩 writing style"也不是"时间 commit", 而是**"验证型早停"** (iter3 的方向): 鼓励模型在得到候选答案后用小 case 验证, 验证通过即 commit (不继续推导), 既提升准确率又自然控制长度。

## 对下一步的建议 (仅建议, 不改代码)

1. **iter3 EarlyCommit (+0.052) 已是有效方向**, 只是 n=30 CI 跨 0。问题在统计功效, 不在方向。
2. 若改 expand prompt, 应引导 LLM 产**"验证后早停"型** (如 "在最小 case 上验证候选答案, 通过即提交, 不重做"), 而非"重算/换方法"型 (会增 token)。
3. **或先解决统计功效**: iter3 的 +0.052 在 n=30 下 CI=-0.033; 若 n=90 (3×样本) CI 会收窄到 ~-0.015, 仍可能跨 0; n=270 才可能 CI>0。增 n 比改方向更快让既有有效候选过门控。
4. **截断守门** (候选 n_trunc > parent 则拒) 会直接拦掉 iter4 (15.6%>5.6%), 让 LLM 在有效方向 (iter3) 上多探索, 不在破坏方向上浪费 CI 计算。

## 关联

[[feedback-loop-failure-mechanism-20261119]] — 我之前说"budget section 反生产力", 本节修正: **只 iter4 反生产力, iter3 有效** (+0.052 纯增益)
[[trunc-efl-bgt-dual-evolve-launch-20261118]] — run3 完整结果; iter3 delta+0.056 峰值正是有效方向的体现
