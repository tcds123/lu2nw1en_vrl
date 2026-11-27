---
name: mathline-codex-audit-20261126
description: codex 数学线评审 + P0 离线审计执行记录 (2026-11-26) — 修正背景文档硬错误 + 回答 7 问 + 统计功效定量
metadata:
  type: project
---

# codex 数学线评审 + 离线审计 (2026-11-26)

## codex 评审要点 (用户转入)

codex 指出: "候选无收益/小收益检不出/局部收益带负迁移" 是三个不同问题, 不应统一归入"饱和"。建议拆成 候选生成/效果测量/候选选择/跨题泛化 四环节判瓶颈。P0=离线统计审计 + 逐题收益分析。详见 `for_review/incoming/20261126_codex_mathline/codex_mathline_review.md`。

## 执行的 P0 离线审计 (零 API)

### 修正的硬错误
1. **LiveMath split 题数**: 背景文档写 train35/val18/test106 (159 不闭合) = 笔误。实际 train18/val53/test106=177 (闭合, 1:3:6≈18:53:106)。源码注释 L59 本就对。已修背景文档。
2. **"两轮方向一致"误导**: Option Comparison iter1 hash=0d319a89 / iter2 hash=5fe9dd30 = 两个不同 patch, 非同一候选重复。不能当独立验证。已修。
3. **"诚实门控识别捷径"不准**: CI 判统计证据非判推理来源; 揭示捷径的是 P8/P9+Step6 非门控。已修。

### codex 7 问答案
- **Q1 not_meta_n=10**: 是题数非 rollout。subset_signal 先每题 mean over k rollouts → 题级 delta → bootstrap 题级。未把 rollout 当独立 (codex 担心的错误没发生)。
- **Q2 两轮同一 skill?**: 否, hash 不同。codex 警告命中。
- **Q3 bootstrap 数据流**: 重采样单位=题, 1000 次 default_rng(42), 2.5 分位。paired 同批同题消噪。**关键缺口**: 原始 per-item delta 未落盘 → 离线无法精确复算 CI。
- **Q4 AIME 输出级证据**: 有。details_*.json 含完整 response/predicted/correct/pass_rate。逐题对比可做 (未做完)。
- **Q5 σ_pair≈0.044**: AIME identity 两次 eval delta 摇摆幅度 = 题级 delta std 量级, 非 SE。

### 统计功效定量 (审点 A)
- n=10, σ=0.044 → CI>0 需 mean > +0.0273 (+2.7pp)。iter1 候选3 +0.023 < 0.0273 → 注定跨零。
- 真实效应 +0.023, n=10 power=**38%** (62% 误拒概率)。n=71 (满 select) power=99%。
- **结论**: 0 晋升负面证据不强, 与功效不足无法区分。但解法是增大 n 或隔离确认数据, 非降门槛 (降门槛放假阳性)。

## 采纳方向
- ✅ P0 离线审计 (本次完成)
- 🟡 逐题收益分析 (AIME details 可做, LiveMath per-item 需补落盘)
- ✅ 两阶段候选选择 (探索A/确认B, 确认数据与搜索隔离)
- ✅ 失败模式可迁移性独立研究 (codex 方向 D, 最有论文潜力)
- ✅ 暂缓 AIME 防截断/堆全局 bullet

## 修正后判断
1. LiveMath 0 晋升 = 功效不足与效应不足无法区分, 非已证无效
2. AIME +0.100 in-sample 不应作独立确认证据 (参与搜索有选择偏差); 25/26 才是外推边界
3. "饱和"应限定"当前实验条件下", 非范式终判

## 关联
[[livemath-forreview-20261125]] [[livemath-newmech-smoke-launched-20261125]] [[feedback-no-absolute-verdict-across-versions]] [[aime-saturation-closeout-20261105]]
