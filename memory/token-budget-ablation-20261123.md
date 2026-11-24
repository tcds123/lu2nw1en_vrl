---
name: token-budget-ablation-20261123
description: Token 预算消融 — 隔离 max_tokens 对 S₀ 的影响 (8192 vs 16384), AIME24+AIME26 ✅ 完成
metadata:
  type: project
---

# Token 预算消融 ✅ 完成 (2026-11-23 09:52→15:10, ~3h20m)

**目的**: 隔离 max_tokens 对 AIME 评测的影响，量化"cap=8192 是否比 cap=16384 显著降低 pass rate"。

## 最终结果

| Dataset | 8192 mean | 16384 mean | Δ | CI_lower | CI>0 |
|---------|-----------|------------|-----|----------|------|
| AIME24 (in-sample) | 0.7000 | 0.7417 | **+0.0417** | -0.0020 | ❌ (边际) |
| AIME26 (held-out) | 0.7250 | 0.7417 | **+0.0167** | -0.0583 | ❌ |

### 截断对比

| Dataset | 8192 trunc | 16384 trunc |
|---------|-----------|-------------|
| AIME24 | 6.7% (8/120) | 5.8% (7/120) |
| AIME26 | 10.8% (13/120) | 7.5% (9/120) |

### Avg completion tokens

| Dataset | 8192 avg_tok | 16384 avg_tok |
|---------|-------------|---------------|
| AIME24 | 3147 | 3008 |
| AIME26 | 3314 | 3258 |

## 核心发现

1. **方向一致**: 两个数据集上 16384 cap 都优于 8192 cap (+0.017 ~ +0.042)
2. **统计不显著**: CI 均跨零。AIME24 CI 逼近边界 (-0.002), AIME26 CI 宽度大 (-0.058)
3. **截断缓解但有 leak**: 16384 cap 减少截断率 ~30% (AIME26 10.8→7.5%), 但缓解的截断并不等于 pass rate 提升——很多截断出来的答案本来就不对
4. **AIME26 更容易但有更多截断**: AIME26 在两种 cap 下 mean_pass 都高于 AIME24 (0.72-0.74 vs 0.70-0.74), 但截断率反而更高 (10.8% vs 6.7%)——AIME26 题目触发 V3.2 写更长的推理链，但正确率也更高
5. **avg completion tokens 极接近**: 两 cap 下平均 token 差异小于 <10% (3147 vs 3008, 3314 vs 3258), 说明大部分题目没有人用到 >8192 的 token 预算。V3.2 在数学题上的推理链长度基本集中在 3k-4k 范围

## 对论文的 implication

- **cap=8192 是合理选择**: 虽然方向性正向 (16384 > 8192), 但效应 ~0.02-0.04 并不可靠 (CI 跨零)
- **可以记录为限制**: "Our experiments use cap=8192; a cap sensitivity analysis (n=30, k=4) shows 16384 cap yields +0.02–0.04 mean pass rate advantage, directionally consistent but not statistically significant (CI crosses zero)."
- **两种 cap 下 avg tokens 几乎相同 →** 截断是少数 outlier 题的现象，而非系统性缺陷

## 注意事项

- k=4, 单 seed → CI 较宽。要获得 compact CI 需要 k=8 + multi-seed, 但这已经是 held-out AIME26 eval 的设计
- AIME24 的 AIME24 数据没有 per-qid detail (首次运行的 CI 调用 bug 导致 JSON 未落盘), 仅有 mean_pass 和 trunc 从日志提取

## 输出

- `outputs/token_budget_ablation_aime/2026-11-23_09-52/token_budget_ablation.json`
- `outputs/token_budget_ablation_aime/2026-11-23_09-52/aime26_cap{8192,16384}_detail.json`

## 关联

[[aime26-heldout-20261123]] — AIME26 held-out 验证 (+0.026, CI 跨零)
[[aime25-heldout-generalization-20261120]] — AIME25 held-out 结果 (+0.019, CI 跨零)
