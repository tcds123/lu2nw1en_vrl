---
name: ib-pruning-evaluation
description: 方向3 IB剪枝评估 — 逐条skill的baseline对照实验设计、脚本与运行记录 (2026-07-10)
metadata:
  type: project
---

# IB 方向 3 — Skill 剪枝评估

## 当前状态

### 数据发现 (2026-07-10)

| 指标 | 值 |
|------|-----|
| Skill 总数 | 402 (files: 403, 含 G0.md) |
| Non-zero ATE | **0/402** |
| Non-zero usage_count | **0/402** |
| GRPO trace 含 skills_used | 仅有 ~2 iterations (6/29 产出) |

**结论**: experience_pool.json 的 ATE 数据全部为初始值 0。证实了 G1（数据源断裂）是最致命的 gap——无法通过 ATE 做剪枝，必须通过**实际实验**来评估每条 skill。

## 实验设计

### 方法: 逐条 Skill Baseline 对照

对每条 skill s 在固定任务集 T = {t₁, ..., t₁₀} 上做 paired 对比：

```
baseline(t)  = B(task=t, skills=∅)
skill(s, t)  = B(task=t, skills={s})
delta(s, t)  = skill(s, t) - baseline(t)
```

汇总每条 skill 在所有 task 上的 delta 均值、正/负 task 数。

### 评估脚本

`agent/b_agent/prune_eval.py` — 逐条 skill × per-task baseline 对比

### 运行计划

| Phase | 数据集 | n_tasks | Skills | API calls | 预计时间 |
|-------|--------|---------|--------|-----------|---------|
| P1 (quick) | BigCodeBench | 10 | top 30 | ~310 | ~15 min |
| P2 (full) | BigCodeBench | 10 | All 402 | ~4030 | ~3.5 hrs |
| P3 (cross) | Humaneval Pro | 10 | top 50 (from P2) | ~510 | ~25 min |

### 剪枝规则

执行完 P1+P2 后:

1. Harmful (mean_delta < -0.1 且 n_negative > n_positive): → DELETE
2. Neutral (|mean_delta| ≤ 0.1): → ARCHIVE (不参与检索，但保留文件)
3. Beneficial (mean_delta > 0.1): → KEEP
4. 目标: 从 402 压缩到 ~50-100 条

## 运行状态

| Phase | 状态 | 开始 | 完成 | 结果 |
|-------|------|------|------|------|
| P0 (pilot) | ✅ 完成 | 11:30 | 11:51 | 5 tasks × 10 skills, G217 best(+0.37), G111 worst(-2.05), 6/10 harmful |
| P1 (medium) | 🔄 运行中 | 11:53 | — | 10 tasks × 30 skills |
| P2 (full) | ⏳ 待运行 | — | — | — |
| P3 (cross) | ⏳ 待运行 | — | — | — |

### P1 发现 (2026-07-10 13:16)

**统计**:
| 指标 | 值 |
|------|-----|
| Mean delta (30 skills) | -0.247 |
| Median delta | -0.197 |
| Std delta | 0.236 |
| Beneficial (Δ>0.1) | 1/30 (3.3%) |
| Neutral | 8/30 (26.7%) |
| Harmful (Δ<-0.1) | 21/30 (70%) |
| Mean harmful Δ | -0.361 |

**Cross-P0 交叉验证**:
| Skill | P0 Δ (5 tasks) | P1 Δ (10 tasks) | 结论 |
|-------|---------------|-----------------|------|
| G217 | +0.37 | -0.08 | 🔴 反转——5 tasks 不足 |
| G18 | +0.19 | +0.003 | → 近零 |
| G111 | -2.05 | -0.02 | P0 outlier |

**关键洞察**:
1. Per-task delta std >> mean delta → skill 效应高度 task-dependent
2. 5 tasks 统计不稳定，10 tasks 最低要求
3. **70% 的 skill 平均有害** → 大量 skill 在不适用的任务上被注入 = 噪音淹没有效信号
4. 仅 1 条 skill (G365) mean Δ > 0.1 → 当前 skill 池质量严重不足

## 相关文件

- [[ib-skill-improvement-plan]] — 四阶段改进总方案
- [[distillation-info-bottleneck]] — IB 理论框架
- [[trace2skill-gap-analysis]] — G1 数据源断裂
- [[theory-directions]] — 4 方向总纲
