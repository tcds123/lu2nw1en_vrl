---
name: ablation-apps-plus-hard
description: APPS_Plus Hard 消融实验完整结果与诊断 (2026-07-01)
metadata:
  type: project
---

# APPS_Plus Hard 消融实验 — n=100

**日期**: 2026-07-01, 14:18~19:15 (运行约 5小时)
**命令**: `python -m agent.b_agent.ablation_test --dataset apps_plus_hard --n 100 --top_k 5 --seed 42`
**结果文件**: `agent/b_agent/ablation_results/ablation_appshard_n100_k5_20260701_191506.json`

## 总体结果

| 指标 | Baseline | Random | Curated | SelfDebug |
|------|----------|--------|---------|-----------|
| **Composite** | 0.59 ± 1.13 | 0.69 ± 1.14 | 0.58 ± 1.07 | 0.48 ± 1.16 |
| **Pass Rate** | -0.15 ± 0.65 | -0.09 ± 0.51 | -0.06 ± 0.42 | -0.14 ± 0.66 |
| **C-Judge** | 1.97 ± 3.76 | 2.29 ± 3.79 | 1.93 ± 3.57 | 1.35 ± 3.10 |

## 统计检验

| 比较 | t-statistic | p-value | 显著性 |
|------|------------|---------|--------|
| Curated vs Baseline | -0.10 | 0.92 | ❌ |
| SelfDebug vs Baseline | -0.87 | 0.39 | ❌ |
| SelfDebug vs Curated | -0.82 | 0.41 | ❌ |
| SelfDebug vs Random | -1.48 | 0.14 | ❌ |
| Curated vs Random | -1.09 | 0.28 | ❌ |
| Random vs Baseline | 0.86 | 0.39 | ❌ |

**Win rates**: SelfDebug > Baseline: 37%, Curated > Baseline: 52%

## Self-Debug 详细统计

- avg_rounds = 2.9 ± 0.4 (几乎所有任务都触发了 3 轮 max)
- 96/100 任务触发了修复循环
- 仅 1 个任务在第 2 轮成功修复 (case 98: pass=1.0, final=7.3)
- 最常见错误类型:
  - **`takes 0 positional arguments but 1 was given`**: 占比最高，说明代码使用无参函数签名而测试要求带参
  - **`AssertionError` (无 message)**: 逻辑错误但无法自动修复
  - **`IndexError: list index out of range`**: 边界条件 bug

## 根因分析

### 为什么 Self-Debug 在两个 Pro 数据集上有效但在 Hard 上无效？

1. **错误类型差异**: MBPP/Humaneval Pro 上 self-debug 修复的主要是局部逻辑 bug。APPS_Plus Hard 的错误大多是**函数签名不匹配**，这需要结构性重写，self-debug 无法 fix
2. **题目难度**: Competition + Interview 级别的 APPS 题目需要更复杂的算法设计，当前 B 模型生成的初始解已经很差，self-debug 无法从差的起点迭代到好解
3. **对论文的利用**: 这提供了有价值的「方法局限性」讨论 — self-debug 在中等难度有效 (p<0.01)，但在极端难度无效 (p=0.39)

## 与历史数据对比

| 数据集 | N | Baseline | SelfDebug | SD p | 模式 |
|--------|---|----------|-----------|------|------|
| MBPP Pro | 100 | 7.31 | 8.16 | 0.0009 | ✅ 突破 |
| Humaneval Pro | 100 | 6.57 | 7.32 | 0.0048 | ✅ 突破 |
| **APPS_Plus Hard** | 100 | 0.59 | 0.48 | 0.39 | ❌ 无效 |

## 教训与后续方向

1. Self-Debug 的天花板是初始解的质量。如果初始解结构完全错误，修复循环无法纠正
2. 下一步应该分析「self-debug 何时能修复 vs 不能修复」的分界线，作为论文讨论部分
3. 可能的改进方向: 让 self-debug 支持更激进的代码重写（扩大 diff token 量），或加入函数签名验证作为质量检测
