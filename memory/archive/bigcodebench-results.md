---
name: bigcodebench-results
description: BigCodeBench Lite Pro 消融实验结果 — n=57 全量分析 (2026-07-03)
metadata:
  type: project
---

# BigCodeBench Lite Pro 消融实验结果

## 实验信息

- 运行时间: 2026-07-03 14:17 → 17:36 (~3h 19min)
- 数据集: BigCodeBench Lite Pro (CodeEval-Pro), 57 items 全量
- Skill: 402 条结构化 skill (WHEN/WHAT/Example/WHY, ~1000-1200字/条)
- 4 组: baseline / random / curated / self_debug
- max_debug_rounds = 3, top_k = 5, random_k = 5

## 核心结果

| Metric | Baseline | Random | Curated | Self-Debug |
|--------|:-------:|:------:|:-------:|:----------:|
| Composite Score | 2.32 ± 2.12 | 2.55 ± 2.19 | **2.86 ± 2.31** | **3.05 ± 2.81** |
| Pass Rate | -0.63 ± 1.32 | -0.70 ± 1.40 | -0.59 ± 1.35 | **0.05 ± 0.68** |
| C-Judge Score | 6.08 ± 2.85 | 6.50 ± 2.79 | 6.99 ± 2.47 | 6.58 ± 3.22 |

## 统计显著性

| 对比 | p-value | 显著 |
|------|:------:|:----:|
| **Curated vs Baseline** | **0.0118** | ✅ SIGNIFICANT |
| **SelfDebug vs Baseline** | **0.0121** | ✅ SIGNIFICANT |
| **Curated vs Random** | **0.0398** | ✅ SIGNIFICANT |
| **SelfDebug vs Random** | **0.0400** | ✅ SIGNIFICANT |
| SelfDebug vs Curated | 0.3806 | ❌ not sig |
| Random vs Baseline | 0.2256 | ❌ not sig |

## Self-Debug 统计

- avg rounds: 2.8 ± 0.6
- fix attempts: 150 (总计)
- tasks with fixes: 52/57 (91.2%)
- SelfDebug > Baseline win rate: 63.2% (36/57)
- SelfDebug > Curated win rate: 42.1% (24/57)

## 关键发现

### 🔥 Curated Skills 首次显著 (p=0.0118)！

这是**首个数据集**上 Curated Skills 显著优于 Baseline。可能原因：
- BigCodeBench 的困难度恰好适合结构化 skill 发挥作用
- 在更难的任务中，skill 指南的边际价值更大
- 402 条结构化 skill 中终于有在 BigCodeBench 风格问题上有效的

### 📊 Self-Debug 仍然有效但不占绝对优势

- Self-Debug p=0.0121 vs baseline（显著）
- 但 SelfDebug vs Curated p=0.3806（不显著），两者效果接近
- **SelfDebug 和 Curated 在 BigCodeBench 上效果相近**（3.05 vs 2.86）

### 🔄 Self-Debug 参与率极高

- 91.2% 的任务触发了 self-debug 循环（vs MBPP Pro 35%, Humaneval Pro 43%）
- avg rounds=2.8（接近 max=3），说明 BigCodeBench 的代码质量普遍较差
- pass rate 从 baseline 的 -0.63 提升到 0.05（勉强正值），绝对水平仍然很低

## 四数据集对比总览

| 数据集 | n | Baseline | Curated | SelfDebug | Curated p | SD p | 难度 |
|--------|---|:-------:|:-------:|:---------:|:---------:|:----:|------|
| MBPP Pro | 100 | **7.31** | 7.26 | **8.16** | 0.73 ❌ | **0.0009** ✅ | 中等-困难 |
| Humaneval Pro | 100 | **6.57** | 6.34 | **7.32** | 0.14 ❌ | **0.0048** ✅ | 中等-困难 |
| **BigCodeBench Lite** | **57** | 2.32 | **2.86** | **3.05** | **0.0118** ✅ | **0.0121** ✅ | 困难 |
| APPS_Plus Hard | 100 | 0.59 | 0.58 | 0.48 | 0.69 ❌ | 0.39 ❌ | 极端 |

## 论文洞察

1. **Self-Debug 在 3/4 数据集上显著**（MBPP, Humaneval, BigCodeBench），仅极端难度的 APPS_Plus Hard 不显著
2. **Curated Skills 有效性出现分化**：在中等难度上不显著（p=0.14~0.73），在 BigCodeBench 上首次显著（p=0.0118）
3. **方法在不同难度梯度上呈现清晰的模式**：
   - 中等难度: SelfDebug >> Curated ≈ Baseline
   - 困难: SelfDebug ≈ Curated > Baseline
   - 极端: 全部不显著

## 相关文件

- [[achievements]] — 完整实验矩阵
- [[next-steps]] — 后续方向
- [[paper-outline-v1]] — 论文大纲
- [[selfdebug-ablation-plan]] — Self-Debug 子组件消融
- [[experiment-bigcodebench-plan]] — 实验计划
