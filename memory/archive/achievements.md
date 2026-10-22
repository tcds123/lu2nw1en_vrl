---
name: achievements
description: 项目已取得的阶段性成果与实验数据 (2026-07-01)
metadata:
  type: project
---

# 已取得成果

## 一、核心实验发现

### Self-Debug 突破性结果（2026-07-01）

在 **两个独立硬数据集** 上验证了 BAgent Self-Debug 模式的论文级显著效果：

| 数据集 | n | Baseline | Curated | Self-Debug | Δ vs Baseline | p-value |
|--------|---|----------|---------|------------|---------------|---------|
| **MBPP Pro** | 100 | 7.31 | 7.26 (-0.05) | **8.16** | **+0.85 (+11.6%)** | **0.0009** |
| **Humaneval Pro** | 100 | 6.57 | 6.34 (-0.23) | **7.32** | **+0.75 (+11.4%)** | **0.0048** |

Self-Debug 修复参与度：
- MBPP Pro: 35/100 (35%) 任务触发修复循环, avg_rounds=1.6
- Humaneval Pro: 43/100 (43%) 任务触发修复循环, avg_rounds=1.77

**结论**: Self-Debug 是当前最有效的单项改进，p<0.01 双向验证，论文可直接引用。

### Curated Skills 现状

| 数据集 | Curated vs Baseline p | Curated vs Random | 判断 |
|--------|----------------------|-------------------|------|
| MBPP Pro n=100 | 0.73 | — | ❌ 不显著 |
| Humaneval Pro n=100 | 0.14 | — | ⚠️ 有趋势但不达标 |
| OpenCodeInstruct n=100 | 0.87 | — | ❌ 无信号 |
| OpenCodeInstruct n=30 | 0.04 | p=0.92 (curated≈random) | ⚠️ 显著但检索无效 |

**结论**: 触发词 bug 已修复（364 条 skill 获得专属触发词），但 curated skills 效果仍不显著。瓶颈已从检索转向 skill 内容质量。

### 消融实验完整矩阵

#### Phase 1（原始基线，OpenCodeInstruct 数据集）

| n | Baseline | Curated | p |
|---|----------|---------|---|
| 100 | 51.3% | 50.8% | 0.87 |
| 50 | 57.6% | 64.6% | 0.059 |
| 30 | 5.48 | 6.36 | 0.04 |
| 20 | 5.57 | 5.27 | 0.39 |
| 15 | 5.69 | 5.93 | 0.74 |

#### Phase 3（改进后，CodeEval-Pro 数据集）

| 数据集 | n | Baseline | Curated | Random | Self-Debug | Curated p | Self-Debug p |
|--------|---|----------|---------|--------|------------|-----------|--------------|
| MBPP Pro | 100 | 7.31 | 7.26 | 7.14 | **8.16** | 0.73 | **0.0009** |
| Humaneval Pro | 100 | 6.57 | 6.34 | 6.35 | **7.32** | 0.14 | **0.0048** |

## 二、技术改进总结

### 已完成（2026-07-01）

| # | 改进项 | 影响 |
|---|--------|------|
| 1 | YAML 触发词格式修复 | 触发词从单字符串变为正确 YAML 列表 |
| 2 | LLM 模式生成 364 条 skill 专属触发词 | 1949 个独特短语，检索精度从随机提升到语义匹配 |
| 3 | ablation_test.py 支持 6 种数据集 | opencode / humaneval_pro / mbpp_pro / bigcodebench / apps_plus / apps_plus_hard |
| 4 | Self-Debug: 逐条测试详细反馈 | 每 test 独立 PASS/FAIL + 错误类型 + 原因 |
| 5 | Self-Debug: 修复 assertion failure 检测 | 从仅检测 Traceback 扩展到检测 [FAIL] + COMPILE ERROR |
| 6 | Self-Debug: 明确修复指令 | 分析失败原因 → 定位具体测试 → 修复 → 重新输出 |
| 7 | .claude/settings.json 自动模式 | 项目级 defaultMode: auto |

### 关键设计决策

- Self-Debug 使用**外部编排**（生成→执行→反馈→修正）而非让 LLM 自主调用工具。原因：DeepSeek API 不支持 OpenAI-compatible tool calling，外部编排更可控、可审计。
- 测试执行在 subprocess 中运行（10s timeout），包括资源安全限制。
- 所有新功能通过 feature flag 控制（--dataset, --no-self-debug）。

## 三、APPS_Plus Hard — 极端难度数据集 (2026-07-01, n=100)

**结果**: 所有 4 组全部无显著差异。APPS_Plus Hard 超出了当前系统的能力边界。

| 数据集 | n | Baseline | Curated | Random | Self-Debug | SD vs Base p |
|--------|---|----------|---------|--------|------------|--------------|
| **APPS_Plus Hard** | 100 | 0.59 | 0.58 | 0.69 | 0.48 | 0.39 (ns) |

- 所有组 pass rate 均为负值（0 通过），C 模型评分在 1.35~2.29 之间（满分 10）
- SelfDebug avg_rounds=2.9（96/100 任务触发修复），但修复无效果 — 错误类型主要是 `takes 0 positional arguments`（函数签名不匹配），self-debug 当前无法修复这类结构性问题
- 复合得分在 0.48~0.69 范围内，远低于 HumanEval Pro (6-8) 和 MBPP Pro (7-9)

**结论**: 
- APPS_Plus Hard（competition + interview 级别）的难度远超 CodeEval-Pro
- 当前 self-debug 机制的瓶颈：只能处理逻辑/断言错误，无法处理函数签名重写、算法层面重构
- 适用于论文的对比分析：展示方法在「极端难度」上的局限性 vs 在「中等难度」上的突破

### 跨数据集对比总览

| 数据集 | 难度 | Baseline | SelfDebug | SD 提升 | SD p | 显著性 |
|--------|------|----------|-----------|---------|------|--------|
| MBPP Pro | 中等-困难 | 7.31 | **8.16** | +11.6% | 0.0009 | ✅✅✅ |
| HumanEval Pro | 中等-困难 | 6.57 | **7.32** | +11.4% | 0.0048 | ✅✅ |
| APPS_Plus Hard | 极端 | 0.59 | 0.48 | -18.6% | 0.39 | ❌ |
| **BigCodeBench Lite** | **困难** | 2.32 | **3.05** | **+31.5%** | **0.0121** | **✅** |

**核心模式**: Self-Debug 在 3/4 数据集上显著（MBPP, Humaneval, BigCodeBench），仅在极端难度的 APPS_Plus Hard 上失效。Curated Skills 首次在 BigCodeBench 上显著（p=0.0118），说明在更困难的任务中 skill 指南的边际价值更大。

### 🔥 BigCodeBench Lite 新发现 (2026-07-03)

- **Curated Skills 首次显著** (p=0.0118): 这是首个 curated 显著优于 baseline 的数据集
- Self-Debug vs Curated 无显著差异 (p=0.3806)，两者效果接近
- Self-Debug 参与率 91.2%（52/57 tasks），avg rounds=2.8 — BigCodeBench 极难
- 完整对比见 [[bigcodebench-results]]

## 四、待完成实验

- APPS_Plus hard (competition + interview) n=100: ✅ 已完成 (2026-07-01)
- BigCodeBench Lite Pro: 57 items 全量
- Self-Debug 子组件消融: detailed output / fix prompt / dataset 难度各自的贡献
- Self-Debug max_rounds 对比: 1 / 2 / 3 / 5

## 四、论文可写作点

1. **Training-Free GRPO + Prompt Optimization** 框架
2. **BAgent Self-Debug**: 外部编排的代码自修复，在两个 hard benchmark 上 p<0.01 显著
3. **Context Curator**: 组件化、可独立进化的 system prompt 组装
4. **Skill 触发词系统**: 从统一触发词到 1949 个专属短语的进化路径
5. **从参数空间到上下文空间**的研究范式转移
