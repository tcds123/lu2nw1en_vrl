---
name: theory-directions
description: 论文方法论深化的4个理论方向 — Context软参数/组合优化搜索/信息蒸馏边界/因果归因 (2026-07-10)
metadata:
  type: project
---

# 方法论深化的理论方向

## 问题诊断

当前 A→B→C 管线本质上是在组装现有技术组件（GRPO + Self-Debug + LLM Judge），缺乏原创的方法论和理论深度。论文若仅呈现工程实践会被批评为"堆砌技术"。

需要从以下 4 个方向寻找理论突破点。每个方向既有独立价值，又能互相支撑形成完整故事线。

## 四个理论方向

### 方向 1: Context 作为"软参数"的等效性理论
**核心问题**: 能否建立严格的数学对应关系——某些 context 变化 ≈ 参数空间的等效变化？
**状态**: 🟡 待展开 → [[context-as-soft-parameters]]

### 方向 2: Skill 组合优化的搜索空间与 Reward Landscape 分析
**核心问题**: \( 2^{402} \) 搜索空间中，什么搜索策略在什么 reward 景观上有效？
**状态**: 🟡 待展开 → [[search-landscape-analysis]]

### 方向 3: 经验蒸馏的知识保留边界 (Information Bottleneck)
**核心问题**: trace → skill 蒸馏过程中信息损失的定量分析
**状态**: 🟡 待展开 → [[distillation-info-bottleneck]]

### 方向 4: Self-Debug 的因果归因与反事实推断
**核心问题**: 从 do-calculus / counterfactual 角度给出更严格的归因结论
**状态**: 🟡 待展开 → [[causal-attribution-selfdebug]]

## 推进计划

1. 逐条深入：每个方向单独成文，分析已有数据能提供什么支撑 + 需要补什么实验
2. 初步理论化后，用新的实验验证
3. 最终整合成一个统一的方法论框架

## 相关文件

- [[achievements]] — 已完成实验数据
- [[next-steps]] — 原有工程路线图
- [[paper-outline-v1]] — 论文大纲 (需重构为理论驱动)
