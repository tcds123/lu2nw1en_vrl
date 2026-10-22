---
name: sd-ablation-running
description: Self-Debug 子组件消融实验运行记录 (2026-07-03)
metadata:
  type: project
---

# Self-Debug 子组件消融实验运行记录

## 实验设计

4 组，Humaneval Pro n=50, seed=42, top_k=5, max_debug_rounds=3

| 实验 | Tag | detailed_feedback | detect_assertion | explicit_fix_prompt | 描述 |
|------|-----|:-----------------:|:----------------:|:-------------------:|------|
| A | full | ✅ | ✅ | ✅ | 完整 Self-Debug (基线) |
| B | nodetail | ❌ | ✅ | ✅ | 无逐条测试详细反馈 |
| C | noassert | ✅ | ❌ | ✅ | 仅检测 Traceback, 不检测 [FAIL] |
| D | nosimple | ✅ | ✅ | ❌ | 简化修复指令 (无分析步骤) |

## 运行进度

| 实验 | 状态 | 开始时间 |
|------|------|---------|
| A (full) | 🔄 运行中 | 18:50 |
| B (nodetail) | ⏳ | — |
| C (noassert) | ⏳ | — |
| D (nosimple) | ⏳ | — |

## 相关文件

- [[selfdebug-ablation-plan]] — 实验设计
- [[achievements]] — 完整实验矩阵
- [[paper-outline-v1]] — 论文大纲
