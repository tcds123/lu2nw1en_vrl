---
name: selfdebug-ablation-plan
description: Self-Debug 子组件消融实验规划 (2026-07-03)
metadata:
  type: project
---

# Self-Debug 子组件消融实验

## 目标
确定 self-debug 的 11.6% 提升来自哪些具体改进。

## 背景
7/1 self-debug 突破了 4 个关键改进:
1. 逐条详细测试反馈 (per-test PASS/FAIL + 错误类型)
2. 正确检测 assertion failures (不只是 Traceback)
3. 更明确的修复指令 (分析→定位→修复)
4. 更难的数据集 (CodeEval-Pro 提供真实调试信号)

## 消融维度

| 实验 | 详细反馈 | Assertion检测 | 修复指令 | 预期 |
|------|:-------:|:-----------:|:-------:|------|
| A (Full SD) | ✅ | ✅ | ✅ | 最强 (当前 baseline) |
| B | ❌ (简化输出) | ✅ | ✅ | 下降 — 反馈质量影响 |
| C | ✅ | ❌ (仅Traceback) | ✅ | 下降 — assertion检测影响 |
| D | ✅ | ✅ | ❌ (简单重试) | 下降 — 指令质量影响 |

## 实现方式

在 `core.py:generate_code_with_self_debug` 中添加 feature flags:
- `detailed_feedback: bool = True` — 是否逐条测试输出
- `detect_assertion: bool = True` — 是否检测 assertion failures
- `explicit_fix_prompt: bool = True` — 是否使用详细修复指令

## 执行计划

数据集: Humaneval Pro n=50 (节省成本但保持统计检验力)
每组预计 ~30-40 min

## 相关文件

- [[achievements]] — Self-Debug 突破数据
- [[paper-outline-v1]] — 论文 Ablation 部分
