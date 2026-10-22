---
name: experiment-bigcodebench-plan
description: BigCodeBench Lite Pro 消融实验计划与启动记录 (2026-07-03)
metadata:
  type: project
---

# BigCodeBench Lite Pro 消融实验 (n=57 全量)

## 启动时间
2026-07-03 14:15

## 背景

- Self-Debug 已在 2 个数据集上验证 (MBPP Pro p=0.0009, Humaneval Pro p=0.0048)
- APPS_Plus Hard 已验证为超出系统能力 (p=0.39)
- 待完成: BigCodeBench Lite Pro 作为第 4 个数据集的对比

## 实验配置

- 数据集: BigCodeBench Lite Pro (CodeEval-Pro), 57 items 全量
- 4 组: baseline / random / curated / self_debug
- Skill 配置: 结构化 skill (402 条, WHEN/WHAT/Example/WHY 格式, ~1000-1200字/条)
- top_k = 5, random_k = 5
- max_debug_rounds = 3
- seed = 42

## 预期

- Self-Debug 预期显著优于 baseline (类似 MBPP/Humaneval Pro 模式)
- Curated 预期仍不显著（历史一致性）
- BigCodeBench 难度介于 Humaneval Pro 和 APPS_Plus Hard 之间

## 执行命令

```bash
source lunwen/bin/activate
python -m agent.b_agent.ablation_test --n 57 --top_k 5 --dataset bigcodebench
```

## 相关文件

- [[achievements]] — 已有成果
- [[next-steps]] — 后续方向
- [[trace2skill-gap-analysis]] — Skill 质量瓶颈分析
