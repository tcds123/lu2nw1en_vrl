---
name: tf-evolver-appsplus-routing-retention-20261113
description: APPS+ 免训练 evolver 路由部署 232/300 + 通道B retention 诊断 (88.94%, regress=23, 中风险)
metadata:
  type: project
---

# APPS_Plus 免训练 skill 演化器 — 路由部署 + 通道B retention 全量结果 (2026-11-13)

正式 run: `rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-retention-20261113/run_20261113_152603/`
脚本: `rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py` (含 eval_retention, L376/L599-705)
backend: deepseek-v4-flash, cap=16384, temp=0, 3 iter, g=4, train_n=80, test_n=300, seed=42

## 四指标并列 (核心结果)

| 指标 | 我们的候选 (路由部署) | SkillOpt 对照 | 差异 |
|------|---------------------|--------------|------|
| **Acquisition (Rescue)** | 24/92 S₀-fail | 17/92 | **+7** ✅ 更强 |
| **Regression (路由结构)** | 0 | 14 | **-14** ✅ 路由结构性零 |
| **Preservation (Retention, 通道B)** | 185/208 = 88.94% | 194/208 = 93.27% | **-9 题** ⚠️ 劣于 |
| **Deployment (Routed Pass)** | 232/300 = 77.33% | 211/300 = 70.33% | **+21** ✅ 超 |

McNemar: routed vs base p=1.19e-7 (disc=24,0); routed vs SkillOpt211 p=1.9e-4 (26 better,5 worse).
通道B: McNemar S0-vs-cand p=2.38e-7, discordant=23, **显著退化**.

## 核心解读 — 候选 skill 是「高 acquisition / 低 preservation」型

候选 skill (3009 chars, S₀=1796, +1213) rescue 比 SkillOpt 强 7 题, 但 regress 比 SkillOpt 多 9 题。
路由 keep-better 部署物 (232) 确实超 SkillOpt (211), 但候选 skill **独立执行**在 S₀-pass 题上有 23 题退化 (11.06%), 统计显著 (p=2.4e-7)。

**这正是 review_request §1D 威胁的正面回答**: 退化客观存在且显著 — 路由不是"掩盖", 而是用 keep-better 机制在部署层对冲了候选的高方差。三概念必须分开报:
- **Acquisition**: 候选能救回 SkillOpt 救不回的题 (+7)
- **Preservation**: 候选在 S₀ 已会的题上退化 (-9 vs SkillOpt)
- **Deployment**: 路由组合系统部署净分 (+21)

## 通道B 诊断细节 (诚实归因)

- 23 regress 全部 **agent_ok=True** (非 infra 超时/网络噪声, 见 #62a 闭环)
- fail_reason 分布: 5 `nonzero returncode=1` (崩溃) + 18 `stdout mismatch` (错答)
- 错答多为"差一位/符号错": `-1 vs 11`, `0 vs 999999998`, `Bob vs Alice`, `6152868115 vs 6152868116`
- difficulty: **全部 interview** (非 competition) — 退化集中在 interview 级
- 闭包: valid(208)+fail(0)+missing(0)==N(208), n_agent_fail=0

## 风险带判定

`🟡 中风险 | vs_SkillOpt=劣于 | McNemar显著退化=是(p=2.4e-7, disc=23)`
- regress_rate 11.06% > SkillOpt 6.73%, 落在 🟡 (6.73%~12%) 带
- McNemar 显著 (p<0.05), 不是噪声
- 但 deployment (routed 232) 仍 ✅ 超 SkillOpt — verdict 不变

## 对论文叙事的影响

1. **不能只报 routed 232 + regress=0**: 那会误导读者以为候选 skill 无副作用。必须并列 retention 88.94%。
2. **诚实叙事**: "路由部署物超基线 (+21), 代价是候选 skill 独立 preservation 略低于 SkillOpt (-9); 路由的价值在于用 keep-better 对冲 acquisition 的高方差, 而非消除退化。"
3. **后续方向**: 保留信号 — 可在 evolver gate 里加 retention 软约束 (低频采样), 平衡 acquisition/preservation。当前 gate 只看 routed delta。

## 产出文件

- `tf_appsplus_routing_result.json` — 含 `retention_analysis` 字段 (full)
- `test_retention_results.json` — 208 题逐题候选独立执行结果
- `test_routed_results.json` — 300 题路由结果
- `tf_global_skill.md` — 最终候选 skill (3009 chars)
- `tf_run_log.json` — 迭代日志

## 关联

- §1D 威胁 (regress=0 是路由结构性特征) — 此结果正面回答: 退化存在且显著, 但部署层对冲
- [[flash-eval-serialize-20261113]] — 本次与 cap 扫描并行 (#62a), 但 agent_fail=0 证明未受 infra 污染
- issue #61 (正式 run 通道B 零测量) — 本次补跑已落盘, 闭环
- SkillOpt 基准 (只读): 211/300, retention 93.27%, regress=14
