---
name: appsplus-flash-skillopt-result-20261111
description: APPS_Plus flash SkillOpt 结果 — test Δ=+1.0pp (+3题), McNemar p=0.72=噪声; 证伪 val=10 饱和假设 (val=100/0.01粒度仍噪声); skill-text 优化在竞赛题到顶 (与 ClassEval 同型)
metadata:
  type: project
---

# APPS_Plus flash SkillOpt 结果 (2026-11-11)

## 一句话
flash + gated slow_update 在 APPS_Plus (val=100/test=300, 远超 ClassEval val=10) 上跑完:
**test Δ = +1.0pp (+3 题), McNemar p=0.7201 = 噪声**.
证伪了 "ClassEval 噪声是 val=10 饱和所致" 的假设 —— 即使 val=100 (hard 粒度 0.01, baseline 0.58 远未饱和), skill-text 优化 Δ 仍不显著. **代码竞赛题 skill-text 优化到顶, 与 ClassEval 4-run 同型.**

## 核心数字 (paired n=300, 同 split 同 harness)

| 指标 | 值 |
|------|-----|
| BASELINE (S₀) test hard | **0.6933** (208/300) |
| BEST-on-val (step4) test hard | **0.7033** (211/300) |
| final test hard | 0.7033 (final=best 复用) |
| **Δ (best vs baseline)** | **+0.0100 = +1.0pp = +3 题** |
| baseline test soft | 0.7261 |
| best/final test soft | 0.7394 |

** McNemar (paired, 同 300 题):**
- both pass=194, both fail=75, **HURT (base pass→best fail)=14**, **HELPED (base fail→best pass)=17**
- discordant n=31, net +3
- exact binomial p (two-sided) = **0.7201** (远 > 0.05)
- chi2 w/ continuity = 0.129
- → 14:17 近五五开, +3 题是 API 噪声级波动 (flash temp=0 端点非完全确定性, ±1 题=噪声底), 非真实 skill 收益.

## val 信号 (selection, n=100) — 证伪 val=10 饱和假设的关键

| 指标 | 值 |
|------|-----|
| baseline selection hard (val) | **0.5800** (非饱和, 远离 0.9 天花板) |
| best selection hard (val) | **0.6400** (+6pp on val) |
| best_step | 4 (epoch 2) |

- ClassEval 噪声的假设根因: val=10 → hard 粒度 0.1, baseline 0.9 已饱和, gate 区分不出 skill 变体.
- 本 run: val=100 → hard 粒度 0.01 (10× ClassEval 分辨率), baseline 0.58 远未饱和 → **gate 分辨力充分**.
- 结果: skill 在 val 上确实 +6pp (0.58→0.64, gate 接受了 step4), **但该 val 提升未迁移到 test (+1pp)** = **过拟合 val 的典型信号**, 非泛化收益.
- 即: 不是 "val 太小看不出 Δ", 而是 "Δ 本身就是 val 过拟合噪声, test 上不存在".

## 训练动态 (8 步, accept=1/reject=7)

| epoch | steps | accepts | rejects | best_score_at_epoch_end |
|-------|-------|---------|---------|------------------------|
| 1 | 1,2 | 0 | 2 | 0.5800 |
| 2 | 3,4 | **1** | 1 | 0.6400 ⭐ (step4 = best) |
| 3 | 5,6 | 0 | 2 | 0.6400 |
| 4 | 7,8 | 0 | 2 | 0.6400 |

- 唯一 accept 在 step4, 之后 6 步全 reject → skill 在 epoch2 后停滞.
- best/final skill = 3000 chars (initial 1811 → +1189, 未膨胀). **gated 成功防住 V3.2 force-accept 的 11639 chars 膨胀退化**.
- 资源: wall=10301s (≈2.87h), calls=2325, total_tokens=8,818,961 (prompt=5.24M completion=3.58M).

## 判据对照 (预注册, 见 [[appsplus-flash-skillopt-launch-20261111]])

| test Δ (paired n=300) | 判定 | 本 run |
|----------------------|------|--------|
| ≥ +3pp | 正 skill 杠杆, 证伪 V3.2 force-accept 根因 | ✗ (仅 +1.0pp) |
| **±2pp 内** | **skill 优化 Δ 噪声, 即使大 val 也区分不出 → 到顶** | ✅ **命中** |
| ≤ −2pp | 退化 (gated 也过拟合 val) | ✗ |

**结论: 命中 ±2pp 噪声带 = 到顶.** (判据原文: "±2pp 内 | skill 优化 Δ 仍噪声, 即使大 val 也区分不出 → 代码竞赛题 skill-text 优化到顶 (与 ClassEval 同型)".)

## 三点收敛证据 (跨实验一致)

1. **ClassEval 4-run**: skill-text Δ 全在 ±1 题内 (val=10 饱和假设).
2. **APPS_Plus flash (本 run)**: Δ=+1pp, McNemar p=0.72 (val=100 非饱和, 仍噪声) → **证伪 val 饱和假设**.
3. **V3.2 APPS_Plus 全量 (07-18)**: best −2.2pp 退化 (force-accept 膨胀); 本 run gated 防住退化但也没拿到正收益.

→ 三类证据 (饱和噪声 / 非饱和仍噪声 / 膨胀退化) 共同指向同一根因 [[code-skill-negative-rootcause-20260721]]: **skill 是 system-prompt 纯文本, 单文件竞赛题无可复用工具/API/多文件杠杆**, 模型本身能解的题 skill 边际收益被压缩; gated 防退化但不产生正杠杆.

## 与 B1 LCB 正信号的差异 (为何 flash 在 LCB 有杠杆, APPS_Plus 没有)

- B1 LCB flash: Δ+4.9pp (gated, 16384 cap). LCB 是 LiveCodeBench, 题型含函数签名+可见测试, skill 可注入结构化解题 checklist 有边际收益.
- APPS_Plus: stdin/stdout 竞赛题, 单文件, 输入输出对验证, skill 无结构化注入点 → Δ 噪声.
- **flash 强模型假设**: flash 在 LCB 翻转 V3.2 负结论, 但在 APPS_Plus 上 (本 run) 未翻转 → 强模型不能凭空创造 skill 杠杆, 需任务有结构化杠杆点 (LCB 有, APPS_Plus 无).

## 产出位置
- `/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/appsplus-skillopt-flash-20261111-v2/train/`
- `best_skill.md` (3000 chars), `summary.json`, `test_eval/` (best 逐题), `test_eval_baseline/` (S₀ 逐题), `test_eval_final/` (仅 summary, final=best 复用)

## 关联
- [[appsplus-flash-skillopt-launch-20261111]] (启动配置 + 判据, 本 run 的对照)
- [[appsplus-full-test-reversal-20260718]] (V3.2 全量负结论, force-accept 膨胀退化)
- [[code-skill-negative-rootcause-20260721]] (代码域负根因: 单文件竞赛题 skill 杠杆受限 — 本 run 再次证实)
- [[classeval-a2-2turn-result-20261111]] (ClassEval skill-text 到顶, 转 APPS_Plus 的动因 — 现 APPS_Plus 也到顶)
- [[b1-flash-skillopt-lcb-result-20261109]] (flash LCB 正信号 Δ+4.9pp — 任务有结构化杠杆点才有收益)
