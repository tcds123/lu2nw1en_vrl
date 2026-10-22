---
name: f7-16384-baseline-result-20261021
description: 16384基线确认结果—截断消除,Δ从−0.0347缩到−0.006,CI跨零,决定性负结论被推翻但正向门仍FAIL
metadata:
  type: project
---

# F7 16384 测量基线确认结果 — 2026-10-21

PID 88093 (aime25 held-out iter2 重测, cap=16384+遥测) 跑完 (~111 min, 干净退出)。**所有数字已对磁盘文件逐点复核** (bootstrap_ci.json + 6 个 details_*.json), 非凭 subagent 摘要。

## 核心结果 (脚本逐字输出)

| 指标 | 旧 (4096 cap) | 新 (16384 cap) | 变化 |
|---|---|---|---|
| pooled Δ | −0.0347 | **−0.0056** | +0.0291 (向 0 缩 ~6×) |
| CI95 | [−0.0681, −0.0042] | **[−0.0389, +0.0278]** | 上界转正 → **跨零** |
| CI lower | −0.0681 | −0.0389 | 仍 <0 |
| gate_pass | false | **false** | 仍 FAIL |
| n_pos_seeds | 0/3 | **1/3** | seed2 转正 |
| per_seed_delta | [−0.0625, −0.0417, 0.0] | [−0.0083, +0.0125, −0.0208] | — |
| s0 mean | 0.5347 | 0.6278 | +0.093 (baseline 自己被截断拉低) |
| ev mean | 0.5000 | 0.6222 | +0.122 |

文件: `rl/train/verl1/outputs/tf_skill_aime_v2/aime25_heldout_iter2_f7_20261021/bootstrap_ci.json`

## F7 遥测 (已对 180 条逐条复核)

- n_entries = 180 (3 seed × 2 skill × 30 q)
- `last_finish_reason` = **{'stop': 180}** — 零 `length` 截断
- `truncated_count > 0` = **0/180**
- `max_completion_tokens` = **全部 16384** (cap 生效)
- max `last_completion_tokens` = 10138 (远低于 cap, 说明 16384 足够)
- max `last_resp_chars` = 23972
- **结论: F7 修复完全生效, 截断混淆因子已彻底消除。**

## 分支判定 (中间结果, 非干净任一极)

这不是设计时的三个预设分支中任何一个的干净命中, 而是中间态:

1. ❌ **"fix didn't take"分支排除** — 截断确为 0%。
2. ❌ **"real signal recovered" (正向门) 未达** — 需 CI lower>0 且 ≥2/3 seed 正向; 实际 CI lower=−0.039, 1/3 正向。
3. ❌ **"pure overfit" (决定性负) 被推翻** — 旧 CI [−0.068,−0.004] 整段 <0, 新 CI 上界 +0.028 跨零, 不能再断言 iter2 是纯过拟合。

**一句话**: F7 截断混淆因子完全消除; 先前"决定性负"(CI 整段 <0)被推翻, 但正向门仍 FAIL → iter2 在 held-out 上既非干净过拟合也非干净泛化, 处于噪声水平 (Δ=−0.006, CI 跨零)。

## 机理拆解 (为什么缩了但没归零)

- 旧 s0 mean 0.5347 本身被 4096 截断拉低 → 16384 下 s0 跳到 0.6278 (+0.093)。**baseline 自己被截断伤得几乎和 iter2 一样多**, 这就是 gap 没完全归零的原因: 两个 skill 都被截断伤, iter2 略多伤一点。
- 估算: 旧 gap −0.0347 中, 约 −0.029 (≈83%) 是截断驱动 (两 skill 截断不对称), 残余 −0.006 在噪声内 (CI 跨零)。
- s0 与 evolver 的零分题高度重叠 (qids 13/14/29 三 seed 全零; qid 12 六次中四次零) — 这些是真难题, 两 skill 都败, 非技能特异失败 → 每题 delta 小, CI 跨零。

## 对各判定的影响

- **DeepSeek "候选是 noise fit" 判定**: 被削弱但未推翻 (mean 仍微负, 点估计 −0.006)。不能再说"候选生成层是干净的真瓶颈", 但也不能说它产生了真跨题信号。
- **[[aime25-heldout-iter2-nogeneralize-20261021]] "纯过拟合 verdict"**: **需下调**。原 verdict 建立在 CI 整段 <0 上; 现在 CI 跨零, "纯过拟合"措辞过强。应改为"iter2 held-out 在 16384 下为噪声水平微负, 截断混淆因子排除后决定性负结论不成立"。
- **[[f7-truncation-confounder-20261021]] F7 证伪**: **方向仍成立** (近端桶 ≤0 / 远端桶 >0 的反证在 4096 数据上仍真), 但 F7 作为"测量基建修复"的价值被这次结果重新确认 — 修了之后 Δ 缩 6×, 说明截断确是实打实的测量噪声源, 只是不是推翻 held-out 负结果的充分机制。
- **Direction 1 优先级**: [[direction1-design-review-20261021]] / [[deepseek-three-layer-reassess-20261021]] 里"Direction 1 = 主攻"的定位**仍成立但论据变弱** — 主攻论据原是"iter2 是纯过拟合", 现在纯过拟合被推翻。但 Direction 1 (fix patcher 跨题泛化) 仍是合理攻击点: iter2 没产生干净正信号 (CI 跨零而非干净正), 说明 patcher 仍未产生可靠跨题信号。

## 待办

- [ ] 更新 [[aime25-heldout-iter2-nogeneralize-20261021]]: "纯过拟合 verdict" 下调措辞
- [ ] 更新 [[deepseek-three-layer-reassess-20261021]]: noise fit 判定从"强化"改"削弱但未推翻"
- [ ] 更新 MEMORY.md 索引行 (F7 行 + held-out 行)
- [ ] 追加 `docs/phase2_issues.md`: 16384 基线结果 (里程碑)
- [ ] Direction 1 实现可启动 (基线已确认; 但论据变弱, 实现前需重判优先级)

关联 [[f7-truncation-confounder-20261021]] [[aime25-heldout-iter2-nogeneralize-20261021]] [[deepseek-three-layer-reassess-20261021]] [[direction1-design-review-20261021]].
