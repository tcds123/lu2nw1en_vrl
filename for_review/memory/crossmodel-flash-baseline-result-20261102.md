---
name: crossmodel-flash-baseline-result-20261102
description: flash 在 AIME24 30题 k=8 的 pass@8 基线 = 0.854 (V3.2 同 skill 0.597); 21/30 题满分 → 饱和, 对 skill 效应无分辨力
metadata:
  type: project
---

# flash 跨模型 AIME24 基线 — 结果 (2026-11-02)

启动记录: [[crossmodel-flash-baseline-launch-20261102]]。run: `rl/train/verl1/outputs/crossmodel_baseline/run_20261102_170450/`
(PID 106121 正常退出, 37.8 min, 240 请求, 零报错)。

## 数字

| 臂 | 模型 | skill | k | all 30 | train 18 | val 12 | 0 分题 | 满分题 |
|---|---|---|---|---|---|---|---|---|
| **flash** | deepseek-v4-flash | init v2 (599 chars, `c11d7dc0c8`) | 8 | **0.8542** | 0.8958 | 0.7917 | 2 | **21** |
| V3.2 | DeepSeek-V3.2 | 同上 | 8 | 0.5974 | — | — | 2 | 5 |

- V3.2 一行为 rep5-fixed run 的 base 臂 (4800 条, 30 题 × k=8 × 5 轮, 逐题均值), 同 skill 同题同 temp。
- flash 长 prompt 实测 ~57s/请求 @20RPM (dry-run 的 25.9s 是 1 题, 不代表均值)。

## 结论

1. **flash 在 AIME24 上显著强于 V3.2** (+0.257), 但**基线已饱和**: 21/30 题 8 次全对。
   → **用 AIME24 做 flash 的跨模型 skill 评测基本无分辨力**: 天花板 ∧ 噪声地板都在 0.1 量级,
   而可用提升空间只有剩 9 题 (其中 2 题为 0 分硬骨头)。**要用 flash 验证跨模型泛化, 必须换更难的题库**
   (AIME25 held-out / AIME-以上难度), 或者去饱和口径 (只看 base<1 的题, n≈9, SE 过大不可用)。
2. val 12 题 (0.7917) 比 train 18 (0.8958) 低 0.10 — 但这是 **30 题里按 seed 42 切的分**, n=12 的 SE 本身约 0.1, 不构成"泛化差"的证据。

## 观测质量的可信度 (必须与数字一起读)

- **没有逐样本持久化**: `precompute_s0` 本就不调 `capture_cb` → 只有 `s0_cache_aime.json` (30 条 `pass_rate`),
  没有 raw_response / finish_reason / completion_tokens。**无法核验是否有截断、无法核验解析失败**。
- 两题 0 分 (`0.0`) 原因未知: 可能是 API 失败 / 空响应 / 解析失败 / 真答错 —— 三种都记 0, 未区分。
  → **0.854 是"乐观上界", 若 0 分里含 API 失败则真实值更低**。判据: 重跑那两题并看 parse_status。
- 若要正式采用该基线, 需给 `precompute_s0` 加 capture_cb (或改用 `eval_paired` 跑 S₀ vs S₀, 后者已持久化)。

## 口径注意

- 本轮 `run_one` 的 4096 字面量被进程内替换为 **16384** (cap 隔离, 见 launch 记录) → 与 rep5-fixed 的
  V3.2 数字 **不是同一 cap 口径**。故上表 V3.2 行仅作量级对照, 不作严格配对比较。
- 顺带: V3.2 base 臂响应 max=16501 chars / mean 7666 / >14000 的 149 条 (占 4800 的 3.1%) →
  这些贴着 ~16.5k 的天花板, **再次支持 in-sample cap 是 4096 tokens 而非 16384** (16384 tok 应允许 ~65k chars),
  详见 `docs/phase2_issues.md` #18。若成立, 则 V3.2 行被截断压低, 0.597 是**悲观下界**, 跨模型差距实际更小。

## 关联

- [[crossmodel-flash-baseline-launch-20261102]] — 启动参数与 cap 隔离说明
- [[cross-model-configs-aimp-20261030]] — flash 接入
- [[measurement-calibration-1a-20261028]] — 噪声地板 / 检出下限口径
- `docs/phase2_issues.md` #18 — F7 in-sample cap 遗留 (本条提供了新证据)
