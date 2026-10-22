---
name: lcb-stratified-baseline-20260721
description: LCB 分层 baseline 实验启动 - 旧窗口(2023Q3-Q4, 肯定见过) vs 新窗口(2024Q4-2025Q1, 没见过) noskill 对比, 直接证明/排除数据污染
metadata:
  type: project
---

# LCB 分层 Baseline (2026-07-21 启动) 🔥

## 背景与判别逻辑
[[lcb-contamination-task-20260720]] 的 LCB noskill baseline 已出: test=79, **hard=0.4304 / soft=0.5243** (零 429)。这远低于 appsplus 的 0.6564, 初步支持"数据污染假说成立"——旧基准高分可能因 DeepSeek-V3.2 见过 mbpp/humaneval/APPS。

但单点 0.43 还不够直接。用户确认要做**分层 baseline**: 同一 DeepSeek-V3.2 noskill, 对比不同 contest_date 窗口, 直接证明污染。

- **OLD 窗口** (2023-07-01 ~ 2023-12-31, Q3+Q4): 肯定见过 (V3 预训练截止 2024-10/11)
- **NEW 窗口** (2024-10-01 ~ 2025-03-31, 2024Q4+2025Q1): 没见过 (训练截止后)
- **判读**:
  - OLD baseline 显著 > NEW baseline → 坐实数据污染
  - 两窗口接近 → 模型本身强, 污染非主因

## 窗口题量与难度分布 (关键混杂变量)
- OLD: n=138, easy50/medium47/hard41 (难度均衡)
- NEW: n=204, easy48/medium48/hard108 (hard 占比高, 更难)
- ⚠️ NEW 更难 → 即使无污染, NEW 也会偏低。**必须按 difficulty 分层统计**, 不能只比总 pass-rate
- split ratio 2:1:7 (seed42 确定性): OLD → train28/val14/test96; NEW → train41/val20/test143

## 实验参数
- BENCH=lcb, mode=noskill, RATIO=2:1:7, workers=8, max_turns=1, T=0.0
- 后端 DeepSeek-V3.2 (aimpapi.midea.com), QWEN_CHAT_MAX_RPM=28 (token bucket 防污染)
- 空技能 (empty_skill.md), 全量 test eval
- OLD: `CONTEST_DATE_AFTER=2023-07-01 CONTEST_DATE_BEFORE=2024-01-01`
- NEW: `CONTEST_DATE_AFTER=2024-10-01 CONTEST_DATE_BEFORE=2025-04-01`

## 关键工程信息
- contest_date_before 已在 dataloader 支持 (`< before` 丢弃, 闭下界开上界)
- results.json 含 id 但不含 difficulty → 跑完后用 dataloader id→difficulty 映射做分层统计
- 命令: `cd /data/zhuldz/skillopt && BENCH=lcb CONTEST_DATE_AFTER=... CONTEST_DATE_BEFORE=... bash run_code.sh noskill > outputs/lcb_stratified_<OLD|NEW>.log 2>&1 &`
- 输出目录: outputs/lcb_2-1-7_noskill_all (两窗口同名! 必须用不同 TAG 或分开跑/重命名)

⚠️ **TAG 冲突**: 默认 TAG=lcb_2-1-7, 两窗口输出都写到 outputs/lcb_2-1-7_noskill_all 会互相覆盖。用 TAG 环境变量区分: `TAG=lcb_OLD` / `TAG=lcb_NEW`。

## 待办
- [x] 跑 OLD 窗口 noskill (test=96) → hard=0.7083 soft=0.8002, agent_ok=96/96
- [x] 跑 NEW 窗口 noskill (test=143) → hard=0.4266 soft=0.5444, agent_ok=143/143
- [x] 分层统计: 按 easy/medium/hard 分别比 OLD vs NEW pass-rate
- [x] 结论写入此文件 + 更新 MEMORY.md 索引

## ✅ 结果 (2026-07-21 完成)

两窗口 agent_ok 均 100% (零 429/零污染), 对比干净。

| 难度 | OLD(seen) hard | NEW(unseen) hard | delta | OLD n | NEW n |
|------|---------------|------------------|-------|-------|-------|
| easy | 0.9714 | 0.9706 | +0.0008 | 35 | 34 |
| medium | 0.5758 | 0.5000 | +0.0758 | 33 | 26 |
| hard | 0.5357 | 0.1807 | **+0.3550** | 28 | 83 |
| **TOTAL** | **0.7083** | **0.4266** | +0.2817 | 96 | 143 |

复现性: 本次 NEW(2024-10~2025-03) hard=0.4266 ≈ 07-20 lcb_noskill_2025(≥2025-01-01) hard=0.4304, 一致 ✅

## 结论 (判读)

1. **污染假说方向性成立**: NEW(unseen) 总分 0.4266 << appsplus(seen) 0.6564, 也 << OLD(seen) 0.7083。旧基准高分确有"见过"成分。

2. **但信号全在 hard, easy 在天花板**: easy 两窗口几乎相同 (0.971 vs 0.971) — 模型对简单题已饱和, 见过与否无差 (ceiling effect, 无信息量)。medium n 小且差距小 (+0.076, 噪声)。**hard 出现巨大 gap: OLD 0.54 vs NEW 0.18 (+0.355)**。

3. **hard gap 的两难归因 (诚实保留)**:
   - 解释A (污染): DeepSeek-V3.2 见过 2023 hard 题 → 背答案 → OLD 0.54; 2025 hard 没见过 → 真实能力 0.18。
   - 解释B (难度漂移): 2025 codeforces/atcoder hard 题客观上比 2023 更难 (算法更复杂), gap 是真实难度差, 非污染。
   - 两者无法仅凭此实验完全分离。但无论哪个解释, **NEW hard 0.18 是干净的真实能力, headroom 巨大**。

4. **总分层差距 (+0.28) 同时受两混杂驱动**: (a) NEW 难度分布更难 (hard 占 83/143=58% vs OLD 28/96=29%); (b) within-hard OLD 更高。所以总分差不全是污染, 部分是分布差。within-difficulty 对比更可信, hard 的 +0.355 是最强污染信号。

## 对论文叙事的意义
- 代码域"模型饱和/不适配"结论需修正: **在无污染 (LCB unseen) 的 hard 题上, DeepSeek-V3.2 真实 pass-rate 仅 0.18, 远未饱和, headroom 巨大**。这给 skill 优化留出真实作用空间。
- 之前 appsplus/mbpp "test 不泛化"的负结果, 部分可归因于 seen 数据天花板 (noskill 已 0.66, 优化无空间) 而非"skill 对代码无用"。
- **下一步建议**: 在 LCB NEW (无污染) 上跑 SkillOpt optimize, 验 skill 能否在 unseen hard 题上提分并 test 泛化。train=41/val=20 偏小是风险。

Related: [[lcb-contamination-task-20260720]], [[appsplus-full-test-reversal-20260718]], [[dataset-transition-narrative-20260716]], [[skillopt-step0-results-20260714]]

Related: [[lcb-contamination-task-20260720]], [[appsplus-full-test-reversal-20260718]], [[dataset-transition-narrative-20260716]]
