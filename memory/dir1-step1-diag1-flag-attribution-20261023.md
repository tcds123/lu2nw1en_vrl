---
name: dir1-step1-diag1-flag-attribution-20261023
description: 诊断1 — ON-flag iter2 晋升的 best_skill 是 facet_edit(吃flag注入路径), OFF-run6d 从未产出该 bullet, flag确有效(但rescue门极薄 ci=+0.004)
metadata:
  type: project
---

# 诊断1: Direction 1 Step 1 flag 归因 (2026-10-23)

## 缘起

用户审核 held-out 结果后指出关键盲点: "你测的根本不是 ON flag 的效果" — 因 best_skill 是 iter1 overlay (n_axes_combined=0), 而 ON flag 的 facet_edit 候选 6/6 admitted 但仅 1 promoted, 全 KEEP-PARENT(NO-CI), 无一因 flag 晋升。推论: held-out +0.0153 是"overlay(与flag无关)在held-out的表现", 非 flag 对照实验。

→ 启动诊断1 (零重跑, 纯读码+archive): 确认晋升路径是否真与 flag 无关。

## ⚠️ 纠错 (Lesson #5): 我此前 memory 文件的晋升来源写错了

我之前在 `dir1-step1-heldout-result-20261022.md` 写 "唯一晋升=iter1 overlay (n_axes_combined=0, 纯加4个section)"。**这是误读 evolver_log 的 best_val_delta 字段** (paired_rescore=on 时它是 paired Δ 非 cumulative)。逐字节复核 best_skill_evolved.md 后发现:

- `best_skill_evolved.md` (hash 9c9bc27e84, 705c) == `archive_iter2.json` node[9] **逐字节相同**
- node[9]: `parent="Computation & Verification"`, delta=+0.0375, ci=+0.0000 → **是 facet_edit 候选, 非 overlay**
- iter1 overlay (hash c11d7dc0c8=S₀本身, delta=+0.075 paired) **不是最终 best_skill**; best_skill_iter1.md 也是 S₀ 原文 (599c)

evolver.log L32 印证: `✅ FACET-RESCUE-REJECT iter2[Computation & Verification]: delta=+0.037 src=subset sig={'meta': -inf, 'not_meta': 0.004}` → 经 subset 救援晋升。

## 真正的晋升路径拆解

| 层 | 机制 | 吃 flag? |
|---|------|---------|
| 候选生成 | `generate_candidates` (L4781) 注入 POSITIVE_SECTION(learn_positive) + META_CONTRAST_SECTION(meta_contrast) 进 LLM patcher prompt | **是** (L1254-1277) |
| 晋升门控 | `subset_promote_decision` / `group_promote_decision` 用 `classify_item_meta`(零API分类器) 切 meta/not_meta 子集看 ci | **否** (与flag独立) |

→ **用户的critique在门控层对(晋升机制与flag无关), 在内容层错(晋升的bullet是flag注入路径生成的)**。

## 决定性证据: ON vs OFF 对比 (2026-10-23 修正归因措辞精度)

ON-run 与 OFF-run6d 共享 **完全相同** 的 probe/s0_cache/seed (唯一变量=flag + iters):
- s0_cache: run4_measure_foundation/run_20261016_140819/s0_cache.json (同)
- shuffle_seed=0 (同), train_n=18 (同), n_select=30 (同), probe_n=30 (同), split_dir=aime24.jsonl (同)
- ON: learn_positive=True, meta_contrast=True, iters=2
- OFF(run6d): learn_positive=False, meta_contrast=False, iters=10

### ⚠️ 归因措辞修正 (用户审核指出 + 逐字复核, 2026-10-23 第二轮)

**此前本节写 "OFF 从未产出此 bullet" 是 grep 口径错误** (见教训 #8)。逐字复核后:

ON 晋升的 bullet (best_skill_evolved.md, Computation & Verification 段):
```
- After each major computation, test the result against extreme or simplified cases to detect gross errors.
  (hash 9c9bc27e84, delta=+0.0375, ci=0.0, 经 subset rescue not_meta ci=+0.004 晋升)
```

OFF-run6d **反复生成同概念的不同措辞候选** (archive_iter2~9 每轮 node[8], hash f0ade174da 跨轮持久化, delta=−0.0083, ci=−0.0792):
```
- Before finalizing, test extreme cases or alternative counting methods if the answer is unexpectedly small or large.
  (hash f0ade174da, delta=−0.0083, ci=−0.0792, iter2→9 连续8轮持续产出, 从未晋升)
```

→ 精确短语 grep 仍证实: "extreme or simplified" / "After each major computation" / "detect gross errors" 三个 ON 专属短语在 OFF-run6d 全目录 NOT FOUND。**即 OFF 产出的是同概念变体, 非 ON 那颗 bullet 逐字复现。**

### 归因真相 (比"全新信号"/"结晶"都精确)

| 表述 | 对错 | 理由 |
|------|------|------|
| "flag 产生了全新的跨题推理信号" | ❌ | extreme-case 概念 OFF iter2 就生成 |
| "flag 把散落/隐含概念结晶成 bullet" | ❌ 偏轻 | OFF 那颗本就是显式 bullet, 只是被门控拒绝 |
| **"flag 把 OFF 反复生成却连续8轮被拒 (delta=−0.0083 ci=−0.0792) 的 extreme-case 概念, steer 成了 in-sample +0.0375 过门的措辞变体"** | ✅ | 同概念, ON 专属措辞, delta/CI 翻正 |

→ 在同 probe/同 cache/同 seed 下, ON 的措辞变体 in-sample +0.0375 过门并 held-out +0.0153, OFF 同概念变体 in-sample −0.0083 连续8轮被拒。**flag 影响了候选措辞使其过门**, 但严谨性更强 (见下)。

### 严谨性强化 (比修正前更脆弱)

- **概念非 flag 独有**: extreme-case 自检概念 OFF 也有 (只是被拒), 故 +0.0153 不能归因于"flag 造出新能力"。
- **措辞变体过门 vs 测量噪声**: ON +0.0375 vs OFF −0.0083 的差 0.046 在 n=30 probe 上**不排除测量噪声**; held-out +0.0153 < 单 seed σ≈0.044。
- **rescue 门极薄**: not_meta ci=+0.004 (仅高于 0 门 0.004, bootstrap 噪声量级)。若 ci=−0.004 就不晋升 → 几乎掷硬币。
- **样本 n=1**: 仅 1 个 ON run (1 个晋升点) 对比 1 个 OFF run。单 run 无法区分 "flag steer 措辞真有效" vs "采样运气/单 bullet 偶然"。
- **overlay 仍是无功机制**: iter2 overlay delta=+0.0208 ci=−0.042 admitted=N (未晋升)。overlay 在本 run 是无功的。

→ **"flag 影响候选措辞过门"已证 (硬证据); "flag 是 held-out +0.0153 的因果来源"仍需 rep 坐实。** 报周报/汇报时不得把"影响候选措辞(已证)"和"因果有效(待证)"混成一句。

## ⚠️ 严谨性标注 (正向但脆弱, 2026-10-23 修正后更脆弱)

- **概念非 flag 独有** (修正后新增): extreme-case 自检概念 OFF-run6d iter2 即生成 (hash f0ade174da), 故 +0.0153 不能归因于 "flag 造出新能力"。
- **措辞变体过门 vs 测量噪声**: ON +0.0375 vs OFF −0.0083 的差 0.046 在 n=30 probe 上不排除测量噪声; held-out +0.0153 < 单 seed σ≈0.044。
- **rescue 门极薄**: not_meta ci=+0.004 (仅高于 0 门 0.004, bootstrap 噪声量级)。若 ci=-0.004 就不晋升 → 几乎掷硬币。
- **iters=2 太短**: ON 只跑了 2 轮, flag 候选 6/6 admitted 但多是 KEEP-PARENT(NO-CI) (iter1 三个 facet_edit ci 全负)。flag 机制打开了但 2 轮内只有 1 个候选经救援晋升。
- **样本 n=1**: 仅 1 个 ON run (1 个晋升点) 对比 1 个 OFF run。单 run 无法区分 "flag steer 措辞真有效" vs "采样运气/单 bullet 偶然"。
- **overlay 仍是无功机制**: iter2 overlay delta=+0.0208 ci=-0.042 admitted=N (未晋升)。overlay 在本 run 是无功的 (与 held-out +0.0153 无关)。

## 判定 (修正后)

- **flag 不是无功的** (推翻用户"测的不是flag效果"的结论): ON 晋升的 bullet 来自 flag 注入路径 (facet_edit L4781), 且 ON 专属措辞在 OFF-run6d 10轮从未产出 (精确短语 grep 证实)。
- **但 flag 的因果强度未坐实** (修正后更明确): 概念 OFF 也有 (只是被拒, delta=−0.0083 ci=−0.0792), 故更精确的归因是 "flag 把 OFF 反复生成却连拒8轮的 extreme-case 概念 steer 成了 in-sample +0.0375 过门的措辞变体", 而非 "flag 造出新能力"。rescue 门 ci=+0.004 极薄, 单 run, 无法排除采样运气。
- **held-out +0.0153 可归因于**: flag 注入路径产出的 ON 专属措辞变体在 held-out 上有效 (而非 overlay), 但需 rep 坐实因果。
- ⚠️ **汇报口径红线**: "flag 影响候选措辞过门(已证)" ≠ "flag 是 +0.0153 的因果来源(待证)", 不得混成一句。

## 下一步 (诊断1 结论驱动的分支, 2026-10-23 修正后)

诊断1 回答了用户的问题 "flag 候选离晋升门多远": **逼近并跨过了** (not_meta ci=+0.004, 刚过门)。修正后更精确: flag 候选不只是 close, 而是 ON 专属措辞变体 in-sample +0.0375 过门 (OFF 同概念变体 delta=−0.0083 连拒8轮)。所以:
- 不是 "flag 候选从不 close → 2轮不够" (flag候选确实 close 并晋升了)。
- 也不是 "flag 机制真的弱 → 转Direction2" (ON 专属措辞 in-sample 正向 + held-out 正向)。
- **是第三种**: flag steer 了措辞变体过门, 但概念非 flag 独有 + rescue 门极薄 + 单 run → 因果未坐实。

→ **最干净的下一步**: 跑 iters≥5 ON-flag rep (同 probe/同 cache), 三件事:
1. "extreme/simplified cases 自检" 类 ON 专属措辞是否稳定产出 (逐轮 skill diff 存档, 用户建议零成本前置);
2. held-out Δ 是否稳定在正区间;
3. 方向一致率能否到 3/3。

若 rep 稳定正向 → flag steer 因果坐实 (措辞变体过门非单次运气), 推进 Step 2 (写 make_staleness_negatives + {staleness_section}); 若 rep 翻负/不稳 → 信号是采样运气, 转 Direction 2。

Step 2 设计可标 "open" 先不写码 (flag 因果仍需 rep 坐实)。

## 产物路径

- ON archive: `outputs/skill_evolver_aime_v2/run_dir1_step1_onflags/run_20261022_091146/archive_iter2.json` node[9]
- OFF archive: `outputs/skill_evolver_aime_v2/run6d_two_batch_paired/run_20261019_162330/archive_iter{1..9}.json`
- 管道验证: `skill_evolution/skill_evolver_aime_v2.py` L4781-4787 (flag 注入), L561-641 (rescue 门), L2684-2714 (_overlay_assemble 纯重组)

关联 [[dir1-step1-heldout-result-20261022]] [[dir1-step1-onflags-launch-20261021]] [[f7-16384-baseline-result-20261021]] [[direction1-design-review-20261021]] [[deepseek-three-layer-reassess-20261021]].
