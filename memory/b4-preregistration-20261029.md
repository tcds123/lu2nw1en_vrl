# B4 预登记 — cross_q_consistency 假设检验 (2026-10-29)

**写于跑数之前**。目的是固定判据, 防止事后挑 theta / 挑口径解释结果。

## 数据

`rl/train/verl1/outputs/skill_evolver_aime_v2/run_dir2_heldout_gate_rep5_20261027/run_20261027_093001/paired_eval_samples.jsonl`
(13,442 行; parse_status: parsed 13,440 / abandoned 2)

- 17 个 evolver 候选: facet ×14 (iter1-5), overlay ×5 (iter1-5, **cand==base 同一 skill_version `c11d7dc0c8`**), expand ×2 (iter4-5)
- 3 个 held-out 候选: `699d772c42`(facet iter1) / `c11d7dc0c8`(overlay iter3) / `82ca7d589e`(expand iter4)
- base 每 iter 重新采样一次 (不是共享缓存) → 每个候选与其同 iter base 配对

## 指标定义 (与 checklist B1 一致)

对每个候选:
1. 每题的 cand pass rate = mean(k=8 的 score) 对照同一 iter 的 base 同题 pass rate
2. `delta_i = cand_i − base_i`, 30 题 → 30 个数
3. `pos = #{delta>0}`, `neg = #{delta<0}`, `zero = #{delta==0}`
4. `consistency = pos / (pos + neg)` (zero 不进分母; 全 zero 则返回 None)

## 预登记判据

**主检验 (训练集 17 候选)**:

| 结局 | 判据 | 解读 |
|---|---|---|
| (a) 有分离 | consistency 的极差 ≥ 0.20 **且** 存在 theta 使"选中候选的 held-out delta > 0" | cqc 可作 selection gate |
| (b) 无分离 | 17 个候选 consistency 全部落在 [0.40, 0.60] (即 pos 在 12-18/30) | 信号是噪 — 转向 generation 层; cqc 放弃 |
| (c) 过度分离 | 多数候选 consistency 极端 (>0.8 或 <0.2) | 说明 per-item 高度相关, 30 题不构成独立样本 → 需换统计量 (如 block bootstrap) |

**预期结局: (b)**。理由: D2 诊断已判 17 候选整体为人 winner's-curse 噪声 (均值 +0.0088 < 1SE, 9 正 5 负对称, identity 候选真效应=0 却 in-sample +0.083)。若效应是噪声, 每题符号也应 ~随机。

**佐证/反证的关键量**:
- **identity 候选** (`c11d7dc0c8` overlay 5 个): 真效应 = 0。它的 consistency 若显著偏离 0.5, 说明**指标本身有偏 / 存在系统性顺序效应**, 而非候选真有信号。这是上界: 任何"有效" 的 gate 都必须让 identity 候选的 consistency 不高于真正有效候选。
- **held-out 段的 3 个候选**: held-out 30 题与训练 30 题不重叠。若训练集排出的"最一致"候选在 held-out 上也是正向, 才算跨集一致。

## 技术限制 (不在 B4 内解决)

- per-item delta 是**单 rep 口径**: 无法估跨 rep 稳定度。设计 §2 的 theta≈0.10 是跨 rep 口径 (2/30), **不能套**。
- k=8 的 cand/base 各自有采样噪声 (SE≈0.02/题, 30 题 ⇒ delta 的 SE≈0.018-0.025)。这本身就会让 pos/neg 对称化。故 (b) 是"两种来源不可分"的弱阴性, 不是"信号不存在"的强证明。
- B4 **零 API**, 不跑 evolver。

## 输出

`memory/b4-result-20261029.md` — 逐候选表 + 结局判定 + 对 B5 (`--cross_q_theta`) 的建议。
