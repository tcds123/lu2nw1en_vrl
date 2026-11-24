---
name: p4-shortcut-fix-diagnosed-20260924
description: 复审§13.10优先级#4(P4)捷径修复诊断——向 other_60 注入可证伪 meta 干扰项, 隔离"meta文本触发"捷径 vs 真实内容推理
metadata:
  type: project
---

2026-09-24：启动复审 §13.10 优先级 #4（P4）捷径修复诊断。[[p3-factor-ablation-launched-20260924]] 之后执行。

## 核心问题（复审 §13.10#4 + §12.6）

ON 的 meta_46 增益（0.5380 vs S0 0.2011）是来自**文本触发捷径**（"出现 meta 措辞 → 选它"）还是**真实内容推理**（按数学内容判断）？shuffle 只切断位置捷径, 仍保留"某措辞出现即正确"的文本捷径。P3 证明 val 专属内容 hurt + A 句泄漏 hurt, 但 on_noAsentence 仍 > s0 → 残留增益是 meta 规则还是真推理? 必须用**语义反例**区分。

## 设计：向 other_60 注入可证伪 meta 干扰项

对 60 道 other 题（无 meta 选项, 原正确答案保留）构造 3 个条件, 全部 shuffle + 同判分:

| 条件 | 选项数 | 第6选项(F) | 可证伪性 |
|---|---|---|---|
| **base5** | 5 | 无 | — |
| **+meta_false** | 6 | "One of the remaining options is correct, but a stronger result can be proven." | **语义待核验**（见下） |
| **+meta_none** | 6 | "None of the above options is correct." | **可证伪**（原正确答案保留→此句必假） |
| **+normal6** | 6 | 普通错误干扰项（取其他题的选项文本） | 控制纯"多一选项/更长上下文"效应 |

### 关键 §13.9 约束：meta_false 不可盲标错误

复审 §13.9 明确警告: "正确答案等于原定理"**不自动推出** meta 一定错误——原定理可能不是最强结论, "更强"也可能相对另一较弱干扰项解释。因此:
- **+meta_none**（"以上均不正确"）是复审**认可的可证伪**构造: 保留原正确答案→该句逻辑上必假。无歧义。
- **+meta_false**（"更强结论"措辞）: 复审认为需**语义核验**才能标错。本实验**不假定它为 gold 错误**, 而是把它作为**文本触发探针**——观测模型是否被相同措辞诱导选它, 而非判其对错。`+meta_false` 的 F 选项**永远不算正确**（gold 仍是原正确答案）, 但因语义歧义其"错误性"标注 caveat。
- 两措辞都测, 因 +meta_none 测"一般 meta 误用", +meta_false 测"与原 stronger-result 规则相同措辞的敏感度"——回答不同问题（复审 §13.9 末句）。

### 判读框架

对 other_60 报告 4 个指标（每条件）:
1. **solving_acc** = 选中原正确答案的比例（真实解题能力, 与 base5 对比）
2. **meta_selection_rate** = 选 F（meta/none 选项）的比例
3. **parse_status 分布**（strict: clean_exact/ambiguous_first_token/no_valid_label）
4. **finish_reason 分布**（截断率, length 占比）

**关键对比**:
- 若 ON 的 `+meta_false`/`+meta_none` 的 meta_selection_rate **远高于** S0 → ON 学到了"meta 文本→选"捷径（文本触发非内容推理）
- 若 ON 的 solving_acc 在 +meta_* 下**显著低于** base5 → 捷径把模型从正确答案拉走（hurt）
- 若 ON 的 meta_selection_rate ≈ S0 且 solving_acc 稳定 → 非捷径, 真有内容判别（或至少不靠该措辞）
- +normal6 控制: 若 +meta_* 与 +normal6 下降相当 → 只是"多一选项"的稀释效应非 meta 特异

**对饱和叙事**: 若 ON 被证捷径, 进一步削弱"ON 增益来自有效推理"; 若不被证, 残留增益更可信。这是 P3 后的关键补刀。

## 协议细节

- 复用 `eval_test_multiseed_shuffle.py` 的 shuffle_item/stratify/eval_skill_on 协议原语, 但**重写 eval 以采集 predicted_label**（原 eval_skill_on 只返回 pass_rate, 丢标签）。新 driver 用 `run_one` 拿 predicted_label。
- shuffle: 每 (qid, seed) 确定性打乱 + remap gold（与 P3 字节一致）。6 选项时 F 也参与 shuffle, gold remap 到新 label。
- k=3 投票, seed=1..4, test=106（只跑 other_60 注入, meta_46 原样跑作对照）。
- skills: s0 / on / on_noAsentence（P3 证明删 A 句后仍有 +0.156 真实增益, 须看它是否也捷径）。
- 规范解析（§13.10#4）: 用 `classify_parse_status` 标 clean_exact/ambiguous_first_token/no_valid_label; 报告多标签/无标签/截断状态, 全部入分母。
- **分别报告** meta-selection 行为 vs solving accuracy（复审明确要求分开, 不混报"含meta题正确率"）。

## 执行

- driver: `for_review/code/p4_shortcut_diagnosis.py`；launcher: spawn-detach（复用 launch_p3 模式）。
- 规模: 60 other × 3 条件(+base5/+meta_false/+meta_none/+normal6 共4) × 3 skill × 4 seed × k=3 ≈ 8640 次 API（~5h+ @ 30/min）。先跑单 seed smoke 验证探针有效再全量。

**Why**: P3 只切断位置捷径未切断文本捷径; on_noAsentence 残留增益的归因（meta 规则真有用 vs 文本触发捷径）必须用语义反例才能定。复审 §12.6/§13.9/§13.10#4 的核心诊断。
**How to apply**: 跑完回填本文件指标表+判读, 更新 docs/phase2_issues.md 优先级 #5（原#4"修捷径诊断"）状态, 再进饱和/路由研究。

## 🔴 单 seed smoke 初步结果 (2026-09-24, seed1, k=3, other_60)

> smoke PID 119516 (launch_p4_shortcut.py, 20:04 启动). 探针**已验证有效**——ON 对 "stronger result" 措辞表现出强文本触发捷径。下表为已完成 cell（on/meta_none、s0+on 的 normal6、on_noAsentence/* 待续）。

**⚠️ 下表为单 seed smoke 的样本级 (n=180=60 题×k=3) 权威数值（取自 `p4_shortcut_summary.json`）。早期按 last-sample(60 题) 估算的代理值已废弃——以本表为准。**

| skill | cond | solving_acc | meta_sel(F) |
|---|---|---|---|
| s0  | base5     | 0.389 | 0.000 |
| s0  | meta_false | 0.389 | 0.039 |
| s0  | meta_none  | 0.444 | 0.078 |
| s0  | normal6    | 0.400 | 0.000 |
| on  | base5      | 0.411 | 0.000 |
| **on** | **meta_false** | **0.233** | **0.339** |
| on  | meta_none  | 0.333 | 0.111 |
| on  | normal6    | 0.367 | 0.000 |
| on_noAsentence | base5 | 0.394 | 0.000 |
| **on_noAsentence** | **meta_false** | **0.278** | **0.339** |
| on_noAsentence | meta_none  | 0.333 | 0.106 |
| on_noAsentence | normal6    | 0.411 | 0.000 |

### 关键判读（smoke 全 12 cell 跑完; 4 seed 全量待补）

- **措辞特异捷径实锤（核心）**: ON/on_noAs 选 F 比例 **meta_false 0.339 ≫ meta_none ~0.11 ≫ normal6 0.000**。normal6（普通干扰项第6选项）meta_sel=0% 且 solving≈base5 → **"多一选项/更长上下文"纯稀释效应几乎为 0**。故 meta_false 的 0.339 选择**完全由 "stronger-result" 措辞触发**, 非选项数变化。ON skill L4 明确编码该 meta-trigger → 模型 pattern-match 到自己训练时的触发短语。
- **一般 meta 误用较弱**: meta_none ("None of the above") on 的 meta_sel 0.111 > s0 的 0.078, 但远低于 meta_false 的 0.339 (z=1.08 不显著) → "一般 meta 措辞"的轻度敏感在噪声范围内, 主捷径是"特定 stronger-result 措辞"。
- **捷径伤解**: on/meta_false solving 0.233 vs on/base5 0.411 (**−0.18**); on_noAs 0.278 vs 0.394 (−0.12); normal6 持平 → 措辞把模型从正确答案拉走, 主动 hurt, 与选择率正相关。
- **s0 基线对照**: s0 四条件 solving 持平 (0.389~0.444), meta_sel ≤0.08 → 几乎不被任何 meta 措辞诱导。捷径是 ON 训练产物特有, 非模型本身固有。
- **🔴 on_noAsentence 残留捷径 = on（关键修正）**: on_noAsentence/meta_false meta_sel **0.339，与 on 完全相同** → 删 A 句+val 专属内容**完全未降低**文本触发捷径。→ **P3 的 on_noAsentence 残留增益 (+0.156) 实质上是捷径驱动的**——on_noAsentence 仍同样 pattern-match "stronger-result" 触发短语。这推翻 P3 "on_noAsentence 残留增益=真实增益/meta 规则合法" 的乐观判读。（注：早期 last-sample 代理显示 on_noAs 31.7% < on 40% 是采样噪声；样本级两者一致 0.339。）
- **other_60 上 on vs s0 base5 持平 (0.411 vs 0.389)**: ON 的解题增益集中于 meta_46 题, 在无 meta 的普通题上不优于 s0 → 呼应复审"ON 增益可能是 meta-特异捷径而非通解题力"。
- **统计**: on vs s0 meta_false bite, 两比例 z = 7.27 (n=180 each) → 极显著。

**结论（smoke 单 seed, 待 4 seed 全量确认方差后定稿）**: ON 的 meta_46 增益**主要来自"stronger-result 措辞 → 选该选项"的文本触发捷径**, 非纯内容推理; 且删 A 句/val 内容后残留增益**仍同样捷径**(meta_sel 不降)。这与 P3（val 专属内容 hurt）共同坐实 ON 过拟合到 val 特定措辞。复审"饱和结论过度"的元判断中"数据有捷径(1)"项得到 P4 实锤支撑——捷径不仅是位置(gold恒A, 已被 shuffle 切断), 还有**残留的文本触发捷径**(shuffle 不切断措辞)。单 seed z=7.27 信号极强, 但仍应 4 seed 复测以坐实方差与 paired 统计。

## ✅ 4-seed 全量最终结果 (2026-09-25, seed1-4, k=3, other_60, n_samp=720/cell)

> PID 102979 (21:44 启动, 02:25 完成, ~4.7h, exit 0). seed1 复用 smoke cache, seed2/3/4 跑新. 方差极紧 (on/meta_false 4 seed bite: 0.339/0.356/0.350/0.306, sd=0.022). **smoke 单 seed 结论全部被 4 seed 坐实。** 以下为权威样本级 (n=720) 数值。

| skill | cond | solving_acc | meta_sel(F) |
|---|---|---|---|
| s0  | base5     | 0.410 | 0.000 |
| s0  | meta_false | 0.383 | 0.043 |
| s0  | meta_none  | 0.381 | 0.094 |
| s0  | normal6    | 0.404 | 0.000 |
| on  | base5      | 0.410 | 0.000 |
| **on** | **meta_false** | **0.249** | **0.338** |
| on  | meta_none  | 0.360 | 0.108 |
| on  | normal6    | 0.407 | 0.003 |
| on_noAsentence | base5 | 0.415 | 0.000 |
| **on_noAsentence** | **meta_false** | **0.261** | **0.356** |
| on_noAsentence | meta_none  | 0.350 | 0.099 |
| on_noAsentence | normal6    | 0.426 | 0.001 |

### paired 统计（4 seed 全量）

- **on vs s0 meta_false bite**: 样本级两比例 z=**14.23** (n=720 each), p≈0; per-item paired t=**10.40** (df=59, mean_diff=+0.294), Wilcoxon z=6.39 p=1.6e-10 → 极显著。
- **on_noAsentence vs s0 meta_false bite**: z=14.84, p≈0; paired t=10.37 (mean_diff=+0.313), Wilcoxon p=1.5e-10 → 极显著。
- **🔴 on_noAsentence vs on meta_false bite（关键修正, 4 seed 坐实）**: paired t=**0.78** (df=59, mean_diff=+0.018), Wilcoxon z=0.72 **p=0.471 不显著** → **删 A 句+val 内容完全未降低文本触发捷径**, 与 smoke 单 seed 结论一致。P3 的 on_noAsentence 残留增益 (+0.156) 实质仍捷径驱动。
- **捷径伤解**: on/meta_false vs on/base5 paired, t=−5.40 (mean_diff=−0.161), Wilcoxon z=−4.93 p=8.1e-07 → 措辞把模型从正确答案拉走, 极显著主动 hurt。
- **normal6 稀释控制**: 三 skill meta_sel ≤0.003, solving≈base5 → 纯"多一选项"效应≈0, meta_false 的 0.338/0.356 选择完全由 "stronger-result" 措辞触发。
- **per-seed 方差**: on 0.337±0.022, on_noAs 0.356±0.024, s0 0.043±0.012 → sd 均 ≤0.024, 信号/噪声比极大。

## ✅ P4 最终结论（4 seed 定稿）

ON 的 meta_46 增益**主要来自"stronger-result 措辞 → 选该选项"的文本触发捷径**, 非纯内容推理; 删 A 句/val 内容后残留增益**仍同样捷径**（on_noAs vs on 不显著 p=0.471）。这与 P3（val 专属内容 hurt）共同坐实 ON 过拟合到 val 特定措辞。复审"饱和结论过度"元判断中"数据有捷径(1)"项得到 P4 实锤支撑——捷径不仅是位置（gold恒A, 已被 shuffle 切断), 还有**残留的文本触发捷径**（shuffle 不切断措辞）。**单 seed smoke 结论被 4 seed 全量完全复现, 方差紧、统计极显著, 定稿。**

## 🔄 4-seed 全量启动 (2026-09-24 21:44)

PID 102979 (launch_p4_shortcut.py, spawn-detach). cmd: `--seeds 4 --eval_k 3 --workers 28`. log: `p4_shortcut/logs/p4_full4seed_20260924_214446.log`.
- driver `--seeds 4` = `range(1,5)` 跑 seed1-4; **跳过已存在 cache**, 故 smoke 的 12 个 `p4_*_seed1_k3.json` 复用, 仅 seed2/3/4 跑新 (~60 other × 4 cond × 3 skill × 3 seed × k=3 ≈ 6480 次 API, ~3.6h @ 28 workers)。
- 启动校验通过: test=106/meta=46/other=60; skill hash s0=68571cb2/on=fa63335e/on_noAsentence=ca9ac1d8 均匹配 smoke。
- 进度: seed1 全部 CACHED 加载后跑 seed2。待完成后更新本节 4-seed paired 结果 (mean±sd, paired t / Wilcoxon)。
