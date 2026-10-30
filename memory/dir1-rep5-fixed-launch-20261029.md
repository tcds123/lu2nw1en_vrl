# 1′-fixed rep5 重跑 — Launch 记录 (2026-10-29)

**状态: ✅ 已完成 (2026-10-30 04:28 正常退出, ~9h27m)。结果见 [[dir1-rep5-fixed-result-20261029]]**。
**判据执行前的两处口径偏离 (结果文件详述)**: (a) 判据③的 identity 空白**本 run 未产出** (overlay 5/5 被 `OVERLAY-SKIP` 跳过, 见下文说明) → 改用 placebo 交叉配对实验补齐同一件事; (b) 判据①提到的 `eval_audit.jsonl` 只记 main 侧 `eval_on` 的 8 条, **eval_paired 内的 dropped 不计入审计文件** (只打日志), 属实现缺口, 已记 issue #17。
启动脚本: `.../run_dir1_step1_onflags/rep5_fixed_launch.sh`。

目的: 在 1′ (abandoned/全异常剔除) 已修的前提下重跑 rep5, 验证偏置修复是否改变
"iter1 污染 bullet 锁死 parent" 这个头号失败, 并同时用 identity 空白标定当前时段的真实噪声。

## 与 rep5 的唯一差异

| 项 | rep5 (2026-10-23) | 本次 |
|---|---|---|
| 1′ (abandoned 剔除 + eval_audit) | ❌ 未修, abandoned 计 0 | ✅ 已修 |
| iters | 5 | 5 |
| flags | `--learn_positive --meta_contrast` | 同 |
| K / eval_k | 4 / 8 | 同 |
| s0_cache | `run4_measure_foundation/run_20261016_140819/s0_cache.json` | 同 (可重复用) |
| QWEN_CHAT_MAX_TOKENS | 16384 | 同 |
| held-out gate | 关 (post-hoc 测) | **关** (与 rep5 可比; 空白另行测) |
| output | `run_dir1_step1_onflags/rep5_20261023` | 新建 `rep5_fixed_20261029` |

## 三条预期判据 (写死, 跑完不得改)

### 判据 ① iter1 污染是否被堵 (首要目的)

rep5 iter1 唯一 PROMOTE = `12064805af`, 靠 base-low (13/30 abandoned, 计 0) 把 paired delta
抬到 +0.1042。1′ 之后这 13 题**整对剔除**, 分母 30→17, delta 应大幅收缩。

- **通过**: iter1 该候选 delta < +0.05 (检出下限) 或根本不被 promote; 且 `eval_audit.jsonl` 中
  iter1 的 abandoned 题笔记数 ≈ 13 (审计可见, 证实剔除确实发生)。
- **口径澄清 (2026-10-29 核对代码后修正)**: 1′ 的"整对剔除"只在 **`eval_paired`** (paired_rescore
  路径) 生效。**`eval_on`** (probe/val, `tf_global_v2_aime.py:326-355`) 仍是"剔题但不配对 —
  只剩单臂, 分母按剩余题重算" (`results` filter, 无 base 对照)。因此 13 题在 `eval_on` 侧是
  **从 delta 中删掉**, 不是"双方都不计 0"。对判定影响小 (都能堵住计 0 的偏置), 但等变差
  **不能**直接按"分母 30→17"预言; 实测 delta 并不可由剔除数精确推出。这一条不改判据阈值,
  只纠正解释口径。
- **不通过**: 仍 promote 且 delta > +0.10 → 说明偏置还有第二个来源 (1′ 没堵住)。
- **中性**: iter1 无候选过门 (zero promotion) — 也算通过, 记作"偏置产物已消失"。

### 判据 ② 干净信号能否过 +0.10 检出下限

rep5 iter3 的 CV delta 是 **+0.042** (0 abandoned, 干净正), 但被 NO-CI 拒 (CI −0.017)。
在 #15 修正确认真实 SE≈0.06 / 检出下限 ~+0.10 之后:

- **通过**: 存在 iter≥2 的干净候选, in-sample delta ≥ +0.10 **且** CI > 0。
- **不通过**: 最高干净 delta < +0.10 → "干净信号太弱" 从 rep5 的观察升级为**已定量的结论**,
  Direction 1 实验侧结案, 转 generation 层。
- 注意: rep5 的 +0.042 在 1a 旧尺子下是 "接近 +0.05", 在新尺子下是 "不到一半"。**这不改变结论方向,
  只改变措辞强度** — 不得据此宣称重跑"变差"。

### 判据 ③ identity 空白对照 (本次新增, 为此才需要双尺子)

rep5 的 overlay 算子 5/5 次产出 identity 候选 (`c11d7dc0c8`, cand==parent), 真效应精确 0。
它们天然是**同时段的测量空白**: 那一轮的实际噪声水平由它们决定, 不用拿别处的点估计当尺子。

rep5 实测 identity in-sample delta: **−0.038 / +0.038 / +0.083 / −0.058 / −0.025**
(sd ≈ 0.058, 与 #15 的 0.0587 吻合)。

- **通过 (空白干净)**: 本次 identity delta 的 |mean| < +0.05 且 sd ≤ 0.07 → 空白与 #15 一致,
  上两条判据按原数判。
- **不通过 (空白脏)**: identity 的 |mean| ≥ +0.05 → **本次所有非 identity 候选的 delta 必须减去
  identity 的 mean 再判** (因为空白本身就带那个偏移)。这**不是事后调判据**, 是预登记的前置规则。

## 双尺子报数 (你选定的口径)

结果同时按两套阈值报, 结论以**主判据 ~+0.10 + identity 空白**为准, +0.05 仅作对照:

| 尺子 | 来源 | 检出下限 | 用途 |
|---|---|---|---|
| 旧 (1a) | `sd_item/√30`, 题真值固定假设 | +0.05 | 与历史 rep5 / run6d / D1-step1 直接可比 |
| **新 (#15)** | identity 5 次重测的经验展幅, **主判据** | **~+0.10** | 真实跨 run 噪声 |
| identity 空白 | 本次 run 的 overlay identity 候选 | 当次 mean / sd | 校正本次的时变偏置 |

## 成本

rep5 实测: evolver 约 ~2 h (8 GPU 不占, 纯 API), held-out eval 另 ~2 h。
本次 evolver 部分同量级; identity 空白**零额外 API** (overlay 算子本来每轮就跑, 只是现在要单独读出来)。

## 不在这份记录里解决

- 本 run 的 held-out 侧 (AIME25) post-hoc 测: **留到 evolver 跑完再决定是否需要**。
  若判据 ② 通过 → 值得测; 若 ② 不通过 → 直接结案, 不必再烧 2 h held-out。
- 不跑 `--staleness` (A-Step2 实验), 不在本 run 叠加任何新变量。

## 关联

- [[dir1-step1-rep5-onflag-launch-20261023]] — rep5 原记录 (基线)
- [[measurement-calibration-1a-20261028]] — 1a 原口径
- `docs/phase2_issues.md` #13 (1′ 修复) / #15 (SE 低估)
- [[dir1-executable-checklist-20261029]] — D1 三处改造依赖本 run
