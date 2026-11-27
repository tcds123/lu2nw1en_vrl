# REVIEW REQUEST — 2026-11-25 (LiveMath 新机制移植 + 门控争议)

## 你是谁 / 你要做什么

你是独立技术评审。本包要你审的核心问题是：**在我们的 skill 自进化框架里，bootstrap CI 下界（CI_lower > 0）作为候选 skill 的"唯一晋升门"是否过严？是否应该让"多轮方向一致但单轮 CI 跨零"的真实信号也进入候选池？**

这不是"帮我们确认结论"。我们要你**反驳或修正**我们的当前判断。

---

## 背景：这个实验从哪来

> **先读** `docs/math_line_background_20261126.md` —— 数学线（AIME × LiveMathBench）完整背景与历史，建立全貌后再读本文件。

我们的框架：`evolver 产出 skill 文本 → skill 作 system prompt 喂推理模型(DeepSeek-V3.2 API) → 评测 pass rate + bootstrap CI → 反馈驱动下一轮进化`。

之前在 AIME 数据集上，我们移植了 6 个新机制（paired_rescore / rescue_on_reject / facet_group_admit / EFL / _veto_regression / F6 cache manifest），其中 **paired_rescore**（同批配对评测消噪，把丢题从 58-68% 降到 3%）在 AIME 上被验证有效。

本次：把同样这 6 个机制移植到 **LiveMathBench**（MCQ 数学选择题，177 题，train35/val18/test106），用 **V3.2 模型**重测，看新机制能否松动旧饱和结论。**只移植测量/门控机制，不碰 LiveMath 的 MCQ taxonomy**（即"LiveMath taxonomy + AIME 测量地基"这个唯一未测的叉乘）。

代码 delta 见 `code/`（4 个核心文件 + `_env.sh`），完整移植记录见 `memory/livemath-newmech-port-exec-20261124.md`。

---

## 实验事实（你要审的素材）

### Run: smoke r2 (PID 127644, 仍在跑 iter2)

- 配置: `iters=2, K=4, eval_k=8, train_n=18, V3.2, 5 flags ON (paired_rescore + rescue + veto + facet_group_admit + subset_admit)`
- S₀ 真实 pass = **0.238**（71 题 select 集, k=8, shuffle_seed=0），无丢题
- best_val_delta 在 paired_rescore 下重置为 0.0（零假设 cand≤parent）

### iter1 全部候选（archive_iter1.json，6 个，3 个 identity 对照 + 3 个真实 patch）

| # | facet | delta | CI_lower | 结果 |
|---|-------|-------|----------|------|
| 3 | Option Comparison | **+0.023** | -0.004 | KEEP-PARENT (CI 跨零拒) |
| 4 | Theorem-Level Precision | -0.011 | -0.041 | KEEP-PARENT |
| 5 | Hypotheses | -0.002 | -0.041 | KEEP-PARENT |

- operator_counts: `facet_edit attempted=3 evaluated=3 admitted=3 **promoted=0**`
- 即 3 个候选**全过准入门**（admitted=3），但 **0 晋升为 best**——全栽在 CI>0 这关
- overlay: delta=-0.012 CI=-0.039 admitted=N
- expand: delta=+0.014 CI=-0.014 **not_meta_ci=0.0**（正好踩线，>0 才救，0.0 不算）not_meta_n=10

### iter2（已完成）

| facet | iter1 delta | iter1 CI | iter2 delta | iter2 CI | 两轮方向 |
|-------|-------------|----------|-------------|----------|---------|
| Option Comparison | +0.023 | -0.004 | **+0.016** | -0.019 | **两轮都正** |
| Theorem-Level Precision | -0.011 | -0.041 | **+0.025** | -0.004 | 翻正 |
| Hypotheses | -0.002 | -0.041 | +0.000 | -0.028 | 近零 |

| 算子 | iter1 delta/ci | iter2 delta/ci |
|------|---------------|---------------|
| overlay | -0.012 / -0.039 | -0.002 / -0.030 |
| expand | +0.014 / -0.014 (not_meta_ci=0.0 踩线) | +0.002 / -0.026 |

**关键观察 1**：Theorem-Level Precision 这个 facet，iter1 delta=-0.011 → iter2 delta=+0.025，**方向都翻了**。这说明单轮 delta 的采样噪声极大。

**关键观察 2**：Option Comparison **两轮 delta 都是正的**（+0.023, +0.016），但两轮 CI 都跨零被拒。expand 算子两轮 delta 也都正（+0.014, +0.002）。如果存在"多轮方向一致"的门控，这些候选本可进入候选池待合并验证。当前框架没有这个机制——单轮 CI 否决把它们两轮都扔了。

### 终局（RF §6 告警 — 完整 run 结果）

```
✅ Facet-Tree done. init val delta=+0.000 → best val delta=+0.000 (CI=+0.000)
   skill: 1017 → 1017 chars | hash 68571cb2a6 → 68571cb2a6   (best_skill = S₀, 未变)

⚠️ RF §6 Step 3 — operator(s) active but 0 promoted all run:
   - facet_edit(attempted=6, promoted=0)
   - overlay(attempted=2, promoted=0)
   - expand(attempted=2, promoted=0)
```

**两轮合计**：10 个候选尝试（6 facet_edit + 2 overlay + 2 expand），**全部 0 promoted**，best_skill 保持 S₀ 不变。rescue_by_source 全程空（无可救子集）。

---

## 我们要你审的 4 个点

### 审点 A（核心）：CI_lower>0 作为唯一晋升门是否过严？

我们的当前判断（供你反驳）：**CI 作为最终防误晋升门是对的，但当前框架把它当"单点否决"偏严**。理由：
1. CI 本意是防噪声误晋升（防止把 +0.02 采样波动当真增益写进论文后 held-out 翻车），目的正确不该放弃。
2. 但 CI 在小样本（not_meta_n=10）上系统性偏保守：n=10 时要 CI>0 需 mean delta≈0.04+，真实 +0.02 效应几乎不可能 CI>0——这是统计功效问题，不是"无效应"。
3. 当前缺"轨迹信号层"：只看单轮 CI，不看"连续多轮 delta>0 且 CI 收窄"。

**请回答**：
- (A1) 这个判断成立吗？若不成立，是哪一步错了？
- (A2) 在 n≈10、σ_pair≈0.044 的 paired 评测下，CI>0 的真实检出下限是多少？+0.023 这种信号是不是注定 CI 跨零？
- (A3) 有没有比"单轮 CI>0 否决"更合适的门控？比如"多轮 delta 方向一致 + 合并样本 CI>0"、"Bayes 因子"、"序贯检验"、"effect size + CI 联合"？
- (A4) **是否改变论文结论**？如果放宽门会放进来多少假阳性？

### 审点 B：候选生成器是否也有问题？

facet_edit 只改一个 slot，是局部微调，先天改动力度小 → delta 天花板低。这是不是"全被拒"的另一层根因？

- (B1) 单 slot 编辑的有效改动力度，在 MCQ 任务上合理吗？
- (B2) expand 算子的 not_meta_ci=0.0（踩线）是不是暗示"新 slot 段有信号但被样本量压在零界"？

### 审点 C：移植正确性

代码在 `code/`。paired_rescore 把 `eval_paired` 的 5-tuple 返回接到了 facet_edit/overlay/expand 三个晋升点。
- (C1) 5-tuple 解包在 MCQ 后端（`math_evaluate` 用 `correct_choice`/`choices`）是否正确？
- (C2) `best_val_delta` 重置为 0 的 paired 零假设语义是否实现正确？

### 审点 D：实验设计是否足够回答"能否提升"

- (D1) smoke 只 2 轮 + train_n=18，够不够下"能否松动旧饱和"的结论？
- (D2) S₀ pass=0.238 偏低（历史 LiveMath test106 上 S₀~0.34-0.41），这是 select 集(train+val=71)还是 shuffle 打乱选项导致的？是否影响结论？

---

## 交付形式

- 一份 `.md` 评审报告放 `for_review/incoming/20261125_codex/REVIEW.md`（若改代码一并放进同目录）。
- 每条意见必须标：**证据等级**（【事实】/【推断】/【假说】）、**是否改变论文结论**、**建议的最小验证实验**（题量/时长/配置）。
- **不要重复** `docs/phase2_issues_202611.md` 里已知的问题。
- `archive/` 里的历史包只在需要追溯时查，不必逐份评审。
- 不要外传包内任何数据/API key。

## 包结构

```
for_review/
├── README.md                          ← 本包总入口（先读这个）
├── review_requests/20261125_codex/    ← 你正在读的这份
├── code/                              ← 最新源码（4 核心 .py + _env.sh + smoke 脚本）
├── outputs/livemath_smoke_r2/         ← smoke 实验产物（archive_iter1.json + evolver_log.json + s0_cache.json + log）
├── memory/                            ← 移植记录 + 修正后的重评估
├── docs/                              ← 设计文档 + 门控逻辑说明
└── archive/2026-11-25_livemath_newmech_pre/  ← 旧 11-13 版主体（归档）
```
