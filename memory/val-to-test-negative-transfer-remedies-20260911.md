---
name: val-to-test-negative-transfer-remedies-20260911
description: val→test 负迁移调研结论 — 三叠加病因(赢家诅咒/题面过拟合/train⊂val数据泄露·已确认) + 三味药(LCB/k-fold min/剔除train+难度分层)
metadata:
  type: project
---

# val→test 负迁移: 三病因 + 三味药 (2026-09-11)

用户确认 val→test 负迁移是本项目反复未越 northstar 的唯一系统性瓶颈，要求集中调研有效解法。本轮产出方法论结论（统计/因果/小样本模型选择视角）＋ 代码级确认了一个一直未被当 bug 的致命问题。

## 三叠加病因（统一了方向E/PathB/floor/warm 的所有失败）

| # | 病因 | 项目对应 | 严重度 |
|---|---|---|---|
| ① | **赢家诅咒/选择偏差** | 每轮从 4 候选选 val argmax，冠军点估计系统性高估真实 delta | 🔴 根本 |
| ② | **题面过拟合**（只在少数 easy 题刷高均值） | 方向E 毒段=只在一个 meta 子集赢；floor 套餐 | 🔴 根本 |
| ③ | **train⊂val 数据泄露** | **select=train+val=71，train 18 题从头到尾在评分/选择集里** | 🟠 更基础、已确认 |

### ③ 已确认（代码级坐实）
- `skill_evolver_math.py:57` 注释 `select=train+val=71`；`3324` 行 `选择集=train+val` 注释还写着"免训练优化无过拟合权重"（误判）。
- `3322-3345`：`val_items=load_split(...,"val")` + `train_items=load_split(...,"train")` → `select_items=train+val=71`，`probe_items`/`diag_items` 都从这 71 题切，`eval_on` 全在这 71 题上计分。
- **结论**：方向E、Path B、direction-B 每一个 champion 都是在含 train 18 题的 71 题上选出来的 = 在"看过答案的题"上计分，天然乐观偏差。这是比 winner's curse 更基础、且至今漏掉的 bug。

## 三味药（均可直接编码，非空谈）

### 药一：Beta-Binomial 收缩 + LCB（下分位）作选择准则（替换现 `ci>0` 均值门槛）
- 候选 pass 数 `k ~ Beta(0.5+k, 0.5+n-k)`（Jeffreys 先验），准则=`scipy.stats.beta.ppf(0.25,...)` 后验 25% 分位。
- 理论 = multi-armed bandit 的 LCB（Audibert 2009），样本越少越悲观收缩，直接抗赢家诅咒。比现有 CI 下界更正确。

### 药二：k-fold min-over-folds（惩罚题面过拟合 = 方向D 的精确统计依据）
- 纯 val（剔除 train 后 35 题）分 k=5 折，候选每折算 delta，准则 = **min over folds** 而非均值。
- 理论 = stability selection（Meinshausen & Bühlmann 2010）。方向E 毒段（只在 meta 子集强）会在"全折为正"这关被筛。

### 药三：剔除 train 重叠 + 难度分层监测（诊断，最先修）
- **先修②**：选择/评分集必须剔除那 18 个 train 题，否则每个冠军都在背题。
- 叠一层：baseline 每题 pass/fail 作难度锚，候选 delta 在 easy 为正、hard 为负 → 一票否决（过拟合信号）。

## 文献级补强（Prompt/Skill 进化领域的现成机制，第二条调研）

- **bootstrap-min / worst-case 评分在 prompt 优化领域几乎是空白**（DIVE/OPRO/EvoPrompt 只用均值/单次 batch，无人做有放回重采样取 2.5% 下界）→ 我们做药一+药二是**已知空白的高价值原创切入点**。
- **迁移比在线早停没人做**：监测 τ_train_val = val_pass/train_pass 单调下降即停，利用 train>val>test 单调漂移的"退化先兆"（train→val 先于 val→test 退化），也是原创点。对应药三的难度分层监测。
- **APE 用四分位均值(interquartile mean)** 抗离群；GrIPS phrase-level 编辑作表征正则；SkillComposer 短 skill 拆分 = 长度/复杂度正则的 skill 维度体现。
- **DSPy 早停+held-out 是防 dev 过拟合的教科书实现**，建议精读 BootstrapFewShot/MIPROv2 的早停逻辑。
- **NSGA-II 多门槛 Pareto**（目标 `[各子集pass门槛, -len]`）= 方向D + 长度罚的正式化（方向D就是这里"各子集设软门槛"的特殊情形）。

## 排期建议
1. 最优先 = 药三（最便宜，坐实一个真 bug，先修数据泄露 —— train 题别混进选择集）。
2. 药一（LCB）+ 药二（min-over-folds，方向D升级版）随后同上一轮，合并 = "bootstrap-min 评分"，是文献空白级原创点。
3. 迁移比 τ_train_val 早停 + NSGA-II 多目标作为进阶层。

## 关联
[[peak-diagnosis-val-to-test-negative-20260909]]（val→test 负相关实证，select 口径含 train 18 曾作旁证，本文件升级为可执行 bug 修复） / [[direction-e-pop-test-verdict-20260910]]（毒段=题面过拟合的极端例） / [[meta-rule-pathB-test-verdict-20260911]]（meta ceiling 封顶） / docs/phase2_issues.md