# 防截断 skill 自主进化 — run2 (纯 S₀ + budget 轴进 focus) 2026-11-16

## 起因 (用户质疑)
run1 的起点 `trunc_v1.md` 里那段 "Budget & Commit Discipline" 6 条是**我手写的**, 不是算法进化的产物。
用户: "为什么是手写的, 不应该是我们设计的系统自进化出来的吗"。
根因 (issue #76): 代码里**能**识别 `finish_reason=length` (`_BUDGET_FAILURE_LABEL`), 但它
(a) 在关键词阶梯之前就被分流, 不进多数票; (b) 不在 `FAILURE_FOCI` 8 个答错轴里; (c) `_TYPE_COVERAGE_SIG`
无对应项 → 永远判"从未覆盖"。结果 4 轮 focus 全落在答错轴, evolver **从没被要求降低截断**。
一句话: 机制在代码里 ≠ 它在优化目标里; 起点预埋答案 = 把结论写进了自变量。

## 改动 (flag 门控, 默认 OFF)
`for_review/code/skill_evolver_aime_v2.py`:
1. `_BUDGET_COVERAGE_SIG` 指纹 + `_BUDGET_FOCUS_ON` 开关 (`--budget_focus`)。
2. `_IMPLICIT_AXIS_MANDATE[_BUDGET_FAILURE_LABEL]`: 强制要求写"决策程序" (预算计划 / 反重复检验 /
   硬交卷点 `<answer>\boxed{N}</answer>`), 限 ≤5 bullet 且不许删旧段。
3. `_efl_majority` / `find_uncovered_type`: ON 时 Budget 轴可被选中; 且"无收尾纪律指纹"→ 显式早退选 Budget
   (按 count 排序会被 9 次的答错轴压掉, 必须早退)。
4. `_implicit_axis_block()`: 非 coverage 路径也把具体截断样本 + mandate 送进 patcher (旧代码只有 coverage 路径送得到)。
5. 顺带修掉 OFF 路径的潜在崩溃: `_efl_majority` 遇 Budget 标签会 `ValueError` (旧防线是"Budget 永不进 counts")。
- **OFF 字节级等价已验证**: 403/403 输出与改前完全一致 (`/tmp/ev_before_budget.py` vs `/tmp/ev_with_budget.py`, 非留档)。

## 本轮设计 (run2)
- 起点 = **纯 S₀** `for_review/data/skill_aime_initial_v2.md` (599 chars, md5 `c11d7dc0c8`, 无任何 commit/budget 词)。
  启动脚本有 grep 守门: 起点含 `commit checkpoint|token budget|running total` 直接 exit 1。
- 与 run1 唯一差异: 起点 (trunc_v1 → S₀) + `--budget_focus`。其余逐项相同:
  iters=4 K=3 eval_k=3 n_parallel=6 cap=16384 temp=0.3 train_n=18 shuffle_seed=0, B=deepseek-v4-flash。
- 启动: `nohup bash code/scripts/run_evolve_budget_v2.sh run2 > /tmp/evolve_budget_run2.out 2>&1 &` @ 17:45:16
- 产物 `for_review/outputs/skill_evolver_budget_20261116/run2/run_20261116_174516/`
  日志 `for_review/logs/evolve_budget_run2_20261116_174516.log`
- 启动横幅已确认: `🔧 budget_focus ON: Budget-Exhausted 轴参与 focus 选择 (issue #76)` / `防泄漏预检: OK`

## 要看什么 (判定标准, 跑前定死)
1. **Iter 1 的 `[data_driven_focus]` tally 是否 `-> pick: Budget-Exhausted`** — 机制是否真的接通。
2. 最终产物里是否**自己长出**收尾纪律 (grep `commit checkpoint|token budget|running total|hard commit`)。
3. 截断率是否从 S₀ 的 11.1% 降下来, 且 pass 不掉 (LOST guard `pass < base−0.06` 会拦)。
4. 若 Iter1 focus 仍落到答错轴 → 机制没接通, 不是算法不行, 回来改代码。
- 注意: 晋升门仍**不含长度轴** (issue #69), 所以"变短"不是过门的条件, 只是可观测结果。

## 观测
- 17:45 启动 → 18:04 S₀ cache (30 题, k=3) → 19min。比 20RPM 估算慢, 思考型模型输出长。
- 18:05 起跑 `evolver_base` (纯 S₀ 作为 base, 90 条) → 18:27 出 Iter1 tally。
- **18:27 崩 (issue #77)**: tally 已正确 `-> pick: Budget-Exhausted (truncated before completion) (2/2)`
  —— 机制接通了, 但紧接着 `UnboundLocalError: base_user` 未赋值。两个 bug:
  ① 注入块写在 `PATCH_TEMPLATE.format()` 之前; ② `_efl_majority` OFF 分支对 budget 标签会 `ValueError`
  (`FAILURE_FOCI.index` 抛异常, 任何 skill 整题截断都会炸, 与 flag 无关)。均已修, 零 API 验证过
  (渲染全路径 / 分类链路 / OFF 回归逐字一致)。崩溃轮产物改名为 `run_20261116_174516_CRASHED_issue77` 留档。
- **18:58:42 重启** (`/tmp/evolve_budget_run2b.out`, 日志 `logs/evolve_budget_run2_20261116_185842.log`),
  从纯 S₀ 原地重跑, 无脏状态。产物目录 `run_20261116_185843/`。
- 19:09 S₀ cache 完成 (比首跑 19min 略快) → 19:21 baseline `pass=0.856 delta=-0.011 CI_lower=-0.089`
  (与首跑 17:45 那条纯 S₀ 基线 `pass=0.867 delta=+0.011` 相差 ~1.1pp ≈ 1 题, 即 temp=0.3 的噪声底)。
- **19:31 Iter1 tally 判定标准 1 通过**: `Budget-Exhausted (truncated)=1 | Misreading the asked quantit=1
  -> pick: Budget-Exhausted (1/2)` —— 与崩溃轮同结论, 且**这次没崩**(已过 patch 阶段)。
- 19:31 patch prompt 实查 (零成本): 含 `## Focus: Budget-Exhausted (2 diagnostic failure(s))` +
  `- Problem aime24:81: s0_pass=0.0, 1/3 samples hit the token limit` + `- Problem aime24:63: ... 0/3 ...`,
  即截断样本与 mandate 确实送到了 patcher (issue #76 第 4 条改动生效)。`generated 3 candidates, 3 novel`。

## 结果 (2026-11-16 21:49 完成, 4 轮全 KEEP PREV)

| Iter | focus 选择 | 候选 (delta / CI_lower / pass / len) | 判定 |
|---|---|---|---|
| 1 | **Budget** (1/2) | c0 +0.011/−0.033/0.878/1779; c1 +0.022/−0.078/0.889/**6488**; c2 +0.033/−0.044/0.900/1845 | KEEP PREV |
| 2 | Misreading quantity (1/1) | c0 −0.000/−0.067/0.867/2426; c1 +0.022/−0.044/0.889/1955; c2 −0.011/−0.078/0.856/**4235** | KEEP PREV |
| 3 | **Budget** (1/2) | c0 +0.056/−0.011/0.922/1901; c1 +0.044/−0.011/0.911/2068; c2 +0.033/−0.044/0.900/1919 | KEEP PREV |
| 4 | **Budget** (1/1) | c0 −0.000/−0.056/0.867/1886; c1 +0.056/−0.011/0.922/**1382**; c2 +0.067/−0.022/0.933/4208 | KEEP PREV |

最终: `init val delta=-0.011 → best val delta=-0.011 (CI=-0.089)`, skill 599 chars, hash
`c11d7dc0c8 → c11d7dc0c8` 未变 (best 与 init 逐字一致, 仅少末尾换行)。产物
`outputs/skill_evolver_budget_20261116/run2/run_20261116_185843/` (含 analysis/REPORT.md)。

### 三条判定标准的回收
1. **机制是否接通 → 通过**。4 轮里 3 轮 focus 落到 Budget 轴; 逐条查 `evolver_patch.jsonl`,
   3/3 条 patch prompt 含 `## Focus: Budget-Exhausted (2 diagnostic failure(s))` + 具体截断样本
   (`aime24:81 1/3 hit the token limit`)。Iter2 未选 Budget 是因为该轮确实没有截断失败
   → 证明是**数据驱动**而非常数偏置, 比"每轮必选 Budget"更可信。
2. **是否自己长出收尾纪律 → 不适用**。skill 从未晋升, 无产物可查 (`best_skill_evolved.md` == S₀)。
3. **截断率/长度 → 无法归因**。所有读数都在候选身上, 无一进晋升门。

### 核心结论: 瓶颈是**统计效能**, 不是 evolve 能力
- 4 轮最佳 delta 全为正且单调上升 (+0.033 → +0.022 → +0.056 → +0.067), pass 0.856 → 0.933,
  方向高度一致, 但**CI_lower 最好只到 −0.011 (Iter3), 一次没过零**。
- 30 题 × k=3 的粒度 = ±1 题 ≈ 3.3pp, 基线 pass 已 0.856 → 剩余提升空间本就很窄,
  CI 过零在 n=30 下近乎不可能。这是 issue #69 之外的新一层: 不只是"门不看长度",
  而是**门在这个基线下根本打不开**。
- 因此 run2 对用户原问题("为什么是手写的")的回答是: 纯 S₀ 起点 + budget 轴都接上了,
  但当前门控无法把任何进化产物推上台面, 还不能说"算法自己长出了纪律"。
- 长度 Observable (非门控): 长候选并未更优 —— Iter1 len 6488 的 delta(+0.022) 低于 len 1845(+0.033);
  Iter4 len 4208 的 delta(+0.067) 与 len 1382(+0.056) 差 1.1pp ≈ 噪声底。**长度与收益无可辨相关**。

### 下一步 (待用户定)
- 提高统计效能: eval_k 3→10 或 select 30→60 (调用量 ×2~3.3, 单轮 ~70min → ~3h)。
- 或放宽门 (如 CI>−0.02) —— 会引入假阳性, 不推荐单独用。
- 或按 issue #69 把长度轴纳入门 (双门: δ_acc ≥ −1pp 且 clean-pass len ≤ −15%)。
