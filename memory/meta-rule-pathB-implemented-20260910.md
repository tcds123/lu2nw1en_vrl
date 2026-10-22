---
name: meta-rule-pathB-implemented-20260910
description: 路径B诊断对照对称化实现完成 — collect_failures 新增 meta_passes 通道 + make_meta_contrast_diagnostics 渲染 + --meta_contrast flag, 让 patcher 从"同构 meta 题对/错对照"自主归纳 meta⇒答案规则, 不注入结论
metadata:
  type: project
---

# 路径B 诊断对照对称化实现完成 (2026-09-10)

## 一句话

[[livemath-meta-option-trap-20260910]] 坐实「meta-option 出现⇒正确答案=meta 选项 100%」，但用户硬性要求：**这条规则不能人手写，须由算法/模型自己发现**。路径B（用户选定）= 只改"喂给 patcher 看哪些数据 + 怎么配对"，把每道 meta 错题配对一道同构 meta 对题并列展示，让「同是 meta 题、做对的选了 meta 选项」这个对称关系自己浮现，模型自行归纳规则。实现已完成，**未烧 API**，未跑 evolver。

## 实现内容（`rl/train/verl1/skill_evolver_math.py`）

1. **`collect_failures` 新增 meta_passes 通道**（`:445`）：签名加 `meta_contrast=False`。在 `else:`（pass_rate>0）分支里，`meta_contrast and classify_item_meta(it)` 时收集 skill 解对的 meta 题（含 id/item/pred/correct/response_excerpt/s0_pass/skill_pass），**不受 `s0_pass<1.0` 限制**（这是三重掩盖第③层 `learn_positive` wins 过滤掉的 meta 对样本）。零额外 API（复用同一 eval pass 的 `run_one` 结果）。
   - 返回协议：默认 3-tuple 不变；`return_all_failures=True` 时 4-tuple 不变（保所有旧调用点字节复现）；**仅 `return_all_failures=True and meta_contrast=True` 才返回 5-tuple** `(failures, n_fail, wins, all_failures, meta_passes)`。
   - `meta_passes.sort(key=id)` 确定性排序，供最近邻配对。

2. **`make_meta_contrast_diagnostics(meta_failures, meta_passes)`**（`make_wins` 之后）：为每道 meta 错题配对一道 meta 对题（最近邻按 id 序），并列渲染 `[MISSED]`（pred/correct）和 `[SOLVED]`（pred/correct）两个原始样本（Theorem/Question/Choices 全量、`_format_choices` 不截断），**通篇无一句结论文字**——不写"meta⇒答案"。

3. **`META_CONTRAST_SECTION` 模板 + `{meta_contrast_section}` 占位符**：挂在 `{positive_section}` 之后（`PATCH_TEMPLATE` 和 `PATCH_TEMPLATE_COVERAGE` 各加一个占位）。文案只命令"derive any GENERALIZABLE procedure yourself"，不预结结论。

4. **`generate_candidates` 加 `meta_contrast=None` 参数**：默认 None（旧路径字节复现）；非 None 时 `meta_contrast=(meta_failures, meta_passes)` 渲染成 `meta_sec` 注入两模板。

5. **主循环调用点**（`3393-3424` + `3485` 附近）：`--meta_contrast` 时令 `collect_failures(..., return_all_failures=True, meta_contrast=True)` 拿 5-tuple → `meta_passes` 单独存变量 → `failure_cache` 存 5-tuple（缓存分支按 `len(cached)>N` 判定），`generate_candidates` 传 `meta_contrast=(failures, meta_passes)`。

6. **`--meta_contrast` argparse flag**（默认 off = 字节复现）+ `_flags` 标记 "路径B:meta_contrast"。

## 边界（诚实标注，呼应设计文档 §3.3）

- **不是变相注入**：结论留空，只摆原始证据（对/错两个原始样本）。
- **仍有人为塑造嫌疑的残留**：配对是确定性最近邻，但"刻意把 meta 题 + 做对摆在一起"本身是人选的呈现结构。缓解=不作任何"保证模型必然看出"的承诺，归纳功劳留给模型侧（体现在 test 评测自主提升上），验证判据见设计文档 §5。

## 验证判据（跑 evolver 后，守 CI 纪律）

1. 机制：产物 skill 是否**自主出现**"识别 meta ⇒ 选 meta"决策段，且这句**不来自**我们任何 prompt（diff 我们未提供该文本）。
2. 效果：val meta 子集 + test（meta 43.4%）pass 提升，且 4-seed CI 不回归 not-meta。
3. 红线：不得在 test 上显著掉（重蹈 direction-E val→test 反向崩）。

## 状态

- 代码已实现 + `ast.parse` 语法校验通过。
- **已跑低成本 smoke（2026-09-10）验证机制**，见下。

## Smoke 验证结果（`run_20260910_174403`，`--iters 1 --eval_k 1 --K 2 --train_n 5 --meta_contrast`，S₀=initial.md）

**机制端到端跑通，verdict=有效，但未产出 patch（判据1 待真跑）：**

1. 日志确认 `meta_passes: 6 (对照)` 与 `meta_contrast=True` 贯穿。
2. `evolver_log.json` 的 `failure_cache` 存的是 **5-tuple**：`[0]` failures(len=8)、`[1]` n_fail=45、`[2]` wins(len=4)、`[3]` all_failures(len=45，未截断)、`[4]` **meta_passes(len=6)**。
3. 配对样本坐实对称性：`[0][0]` = id 202512:50 pred=C correct=A（meta 错题）；`[4][0]` = id 202512:16 pred=A correct=A（meta 对题，同构对照）。同是 meta 题，做对的选了 meta 选项 A，做错的选了实质选项 C —— 对称关系可被 patcher 观察。

**暂未出 patch 的原因（纯 smoke 配置所致，非机制问题）：** 仅 1 iter + focus=「Implication vs biconditional」，2 候选均 CI<0（cand0 Δ+0.055 CI=-0.017、cand1 Δ-0.066），`KEEP PREV` 守门正确拒收 → best_skill 仍是 1017 字 init。

**下一步（要判定判据1/2/3，须真跑）：** 用默认 `--iters 10 --eval_k 5 --train_n 35`（或至少 iters≥3）跑一个非 smoke 的 `--meta_contrast` run，观察产物 skill 是否自主长出 meta 决策段（非我们 prompt 文本），再 4-seed test 复核效果 + not-meta 不回归（红线）。

## 🎯 真跑判据① 已达成：模型自主归纳出 meta 规则（2026-09-10）

`run_20260910_175957`（`--iters 10 --eval_k 5 --K 4 --train_n 35 --meta_contrast`，S₀=initial.md，运行中）

**判据①（机制）达成 — 产物 skill 自主长出 meta 决策段，且非我们 prompt 文本：**

iter1 接受 cand0（`Δ+0.223 CI+0.152 pass=0.454`，hash 68571cb2a6→e385c57b7a），写出的 `skill_iter1.md`（1882 字）Option Comparison 节新增一句（第 6 行）：

> "An option stating **'One of the remaining options is correct, but a stronger result can be proven' is often correct** when other options incorrectly assert a two-way equivalence or a universal property that the theorem only establishes in one direction or under stricter conditions."

**三点坐实「非人手写、模型自己得到」：**
1. 引用了 meta 签名**逐字**（`"One of the remaining ... stronger result can be proven"`），这条原文只出现在我们给的**原始对/错对照样本**里，不在任何 prompt 模板的结论位。
2. `is often correct` 是模型自行归纳的「meta 选项 ⇒ 答案」概率规则，我们的 `META_CONTRAST_SECTION`/`PATCH_TEMPLATE` 全文只写「derive any GENERALIZABLE procedure yourself」，无此结论（已 grep 核验：`is often correct`/`must be selected`/`always the answer` 均 0 命中于代码）。
3. 模型还自己把 meta 规则**接到了既有助记框架**上：把「meta 正确的时机」与「其他选项错误地断言双向等价/过分强」绑定 —— 复用了同 skill 的 overstrong 判定逻辑，说明是归纳不是抄录。

**诚实边界（保留）：**
- 结论写成 `often correct`（概率化）而非我们统计真相 `100%`（train 12/12 / val 27/27 / test 46/46）。模型从有限对照里归纳的是弱化版。这是模型侧合理的 epistemics，但真实泄露度更高（见 [[livemath-meta-option-trap-20260910]]）。利好诚实性与 generalizability，代价是可能没吃满 test 的 46/46。
- 另外「配对呈现代本身」仍有人为塑造嫌疑（见设计文档 §3.3），但归纳文本确实非我们提供。

**判据②③（效果 + 红线）待跑完 4-seed test 复核**：run 还只到 iter2（在线）。等 10 iter 收敛后，用 `eval_test_multiseed.py` 跑 s0 vs evolver 的 4-seed test=106 对比，验证 meta 子集提升 + not-meta 不回归 + 不得掉过 direction-B(0.5393)/skillopt(0.5605) 红线。

## 关联

[[livemath-meta-option-trap-20260910]]（根因实证）/ [[direction-e-pop-test-verdict-20260910]]（被纠正的证伪）/ docs/discover-meta-rule-pathB-design.md（设计文档）