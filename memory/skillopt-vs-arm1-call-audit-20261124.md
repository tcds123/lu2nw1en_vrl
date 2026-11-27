# SkillOpt vs Arm1 调用预算对账 — 2026-11-24 (0 API, 只读日志)

用户任务: Issue #117 之后, 先回答 **Arm1 (+74) 是否只是"调用更多/预算更大"**。只读日志, 不补跑、不调 API、不改代码。
这是进入跨模型独立复现之前必须完成的审计 (顺序: 数据审计 #13 → 预算审计 #15 → 跨模型复现 #14)。

## 口径 (用户指定的六阶段, 避免重复计)

1. **① 优化器调用** / **② 候选生成 (Rollout)** / **③ Selection Eval** 三者构成**训练预算**。
2. **⑥ 正式评测不计入训练预算** — 2765 题正式评测与 SkillOpt 的 300 题 test chunk 均单列; 2765 两臂共享。
3. **④ Smoke Test 单列** (本次两臂主 run 均 0; 独立 `smoke_48_20261117` = 9 次不算)。
4. **⑤ 重试单列** — 语义是"**0 可观测**", 不是"确实为 0" (见下"重试盲区")。
5. 请求数 = HTTP 请求; SkillOpt "Rollout 320" 是**有效样本数**, 实际 **1120 请求** (40/(4+2)/步 × 8)。
6. 被拒候选照样计入; 优化器与目标调用分列 (无一次调用同时生成+打分)。

## 实测数 (逐项可追溯到日志, 见台账 json)

| 阶段 | SkillOpt | Arm1 |
|---|---|---|
| ① 优化器 | 85 (analyst 47 / merge 24 / ranking 8 / guidance 3 / meta 记忆 3) | 411 (rewriter 324 / distill 81 / merge+compose 6) |
| ② 候选生成 | 1240 (1120 rollout + 120 slow 对比) | 429 (324 B-eval + 105 val 门控) |
| ③ Selection Eval | 1200 (800 步内 + 100 S₀ 基线 + 300 slow) | 105 (展示上已含在 ② 的 429 内) |
| **训练合计** | **2525** (带 token 记录下界 1622) | **1050** (台账口径; 代码推导 840, 差 210 **未闭合**) |
| ④ Smoke | 0 | 0 |
| ⑤ 重试 | 0 可观测 | 0 可观测 |
| ⑥ 正式评测 (不计训练) | 600 (300 ⊆ 2765) | 2765 + 215 + 85 |
| 训练 token | 4.96M 已记 + ~1.24M 未记 ≈ 6.2M | 未记录 (估 ~1.3M) |
| 净增益 (2765, 配对) | **−47** | **+74** |

方法学花费 (训练 + 非共享评测): SkillOpt **3125** vs Arm1 **1350** (0.43×); 含共享 2765 总计 5890 vs 4115。

## 两个计数差额 (**Arm1 的 210 已撤回定案, 2026-11-24 二次核对**)

**Arm1 的 210 (411+429=840 vs 1050)**: ~~差 210 = 2 × 105, 被拒轮门控仍付费~~ — **该定案已撤回**。
代码推导的每轮训练 = 279 + G 请求 (rewriter 108 + distill 27 + merge 1 + compose 1 + B-eval 108 + 门控 G)。
门控 G 已核实为 **35** (2026-11-24 代码复核: L1118 读 args 缓存 → L1131 D5 重写 → L1216 用它评测;
原始缓存的 42 中 7 道 fn-call 被 D5 判为通过) → 每轮 280, 3 轮 **840**, 与台账差 **210**, 状态 **未闭合**。
历史 run 无 TRACE_DIR → 差额不推算、不填补。
另发现历史 run 的 `val_new_pass`/`val_new_delta` 与 35 道门控题的逐题软分对位矛盾 (10 道 partial pass),
该逐题口径同样**未闭合**; 不影响请求数。详见 [[issue14-crossmodel-budget-v2-20261124]] §3。

**SkillOpt 的 74 → 85 (+11)**: guidance 3 + meta 记忆 3 只解释 6 次; 剩余 5 次**无法从日志重构**。
`step_record.tokens` 里的 47+24+8 = 79 是带 token 记录的子集, 74 与之都不自洽。
**以 85 为准** (5 次落在未记 token 的写入调用上, 具体是哪 5 次日志不提供), 不再追。

## 重试盲区 (论文措辞必须遵守)

`code/skillopt/model/azure_openai.py::_chat_impl` (L362–436) 对任意异常 `sleep(min(2**attempt,30))`
后整条重发, `retries=5`, **不打印日志**; Arm1 的 `llm()` 与 SkillOpt 每步都走该路径。
→ 日志无 retry 行只能写 "**未观测到重发**", 不能写 "重试为 0"。
Arm1 `eval_on_routed` 另有 future-exception 兜底: 5 次全失败该题不再产生请求 (只会减少, 不会增加)。

## 另一处代码更正: compose 是 LLM 调用

`compose_global_skill` L575–589 实际调 `llm(..., stage="tf_compose")`, **不是本地拼接** (旧稿写错)。
故 Arm1 每轮 = rewriter 108 + distill 27 + **merge 1 + compose 1** + B-eval 108 + val 门控 G = 279 + G; G=35 (已验证) → 280。
(初稿的 "350" 把 3 轮门控 105 塞进了单轮, 已更正。)

## 失败池 × 训练请求 (为 #14 核实)

- 代码 L1157–1161: `fail_items_all` = train 中 hard==0 的题 (hard 优先排序), `train_items = fail_items_all[:train_n]`, train_n 默认 80。
- flash run.log L8: train 80 / **S₀-fail 27, 取前 27** / val cache 100 / test cache 300。
- **关系**: 每轮训练 = 243 + 2 + G; 其中 G 只取决于 **val 缓存的 S₀ 失败题数, 与训练池无关**。
  池 ≤ train_n 时只有 B-eval/rewriter/distill 随池线性增长 (g+1+... 共 9 次/题)。
  池 27 → 840; 池 40 → 1,191; 池 80 → 2,271 (G=35, 3 轮)。门控不随池变 (初稿"门控随池"已更正)。
  每模型合计 (含 train S₀ 80 + val S₀ 100 + 独立 2765): 3,785 / 4,136 / 5,216。

## 三个结论

1. **调用预算差异**: Arm1 训练 1050 vs SkillOpt 2525 (0.42×; 用带 token 记录的下界也是 0.65×), token 约 1/4.5。
   → "Arm1 赢在预算更多" **已排除, 方向相反**。
2. **优化效率差异**: Arm1 39% 训练调用在优化器侧 (rewriter+distill), SkillOpt 仅 3.4%, 且 52% 调用打在与其最终 2765 重叠的 100/300 题 chunk 上 → 目标错位/负迁移。
3. **归因边界**: 已排除 ①调用更多 ②测试泄漏 ③重试放大; 部分排除 ④样本选择 (同 split, 但 Arm1 只训 27 道 S₀-fail);
   **无法排除** ⑤优化目标/门控凭据 (最主要) ⑥候选生成方式 ⑦缺 Arm0 同预算对照 ⑧SkillOpt 仅 8 步。
   → 只能写"该完整方法配置在本文评测下更优", **不能**写"优势来自 Evolver 机制"。

## 论文措辞

- ✅ "在相同模型与解码配置下, 我们的框架以更少的训练调用 (1050 vs 2525) 与约 1/4 的 token 取得 +74 配对净增益, SkillOpt 为 −47"。
- ❌ 单调链 "Arm1 > SkillOpt > S₀"; ❌ 因果归因 Evolver 机制; ❌ "两臂均 0 重试"。

## 未解决 / 需授权

- Arm1 不记录 token → token 对比只能以**估算**呈现; 要硬证据需加 usage 计数后重跑 (未授权)。
- 补 ⑦ 的优先级是 **Arm0**, 而非把 SkillOpt 跑满 16 步 (只补 ⑧)。均未授权。
- 新 run 必须在 `stage` 级计 retry, 否则费用/请求数不可核。

## 产物 (outputs/ gitignore, 未入库)

- `rl/train/verl1/outputs/crossmodel_baseline/phase4a-audit-20261120/skillopt_vs_arm1_call_audit.md`
- 同目录 `skillopt_vs_arm1_call_ledger.20261124.json` (headline/grand_check/handover_reconciliation/fail_pool)
- 同目录 `issue14-crossmodel-repro-budget-v1.md` (#14 设计与条件化预算)

## 教训

- 旧稿把 800 次步内 selection_eval 归入"方法自测"、漏了 100 次 S₀ 基线、又把 Arm1 的 105 次 val 重复计进 639 → 均已更正。
- 两法 grand total 不能直接比: Arm1 的 3065 测试期调用里 2765 是共享成本。
- SkillOpt 的 "Rollout 320 / Optimizer 74" 是**有效样本 / 无法重构的旧数**, 实测是 1120 请求 / 85 调用。
- 计数差额先怀疑"展示归并", 再怀疑"漏记"; 本例两处都是展示问题, 不是日志缺行。

关联: [[skillopt-vs-s0-paired-20261124]] [[phase4a-verdict-20261120]] [[ablations-consolidated-20261124]]
