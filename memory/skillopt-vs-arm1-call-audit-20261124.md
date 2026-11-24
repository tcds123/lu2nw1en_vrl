# SkillOpt vs Arm1 调用预算对账 — 2026-11-24 (0 API, 只读日志)

用户任务: Issue #117 之后, 先回答 **Arm1 (+74) 是否只是"调用更多/预算更大"**。只读日志, 不补跑、不调 API、不改代码。
这是进入跨模型独立复现之前必须完成的审计 (顺序: 数据审计 #13 → 预算审计 #15 → 跨模型复现 #14)。

## 口径 (用户指定的六阶段, 避免重复计)

1. **① 优化器调用** / **② 候选生成 (Rollout)** / **③ Selection Eval** 三者构成**训练预算**。
2. **⑥ 正式评测不计入训练预算** — 2765 题正式评测与 SkillOpt 的 300 题 test chunk 均单列; 2765 两臂共享。
3. **④ Smoke Test 单列** (本次两臂主 run 均 0; 独立 `smoke_48_20261117` = 9 次不算)。
4. **⑤ 重试单列** (两臂均 0; 无空响应/无 retry 日志)。
5. 请求数 = HTTP 请求; SkillOpt "Rollout 320" 是**有效样本数**, 实际 **1120 请求** (40/(4+2)/步 × 8)。
6. 被拒候选照样计入; 优化器与目标调用分列 (无一次调用同时生成+打分)。

## 实测数 (逐项可追溯到日志, 见台账 json)

| 阶段 | SkillOpt | Arm1 |
|---|---|---|
| ① 优化器 | 85 (analyst 47 / merge 24 / ranking 8 / guidance 3 / meta 记忆 3) | 411 (rewriter 324 / distill 81 / merge+compose 6) |
| ② 候选生成 | 1240 (1120 rollout + 120 slow 对比) | 429 (324 B-eval + 105 val 门控) |
| ③ Selection Eval | 1200 (800 步内 + 100 S₀ 基线 + 300 slow) | 0 单列 (105 已并入 ②) |
| **训练合计** | **2525** (带 token 记录下界 1622) | **1050** |
| ④ Smoke | 0 | 0 |
| ⑤ 重试 | 0 | 0 |
| ⑥ 正式评测 (不计训练) | 600 (300 ⊆ 2765) | 2765 + 215 + 85 |
| 训练 token | 4.96M 已记 + ~1.24M 未记 ≈ 6.2M | 未记录 (估 ~1.3M) |
| 净增益 (2765, 配对) | **−47** | **+74** |

方法学花费 (训练 + 非共享评测): SkillOpt **3125** vs Arm1 **1350** (0.43×); 含共享 2765 总计 5890 vs 4115。

## 三个结论

1. **调用预算差异**: Arm1 训练 1050 vs SkillOpt 2525 (0.40×; 用带 token 记录的下界也是 0.65×), token 约 1/4.5。
   → "Arm1 赢在预算更多" **已排除, 方向相反**。
2. **优化效率差异**: Arm1 14.2 请求/净增益题; SkillOpt 负效率。结构上 Arm1 39% 训练调用是优化器侧 (rewriter+distill),
   SkillOpt 只有 3.4%, 且 52% 的调用打在与其最终 2765 重叠的 100/300 题 chunk 上 → 目标错位/负迁移。
3. **归因边界**: 已排除 ①调用更多 ②测试泄漏 ③重试放大; 部分排除 ④样本选择 (同 split/train_size=80, 但 Arm1 只训 27 道 S₀-fail);
   **无法排除** ⑤优化目标/门控凭据 (最主要) ⑥候选生成方式 ⑦缺 Arm0 同预算对照 ⑧SkillOpt 仅 8 步。
   → 只能写"该完整方法配置在本文评测下更优", **不能**写"优势来自 Evolver 机制"。

## 论文措辞

- ✅ "在相同模型与解码配置下, 我们的框架以更少的训练调用 (1050 vs 2525) 与约 1/4 的 token 取得 +74 配对净增益, SkillOpt 为 −47"。
- ❌ 单调链 "Arm1 > SkillOpt > S₀"; ❌ 因果归因 Evolver 机制。

## 未解决 / 需授权

- Arm1 不记录 token → token 对比只能以**估算**呈现; 要硬证据需加 usage 计数后重跑 (未授权)。
- 补 ⑦ 的优先级是 **Arm0** (同预算同候选生成、无训练信息无门控), 而非把 SkillOpt 跑满 16 步 (只补 ⑧)。均未授权。

## 产物 (outputs/ gitignore, 未入库)

- `rl/train/verl1/outputs/crossmodel_baseline/phase4a-audit-20261120/skillopt_vs_arm1_call_audit.md`
- 同目录 `skillopt_vs_arm1_call_ledger.20261124.json` (headline/grand_check 已按六阶段重写)

## 教训

- 旧稿把 800 次步内 selection_eval 归入"方法自测"、漏了 100 次 S₀ 基线、又把 Arm1 的 105 次 val 重复计进 639 → 均已更正。
- 两法 grand total 不能直接比: Arm1 的 3065 测试期调用里 2765 是共享成本。
- SkillOpt 的 "Rollout 320 / Optimizer 74" 是**有效样本 / 近似值**, 实测是 1120 请求 / 85 调用。

关联: [[skillopt-vs-s0-paired-20261124]] [[phase4a-verdict-20261120]] [[ablations-consolidated-20261124]]
