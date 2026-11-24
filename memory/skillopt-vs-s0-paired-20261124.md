# SkillOpt vs S₀ 配对四格表 — 2026-11-24 (0 API)

背景: Phase 3b 同跑三臂, SkillOpt 1956/2765 = 70.74% **低于** S₀ 2003/2765 = 72.44% (−47 题, −1.70 pp)。
用户要求先分清三层 (**结果 / 统计 / 机制**), 不允许直接把差异解读为"SkillOpt 算法有缺陷"。
本节只做统计层核销, **事后补算, 非预注册**, 不改动 Phase 4-A 任何结论。

产物 (机器数据, 未入库):
- `rl/train/verl1/outputs/crossmodel_baseline/phase4a-audit-20261120/paired_skillopt_vs_s0.full.posthoc.json`
- `.../verdict_skillopt_vs_s0.md` (推理与公平性核查全文)
脚本: `frozen_phase4a.py::run_pair(a=skillopt, b=s0)` + `paired_stats` / `paired_net_gain_ci` / `mcnemar_paired` (Clopper-Pearson 映射 + exact binomial)。

## ① 结果层

| 臂 | hard pass | 率 | 单样本 CP 95% |
|---|---|---|---|
| S₀ | 2003 | 0.7244 | [0.7073, 0.7410] |
| SkillOpt | 1956 | 0.7074 | [0.6901, 0.7243] |
| Arm1 | 2077 | 0.7512 | [0.7346, 0.7672] |

单样本区间两两重叠 → 光看边际率不足以判退化, 必须配对。

## ② 统计层 (四格表, n=2765)

| | SkillOpt=1 | SkillOpt=0 |
|---|---|---|
| **S₀=1** | 1788 | 215 |
| **S₀=0** | 168 | 594 |

- 不一致对 d = 383; g = 168 − 215 = **−47**; rate −0.0170
- **CI [−0.0309, −0.0028], 整段在 0 以下**; exact McNemar p = **0.0186**
- 去重 2744: g = −48, CI [−0.0314, −0.0033], p = 0.0153
- 结论: **真实退化信号, 不是点估计波动** (点估计留在同一跑内, 口径一致)。

## ③ 机制层 — 退化发生在哪里

| 分层 | n | SkillOpt-only | S₀-only | g | p |
|---|---|---|---|---|---|
| L1 (plain) | 1791 | 121 | 136 | −15 | (见 json) |
| L2 | 341 | 36 | 33 | +3 | |
| L3.func | 373 | 22 | 21 | **+1** | 1.0 (无差) |
| L3.class | 260 | **7** | **24** | **−17** | **0.0033** |

- 损失主体在 L1 (215 道 S₀-only 中 136 道是 L1), 但 L1 比例大, 相对量小。
- **SkillOpt 在 class 层崩得最狠 (7 vs 24)**; 而 Arm1 的全部增益恰在 class (132 vs 45) → 两个方法不是在同一轴上的高低, 是**方向相反**。
- L3.func 上 SkillOpt 与 S₀ 无差 → 它的 skill 没有命中 fn-call 接口重建这一层。

## 公平性核查 (回答"SkillOpt 是否真的做了它该做的事")

1. **同一跑内交错**: ticket_order = [s0 id_i, skillopt id_i] 交替, 8 workers → 无跑间漂移、无批次顺序效应。
2. **同模型同解码**: deepseek-v4-flash, temp 0.0, max_tokens 16384, enable_thinking=false, 同一 evaluator `_evaluate_d5`。
3. **skill 文本确实不同**: SkillOpt 2986 字符 vs S₀ 1797, difflib ratio 0.63 → 不是"没改动"。
4. **训练预算对等**: 同 `split_dir`, 同 train_size=80 → 不是"少训练"能解释的差异。
5. **SkillOpt 在其自身目标上是在提升的**: 300 题选择/测试 chunk 0.58→0.64 与 0.6933→0.7033, 却在 2765 上净跌 → **典型的负迁移 / 目标错位特征** (优化器追的是自己 300 题上的代理信号)。
6. ⚠️ **深度限制**: SkillOpt 只跑了 **8 步** (4 epoch × 2), 接受 1 / 拒绝 7; 且其 300 题 chunk **与 2765 测试集重叠**, 不是独立集。→ 不得用这一行证明"SkillOpt 算法本身不行", 只能写"在 8 步预算与本配置下低于 S₀"。

## 对论文措辞的约束

- ✅ 可说: Arm1 在 2765 上**同时**优于 S₀ (+74, p=8.0e-4) 与 SkillOpt (+121, p=3.3e-8); 两个比较各自无条件成立 (共享同一 S₀/SkillOpt 配对, 互不派生)。
- ❌ 不可说: "Arm1 > SkillOpt > S₀" 的单调链 — SkillOpt < S₀ 已经把中段打断。
- ❌ 不可说: 差异源于 Evolver 机制。**Arm1 的 evolver 调用预算尚未与 SkillOpt 的 320 rollout + 74 optimizer 调用对账** → "预算/样本选择/设计差异" 这条替代解释还未排除。
- ✅ 可在讨论节写: SkillOpt 在训练池上局部改进却整体退化 → **小样本 fail 池上的优化目标与最终 hard pass rate 存在系统性错位**, 这本身是支持"需门控/需 held-out 判据"的论据 (与双门控设计呼应), 但不能当作对它算法的判决。

## 未完成

- Arm1 evolver 精确调用数尚未从 `tf48-arm1-full-20261117` 日志拉出 → 预算对账 PARTIAL。
- Issue #93 问题 2 (Interface 条款在 9 题 fn-call 上**实际是否干预**) 仍未解; 本节不涉及。

关联: [[phase4a-verdict-20261120]] [[ablations-consolidated-20261124]] [[phase3b-baseline-run-progress-20261120]]
