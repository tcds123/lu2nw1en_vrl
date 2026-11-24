---
name: 48-preassessment-0api-20261117
description: #48 Pre-assessment 0-API 闸全过 — Arm2/Arm4 训练干预为零 (P0-3 量化), 待用户决策实验矩阵
metadata:
  type: project
---

# #48 Pre-assessment 0-API 闸完成 (2026-11-17)

承接 [[d5-fncall-harness-fix-20261117]] (D5 解锁 #48) + 用户审查闸 (issues #82, 暂缓正式训练)。
脚本: `eva/preassess_48_0api.py` (0 flash 调用, 离线统计)。

## 0-API 闸 (5 项全过)

| 闸 | 结果 | 量化 |
|----|------|------|
| 闸1 P0-3 Arm2/Arm4 覆盖率 | PASS (干预=0) | train fail 池 27 题全 stdin, fn-call=0 → Arm2 删 0 / Arm4 抑制 0 |
| 闸2 Arm3 s0='' 静态可行性 | PASS | make_rewriter_user→2857 chars 非空; compose→空→best_S 留真 s0; smoke 待验证 |
| 闸3 gate 隔离 | PASS | Arm2/3/4 不读 val cache → gate 不变 |
| 闸4 训练题ID+缓存一致 | PASS | split/cache ID 80/80 完全一致, 共享 precompute 幂等 |
| 闸5 P0-2 agent_ok 边界 | PASS | 纯 infra (源码核实 #82); train cache agent_ok True=80 False=0 |

## ⚠ 重大发现: Arm2/Arm4 训练干预为零 (P0-3 量化)

train split 80 题, S₀ fail 池 27 题, **其中 fn-call = 0 道** (全是 stdin 题)。
- Arm2 (drop_interface) 实际删除 = **0 道** (fail 池没有 fn-call 可删)。
- Arm4 (suppress_exp) 抑制 = **0 道** (train_used 里也没有 fn-call)。
- → Arm2/Arm4 训练输入与 Arm1 完全相同 (训练输入层面无差异)。

**根因**: D5 修复后, train split ~9 道 fn-call 题多数 S₀ 已 hard=1 → 不进 fail 池。余 27 fail 全 stdin。
用户 P0-3 预判"删 0~3 道", 实际更极端: **删 0 道** (干预为零, 非功效不足)。

**含义 (表述经用户 2026-11-17 纠正)**: 即使训练输入相同, 独立运行仍可能因模型采样/API 随机性产生不同结果, **不能预先断言两组分数必然相同**。准确说法: Arm2/Arm4 **没有形成有效训练干预**, 组间结果差异只能反映运行随机性及其他非干预因素, **不能用于识别接口反馈通道 / 经验蒸馏通道的因果效应**。这不是"功效不足"(干预真实但样本少), 而是"干预为零"。

## 实验矩阵冻结 (用户 2026-11-17 批准)

正式矩阵 = **Arm1 (full) + Arm3 (drop_scaffold)** 两臂。Arm2/Arm4 标记 `zero_intervention_deferred`, 不消耗正式训练预算。
- Arm2/4 若要做, 需先构造含 fn-call fail 题的训练 split (扩大 train_n / 换 split), 属另一实验设计, 单独记录。
- **执行顺序**: (1) ≤5 call smoke (Arm3 空 S₀ 1题g2 + eval_independent 5题混合) → (2) 正式 Arm1 → (3) 正式 Arm3 (独立, 不复用 Arm1 skill/经验池) → (4) 配对统计。

## Arm3 干预范围 (用户明确)

Arm3 检验的是 **S₀ scaffold 在整个 Evolver 训练流程中的综合作用** (4-mechanism 联动):
移除 Rewriter 输入 S₀ + 全局 Skill 合成空 + 失败回退空 + 退化判断失效。
**不能**简单解释为"S₀ 提示词是否有用", 也**不能**把四个机制分别归因。
- Arm1 明显优于 Arm3 → scaffold 对当前 Evolver 流程有贡献。
- 无明显差异 → 当前配置下未观察到 scaffold 带来可识别增益。

## 统计锁定 (用户 2026-11-17)

- **δ=0.03 固定** (300 题约 9 道净增益), 不看结果后调阈值。
- **核心配对 = Arm1 vs Arm3**。Arm1 vs S₀(215) / Arm1 vs SkillOpt(218) 是参考基线 (S₀/SkillOpt 缓存零 API 但生成预算/选择过程不同, 不能单独证明算法优越性)。
- 共同有效题集 (valid_ids 交集) 上做 McNemar + 配对净增益 + CI。

## 研究边界保留

本轮可检验 S₀ scaffold 对 Evolver 训练流程的贡献, **仍不能**单独证明 Evolver 整体优于等预算无进化方法 (需后续 No-Evolution Control / Arm0)。

## 待用户决策 (实验矩阵)

- **选项 A (推荐)**: 正式矩阵只跑 **Arm1 (full) + Arm3 (drop_scaffold)** 两臂。Arm2/4 暂缓记"干预为零"。省 ~10h。
- **选项 B**: 仍跑 4 臂, 结论标 Arm2/4 干预为零、数字仅作基线可复现性参考。多花 ~10h。
- **选项 C**: 改 train split 让 fn-call fail 题进池 (扩大 train_n / 换 split), 属另一实验设计。

## 各 arm 训练输入对比 (0-API 实测)

| Arm | train_items | fn-call in pool | 干预 | 预期 |
|-----|------------|-----------------|------|------|
| Arm1 full | 27 fail | 0 | 无 | 基线 |
| Arm2 drop_interface | 27 fail (过滤 0) | — | **空操作** | ≡ Arm1 |
| Arm3 drop_scaffold | 27 fail | 0 | s0_train='' (4-mechanism) | 真干预, 待 smoke |
| Arm4 suppress_exp | 27 fail | 0 (抑制 0) | **空操作** | ≡ Arm1 |

## 输入数据状态

- S₀ = 1797 chars (appsplus initial.md)。
- train split 80 题 (cache: 20261113 正式 run train_s0_cache.json, hard=53 fail=27, 幂等)。
- val cache 100 题 hard=58 (D5 前 20261111-v2 口径, gate 用)。
- test cache 300 题 hard=208 (D5 前; 正式 run 内 Path-B 重建 215)。
- 共享 `--s0_cache_train` 路径 → precompute 幂等 (各 arm 串行, 首个触发, 后续复用)。

## 下一步

1. **≤5 call smoke** (`eva/preassess_48_smoke.py`): Arm3 空 S₀ 1题g2 (验证 rewriter s0='' 产非空候选 + 不中断训练流程) + eval_independent 5题混合(stdin+fn-call) (验证 API/评测分流/infra 分类/落盘)。
2. smoke 通过 → 正式 run: Arm1 (full) → Arm3 (drop_scaffold), 串行, 各独立 out_dir, 不复用 skill/经验池。
3. 配对统计: 共同 valid_ids 交集上 McNemar + 配对净增益 + CI, 对照 δ=0.03; 三层结论边界。

详见 issues #82 (审查闸) + #83 (P0-3 量化) + #84 (矩阵冻结) + plan `luminous-churning-avalanche.md`。
