---
name: direction-global-not-pertask-20260723
description: 🚨方向纠正 - 目标是global(至少领域级), per-task非主推方向; TF-global比S_0差是真问题待解; SkillComposer hybrid是可借鉴路径
metadata: type: project
---

# 方向纠正: 目标是 global, per-task 非主推 (2026-07-23) 🚨

## 用户决策 (2026-07-23)
- **目标是 global** (至少领域级 global, 不追求跨领域)。per-task 不是未来方向。
- per-task 本质是"每题一份 skill", 泛化性弱、推理成本高、不符合"优化出一个可复用 skill"的研究目标。
- per-task 实验结果先记录, **不作为主线**。

## 现状 (同条件对位, LCB test=143)
| 方法 | overall | hard | 性质 |
|------|---------|------|------|
| S_0 (同条件) | 0.6005 | 0.3012 | 基线 |
| **TF global** | 0.5828 | 0.253 | ❌比S_0差 — **真问题待解** |
| TF per-task | 0.6698 | 0.3855 | ✅比S_0好但**非目标方向** |

**真问题**: 怎么让 global skill 超过 S_0? (当前 TF global 反而干扰强模型)

## SkillComposer 三模式核实 (arXiv:2606.06079 原文)
> Offline builds a generalized skill library; Online creates and iteratively refines skills per task; Hybrid initializes from an offline library and specializes via online evolution.

- offline (全局library): 在 LCB 差 (覆盖不够任务分布) — 与我们 TF-global 比S_0差一致
- online (per-task): 在 LCB 强 (task-specific specification 最有用) — 与我们 per-task +8.4pp 一致
- **hybrid**: Hard 级最强 (31.3 vs online) — **可借鉴路径**

## 可借鉴路径: hybrid (领域 global + 轻量特化)
SkillComposer hybrid = offline 全局 library 初始化 + online 特化。对应我们的"领域 global"目标:
- 先有一个领域级 global skill (LCB 竞赛题领域)
- 必要时轻量 per-task 特化 (但 global 是主体)
- 这可能解决"纯 global 干扰强模型" + "纯 per-task 不泛化"的两难

## per-task 结果存档 (不作主线)
- LCB test=143: overall 0.6698 / hard 0.3855, vs 同条件S_0 hard +8.4pp
- 产出: outputs/eval_lcb_tf_pertask_20260723.json
- 价值: 印证 SkillComposer per-task 有效, 但**非我们目标方向**; 可作论文消融/上界对照
- 详见 [[tf-pertask-positive-20260723]] (已降级标注)

## 下一步 (重新聚焦 global)
1. ⏳ 诊断 TF-global 为何比 S_0 差: rewriter 加的通用启发式如何干扰强模型? 是 skill 太长? 太通用? 逐条经验做消融
2. ⏳ 借鉴 SkillComposer hybrid: offline 领域 library + 轻量特化, 看能否让 global 超过 S_0
3. ⏳ 修 TF 训练循环: s0_cache 同条件重算 (当前 delta 信号污染)
4. ⏳ global 多变体/多轮: 当前 global 只用 iter1, 尝试不同 rewriter 策略 (如只改特定失败模式, 不加通用废话)
5. ⏳ 8卡4B GRPO 训练线: A模型学改写 global skill, 与 TF-global 同表对位

Related: [[tf-global-reversal-samecond-20260723]], [[tf-pertask-positive-20260723]], [[skillcomposer-sepo-verified-20260721]], [[training-vs-tf-parallel-plan-20260723]]
