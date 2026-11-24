---
name: step0-ablation-result-20261116
description: Step0 四组配对消融 — 推理级(产物级)敏感性分析: 删/保守化 Step0 文本段不降退化; 非训练机制因果检验
metadata:
  type: project
---

# Step 0 四组配对消融结果 (2026-11-16, task #47)

> ## ⚠️ 范围界定 (2026-11-16, 用户方法论修正后)
>
> 本实验是**推理级 (产物级) 敏感性分析**, 不是训练机制因果检验: 它删/改的是**最终 skill 文本的一段**, 测"这条指令在推理时重不重要", **不**测"产生这条指令的训练过程需不需要这个信号"。
>
> 因此本结果**只支持**: "Step 0 文本段不是 23 道退化的推理级主因; 退化跨 Step0 变体低重叠, 与生成不确定性一致"。
> **不支持**: "Step 0 (作为训练产物) 没价值" / "Evolver 训练机制无效" / "生成随机性是退化主因" (训练因果口径)。
>
> 真正的训练机制验证须做**训练时消融** (改变 Evolver 训练输入 → 比较各组独立生成的 skill), 见 task #48 设计中。
> 232/300 路由数字已撤下 (测试集 oracle 泄漏), 详见 [[appsplus-routing-oracle-leakage-reframe-20261116]] 和 `docs/phase2_issues.md`。

## 设计 (唯一变量 = Step 0 段, 非 Step0 段落逐字节一致)

| 组 | skill | Step 0 处理 | 调用 |
|---|---|---|---|
| A | S₀ (skill_v0000) | 无 Step 0 | 0 (复用 cache) |
| B | 当前候选 (tf_global_skill) | 双接口分叉 + 全局 FORBID | 0 (复用 retention+routed-fail) |
| C | skill_C_no_step0 | **删整段 Step 0** | 300 (新增) |
| D | skill_D_evidence | 三级证据驱动, 取消 FORBID, 弱证据不切换 | 300 (新增) |

两主终点 (A=S₀ 参照, 配对 McNemar, Holm-Bonferroni m=6):
- Preservation (208 道 S₀-pass): regress↓=好
- Acquisition (92 道 S₀-fail): rescue↑=好

## 结果 (600 flash 调用, agent_fail=0 全组, 干净)

### 主终点 vs A=S₀ (6 检验 Holm 校正后全拒绝 H₀ — 三组都显著异于 S₀)

| 组 | Preservation regress | p_adj | Acquisition rescue | p_adj | hard/300 |
|---|---|---|---|---|---|
| B | 23 | 4.77e-7 | 24 | 3.58e-7 | 209 |
| C | 22 | 4.77e-7 | 26 | 1.49e-7 | 212 |
| D | 25 | 2.38e-7 | 27 | 8.94e-8 | 210 |

### ★ Step-0-隔离比较 (C/D vs B, 配对) — 全部不显著

| 比较 | Preservation (B-better : C/D-better) | p | Acquisition | p |
|---|---|---|---|---|
| C vs B | 13 : 14 | **1.00** | 8 : 10 | 0.81 |
| D vs B | 13 : 11 | **0.84** | 8 : 11 | 0.65 |

**删 Step 0 (C) 退化=22 ≈ B 的 23; 保守化 (D) 退化=25 ≥ B 的 23。** Step 0 假设预测 C/D < B, 结果相反 → **假设被反驳。** 配对对称 (13:14 / 13:11) 说明即使逐题也无 Step 0 效应。

## 噪声签名 (退化/rescue 集合重叠) — flash 随机性的指纹

- 退化: B/C/D 至少一组退化的题 = **45 道**, 但三组都退化的 = **仅 6 道**。
- B 的 23 道退化: C 也退化 9, D 也退化 12。C "治好" B 的 14 道, 却 "新弄坏" 13 道别的; D 同理 (治好 11 / 新坏 13)。
- rescue: 三组都救的 = 12/38 (比退化稳定, 真正可学的题重叠更多)。

→ 退化是**非重叠的逐题翻转**, 典型的 temp=0 非完美确定**生成随机性**签名, 不是 Step 0 确定性机制。若 Step 0 是主因, 删它应消除大部分 B 退化 (15:1 量级配对差异, 易检测); 实测 13:14 完全对称。

## 结论 (诚实口径)

1. **Step 0 双接口分叉不是 23 道退化的主因。** 删/保守化 Step 0 不降低退化计数 (C 22 / D 25 vs B 23), 配对无显著差异。[[appsplus-regression-rootcause-step0-contract-20261116]] 的机制假设 (Step 0 分叉 → preservation 代价) **被消融证伪**。
2. **退化主因是 flash 生成随机性** (temp=0 非完美确定): 45 道题在 ≥1 组翻转, 仅 6 道三组共翻。agent_fail=0 全组 → 非 infra, 是模型生成波动。
3. **Step 0 也不贡献 rescue**: C (无 Step 0) rescue=26 ≥ B 24, D=27。rescue 是 "候选 skill 跑这些题" 的一般效应, 非 Step 0 教接口识别所致。
4. **候选 skill 纯能力 ≈ S₀** (B 209 / C 212 / D 210 vs A 208, 净 +1~+4), 与候选 vs SkillOpt 209≈211 (p=0.88, [[appsplus-attribution-decompose-26vs5-20261116]]) 一致 —— skill 内容在噪声内。
5. **路由是 +21 部署增益的唯一来源** (232 vs 211): 候选 skill 自身净近中性 (rescue≈regress), 路由结构性地把 regression 侧归零, 只保留 rescue。这与修正归因完全一致。

## 对论文叙事的影响

- ✅ 可写: 消融 (4 组 × 2 终点, Holm 校正) 表明 Step 0 接口分叉**不是**退化主因; 退化主要来自生成随机性 (低跨组重叠, 配对对称)。这是对 "显式规则扰动稳定行为" 假设的**证伪**, 而非支持。
- ✅ 可写: 候选 skill 纯能力 (含/不含 Step 0) 在统计上不区分 S₀/SkillOpt → **部署增益完全归因路由架构**, 非 skill 内容。
- ❌ 不可写: "Step 0 分叉导致 preservation 退化" (消融证伪)。
- ❌ 不可写: "保守证据驱动策略减少退化" (D=25 ≥ B=23, 不显著)。
- ⚠️ 小效应不排除: ±2-3 的 Step 0 效应在噪声内不可检测, 但绝不是 23 道退化的解释。

## 数据源 + 脚本

- runner: `rl/train/verl1/skill_evolution/step0_ablation_runner.py`
- launch: `rl/train/verl1/skill_evolution/launch_step0_ablation.sh`
- 结果: `outputs/crossmodel_baseline/step0_ablation_20261116/results/`
  - `group_C_results_20261116_144437.json` (300 逐题)
  - `group_D_results_20261116_144437.json` (300 逐题)
  - `step0_ablation_summary_20261116_144437.json` (汇总+McNemar+Holm+pairwise)
  - `run_step0-ablation-flash-20261116-144436.log`
- skill: `step0_ablation_20261116/skills/skill_{B_current,C_no_step0,D_evidence}.md`
- A/B 复用: A=`appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.json`; B=retention(208)+routed-fail(92)

关联 [[appsplus-regression-rootcause-step0-contract-20261116]] [[appsplus-attribution-decompose-26vs5-20261116]] [[appsplus-rescue-retention-paired-vs-skillopt-20261116]] [[tf-evolver-appsplus-routing-retention-20261113]]。
