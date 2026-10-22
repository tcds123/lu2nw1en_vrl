---
name: p6-routing-saturation-20260925
description: 复审§13.10#5/§12.1-12.2/§13.8 路由饱和研究——用 P3 干净 shuffle 协议数据建题目×skill 互补矩阵, split-half 检验可学路由信号是否存在
metadata:
  type: project
---

2026-09-25：复审 §13.10 优先级 #5（P6）路由/饱和研究。[[p4-shortcut-fix-diagnosed-20260924]] 之后执行。零 API 离线分析。

## 核心问题（复审 §12.1/§13.8）

不同 skill 是否存在**稳定、可预测的互补**？若有可学信号才值得做路由；若无（只是采样噪声），先造复杂路由只增参数/搜索成本。§13.8 的历史互补矩阵基于**未打乱、采样污染**数据（"互补包含采样因素, 尚不能直接推出可学习的路由收益"），§13.8 明确要求"真正决策所需的互补性应在打乱、规范判分的数据上重测"。P3/P4 已产出干净 shuffle 协议数据（7 skill × 4 seed × 106 × k=3），正好满足。

## 数据

干净 shuffle 协议（shuffle_item per qid/seed, parse_choice_label, k=3）per-cell cache `per_{skill}_{hash}_seed{N}_shuf.json`（phaseA_shuffle_20260917 目录），结构 `{qid: pass_rate}`。4 个**真正不同**的 expert 候选：s0(68571cb2 基线) / on(fa63335e ON) / off(02a7f019 OFF 另一条 run) / skillopt(cfae156d)。(on_noimpl/on_noAsentence 是 ON 消融变体且 P4 证明捷径耦合, rule_meta_guess 是零 API 平凡规则, 不作路由候选。) test=106 → meta_46（捷径子集）+ other_60（**无捷径**子集, 路由研究的干净靶标）。

## 方法（§12.1 line 336 要求）

Split-half reliability: 4 seed 拆 selection 半 {1,2} / eval 半 {3,4}。selection 半选每题 best skill → 看 eval 半是否仍 best（可预测性）。避免在同一批噪声分数上挑最大值的乐观偏差。oracle=eval 半 per-item best（用 eval 标签, 不可部署, 仅诊断上界）。

## 结果（4 skill, other_60, 干净 shuffle 协议）

### [A] 各 skill 平均正确率（sample-level: 60 题 × 4 seed）

| skill | other_60 | meta_46 | all_106 |
|---|---|---|---|
| s0      | 0.4139 | 0.2011 | 0.3215 |
| on      | 0.3750 | 0.5380 | 0.4458 |
| off     | 0.4125 | 0.4475 | 0.4277 |
| skillopt| 0.3875 | 0.7880 | 0.5613 |

注: meta_46 上 on/skillopt 高 → 但 P4 证明这是**文本触发捷径**(meta_sel 0.338 vs s0 0.043), 非通解题力。other_60(无捷径) 上四 skill 持平 0.375~0.414, 无显著差异。

### [B] Split-half 路由（other_60, sel={1,2} → eval={3,4}）

- best fixed skill on eval half: **off** acc=0.4111
- **routed (by sel-half best) eval acc=0.4194 → gain +0.0083**（≈0）
- oracle (eval-half best, 不可部署) acc=0.5333 → gain +0.1222
- sel-best==eval-best agreement=0.550（4 skill 随机≈0.25, 0.55 弱高于随机但 translate 到增益≈0）

### [C] 成对 skill per-item Δ 的 sel/eval 相关（other_60）

| pair | sel meanΔ | eval meanΔ | r(sel,eval) |
|---|---|---|---|
| on−off | −0.036 | −0.039 | **0.024** |
| on−s0 | −0.042 | −0.036 | **0.017** |
| off−s0 | −0.006 | +0.003 | 0.155 |
| on−skillopt | −0.028 | +0.003 | **0.014** |
| off−skillopt | +0.008 | +0.042 | 0.210 |

per-item 跨两半的 skill 偏好相关 r≈0.01–0.21 → 选半选出的 best **不预测** eval 半的 best。

### [D] 噪声地板（同 skill sel/eval per-item 正确率 r）

s0=0.825, on=0.731, off=0.828, skillopt=0.761 → 单 skill 难度稳定, 但**skill 间偏好不稳** → 互补是采样噪声非可学信号。

### [E] 统计检验

- routed gain +0.0083, **bootstrap 95% CI [−0.047, +0.042]**（含 0, 不显著）
- 随机路由 permutation null: mean −0.021, 97.5 分位 +0.017; observed +0.0083 **在 null 内**（p>0.025）→ 路由 ≈随机
- oracle +0.1222 不可部署（用 eval 标签）, 非 learnable routing 收益

## ✅ 结论（P6 定稿）

干净 shuffle 协议下, distinct skill 间在**无捷径 other_60** 上的互补是**采样噪声, 非可学路由信号**: routed gain +0.0083（CI 含 0）, permutation 内, per-item Δ 跨半相关 r≤0.21, 而 oracle(不可部署)虽 +0.12 但非 deployable。→ **per-item / facet / topic 路由在当前 skill 集上无可学习信号, 不值得构建复杂路由框架。** 与 §13.8 历史矩阵结论一致, 但本次基于更强证据（打乱+规范判分+split-half），坐实"饱和"元判断中"互补=噪声"分支——不是"所有 skill 无 headroom"（那是过强普遍化）, 而是**这组候选 skill 在 other_60 上无可预测互补**; meta_46 上的差异 P4 已证明是捷径非能力。

**对论文叙事**: 单全局 skill 进化在 LiveMath(other_60) 上确已饱和(四 skill 持平, 路由无增益); meta_46 上的"超越"是捷径驱动。方向应转向换表示层/换目标(AIME)或堵捷径后重测, 而非 per-topic 路由库。
