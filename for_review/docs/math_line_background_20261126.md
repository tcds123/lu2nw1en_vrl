# 数学线背景与历史 — AIME × LiveMathBench (供 codex 分析解读)

> 本文档仅聚焦数学线两条数据集（AIME / LiveMathBench），供独立评审者快速建立完整背景。所有数字均与 `memory/` 原始记录及产物文件核实。时间线按实际发生顺序。

---

## 0. 一句话定位

我们在做 **training-free skill 自进化**：不改模型参数，靠 evolver 优化 system prompt（"skill"）文本，引导推理模型（DeepSeek-V3.2 API，单次生成，无工具/无迭代）的行为。数学线是这条研究最早、跑得最久的两条数据集，承载了**方法有效性边界的几乎所有关键证据**。

```
evolver 产出 skill 文本 → skill 作 system prompt 喂推理模型 → 评测(pass rate + bootstrap CI) → CI 门控反馈 → 下一轮进化
```

两条线先后证伪了同一范式（"单全局 skill 文本进化"），但证伪方式截然不同——这正是论文的核心叙事张力。

---

## 1. 两条数据集的结构差异（理解后续一切的根基）

| 维度 | LiveMathBench | AIME |
|------|---------------|------|
| 形式 | **theorem-level MCQ**（每题 4 选项，正确答案恒 label=A） | 自由作答（填整数 0-999） |
| 规模 | 177 题（**train18/val53/test106**，1-3-6 split seed42，1:3:6≈18:53:106） | 每年 30 题（AIME24/25/26） |
| 捷径 | **有**：meta-option 文本 "One of the remaining options is correct, but a stronger result can be proven" 出现 → 100% 正确答案 | 无（须真算） |
| S₀ 表现 | MCQ 0.34~0.41（受 shuffle 打乱选项影响） | V3.2 S₀≈0.74 / flash S₀≈0.85 |
| 失败模式 | 高度同质（93% 难题回退到同一 meta-option 失败类型） | 高度异质（逐题特异，组不成泛化 global skill） |
| 饱和机制 | **捷径吃光**全部增益 + 非捷径题无 headroom | **异质任务** → 门控正确拒 → best==S₀ |

**关键洞察**：LiveMath 有可被 skill 放大的结构捷径；AIME 没有，但 AIME 的失败模式太异质，单条 global skill 无法跨题泛化。两条线从相反方向撞上同一堵墙：**单全局 skill 文本进化只在 (a) 有结构捷径可吃 或 (b) S₀ 已覆盖形态级时有效；在需真实推理且 S₀ 已饱和处打不过 S₀。**

---

## 2. 时间线：LiveMathBench 线（2026-07 → 2026-09 饱和 → 2026-11 新机制重测）

### 2.1 早期坐实 +16.1pp（2026-07）
- evolver 4-seed 在 LiveMath 上稳定产出 **+16.1pp over S₀**（test=106 独立评测, CI[+0.126,+0.196]；优化过程 in-sample best_val_delta=+0.254 在 select=71）。这是数学线第一个"有真实效果"的信号。
- 注：+16.1pp 原始产物(2026-07 v2)已清理、不可复算；可复现最近 L2 = +10.8pp(2026-09 multirun1, 2-seed) / +18.6pp(2026-09 onoff_shuffled, 3-seed)。故 +16.1pp 作"参考效果"非"对齐基线"。
- 但同一批实验发现 AIME24 跨任务泛化失败（泛化边界负结论）——提示效果可能不迁移。

### 2.2 捷径三证（2026-09-10 → 09-29）—— 决定性发现
LiveMath 的 +16.1pp 不是"算法变聪明了"，是 skill 学会了**吃 meta-option 捷径**。三证收敛：

1. **P9 语义证**：meta-option 文本出现即 100% 正确答案（label-A），零 API 子串匹配 `rule_meta_guess` 达 meta_46=**1.0000**，高于 OFF 0.7053 / skillopt 0.7995。即"超越 skillopt"可被 trivial 手写规则拼装达成（trivial ensemble ≈0.67 > skillopt 0.576），作研究里程碑无意义。
2. **P8 因子证**：other_60（非 meta 题）所有 skill（含 skillopt）≤ S₀，无 headroom。加任何 meta-option skill 让真数学题微降。
3. **Step6 形式证**：同题 MCQ vs 自由作答对照，gain_MCQ=+0.1682（p=0.00025 显著）vs gain_freeform=+0.0047（≈0）→ **去掉 MCQ 选项，增益即塌缩**。

### 2.3 MCQ 机制耦合发现（2026-09-30）
试图把 LiveMath 的 evolver 机制移植到 AIME 时发现：**新机制本身就是 MCQ 耦合的**。`classify_failure_type`/`FAILURE_FOCI`/`_FIVE_PEAKS`/`--facet_axes` 默认值几乎每一处都锚定 theorem-level MCQ 结构——正是捷径所在的结构。`facet_tree/expand/pool_filter` 的区分性特征恰恰锚定 MCQ，**捷径不是副产品，是机制设计目标**。

### 2.4 饱和合成结论（2026-09-24）
LiveMath 对"单 skill 进化"已饱和：meta=trivial 捷径（子串匹配即完美），other=无 headroom。当时裁决"不可提升"。

### 2.5 ⚠️ 重新评估与修正（2026-11-24）—— 转折点
AIME 放弃后重看 LiveMath，**用户两次纠正了过死的判决**：
1. 2026-09 的饱和判决用的是**旧 evolver**，而 evolver 在 2026-10/11 已发生算法改造（加 paired_rescore 等新机制）——不能跨版本转移判决。
2. "无法超越 skillopt"≠"无价值"——+16.1pp over S₀ 本身就是可发表的显著效果。论文门槛 = 真实增益 over S₀，不是"打平 skillopt"。

代码核实发现：math 后端已含 learn_positive/meta_contrast/subset_admit（2026-10-22），但**未含** AIME 后端 2026-11-02 才有的 6 个测量/门控新机制（paired_rescore / held_out_gate / EFL / rescue / facet_group_admit / cache manifest）。即"新机制在 LiveMath 上的效果"=**真欠一次实验**。

### 2.6 新机制移植 + V3.2 smoke 重测（2026-11-24 → 11-25）
移植 6 机制到 math 后端（+610 行/2 文件，MCQ taxonomy 逐字未动 = "LiveMath taxonomy + AIME 测量地基"唯一未测叉乘），V3.2 跑 smoke（iters=2, eval_k=8, 5 机制 ON）。

**smoke 结果（本次交付包的主角）**：6 移植机制端到端跑通（paired_rescore 5-tuple 在 MCQ 后端不崩、F6 manifest 写入、bootstrap CI、rescue 正确不触发）。但 **10 个候选（6 facet_edit + 2 overlay + 2 expand）全部 0 promoted**，best_skill 保持 S₀ 不变。

⚠️ **关于"两轮方向一致"的重要修正（codex 评审抓出）**：Option Comparison iter1（hash `0d319a89`, +0.023）与 iter2（hash `5fe9dd30`, +0.016）是**两个不同的 skill patch**，不是同一候选的独立重复。所以"两轮 delta 都正"是同 facet 下两个不同候选各自的正向点估计，**不能当独立重复验证**。真正待审的是：CI 在 n=10（题数）、σ≈0.044 下检出下限 +0.027，而候选 +0.023 < 0.027 注定跨零——是统计功效不足，还是候选效应不足，**两者无法从 0 晋升区分**。详见 `review_requests/20261125_codex/` + `incoming/20261126_codex_mathline/AUDIT_RESPONSE_20261126.md`。

---

## 3. 时间线：AIME 线（2026-08 → 2026-11 三重饱和 → 收尾）

### 3.1 从零起步 + 泛化阴性（2026-08）
- AIME fresh rerun：CI-gated 阴性，无共享元结构。早期即发现失败模式逐题特异，组不成泛化 global skill。

### 3.2 防截断线（2026-11-16 → 11-23）—— 一条路线的系统性否证
预设方向是"让 skill 自主生成收尾纪律、控制推理长度"，6 条路径**全部无 CI>0 收益**：

| 路径 | 结果 |
|------|------|
| run1 实时进化 | +0.100 但来自手写起点预埋（起点污染），非进化产物 |
| run2 纯 S₀ 起点 | 4 轮全 KEEP PREV，CI 从未过零（零进化） |
| run3a (cap16384) | tr=0%（不截断），无信号 |
| run3b (cap8192, retry1) | tr 漏计 82%（retry 覆盖 meta） |
| run3c (cap8192, retry0) | 4 budget 候选全 CI≤0，0/4 过门 |
| 旁路蒸馏 base_rule | 注入即降准确率 -3.4pp、截断率翻倍（反效 kill） |
| 单 bullet 消融 | ≈0（selection on noise） |

**核心诊断**：自然语言 skill 无法约束模型 token 预算行为（模型没有 token counter，无法执行"超 6000 token 停止"这类阈值指令）——**工具匹配错误**，是 skill 优化的边界识别，非方法论失败。

### 3.3 准确率线（2026-11-19 → 11-23）—— 唯一真实信号
evolver 常规 facet_tree 机制**自然涌现**（非预设）出 Quantity Parsing 进化点（答案形式转换规则）：

| 数据集 | 角色 | delta | CI | 性质 |
|--------|------|-------|-----|------|
| AIME24 | in-sample | **+0.100** | +0.011 | 显著 |
| AIME25 | held-out 1 | +0.019 | 跨零 | 弱正向不显著 |
| AIME26 | held-out 2 | +0.026 | per-seed CI 不跨零 | 弱正向，3 seed 全正 |

衰减 **74~81%**。逐题归因：稳定救回的全是答案形式转换题，稳定搞坏的全是几何构造题——方向与作用面一致，证明 +0.019 不是纯噪声，是"作用面窄"的真实信号。

### 3.4 三重饱和（2026-11-05 flash + 11-23 V3.2 第三次确认）
1. **结论饱和**：in-sample 显著 / held-out 不泛化，被 8+ run、两个模型反复确认（V3.2 0promote；run6d iter2 +0.042→−0.035 翻转；rep5 +0.044<placebo max +0.049 p=0.21；flash +0.078→+0.017）。
2. **模型饱和**：flash S₀ 0.854/0.838，信号点仅 9~14 个；检出下限 ~+0.10（k=8, n=30），候选实际 +0.04~0.08 → 永远淹在噪声里。
3. **瓶颈在 generation 层**：rep5-fixed 证明候选信号与时间漂移不可区分；flash 排除"模型能力"解释。根因 = patcher 从"修单题"出发，不携带跨题通用决策过程。

**11-23 的 AIME26（+0.026）没推翻 11-05 的饱和判断，而是用第三个数据集第三次确认它**。两个模型、18 天跨度、同一模式：in-sample 显著 / held-out ~+0.02 不显著 / 衰减 75~80%。

### 3.5 放弃 AIME（2026-11-24）
三重饱和后，AIME 改作"回归测试 / 第二关卡基准"，精力转向。**这触发了 LiveMath 重评估**（§2.5）。

---

## 4. 两条线的命运对照（论文核心叙事）

| 维度 | LiveMathBench | AIME |
|------|---------------|------|
| 有无捷径 | **有**（meta-option） | 无 |
| 增益来源 | 捷径放大（+16.1pp） | 答案形式转换（+0.100 in-sample） |
| held-out | （旧 split，未用 held-out 门控） | +0.019/+0.026，衰减 74~81% |
| 饱和机制 | 捷径吃光 + other 无 headroom | 异质任务 + 检出下限 > 效应量 |
| 验证手段 | Step6 形式证 + P8/P9 语义证 | 8+ run 两模型三数据集 |
| 诚实门控价值 | **能识别并量化捷径占比**（方法论卖点） | **正确拒异质候选**（不造假增益） |
| 论文定位 | "捷径发现 + 诚实门控"案例 | "有效性边界 = 失败模式跨题可泛化性" |

**共同根因**：单全局 skill 文本进化只在 (a) 有结构捷径可吃 或 (b) S₀ 已覆盖形态级时有效；在需真实推理且 S₀ 已饱和处打不过 S₀。两条线先后证伪同一范式，但从相反方向——LiveMath 证伪"增益全是捷径"，AIME 证伪"无捷径则无泛化"。

---

## 5. 当前未决问题（本次交付包交 codex 审的焦点）

LiveMath 新机制移植后 smoke 的 **10 候选 0 晋升**，暴露了一个此前未充分审视的测量问题：

**bootstrap CI_lower>0 作为唯一晋升门，在小样本（not_meta_n=10, σ_pair≈0.044）上是否系统性偏严？**

证据：
- iter1/iter2 共 10 候选，全部 CI 跨零被拒，但多个候选 delta 方向正向（Option Comparison 两轮 +0.023/+0.016，expand 两轮 +0.014/+0.002）。
- n=10 时 CI>0 需 mean delta≈0.04+，真实 +0.02 信号注定跨零——这是统计功效问题，可能不是"无效应"。
- Theorem-Level Precision iter1 -0.011 → iter2 +0.025，单轮 delta 方向都翻，说明单轮噪声极大。

**我们的当前判断（供反驳）**：CI 防误晋升目的对，但单点否决偏严；建议加"多轮方向一致"轨迹信号层，而非放宽 CI。详见 `review_requests/20261125_codex/REVIEW_REQUEST_20261125.md` 审点 A。

---

## 6. 关键文件索引

| 文件 | 内容 |
|------|------|
| `memory/livemath-saturation-synthesis-20260924.md` | LiveMath 旧饱和合成（捷径三证数据） |
| `memory/aime-port-mcq-coupling-finding-20260930.md` | MCQ 机制耦合发现（为何机制锚定捷径） |
| `memory/aime-saturation-closeout-20261105.md` | AIME 三重饱和 + flash 全流程数字 |
| `memory/aime-progress-20261117-20261124.md` | AIME 周报（防截断 6 路径 + 准确率线） |
| `memory/livemath-reassessment-20261124.md` | LiveMath 重评估（含修正说明） |
| `memory/livemath-newmech-port-exec-20261124.md` | 新机制移植执行记录（代码 delta + 零回归验证） |
| `memory/livemath-newmech-smoke-launched-20261125.md` | smoke 启动 + 3 个坑（含 deadline 修复） |
| `docs/truncation-defense-final-report-20261123.md` | 防截断线终版报告 |
| `docs/gate_logic_walkthrough_20261125.md` | 门控逻辑代码导读（审点 A 用） |
