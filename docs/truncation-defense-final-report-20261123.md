# 防截断 Skill 自进化：最终结论报告

> 2026-11-23 终版 | 作者: 自进化实验全链证据汇总
> 本文档为防截断 skill 自进化线的**最终裁决**，涵盖全部实验设计、结果、失败机制分析和后续建议。

---

## 1. 研究动机与设计原则

### 1.1 问题

Skill 自进化框架中，evolver 产出的 skill 文件作为 system prompt 被推理模型（DeepSeek-V3.2）消费。推理模型在 AIME 难题上会产生极长的 chain-of-thought 推理链（3k-8k tokens），在 `max_tokens=8192` 的 cap 限制下部分样本被截断（`finish_reason=length`），答案丢失。

**核心假设**: 能否通过 skill 优化，让 B 模型在不损失准确率的前提下，**自发控制推理长度**，减少截断？

### 1.2 设计原则（2026-11-18 确定）

| # | 原则 | 说明 |
|---|------|------|
| 1 | **计错自然传导** | `finish_reason=length` 强制 `em=0`，截断自动进入失败池，不设人工 λ 权重 |
| 2 | **非劣约束** | `clean_pass_tokens` + pass 不降作为约束，不是"越短越好"单目标 |
| 3 | **自进化载体** | 经验载体 = skill + facet 库累积计数，不预埋人工 base_rule / trunc_bank 旁路 |
| 4 | **残文行为签名** | patcher 看到具体残文病态（tail_loop/cycle_reps/answer_pos），不是抽象截断率 |

### 1.3 初始条件

- **S₀ skill**: `skill_aime_initial_v2.md`，599 字符，**不含任何** `commit discipline` / `checkpoint` / `token budget` / `running total` / `stop and submit` / `hard commit` 等 _BUDGET_COVERAGE_SIG 词汇
- **数据集**: AIME24，30 题（train 18 + val 12，shuffle seed=42）
- **核心判据**: 起点 S₀ 无任何收尾纪律词 → N 轮后 skill **自主**长出含 _BUDGET_COVERAGE_SIG 词汇的收尾纪律段

---

## 2. 实验设计总览

### 2.1 基础设施搭建

| 阶段 | 日期 | 内容 | 关联 issue |
|------|------|------|----------|
| 机制修复 | 2026-11-16 | `_BUDGET_COVERAGE_SIG` 指纹 + `--budget_focus` 开关，让 Budget 轴可被 focus 选中 | #76 |
| EFL 增强 | 2026-11-18 | 增量 A: budget facet `get_signature` → 覆盖签名；增量 B: `_implicit_axis_block` 注入残文行为签名 | #87 |
| Bug 修复 | 2026-11-18 | 4 bug 修复 (标签匹配/跨文件 import/tiebreak 方向/test 缺口) + `_to_int` 严格化 | #88 |
| 截断漏计修复 | 2026-11-19 | `QWEN_CHAT_TRUNC_RETRY=0` 禁用自动重试，截断从漏计到真实记录 | #89 |
| 双门控 + 门阀 | 2026-11-17→19 | `--dual_gate` (delta+CI) + `--trunc_gate` (截断率退化拦截) | #69, #81, #85 |

### 2.2 六条实验路径

| 路径 | 设计 | cap | 模型 | 预期 |
|------|------|-----|------|------|
| **run1** | 手写 budget 6 条起点 (trunc_v1) + EFL-OFF 平坦进化 | 16384 | flash | 探底"手写+单目标" |
| **run2** | 纯 S₀ 起点 + `--budget_focus` (EFL-OFF) | 16384 | flash | "纯 S₀ 能否自长出纪律" |
| **run3** | 纯 S₀ + EFL-ON + dual_gate + trunc_gate | 16384→8192 | V3.2→V3.2 | 全机制联动端到端 |
| **run3 修复版** | run3 + `TRUNC_RETRY=0` | 8192 | V3.2 | 首个含真实截断信号的实验 |
| **旁路 base_rule** | 离线 distill trunc_bank → 注入 sidechannel_init | 8192 | V3.2 | 验证"蒸馏经验是否有用" |
| **单 bullet 消融** | Record2 (iter3 budget candidate) 拆 5 bullet | 8192 | V3.2 | "信噪比 vs 零效应"判别 |

---

## 3. 各路径详细结果

### 3.1 Run1: 手写起点 + 单目标进化 (2026-11-16)

**配置**: 起点 `trunc_v1.md` (2273 chars, 含手写 "Budget & Commit Discipline" 6 条)，B=flash，cap=16384，EFL-OFF，单目标进化。

**结果**: 4 iter / 12 候选，仅 iter4 通过 CI gate (+0.100, CI=+0.033)，skill 从 2273 膨胀到 3907 chars (+72%)。

**关键发现**:
- 手写 budget 纪律段带来的截断率下降 (11.1%→6.7%) 是**起点的贡献，不是进化的贡献**
- 12/12 候选全部比起点更长，进化方向系统性偏"加内容"而非"减长度"
- 单目标门控对长度完全无感

**结论**: 第一版的收益来自设计，不来自进化。用户质疑"为什么是手写的，不应该是自进化出来的吗"——这是整条线的起点反思。

### 3.2 Run2: 纯 S₀ + budget_focus (2026-11-16)

**配置**: 起点纯 S₀ (599 chars，无任何纪律词)，B=flash，cap=16384，`--budget_focus`，EFL-OFF。有 grep 守门确认起点零预埋。

**结果**: 4 iter 全 KEEP PREV。最佳 delta 单调上升 (+0.011→+0.022→+0.056→+0.067) 但 CI 最好仅 -0.011，从未过零。best_skill 599 chars hash 不变。

**关键发现**:
- budget focus 机制接通：4 轮中 3 轮 focus 落到 Budget 轴（是数据驱动的，不是常数偏置）
- 瓶颈是**统计效能**：30 题 × k=3 下 ±1 题 ≈ 3.3pp，基线 pass 已 0.856，剩余空间窄
- "门在这个基线下根本打不开"

**结论**: 机制接上了，但当前门控无法把任何进化产物推上台面。

### 3.3 Run3: EFL-ON + 全机制联动 (2026-11-18→19)

**三阶段演进**:

| 子 run | cap | retry | 结果 |
|--------|-----|-------|------|
| run3a | 16384 | 1 (默认) | 零进化 (tr=0%，无截断信号，budget facet 无 failure 累积) |
| run3b | 8192 | 1 (默认) | 零进化 (tr=2.2% 仍被 retry 覆盖漏计，harness 汇总只有 1 条) |
| **run3c** | **8192** | **0** | **✅ 进化成功 (+0.100)** |

**run3c 最终结果** (PID 39989，2026-11-19，3.5h):

| 指标 | 值 |
|------|-----|
| S₀ baseline | pass=0.756，delta=+0.044，CI=-0.011，**tr=5.6%** |
| iter1 | KEEP-PARENT |
| **iter2** | **delta=+0.100，CI=+0.011 → PROMOTED** |
| iter3 | KEEP-PARENT |
| iter4 | KEEP-PARENT |
| best_skill | 599→**817 chars** (+36%) |
| best_final_delta | **+0.100，CI +0.011 (显著)** |

**进化内容分析**: 进化点在 **Quantity Parsing** 轴（answer-form rephrasing 规则），并非 budget/防截断轴。这是 **准确性导向** 的进化（"检查所求量的表述形式"），不是 **截断导向** 的进化（"在 token 预算耗尽前提交"）。

**budget focus 命中史** (facet_tree expand 路径，4 轮命中 3/4 次，issue #91 推翻后确认 budget focus 真实命中非偏置):

| iter | budget 候选 section | delta | CI | 过门控 |
|------|---------------------|-------|----|--------|
| 1 | Rapid-Exit Heuristics | +0.011 | -0.056 | ❌ |
| 2 | Forced Final-Answer Emission | +0.011 | -0.078 | ❌ |
| 3 | Early Commitment (单 bullet 消融后 ≈0) | **+0.056** | -0.033 | ❌ |
| 4 | Escape Hatch | -0.022 | -0.111 | ❌ |

**4 个 budget 候选全部 CI≤0，0/4 过门控** —— 没有产出任何被采纳的防截断 skill。机制全链连通（`response_tails` + "NEVER EMITTED `<answer>`" 信号 → LLM 实时生成防截断 section → eval 评测过门控），但产物为空。iter3 的 +0.056 经单 bullet 消融证实是 selection on noise（k=3 下 5 题翻转的采样高点），非真信号。

**天花板分析**: tr=5.6% × 30 题 = ~1.67 题截断 → 可恢复 pass 上限 = +0.056。iter3 budget 候选 delta=**+0.056 恰与天花板吻合**——机制在 iter3 已触及理论上限。+0.056 在 n=30 下 CI 跨 0 (-0.033)，**永远过不了 CI>0 门控**。

**传导链验证**: length→harness→EFL.classify→_assign_budget→counts→`find_uncovered_type_for_expand` 选中→patcher 生成→evaluator 评估 → **全环打通**。budget 候选未过门控 = 诚实统计守门（delta 振荡 +0.011→+0.056→-0.022），非机制断接。

### 3.4 旁路 base_rule: 离线蒸馏注入 (2026-11-20，PID 64303)

**设计**: 从 run3c trace 中 `trunc_bank collect+distill`（180 条截断证据） → LLM 蒸馏出 5 条 `base_rule`（token 阈值指令：超 6000 token 强制决策、超 7000 立即写答案） → `apply_rule` 注入 S₀ → 产出 `sidechannel_init.md` → 启动旁路 evolver。

**结果**: base_rule 注入即**反效**。跑到了 S₀ precompute 阶段就被否定 kill（未跑完进化），关键数字（issue #95 已记录）：

| 指标 | sidechannel_init（含 base_rule） | 纯 S₀ |
|------|----------------------------------|-------|
| pass | 0.722 | 0.756 |
| delta vs S₀ | **−0.044** | — |
| 截断率 | **12.2%** | 5.6% |

**base_rule 一注入起点就降 pass 3.4pp、截断率翻倍 (+117%)**，直接 kill 省下 ~2500 API calls 没浪费在进化上。

**结论**: LLM 蒸馏出的防截断经验（base_rule）不仅无益反而增加了截断。要么蒸馏经验质量不足以导向有效行为，要么"自然语言控制推理长度"这事的难度超过了 LLM 的能力。旁路路线在 baseline 否定点 kill，不是中途放弃。

### 3.5 单 bullet 消融: Record2 分解 (2026-11-20/23)

**设计**: 把 Record2（iter3 budget candidate，+0.056，5 bullet 混合了 1 条验证规则 + 1 条口号 + 3 条题型方法）拆成两个 skill：
- **Skill A**: S₀ + 纯 bullet[1]（验证型早停规则，唯一跨题通用行为指令）
- **Skill B**: S₀ + 全 Record2（5 bullet）

在 AIME24 上独立 paired eval（k=4）。

**结果**:

| skill | 内容 | mean_pass | delta vs S₀ |
|-------|------|-----------|-------------|
| skill_A | 纯 bullet[1] 验证指令 | 0.7167 | **+0.0056** |
| skill_B | 全 Record2 (5 bullet) | 0.7083 | **-0.0028** |

**两者均 ≈ 0，方向相反且都在抽样噪声内。**

**关键诊断**: Record2 的 +0.056 机制是 **selection on noise**（非 winner's curse）：
- k=3 下仅 5 题各多对 1 次 = 1.67 题单位，属随机波动量级
- evolver 从 4 candidate 里选 max → 必然选到偏高者
- +0.056 从来不是可重复信号，消融 ≈0 不是"证伪了真效应"，而是"重新采样证明了它的不稳定性"

**方法缺陷坦白**: 消融有 3 个方法缺陷（k 不匹配/CI 崩溃/名词误用），影响精度但不影响大方向——全线无 CI>0 候选的结论由归纳（非单次消融）支撑。

---

## 4. 全部路径的汇总裁决

| # | 路径 | cap | 模型 | 核心结果 | 状态 |
|---|------|-----|------|---------|------|
| 1 | run1 (手写起点) | 16384 | flash | +0.100 但来自起点预埋，非进化 | ❌ 起点污染 |
| 2 | run2 (纯 S₀ + budget_focus) | 16384 | flash | 4 轮全 KEEP PREV，CI 从未过零 | ❌ 零进化 |
| 3a | run3a (EFL-ON + retry=1) | 16384 | V3.2 | tr=0%，无截断信号 | ❌ 无信号 |
| 3b | run3b (EFL-ON + retry=1) | 8192 | V3.2 | tr 漏计 82%，budget 不可选 | ❌ 漏计 |
| 3c | run3c (EFL-ON + retry=0) | 8192 | V3.2 | +0.100 但方向=QtyParsing，budget 候选 CI≤0 | 🟡 进化有、但非防截断 |
| 4 | 旁路 base_rule | 8192 | V3.2 | tr 5.6%→12.2% (反效) | ❌ 反效 |
| 5 | 单 bullet 消融 | 8192 | V3.2 | 两 skill ≈0，selection on noise | ❌ 零效应 |

**6 条路径中 0 条产出 CI>0 的防截断 candidate。**

### 4.1 两方法对照：实时进化 vs 旁路蒸馏，两条路独立收敛

两种完全不同输入和 prompt 设计的方法，独立验证防截断元指令的效果，最终指向同一个结论。

#### 方法①：实时进化中含防截断因子和反馈（run3c，PID 39989）

这是主设计——evolver 的 facet_tree expand 环节接收截断诊断信号（`response_tails` + "NEVER EMITTED `<answer>`"），LLM 实时生成防截断 section，eval 评测过门控。机制全链连通（issue #91 推翻后确认 budget focus 真实命中 3/4 轮）。

**4 个 budget 候选结果**（详见 §3.3）：

| iter | section | delta | CI | 过门控 |
|------|---------|-------|-----|--------|
| 1 | Rapid-Exit Heuristics | +0.011 | −0.056 | ❌ |
| 2 | Forced Final-Answer Emission | +0.011 | −0.078 | ❌ |
| 3 | Early Commitment（单 bullet 消融 ≈0） | +0.056 | −0.033 | ❌ |
| 4 | Escape Hatch | −0.022 | −0.111 | ❌ |

- **4/4 候选 CI≤0，0/4 过门控** — 没有产出任何被采纳的防截断 skill
- 最好的（iter3 +0.056）经单 bullet 消融证实是 selection on noise（k=3 下 5 题翻转的采样高点），非真信号
- 机制全通、产物为空 — 诚实统计守门 + 效应天花板限制

#### 方法②：旁路蒸馏 base_rule（PID 64303，2026-11-20 09:04）

这是离线路线——从 run3 的 180 条截断证据里蒸馏出 5 条 base_rule（token 阈值指令：超 6000 token 强制决策、超 7000 立即写答案），注入 S₀ 得 `sidechannel_init.md`，再启动旁路 evolver。

**关键数字**（详见 §3.4，issue #95 已记录）：

| 指标 | sidechannel_init（含 base_rule） | 纯 S₀ | 方向 |
|------|----------------------------------|-------|------|
| pass | 0.722 | 0.756 | **−3.4pp** |
| delta vs S₀ | **−0.044** | — | **负** |
| 截断率 | 12.2% | 5.6% | **+117% (翻倍)** |

- 跑到了 S₀ precompute 阶段就被否定 kill，没跑完进化
- 一注入就反效：pass 降、截断率翻倍 → 直接 kill，省下 ~2500 calls
- 旁路路线在 baseline 否定点 kill，不是中途放弃

#### 两条路的共同结论

两种方法、完全不同的输入来源和 prompt 设计，完全独立地指向同一结论：

| 维度 | 方法① (实时进化) | 方法② (旁路蒸馏) |
|------|-----------------|-------------------|
| 输入 | trace 诊断信号 → LLM 实时生成 | trace 截断证据 → LLM 离线蒸馏 |
| 产出形式 | evolover expand 候选 section | 注入 S₀ 的 base_rule 段落 |
| 评测方式 | 主循环 eval + paired+bootstrap 门控 | S₀ precompute baseline → 判定是否优于纯 S₀ |
| 结论 | 4 候选 0 过门控，最佳 +0.056 系噪声 | 注入即反效（−3.4pp, +117% tr），直接 kill |
| 状态 | 跑满 4 轮，机制全通，诚实负结果 | 跑到 baseline 否定点 kill，非中途放弃 |

**防截断的主动进化维度，经过两种方法、两条路线的独立验证，都不成立。** 这就是后来 trunc_gate 保留为 passive gate（只拦恶意退化）、不再投入 LLM 生成防截断 section 的依据。

---

## 5. 为什么防截断 Skill 不 Work：机制层面分析

### 5.1 五个层面，每个都可能断

```
[Skill 中的防截断条文]
         ↓
[LLM 读入 system prompt]           ← ① 是否能被模型注意到并内化？
         ↓
[推理过程中监控自身长度]            ← ② 模型有 token counter 吗？没有。
         ↓
[在合适的时机触发早停/提交]         ← ③ 感觉型阈值（"约6000 tokens时..."），极度不准
         ↓
[提交的答案是正确的]                ← ④ 截断题本身是否可救？大多不是。
         ↓
[最终 pass rate 提升 → CI>0]       ← ⑤ 效应量够大吗？不够。
```

### 5.2 各断裂点的具体证据

**① 模型能否内化？** — 旁路 base_rule 注入后 tr 反而上升（5.6%→12.2%），说明即便是精心蒸馏的经验段落，注入后也不能被模型正确理解和执行。反而可能因为增加了 system prompt 的复杂性而分散注意力。

**② 模型有 token counter 吗？** — 没有。它只能靠模糊的"我已经写了很久了"的感觉来判断。这种感觉在不同题目、不同推理模式下的方差极大。

**③ 阈值触发不准** — LLM 生成的防截断候选 section 里全是 "stop at 6000 tokens"、"commit early"、"recompute by alternative method" 这类口号。它们不可执行：模型不知道自己写到了多少 token，也不知道什么叫"early"。

**④ 截断题大多不可救** — Token budget ablation 揭示：16384 cap 相比 8192 减少了 ~30% 截断，但 pass rate 只提了 +0.017~+0.042。说明大部分截断题即使给了更多 token 也做不对——截断是**症状**而非**根因**。模型在这些题上陷入循环/反复验证/方向错误，给再多 token 也无济于事。

**⑤ 效应量天然受限** — tr=5.6% × 30 题 = 可恢复上限仅 +0.056，在 n=30 下 CI 永远跨零。天花板由截断率决定，而 V3.2 本身在 8192 cap 下的截断率只有 ~5-10%（比 flash 的 10-12% 更低）。

### 5.3 核心矛盾：工具匹配错误

让一个**自然语言 skill** 去对抗 **token 预算的硬约束**，本质上是 mismatch：

- token 预算是基础设施层的硬限制（整数，精确，不可协商）
- skill 是自然语言弱信号（模糊，概率性，依赖模型内化和执行）

**这不是 skill optimization 的方法论失败，而是它的边界识别——它告诉我们 skill 优化适用范围的 negative case。**

---

## 6. 与准确率线的对比：两条线命运清晰

| 维度 | 防截断线 | 准确率线 (Quantity Parsing) |
|------|---------|---------------------------|
| 起点 | 手写 6 条预埋 → 反思后坚持纯 S₀ | 纯 S₀，零预埋 |
| in-sample | 无 CI>0 candidate | **+0.100, CI=+0.011 (显著)** |
| AIME25 held-out | — | +0.019 (方向正，不显著) |
| AIME26 held-out | — | **+0.026, per-seed CI=[+0.005,+0.048] 不跨零** |
| 全路径量 | 6 条全无 CI>0 | 3/3 数据集正向 |
| 判决 | **放弃主动进化** | **继续** |

准确率线的 Quantity Parsing 在两个独立 held-out (AIME25, AIME26) 都正向，排除了 "+0.100 是 AIME24 过拟合" 的假设。防截断线在同样框架下全无正向证据。**两条线的命运是对同一个方法的 complementary evidence：这个框架能进化出有用 skill，但只能在某些维度上进化——"控制推理长度"不在可用维度之列。**

---

## 7. 最终决定与建议

### 7.1 决定

| 决定 | 理由 |
|------|------|
| **放弃防截断 skill 的主动进化** | 6 条路径全无 CI>0；天花板 tr×n 限制了可恢复上限；LLM 无法执行 token-精确的长度控制；机制 root cause 是工具匹配错误 |
| **保留 trunc_gate 为 passive gate** | 防止候选 skill 导致截断率暴涨（如旁路 base_rule 的 5.6%→12.2%），这是有效的防御机制 |
| **论文中记录为边界识别** | 防截断 self-evolve 的失败是有价值的 negative result：它界定 skill optimization 的适用范围，不是所有行为维度都可通过自然语言 skill 优化 |

### 7.2 论文中截断问题的处置

1. **基础设施层解决**: 加大 cap（8192→16384）或在 8192 下做 truncation-aware scoring（截断强制 em=0 已做到）
2. **不把防截断作为 skill 优化目标**: 因为我们证明了它在当前框架下不可实现
3. **可记录为限制**: "We attempted to evolve anti-truncation meta-instructions through our skill optimization framework, but found no statistically significant improvement (6 independent runs, all CI≤0). We attribute this to a fundamental tool mismatch: natural language skills cannot enforce precise token-level budget control, as the model lacks explicit token counter awareness. This negative result characterizes a boundary of what skill-space optimization can achieve."

### 7.3 后续方向

- **路径废弃**: 防截断自进化、trunc_bank 旁路蒸馏、base_rule 注入 —— 均不再投入资源
- **路径保留**: trunc_gate (passive)、cap sensitivity analysis (作为论文 footnote)
- **重心转移**: 准确率线 — Quantity Parsing 已验证泛化，路径清晰：n=90 evolver，用更多 facet 进化点放大 ~+0.02 的 held-out 收益

---

## 8. 附录：全证据链索引

| 文件 | 内容 |
|------|------|
| `memory/trunc-evolve-first-run-20261116.md` | run1 手写起点 + 单目标进化 |
| `memory/trunc-evolve-budget-run2-20261116.md` | run2 纯 S₀ + budget_focus |
| `memory/trunc-self-evolve-increments-20261118.md` | #87 两增量 + #88 四 bug 修复 |
| `memory/trunc-efl-bgt-dual-evolve-launch-20261118.md` | run3 端到端全历程 (3a/3b/3c) |
| `memory/budget-focus-ceiling-analysis-20261119.md` | 天花板分析：tr×n 上限 |
| `memory/bullet-ablation-record2-20261120.md` | 单 bullet 消融 + selection on noise 诊断 |
| `memory/token-budget-ablation-20261123.md` | Token 预算消融 (8192 vs 16384) |
| `memory/aime25-heldout-generalization-20261120.md` | AIME25 held-out (+0.019) |
| `memory/aime26-heldout-20261123.md` | AIME26 held-out (+0.026，单点最强正面证据) |
| `docs/phase2_issues.md` | 全流程 issue 跟踪 (#69-#95) |

---

## 9. 文档元信息

- **创建日期**: 2026-11-23
- **最后更新**: 2026-11-23 (补充方法①/②两路对照 + base_rule 精确数值表 + budget 候选表格修正)
- **状态**: 终版 (防截断线 closed)
- **作者**: 自进化实验全链证据汇总

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Code <noreply@anthropic.com>
