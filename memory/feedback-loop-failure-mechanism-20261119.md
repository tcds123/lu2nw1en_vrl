---
name: feedback-loop-failure-mechanism-20261119
description: 196 次截断未成功反馈的根因 — budget section 反生产力 (新增截断+搞坏答对题), 非"信号不足"非"渲染差"
metadata:
  type: project
---

# 反馈环失败机制 (2026-11-19, run_20261119_140908 复核)

**用户问题**: run3 全程 196 次截断 (7.92%), budget focus 命中 3 次, 为何 0 个 budget section 过门控?

**结论**: 根因不是"信号不足"(我之前天花板分析的前提错了 — 196 次信号充足) 也非"渲染差"(#87 增量 B), 而是 **budget section 在 cap=8192 下反生产力** — 它不降反升截断数, 还把原本答对的题搞成截断 (0 分), 候选 delta 被净拖负 → 永远过不了 CI>0 门控。

## 完整反馈环 (10 环, 断在环 8)

```
1. 截断发生 (trace length=196)           ✅
2. harness.n_truncated 记录              ✅ (retry=0 修复后)
3. collect_failures 组装 failure.harness  ✅
4. EFL.classify → budget_signal → F11    ✅ (cnt=8 actionable)
5. counts 含 "Budget Exhausted"          ✅
6. expand 选 budget (命中 3/4 轮)        ✅
7. LLM 生成 budget section               ✅ (4 个: Rapid-Exit/EarlyCommit/EscapeHatch/ForcedFinalAns)
8. eval 该候选: 截断题答对了吗?          ❌❌❌  ← 反馈失败在这
9. delta/CI 计算                          → delta 振荡 +0.011→+0.056→-0.022, CI 跨 0
10. 门控 admitted?                       ❌ 0/4 admitted
```

## 环 8 的决定性证据 (expand_val trace 逐 call 重建)

**所有 32 次 expand_val 截断调用, 0 次产出 `<answer>` 标签 → 截断 = 必错 (0 分)**:

| 候选 | budget section | 截断题数 | 截断题有`<answer>` | 非截断有`<answer>` |
|------|---------------|---------|-------------------|-------------------|
| iter1 | Rapid-Exit Heuristics | 5 | **0** | 85/85 |
| iter2 | (Missing Final Answer) | 7 | **0** | 83/83 |
| iter3 | Early Commitment | 6 | **0** | 84/84 |
| iter4 | Escape Hatch | 14 | **0** | 76/76 |

→ budget section **没救回任何一个截断题** (截断题 0 分, 与 issue #331 截断审计一致)。

## 更致命的发现: budget section 反生产力 (截断不降反升)

同题对比 S₀ baseline (4 题截断) vs 各 budget 候选:

| 候选 | 截断题数 | 与 S₀ 重叠 | 新增截断题 |
|------|---------|-----------|-----------|
| S₀ | 4 | — | — |
| iter1 Rapid-Exit | 4 | 1 | **+3** |
| iter3 EarlyCommit | 6 | 2 | **+4** |
| iter4 EscapeHatch | **10** | 3 | **+7** |

budget section 加上后, 截断题从 4 扩到 10 (iter4)。**新增截断题里有 3 个 (aime24:66/78/85) 在 S₀ 下原本答对且不截断, 被 budget section 搞成截断 0 分**:

```
aime24:66  S₀ pass=0.333 trunc=0  → budget候选截断0分  ❌搞坏
aime24:78  S₀ pass=0.333 trunc=0  → budget候选截断0分  ❌搞坏
aime24:85  S₀ pass=0.333 trunc=0  → budget候选截断0分  ❌搞坏
```

→ budget section **净负**: 0 救回 (截断必错) + 3 搞坏 (答对变截断) = delta 被拖 -0.1 量级。这解释了 iter4 budget 候选 delta=-0.022 (负值)。

## 为何 budget section 会增加截断 (机制推测, 待验证)

"Early Commitment / Escape Hatch / Rapid-Exit" 措辞可能改变模型输出分布:
- 鼓励"先 commit 再说"→ 模型可能更早开始写正式解而非简略探索 → 输出更长 → 更易截断
- 或让模型误以为可以放开推导 → 冗长化
- 关键: 这些 section 没有提供"如何在 8192 token 内收尾"的可操作策略, 只是抽象口号

## 为何反馈环仍"诚实工作"却不收敛

机制本身没断 (环 1-7 全通, expand 命中 3 次), 但环 8 的负反馈让 budget 候选 delta 永远 ≤ ~0 → 门控诚实拦下 (这是正确行为, 不是 bug)。系统在**诚实地拒绝一个反生产力的方向**。问题在生成端 (LLM 产出的 budget section 质量差/方向错), 不在评估端。

## 对下一步的修正 (推翻 budget-focus-ceiling-analysis-20261119 的部分结论)

- 我之前说"瓶颈是 tr 天花板, 建议降 cap" — **部分错**: 降 cap 提高 tr 会让 budget section 更反生产力 (更多题被搞成截断), delta 可能更负, 更难过门控
- **真正瓶颈是 budget section 生成质量** (方向反了): LLM 产出的"早 commit"类策略在 cap 约束下增加截断。需要的是**真正缩短输出**的策略 (如"分阶段预算: 探索≤30% tokens, 验证≤20%, 余量收尾"), 而非"早 commit"
- **#87 增量 B (增强渲染) 仍是对的方向**, 但要渲染的不是"截断多严重", 而是"如何在该题 token 预算内完成"的可操作策略 — 让 LLM 看到截断的具体残文 + 该题的 token 预算, 产出"压缩策略"而非"commit 口号"

## 建议 (仅建议, 不改代码)

1. 验证方向 1 (Quantity Parsing +0.100 的 AIME25 泛化) 仍最优先 — 它是已验证的正向进化
2. 若推进防截断: 不降 cap, 而是**改 expand prompt 让 LLM 产出"压缩/收尾策略"而非"早 commit 口号"**; 或在 expand_val 前加**截断守门** (候选 n_truncated > parent 则直接拒, 不浪费 CI 计算)
3. 可做对照: 同 cap 下, 人工写一个"压缩策略" budget section (如 "token budget: 探索≤2k, 计算≤2k, 答案≤1k, 超 6k 立即停止推导转写答案"), 看 delta 是否为正 — 若是, 证实"生成方向"是瓶颈

## 关联

[[trunc-efl-bgt-dual-evolve-launch-20261118]] — run3 结果; budget 候选 delta 振荡 +0.011→+0.056→-0.022 正是反生产力的体现
[[budget-focus-ceiling-analysis-20261119]] — 我之前的天花板分析, **本节部分推翻** (降 cap 建议有误, 会加重反生产力)
[[truncation-audit-20261116]] — "截断必错" 经本次再次验证 (32/32 截断 0 个 answer)
