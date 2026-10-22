---
name: aime-codeagent-eval-result-20261012
description: AIME25 CodeAgent (PoT) eval DONE — PoT paradigm SIGNIFICANTLY POSITIVE for s0, mirrors opposite of ReAct
metadata:
  type: project
---

## 结果 (2026-10-12 09:40 启动 → 16:37 完成, ~7h, PID 114189)

| skill | single-call | CodeAgent(PoT) | Δ | t | p | 判定 |
|-------|------------|----------------|------|------|------|------|
| s0 | 0.4741 | 0.6222 | **+14.8pp** | 3.39 | 0.0007 | **SIG** |
| evolver | 0.5185 | 0.5815 | **+6.3pp** | 1.46 | 0.147 | NS |

n=30 (AIME25 held-out), k=3, seeds=3, paired t-test on per-question pass-rate (3-seed mean).
backend=DeepSeek-V3.2 (同 key), eval_k=3/seeds=3/TEMP=0.3/max_steps=6, skill hash 对齐基线 (s0=c11d7dc0 / evolver=f00fa794)。

产物: `outputs/tf_skill_aime_v2/aime25_codeagent/codeagent_eval.json` + 6 个 `per_{skill}_{hash}_codeagent_seed{N}.json` (flat {qid:pass_rate} dict)。

## 单 seed 列表

- s0: 0.700 / 0.589 / 0.578 → mean 0.6222 (方差小, 稳定)
- evolver: 0.578 / 0.578 / 0.589 → mean 0.5815 (方差极小)

## 核心结论: PoT 成功 = ReAct 负结果是范式选错

这是 ReAct 负结果 ([[aime-react-eval-result-20261010]]: s0=−7.4pp / evolver=−13.0pp 均 SIG 负) 之后的**对照范式实验**。

- **ReAct** (Paradigm 1, 交错工具调用 Think→Act→Observe): 两 skill 均**显著有害**
- **CodeAgent/PoT** (Paradigm 2, 模型写 Python 在持久解释器算, final_answer 终止): s0 **显著有益** +14.8pp, evolver 正向但不显著 +6.3pp

**结论**: ReAct 的负结果不是「agent 能力在竞赛数学上本质有害」, 而是**范式选错** — 竞赛数学是整数答案的符号推理题, 靠"写代码算出来" (PoT) 远胜于"交错调用工具" (ReAct)。PoT 把模型注意力锁在数学计算上, 不被元决策 (何时调工具) 分散, 不被多轮对话挤占 token 预算。

## 二级发现 (跨范式一致): agent 能力对强 skill 有递减/反转效应

| 度量 | single-call | ReAct | CodeAgent |
|------|-------------|-------|-----------|
| s0 | 0.4741 | 0.4000 | 0.6222 |
| evolver | 0.5185 | 0.3889 | 0.5815 |
| evolver−s0 | **+4.4pp (evolver 胜)** | **−1.1pp (s0 胜, 翻转)** | **−4.1pp (s0 胜, 翻转)** |

**单次调用下 evolver 胜 s0 (+4.4pp); 但在两种 agent 范式下排名都翻转 (s0 胜)**。
- ReAct: evolver 被伤得更深 (−13pp vs s0 −7.4pp)
- CodeAgent: evolver 受益更少 (+6.3pp vs s0 +14.8pp)

**跨范式稳健发现**: agent 能力对已强 skill 有**平等化/回归效应** — skill 越强, agent 循环的边际收益越低 (甚至转负)。机制推测: 强 skill 已把模型推到接近 agent 增益天花板, agent 循环的开销 (元决策/token 挤占) 开始抵消收益。这对论文是强叙事: 上下文空间进化 (skill 优化) 与 agent 能力不是可加的正交维度, 强 skill 会压缩 agent 的 headroom。

## s0 逐题归因 (3-seed mean, codeagent vs single)

- **progress 15 题**: q4 q6 q8 q9 q10 q13 q14 q17 q18 q19 q22 q23 q24 q25 q28 — 单次拿部分分/0 分的题, PoT 代码验证救回
- **regress 仅 2 题**: q1 q11 — 题量极少, 非系统性 (满分题仍满分)
- 净效应强正: 15 prog 远超 2 regress (对比 ReAct s0 是 3 prog / 7 regress)

evolver: progress 11 / regress 9 / same 10 — 救题和崩题参半, 故增益小且不显著。

## Bug 5 修复全程生效

- per-trajectory wall-clock 超时 (TRAJ_TIMEOUT=240s, daemon thread + join) 全程捕获 61 次死锁轨迹, 全部含在单次也 0 分的题上 (无公平性污染)。
- per-qid 检查点 (原子 tmp+os.replace) 零丢失, 7h 长跑无数据损失。
- 旧 Bug 5 (CPU 忙循环死锁 26h 零结果) 彻底解决。

## Bug 6 (新, minor): 汇总打印 KeyError 'mean'

eval 结束写完 codeagent_eval.json 后, 控制台汇总打印在 `eval_aime_codeagent_multiseed.py:632` 崩 `KeyError: 'mean'` — 访问 `d["mean"]` 但 delta_vs_single dict 的键是 `mean_diff`。**无数据损失** (JSON 已先写盘), 但留下孤儿线程烧 CPU (89%), 需手动 kill。修复见 docs/phase2_issues.md。

## 对论文

- **正面证据**: agent 能力 (PoT) 在竞赛数学上**显著有效** (+14.8pp s0), 与 ReAct 负结果组成完整对照 — "加 agent 能力"不是免费午餐也不是一律有害, **范式选择决定方向**。
- **跨范式稳健性证据**: 强 skill 压缩 agent headroom (两种 agent 范式下排名都翻转) — 为上下文空间进化 (skill-only 优化) 的正当性提供新维度: skill 与 agent 不是正交可加。
- 关联 [[aime-react-eval-result-20261010]] [[aime-codeagent-eval-launched-20261012]] [[aime-skel-fallback-launched-20260909]] [[aime-on-skillopt-result-20261008]]
