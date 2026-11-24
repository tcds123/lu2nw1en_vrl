---
name: aime-react-eval-result-20261010
description: AIME25 ReAct+execute_code agent eval DONE — agent delta SIGNIFICANTLY NEGATIVE for both skills
metadata:
  type: project
---

## 结果 (2026-10-10 14:17 启动 → 15:15 完成, exit 0)

| skill | single-call | ReAct | Δ | t | p | |
|-------|------------|-------|------|------|------|---|
| s0 | 0.4741 | 0.4000 | **−7.4pp** | −2.00 | 0.046 | SIG |
| evolver | 0.5185 | 0.3889 | **−13.0pp** | −3.58 | 0.0003 | SIG |

n=30 (AIME25 held-out), k=3, seeds=3, paired t-test on per-question pass-rate (3-seed mean).

产物: `outputs/tf_skill_aime_v2/aime25_react/react_eval.json` + 6 个 `per_{skill}_{hash}_react_seed{N}.json`

## 公平性 (与 single-call 基线逐字对齐, 唯一变量=agent 能力)

- backend=DeepSeek-V3.2 (同 key), data=AIME25, eval_k=3/seeds=3/TEMP=0.3/max_completion_tokens=4096 全部同基线
- skill hash 对齐基线缓存: s0=c11d7dc0 / evolver=f00fa794
- 答案提取 (`evaluate_aime`/`_build_system`/`_build_user`/`chat_target_messages`) = single-call 基线**复用同一套 import** → 提取逻辑差异不是假阴性来源
- ReAct 循环逻辑 (`run_one_aime_react`): 每 k 采样独立 messages, tool 结果回填, max_steps=6 耗尽则 nudge 一次 final, `evaluate_aime(final)` 判定 — 正确, 无结构性 bug

## 二级发现: agent 循环抹掉 skill 优势

- evolver 在 single-call 下 > s0 (+4.4pp: 0.5185 vs 0.4741)
- 但开 ReAct 后 evolver-react (0.3889) **反低于** s0-react (0.4000) → skill 优势被 agent 循环抹平/反转
- 越强的 skill 被 agent 循环伤得越深 (evolver Δ=−13pp > s0 Δ=−7.4pp)

## s0 逐题归因 (手算 3-seed mean, react vs single)

- progress 3 题: q6(+0.44) q25(+0.22) q28(+0.22) — single 本来 0 分的难题, agent 代码验证救回
- regress 7 题大退步: q26(1.0→0.44) q4(0.89→0.44) q3(1.0→0.67) q23(0.67→0.22) q17(0.33→0) — single 能部分/全部解的中等题被 agent 循环拖崩
- 净效应负: 7 题大 regress 远超 3 题 progress
- **非系统性提取 bug**: 满分题 q0/q5/q7/q15/q16 react 仍满分 (若是 bug 全题中招)

## 机制推断 (待轨迹证实, 但数据模式已指向)

AIME 是整数答案的竞赛数学题, 靠符号推理非计算. agent 循环的负向机制最可能是:
1. **注意力分散**: Think→Act→Observe 多轮把模型从纯数学推理拉向"何时调工具"的元决策
2. **token 预算挤占**: 多轮累积对话, 4096 token 给单步可能不够 → finish_reason=length 截断 → 没出 boxed
3. **过度验证**: 模型倾向对已对答案反复 execute_code, 多步循环中走偏

注: 轨迹未落盘 (per-seed JSON 只存 per_qid pass 率), 无法事后回看具体 fail 题。如需坐实机制需重跑带 trace dump 的版本。

## 结论

- **agent 能力 (ReAct+execute_code) 在 AIME 数学题上显著有害**, 不同于编程类任务 (self-debug 在消融实验中显著优于 baseline)
- 用户原始诊断"把有 self-debug 能力的 agent 当单次答题器用"方向对, 但**解药不是开 agent** — 单次调用在此任务/模型上已是更优配置
- 对论文: 这是一组干净的**负对照** — 在非编程任务上 agent 循环不是免费午餐, 为 skill-only 优化 (上下文空间进化) 的正当性提供了反面证据 (纯 prompt 优化 +4.4pp > 加 agent 能力 −13pp)

关联 [[aime-react-eval-launched-20261010]] [[skillopt-skel-finished-20261010]]
