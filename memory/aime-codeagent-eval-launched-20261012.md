---
name: aime-codeagent-eval-launched-20261012
description: AIME25 CodeAgent (PoT) eval RELAUNCHED after Bug 5 fix — per-trajectory timeout + per-qid checkpoint
metadata:
  type: project
---

## 启动 (2026-10-12 09:40:38)

AIME25 CodeAgent (Program-of-Thoughts) eval 重跑, 在 Bug 5 (CPU 忙循环死锁 26h) 修复后启动。

- launcher PID 114187, eval 子进程 PID 114189
- log: `outputs/tf_skill_aime_v2/aime25_codeagent/run_20261012_094038.log`
- 启动方式: `source lunwen/bin/activate && python -u _run_codeagent_eval.py` (venv 必须激活 — smolagents 装在 venv site-packages, plain /usr/bin/python 找不到)
- 配置: `EVAL_TEMP=0.3 k=3 seeds=3 max_steps=6 workers=3` — 与 single-call 基线 + ReAct eval 逐字对齐
- skills: s0 (hash `c11d7dc0`) + evolver (hash `f00fa794`) — 哈希对齐基线缓存
- backend: DeepSeek-V3.2 via aimpapi.midea.com, 28 RPM 共享 bucket
- 规模: 2 skills × 3 seeds × 30 题 × k=3 = 540 轨迹, 预计数小时

## Bug 5 修复 (本次重跑的前置)

| 层 | 修复 | 验证 |
|---|------|------|
| per-trajectory 超时 | `_run_agent_with_timeout`: daemon 线程跑 `agent.run()`, `join(timeout=TRAJ_TIMEOUT=240s)`, 超时放弃该 k-vote (pass 计 0) | 冒烟测 fast/slow 两路径 OK |
| per-qid 检查点 | `eval_codeagent_on` 每完成一题原子写 cache (`tmp+os.replace`), rerun skip 已完成 | out_dir 干净, 全量重跑 |
| main 缓存判定 | 部分缓存 (`len < n_items`) 视为 resume 而非完成; 损坏 json 降级 fresh | 代码审查 |

详见 `docs/phase2_issues.md` Bug 5 (根因 + 修复落地)。

## 研究意义

这是 ReAct 负结果 (-7.4pp s0 / -13.0pp evolver, 见 [[aime-react-eval-result-20261010]]) 之后的**对照范式实验**: 同样是"加 agent 能力", 但换成 PoT (模型写 Python 在持久解释器里算, final_answer 终止) 而非 ReAct (交错工具调用)。如果 PoT delta 仍为负 → agent 能力在竞赛数学上确无帮助 (强化 single-call 优化的正当性); 如果 PoT delta 为正 → ReAct 的负结果是范式选错而非 agent 能力本身有害。

## 待办 (✅ 全部完成, 见 [[aime-codeagent-eval-result-20261012]])
- [x] 收集 540 轨迹结果, 写 per_qid JSON (6 个 per-seed cache 全 30/30)
- [x] 算 delta vs single-call baseline (paired t-test) — s0 +14.8pp p=0.0007 SIG / evolver +6.3pp p=0.147 NS
- [x] 写最终 memory 记录结论 + 更新 MEMORY.md 索引

**结果一句话**: PoT 范式显著正向 (s0 +14.8pp SIG), 是 ReAct 负结果 (−7.4pp) 的反面 — ReAct 负结果归因为范式选错而非 agent 能力本质有害。强 skill 跨两种 agent 范式都翻转排名 (压缩 headroom)。
