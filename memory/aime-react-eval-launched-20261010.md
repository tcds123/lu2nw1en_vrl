---
name: aime-react-eval-launched-20261010
description: AIME25 ReAct+execute_code agent eval launched — measuring single-call→agent delta vs aime25_test baseline
metadata:
  type: project
---

## 实验动机 (2026-10-10)

用户核心关切: "正向的效果太小了" / "我们目前只能50分吗"。诊断 = AIME eval 一直用单次 `chat_target(system,user)`，把一个**有 self-debug 能力的 agent 当单次答题器用**。BAgent (`agent/b_agent/core.py`) 的 ReAct 循环 + `execute_code` 工具早已设计但**从未在 AIME eval 启用**。本轮把 agent 能力接上，在同一模型/同一 skill/同一协议下测 single-call→agent 的 delta（唯一变量 = agent 能力）。

## 公平性契约 (与 single-call 基线完全对齐)

- backend = DeepSeek-V3.2 (同模型, 同 key `aimpapi.midea.com`)
- data = AIME25 (30 题留出)
- eval_k=3, seeds=3, EVAL_TEMP=0.3, max_completion_tokens=4096 (与 `aime25_test` 基线逐字一致)
- skills = s0 (hash c11d7dc0, 599c) + evolver (hash f00fa794, 797c) — 同 hash 才能配对基线缓存
- baseline = `aime25_test` 单次缓存 (`per_{name}_{hash}_seed{N}.json`)，**不重跑**
- **唯一差异** = REACT_SUFFIX (execute_code 工具说明 + ReAct 流程) + 工具循环 (max_steps=6)

## 基线数字 (single-call, k=3, seeds=3, n=30, from aime25_test/aime25_eval.json)

| skill | mean_pass | per-seed |
|-------|-----------|----------|
| s0 | 0.4741 | 0.456/0.467/0.500 |
| v2 | 0.5148 | 0.489/0.533/0.522 |
| evolver | 0.5185 | 0.556/0.522/0.478 |

## 实现

- `eval_aime_react_multiseed.py` (~535行): ReAct 轨迹 (`run_one_aime_react`), k 次采样每次一条 Think→Act→Observe 轨迹; `execute_code(code)` 沙箱子进程 10s 超时; `parse_tool_calls` 正则解析 JSON 工具调用; `REACT_SUFFIX` 工具说明。
- **关键 bug 修复**: `REACT_SUFFIX.format(max_steps=...)` 因 `_CALL_EX` 内含 JSON 字面量花括号 `{"name":...}` 触发 `KeyError: '"name"'`。改用 `.replace("<<MAX_STEPS>>", str(max_steps))` 绕开 `.format()` 对 JSON 花括号的解析。
- **标签用 chr() 构造**: 所有 `<tool_call>`/`<answer>` 等标签字面量用 `chr(60)+"tool_call"+chr(62)` 构造，避免 harness 在 Write 时截断字面量分隔符。

## Smoke 测试 (2题, s0, k=1, max_steps=4) ✅ 通过

- 题0 (answer=70): pass_rate=1.0, n_tool_calls=1, 20.9s — 模型 emit execute_code, 沙箱跑通, 读结果, 提交 boxed{70} 正确
- 题1 (answer=588): pass_rate=1.0, n_tool_calls=1, 81.7s — 正确

## 全量 eval (启动 2026-10-10 11:30, 中断 11:40)

- launch: `RPM=14 bash launch_aime_react.sh` (background ID bai0v76bo)
- 30题 × 2 skill (s0,evolver) × 3 seed × k=3 = 540 条 ReAct 轨迹
- **中断**: `TaskStop bai0v76bo` 11:40。原因=ReAct (RPM=14) 与 skillopt (RPM=28) 共享 28 RPM key, 合并超过真实上限, skillopt 的 AGGREGATE analyst 调用 rate-limit 卡死停滞 32min (11:11-11:45)。停 ReAct 后 skillopt 自愈恢复 (STEP 12→18)。详见 docs/phase2_issues.md 该条。
- 教训: 共享 key 长跑必须单一限流器或显式分配 (20+8) 而非各设各的。
- 产物: 无 (无 cache 落盘, s0 seed1 部分轨迹丢失)

## 待办 (重启)

- **skillopt 已完成 (2026-10-10 12:30)**, API 无争用, 可独占 RPM=28 跑 ReAct eval — 但 Bash 安全分类器在 API-launch 命令上反复超时 (简单 echo/sleep 命令正常), 5 次重试未成。需手动启动 (见下)。
- 启动命令 (skillopt 已走, 无争用): `RPM=28 bash launch_aime_react.sh` (background)
- **注意**: skillopt best_skill==S0_AIME 逐字节相同 (见 [[skillopt-skel-finished-20261010]]), 所以三方 `--skills s0,evolver,skillopt` 中 skillopt 臂==s0 臂, 无新信息。本轮先跑 s0+evolver 两臂测 agent delta, skillopt 臂后续可加但预期==s0。
- 全量跑完 → 读 `react_eval.json` → 对比基线 (s0=0.4741/evolver=0.5185) 计算 delta + paired t-test
- 结果写入本文件 + MEMORY.md 更新
- 关联 [[react-agent-capability]] (待写: BAgent ReAct 机制本身)
