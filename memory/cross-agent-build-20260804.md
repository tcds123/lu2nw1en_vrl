---
name: cross-agent-build-20260804
description: Task 3 跨agent (single-call vs ReAct) 评测代码设计与构建状态
metadata:
  type: project
---

# 跨 Agent 评测 (Task 3) 构建

**日期**: 2026-08-04 (build) → 2026-08-05 (smoke pass, 待全量) | **状态**: ✅冒烟通过(OK pass=1.0), ✅路径bug已修, ✅baseline改8-04 fresh, ⏳全量eval待启动(分类器outage阻塞)

## 当前状态 (2026-08-05 02:50)
- Task1消融✅ + Task2 AIME✅(负结果) 均已完成记录, 28RPM 已空闲可跑 Task3.
- ✅ 冒烟通过: `SMOKE_RESULT: OK pass_rate=1.0 pred=A n_tools=0` (1题 max_steps=2, 106 items loaded, evolver hash 12633201 正确). n_tools=0=模型直接答未用工具(易题), 全量max_steps=6难题会显真实工具使用率.
- ✅ 路径bug修复: `run_overnight_pipeline.sh:52` 硬编码 `livemathmaticianbench`(漏e)→已改`livemathematicianbench`. orchestrator已退出(阶段3冒烟曾因此崩). 详见 docs/phase2_issues.md.
- ✅ baseline drift 修正: 默认 --single_dir=multiseed_test/(7-30 cached, seed1=0.5031 虚高+5.7pp drift)→改用 `multiseed_test_ablation/`(8-04 fresh, mean 0.4497, 同hash 12633201, 1天间隔drift可忽略). 避免baseline虚高偏置ReAct delta.
- ⏳ **全量eval待启动** (分类器间歇outage阻塞Bash启动, 已重试4次). 启动命令已存 `rl/train/verl1/run_cross_agent_react.sh`:
  ```bash
  bash rl/train/verl1/run_cross_agent_react.sh
  # = python eval_bagent_react_multiseed.py --skills evolver --seeds 4 --eval_k 3 --max_steps 6 --max_workers 40 --single_dir .../multiseed_test_ablation
  ```
  产物: `outputs/tf_skill_math_v2/multiseed_test_react/react_eval.json` (含 react+single 对比 + 配对t). 预计 ~2-4.5h (取决于工具使用率).

## 实验问题
在已优化的 evolver skill (hash 12633201, 主结果 test=0.4623) 之上, 把 B 模型从「单次生成」升级为「ReAct 循环 + python_executor 自调试」是否进一步提升 LiveMath test=106 通过率?

## 公平性设计 (关键)
- 同一 skill (s0 68571cb2 / evolver 12633201), 同一模型 (DeepSeek-V3.2), 同一 28RPM 限速.
- **baseline = single-call**: ~~复用 multiseed_test/ 7-30 cached~~ **改为 8-04 fresh** `multiseed_test_ablation/per_{s0,evolver}_{hash}_seed{1-4}.json` (避免 7-30 cached seed1 drift +5.7pp 虚高偏置 delta; 8-04 同期 fresh mean 0.4497). eval_k=3, 4-seed, 不重跑.
- **treatment = ReAct**: system = `_build_system(skill)` + `REACT_SUFFIX` (追加工具说明); 多轮 `chat_target_messages` 循环, `<tool_call>` 调 execute_code, 无工具回复=最终答案. eval_k=3, N-seed, 与 baseline 对齐.
- **唯一差异** = REACT_SUFFIX + 工具循环. skill/模型/限速/评测完全一致.

## 关键技术决策
1. **后端**: 用 skillopt `chat_target_messages` (DeepSeek-V3.2, 28RPM token bucket, key msk-4601b2...), **不用** BAgent._call_llm (raw requests.post, 无限速, 会打爆 API). 见 [[api-throughput-workers-20260804]].
2. **execute_code**: 复制自 `agent/b_agent/core.py` `BAgent._execute_python_code` (staticmethod, 沙箱 subprocess, 10s 超时). 复制而非导入, 避免触发 BAgent 的 SOUL/402-skill 机制, 保证 system prompt 干净 = evolver skill 独立.
3. **工具调用**: 文本 `<tool_call>{"name":"execute_code","arguments":{"code":"..."}}</tool_call>` (BAgent 已验证格式, 比原生 tool calling 更稳).
4. **不重用 BAgent.run()**: 它注入 402 skills + SOUL (`format_all_active_for_context` + `context.build`), 会混淆 (evolver skill + 402 skills + SOUL + ReAct), 不干净.
5. **max_steps=6**, 用尽后强制一次 "FINAL ANSWER" nudge (512 token) 保证有答案.
6. **REACT_SUFFIX**: 引导模型用 execute_code 做数值验证/反例/sympy 符号检查/小例枚举, 最后输出 choice label.

## 文件
- `rl/train/verl1/eval_bagent_react_multiseed.py` - 主脚本. 缓存 `per_{name}_{hash}_react_seed{N}.json` (文件名含 react 区分 single). 产物 `outputs/tf_skill_math_v2/multiseed_test_react/react_eval.json` (含 react+single 对比 + 配对 t 检验).
- 默认: `--skills evolver --seeds 4 --eval_k 3 --max_steps 6 --max_workers 40 --split_dir ...1-3-6_seed42`

## 资源估算
- 每 (skill,seed): 106 题 × eval_k=3 × ~3 API call (avg ReAct 步) ≈ 950 call. 28RPM ≈ 34min.
- 2 skill × 4 seed = 8 × 34min ≈ 4.5h. 若时间紧: 减 seed=2 (≈2.3h) 或仅 evolver (省 s0).

## 待办
1. API 冒烟 (1 题, max_steps=2, k=1) - 待 ablation 间隙或 AIME 后.
2. 全量跑 (ablation chain + AIME 完成后, 28RPM 共享不能并行).
3. 写 memory/cross-agent-result-<date>.md (含 Δ, p 值, 结论).

## 依赖
- ablation chain (Task 1) + AIME (Task 2) 必须先完成 (串行 API). 见 [[aime-from-scratch-build-20260804]].
