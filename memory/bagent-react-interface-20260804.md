---
name: bagent-react-interface-20260804
description: Task3跨agent接口调研: BAgent.run()=ReAct(read_file/write_file/execute_code); b_agent_interface=单次无工具; 同模型可对照
metadata:
  type: reference
---

Task3 (跨agent: single-shot -> ReAct) 接口调研结论. 代码在 agent/b_agent/core.py + rl/train/verl1/b_agent_interface.py.

## 两条 B 路径
1. **单次 single-shot** (当前 eval 用): `tf_global_v2_math.run_one` -> `chat_target(system=skill, user=problem)` 单次生成. 无工具. eval_test_multiseed / eval_aime24_multiseed 都走这条.
2. **BAgent 单次** (GRPO reward 用): `b_agent_interface._generate_one_with_skills` -> `agent.generate_code(task, system_prompt=full_system)`. 也是单次无工具 ("简化模式: 单次生成, 无工具"), 但额外注入 402 条 BAgent skill + Meta-Skill 引导语. **不是 ReAct**.
3. **BAgent ReAct** (core.py:268 `run()`): Think->Act->Observe 循环, max_steps=10. 工具: `read_file`/`write_file`/`execute_code`(沙箱 Python, 10s 超时). 无工具调用时输出 final_answer 并 break. **这才是 ReAct**.

## Task3 实验设计 (待实现)
- **变量**: 同 skill (evolver 主结果 12633201) + 同模型 (DeepSeek-V3.2 via chat_target), 仅切换 single-shot vs ReAct. 测 skill 是否跨 agent 架构泛化.
- **关键**: BAgent.run() 注入自己的 402 skill (self.skills.format_all_active_for_context), 不接受外部 skill 字符串. 要测 evolver skill 在 ReAct 下的效果, 需绕过 self.skills, 用 evolver skill 作 system prompt base + 工具描述 + ReAct 格式指令, 复用 BAgent 的 _execute_python_code / _call_llm / _parse_tool_calls.
- **execute_code 工具 = self-debug 能力** (Phase 2 issue #5: "B 缺少 python_executor tool，无法 self-debug"). ReAct 让 B 能写 Python 验算/检查答案. 对 AIME 数值题尤其有用 (直接算+校验), 对 livemath MCQ 也可验证计算 (极限/行列式/反例).
- **模型后端待确认**: BAgent __init__ 接受 model_name, docstring 说 "封装 Qwen2.5-72B API". 须确认 _call_llm 是否走 chat_target (DeepSeek-V3.2) 还是另一后端 -- 若不同模型则 cross-agent + cross-model 混淆, 需对齐到同模型才干净.

## 实现路径 (推荐)
新建 `eval_bagent_react_multiseed.py`: `run_one_react(item, skill, k, max_steps)` 自建 system prompt (skill + 工具 + ReAct 格式), 复用 BAgent._execute_python_code (静态方法) + chat_target, 跑 ReAct 循环, extract_int_answer/evaluate 抽答案. 先 smoke 1-seed k=1, 再 4-seed. 对照 single-shot run_one, 比 ΔvsS0.

关联: [[multi-agent-generalize-plan-20260731]] [[aime-from-scratch-build-20260804]] [[api-throughput-workers-20260804]]
