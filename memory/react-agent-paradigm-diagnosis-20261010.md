---
name: react-agent-paradigm-diagnosis-20261010
description: Two mature agent frameworks (smolagents + sema-code-core) explored — my ReAct -13pp is implementation flaws + wrong paradigm (interleaved tools interrupt symbolic reasoning)
metadata:
  type: project
---

## 背景

ReAct eval 负结果后 (见 [[aime-react-eval-result-20261010]])，用户纠正我的过结论："说明你这个 react 的 agent 设计的很糟糕。看看有没有优化空间，比如使用这些好的 agent：lunwen/agent"。遂研究 agent/ 下两个成熟框架，对照诊断我的实现缺陷。

## 两种范式 (核心 insight)

| 范式 | 代表 | 机制 | 对 AIME 数学题 |
|------|------|------|---------------|
| 范式1: interleaved ReAct | sema-code-core / Claude Code / 我的 eval | Think→Act→Observe 多轮, 工具中途插入推理链 | 可能有害: 每次工具往返打断符号推理链 |
| 范式2: Program-of-Thoughts | smolagents CodeAgent | 整道题翻译成 Python, 在持久化解释器里跑, final_answer(x) 触发终止 | 可能有益: 消除心算错误, state 跨步保持 |

我的 eval 用的是范式1, 且实现粗糙。

## sema-code-core 探索结论 (Claude-Code 同款架构)

- 核心循环 src/core/Conversation.ts:30 的 ReAct async generator: 原生 tool_use block (非文本解析), 递归, 无迭代上限
- 唯一终止条件: 模型 emit 纯文本无 tool_use (模型自己决定何时停) — 我的 eval 是"输出里没 tool_call=final", 残破输出误判为 final
- 截断处理 (Conversation.ts:106-126): stop_reason==='max_tokens' 且有 tool_use → 丢弃残缺 tool call + 注入 reminder "请用更简洁方式重试"。我的 eval 丢弃 finish_reason (resp, _ = chat_target_messages(...)), 静默吞截断
- 无数学 prompt: system prompt 是 coding 导向 ("lead with the answer, not the reasoning" — 对 AIME 反向), 数学仅作 KaTeX 渲染指令
- 无答案提取: 全 src/ 无 boxed/整数 regex/final_answer tag
- 无 verification-via-code loop: run_shell 是通用 shell, 无"写代码验证答案"的脚手架
- 它的 synthesis 原话: "Consider NOT looping for pure-reasoning problems... interleaved tool can hurt because each tool round-trip is a reasoning interruption... make code-verification a distinct, optional, terminal step rather than an interleaved tool the model can fire mid-derivation"

## smolagents 探索结论

- CodeAgent (范式2): 模型写 Python 在 LocalPythonExecutor 持久解释器跑, state 跨步保持; final_answer(x) 触发 FinalAnswerException 终止 (模型必须主动调)
- max_steps 耗尽 → _handle_max_steps_reached 做一次带完整 memory 的总结 LLM 调用 ("agent 卡住了, 这是它尝试的全部, 给个答案")
- 错误当 observation 回灌 + "retry, try a completely different approach" (我的 eval try/except 吞掉, 模型看不到错误)
- ToolCallingAgent: 原生 function calling + tool_choice="required"
- 单条 observation 截断 20K 字符 (我的 eval 无截断, 多轮累积撑爆)
- system prompt 有数学示例 (88**0.36) 教模型"用 Python 算别在脑子里算"

## 我的实现缺陷 (4 条, 对照两框架)

1. 无 token 截断处理: eval_aime_react_multiseed.py:195 `resp, _ = chat_target_messages(...)` 丢弃 finish_reason, finish_reason=length 静默吞 → 没 boxed 输出被当 final
2. 强制过程: REACT_SUFFIX 推模型走 Think→Act→Observe, 平均 2.6 tool_call/题, 拖入元决策
3. 文本解析而非原生 tool_call: 正则解析 JSON tool call (现过时, aimpapi 2026-07-24 升级后 V3.2/235B 都支持原生 tool_call)
4. max_steps 耗尽无反思: 只 nudge 一次 512 token final, 不带轨迹 memory

## 根因分层 (关键)

-13pp = 实现缺陷 (缺陷1-4, 可修) + 范式选错 (范式1 对 AIME 本就不优, sema-core 自己都承认)
- 若只修缺陷1-4 → 可能把 -13pp 拉回 0 附近, 但难转正 (范式1 天花板)
- 若换范式2 (CodeAgent/PoT) → 才是"agent 能力帮数学题"的正确打开方式

## 下一步实验选项 (待用户拍板)

- A. 修范式1: 改 eval_aime_react_multiseed.py — 原生 tool_call + 修 finish_reason + 模型自决停 + graceful max-steps。工作量中。预期: -13pp → ~0, 难转正
- B. 换范式2 (smolagents CodeAgent): 用 smolagents CodeAgent + V3.2 跑 AIME25, 测 PoT 是否转正。工作量中。预期: 可能 +Δ (PoT 消除心算错误)。最高信号
- C. CCR 对照: 路由 Claude Code (范式1 成熟 scaffold) 经 CCR 到 V3.2 跑 AIME25。工作量大。若 CCR+V3.2 也负 → 坐实"范式1 对 AIME 有害" (非我实现差); 若正 → 证明我实现差。但 sema-core 探索已预示 CCR (Claude Code=范式1) 大概率也负

## CCR 现状

- ccr v3.0.19 已装, claude CLI 2.1.296 已装, headless 完整支持 (-p/--print, --system-prompt-file, --model, --output-format)
- ~/.claude-code-router/global-profile-takeover.json 已读到: 定义 claude-code + codex 两个 profile, codex 走 providerId="claude-code-router"
- 主配置 config.json 不存在 (实际是 config.sqlite, sqlite3 二进制未确认) — provider 路由到哪个模型待查。Bash 分类器在探查命令上反复超时, 暂受阻

## 关联

[[aime-react-eval-result-20261010]] [[aime-react-eval-launched-20261010]] [[skillopt-skel-finished-20261010]]
