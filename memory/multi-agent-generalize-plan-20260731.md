---
name: multi-agent-generalize-plan-20260731
description: 换agent框架测skill泛化实验方案(BAgent单轮→ReAct/nanobot)+周末长跑候选(evolver全量10轮)
metadata: type: project
---

# 换 Agent 框架测 skill 跨 agent 泛化 — 实验方案 (2026-07-31) 📋

## 背景
AIME24 跨任务泛化失败 ([[aime24-generalize-result-20260731]]): livemath 选择题 skill 不迁移到数值题.
用户提出新泛化维度: **换 agent 框架** — 数据集不泛化, 那 skill 能否在不同 agent 架构上泛化?
(跨模型换 LLM 也是一条路, 但当前网关可用模型不多, 留作备选, 见末节.)

## 现状 (调研结论)
当前数学 skill 评测链路: `_build_system(skill_content)` → `skillopt.model.chat_target` → DeepSeek-V3.2, **单轮 chat, 无 ReAct, 无工具**. skill 直接拼进 system prompt.
agent/ 下可用框架 (详见 [[调研]]):
- **BAgent** (agent/b_agent/core.py): 自研, 走独立 DeepSeek-V3.2 endpoint (不经 skillopt). 两条路径:
  - (a) ReAct 多步循环 (带 read_file/write_file/execute_code 工具 + skill 经 SkillLibrary 注入)
  - (b) self-debug 多轮 (外部编排执行 unit test 反馈)
  - 当前 v2 评测用的是 (b) 之外的最简单单轮, 没用 BAgent 的 ReAct 能力
- **nanobot**: 完整开源框架, skill 静态 SKILL.md 格式, 无 ATE 评分. 换框架变量多.
- **smolagents**: 无内置 skill 系统, 不推荐.

## 实验方案 (优先级排序)

### 方案 1 ⭐: BAgent 单轮 → BAgent ReAct (同模型, 加工具循环)
**问题**: skill 在单轮 chat 上有效 (+16.1pp), 换到带工具的 ReAct agent 上是否仍有效? 还是 skill 的诊断程序被工具调用"淹没"?
**设计**:
- 控制变量: 同一 DeepSeek-V3.2, 同一 livemath test=106, 同一 S0/v2/evolver skill
- A 组 (baseline-agent): 当前单轮 chat_target (无工具) — 已有 4-seed 结果
- B 组 (react-agent): BAgent.run() ReAct 循环 (read_file/write_file/execute_code 三工具 + skill 注入)
- 对比 B 组内 S0 vs evolver 的 ΔvsS0, 看 skill 在 ReAct agent 上是否仍 +16pp
**改动**: 写一个 `eval_bagent_react_multiseed.py`, 复用 BAgent.run() 替换 chat_target, 评测逻辑不变 (livemath 多选 evaluator)
**判据**: B 组 evolver ΔvsS0 显著正 → skill 跨 agent 形态泛化; 若 Δ≈0 或负 → skill 依赖单轮形态, ReAct 下失效
**风险**: BAgent ReAct 路径调 DeepSeek-V3.2 但有工具调用, 单题耗时大增 (多轮 LLM), 28RPM 下 106题×k×seed 可能 >10h

### 方案 2: BAgent self-debug (同模型, 加执行反馈循环)
**问题**: skill + self-debug 反馈循环是否协同? ([[work-summary-20260722]] 提到 self-debug 组曾显著优于 baseline)
**设计**: 用 BAgent.generate_code_with_self_debug(), 对比 单轮 vs self-debug 下 skill 的 ΔvsS0
**注**: 这是代码生成场景的 skill (b_agent/skills/), 与数学 skill (livemath skills/initial.md) 不同体系. 数学题无 unit test, self-debug 不直接适用. **方案2 仅适用于代码基准, 非当前数学主线**.

### 方案 3: 换到 nanobot 框架
**问题**: skill 跨完全不同的 agent 框架能否迁移?
**设计**: 把 v2/evolver skill 转 nanobot SKILL.md 格式, 用 nanobot 跑 livemath test=106
**改动大**: skill 格式转换 + nanobot 配置 + 后端对齐. 变量多 (框架+prompt模板+后端), 归因难. **优先级低**.

## 推荐执行顺序
1. ⭐ **方案1 (BAgent ReAct)**: 最贴近"换 agent"且变量可控 (只加工具循环). 先跑 1 seed k=1 冒烟确认 ReAct 路径能跑通 livemath 选择题, 再上多 seed.
2. 方案3 (nanobot) 备选, 若方案1 显示 skill 在 ReAct 上失效, 再验证是否框架级不泛化.
3. 跨模型 (备选): BAgent 构造参数切后端, DeepSeek-V3.2 → Qwen3-235B (若网关有). 待确认可用模型.

## ⏳ 周末长跑候选 (2026-08-01 ~ 08-03)
周末两天可跑的长实验 (不占人, 后台 28RPM):

### 候选 A ⭐: evolver 全量 10 轮 (从 v2 起点进化到收敛)
- 命令: `python rl/train/verl1/skill_evolver_math.py --iters 10 --eval_k 5 --K 4 --train_n 35`
- 起点: v2 skill (run_20260728_131934)
- 预估: 28RPM 下 ~20-24h (07-30 冒烟 iters=2/eval_k=3/train_n=5 跑了 4h)
- 价值: 看完整进化曲线是否收敛/超过当前 evolver 2-iter 产物 (test +16.1pp). 若 10 轮产物更强, 进论文主结果.
- ⚠️ 风险: 28RPM 限速 + 429 退避可能拖到 >24h; eval_k=5 比 07-30 的 eval_k=3 更慢. 可降 eval_k=3 缩到 ~15h.
- **建议配置**: `--iters 10 --eval_k 3 --K 4 --train_n 35` (复用 07-30 冒烟的 eval_k=3, 产物身份校验已修, ~15-18h 适合周末)

### 候选 B: 方案1 (BAgent ReAct) 多 seed 全量
- 依赖方案1 冒烟先跑通. 若周六冒烟通过, 周日跑 3-seed k=3 全量 (~8-10h)

### 候选 C (低优先): v2/evolver 扩到 6-8 seed 压 CI
- 当前 4-seed CI 已显著 ([+0.126,+0.196]), 边际收益低. 除非要投稿需更窄 CI.

## 待办
- [ ] 周五前: 方案1 冒烟 (BAgent ReAct 跑 livemath 1题, 确认工具循环能产出选项答案)
- [ ] 周五启动: 候选A evolver 全量 10 轮 (eval_k=3), 周末跑
- [ ] 周末查 evolver 进度 + 跑完后 multiseed 复评 test=106
- [ ] 方案1 冒烟通过 → 周日跑 BAgent ReAct 多 seed

Related: [[aime24-generalize-result-20260731]], [[evolver-test-generalize-20260730]], [[next-steps-20260729]]
