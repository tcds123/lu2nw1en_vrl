---
name: classeval-a2-2turn-launch-20261111
description: A2 启动 — ClassEval rollout 改 2-turn self-test (max_turns=2), 单变量对照 gated-hard Δ0.0
metadata:
  type: project
---

# A2 启动 — ClassEval 2-turn self-test (2026-11-11)

## 一句话
把 ClassEval rollout 从 single-shot (max_turns=1) 改成 2-turn self-test:
turn1 出码 → evaluate 跑 unittest 拿 failure_trace → turn2 把失败反馈喂回模型修 →
重新 evaluate 取更优. **唯一变量 = rollout.py 改动** (initial.md 不变, gate 仍 hard+gated),
对照 gated-hard baseline Δ=0.0 ([[classeval-gated-skillopt-result-20261111]]).

## 假设 (为何对症)
- 4 run 全在 ±1 题噪声 → 两条瓶颈: 关卡1 (guidance 未固化, single-shot 未证实),
  关卡2 (gate 太粗, 已证伪 [[classeval-softgate-result-20261111]]).
- soft 0.88 的 12% tail 正是 doctest 能直接报错的部分. single-shot 下模型读 checklist
  靠自觉 internal-trace doctest (没真执行). A2 给模型**实际执行 doctest 的回合**:
  turn1 失败的具体 traceback (assertIsNone / AssertionError) 喂回去, 模型对着真错误修,
  而非对着散文 "请仔细读 docstring".
- 预期: 若 doctest 自测真有效, test Δ 应跳出 ±1 题噪声 (≥+2-3题).
  若仍 ±1题 → tail 连反馈都补不动, val=10 过拟合成硬瓶颈 → 转 APPS_Plus 大数据集.

## 单文件改动 (已落地 + 冒烟通过)
`/data/zhuldz/skillopt/skillopt/envs/classeval/rollout.py`:
- import 加 `chat_target_messages`
- 新增 `_build_repair_user(eval_result)`: turn2 user prompt, 含 pass_rate + error_type +
  failure_trace[:2500] + 修指令 (只改错的, 不重写已过的, 注意 return shape/sentinel/措辞/公式)
- 新增 `_eval_better(a,b)`: hard 优先, hard 同比 soft, 平局取 turn2
- `process_one` 改: turn1 chat_target → evaluate(eval1) → 若 max_turns>=2 且 eval1.hard==0:
  messages=[system,user1,asst1,user2_repair] → chat_target_messages → evaluate(eval2) →
  取 _eval_better(eval1,eval2). 全过的 item 不走 turn2 (省 API). conversation.json 记两轮.
- `run_batch` summary 加 `n_two_turn` 计数.
- `max_turns` 经 adapter.max_turns → run_batch → process_one 透传. config `env.max_turns`
  经 flatten_config 到 top-level `max_turns`, adapter 接收. **已验证 env.max_turns=2 透传生效**.

## 冒烟 (2 item, 通过)
- ClassEval_0: hard=0 soft=0.8947 **n_turns=2** (turn1 fail→turn2 触发), fail_reason=真
  unittest traceback (test_filter_7 assertIsNone AssertionError) ← 这就是 turn2 反馈料
- ClassEval_10: hard=1 soft=1.0 **n_turns=1** (turn1 全过→turn2 跳过, 省 API)
- chat_target_messages 与 flash 端点兼容, 无异常.

## 运行参数
- launch: `RUN_TAG=classeval-a2-2turn-flash-20261111 EXTRA_CFG="env.max_turns=2" bash launch_skillopt_flash_classeval.sh`
- skill_init: **initial.md** (与 gated-hard baseline 同, 非 checklist) ← 单变量是 rollout
- gate_metric: hard (默认, 与 gated-hard baseline 同)
- slow_update_gate_with_selection: true (gated, 同 baseline)
- epochs=4 seed=42 split ratio 2:1:7 (val=10 test=70, 与 4 run 全同)
- 成本: ~50% item fail → +50% API 调用. 20 RPM 下 baseline ~25min → 预计 ~40min.
- 对照 baseline (gated-hard, [[classeval-gated-skillopt-result-20261111]]):
  S0 test=0.4571, best test=0.4571, Δ=0.0, 3×REJECT.

## 判据
| test Δ | 结论 |
|--------|------|
| ≥ +2题 (+0.0286) | doctest 自测有效, 关卡1 single-shot 是瓶颈 → skill 杠杆证实 |
| ±1题 ([-1,+1]) | tail 连反馈都补不动, val=10 是硬瓶颈 → 转 APPS_Plus |
| ≤ -2题 | turn2 引入退化 (flash 自我修复越改越错) → 也算跳出噪声, 记录分析 |

## 后续
A2 完成后 → APPS_Plus (大数据集 val≥30, [[classeval-a2-biggerdataset-plan-20261111]]).
两 run 共享 20 RPM, 串行 (A2 先).

## 关联
- [[classeval-a2-biggerdataset-plan-20261111]] (A2 + APPS_Plus 双路径规划)
- [[classeval-skill-leverage-reanalysis-20261111]] (两关卡研判, A2 是关卡1 的结构验证)
- [[classeval-gated-skillopt-result-20261111]] (gated-hard Δ0.0, 本 run 对照 baseline)
- [[classeval-checklist-result-20261111]] (方向 A checklist, single-shot 关卡1 未证实 → A2 动机)
