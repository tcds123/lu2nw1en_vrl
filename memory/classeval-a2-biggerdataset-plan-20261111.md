---
name: classeval-a2-biggerdataset-plan-20261111
description: A2 (2-turn self-test) + 更大数据集两条路径的规划与候选数据集调研
metadata:
  type: project
---

# A2 + 更大数据集 两条路径规划 (2026-11-11)

## 背景
4-run ClassEval 全在 ±1 题噪声内, 两条瓶颈被用户确认:
1. **val=10 过拟合** — best-on-val 到 test 不泛化 (4 run 全 Δ≤0 或 ±1题)
2. **基线已高** — soft 0.88 (平均每 class 过 88% test function), 可提升 tail 仅 12%, 且是题级规格 (通用 skill 文本边际最小)

用户定方向: "简单试一下 A2, 然后尝试换大数据集".

## 路径 1: A2 — ClassEval rollout 改 2-turn self-test

### 机制 (不改数据集, 不建沙箱)
- 现状: `rollout.py:process_one` single-shot (chat_target 1 调), 模型读 checklist 靠自觉 internal-trace doctest
- A2: turn1 出码 → `evaluate` 拿 failure_trace → turn2 把失败反馈喂回模型修 → 重新 evaluate 取最终结果
- **为什么对症**: soft 0.88 的 12% tail 正是 doctest 能直接报错的部分; turn2 给模型实际执行反馈 (而非文本"请仔细读"), 精准定位每个 tail method 的具体错误
- **判据**: 若 test Δ 跳出 ±1 题 (≥+2-3题) → 证实 single-shot 是关卡1瓶颈, doctest 反馈有效; 若仍 ±1题 → tail 连反馈都补不动, val=10 过拟合成硬瓶颈

### 实现 (单文件改 classeval/rollout.py)
- 已确认 `chat_target_messages(messages, ...)` 支持多轮 (qwen_backend `_chat_messages_impl` 直传 messages list, OpenAI 格式)
- messages = [system, user1(skeleton), assistant(turn1码), user2(失败反馈+修指令)] → chat_target_messages 返回 turn2 码
- turn1 用现有 evaluate() 拿 failure_trace (evaluator 已返回); turn2 重新 evaluate 取最终 hard/soft
- 注意: 只当 turn1 没全过 (hard==0) 时才走 turn2 (全过不必修), 省 API 调用
- A2 是 rollout 层改动, SkillOpt loop (reflect/aggregate/select) 不变 → 同 initial.md 跑, 单变量

### 成本
- 每个 fail item 多 1 次 API 调用 (turn2). ClassEval ~50% item fail → calls +50%. 20 RPM 下 38min→~55min, 可接受.

## 路径 2: 更大数据集 (val≥30, 避开沙箱)

### 候选调研结果 (本地 data/)
| 数据集 | 规模 | val (2:1:7) | 执行模式 | 沙箱需求 | 已适配 |
|--------|------|-------------|---------|---------|--------|
| ClassEval | 100 | 10 | 进程内 unittest | 无 | ✅ |
| **mbpp_pro** (CodeEval-Pro) | 378 | ~53 | 进程内 exec | 无 | ✅ mbpppro env |
| humaneval_pro | 164 | ~23 | 进程内 exec | 无 | ❌ (可仿 mbpppro) |
| **APPS_Plus** | 7413 | 2766 | subprocess stdin/stdout | 无 (stdlib) | ✅ appsplus env |
| LCB | 400 | ~57 | — | — | ✅ lcb env (已跑 Δ+4.9pp) |

### 选 APPS_Plus (已适配 appsplus env, 最省力)
- env 完整: adapter/dataloader/evaluator/rollout/prompts/skills 全有
- 执行: subprocess `python3 code_path` stdin/stdout 对比 (无 docker, 竞赛题只用 stdlib)
- split: 7413 题 2:1:7 → val=2766 (远超 ClassEval 10), 彻底解决过拟合
- **基线对照**: 上游 V3.2 全量跑过 (noskill 0.656 / seed 0.650 / best 0.634 — best 在 V3.2 上退化). flash 在 LCB 已翻转 V3.2 负结论, APPS_Plus 是验证"flash+skill 是否也能翻转 V3.2 在竞赛题上的负结论"的最佳场.
- **成本问题 (关键约束)**: 全量 test=2765 + 4 epoch + 2-turn 太重 (V3.2 全量 ~48h). flash 20 RPM 下必须采样. 用 `--test_env_num N` / batch_size 控制.

### 成本估算 (flash 20 RPM)
- 每步 rollout: batch_size 题 × max_turns 调用 + reflect/aggregate 调用
- 4 epoch × 4 step/epoch = 16 step. 每 step rollout batch 题.
- 想让 val≥30: train 60 / val 30 / test 100 采样 (split 改 2:1:7 仍按 ratio, 但整体采 190 题)
- **但 appsplus env 的 ratio split 是全量 7413 切**, 不是采样 190. 需确认 `limit`/`train_size`/`test_env_num` 如何控制实际跑的子集.
- 待定: 采样规模 N (val≥30 前提下尽量小, 控 wall<2h)

### 待确认决策点
1. **A2 先跑还是并行**: flash 20 RPM 串行 (两 run 共享配额, 并行会 429). A2 先 (单文件改动, 快), 完成后跑 APPS_Plus.
2. **APPS_Plus 采样规模**: 倾向 train 60 / val 30 / test 100 (val=30 满足分辨率, test=100 与 ClassEval 70 同量级可比), 但需确认 env 是否支持这种"全量切 split 后只跑子集".
3. **APPS_Plus 后端**: 必须 flash (对齐 B1/LCB 翻转叙事), 不是 V3.2. 需新写 launch shim (仿 classeval shim: 读 flash key from yaml, unset AIGC_USER).

## 关联
- [[classeval-skill-leverage-reanalysis-20261111]] (两关卡研判, A/B 结果已更新)
- [[classeval-checklist-result-20261111]] (关卡1 single-shot 未证实 → A2 动机)
- [[classeval-softgate-result-20261111]] (关卡2 证伪)
- [[b1-flash-skillopt-lcb-result-20261109]] (LCB Δ+4.9pp 正杠杆, flash 翻转 V3.2)
