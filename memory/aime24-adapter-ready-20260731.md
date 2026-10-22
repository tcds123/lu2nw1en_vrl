---
name: aime24-adapter-ready-20260731
description: AIME24跨任务泛化评测脚本已就绪(eval_aime24_multiseed.py), 数据缓存30题, 冒烟通过, 待扩测完成后启动
metadata: type: project
---

# AIME24 跨任务泛化评测脚本就绪 (2026-07-31) 🔄

## 背景
[[next-steps-20260729]] task 4: v2/evolver 在 LiveMathematicianBench (选择题) 上的增益能否泛化到 AIME24 (数值题, 不同题源, 完全 held-out). 这是终极泛化判据. 07-31 完成 adapter 编写 + 冒烟.

## 已完成
1. **数据**: `data/aime24.jsonl` — HuggingFaceH4/aime_2024, 30 题, 答案 0-999 整数 (range 23-902), 字段 id/problem/answer/year. 本地缓存避免每次访问 HF.
2. **脚本**: `rl/train/verl1/eval_aime24_multiseed.py`
   - 复用 v2 框架: `chat_target`/`set_temp`/`EVAL_TEMP`/`md5`/多 seed 配对/断点续跑缓存 (skill hash 前缀)
   - 新增 AIME 专用: `_build_system` (数值题+\\boxed{N} 输出指令) / `_build_user` / `extract_int_answer` (\\boxed{} → <answer> → 末整数 三级兜底) / `evaluate_aime` (数值比对)
   - 评测逻辑单元测试 7/7 通过 (含多 \\boxed 取最后, 无数字返回 None)
3. **冒烟**: S0 单题 k=1, 第1题 gt=204 答对 (pred=204), 端到端链路通.

## 关键设计差异 (vs livemath 多选)
- livemath evaluator 比对选项 label (A/B/C); AIME 比对整数
- livemath _build_system 要求输出选项字母; AIME 要求 <answer>\\boxed{N}</answer>
- skill (S0/v2/evolver) 原为选择题写, 在数值题上未必增益 — 泛化失败本身也是有价值结论 (skill 过拟合选择题形态), 无论正负都记录

## 启动命令 (待扩测 PID 23868 完成后)
```bash
source lunwen/bin/activate
python rl/train/verl1/eval_aime24_multiseed.py --skills s0,v2,evolver --eval_k 3 --seeds 3
```
- 默认 evolver_best 指向 07-30 新 skill (run_20260730_111306, hash 12633201) ✅ 已在脚本常量修正
- 工作量: 3 skill × 3 seed × 30题 × k=3 = 810 次调用, 28RPM 下约 30-40 min (不与扩测争抢时)
- 输出: rl/train/verl1/outputs/tf_skill_math_v2/aime24/aime24_eval.json

## ⚠️ 时序约束
- 28RPM 共享配额: 现起 AIME 会与扩测 (PID 23868, 预计 3-5h) 争抢致双方 429 退避变慢. 决策: **等扩测完成再启动 AIME**.
- 扩测完成后 AIME 可立即跑, 不阻塞.

## 判据
- evolver/v2 ΔvsS0 在 AIME24 上均正且显著 → 跨任务泛化坐实, skill 非过拟合选择题
- 若 livemath 正但 AIME24 零/负 → skill 过拟合选择题形态, 泛化边界明确 (也是有价值的论文结论)

Related: [[evolver-test-generalize-20260730]], [[next-steps-20260729]]
