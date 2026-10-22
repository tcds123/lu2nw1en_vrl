---
name: next-steps-20260729
description: 0729复测后下一步: evolver机制消融(最优先)/多seed压CI/AIME24跨任务泛化/论文主结果定稿
metadata: type: project
---

# 下一步方向 (2026-07-29 复测后, 已修订) 🎯

> ⚠️ **2026-07-29 修订**: 原 task1 "evolver 机制消融" 前提 (evolver +15.7pp 真实) 已被 [[evolver-unchanged-skill-bug-20260729]] 推翻. evolver run_20260727_163814 全 8 轮未改 skill (best==init, md5 全同 911597f7df), 原 "+15.7pp" 是两版 v2 (0727 vs 0728) 对比非 evolver 增益. 优先级重排如下.

## 当前状态快照 (修订)
- ✅ **v2 +6.6pp 是当前唯一坐实的正结果** (k=3/2seed, t=2.78 p≈0.006, CI[+0.020,+0.113]) — 见 [[multiseed-test-replicate-20260729]]
- ❌ evolver +15.7pp 推翻 (产物未改动), evolver 主结果候选撤回
- 教训: 优化器产物须先校验 hash(output)!=hash(input); val_passes=1 单次采样噪声驱动假 improved

## 待办 (按优先级, 修订后)

### 1. ⭐ 修复 evolver 让其真正改 skill (✅ 代码已改完, 待跑)
根因: `skill_evolver_math.py` 的 `val_passes=1` 单次采样 + `cands.append(best_skill)` 保底 → winner 常是 current_skill 重采样, `improved=True` 但 best_skill 重赋值为同一字符串, 8 轮原地踏步. 修复 (✅ 已完成于 skill_evolver_math.py, 详见 [[evolver-unchanged-skill-bug-20260729]] "修复已完成" 节):
- a. ✅ 评测改 k 投票连续 pass_rate (复用 v2 的 eval_k + EVAL_TEMP=0.3), 删 val_passes; candidate 与 baseline 同条件
- b. ✅ winner 必须 delta 显著 > current 且 `skill_hash != current_hash` 才接受; 保底 current 只作下界不进候选池
- c. ✅ LOST/CI 门控在 k 投票下重新有效
- d. ⏳ 修复后重跑, 先验证 best_skill md5 != init, 再 multiseed 重评测 vs v2 同条件
- 已通过 py_compile + import 测试; 冒烟因争 28RPM 配额取消, 待空闲跑
- 入口: rl/train/verl1/skill_evolver_math.py

### 2. (原1, 推迟到 evolver 修复后) evolver 机制消融
evolver 修复并坐实真增益后再做机制拆解 (GEPA选择/失败patch/慢更新/防退化). 现在做无意义.

### 3. v2 多扩 1-2 seed 压窄 CI
v2 当前 2 seed CI[+0.020,+0.113] 已显著但较宽. 扩到 3-4 seed 坐实主结果. 复用 eval_test_multiseed.py 断点续跑.

### 4. AIME24 跨任务泛化验证
v2 (及修复后的 evolver) 在 LiveMathematicianBench test 上的增益是否泛化到 AIME24 (不同题源). 非 val 筛选, 终极泛化判据. 需写 AIME24 adapter.

### 5. 论文主结果定稿 (修订)
主结果 = v2 +6.6pp (当前唯一坐实). 故事线修订:
v2 单次 +15.3pp(假阳性,评测bug) → 3-seed全拒 → 诊断评测bug → 修复 → v2 +6.6pp坐实 → (evolver 修复后若更强则补入, 否则止于 v2).
**不再**以 evolver +15.7pp 为主结果.

## 关键脚本/文件
- 评测: rl/train/verl1/eval_test_multiseed.py (k投票多seed配对, 行缓冲+断点续跑)
- evolver (待修): rl/train/verl1/skill_evolver_math.py
- v2: rl/train/verl1/tf_global_v2_math.py
- 合并归档 (含被推翻的 evolver 数据): rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/archived_combined_results.json

Related: [[evolver-unchanged-skill-bug-20260729]], [[evolver-reeval-test-20260729]], [[multiseed-test-replicate-20260729]], [[v2-eval-bug-fix-20260728]]
