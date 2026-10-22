---
name: v2-multiseed-replication-20260727
description: 🔄3-seed 复测 v2 +15.3pp 进行中; 因与 skill_evolver_math 抢 28RPM 配额暂停, watcher 自动接力; 计划在 plan.md
metadata: type: project
---

# v2 global +15.3pp 多 seed 复测 (2026-07-27 启动, 进行中) 🔄

## 目的
0727 v2 test=0.4839 (+15.3pp) 是单次优化采样结果。项目史多次小样本假阳性→复测反转
(APPS_Plus 07-17 +6pp→07-18 -2.1pp)。必须多 seed 复测坐实。

## 关键洞察:随机性定位
- **评测端确定性**: skillopt eval_only.py greedy(temp=0.0)+固定 split seed42+test=124 →
  同一 skill 文件 test acc 可重现, 无评测噪声。
- **优化端随机**: v2 rewriter temps 0.7~0.9, distill/merge temp=0.3 → 每次全量重跑产出不同 skill。
- **复测本质**: 重跑 N 次全量 v2 → 各产出不同 skill → 各自 greedy 评测 test=124 → 看 +15.3pp 是否稳定重现。
- 无需改代码加 seed 参数: 每次 API LLM 采样自带随机性 = 一次独立 seed。

## 复测方案 (plan.md)
3 seed 串行 (28RPM 全局限制, 并行只抢配额致 429 污染):
- 复用 s0_cache (`rl/train/verl1/outputs/tf_skill_math/s0_cache_math_full.json`, 53 题) 省预计算
- 每个 seed: `python tf_global_v2_math.py --iters 3 --g 4 --train_n 35 --val_passes 2 --s0_cache ...`
- 每个产出独立 run_<ts>/tf_global_skill_v2_math.md
- 评测: `cd /data/zhuldz/skillopt && bash run_midea.sh evalskill <skill>` test=124
- 汇总: 3 test acc + 原 0.4839, mean±范围, 与 S₀ 0.3306 对位

## 阻塞: 与 skill_evolver_math 抢 API 配额 ⚠️
- 启动 seed1 时发现 `skill_evolver_math.py --iters 8 --K 3` (pid 66951, 16:38 启动) 正在跑。
- evolver iter1 val delta=+0.302 (CI+0.170), **强于 v2 iter3 (+0.222)**, 是更先进方法
  (局部编辑+多候选+具体决策程序, 目标超 SkillOpt-best 0.5161)。
- 28 RPM 是全局硬限, 两实验并行必互相 429 污染 (项目史已知坑)。
- **用户决策 (2026-07-27)**: 暂停我的 3-seed 复测, 让 evolver 先跑完。已 kill 我的 seed1, evolver 存活。

## 自动接力 (已设置)
- watcher `/tmp/wait_and_relay.sh` (pid 24085): 每 60s 检测 evolver 66951 是否结束
  → 等 5min 让 RPM 恢复 → 自动串行跑 3 seed 全量 v2 → 各产出 skill
- 日志: `/tmp/relay.log` (接力状态), `/tmp/tf_v2_math_seed{1,2,3}.log` (各 seed 详情)
- 时间: evolver iter2/8 (17:33), 预计 ~6-7h 跑完 → 接力 ~9h → 次日 ~9-10:00 全完成

## 验收标准
- ✅ 3 seed test acc 均 > S₀ 0.3306 → +15.3pp 非单次假阳性, 方法坐实
- ⚠️ 若某 seed ≤ S₀ (如门控全拒→best_S=S₀) → 诚实记录, 量化方法方差
- v2 门控已诚实 (iter1/2 拒, 仅 iter3 接受): 复测可能出现"门控全拒→test≈S₀"的 seed, 属真实方差

## 待 evolver+复测完成后做
1. 评测 3 skill (test=124 greedy, ~45min)
2. 汇总 mean±范围 + 与 S₀ 对位
3. 更新本文件为最终结论 + 更新 MEMORY.md 索引
4. 顺带记录 evolver 结果 (若超 0.5161 则是新主线)

Related: [[tf-global-v2-math-diagnosis-20260727]], [[tf-global-math-null-20260724]]
