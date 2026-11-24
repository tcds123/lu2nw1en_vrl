---
name: classeval-flash-skillopt-result-20261110
description: ClassEval flash SkillOpt 4-epoch 全量 run 结果 — flat/负 (Δbest=+0.0, Δfinal=-4.3pp), 对照 B1 flash-lcb Δ+4.9pp
metadata:
  type: project
---

# ClassEval flash SkillOpt 全量 run 结果 (2026-11-10)

## 一句话结论
ClassEval (class 级, 100 题) 上 flash SkillOpt 4-epoch = **flat/负**:
best-on-val Δ=**+0.0000** (gate 全 reject, best=initial), final/last skill Δ=**−0.0429**.
**与 B1 (flash-lcb Δ+4.9pp) 方向相反** —— 复杂任务上 skill 优化未释放杠杆, 反而 force-accept 的 slow-update 把 skill 撑到 8273 chars 导致退化.

## 实验配置
- run: `classeval-skillopt-flash-20261110-181139`
- out_root: `rl/train/verl1/outputs/crossmodel_baseline/classeval-skillopt-flash-20261110-181139/train`
- model: deepseek-v4-flash (optimizer=target=qwen_chat), reasoning=off, T=0.0, cap=16384
- data: ClassEval 100 class, ratio split 2:1:7 seed=42 → train=20/val=10/test=70
- train: epochs=4, batch=40, steps/epoch=1 (auto), total_steps=4, minibatch=8
- skill: patch mode, edit_budget=4, lr=cosine/fixed, slow_update=true(force-accept), meta_skill=true
- gate: metric=hard, eval_test=true
- skill_init: `skillopt/envs/classeval/skills/initial.md` (1849 chars)
- 启动: `launch_skillopt_flash_classeval.sh` (venv python, ratio 默认). 见 issue #52 (首次 bare --cfg-options bug, 已修).

## 结果 (test set = 70 items, split=valid_unseen)
| skill | test_hard | test_soft | Δ vs S₀ |
|---|---|---|---|
| S₀ baseline (initial) | 34/70=**0.4857** | 0.8840 | — |
| best-on-val (step 0=initial) | 34/70=**0.4857** | 0.8846 | **+0.0000** |
| final/last skill (8273 chars) | 31/70=**0.4429** | 0.8693 | **−0.0429** |

- baseline_selection(val) hard=0.7000, best/final_selection hard=0.7000 (gate 全 reject)
- steps=4 accept=0 reject=4 skip=0; best_step=0, best_origin=initial_skill
- current_origin=slow_update_epoch_04 (final skill 来自 force-accept 的 slow update, 非 gate 接受)
- wall=2191s (~36.5min), calls=496, tokens=1,866,688 (prompt=1,456,129 completion=410,559)
  - rollout 460 calls/1.32M tok (大头), analyst 15, merge 11, slow_update 3, meta_skill 3, ranking 4

## token 对比 B1
| run | wall | calls | tokens | test S₀ | Δbest | Δfinal |
|---|---|---|---|---|---|---|
| B1 flash-lcb (单函数) | 3.3h | 963 | 8M | 0.6923 | **+4.9pp** | — |
| 本次 flash-classeval (class 级) | 36.5min | 496 | 1.87M | 0.4857 | **+0.0pp** | **−4.3pp** |
(本次 wall/calls/tokens 远小于 B1: ClassEval 单 class 一次 chat 出整类, 不像 lcb 每题多次采样.)

## 根因分析 (为何与 B1 相反)
1. **gate 太粗 → 全 reject**: val=10 items, hard 指标粒度 0.1/步 (0.6↔0.7↔0.8). 4 个 candidate 的 soft 都在 0.95~0.98 (接近), hard 抖动 > 真实信号 → gate 全判 ≤0.7000 reject. best 恒留 initial → best-test delta=0 是**构造性的**, 非真改进.
2. **force-accept slow_update 反噬**: slow_update 无条件接受 (acceptance=force-accept). 它把每轮 reflect 产物堆进 skill → 1849→6122→6324→8273 chars. 这个膨胀的 final skill 在 test 上 −4.3pp: 长 skill 偏离了 initial 的简洁正确性约束, 误导模型.
3. **class 级任务 skill 杠杆未释放**: 原假设 "class 级多方法协作 → skill 边际收益更大" 未兑现. flash 已能单次出整类 (rollout soft 0.88), skill 文本对 hard 的边际 < 单函数 lcb (那里模型本身解不完整, skill 补位有效).

## 关键教训
- **force-accept slow_update 在 gate 全 reject 时有害**: 当 gate 拒绝所有 candidate, slow_update 仍 force-accept → skill 单向膨胀 → final 退化. 应在 "gate 全 reject 的 epoch" 跳过 slow_update 或回退.
- **n=10 val + hard 指标 = 信号不足**: hard 粒度 0.1, 10 题 → 1 题翻盘=0.1 抖动. 需更大 val 或用 soft 做 gate (但 soft 已 0.95+ 饱和). 这是 class 级 reward 信号的内在难题.
- **best_skill = initial ≠ "skill 无用"**: 本次 best 留 initial 是 gate 保守性的结果, 不证明 skill 文本本身无改进空间 — 只证明 4 epoch / n=10 val 配置下没找到能过 gate 的 candidate.

## 下一步候选 (待用户定)
1. **关 slow_update 的 force-accept**: 改成 gate-gated (slow_update 也过 gate), 防止 final 膨胀退化. 重跑看 final 是否守住.
2. **加大 val / 多 seed 投票**: val 10→20 或 multi-seed eval 降 hard 抖动, 让 gate 能区分 candidate.
3. **回到 B1 lcb 正收益路径**: ClassEval 负结果不否定方法, 只否定 "class 级更易显 skill". 可在 lcb 上加深 (更多 epoch / 更大 train) 巩固 +4.9pp.
4. **保留 ClassEval 作为负对照**: 记录 "复杂任务上纯文本 skill 杠杆有限", 支持论文 ablation.

## 产出物路径
- best_skill: `.../classeval-skillopt-flash-20261110-181139/train/best_skill.md` (=initial)
- summary: `.../train/summary.json`
- skills: `.../train/skills/skill_v0000~v0004`
- test_eval: `.../train/test_eval{,_baseline,_final}/` (各 70 题 results+summary)
- 关联: [[classeval-env-build-20261110]] (env 适配 + baseline 96/100), [[crossmodel-flash-baseline-result-20261102]] (B1 flash-lcb Δ+4.9pp 对照)
