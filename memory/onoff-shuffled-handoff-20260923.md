---
name: onoff-shuffled-handoff-20260923
description: ON/OFF shuffled 完整重跑交接 (会话中断落盘). ON arm ✅完成(+0.2864,无anchor), OFF arm 4/8 iters 运行中(PID 80773). 全部 detached. 下次会话接: 等 OFF 完成做最终 ON vs OFF 对比
metadata:
  type: project
---

# ON/OFF shuffled 完整重跑 — 交接 (2026-09-23 中断落盘)

## 一句话状态

完整 ON/OFF shuffled 因果重跑 (iters=8, 两臂 shuffle_seed=0, 串行) **ON arm 已完成, OFF arm 4/8 iters 运行中**. 全部 detached (PPID=1), 不受会话中断影响. 下次会话接手: 等 OFF arm 完成做最终 ON vs OFF 对比.

## 实验设计 (已落盘 docs/phase2_issues.md 2887行起)

用户 R4/R5 治本修复: `--shuffle_seed` 默认 None→0, 让 val gold 散 A-E, anchor-A skill 在 val 上拿不到正向 fitness 被自然淘汰. 完整因果对照验证: 两臂均 shuffle_seed=0 + iters=8, 看 ON(--pool_filter) 是否仍涌现 anchor-A, +15.6pp 是否再现. R4 预测: ON≈OFF, +15.6pp 坍缩.

- **唯一变量** = `--pool_filter` (ON 有, OFF 无).
- 串行 (ON 先 OFF 后): TokenBucket 模块级单例只同步进程内线程不跨进程, 并行会双吃 28RPM DeepSeek 配额.
- 启动脚本: `rl/train/verl1/skill_evolution/launch_onoff_shuffled_20260922.sh`.
- out_dir: `rl/train/verl1/outputs/onoff_shuffled_20260922/{on_arm,off_arm}`.

## 进程拓扑 (全部 detached, PPID=1)

| 组件 | PID | 状态 |
|---|---|---|
| orchestrator (bash, wait OFF) | 129280 | 活, OFF 完自动退出 |
| ON arm evolver (--pool_filter) | 129300 | ✅ 已完成退出 |
| OFF arm evolver (无 pool_filter) | 80773 | 🔄 运行中 4/8 |
| watcher (30min 心跳) | 19620 | 活, 两臂完成即退出 |

- watcher 心跳: `rl/train/verl1/outputs/onoff_shuffled_20260922/watcher_heartbeat.txt`
- 完成判定: on_arm.pid 的进程 gone + off_arm.pid 存在 + off_arm.pid 的进程 gone.

## ON arm 结果 (✅ 已完成, R4 预测确认)

- run_dir: `onoff_shuffled_20260922/on_arm/run_20260922_162135/`
- **完整 8 轮 delta**: baseline +0.0282 → iter1 +0.0657 → iter2 +0.2488 → iter3 +0.2676 → iter4/5 +0.2676 → iter6 +0.2864 → iter7/8 +0.2864. 最终 **+0.2864 (CI+0.2019)**.
- best_skill_evolved.md: 1933 chars, hash 6184ddc256 (vs initial 1018 chars). 累计新增 3 条内容性 bullet: 区分全称/存在命题 / 识别 meta-option / meta-option 验证规则.
- **grep anchor-A "option X" 锚句 = 无** ✅. 所有匹配均为内容性 phrasing (讨论 meta-option 数学含义).
- R4 预测确认: shuffled val → anchor-A 无 fitness → 演化转向内容 → 位置不变性自然满足, 无需额外门.

## OFF arm 结果 (🔄 运行中, 4/8 iters)

- run_dir: `onoff_shuffled_20260922/off_arm/run_20260923_022424/`
- baseline +0.0141 → iter1 +0.2723 → iter2 +0.3756 → iter3 +0.3803 → iter4 +0.3803 (0晋升).
- OFF iter1 跳更猛 (+0.2723, 2晋升) vs ON iter1 (+0.0657, 1晋升) — pool_filter 三门准入门在前期抑制候选通过.
- OFF delta (+0.38) 高于 ON (+0.2864), 差 ~0.11pp, 远非旧 +15.6pp 量级.
- 预计 OFF 完成 ~13:00 (每轮 ~70-82min, 剩 4 轮).

## 下次会话接手任务

1. 检查 OFF arm 是否完成: `kill -0 80773` / 读 watcher_heartbeat.txt 尾部 (off gone = 完成).
2. **最终 ON vs OFF 对比** (按硬性规则落盘):
   a. 读 `off_arm/run_*/best_skill_evolved.md` 检查是否含 anchor-A 锚句 (R4 预测 ON/OFF 都无).
   b. 对比两者 `evolver_log.json` 最终 best_val_delta: ON +0.2864 vs OFF +0.3803. +15.6pp 是否再现/坍缩 (R4 预测坍缩, 因两臂 delta 同量级 ~0.3).
   c. 写 `docs/phase2_issues.md` 最终对比段落 + 新建 `memory/onoff-shuffled-final-verdict-20260923.md` + 更新 `MEMORY.md` 索引.
3. 删 session-only cron `e1d8a296` (若会话还在; 否则随会话已死, 无需处理).
4. 若 OFF 未完成: 报进度, 等 watcher 退出.

## 已落盘记录 (无缺口)

- `docs/phase2_issues.md` 2887-2960 行: 启动设计 + F07确认 + iter1 + iter1-3轨迹 + ON完成 (5段落).
- `memory/shuffle-val-smoke-r4-confirmed-20260922.md` (冒烟 R4).
- `memory/exp-causal-anchor-rotation-20260922.md` (§8.1 评测侧因果).

## 关联

[[shuffle-val-smoke-r4-confirmed-20260922]] (冒烟) / [[exp-causal-anchor-rotation-20260922]] (§8.1) / [[phaseA-shuffle-verdict-20260917]] (旧 on/off 0.437vs0.434) / [[pool-filter-on-off-verdict-20260917]] (+15.6pp 出处) / [[criterion-shuffle-invariance-not-natural-evolution-20260922]] (判据)
