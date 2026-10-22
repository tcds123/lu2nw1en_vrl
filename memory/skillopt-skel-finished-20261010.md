---
name: skillopt-skel-finished-20261010
description: skillopt skeleton GA finished 24 steps — best=initial skill (no edit beat S0), best_skill identical to S0_AIME
metadata:
  type: project
---

## 实验完成: skillopt skeleton GA (2026-10-10 09:55 启动 → 12:18 loop 完成)

- run dir: `outputs/tf_skill_aime_v2/skillopt/aime-skillopt-skel-20261010-095515`
- launcher PID 109872, train.py PID 109891
- 配置: 24 steps, 8 epochs, train pool AIME24, selection 6 题, DeepSeek-V3.2 backend, RPM=28

## 核心结论: GA 全程未超越初始 skill

`runtime_state.json`:
- `last_completed_step`: 24 (全部跑完)
- `best_step`: **0**, `best_origin`: **"initial_skill"**, `best_score`: 0.8333 (5/6 on 6-item selection set)
- 即 STEP 1–23 的每一次 proposed edit 全部 REJECT (hard <= current=0.8333), STEP 24 同样 reject

`best_skill.md` (600 chars, md5 735f9e8b) 与 `S0_AIME` (`/data/zhuldz/lunwen/data/skill_aime_initial_v2.md`) **逐字节相同**。

## 解读

- skeleton GA 在 6 题 selection 集上 baseline 已 0.8333, 天花板效应 + 题量太少 → 任何 edit 的提升都进不了统计噪声之外, 全 reject
- 对比: 我们的 evolver (skill_evolver_aime) 在 held-out AIME25 上 = 0.5185 vs s0=0.4741 (+4.4pp, 见 aime25_test 基线)
- skillopt 的 best_skill == s0 → 在三方对比 `eval_aime25_multiseed.py --skills s0,evolver,skillopt` 中 **skillopt 臂 == s0 臂** (同文件), 不会提供新信息
  - 仍可跑 (验证 skillopt 流程正确性), 但预期 skillopt==s0

## 产物状态 (最终确认)

- ✅ train.py 已退出 (12:30 写完 `summary.json` 后干净退出), `runtime_state.json`, `best_skill.md`, `history.json` (46KB), `skills/skill_v0024.md` 全部落盘
- ✅ `summary.json` (7375B) 已产出, 含全部统计量
- `skill_final.md` 未产出 (skillopt 用 `best_skill.md` 作最终产物, 非 skill_final.md)

## summary.json 关键指标 (最终确认)

- `total_steps`: 24, `total_accepts`: **0**, `total_rejects`: **24** (每次 proposed edit 全 reject)
- `best_step`: 0, `best_origin`: "initial_skill"
- `baseline_selection_hard`: 0.8333 (5/6), `best_selection_hard`: 0.8333 (未变), `final_selection_hard`: 0.8333
- **`baseline_test_hard`: 0.6667 (4/6), `final_test_hard`: 0.6667 (未变), `test_delta_hard`: 0.0** — 内部 held-out test 集同样零提升
- `total_wall_time_s`: 9199.5 (~2.55h)
- token: 569 calls / 3.57M tokens (rollout 2.04M, analyst 738K, slow_update 340K, meta_skill 320K, merge 91K, ranking 46K)
- 配置: train_size=18, batch_size=6, num_epochs=8, steps_per_epoch=3 → 24 steps; split_ratio 6:2:2 (18 train / 6 sel / 6 test); edit_budget=4 min=2

## 两次 stall (均在 AGGREGATE 阶段)

1. 11:11–11:45 (32min): ReAct eval (RPM14) + skillopt (RPM28) 共享 28RPM key 争用 → 停 ReAct 后自愈
2. ~11:54–12:18 (~24min): 无争用 (ReAct 已停), DeepSeek endpoint 对 analyst 调用 hang, 自愈

## 下一步

- 待 train.py 确认退出 + summary.json 落盘 → 跑三方对比验证 skillopt==s0 (预期)
- 主线: 重启 ReAct agent eval (Task #3) 测 single-call→agent delta — 见 [[aime-react-eval-launched-20261010]]
