---
name: evolver-smoke-passed-20260730
description: evolver修复后冒烟通过(2轮全ACCEPT, 产物hash改变, val delta +0.038→+0.199), 待test=106复评泛化
metadata: type: project
---

# evolver 修复后冒烟通过 (2026-07-30) ✅✅✅

## 背景
07-29 修复 evolver (产物身份校验 + k投票 + 删保底回退), 07-30 冒烟暴露并修复第二个 bug: generate_candidates 把 dict 当位置参传给 keyword-only llm() 致 0 novel (详见 [[evolver-unchanged-skill-bug-20260729]] 07-30节 + docs/phase2_issues.md 末两条). 修复后重跑冒烟.

## 冒烟配置
- iters=2, eval_k=3, train_n=5, K=4
- 起点 = v2 skill (run_20260728_131934, 3655 chars, hash b98a9abd10)
- select set = train+val = 58 题, DeepSeek-V3.2 API, 28RPM
- run: rl/train/verl1/outputs/skill_evolver_math/run_20260730_111306
- 总耗时 ~4h (28RPM + 429退避严重, 单候选评测最慢~40min)

## 结果 ✅
| 阶段 | val delta | CI_lower | skill_len | hash |
|------|-----------|----------|-----------|------|
| init(v2) | +0.038 | -0.018 | 3655 | b98a9abd10 |
| Iter1 ACCEPT | +0.176 | +0.109 | 3906 | 1053fda28a |
| Iter2 ACCEPT | +0.199 | +0.124 | 4670 | 1263320145 |

- **产物身份校验通过**: best_skill_evolved.md hash (1263320145) ≠ init (b98a9abd10) — evolver 真正改了 skill, 07-29修复核心目标达成
- 两轮共 8 候选全部 novel (4/4 + 4/4), 07-30 签名 bug 修复确认有效 (修复前 0/8)
- 增益单调递增 +0.038→+0.176→+0.199, CI 下界全正且递增
- pass 0.276→0.414→0.437 (select set +16.1pp)
- Iter2 winner=cand0 delta+0.199 (cand1+0.193/cand2+0.164/cand3+0.090 均 novel+CI>0)

## ⚠️ 关键提醒: select set 信号 ≠ test 泛化
+0.199 是 select set (train+val) 上的训练信号, evolver 是 test-free 选择. 能否泛化到 test=106 未知 — 正是 v2 +15.3pp 假阳性教训 (val强但test不泛化). **必须跑 test 复评才能定论**, 不能直接当 evolver 主结果.

## 待办
1. ⭐ test=106 复评: `python rl/train/verl1/eval_test_multiseed.py --skills s0,evolver --evolver_best <run>/best_skill_evolved.md --eval_k 3 --seeds 2` (同 v2 复测条件, 配对 delta vs S0)
2. 若 test 显著正: evolver 坐实, 进机制消融 (GEPA选择/失败patch/慢更新/防退化)
3. 若 test 不泛化: 同 v2 假阳性模式, 诊断是否过拟合 train 题

## test=106 复评启动 (2026-07-30, 🔄 进行中)
- 首次启动秒完报 evolver ΔvsS0=+0.1572 — **无效**, 缓存 key 不含 skill hash, 误用 07-29 旧 evolver skill (md5 911597f7df) 的 per_evolver_seed*.json. 已修 eval_test_multiseed.py 缓存 key 加 md5[:8] 前缀 + 删旧缓存 (详见 docs/phase2_issues.md 末条).
- 重跑 PID 30147, 日志确认 `evolver: 4670 chars (hash 12633201)` 用 07-30 新 skill, 无 CACHED. 配置 k=3/2seed/test=106, 4 轮 (s0/evolver × 2seed) ≈ 2h, ~21s/题.
- 输出: rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/multiseed_test_eval.json
- ⏳ 待 2h 后查结果: evolver ΔvsS0 两 seed 均正且显著 → 泛化坐实; 否则同 v2 假阳性模式.

## 教训 (执行层)
- 28RPM 限速 + 429 退避使 evolver 极慢: 2轮冒烟~4h, 全量10轮将~20h不可接受. 正式跑须降 eval_k 或缩 select set 或题内 k 采样改并发.
- 单候选评测 (58题×k=3=174调用) 在 429 退避下最慢~40min. run_one 题内 k 次串行是瓶颈之一.

Related: [[evolver-unchanged-skill-bug-20260729]], [[multiseed-test-replicate-20260729]], [[next-steps-20260729]]
