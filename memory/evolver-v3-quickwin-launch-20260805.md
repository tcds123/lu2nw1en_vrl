---
name: evolver-v3-quickwin-launch-20260805
description: 6-iter evolver + 3 quick-win flags (C1/A1/D1) 实验启动参数与对照, 2026-08-05
metadata:
  type: project
---

# evolver v3 quick-win 实验 (2026-08-05 启动)

## 动机
deep-review 发现 3 个 target-agnostic quick win (默认关, 不破坏坐实的 +16.1pp):
- **C1 fix_exploration**: `generate_candidates` 原本接 `temps` 形参但 body 从不使用 (line 201-232), 探索模式 (no_improve>=patience 时) 打印 "bolder angle hints" 却生成与正常模式相同的候选 = **死代码 no-op**. 修复: explore=True 时切到 6 条 bolder angle_hints 池.
- **A1 select_by_ci**: winner 原本 `max(novel_positive, key=lambda s: s["delta"])`, CI 仅作二元 gate. max-of-K 的 delta 有向上偏 (val-overfit, 老 10-iter val +0.291 但 test 仅 +13.7pp 没 transfer). 改为按 CI 下界选最稳.
- **D1 cache_failures**: skill 未变 (hash 同) 时复用上轮 failures, 省 ~159 call/轮.

## 启动参数
```
python -u rl/train/verl1/skill_evolver_math.py \
  --iters 6 --K 4 --eval_k 3 --train_n 35 \
  --select_by_ci --fix_exploration --cache_failures \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_v3_quickwin
```
- 起点: V2_SKILL_DEFAULT (run_20260728_131934, 与 multiseed baseline 同源)
- split: livemathematicianbench_1-3-6 (train18/val53/test106, select=train+val=71)
- s0_cache: s0_cache_math_1-3-6_k5.json (k=5 连续值固定快照)
- LiveMath=MCQ, pass=精确匹配, **纯精确匹配无 LLM 评判** → 每轮 ~852 B-call @28RPM ≈ 30min, 6 iter ≈ 4h
- background task id: belbxy8zl

## 对照基准 (已坐实)
| 配置 | val delta | test (106题) | 备注 |
|------|-----------|-------------|------|
| S0 baseline | 0 | 0.301 | 官方 SkillOpt seed |
| v2 2-iter | — | 0.478 (+16.1pp) | t=9.02, 4-seed, **主结果** |
| evolver 10-iter (老, 无 flag) | +0.291 | +13.7pp (0.438) | val-overfit, 10 iter 后停滞 |
| SkillOpt-best | — | 0.546 | 强基线 |
| evo_seed | — | 0.559 | parity p=0.42 |

## 判定标准
- 6-iter+flag test > +13.7pp (0.438) → flag 有效 (至少不伤)
- 6-iter+flag test 接近/超 +16.1pp (0.478) → flag 显著有效, 可扩到 10-iter + multiseed
- 6-iter+flag test < +13.7pp → A1(CI选)太保守 或 C1(bolder)跑偏, 关对应 flag 重试
- 关注: patience=3, 6 iter 中若 1-3 stall 则 iter4-6 触发 C1 探索 (关键测试点)

## 后续
- 若阳性 → multiseed(4 seed) 坐实 + 攻 AIME (per-topic 库 / B1-B2 信号密度)
- 见 [[algorithm-future-optimization-directions-20260805]] 的 14 点 code-level 评审

## 进度日志
- **iter1 ✅ ACCEPT** (16:25): delta=+0.136 CI=+0.072 hash b98a9abd10->34683e8aab. 4 候选全 novel: cand0(d+0.099/CI+0.052) cand1(d+0.113/CI+0.062) cand2(d+0.136/CI+0.072<-winner) cand3(d+0.061/CI+0.004). base delta+0.056. A1(CI选)与 argmax-delta 同选 cand2, 差异待后续 iter. iter1 耗时 ~1h (API ~17call/min, 延迟瓶颈非限速).
- 待续: iter2-6 + test=106 复评.
- **iter2 ✅ ACCEPT** (17:34): delta=+0.188 CI=+0.123 hash 34683e8aab->cab568a153 (cand0 胜). failures 40->33->32. 轨迹 0.056->0.136->0.188 增速放缓(收敛迹象). A1 仍与 argmax-delta 同选. ⚠️test 复评是唯一判据, 警惕 val-overfit.
- **iter3 收敛** (18:05): cand0/cand1 delta=+0.155 < iter2 bar +0.188 -> 大概率 stall. 预期 iter3-5 stall(no_improve=1,2,3), **iter6 C1 探索触发**(no_improve≥patience=3, bolder hints, 关键测试点). 最终 best 多为 iter2 skill(+0.188 val) 除非 iter6 突破. +0.188 < 老10-iter +0.291 (若 A1 更保守则 val 低但或更泛化, 待 test 裁决).
- **iter3 ✗ KEEP PREV** (18:36): no_improve=1 (4 候选 delta 全 < +0.188 bar). **M3 consolidate 触发**(t%3==0, len>4000): 4685->3905 chars, hash cab568a153->beaed23efc. ⚠️观察: consolidate **不重评 val delta** (盲压缩), iter4 failures 32->34 微升 (疑压缩掉有用片段 or k=3 噪声). 记为方法学点: M3 应加 re-eval+门控 (现 blind). iter4 在跑(focus=Regularity).
- **iter4 大突破** (19:07): cand0 d+0.258/CI+0.178/pass0.484, cand1 d+0.249/CI+0.177 -- 远超 +0.188 bar! 轨迹 0.056->0.136->0.188->(stall+consolidate)->0.258. 突破在 consolidate 之后 (压缩去冗余->patcher 拿干净材料 + Regularity focus 命中). CI+0.178 极高=稳健非走运, 利 test 泛化. A1 与 argmax-delta 仍同选 cand0. no_improve 重置->C1 探索或不再触发(不卡了). val+0.258 接近老10-iter+0.291, 但 CI 更高.
- **iter4 ✅ ACCEPT** (19:16): d+0.258/CI+0.178 hash beaed23efc->fbddc57e2f. iter5 在跑(focus=Equality). ⚠️**C1 本 run 不会触发** (iter4 跳变重置 no_improve, 仅剩 iter5-6 达不到 patience=3) -> 本 run 对 C1 零信号, 主要测 A1. C1 须更长 run(10 iter)或多 stall 才能测. 最终 best 多为 iter4 skill(+0.258) 除非 iter5/6 突破.
- **iter5 ✅ ACCEPT** (20:04): d+0.272/CI+0.204 hash fbddc57e2f->35890a250e (cand1 胜). failures 34->23 (iter4 patch 修11题!). 轨迹 0.056->0.136->0.188->0.258->0.272, 5 iter 达成, CI+0.204 极高. iter6 末轮在跑(focus=Meta-options, no_improve=0, C1 不触发). 当前 best=iter5 skill. iter6 完成后 watcher 自动接 test=106 复评(~75min, ~22:15 出结果).
- **iter6 ✗ KEEP PREV + 末次 consolidate** (20:58): cand0/1/2 全 < +0.272 bar (stall). M3 consolidate (t=6%3==0): 4986->4564 chars, hash 35890a250e->ee3f38d1ee (盲压缩未重评 val).
- **🎉 EVOLVER 完成** (20:58): init val delta +0.056 -> **best val delta +0.272 (CI=+0.204)**. skill 3655->4564 chars, hash b98a9abd10->ee3f38d1ee (产物已改动). 最终 best = consolidated iter5 skill. 6 iter 总耗时 ~5.8h (15:08-20:58).
- **test=106 复评启动** (watcher 自动衔接, 20:59): [s0] seed1 CACHED pass=0.3082; [evolver seed1] 进行中 (20/106 pr=0.33 早期). 仅评 evolver (s0 缓存), 636 call ~40min, 结果 ~21:45. ⚠️最终 skill 是盲压缩产物, 若 test 低于预期 consolidate 或为因素.
