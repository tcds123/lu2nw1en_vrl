---
name: evolver-full-run-start-20260801
description: evolver全量10轮长跑✅已完成(07-31 18:19启动->08-01 07:14完成, 产物hash改b98a9abd->3888edefc1, val+0.291); test=106复评2026-08-03 09:40已启动PID26211
metadata: type: project
---

# evolver 全量 10 轮长跑 - ✅ 已完成 (08-01 07:14) + test 复评进行中

## 完成状态 (2026-08-03 核实)
- 进程已退出，无残留（ps 无 evolver/eval 进程在跑）
- run 目录: `rl/train/verl1/outputs/skill_evolver_math/run_20260731_181928/`（实际启动 07-31 18:19，跑 ~13h，08-01 07:14 完成）
- 产物身份校验通过: best_skill_evolved.md md5 = `3888edefc14d8a18ee49b3811298f49b`（hash 3888edefc1）不等于 init `b98a9abd10` - evolver 真改了 skill，非原地踏步（07-29 教训已规避）
- 进化曲线（val delta, select set train+val=71题）: init +0.042 -> iter9 ACCEPT +0.291 (CI+0.229) -> iter10 KEEP_PREV（收敛在 iter9 附近）
- skill 长度: 3655 -> 4539 chars

## test=106 复评 - 2026-08-03 09:40 已启动 (后台 PID 26211)
- 命令: `/data/zhuldz/lunwen/lunwen/bin/python rl/train/verl1/eval_test_multiseed.py --skills s0,v2,evolver --evolver_best rl/train/verl1/outputs/skill_evolver_math/run_20260731_181928/best_skill_evolved.md --eval_k 3 --seeds 4`
- 日志: `rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/eval_10iter_evolver.log`
- 已确认正常启动: test=106/k=3/seeds=4; s0(hash 68571cb2)/v2(hash b98a9abd) 4-seed 全走缓存不重算; 仅新算 evolver 10-iter(hash 3888edef) seed1-4
- 缓存 key 含 skill hash, 10-iter(3888edef) 不与 2-iter(12633201) 旧缓存冲突
- 估 ~3-4h (4 seed x 106 题 x k=3 = 1272 调用, 28RPM 限速)
- 完成后: 算 evolver ΔvsS0 配对 t, 对比 2-iter +16.1pp (t=9.02, CI[+0.126,+0.196]); 10-iter 显著更高则更新主结果, 持平/退化则 2-iter 已够(收敛~iter9)

## 启动配置 (原记录，时间更正为 07-31 18:19)
- 命令: `python rl/train/verl1/skill_evolver_math.py --iters 10 --eval_k 3 --K 4 --train_n 35`
- PID: 120405 (真 python, parent bash 120402)
- 日志: rl/train/verl1/outputs/skill_evolver_math/run_weekend_full.log
- 输出目录: rl/train/verl1/outputs/skill_evolver_math/run_20260731_181928

## 参数
- 起点 skill: v2 (run_20260728_131934, 3655 chars, hash b98a9abd)
- select set: train+val = 71 题 (train_n=35 + val 36)
- eval_k=3, K=4 (每轮 4 候选), iters=10, consolidate_every=3, patience=3
- S0 cache: 71 题已加载 (现算 k=3, 非 k5 文件)
- 28RPM 限速

Related: [[evolver-test-generalize-20260730]], [[evolver-smoke-passed-20260730]], [[multi-agent-generalize-plan-20260731]], [[aime24-generalize-result-20260731]]
