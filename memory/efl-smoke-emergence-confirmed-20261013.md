---
name: efl-smoke-emergence-confirmed-20261013
description: EFL smoke 真实 LLM 链实证 — F8 Geometry Visualization + F9 Nested Composition Periodicity 从失败数据涌现
metadata:
  type: project
---

## 成果 (2026-10-13 14:03)

EFL smoke run (run_20261013_134050, train_n=6/iters=1/eval_k=3) 在真实 DeepSeek-V3.2 LLM 链上验证了**约束 #3 (自动涌现)**:

- `facet_library.json` 从 8 个 seed facet (F0-F7) **长出到 10 个**: F8 "Geometry Visualization" + F9 "Nested Composition Periodicity" EMERGE。
- F8 keyword_signature = `['torus','sphere','tangent','radius','revolve']`，example failure = `aime24:88` (torus revolving 圈题)。**这是几何题**，原关键词路由器会把它误判到 foci[5] (答案格式)。
- F9 = `['absolute value','trigonometric','periodicity','composition','piecewise']`，example = `aime24:63` (nested ||x|-1/2|| 周期性题)。
- seed F5 is_meta=True 正确; F8/F9 is_meta=False 正确 (非 meta)。
- F8/F9 status=candidate (cnt=1, seen=1) — smoke 样本太小未触发 promote (需 cnt≥2 且跨题 seen≥2)。

**意义**: evolver 此前完全做不到的语义发现，现在 LLM 分类器自动完成了。这是机制的端到端实证 (非 pass-rate 增益，增益待 full run)。

## smoke 未完成的部分

- smoke 被 `timeout 1500` 杀掉 (exit 124 "Terminated")，iter1 的 GA 循环 (generate_candidates/eval/LCB gate) 未跑完，故无 evolver_log.json / best_skill_evolved.md 产物。
- EFL classify 在 `collect_failures → compute_failure_distribution` (line 4270) 阶段触发，故即使 iter1 GA 未完成，facet_library 也已涌现。
- **promote gate + focus 路由命中 Geometry 未验证** — 需 full run (更多题 + 多轮累积) 才能让 F8 cnt≥2/seen≥2 promote 到 actionable，然后 flat path focus 才会选它。

## flat path 启动参数 (已查清)

- 默认: `--iters 10 --eval_k 5 --train_n 35 --K 4 --n_diag 8 --split_dir aime24.jsonl`
- flat path 产物: `best_skill_evolved.md` (line 4501) + `evolver_log.json` (每轮增量写 line 4483, line 4503 收尾)。
- **必须用绝对路径 --out_dir** (相对路径触发双重路径 bug, 见 [[aime-on-skillopt-launched-20261008]] line 58)。
- **必须 nohup 重定向文件** (tee 缓冲 stdout, 监控不可见)。
- **必须显式 --init_skill /data/zhuldz/lunwen/data/skill_aime_initial_v2.md** (V2_SKILL_DEFAULT 默认指向不存在的 run_LATEST 目录)。
- 9-30 基线对比基准: pass=0.633@k3, config `iters=10 eval_k=3 K=4` (见 [[aime-on-skillopt-launched-20261008]])。
- **不要 --ablate_antideg** (它会跳过 collect_failures, EFL 无失败可分类)。

## flat run 命令模板

```bash
cd /data/zhuldz/lunwen && source lunwen/bin/activate && cd rl/train/verl1/skill_evolution
nohup python -u skill_evolver_aime_v2.py \
  --emergent_facets --coverage_diagnosis --select_lcb \
  --init_skill /data/zhuldz/lunwen/data/skill_aime_initial_v2.md \
  --out_dir /data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime_v2/efl_flat_run \
  --iters 10 --eval_k 3 --K 4 --train_n 18 \
  > /data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime_v2/efl_flat_run.log 2>&1 &
```

## 关联

[[efl-implementation-complete-20261013]] [[efl-implementation-started-20261013]] [[aime-on-skillopt-launched-20261008]]
