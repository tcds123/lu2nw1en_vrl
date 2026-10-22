---
name: for-review-package-built-20260916
description: for_review/ 最小可运行包已构建并脱敏 — facet-tree 演化核心链路 4 脚本 + skillopt 子集 + split + 3break 轨迹 + 4 设计文档, 127 文件 2.2M, 无明文 key
metadata:
  type: project
---

# for_review/ 评审包构建完成（2026-09-16）

> 起因：请其他 LLM 评审当前结果/方法。打包**当前活跃的 B 模型技能进化链路**（facet-tree 演化器），不含已休眠的 GRPO 参数训练线。

## 包结构与规模

```
for_review/                        # 127 文件, 2.2M, 0 pyc/pycache
├── README.md                      # 一页定位 + 结果表 + 代码索引 + 3 review 问题
├── code/                          # 最小可运行核心链路（已脱敏）
│   ├── skill_evolver_math.py      # 演化器主算法（3998 行）
│   ├── tf_global_v2_math.py       # eval 引擎（524 行）：run_one/eval_on/bootstrap_ci/precompute_s0
│   ├── eval_test_multiseed.py     # test 复测（4-seed bootstrap）
│   ├── precompute_s0_math.py     # S₀ cache 补全
│   └── skillopt/                  # 外部依赖子集（envs/livemathmaticianbench/+model/, 168 文件 1.2M）
├── data/split_1-3-6_seed42/       # train18/val53/test106 + split_manifest.json（955K）
├── results/3break_trajectory/     # 9/2 run 幸存备份（嵌套副本抢救自 outputs/ 误删事故）
│   ├── best_skill_evolved.md      # S0 4 段 + 涌现 "Theorem-Level Precision" 段
│   ├── evolver_log.json           # 8 iter, best_val_delta 0.127→0.357
│   └── s0_cache.json              # 71 题 k=3 快照, 值分布 {1.0:6,0.0:44,0.333:14,0.667:7}
└── docs/                          # training_free_grpo_fusion_design / facet_tree_design / 进展汇报_2026-09-14 / ea-skill-evolution-survey
```

## 脱敏

- `code/*.py` 中所有硬编码 API key（`msk-4601b27a...`）→ `os.environ.get("QWEN_CHAT_API_KEY", "")` 占位符。
- 4 个核心脚本均已 grep 确认无明文 key。
- 绝对路径（`/data/zhuldz/...`）保留为真实实验配置（非机密）。
- `code/skillopt/` 子集经全量 grep 确认无硬编码 key。

## README 中固化的结果数字（已对 evolver_log.json 逐字段核实）

| skill | mean_pass (test=106) | vs S₀ |
|---|---|---|
| S₀ | 0.2814 [0.2799,0.2799,0.305,0.261] | — |
| evolver theo_iter6 (9/2 3break) | **0.5542** | Δ+0.273, CI[+0.190,+0.329], p<0.001 |
| evolver pool_filter (9/15) | 0.5645 [0.5566,0.5566,0.566,0.5786] | Δ+0.283, 4/4 seed 全正 |
| SkillOpt (对照, 参数空间) | 0.5629 | — |

evolver_log 关键字段：iter1 val_delta 0.1268 → iter8 0.3568 (CI 0.2817); baseline_val.delta 0.0235; init_skill_hash 68571cb2a6 (=S0); archive_size 42; crossover_fired 3。

## 诚实披露（README §2 + §6 Q1）

0.5645 vs 0.5542 的 **+0.0103 不能归因于 `--pool_filter` 单变量**——两条轨迹代码版本不对齐 + `expand_every` 1 vs 2。是独立从零轨迹的点估计差，非受控净效应。`--pool_filter` 独立贡献待 on/off 受控对照坐实（见 [[pool-filter-controlled-run-launch-20260916]] / [[pool-filter-correction-20260916]]，产物因 outputs/ 误删丢失，待重跑）。

## 3 个 review 问题（README §5/§6）

- **Q1 实验严谨性**：val→test 负迁移风险（val 选/test 评，8iter val_delta +0.127→+0.357 但 test 是否等价）；`--pool_filter` +0.0103 归因未闭合。
- **Q2 代码实现正确性**：s0_cache 固定快照 vs 候选现场采样对位偏差；bootstrap per-item 配对性；admission 5 峰多检验是否需 Bonferroni；git_diff_similarity 分母对长度悬殊 skill 对是否合理。
- **Q3 论文叙事**：GRPO 已休眠 3 个月，当前主线纯 facet-tree 演化——是否应去掉 GRPO 定位为 "Skill Evolution"；涌现段 vs val 失败模式结构化复述的声称强度；与 SkillOpt 对比公平性。

## 复现命令（README §4）

```bash
export QWEN_CHAT_API_KEY="<key>"
python code/precompute_s0_math.py <cache_path>          # ~1h（若无现成 cache）
python -u code/skill_evolver_math.py \
  --s0 code/skillopt/envs/livemathematicianbench/skills/initial.md \
  --init_skill code/skillopt/envs/livemathematicianbench/skills/initial.md \
  --split_dir data/split_1-3-6_seed42 --s0_cache <cache_path> \
  --iters 8 --K 4 --eval_k 3 --n_diag 8 --train_n 35 \
  --facet_tree --expand --expand_every 2 --out_dir <out_dir>
python code/eval_test_multiseed.py --skills s0,evolver --eval_k 3 --seeds 4   # ~103min
```

## 关联

[[verl1-reorg-complete-20260916]]（同步完成的 verl1 脚本整理）/ [[pool-filter-correction-20260916]]（+0.0103 归因勘误）/ [[same-window-skillopt-paired-ci-20260907]]（evolver vs SkillOpt 同窗口无定论）
