---
name: run4-measure-foundation-20261016
description: run#4 端到端启动配置 (F6+F2 ON, k=8), 验证 s0_cache 复用 + 救援可达; 计划 ~7h API
metadata:
  type: project
---

# run#4 — 修测量地基端到端验证 (2026-10-16 启动)

承接 [[measure-foundation-layerB-f2-done-20261016]] (Layer A F6 + Layer B F2 均完成并自测通过)。本文件记 run#4 启动配置 (硬性规则: 实验启动前写)。

## 启动参数 (已逐项核对 argparse, 修正 plan doc 两处错误)

plan doc `/root/.claude/plans/wise-bubbling-crayon.md` 命令有两处 flag 名错误, 已修正:
1. `--output_dir` → **`--out_dir`** (line 4050, evolver 自动建 `run_<ts>` 子目录)
2. `--rescue_upper_veto` → **`--veto_upper_bound`** (line 4221, `--rescue_upper_veto` 不存在会 argparse 报错崩)

实际命令:
```bash
source /data/zhuldz/lunwen/lunwen/bin/activate
cd /data/zhuldz/lunwen/rl/train/verl1/skill_evolution
python skill_evolver_aime_v2.py \
  --emergent_facets --facet_tree --expand \
  --facet_group_admit --subset_admit --crossover --overlay \
  --rescue_on_reject --veto_upper_bound --veto_skip_n1 \
  --iters 10 --eval_k 8 --K 4 --Kc 4 --train_n 18 --consolidate_every 3 \
  --init_skill /data/zhuldz/lunwen/data/skill_aime_initial_v2.md \
  --split_dir /data/zhuldz/lunwen/data/aime24.jsonl \
  --s0_cache /data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime_v2/efl_tree_run/run_20261013_193841/s0_cache.json \
  --out_dir /data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime_v2/run4_measure_foundation
```

## 关键 flag 说明 (vs run#3)

| flag | run#3 | run#4 | 作用 |
|------|------|------|------|
| `--eval_k` | 3 | **8** | F1 提功效 (C1): 8 次投票降采样噪声, CI 更窄 |
| `--rescue_on_reject` | (无) | **ON** | F2: 整体 CI≤0 时救援可达 else 分支 |
| `--veto_upper_bound` | (无) | **ON** | F2 veto 软化: 用 ci_upper<0 否决 (连最好情况都负才否决) |
| `--veto_skip_n1` | (无) | **ON** | F2: n<2 子集不否决 (AIME meta n=1 不再永远 -inf 阻断) |
| `--facet_group_admit` | ON | ON | 同 (group rescue 通道) |
| `--s0_cache` | 默认(不存在) | **run#3 cache** | F6: 故意指向无 manifest 的 run#3 cache → loader 拒绝 → 重算+写 manifest |

**其余与 run#3 对齐**: `--init_skill` / `--split_dir` / `--s0` (默认同 init) / `--shuffle_seed 0` / `--iters 10` / `--K 4` / `--Kc 4` / `--train_n 18` / `--consolidate_every 3`。

## F6 验证逻辑

- run#3 cache (`.../run_20261013_193841/s0_cache.json`) 已确认 **无 `_manifest`** (F6 bug 条件)。
- run#4 `--s0_cache` 指向它 → `load_s0_cache_with_manifest_aime` 返回 None → 现场重算 (k=8) + `write_manifest=True` → 落盘 `run4_measure_foundation/run_<ts>/s0_cache.json` (带 manifest)。
- **run#5** (待 run#4 完成后跑): `--s0_cache` 指向 run#4 写出的 cache + 同 `--s0`/`--eval_k 8`/`--split_dir`/`--shuffle_seed` → loader 必须打印 `manifest 校验通过` 且 **0 S0 API 调用** (重测 S₀ 的 precompute 路径不触发)。证明 F6 复用。

## 成功判据

1. **F6 复用** (run#5 验证): manifest 校验通过, 0 S0 API。
2. **F2 救援可达** (run#4 本身): `evolver_log.json` 的 `rescue_by_source` 非空 (至少一次从 rejected 分支触发救援), 日志出现 `✅ *-RESCUE-REJECT` 标记之一。
3. **geometry-group 正信号晋升** (run#4): 存在某候选整体 CI≤0 但 geometry 分组 CI>0 → 经 `group:Geometry` 救援晋升 (即使不达整体显著)。

## run#3 基线对照 (已查 evolver_log)

- run#3: `rescue_by_source: None` (从未触发 — 印证 F2 bug), `eval_k=3`, `baseline_val` 见 log。
- run#3 operator_counts 见 `evolver_log.json["operator_counts"]`。
- 注意: run#3 用 `efl_tree_run` 子目录; run#4 用 `run4_measure_foundation` (evolver 内部再建 `run_<ts>`)。

## 风险/注意

- **~7h API**: k=8 (vs run#3 的 k=3) 每 eval_on 投票翻倍, 30 题 × 多算子 × 10 轮。后台运行。
- **首 run 重算 S₀**: 因 run#3 cache 无 manifest, run#4 必然重算 (30 题 × k=8 ≈ 4-8 min)。这是预期, 非异常。
- **零回归已自测**: flags 全 OFF 时 `--self_test_facet` 全过 (含 Bug C regression OK); F2 reachability 自测 6 组全过。run#4 是 flags ON 的首跑。

## 待做 (run#4 完成后)

1. 读 `evolver_log.json` → 记 `rescue_by_source` / `operator_counts` / `baseline_val` / best_skill delta → 写 `memory/run4-results-*.md` + 更新 `MEMORY.md`。
2. 验证 `s0_cache.json` 带 `_manifest` (skill_hash/answer_hash/k/seed 在)。
3. 跑 run#5 (复用 run#4 cache) 证 F6。
4. 任何报错/异常 → 追加 `docs/phase2_issues.md`。

关联 [[measure-foundation-layerA-f6-done-20261016]] [[measure-foundation-layerB-f2-done-20261016]] [[gpt-review-verdict-20261016]]。
