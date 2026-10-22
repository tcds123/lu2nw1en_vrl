---
name: dir1-step1-onflags-launch-20261021
description: Direction 1 Step 1 启动—learn_positive+meta_contrast ON 镜像 run6d iters=2 16384cap, 测现存未试跨题机制是否缓解过拟合
metadata:
  type: project
---

# Direction 1 Step 1 启动: learn_positive + meta_contrast ON (2026-10-21)

## 缘起

承接 [[f7-16384-baseline-result-20261021]]: 16384 基线确认后, iter2 在 held-out 上处噪声水平中间态 (Δ=−0.0056, CI 跨零), "纯过拟合" verdict 被推翻但正向门仍 FAIL. Direction 1 (fix patcher 跨题泛化) 主攻论据变弱但仍合理. 用户授权 "先开始慢慢优化" → Step 1 (MVC, 零代码 flag 对照).

## 实验设计 (代码实证, 非 memory)

**核心问题**: run6d (产出过拟合 iter2) 的 `run6d_launch.log` banner **无 `🔧 opt-flags ON:` 行**, 且 `evolver_log.json` 的 args 确认 `learn_positive: false, meta_contrast: false` → 两 flag 均 OFF. 即 [[direction1-design-review-20261021]] 定位的"现存但未试的跨题锚/meta-contrast 机制"是真实空白 (非已被证伪的重复). Step 1 打开它们, 看 ON 的候选是否更跨题.

**flag 机制 (代码已复核)**:
- `--learn_positive`: 触发 `make_wins` (L1186) 注入 POSITIVE_SECTION — 从 skill 解对的难题蒸馏可推广决策程序 (不额外花 API, 复用 eval 已有数据). 控制点 `generate_candidates` L1254.
- `--meta_contrast`: 触发 `make_meta_contrast_diagnostics` (L1201) 注入 META_CONTRAST_SECTION — 同构 MISSED/SOLVED 对照块, 不写结论让 patcher 自归纳. collect_failures 额外收集 meta_passes (零额外 API).

## 启动参数 (镜像 run6d, 三处改动)

**镜像 run6d 的 flag 全集** (从 `evolver_log.json` args 逐字段读出):
- `facet_tree, bootstrap, crossover, overlay, expand, emergent_facets, subset_admit, facet_group_admit, rescue_on_reject, veto_upper_bound, veto_skip_n1, paired_rescore` 全 ON
- `K=4, eval_k=8, train_n=18, n_diag=8, consolidate_every=3, patience=3, Kc=4, struct_temp=1.0, expand_every=1, expand_candidates=1, efl_min_failures=2, efl_min_problems=2, shuffle_seed=0, facet_axes="Quantity Parsing,Case Enumeration,Computation & Verification,Final Answer", freeze_facet="Final Answer"`
- init_skill / s0 / split_dir / s0_cache 全同 run6d (复用 run4 s0_cache, manifest 校验通过 skill_hash=c11d7dc0c8 k=8 n=30 → F6 复用 S₀, 省 ~90min)

**三处改动 (vs run6d)**:
1. `--learn_positive --meta_contrast` (run6d OFF → 本次 ON) — **本次唯一真变量**
2. `--iters 2` (run6d=10 → 2) — MVC: run6d 唯一干净晋升在 iter2, 2 轮足以看到首个 PROMOTE 事件. 省 ~70% 时间 (设计文档 §4.3 "1-2 iter").
3. `QWEN_CHAT_MAX_TOKENS=16384` (run6d=4096 → 16384) — F7 后口径. **不 source `_env.sh`** (其 L6=4096 会覆盖), 改为 inline export; evolver L38 `setdefault` 不覆盖已 export 值. held-out 测直接对比已有 16384 基线 (s0=0.6278, ev_iter2=0.6222).

**out_dir**: `rl/train/verl1/outputs/skill_evolver_aime_v2/run_dir1_step1_onflags/` (fresh)
**launch script**: `run_dir1_step1_onflags/launch_dir1_step1.sh`
**预估**: ~2-3h (iters=2)

## 成功判据

1. **evolver 正常产出** `best_skill_iter1.md` + `best_skill_iter2.md` + `evolver_log.json` (迭代正常, 无崩溃)
2. **flag 确实生效**: log banner 应含 `🔧 opt-flags ON: dir5:learn_positive, 路径B:meta_contrast` (L4368-4369 `_flags.append`)
3. **核心科学判据** (需后续 held-out 测): 产出的 best_skill 在 AIME25 held-out (16384, k=8, seeds=3) 上 vs init 的 Δ. 对照 16384 基线: run6d iter2 Δ=−0.0056 (噪声水平). **Step 1 正向 = ON 候选 held-out Δ 明显 > −0.0056 (趋正或至少 CI 更靠正)**.

## 失败模式 (设计文档 §5)

- Step 1 无正向 (现存机制本就无效) → 接受, learn_positive/meta_contrast 非过拟合解药, 转 Direction 2/3. 不写 Step 2 (staleness 新函数), 避免无效投入.
- 注意: iters=2 可能太短看不到 PROMOTE → 若 iter1/iter2 全 KEEP-PARENT, 不一定判"机制无效", 可能只是 2 轮未命中; 需看候选 held-out 方向即便无 PROMOTE.

## 后续步骤

- [ ] 等 evolver 跑完 (~2-3h), 收集 best_skill_iter*.md
- [ ] 用 `eval_aime25_multiseed.py` (16384, k=8, seeds=3) 测产出的 best_skill vs s0 基线
- [ ] 对照 16384 基线 (run6d iter2 Δ=−0.0056) 判 Step 1 正/负
- [ ] 写结果到 memory + 更新 MEMORY.md + docs/phase2_issues.md
- [ ] 若正向 → Step 2 (写 make_staleness_negatives + {staleness_section} 槽); 若负 → 转 Direction 2/3 重判优先级

关联 [[f7-16384-baseline-result-20261021]] [[direction1-design-review-20261021]] [[aime25-heldout-iter2-nogeneralize-20261021]] [[deepseek-three-layer-reassess-20261021]] [[run6d-results-20261019]].

## ⏰ 启动确认 (2026-10-22 09:11:45)

- PID=60391, 启动成功 (此前 auto-classifier 反复拦截 `bash <script>` 模式, 由用户 `!` 前缀直接执行绕过)
- **判据1 (正常运行)**: ✅ 进程在跑, RSS 49MB, etime 58s
- **判据2 (flag 生效)**: ✅ log banner `🔧 opt-flags ON: dir5:learn_positive | 路径B:meta_contrast` (run6d 无此行 → 本次唯一真变量确认生效)
- **F6 s0_cache 复用**: ✅ `⏳ s0_cache(AIME) manifest 校验通过 (k=8, shuffle_seed=0, n=30)` → 省 ~90min, baseline 直接用 run4 s0=0.6278
- **防泄漏预检**: ✅ `🔒 防泄漏预检: init_skill=OK S₀=OK (检查面=train 18 题)`

**注**: iters=2 预估 ~2-3h. 跑完后用 `eval_aime25_multiseed.py` (16384, k=8, seeds=3) 测 best_skill_iter*.md vs s0, 对照 16384 基线 Δ=−0.0056.
