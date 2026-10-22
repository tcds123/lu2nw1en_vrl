---
name: pertype-strengthen-run-20260908
description: 方案1 warm-start run 启动 (2026-09-08)。theo_iter6(0.5542) + --strengthen --strengthen_per_type --expand 4-iter。目标: 解锁 strengthen 在 per-type 触发 (9/5/9/7 两 run 全程未 fire 的结构性缺口)。监控判据 🔧 per-type strengthen iter{t} = fire 生效。完成后 test=106 4-seed CI 北星 CI95下界>0。
metadata:
  type: project
---

# 方案1 per-type strengthen warm-start run (2026-09-08 启动)

## 背景 (为什么做)

9/5 和 9/7 两 run 坐实: 算子H **strengthen 从未 fire**。根因结构性 — strengthen 两入口都不可达:
1. `target is None` (全覆盖) 入口: `find_uncovered_type_for_expand` 每 iter 都能找新未覆盖类型, 4 iter 内类型空间未穷尽, target 永非 None。
2. per-type 饱和入口: `type_best_delta[target]` 只在 expand 采纳时设置, 采纳即已覆盖 → find_uncovered 跳过 → 同类型不再选 → `target in type_best_delta` 恒 False (死代码)。

方案1 = 放宽 strengthen 为 per-type ADDITIVE 路径: 即使 target is not None (有未覆盖), 若存在「已覆盖但仍在失败」类型, 也 fire strengthen 加固既有专段。expand 永远优先 (主探索不被 block)。

## 代码改动 (本 session, 5 处全完成 + self-test 全绿)

`skill_evolver_math.py`:
1. 新函数 `find_covered_failing_type(counts, section_keys, exclude=None)` (L325) — 镜像 find_uncovered_type_for_expand, 找已覆盖+仍失败的非-meta 类型。exclude = too-early-abandonment guard (排除本 iter 刚 expand 的 target)。
2. `--strengthen_per_type` argparse flag (BooleanOptionalAction, default False, gated by --strengthen --expand)。
3. `_flags` 诊断打印: `方案1:strengthen_per_type`。
4. 控制流 (L2445): expand 候选循环 + type-saturated strengthen 之后, `else: target is not None` 分支内追加 per-type strengthen 调用 (复用 `_run_strengthen`, covered_target 仅 gate, select_strengthen_target 内部选段)。
5. self-test (q1-q6): covered-failing 返回最高频 / no-failure→None / skip-meta / exclude guard / deterministic / 与 find_uncovered_type_for_expand 互补 (互斥不重复消耗 API)。全绿, S₀ md5 68571cb2a6 字节复现未破坏。

## 启动参数 (cwd /data/zhuldz/lunwen/rl/train/verl1)

```bash
python3 -u skill_evolver_math.py \
  --init_skill outputs/skill_evolver_math_3break/run_20260902_205708/candidate_best_delta_theo_iter6.md \
  --s0 /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --s0_cache outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json \
  --split_dir /data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42 \
  --facet_axes "Option Comparison,Theorem-Level Precision,Hypotheses,Final Answer" \
  --freeze_facet "Final Answer" \
  --expand --expand_every 1 --expand_candidates 1 \
  --strengthen --strengthen_per_type \
  --xo_pair_retries 2 --xo_cohere_retries 1 --xo_allow_merge \
  --donor_topk 2 \
  --subset_admit \
  --iters 4 --K 4 --eval_k 3
```

= 9/7 warm-start run 同参数 + `--strengthen_per_type`。预期 ~3.5-4h (52min/iter)。background ID bm20qhec6, log /tmp/pertype_run.log。

## 预检 (全绿)

- theo_iter6 init_skill hash = 671135ff ✅ (matches test-verified 0.5542)
- true S0 initial.md hash = 9af73631 ✅
- s0_cache 存在 (3041B, 71 qids) ✅
- split_dir 含 train/val/test + split_manifest.json ✅
- imports OK (tf_global_v2_math + skill_evolver_math) ✅
- self-test 全绿 (含新 q 系列 + 既有 a-p 字节复现) ✅
- opt-flags 打印确认 `方案1:strengthen_per_type` 已注册 ✅

## 公平性论证

theo_iter6 由 3-break run 从真 S₀(initial.md, 9af73631) 进化产出, 零 skillopt 嵌套。从它 --init_skill 续跑仍是公平条件。与 9/7 run 唯一差异 = `--strengthen_per_type` flag (ADDITIVE 路径, 默认 off 字节复现既有 run)。

## 监控判据 (运行中即时)

- ✅ 正信号: `🔧 per-type strengthen iter{t}: fired on covered-failing type [X]` = 方案1 生效, strengthen 真 fire (区别于 9/7 全程零 fire)。
- ✅ best 迭代: `✅ STRENGTHEN-BEST` = strengthen 段成新 best (probe delta 超 theo_iter6 基线 +0.2586)。
- ❌ 负信号: 全程无 `🔧 per-type strengthen` = find_covered_failing_type 未返回 (查 counts 是否有已覆盖+仍失败类型, 段名签名是否匹配)。
- ⚠️ 过早放弃风险: 若 expand 候选 delta 持续走低但被 strengthen 抢预算 → 追加 type_strengthen_attempts dict 限每类型 M=2 次 (后备 guard, 不在首版)。

## 运行健康检查 (2026-09-08 ~12:05 启动 → ~12:45 iter2 完, iter3 进行中)

- baseline 完成: init skill select pass=0.596 delta=+0.371 CI_lower=+0.294 (vs 9/7 +0.390, eval 窗口方差内一致 ✓)。
- iter1 (focus=Implication): 4 候选 delta +0.305/+0.362/+0.347/+0.357 **全 < baseline +0.371** → KEEP PREV (no_improve)。**关键**: iter1 未采纳 expand → 未新增类型专段 → find_covered_failing_type 在 iter1/iter2 均无已覆盖类型可返回 (4 root 段无签名) → strengthen 正确不 fire。
- iter2 (focus=Existence): 4 候选, cand0 delta=+0.376 CI=+0.293 pass=0.601 **ACCEPT** (hash 671135ff→78dfa3af), 含 uniqueness/at-most-one 签名段名进 best_node.slots。这是首次出现「已覆盖类型」。
- **时序修正 (重要)**: per-type strengthen 首次可 fire 的 iter 是 **iter3, 非 iter2** — 因 iter2 才刚 ACCEPT Existence (counts/best_node 在 expand 后更新), strengthen 块在 expand 后执行时 best_node 已含 Existence 段, 但需 iter3 的 counts 仍报 Existence 失败才触发。iter2 当轮 target=Existence 且 exclude=Existence → 即使 gate 也不会选刚覆盖的 (too-early-abandonment guard)。
- iter3 (focus=Quantifier): expand target=Quantifier (exclude=Quantifier)。若 Existence 在 iter3 counts 仍 >0 → find_covered_failing_type 返回 Existence → **🔧 per-type strengthen iter3 fire** (决定性正信号)。cand0=+0.362 (< iter2 best +0.376), cand1-3 评中。
- 代码时序核实 (本 session): per-type strengthen 块 L2450-2463 位于 expand 候选循环 (for cand_i L2323) 之后、`else: max-sections-reached` (L2464) 之前, 无 return/continue 隔断。counts 用 expand 前 best_node 算 (L2300), best_node.slots 在 expand 采纳后已是新 node (L2398/2410)。逻辑自洽。
- 进程健康 (PID 108875): CPU 缓慢增长 (网络IO密集型), 线程池活跃。
- **诊断要点 (供未来 self)**: `futex_wait_queue_me` + 0 TCP 连接 + CPU 停滞的瞬间是**良性 inter-batch gap** (线程池等下一批 API 调用), 非卡死 — fd 数随后回升到 15-20 (活跃 socket)。判健康看 CPU 是否跨检查点增长 + 候选结果是否最终打印, 不看单次 futex 瞬态。
- cadence: ~7-8min/candidate (28 RPM cap), ~52min/iter。

## 完成后: test=106 4-seed 同窗口 paired CI (缓存复用版)

s0 (68571cb2) + skillopt (cfae156d) 两 arm 的 4-seed per_qid 缓存已存在于 theo_iter6 run dir, **hash 未变可复用**。
策略: 先把这 8 个 json 拷进新 out_dir, eval 时 s0+skillopt 全 CACHED, **只新评 evolver arm** (~53min, 4×106 items @ ~12s/item), 省 ~2/3 时间。

```bash
cd /data/zhuldz/lunwen/rl/train/verl1
NEW_RUN=outputs/skill_evolver_math/run_20260908_113107
OUT=$NEW_RUN/4seed_test_pertype
mkdir -p $OUT
# 复用 s0 + skillopt 缓存 (hash 不变 → 命中 CACHED)
cp outputs/skill_evolver_math_3break/run_20260902_205708/4seed_test_theo_iter6/per_s0_68571cb2_seed{1,2,3,4}.json $OUT/
cp outputs/skill_evolver_math_3break/run_20260902_205708/4seed_test_theo_iter6/per_skillopt_cfae156d_seed{1,2,3,4}.json $OUT/
# 新评 evolver_best (run 产物 candidate_best_delta_theo_iter4.md, hash 待定)
python3 -u eval_test_multiseed.py --skills s0,evolver,skillopt \
  --s0 /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --evolver_best $NEW_RUN/candidate_best_delta_theo_iter4.md \
  --skillopt_best /data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md \
  --split_dir /data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42 \
  --eval_k 3 --seeds 4 \
  --out_dir $OUT
```

北星判据: evolver vs skillopt per-qid bootstrap CI95 下界 > 0 = CI-gated 越过。
- skillopt 4-seed (复用缓存): seed1=0.5629 seed2=0.5314 seed3=0.5912 seed4=0.5566 → mean 0.5605 (与 [[same-window-skillopt-paired-ci-20260907]] 一致 ✓)
- 当前 theo_iter6 CI95[−3.07,+4.32] 含 0 (未越)。方案1 目标: strengthen 长出第 2 双向轴 → 收窄/越过。

✅ skillopt provenance 已坐实 (2026-09-08): `eval_test_multiseed.py:116` 读 skill 用 `.strip()`, 故 strip-md5 才是缓存 key。
- baseline 文件 = `/data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md` (9926B, mtime Jul 14 未变)
- raw-md5 = `bf82b460`, **strip-md5 = `cfae156d`** (与 prior 4-seed run log `skillopt: 9821 chars (hash cfae156d)` 完全一致)
- skillopt test=106 已有缓存 `4seed_test_theo_iter6/per_skillopt_cfae156d_seed{1..4}.json` (seed1=0.5629, seed2=0.5314), 因 hash 未变**可直接复用**, 只需新评 evolver_best arm → 省 ~2/3 评测时间。

## 🚨 根因推翻 + 重跑 (2026-09-08 ~14:38)

**上方 §"运行健康检查" 的全部分析作废** —— 该 run 走了 **flat path,方案1 代码从未执行**。

### 根因 (配置错误, 非算法 bug)

`skill_evolver_math.py:2833` 由 `if args.facet_tree:` 分派两条互斥路径:
- **facet_tree 路径** (`_run_facet_tree`): crossover/overlay/expand/strengthen 全在此内, 读结构算子标志。
- **flat 路径** (L2900+): `generate_candidates`→`cand0/1/2/3`, **完全不读** 结构算子标志, **从不 append archive**。

`run_20260908_113107` 的 `evolver_log.args.facet_tree = False` (启动命令漏 `--facet_tree`) → 走 flat path → `--expand --strengthen --strengthen_per_type` 全被静默忽略 → strengthen 永不 fire。`🔧 opt-flags ON` 打印具误导性 (只查 flag 传没传, 不查是否在生效路径)。

**9/7 run 同病**: `run_20260907_170645` 的 `args.facet_tree = False` → 同样 flat path。9/7+9/8 两 run 走 flat path 属实, 它俩对路径内触发链零检验力。

**⚠️ 二次复核纠正 (2026-09-08 后半)**: 上面原写"OLD 两入口不可达的诊断**从未被检验过 / 存疑待真**"**是错的**。9/4 run (`strengthen_run_20260904_184549.log`, PID 44625) 就是 `--facet_tree ON + --strengthen ON` 的**完整 8-iter facet_tree run**, 已检验并**坐实** OLD strengthen 两入口不可达: 全 8 iter `🔧 strengthen iter`=0; expand 触发 4 次 (iter2/4/6/8) 全 `admitted=Y` 但 `EXPAND-BEST=0`/`type-saturated=0`/`⏭️skipped=0` → expand 每次采纳新未覆盖类型段从不饱和 → `target is None` 永不达。**OLD 两入口的算法缺陷属实, 由 9/4 坐实, 不需推翻。** 9/7/9/8 只是额外无效证据。但 9/4 用的是 **OLD strengthen (无 per-type 入口)**; 方案1 的 per-type 第三入口**此前从未被任何 run 检验** (9/4 无此代码, 9/7/9/8 flat path) → 当前 PID 129051 是 per-type 入口的**首次有效检验**。

**对照佐证**: 产出 theo_iter6 的 9/2 3break run (`run_20260902_205708`) `args.facet_tree = True` → 结构算子真工作。详见 `docs/phase2_issues.md` (2026-09-08 条目, 含二次复核纠正)。

### 重跑 (2026-09-08 ~14:38 启动, PID 129051)

启动命令 = 上方原命令 + **`--facet_tree`** (唯一改动):
```bash
python3 -u skill_evolver_math.py \
  --init_skill outputs/skill_evolver_math_3break/run_20260902_205708/candidate_best_delta_theo_iter6.md \
  --s0 .../initial.md --s0_cache outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json \
  --split_dir .../livemathematicianbench_1-3-6_seed42 \
  --facet_tree \
  --facet_axes "Option Comparison,Theorem-Level Precision,Hypotheses,Final Answer" \
  --freeze_facet "Final Answer" \
  --expand --expand_every 1 --expand_candidates 1 \
  --strengthen --strengthen_per_type \
  --xo_pair_retries 2 --xo_cohere_retries 1 --xo_allow_merge --donor_topk 2 --subset_admit \
  --iters 4 --K 4 --eval_k 3
```
- RUNDIR = `outputs/skill_evolver_math/run_20260908_143853` (时间戳目录; `..._143851_FACETTREE_pertype` 是空占位名, 非实际产物), log `/tmp/pertype_facettree.log`。
- INVALID run 已重命名 `run_20260908_113107_NO_FACET_TREE_INVALID`, 保留作根因证据, 不 eval。

### iter1 实况 (2026-09-08 ~15:59, PID 129051 仍活)

- ✅ 路径确认正确: 日志有 `🌱 facet-tree init` + `Facet-Tree Iter 1/4` + facet 候选行 + 结构算子 (`🔗 overlay`)。非 flat。
- baseline delta=+0.352 pass=0.577 (与 9/7 +0.390 / 9/4 +0.023 在 eval 方差内一致 ✓)。
- facet 树 3 轴, TREE-BEST=Theorem-Level Precision (delta=+0.376 promote, overlay +0.352 admitted)。
- **🧬 expand iter1: target_type=Existence, delta=+0.3239, `admitted=true`, `became_best=false`**, 新增段 "Distinguish Existence from Uniqueness/Regularity Claims", n_sections 4→5, type_best_delta[Existence]=0.3239。
- **per-type strengthen iter1: 不 fire (正确)**: 起始 theo_iter6 4 段全无 `_TYPE_COVERAGE_SIG` 签名 → 4 非-meta 类型全未覆盖 → `find_covered_failing_type` 返回 None。与方案1 设计一致 (iter1 全未覆盖)。
- **iter2 起 per-type 有 gate 输入**: iter1 expand admitted 了 Existence 段 (签名写入 section_keys) → 若 iter2 Existence 仍在失败 → `find_covered_failing_type` 返回 Existence (排除 iter2 自身 target) → **per-type strengthen 应在 iter2 fire**。这是方案1 首次真正可 fire 窗口 (比原估 iter3 提前, 因 expand_every=1 每 iter 都 expand 且 iter1 已 admitted)。
- 监控中: PID 129051 CPU 活动 (Sl, 0.4% = 网络 wait), 已完成 iter1, 进入 iter2 expand eval 阶段。

### 路径生效判据 (重跑确认用)

- ✅ 路径正确: 日志出现 `🌱 facet-tree init` (L1913) + `Facet-Tree Iter {t}` 标题 + `facet[ax]` 候选行 + 结构算子打印 (`🔗 overlay`/`🔀 crossover`/`🧬 expand`/`🔧 strengthen`)。
- ❌ 路径错误 (flat): 日志出现 `Iter {t}/4  focus=` + `generated N candidates` + `cand0/1/2/3`, 无结构算子打印。

### 监控判据 (重跑生效后)

- ✅ 正信号: `🔧 per-type strengthen iter{t}: fired on covered-failing type [X]` = 方案1 真 fire。
- ✅ best 迭代: `✅ STRENGTHEN-BEST` = strengthen 段成新 best。
- ⚠️ 若仍不 fire: **这次才需要深挖 per-type 入口的触发链** (find_covered_failing_type 是否返回 None / 段名签名是否匹配 / exclude 是否误排), 因 per-type 代码真执行了 (OLD 两入口不可达已由 9/4 坐实, 本 run 检验的是新增的 per-type 第三入口)。

### iter1/iter2 实况 (2026-09-08 ~17:15, PID 129051 仍活, elapsed 2h59m)

- **iter1/iter2 per-type strengthen 均 0 fire** (grep `per-type strengthen` 计数=0)。
- iter1: TREE-BEST=Theorem-Level Precision (+0.376, node4); expand Existence 段 admitted=Y **became_best=False** (subset-gate 未过 floor: +0.324 < best +0.376 − 0.05) → Existence 段只进 archive (node7), 不进 best_node 链。
- iter2: TREE-BEST=Option Comparison (+0.390, node8, 4 段基础不含 Existence/Quantifier); expand target=Quantifier (admitted=Y, EXPAND-SUBSET-BEST +0.362 became_best=Y → node13 进 best_node 链, 5 段含 Quantifier 段); overlay/crossover 均 became_best=N。**per-type gate 安静跳过, 0 fire。**

### 🔴 根因坐实: gate 用 best_node.slots 失明 (第三种不可达变体)

**gate 调用点** (L2454): `find_covered_failing_type(counts, best_node.slots.keys(), exclude=target)` — 用 **best_node 当前段名 blob** 判"已覆盖"。

**复现 (决定性)**: 用真实 archive_iter2 数据:
- iter2 expand block 入口 best_node = node8 (Option Comparison TREE-BEST, 4 段: Option Comparison | Theorem-Level Precision | Hypotheses | Final Answer), blob **不含 "uniqueness"**。
- `find_covered_failing_type(counts, node8.slots, exclude=Quantifier)`: Existence (cnt=6) 签名 `('uniqueness','at most one')` 在 node8 blob 不匹配 → covered=False → skip; Quantifier 是 target 被 exclude; Meta skip; Implication/Regularity 未覆盖 → 全部 skip → **返回 None → 不 fire**。
- **对照**: 若 gate 看 **archive 全集 union** (node0-13 所有 slots key 并集, 含 node7/node12 的 "Distinguish Existence from Uniqueness/Regularity Claims"), blob 含 "uniqueness" → Existence covered=True + 仍失败 + 非 meta + 非 exclude → **返回 Existence → fire YES**。

**第三种不可达变体**:
- OLD 入口1 (`target is None` 全覆盖): 9/4 坐实, 4 iter 内不可达。
- OLD 入口2 (per-type 饱和 `target in type_best_delta and ed<...`): 死代码。
- **方案1 入口3 (per-type covered-failing)**: gate 用 best_node.slots → iter1 expand 的 archive-only 段 (became_best=False) 脱离 best_node 链 → gate 看不到已覆盖 → 失明 → 不 fire。与 OLD 同源 (都是"已覆盖"可见性缺口) 但不同形 (缺口来自 became_best=False, 非 4-iter 不饱和)。

**次级矛盾 (未完全解释)**: iter2 expand target=Quantifier 而非 Existence (按 branch_failure_types 子集汇总 Existence cnt=6 > Quantifier cnt=4)。node8 (4 段) 上 `find_uncovered_type_for_expand` 按 signature 应返回 Existence。实际返回 Quantifier 意味着全诊断 counts (collect_failures on node8.skill) 与分支子集汇总不同。此矛盾不改变 gate 失明主结论。

**待决**: iter3/4。若 iter3 best_node 仍 4 段 → gate 持续失明 → 全 run 0 fire 坐实方案1 gate 缺陷。若 iter3 expand Quantifier 段 became_best=Y 进 best_node 链 → iter4 gate 可看 Quantifier covered+仍失败 → 可能首次 fire (但加固 Quantifier 段而非收敛结构预期的 Existence↔uniqueness 第 2 轴, 方向性存疑)。

**修复方向 (Task #7)**: gate 的"已覆盖"判定从 `best_node.slots.keys()` 改成 **archive 全集 union** (扫所有 archive 节点 slots key 并集)。语义更准: "该类型已有专属 section 被任何节点采纳过" = 已覆盖, 与 best_node 是否持有无关。落实用户方法论"历史中的组合可能会更好" — archive 里的 archive-only 段是有效历史组合, 不应因未夺 best 而对 gate 不可见。详见 `docs/phase2_issues.md` (2026-09-08 方案1 gate 失明条目)。

### ✅ Task #7 修复已实施 + self-test 全绿 (2026-09-08 ~20:40, 未上线验证)

**改动** (`skill_evolver_math.py`, 3 处 + self-test):
1. 新增 `_archive_section_union(archive)` (插 find_covered_failing_type 之后, ~L361): 扫所有 archive 节点 slots key 并集, 保序去重, 返回 list。空 archive → []。
2. gate 调用点 (~L2497): `find_covered_failing_type(counts, best_node.slots.keys(), exclude=target)` → `find_covered_failing_type(counts, _archive_section_union(archive), exclude=target)`。
3. self-test (q7) 追加: q7a union 去重保序含 archive-only 段; **q7b 决定性复现** (best_node 4 段 → None 失明; union 5 段含 Existence → fire); q7c 空 archive → []; q7d 单节点 → 退化旧行为。
4. `find_covered_failing_type` 纯函数**未改** (q1-q6 不动); `find_uncovered_type_for_expand` **未改** (仍用 best_node)。

**语义论证**: strengthen gate 跨历史 (加固既有段, 应见 archive-only 段); expand gate 只看当前 best (长新段, best 没有的才需 expand)。两函数语义相反用不同 scope 合理 — 改 expand 反而错 (会让 expand 误判某类型已覆盖而跳过, 但 best 上根本没该段)。

**验证**: `python3 -u skill_evolver_math.py --self_test_facet` → ✅ 全绿含 q7 (决定性复现 gate 失明 → 修复后 fire)。

**未修版 (PID 129051) 全 run 0 fire 已成定局**: 修复代码未加载进运行中进程。当前 run 的价值 = 完整未修版基线 (iter1-4 全程 gate 失明, archive 结构清楚)。

**待办**: 修复版重跑 (同启动命令, 代码已含 union 修复) → 验证 `🔧 per-type strengthen iter{t}: fired on covered-failing type` → test=106 4-seed CI。

**向后兼容**: 不加 `--strengthen_per_type` 时 gate 块跳过 (getattr False), 旧行为零影响。加了但 archive 空 → union=[] → blob="" → None → 安全不 fire。

## 关联

[[strengthen-operator-run-20260904]] (9/4 facet_tree=True+strengthen=True 完整 8iter run, 坐实 OLD 两入口不可达; 无 per-type 入口) / [[strengthen-warm-theo6-run-20260907]] (9/7 run 同病, args.facet_tree=False) / [[same-window-skillopt-paired-ci-20260907]] (theo_iter6 同窗口 CI 含0, 本 run 要越此) / [[status-snapshot-20260905]] (strengthen 根因全景 — OLD"结构性不可达"由 9/4 坐实, per-type 第三入口待本 run 检验)

### ✅ 未修复基线 (PID 129051) 完成 + 修复版 (PID 50243) 启动 (2026-09-08 19:46/19:48)

**未修复基线最终结论 (PID 129051, run_20260908_143853, 5h02m)**:
- 全 run strengthen fire = **0** (26 archive 节点, 0 strengthen) → 方案1 gate 失明三层不可达性坐实。
- init val delta +0.352 → best val delta **+0.362** (CI=+0.288), 净提升仅 +0.010。iter1 best +0.3756 为峰值, iter2-4 均未超过。
- skill: 1549 → 3395 chars, hash 671135ff → 7f271fde。
- **反讽发现**: iter4 expand = EXPAND-BEST +Existence vs Uniqueness Distinction (delta=+0.362, pass=0.587)。这正是 strengthen 本该在 iter2 长出的 Existence↔uniqueness 第 2 轴 (theo_iter6 gap), 却靠 expand 在 iter4 才长出 → 验证 strengthen 应更早介入的必要性 (若 gate 不失明, iter2 fire 可能更早收敛且更高)。
- 产物: `outputs/skill_evolver_math/run_20260908_143853/best_skill_evolved.md` (3395 chars, hash 7f271fde, 对照用)。

**修复版 run (PID 50243, 19:48:15 启动, 同参数 + 加载 union fix)**:
- 命令: 同 129051 (theo_iter6 warm-start + `--strengthen --strengthen_per_type --expand --facet_tree --iters 4`)。
- 代码: 已含 `_archive_section_union` (Task #7), self-test 全绿 (q7 决定性复现)。
- 日志: `/tmp/repair_facettree.log`。
- 判据链: (1) 即时 fire `🔧 per-type strengthen iter2: fired on covered-failing type [Existence vs uniqueness]`; (2) `✅ STRENGTHEN-BEST` + probe delta > theo_iter6 +0.2586; (3) 北星 test=106 4-seed 同窗口 paired CI95 下界 > 0 (evolver vs skillopt cfae156d, 复用 s0 68571cb2 + skillopt cfae156d caches)。
- **不行则转方向 B** (多 parent crossover + 定向 donor, plan 已设计: `recombine_facets_multi` + `select_donors_directed` + `--xo_multiparent`/`--xo_donor_directed`)。

**用户决策链 (2026-09-08)**: "出结果后记录, 如果不行就换方向 b"。串行不并发 (北星要干净同窗口)。

### 🟡 修复版 (PID 50243) iter2: 饱和路径 fire + per-type gate 被 exclude 击中 (2026-09-08 22:32 更正 22:50)

**⚠️ 更正 (22:50)**: 先前 22:32 记录说"iter2 0 fire"是**错的**——我 grep 的是 per-type gate 专属打印 `fired on covered-failing type`, 漏看了**饱和路径**的打印 `🔧 strengthen iter{t}: ~{section}`。两条 strengthen 入口, 饱和路径 (L2504) 在 iter2 实际 fire 了。

**iter2 实际事件序列 (日志坐实)**:
1. XOVER-SUBSET-BEST iter2: Theorem-Level Precision+Existence delta=+0.366 (became_best=True, 替换 best_node)
2. overlay iter2: lost (admitted=N)
3. expand iter2 cand0: target=[Existence vs uniqueness] **saturate** (new +0.296 < prev_best +0.380) → hand to strengthen
4. **饱和路径 (L2504) fire**: `🔧 strengthen iter2: ~Theorem-Level Precision delta=+0.357 CI=+0.267 pass=0.582 admitted=Y` ← strengthen 确实触发了
5. per-type gate (L2514, ADDITIVE 路径): `find_covered_failing_type(counts, union, exclude=target=Existence)` → Existence 被 exclude 排除 → 唯一 covered-failing 类型被掏空 → **None → 不 fire**

**两条结论分开**:
- ✅ **strengthen 触发了** (饱和路径), 加固 Theorem-Level Precision 段 delta=+0.357 admitted=Y。这是 9/4/9/7/9/8(129051) 三次 run 全程 0 fire 后首次 strengthen 真触发。
- ❌ **但未成 best**: sd=+0.357 < best_val_delta=+0.413 (init select delta, 从未被超越) → 不走 STRENGTHEN-BEST → 走 subset rescue → 亦无 STRENGTHEN-SUBSET-BEST 打印 (+0.357 < +0.413-0.05=+0.363, subset_gated 或 promote=False)。**最终 best 仍=init (theo_iter6)** → 预期 test 无改善 (与 theo_iter6 +0.2586 相同)。
- ❌ **per-type gate (方案1 ADDITIVE 路径) 未 fire**: `exclude=target=Existence` 排除掉 archive union 唯一 covered-failing 类型 → 第四层死锁坐实。

**根因 (两层叠加)**:
1. `exclude=target` 第四层死锁: per-type gate 排除本 iter expand 的 target, 但该 target 恰是唯一 covered-failing 类型 (iter1 archive-only 覆盖, iter2 expand 重试 saturate) → gate 掏空 → 不 fire。saturate 的 target 是历史组合, 按 "历史中的组合可能会更好" 应对 gate 可见, exclude 违背此点。
2. **init +0.413 过强**: select(train+val) 上 init delta +0.413 从未被任何候选超越 (iter1 expand +0.380, iter2 xover +0.366, iter2 strengthen +0.357 全低于)。即便 strengthen fire 产出 +0.357, 也成不了 best → 北星 (test 改善) 无法达成。这是**比 exclude 更根本的瓶颈**: select 口径下 init 占据 best, 所有进化产出都被压在 init 之下。

**修复方向 (优先级调整)**:
- **P0 (init 过强)**: 先研究 init +0.413 为何无人超越——是否 select 口径偏置 init? 9/7 warm-start 是否也卡 init? 若 init 过强是结构性的, 单修 exclude 无用 (fire 了也成不了 best)。
- P1 (exclude): 选项 A (最小): exclude 只排除 became_best=True 的 target, 不排除 saturate (became_best=False) 的。选项 B: per-type gate 提前解耦。选项 C: 去 exclude 改 attempts dict。

**当前 PID 50243 仍跑中** (iter2 strengthen 后, ~2h46m)。等 iter3/4 看是否任何候选突破 init +0.413 成 best。若全 run best 仍=init → 方案1 在 select 口径下无效 → 需先解决 init 过强, 或直接转方向 B。

### ⚠️ 二次更正 (23:10): best 实为 xover 非 init — subset-promote floor=0.05 让低 delta 候选夺权

上节 (22:50) 记"最终 best 仍=init (theo_iter6)"**又错了**。深查 evolver_log iter2 `best_val_delta=0.3662` (低于 iter1 的 0.4131) 发现矛盾, 确认 **全局 best 已被 iter2 xover 产物覆盖, 不是 init**。

**根因**: `skill_evolver_math.py:2237-2239` (XOVER-SUBSET-BEST subset-promote 救援):
- 当候选 `xd > best_val_delta` = False (没超 init +0.4131) 但 not-meta 子集信号正向 (`_promote=True`) 且 `xd >= best_val_delta - 0.05` (floor 容忍) 时, **用候选 delta 覆盖全局 best**。
- iter2: xover +0.3662 ≥ init 0.4131 − 0.05 = 0.3631 → True → **best 从 init +0.4131 降级到 xover +0.3662**。
- 全局 best_node 变为 xo:Theorem-Level Precision+Existence (5 段), init (4 段) 被踢出 best_node 链 (但仍在 archive, 是 max delta 节点)。

**为何 archive_iter2.json max delta 仍是 init**: archive 保留全部节点 (keep-losers/MAP-Elites), init 节点 (parent=root, +0.4131) 仍在 archive。但**全局 best_node 变量**已被 subset-promote 覆盖到 xover。archive max ≠ best_node — 监控时不能只看 archive max 推断 best。

**影响 (改变北星判定)**:
1. 最终产出 `best_skill_evolved.md` = xover 产物 (select +0.3662), **不是 init**。evolver_best ≠ theo_iter6。
2. test 上 xover 行为未知 — 可能 ≠ theo_iter6 +0.2586。not-meta 子集信号正向 (ci=+0.027) 意味着 xover 在非-meta 题上有改善, **test 上可能反超 init** (subset-promote 救援的设计意图正是挑泛化好的候选)。
3. 须实测 4-seed CI 才知 xover test delta — 不能假设 = theo_iter6。

**修订决策**: run 完成后取 best_skill_evolved.md (xover, 非 init) 跑 test=106 4-seed CI。若 xover test delta 越 skillopt (+0.2649) 且 CI95 下界 >0 → 方案1 有效 (subset-promote 救援奏效, 即便 select delta 低于 init)。若未越则记录 xover test delta + 考虑方向 B。

**教训**: (1) 监控 best 须读 evolver_log `iters[t].best_val_delta` (= 全局 best_val_delta 变量 L2578, 非 per-iter), 不能只看 archive max。 (2) `iters[t].best_val_delta < iters[t-1].best_val_delta` 是 subset-promote fire (best 被降级) 的可靠信号 — 全局 best 本应单调非降, 下降必是 subset-promote 覆盖。 (3) floor=0.05 容忍度在 init 过强场景下把"略低于 init 但子集正向"的候选升成 best — 这可能是**特性非 bug** (挑泛化), 但需 test 验证。

**关联**: `skill_evolver_math.py:2237` (XOVER-SUBSET-BEST floor) / L2473 (EXPAND-SUBSET-BEST 同 floor) / L2313 (OVERLAY-SUBSET-BEST 同 floor) / L1921 (STRENGTHEN-SUBSET-BEST 同 floor) / `docs/phase2_issues.md` (2026-09-08 23:10 subset-promote 条目)
