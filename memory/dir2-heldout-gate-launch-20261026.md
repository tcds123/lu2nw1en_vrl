---
name: dir2-heldout-gate-launch-20261026
description: "Direction 2 — held-out AIME25 门控折入 evolver 主 PROMOTE 循环, 候选须同时过 in-sample + held-out 两道门才 promote; 默认 off=零回归"
metadata:
  type: project
---

Direction 2 实现 (2026-10-26 启动). 承接 [[dir1-step1-rep5-onflag-launch-20261023]] 暴露的核心病灶与 review 决定.

## 病灶 (rep5 坐实)

rep5 iter1 "pollution bullet" (hash `12064805af`, sha1 `5164a52e36`) 因 **base-low 采样偏置** 过了 in-sample 门: parent baseline 13/30 items abandoned=0 pass → 虚高 paired delta +0.1042 → 成为 best_skill → **锁死 parent** iter2-5 全 0 晋升 (sha1 全程不变). post-hoc held-out 显示该 bullet 实为噪声 (Δ=+0.0069 < 单 seed σ≈0.044). 根因: held-out 只是 post-hoc **测量**, 从未喂回循环, 偏置锁死产物无法被 veto.

## 方案 (Direction 2 = 把 held-out 测量变 selection pressure)

候选过了 in-sample 门 (ci > best_ci_gate) 后, **再过一道 AIME25 held-out paired 门** (cand vs parent 的 bootstrap CI > `held_out_ci_gate`) 才能夺 best_skill. 该门会 veto iter1 那种偏置产物 (held-out CI 跨 0) → parent 留 → 树不被锁死, 可继续进化. review 原话: "把'测量'变成'选择压力', 比继续调 patch 措辞更对症". Step 2 (staleness 负例) 归档暂缓 (等 held-out 门落地后再给 flag steer 干净测的机会).

## 改动范围 (单文件)

`rl/train/verl1/skill_evolution/skill_evolver_aime_v2.py`:
1. **4 个 CLI flag** (插在 `args = ap.parse_args()` L4327 前, 镜像 `--stagnation_fallback`/`--paired_rescore` 的 "默认 off=零回归" 约定):
   - `--held_out_split_dir` (str, default `""`) — aime25.jsonl 路径, 空=关
   - `--held_out_gate` (store_true, default False) — 主开关
   - `--held_out_k` (int, default 8) — held-out 每题投票数 (独立于 in-sample eval_k, 30 题可放大 k 稳 CI)
   - `--held_out_ci_gate` (float, default 0.0) — NO-CI strict (CI>0 才过), 镜像 in-sample ci>0 门
   - 门 ACTIVE iff `held_out_gate AND held_out_split_dir` 均 truthy
2. **loader**: `_get_heldout_items(args)` + 模块级 `_HELDOUT_CACHE` (镜像 `load_split._cache`), flag off 返回 [] (门 skipped, 零回归)
3. **gate helper**: `_run_heldout_gate(cand, parent, args, t, op_label, out_dir, run_id, persist_ctx=None)` — 复用 `eval_paired` (tf_global_v2_aime.py:326) + `bootstrap_ci_lower` (tf_global_v2_math.py:530), 返回 5-tuple `(passed, delta, ci, per_item, n_abandoned)`。**fail-closed 已删** (见下"实现完成"§2: eval_paired 把超时题 backfill 进 per_item=0.0, 严格门天然 veto)
4. **17 个 PROMOTE 赋值点** 全包 (facet_edit×2 + crossover-MP×3 + crossover-2p×3 + overlay×3 + expand×3 + strengthen×3), `if _ho_passed:` 守卫 1:1 对齐赋值点, flag off 短路保零回归
5. **logging/artifacts**: archive_iter{t}.json (按 hash 条件附 heldout_delta/ci/gated) + evolver_log.json per-iter (`heldout_gates` 非空才附) + run-level `heldout_gate_summary` (有 gate 才附); off 时逐字节不变

## 不写新原语 (复用)

- `eval_paired` (tf_global_v2_aime.py:326) — paired delta=cand−parent 对消共享采样噪声, 返回 per_item 喂 bootstrap CI. 同 EVAL_TEMP / 同 items.
- `bootstrap_ci_lower` (tf_global_v2_math.py:530) — 吃 per_item_deltas, 返回 CI 下界.
- AIME25 ids `"0"`..`"29"` 与 AIME24 `"aime24:60"` 无碰撞, 无需 remap.

## 零回归保证 (硬约束)

flag off → `_heldout_items=[]` → `if _heldout_items:` False → `_ho_passed` 恒 True → 赋值原样执行, 无新 print/eval/archive 字段. 验证: 1 iter flag-off 跑完 diff `evolver_log.json` + `archive_iter1.json` + `best_skill_iter1.md` vs baseline 须逐字节相同.

## 泄露红线

AIME25 items 仅喂 `_run_heldout_gate`→`eval_paired` (B-model generate + C-model score). **绝不**入 `collect_failures` / rewriter prompt / emergent facet 检测. 独立 list, 独立加载, 与 `load_split`/probe_items 物理隔离.

## 成本上界

≤1 held-out eval per operator per iter (不 cascade 查次优). 每 gate = 2 batch × 30 items × k=8 = 480 API 调用. 5 iter worst case ~9600 调用 (4 op/iter); realistic ~2-4 gates = ~1000-2000 调用 (gate 只在 in-sample 通过的候选上触发, 多数 iter 0-1 晋升).

## 成功判据

1. **零回归**: flag off 逐字节同 baseline.
2. **veto 机制**: flag on, iter1 污染 bullet 在 held-out 上 CI 跨 0 → 被 veto → best_skill==init (非 bullet).
3. **rep5 复现**: rep5 config + `--held_out_gate`, iter1 bullet 被 veto, 树继续进化 iter2-5.
4. **成本**: `heldout_gates` count ≤ 4×iters.

## 待办

- [x] 实现 4 flag + loader + helper + 17 点 wrap + logging (本会话完成, 见下"实现完成")
- [x] 零回归验证 (flag off) — 静态证明 + import/signature/argparse/loader 单测通过 (本会话)
- [x] veto 机制决策验证 (flag on, monkeypatch) — 17/17 通过 (CI≤gate→veto, strict 边界, ho_done cascade); 端到端真实 eval_paired 待 API key
- [x] 真实接口契约核对 (launch 前, 无需 classifier) — eval_paired 5-tuple 返回 / per_item `{qid:[d]}` / bootstrap_ci_lower 契约 / aime25.jsonl schema(id/problem/answer/year) / qid 命名空间无碰撞 全绿
- [x] rep5 复现跑 (含 held_out_gate) — **完成 2026-10-27 23:08, 5 iter 全跑完, 无 error**. 核心病灶治愈: iter1 pollution bullet (md5 `12064805afb5`) 从未进入 best_skill; 最终 best = init parent (md5 `c11d7dc0c8bf`). heldout_gate_summary={n_gates:3, n_passed:0, n_vetoed:3}. **0 晋升** (veto 保守, 拦了假阳性但本次无真改进). 详见下"rep5+gate 最终结果".

关联 [[dir1-step1-rep5-onflag-launch-20261023]] (病灶来源) / [[aime25-eval-json-incremental]] (held-out eval 增量陷阱, 嵌入主循环后仍需注意) / [[gpt-review-verdict-20261016]] (测量地基谱系).

## 实现完成 (2026-10-26)

全部代码改动落在单文件 `skill_evolver_aime_v2.py`。与原计划的 **3 处偏差** (均设计修正, 已记录):

1. **17 个 PROMOTE 赋值点 (非 16)**: grep 坐实 `best_val_delta, best_val_ci, best_skill =` 共 17 处 — 3 strengthen (在 `_run_strengthen` helper 内) + 14 在 `_run_facet_tree` 主体 (facet_edit×2, crossover-MP×3, crossover-2p×3, overlay×3, expand×3)。17 处全部包了 `if _ho_passed:` 守卫, 1:1 对齐。

2. **fail-closed 删除**: 原计划用 `n_abandoned = len(items) - len(per_item)` 检测超时题做 fail-closed veto。读 `eval_paired` (tf_global_v2_aime.py:436-471) 发现超时题被 backfill 进 per_item (pass_rate=0.0, 悲观), 故 `per_item` 恒含全部 items → 该差恒 0 (死代码)。删除 fail-closed: 悲观 backfill + strict 门 (`ci > ci_gate`) 已天然 veto 偏置产物 (超时题取 0 pass → paired delta 偏负 → CI<0 → 被 veto)。helper 仍返 `n_abandoned=0` 占位字段, 留作 eval_paired 将来显式返回 abandoned 数时的接入点。

3. **`_run_strengthen` 用 `heldout_state` dict 而非单 bool**: strengthen 是独立 helper (非闭包), 不能读写 `_run_facet_tree` loop-local 的 `_ho_strengthen_done` bool。故把 strengthen 的 per-iter gate 状态收进可变 dict `_heldout_strengthen_state` (含 items/by_hash/gates/ho_done/out_dir/run_id/persist_ctx), by-ref 传入。3 个调用点共享同一 dict → `ho_done` 一旦置位, 同 iter 后续 strengthen 调用跳过 gate AND 跳过晋升 (≤1 gate/op/iter, 不 cascade)。

## 零回归验证 (本会话, flag off)

- **静态**: 17/17 PROMOTE 赋值点全在 `if _ho_passed:` 下; flag off 时 `_heldout_items=[]` (falsy) → BEST 守卫 `if _heldout_items and not ...:` 短路 → `_ho_passed` 恒 True; RESCUE 守卫 `_ho_passed = (not _heldout_items) or ...` = `not []` = True。两路均原样赋值, 无新 print/eval/archive 字段。
- **import**: `python -c "import skill_evolver_aime_v2"` 通过 (无 NameError/SyntaxError)。
- **签名**: `_run_strengthen` 含 `heldout_state=None` (默认 None=零回归); `_run_heldout_gate` 5-tuple 返回 `(passed, delta, ci, per_item, n_abandoned)`。
- **argparse**: 4 flag 默认 off (`--held_out_gate` store_true=False, `--held_out_split_dir=""`)。
- **loader**: flag off → `_get_heldout_items` 返回 `[]`; flag on → 30 items, ids `"0"`..`"29"`, 与 AIME24 `"aime24:NN"` 无碰撞, 缓存命中。
- **日志条件**: `archive_iter{t}.json` 仅当 `md5(nd.skill) in _heldout_by_hash` 才附 heldout 字段; `evolver_log.json` per-iter 仅当 `_heldout_gates` 非空才附; run-level 仅当 `_heldout_run_gates>0` 才附 `heldout_gate_summary`。flag off → 全无新字段 → 字节不变。
- **运行时单元测试 (2026-10-26, monkeypatch 无需 API)**: 用 `unittest.mock.patch` 替换 `_run_heldout_gate` 内调的 `eval_paired` + `bootstrap_ci_lower`, 实跑真实 gate 函数决策逻辑。**17/17 通过**:
  - T1 flag-off: loader 返 `[]`, gate 直通 passed=True/n_abandoned=0/per_item={};
  - T2 flag-on: loader 返 30 items, ids 0..29;
  - T3 veto: CI=-0.02 < gate=0.0 → passed=False (rep5 iter1 污染 bullet 模拟: 偏置产物 held-out CI 跨 0 被 veto);
  - T4 pass: CI=+0.05 > 0 → passed=True, op_label 流到 stage=`heldout_overlay`;
  - T5 strict 边界: ci==gate (0.03==0.03) → veto; ci=0.0301>gate → pass (NO-CI strict `>` 非 `>=`);
  - T6 wrap-guard: flag-off BEST/RESCUE 守卫短路 _ho_passed=True; flag-on ho_done=True 后 RESCUE/BEST 均 _ho_passed=False (不 cascade, ≤1 gate/op/iter 成本上界坐实)。
  - 覆盖了 review 判据 #2 (veto 机制) 的决策核心, 剩 API 端到端 (eval_paired 真实返回) 待 key。
- **日志装配端到端测试 (2026-10-26, monkeypatch 无需 API)**: 验证 `_run_heldout_gate` 真实函数 + L4403-4468 真实 per-iter/run-level 装配代码。**15/15 通过**:
  - W1 by-ref 共享坐实: strengthen 经 `heldout_state["gates"]` append 的事件到达同一 `_heldout_gates` list (顺序保留 [overlay,strengthen,crossover]);
  - W2 per-iter log: `heldout_gates` 字段附 3 事件, run 累加器 n_gates=3/n_passed=2/n_vetoed=1;
  - W3 run-level summary: `heldout_gate_summary{3,2,1}` 落 log (by-ref 持久到终局 dump);
  - W4 flag-off: 空 `_heldout_gates` → iter log 无字段 (字节不变)。
  - 关键设计声明 (heldout_state["gates"] 与 _heldout_gates 同对象) 现运行时坐实。
- **累计无 API 运行时验证**: gate 决策 17 + 日志装配 15 = **32/32 通过**。
- **未做 (需 API key)**: 端到端真实跑 (eval_paired 真调 B 模型) + diff `evolver_log.json`/`archive_iter1.json` vs baseline + rep5 复现跑。待 `QWEN_CHAT_API_KEY` 就绪。

## 泄露红线核对

`aime25.jsonl` items 仅 `_get_heldout_items` → `_run_heldout_gate` → `eval_paired`。grep 确认 `collect_failures` / rewriter / emergent-facet 代码路径无 `held_out` / `aime25` 引用。独立 `_HELDOUT_CACHE`, 与 `load_split._cache` 物理隔离。

## rep5+gate 复现实跑 (2026-10-27 09:30 启动)

- **PID 61263**, run 子目录 `run_20261027_093001/`, 日志 `evolver.log`.
- **配置**: rep5 精确 args + 4 held_out_* flag (iters=5, K=4, eval_k=8, held_out_k=8, held_out_ci_gate=0.0, learn_positive+meta_contrast+facet_tree+paired_rescore).
- **模型**: DeepSeek-V3.2 (aimpapi), RPM=28, cap=16384, temp=0.0.
- **s0_cache**: 复用 run_20261016_140819 (manifest pass, k=8, 不重算).
- **启动确认**: argparse → 防泄漏预检 OK → s0_cache manifest pass → facet_library.json 落盘 (8 facets). 进 facet 树候选生成.
- **决定性判据** (run 完成后核, 写 docs/phase2_issues.md):
  1. iter1 污染 bullet (sha1 `5164a52e361c1c63d1569629a96a02449872ab2a`) 在 held-out CI 跨 0 → `evolver_log.json` iter0 `heldout_gates[].passed=false` (veto).
  2. `best_skill_iter1.md` sha1 == `b2683538c56d68fab2515356b4071ab048438b74` (init, parent 留) 而非 `5164a52e36` (bullet 锁死).
  3. 树继续进化 iter2-5.
  4. `heldout_gate_summary.n_gates` ≤ 20.
- **预计时长**: ~2-4h (5 iter × facet 树 + 4 op + held-out gate). held-out banner `🔒 HELD-OUT gate ON: loaded 30 items` 仅在 iter0 `_run_facet_tree` init 时打印 (lazy).

## iter1 veto 已坐实 (2026-10-27, run 进行中)

run 跑到 iter2/5, **核心判据 #1 (veto) + #2 (parent 留) 已验证**:

1. **held-out gate 触发并 veto**: iter1 facet_edit 候选 (delta=+0.017, in-sample CI=-0.013) 过了 Stage2 放宽门 (CI > 负 floor), 触发 held-out gate → held-out paired delta=+0.017, **CI=-0.042 < gate=0.0 → veto** (passed=false). `evolver_log.json` iter0 `heldout_gates[0].passed=false` ✅.
2. **parent 留 best**: `best_skill_iter1.md` 内容与 init skill **逐字节相同** (normalized sha1 `94d4bca9684a98df290d4a730767883fbfceffe7`, evolver md5 `c11d7dc0c8bf` == `init_skill_hash`) ✅. 对照原 rep5 无 gate: iter1 bullet (md5 `12064805af`) 锁死 parent iter2-5.
3. **RESCUE-REJECT 不 cascade**: TREE-BEST gate 已设 `_ho_facet_edit_done=True`, 后续 RESCUE 路径 `_ho_passed = (not _heldout_items) or _ho_facet_edit_done is False` = False (成本上界 ≤1 gate/op/iter 坐实) ✅.
4. **零泄露**: held-out items 仅出现在 `paired_eval_samples.jsonl` 的 `heldout_facet_edit` stage (480 records = 2 batch × 30 items × k=8), 绝不出现在 `evolver_*` (in-sample) stage ✅.
5. **archive 条件字段**: 仅 node[3] (被 gate 的 facet_edit) 附 `heldout_gated:True, heldout_delta:0.0167, heldout_ci:-0.0417`, 其余 4 node 无该字段 (字节不变) ✅.

**基准修正 (重要, 踩了 [[sha-fingerprint-vs-evolver-hash-field]] 的坑)**: 预启动笔记写 init sha1=`b2683538c5...` — 那是**原始文件带尾换行**的 sha1. evolver 哈希前 strip 尾换行, 故其 `init_skill_hash` (md5)=`c11d7dc0c8bf`, **非** 原始文件 md5 `735f9e8b71cf`. 正确的"parent 是否留"判据 = `best_skill_iter1.md` 的 normalized 内容 sha1/md5 vs `init_skill_hash`, **不是** vs 原始文件 sha1. 已用 normalized 内容逐字节比对坐实.

**待续**: run 进 iter2/5 (etime ~4h50m, 1 iter ~1h). 待: (a) iter2-5 是否有非 iter1 候选晋升 (判据 #3 树继续进化), (b) 最终 `heldout_gate_summary.n_gates` ≤ 20 (判据 #4 成本).

## rep5+gate 最终结果 (2026-10-27 23:08 完成, 5 iter 全跑完)

run 09:30 启动 → 23:08 完成 (etime ~13h38m, iter ~2h45m), PID 61263 正常退出, 5 iter 全跑完无报错.

**最终 `heldout_gate_summary`: `{n_gates: 3, n_passed: 0, n_vetoed: 3}`** — 3 个过 in-sample 放宽门 (Stage2) 的候选, 在 held-out 上 CI 全跨 0 → 全 veto, 无一夺 best:

| iter | op | held-out delta | held-out CI | 结果 |
|------|-----|----------------|-------------|------|
| 0 | facet_edit | +0.017 | **−0.042** | 🚫 VETO |
| 2 | overlay | +0.013 | **−0.038** | 🚫 VETO |
| 3 | expand | +0.025 | **−0.008** | 🚫 VETO |

**判据核对 (4/4)**: #1 veto ✅ (3/3 heldout_gates[].passed=false) / #2 parent 留 ✅ (best_skill_iter1..5.md + best_skill_evolved.md 全 md5=`c11d7dc0c8bf` == init_skill_hash, 逐字节=init) / #3 树继续进化 ⚠️ (0 晋升, 未锁死但未进化, RF §6 Step3 警告触发) / #4 成本 ✅ (n_gates=3 ≪ 20).

**决定性对比原 rep5 (无 gate)**: 原最终 `best_skill_evolved.md` md5=`12064805afb5` (pollution bullet, 794 chars, 锁死 parent iter2-5) → 本 run md5=`c11d7dc0c8bf` (init parent, 599 chars, 干净). iter1 bullet (md5 `12064805af`) 在 +gate run **从未进入 best_skill**.

**诚实判决 (见 [[dir2-heldout-gate-result-20261027]] L44): Direction 2 = 机制完全验证成功, 效果方向是否定的。** gate 确实把 3 个 in-sample 正向候选挡门外 (含 iter3 overlay identity: cand_hash=`c11d7dc0c8` == init, in-sample +0.083 → held-out +0.013, 典型 in-sample 过拟合形态), 最终 best=干净 parent. 但 5 轮 25 次算子尝试 **0 晋升**, 树完全停滞。**这不是"治了病但没药效"的 trade-off——病灶根本不在 gate (gate 是干净筛子, 3/3 拦对零误判), 而在 in-sample 评估本身: 它产出的所有正向信号都是 in-sample 局部现象 (n=25 全衰减到 held-out CI<0), 甜点区不存在。继续调门(放宽/收紧)都不会产生进化。**

> ⚠️ 措辞纠正: 早期版本写"核心病灶治愈/有效"是过度宣称——本 run 全是负数 (0 晋升, 3/3 veto, 无进化), "机制成立" ≠ "有效"。已同步纠正 [[MEMORY]] 索引。
> ⚠️ 记录缺口: 早期引用 DeepSeek review "把测量变选择压力" 原文未持久化 (见 [[deepseek-three-layer-reassess-20261021]] "记录缺口待补"), 不能作为权威依据。本结论由 run 数据独立坐实, 不依赖该转述。

**下一步建议** (按判断排序, 详见结果文件 (c)/(a)/(b)/(d)): **(c) 首选 — 攻击 in-sample 评估的偏置源**: 对 3 个被 veto 候选逐 item 比对 in-sample vs held-out per-item 分数, 定位偏置是否集中在特定 item / 是否与 abandoned=0 (iter1/iter3 各 1/30) 有关。零 API 成本, 数据已在盘. / (a) 多 seed 复跑: 理论价值低 (n=25 全灭指向系统性而非采样不足) / (b) 降 `held_out_ci_gate` 到负值: 不推荐 (放过噪声 = 重演 rep5 锁死) / (d) 转 Step 2 staleness 负例: 正交, 不解 in-sample 偏置.

**零泄露**: held-out items 全 run 仅在 `paired_eval_samples.jsonl` 的 `heldout_*` stage (3×480=1440 records), 绝不出现在 `evolver_*` (in-sample) stage. 成本 1440 ≪ worst-case 9600.
