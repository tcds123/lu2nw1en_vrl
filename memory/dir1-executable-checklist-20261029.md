# Direction 1 三处未写改造 — 可执行清单 (2026-10-29)

前置: [[measurement-calibration-1a-20261028]] (检出下限 +0.05, D2 "0 产出" 与 0 不可区分)。
设计来源: `docs/direction1-cross-question-patch-design.md`。1′ 已完成 (见文末)。

**三处改造现状 (2026-10-30)**: A-Step2 ✅ 代码已实现 (实验归档, 见 A-Step2 段末 ⛔); B ✅ B4 否证 → B2/B3/B5 不实现; C 未写。A-Step1 (`--learn_positive --meta_contrast`) 已落地,
在 dir1_step1 中跑过 (见 [[dir1-step1-heldout-result-20261022]])。

---

## 改造 A-Step2 — 题级 staleness 负例 `make_staleness_negatives`

**为什么只有题级增量**: `make_coverage_diagnostics` (L1181) 的 tally 是**类型级** (COVERED/UNCOVERED 按
`_coverage_sig` 标签), `make_wins` (L1217) 是**正例**。缺的是"把具体已解题渲染成别弄坏"的题级负例。

**落点清单** (行号为 2026-10-29 视图, 改前需复核):

| # | 文件:位置 | 改动 | 零回归 |
|---|---|---|---|
| A2.1 | `skill_evolver_aime_v2.py` L1217 后 | 新纯函数 `make_staleness_negatives(stable_solved, focus, max_n=2) -> str`。入参 `stable_solved` = `list[{"id","item","skill_pass","s0_pass"}]`。选出 `focus` 失败面对立 (skill 已稳、S₀ 难的题), 按 id 排序确定性取前 `max_n` 题 | 纯函数, 零 API; 可单测 |
| A2.2 | 同上, `POSITIVE_SECTION` (L1139) 旁 | 新常量 `STALENESS_SECTION = "## DO-NOT-BREAK: PROBLEMS THE CURRENT SKILL ALREADY SOLVES RELIABLY\n{negatives}\nDo NOT regress these. Your edit must keep them correct.\n"`。措辞对齐"数据非规则": 只摆题与答案, 不写推导 | — |
| A2.3 | `PATCH_TEMPLATE` (L1070) 与 `PATCH_TEMPLATE_COVERAGE` (L1096) | 在 `{meta_contrast_section}` 后加一行 `{staleness_section}`。**默认传 `""`**, 渲染出空行 → 与现状字节差仅一个空行; 若要求逐字节一致, 改为在 `generate_candidates` 里字符串拼接而非模板槽 | ⚠️ 逐字节: 需把空串拼在 `## YOUR OUTPUT` 前, 用 `(sl and "\n"+sl)` 形式 |
| A2.4 | `generate_candidates` (L1274) 签名 | 加 `staleness_negatives=None` 形参; 在 L1285-1294 的 `pos_sec/meta_sec` 后: `st_sec = STALENESS_SECTION.format(negatives=make_staleness_negatives(staleness_negatives, focus)) if staleness_negatives else ""`; 两处 `.format` 补 `staleness_section=st_sec` | 默认 None → 空串 |
| A2.5 | `main` (L4525 附近, `--learn_positive` 旁) | 新 flag `--staleness`, default False | off 零回归 |
| A2.6 | `main` 循环 `collect_failures` 之后 (L5077-5100) | 构造 `stable_solved`: `[{"id":qid,"item":it,"skill_pass":1.0,"s0_pass":s0_cache[qid]["pass_rate"]} for qid 在 ds 中 skill 全对且 s0_pass<0.5]`。复用 `collect_failures` 已有 run_one 结果, **零额外 API** | — |
| A2.7 | L5193 `generate_candidates(...)` 调用 | 传 `staleness_negatives=(stable_solved if args.staleness else None)` | off → None |

**单测**: 3 题假 `stable_solved`, 断言 (a) 输出含 2 题且 id 升序; (b) `focus=None` 不崩; (c) 空输入返回 `""`。

**门槛**: 设计 §4.4 —— 仅当 A-Step1 对照 held-out 方向一致率正向才做。

**2026-10-29 复核状态 (核实后更新)**:
- **形式门槛: 过(边缘)** — A-Step1 held-out Δ=+0.0153, 2/3 seed 同向, 满足"Δ > −0.0056 且 ≥2/3 同向"。出处 [[dir1-step1-heldout-result-20261022]] L17/L25/L29 (搜不带符号的 `0.0153`/`0.0250`, 文件减号是 Unicode U+2212 非 ASCII `-`)。
- **⚠️ 但 rep5(5-iter 复测)已坐实 Direction 2 (否证 flag-steer)** — 详见 [[dir1-step1-rep5-onflag-launch-20261023]] L260-274:
  - 判据① bullet 稳定性: 目标 bullet `9c9bc27e84` **0/5**(需 ≥3/5)→ 未复现 flag steer 措辞变体过门。
  - 判据② in-sample: 唯一 PROMOTE = iter1 base-low(13/30 abandoned)偏置推上的污染 bullet `12064805af`, iter2-5 零晋升。**这正是 1′ 要堵的偏置**。
  - 判据③ held-out Δ=+0.0069, 2/3 同向 — 形式过但 **< 1a 检出下限 +0.05**, 噪声区间微正; 且极可能就是污染 bullet 的 held-out 表现(L274)。
- **⚠️ σ 口径更新 (1a, 2026-10-28)**: rep5 文件 L240/272/289/293 仍用旧尺子 `σ≈0.044`(1a 前的单 seed 口径)。1a 校准后同协议真值: `S₀@k=8 均值级 sd=0.0228` / `mean-delta SE paired=0.0253` / **检出下限 +0.05**(非 +0.13)。按新尺子 +0.0153 约为检出下限的一半, +0.0069 远低于它。方向结论不变, 但论文若用 0.044 当尺子与 1a 表格冲突。下次引用 rep5/原 ON-run 数字须标注"旧口径 σ≈0.044, 已被 1a 校准为 SE≈0.022–0.025"。

**门槛结论**: 形式过但**不应直接当"可推进 A-Step2 实验"的前提**。A2.1-A2.3 写代码 + 单测(零 API, 不依赖门槛)可立即做; **开 `--staleness` 跑实验前, 须先用 1′-fixed evolver 重跑 rep5**, 验证偏置修复是否改变 iter1 锁死这个头号失败。1′ 堵了污染机制, 但 rep5 揭示的"干净信号 <+0.05 检出下限"这点 1′ 没修 — 重跑结果未知, 是开放问题。

### ⛔ A-Step2 `--staleness` 归档 (2026-10-30) — 代码已就绪, 实验侧不跑

**代码状态 (已实现, 非待写)**: A2.1–A2.7 全部落盘 `skill_evolver_aime_v2.py` + `tf_global_v2_aime.py`,
6/6 单测过 (`_test_make_staleness_negatives.py` 🎉), `_test_eval_paired.py` 回归修 (#14), `git diff` 768 ins/217 del。
锚点: `make_staleness_negatives` L1234 / `STALENESS_SECTION` L1311 / `generate_candidates(...,staleness_negatives=None)` L1318 /
`--staleness` flag L4590 / `stable_solved` 构造 L5264-5279。**默认 off → `staleness_section` 空串 → 模板字节近不变 (零回归已验)。**

**为什么仍归档 (不跑 `--staleness` 实验)**:

1. **依赖图判据已死**: 1′-fixed rep5 落在"不变"枝 (0 晋升 / 干净信号最高 in-sample Δ=+0.044 / placebo p=0.21),
   按 L99 分叉 → "D2 结论坐实 → 实验侧暂缓, 转 generation 层修"。该分叉问的是"偏差修复后有没有干净信号可以放大", 答案**否**。
2. **`--staleness` 是 evolver 内 patcher guard, 不是信号源**: 它往 599 字 parent prompt 注"别弄坏这几题"的题级负例,
   护栏语义是**保护存量信号不被改坏**, 不制造新信号。在检出下限 +0.10 (1.96×真实SE≈0.115, 见 [[measurement-calibration-1a-20261028]] 口径修正 + #15)
   之下, 把 +0.044 推到 +0.06 仍在 placebo 带 (max +0.048) 内 → **5 轮分辨不出, 与 rep5 不可区分**。
3. **候选质量是代际问题 (B4 + rep5 共同指向)**: 0 晋升不在"选谁"层 (B4 已否证 cross_q_consistency gate;
   rep5 已否证 promote gate 加严), 而在 generation 层 (operator 提交 identity / 微调幅候选)。
   `--staleness` 改的是选择层的输入质量提示, 不到 generation 层 → 同天花板下换姿势。
4. **成本/信息比**: 代码就绪后边际成本 = ~10h API (rep5 量级), 预期产出 = 0 晋升 / Δ 在噪声带内 → **低信息高成本**。

**重启条件 (写在归档里, 解冻时照此判)**:

> generation 层改造后, 用 1′-fixed evolver 在 AIME24 rep5 **同协议**重跑, 产出 ≥1 候选满足:
> (i) in-sample Δ ≥ **+0.10** (检出下限, 1.96×真实SE≈0.115; **非** 1a 旧口径 +0.05);
> (ii) 该 Δ 超过**交叉时段 placebo 的 max** (非均值; placebo max ≈ +0.048 量级, 见 [[placebo-crossiteration-noise-ruler-20261030]])。
> 满足后 `--staleness` 作**增量重启** = 开 flag + ~10h API, **代码已就绪无需新写**, 清单 A2.1-A2.7 锚点仍有效。

**机制注**: `--staleness` 护栏语义决定它只在"有存量干净信号怕被改坏"时增益; 在信号本身低于地板时, 护栏无靶可护。
故重启判据锚定 generation 层先产出可放大的干净信号, 而非锚定 patcher 自身。

---

## 改造 B — 选择层 `cross_q_consistency` 泛化惩罚

**落点清单**:

| # | 位置 | 改动 |
|---|---|---|
| B1 | `skill_evolver_aime_v2.py` 新纯函数 (建议放 L1330 前, `bootstrap_ci_lower` import 旁) | `cross_q_consistency(per_item: dict[str, list[float]], theta: float) -> (float, bool)`: 返回 (方向一致率, 是否过门)。定义: 每题先取 `mean(ds)`; 记 `pos = #{mean>0}`, `neg = #{mean<0}`; `consistency = pos/(pos+neg)` (全零题不计分母); `pass = consistency >= theta`。**阈值口径与设计 §2 的 theta≈0.10 冲突** —— 设计里 theta=0.10 是"退化为几乎不拒", 与"淘汰噪声 patch"意图相反, 需在实现前重新定标 (见 B4) |
| B2 | promote gate (A) L3446-3453 `ci_pos = [...]; promoted_node = max(ci_pos, ...)` | `ci_pos = [c for c in sv_nodes if c.val_ci > 0 and (not use_cqc or cross_q_consistency(citem_of[c])[1])]`。**注意**: 当前 `sv_nodes` 只带 `val_delta/val_ci`, 不带 per-item → 需把 `evald` 里的 `citem` 一起带过去 (改为 `sv = [(c, citem)]`) |
| B3 | 另外三处 promote (crossover L3589 / expand L3784 / overlay L3953 区) | 同 B2 模式, 每处都有本地 per-item 变量 (`mp_item/xitem/oitem/eitem`) |
| B4 | **假设检验**(定位从"定标"修正; 2026-10-29) | 用 D2 run 的 `paired_eval_samples.jsonl` 离线算每候选的 `cross_q_consistency`, 检验它能否在 D2 数据上**区分信号与噪声**。**预期结局 (b)**: D2 全 17 候选已被 winner's-curse 诊断判为噪声(均值 +0.0088 < 1SE, 9 正 5 负围绕 0 对称, identity 候选真效应=0 却 in-sample +0.083), 若候选整体效应是噪声, per-item 方向也应 ~随机 → consistency ≈ 0.5 全员 → **找不到分离 theta**。这本身是有用的阴性结果, 把"该先修 generation 还是先修 selection"钉死。**⚠️ 口径限制**: eval_paired 下 per-item delta = `mean(cand k=8) − mean(base k=8)`, 每题**一个数**, 30 题 = 30 个 delta → consistency 只能基于这 30 个符号算, **算不了跨 rep 稳定度**(paired 数据单 rep)。设计 §2 的 theta≈0.10(2/30)是跨 rep 口径, 与 B4 题级单 rep 口径**不是同一个量, 不能直接套**。**零 API**, 跑前须写明预期, 免得事后挑 theta 解释结果 |
| B5 | flag | `--cross_q_gate`, default False; `--cross_q_theta`, default 取 B4 结果 |

### ⛔ B4 已跑完 (2026-10-29) → **B2/B3/B5 不应实现**

结果见 [[b4-result-20261029]]。一句话: 预登记判据判 (a) 有分离, **但分离几乎全部来自 identity 候选**
(`c11d7dc0c8` overlay ×5) — 它们的 cons 在 0.154~0.846 之间乱跳, 3/5 超出纯采样噪声 p95。
21 个真候选 cons 全在 [0.267, 0.600], 无一显著。identity train cons 0.846 → held-out 0.500。

**机制**: per-item delta 被题级难度主导 (base 全对的题做多少 k 都 delta=0, base 全挂的题也一样),
30 题的有效独立样本远小于 30。用一致率做门 = 把 winner's curse 从 mean-delta 搬到 consistency。

⇒ 若仍要做选择层, 必须换统计量 (block bootstrap by 题, 且同时要求 meanΔ_I > 0), 不能只用一致率。
本轮建议: **B 改造搁置**, 转向 generation 层。B1-B5 保留为本清单的"已否证设计", 不删。

**风险** (仅当有人坚持实现时仍然成立): 与 `fb1/fb2` (stagnation fallback) 交互 —— 松门阶段 `ci > -pool_tol` 的候选可能被 cqc 全拒,
使 Stage1 兜底失效。缓解: cqc 只作用于 promote (best) 选择, 不作用于 pool admission。

---

## 改造 C — held-out 哨兵 `--heldout_watch_every N`

**落点清单**:

| # | 位置 | 改动 |
|---|---|---|
| C1 | `_get_heldout_items` (L94) | 复用即可。**但它要求 `args.held_out_gate=True`** → 哨兵模式需额外条件 `(args.held_out_gate or args.heldout_watch_every)`。改 L102 判断, 保持 gate 语义不变 |
| C2 | 主循环 for t (L5044) 起始 | `if args.heldout_watch_every and t % args.heldout_watch_every == 0:` → 对**当前 best_skill** 跑 `eval_on(_heldout_items, best_skill, {}, stage="heldout_watch", k=args.held_out_k, persist_ctx={"out_dir":out_dir})`; 结果 append 到 `heldout_watch = [{"iter":t,"pass":p,"delta":d vs 上次,"ts":...}]`; 落 `out_dir/heldout_watch.json` |
| C3 | 同上 | 若 `pass < prev_pass - 0.03` (复用 `GATE_MARGIN`): `print("🟠 HELDOUT-WATCH drift ...")` + 记文件。**不回滚、不进 selection** |
| C4 | 结束时 | `evolver_log.json` 加 `"heldout_watch": [...]` |
| C5 | 泄露红线 | 哨兵结果**绝不**写回 `s0_cache` / `collect_failures` / rewriter。物理隔离同 D2 gate: 只用 `heldout_watch` stage 名 |
| C6 | 成本 | 每次 30 题 × k=8 = 240 调用; N=2 时 5 iter = 2 次 = 480 调用 (≈ D2 一个 gate 的量) |

**注意**: `eval_on` 对 `s0_cache={}` 时 delta = pass − 0 = pass, 故哨兵只看 **pass 绝对值跨轮比较**,
不看 delta。这正是"诊断指标非 selection 目标"。

---

## 依赖顺序

```
1′ (✅ 已做 2026-10-29)  →  A-Step1 结果确认 (✅ 核实: 形式过边缘, 但 rep5 坐实 D2 否证)
   ├→ A2.1-A2.3 写代码 + 单测 (零 API, 不依赖门槛, 可立即做)
   ├→ B4 离线假设检验 (零 API, 可并行)
   └→ 【跑实验前必做】1′-fixed rep5 重跑 → 验偏置修复是否改变 iter1 锁死
        ├ 翻转(iter1 不再污染/有干净晋升) → 信号或可坐实 → 再开 --staleness / B / C 实验
        └ 不变(仍锁死/干净信号 <+0.05) → D2 结论坐实 → 实验侧暂缓, 转 generation 层修
```

B4 与 1′ 都是**零 API 离线**, 可并行; A2.1-A2.3 写代码不依赖门槛; 真正烧 API 的实验步骤
(`--staleness` / `--cross_q_gate` / `--heldout_watch_every` 全开) **统一排在 1′-fixed rep5 之后**,
避免在被 rep5 否定的路线上叠加新变量。

---

## 1′ 已完成 (2026-10-29)

`tf_global_v2_aime.py` + `skill_evolver_aime_v2.py`:

1. `run_one`: 新增 `k_err` 计数 → 返回 `n_error`。区分"真 0 次通过"与"k 次全异常"。
2. `eval_on`: abandoned / `n_error>=k` 的题**排除出 delta** (原计 0); 写 `out_dir/eval_audit.jsonl`
   (原只 print, 不落盘 → 基线侧偏置不可见)。签名加 `persist_ctx=None` (默认行为不变)。
3. `eval_paired`: 任一侧 abandoned / 全异常 → **整对剔除** (两臂对齐), 不再计 0。
4. `skill_evolver_aime_v2.py`: 8 处 `eval_on` 调用点注入 `_eval_audit_ctx`。

**冒烟测试通过** (stub `run_one`): eval_on 3 题中 1 题全异常 → 分母 3→2, audit 落 1 行;
eval_paired cand 侧 x0 全异常 → "1/3 pairs dropped", 分母 3→2, 两臂对齐。

**未做**: 重排队 (re-queue) 而非剔除。剔除会让分母随超时变化, 在 n=30 下 1 题 = 分母 −3.3%;
但重排队会让单 iter 时长不可控。**当前选剔除 + 审计可见**, 待观察 eval_audit 频次后再决定是否加 re-queue。

## 关联

- [[measurement-calibration-1a-20261028]] — σ 真值与检出下限
- [[dir2-heldout-gate-result-20261027]] — D2 结论 (与 0 不可区分)
- [[dir1-step1-heldout-result-20261022]] — A-Step1 结果 (A-Step2 的门槛)
- `docs/phase2_issues.md` #12 — abandoned 偏置
