---
name: facet-tree-next-experiments-20260826
description: Facet-Tree 下一步实验定稿 — 主线(pro)=Step1 N并行突变多代冲0.5464→Step2拆val找正交轴; 并行侦察(GLM)=Phase A穷举轴组合探查加性+Phase B②c诊断; 两条修正(全sub-additive≠方向判死需②c中间层/探查放开ci门k=10); 侦察已摸清代码落点
metadata:
  type: project
---

# Facet-Tree 下一步实验定稿 (2026-08-26)

## 背景
前序结论链: [[facet-tree-no-cumulative-20260825]](累积进化从未发生, best恒"单轴最优+S0其余3段") → [[facet-tree-overlay-fix-20260825]] + [[facet-tree-overlay-smoke-20260825]](修法1 overlay 实现正确但 cable sub-additive: overlay delta==单轴delta, Hypo边际≈0) → [[facet-tree-retest-result-20260825]](test 最强=1dbbb821 0.5220, 修bug后2c33e2f5退步0.4135, 判据线0.5464未闭合, gap 0.0244=2.8SE) → [[facet-tree-preset-vs-emergence-20260825]](facet轴是预设非涌现, 涌现是终局C组未兑现).

核心未解问题: **sub-additive 是数据硬约束(93% meta-option 同质)还是轴选择问题(现行3轴没切到正交信号)?** 这决定后面所有路径命运.

## 决策: 采纳「主线(pro) + 零成本并行侦察(GLM)」双线

曾争论两条路线 (pro 提的 N并行突变冲线 vs GLM 提的轴组合穷举探查), 收敛为**双线并行**:
- 主线解决"搜索方差够不够"(动搜索力, 正面打 8-25 暴露的 t=2.935 边际/修bug后 t=-5.89 退步);
- 并行侦察解决"正交轴存不存在"(动结构前提, 回答主线 Step2 要不要造新轴).

### 主线 (pro, 优先)
1. **侦察**: skill_evolver_math.py 每轴突变落点 + val 失败分布结构 (✓ 已完成, 见下「侦察结论」).
2. **Step 1**: 以 1dbbb821(0.5220) 为种子, 加「每代每轴 N 并行突变取最优」, 跑多代冲 0.5464.
   - 破线 → 搜索方差是病根, 收工;
   - 卡~0.52 → 坐实 sub-additive 硬天花板 (搜过再下"天花板"结论才硬).
3. **Step 2**: 拆 val meta-option 失败模式, 逼出第二条正交轴, 让交叉/叠加从"没用"变"有用".

### 并行侦察 (GLM, 非阻塞塞进主侦察窗口)
- **Phase A 穷举组合探查**: 复用 183940 archive 单轴改进段, 测轴组合是否有加性增益.
- **Phase B ②c 诊断** (仅 Phase A 全 sub-additive 才触发): 拆 val 失败分布, 看能否长出未预设第2条正交信号.

## 两条已接受的修正 (关键)
1. **修正一 (语义收紧)**: "全 sub-additive" 只能判「现行3轴无加性增益」, 不能判「数据里无可涌现第5轴」. 正确链路:
   `穷举探查 → 全sub-additive → ②c诊断(验证未预设正交轴) → 仍无才转方向`.
   不垫②c中间层, 会把"轴选错"误读成"数据没东西". 穷举探查只遍历预设轴, 对"未预设轴"是盲的.
2. **修正二 (CI守门定义)**: 探查测「加性」不测「donor可信度」→ 探查放开 ci>0 守门. CI 降级为「组合可信度」解读指标, 不作准入门. 原守门逻辑原样保留给正式 overlay, 探查只旁路测一组加性, 不动正式链路.
   - 落地: 绕开 select_axis_donors L888 `if nd.val_ci <= 0: continue` (它硬排除了 Theorem ci=-0.033).

## 边界约束 (钉死)
- **禁止单次 k=5 下结论** (8-25 栽过 t=2.935 边际). 组合测试至少 k=10 或 3-seed.

## 侦察结论 (代码落点已摸清, 2026-08-26)
- 文件: `rl/train/verl1/skill_evolver_math.py` (1798行).
- **Stage0 每轴每代只产 1 个 child** — 突变循环 L1063-1101: 单 slot patcher (L1063-1072) → facet_edit_guard → compose → eval_on → 精英晋级. Step1 的 N 并行突变插入点即此块 (每轴并行 N 个 child 取 delta 最优, 再走既有晋级逻辑).
- **overlay 组装** L1166-1210: `select_axis_donors`(L865) 选 donor → L1172-1176 组装 → eval_on(k=eval_k) → admitted(ci>0 且 pass≥base-0.06).
- **select_axis_donors 三道门槛** (L884-894): facet_label∈axes / val_ci>0 / 本轴段md5≠root. 第二道 ci>0 守门硬排 Theorem → Phase A 旁路入口.
- **probe 重建** (零歧义): split_dir=`livemathematicianbench_1-3-6_seed42`, probe_seed=42, probe_ratio=0.5, train_n=35, 切分随机种子固定 → `_idx[:_np]` 前36题 (L1412-1417). 同 seed+split_dir 重建即逐题一致.
- **评测口径复现**: `eval_on(probe_items, skill, s0_cache, stage, k)` + `bootstrap_ci_lower(item)`, 原 run 同函数, 苹果对苹果.
- **183940 数据齐备**: `outputs/skill_evolver_math/run_20260824_183940/{archive_final,tree_iter1,tree_iter2,evolver_log}.json`.
- **单轴 donor 段 (183940 archive 实读, 2026-08-26 订正)**: 注意 archive 只存 `skill` 全文不存 `slots` — Phase A 脚本必须 `facet_decompose(skill)` 逐个重建各轴段, 不能假定有现成 slots 字段。实读 10 个节点:
  - 3 个 `root` 叶子 (delta=+0.0056, ci=-0.0333, hash 68571cb2a6) = init_skill(=S0) 的根, 各轴段均 = S0 原文, 是 md5-vs-root 门要排掉的「未变异」叶子。
  - Opt-Comp: 2c33e2f5 **delta=+0.1944 ci=+0.1222** (gen1 best) / da155f354e delta=+0.1222 ci=+0.0556 (gen2)。
  - Hypo: 2c5e3b006d **delta=+0.1389 ci=+0.0389** (gen1) / 1d6df56eff delta=+0.1167 ci=+0.0611 (gen2)。
  - Theorem: 6501d6edc1 delta=+0.0167 ci=-0.0389 (gen1) / 483335ada1 delta=+0.0000 ci=-0.0333 (gen2) — **全 ci<0, 正式 overlay 守门会拒, Phase A 必须旁路**。
  - xo:Hypo+Opt-Comp 3d3100d897 delta=+0.1889 ci=+0.1167 (交叉产物)。
  - ⚠️ 上条旧记忆里「Opt-Comp +0.344 / Theorem +0.200 / Hypo +0.194」是 **1dbbb821 retest** 的 delta, 不是 183940。183940 正确值见上。基线 183940 = pass 0.1833 / delta +0.0056 / ci -0.0333。
- **1dbbb821 4-seed 数据在**: `outputs/multiseed_test_stage0_retest_20260825/per_evo_seed_1dbbb821_seed{1-4}.json`.

## 口径 (已拍板 2026-08-26)
Phase A 的 k 与 baseline 口径:
- **(a) 全部统一 k=10 重测** — 4组组合 + 各单轴 baseline + s0 全部用 k=10 同次重测. ✅ 已定.
- 否决 (b): combo 用 k=10 baseline 沿用 archive k=5 是 k 口径混用, = [[facet-tree-retest-result-20260825]] 第4条"基线选错论背离"的同型错误翻版, 会系统性高估 combo 的"加性" → 废掉 Phase A 决策价值. 多~30% API 换判据线可信度, 值得.
- **判据线重算 (钉死)**: Phase A 的"combo 要超越的判据线"= **同次 k=10 重测的 Opt-Comp 单轴 delta**, 非 archive 历史 k=5 的 0.344. 否则 combo(k=10) vs 判据线(k=5) 仍是同型口径混用. combo 与判据线同次同口径重测, 连"历史值"层都不掺.

## Phase A 正确实现对照表 (预读后核对 GLM 产出用, 2026-08-26)

### A. 数据源与段重建（最容易错的坑）
1. archive **只存 `skill` 全文**, 无 `slots` 字段。每轴段必须 `facet_decompose(skill)` 重建, 取 `slots[axis]`。不能假定 `node["slots"]` 存在。
2. root 叶子 = init_skill=S0 (hash 68571cb2a6), 各轴段逐字节 = S0。**必须用 md5(段) != md5(root_slots[axis]) 门排掉**, 否则会把 S0 原文段当"该轴改进"叠进 combo (虚高加性)。
3. 4 段固定 key: `Option Comparison` / `Theorem-Level Precision` / `Hypotheses` / `Final Answer`(freeze)。`facet_decompose` 返回 `(title, slots)`, title=`# Live Mathematical MCQ Heuristics`。combo 只动前 3 段, Final Answer 保持 S0。
4. 判据线 = 同次 k=10 重测的 **Opt-Comp 单轴 delta**, 非 archive 历史 +0.1944。

### B. k 口径（钉死, 不得混用）
5. **4 组 combo + 各单轴 baseline + S0 全部 k=10** (口径(a), 已拍板)。任何地方出现 `k=5` 评测都错。
6. `s0_cache` 必须用 **k=10** 版; 现存 `s0_cache_math_1-3-6_k5.json` 是 k=5, 不能直接喂。应现场 `precompute_s0(select_items, s0, k=10)` 重建, 存 `s0_cache_math_1-3-6_k10.json`。
7. delta 对位必须是 `pass_rate - s0_cache[qid]["pass_rate"]`, 且 s0_cache 是 k=10 快照 (同 `eval_on` 的口径)。

### C. 旁路 ci>0 守门（修正二的核心）
8. 探查测「加性」不是「donor 可信度」→ **不放 select_axis_donors 的 `if nd.val_ci <= 0: continue`** (L888)。Theorem 两代 ci<0 必须能进组合。CI 只作「组合可信度」解读, 不作准入门。
9. 若误抄 select_axis_donors, Theorem(ci=-0.0389/-0.0333) 会被硬排除 → 只能穷到 2 轴, 结论「无加性」是漏轴假阴性。这是 Phase A 最大的单一 bug 风险点。

### D. probe 重建（苹果对苹果）
10. `split_dir = livemathematicianbench_1-3-6_seed42`, `train_n=35`, `probe_seed=42`, `probe_ratio=0.5`, 切分用 `random.Random(42).shuffle`。→ `_idx[:_np]` = 前 36 题。偏离任何一项 → probe 题集与 183940 不一致, combo delta 不可比。
11. 评测函数必须用同一 `eval_on(items, skill, s0_cache, stage, k=10)` + `bootstrap_ci_lower(per_item)` (import 自 `tf_global_v2_math`), 不要自写一套 sampling/CI, 否则口径漂移。

### E. 组合枚举完整性（别漏对照组）
12. 单轴 baseline 对照组必须全覆盖: Opt-Comp 单轴 / Theorem 单轴 / Hypo 单轴 / S0 各重测一次。缺任何一轴, combo 的「加性增益」无从比对。
13. 组合组覆盖 3 轴连 2 轴 = Opt+Theo / Opt+Hypo / Theo+Hypo 3 组 pair + 1 组 3 轴全叠 = 4 组 combo。每组 = 取各轴"该轴 delta 最高且 md5≠root"的段叠到 S0 上其余段, Final Answer 不动。
14. 加性判据 = combo_delta vs 组成各单轴 delta 之和 (或 vs 组内最大单轴 delta)。用同次同 k=10 的 delta, 不用历史值。

### F. 边界（总闸）
15. **禁止单 k=5 下结论**; combo 至少 k=10 (或 3-seed)。8-25 栽过 t=2.935 边际, 这是硬约束。

## Phase A 运行状态 (2026-08-26, live)
- 脚本: `/tmp/eval_axis_combo.py`. PID 94009, harness 真后台 (非 nohup, 不会被 reap).
- 进度: S₀ cache done in 1236s (360调用=**17.4/min**), mean pass=0.1972 (k=10 fresh). s0 eval 组 done in 1903s (360调用=**11.4/min**), delta=-0.0111 CI=-0.0500. 现在第 2/8 组 (single:Option).
- ⏳ **ETA 修正**: 实测 eval_on 吞吐 11.4/min (非预估 28RPM). 余 7 组×360=2520 调用 → 11.4/min 下 ~3.7h, 17.4/min 下 ~2.4h. 取 **~2.5-3.5h 余** (远慢于"2-2.5h"旧估). 见下吞吐根因.
- IO 健康 (非死锁). 0 ESTABLISHED 是 call 间隙快照, 非断连.
- 结果落盘: `rl/train/verl1/outputs/axis_combo_probe_20260826/axis_combo_probe_result.json`.

### ⚠️ 吞吐根因 (实测, 2026-08-26) — 影响 Step1 成本估算
- `tf_global_v2_math.py` eval_on/precompute_s0 确有 `ThreadPoolExecutor(max_workers=40)` (L294/327), "40 并发"非虚构.
- `qwen_backend.py` `_TokenBucket(rate=28/60, capacity=burst=2)` (L119-150) 是**限流器非并发数**: acquire() 消耗一个 token 后立即返回, token 按 0.467/s 回填, 不锁住整个调用时长 → 40 线程可同时 acquire, 上限 28/min.
- 但实测 11.4-17.4/min << 28/min → **真瓶颈是 API 端 (延迟/retry 退避), 非客户端桶**. qwen_backend.py:331 `sleep(30+2^attempt)` 在 429 时触发, 拖低有效吞吐.
- **结论**: 调高 `QWEN_CHAT_BURST`/`MAX_RPM` 无益 (API 是墙). Step1 成本须按 ~11-17/min 重估 (非 28RPM), 见 [[step1-nparallel-impl-20260826]].

## ⚠️ donor 源偏差待确认 (153302 vs 183940, 2026-08-26)
- pro 书面计划 (本文件 line 29/49) 写 "复用 183940 archive".
- 脚本实际用 **153302** archive. 理由 (已核验):
  1. Step1 主线种子 = `1dbbb821`, 而 1dbbb821 **是 153302 run 产出的** (153302 archive Opt-Comp 节点 hash=1dbbb821c3, delta +0.3444). 测 153302 加性 = 直接答"Step1 自己的轨迹要不要搞交叉", 比 183940 (另条轨迹, Theorem ci<0) 更贴主线.
  2. 153302 三轴 donor 全真阳性 (Theo ci+0.100 / Hypo ci+0.111), 是"最强加性测试": 连最强单轴段都不加性 → sub-additive 最硬证据.
  3. 修正二 (旁路 ci>0 守门) 为 183940 设计 (其 Theorem ci<0); 153302 Theorem 本就过门, 旁路自动满足, 测仍有效.
- 判据线 = 153302 的 Opt-Comp(1dbbb821) 同次 k=10 delta, 与主线种子对齐, 连贯.
- 待办: 结果出来后, 若需 183940 复验可补跑 (但 153302 已是贴主线的有效 Phase A).
- **解读纪律 (用户钉死)**: 看 CI 带非单点 gain, 别只看 overall_verdict 字符串. 全 sub-additive → 触发 Phase B ②c, 不直接判"数据无第5轴".

## 关联
[[facet-tree-no-cumulative-20260825]] / [[facet-tree-overlay-fix-20260825]] / [[facet-tree-overlay-smoke-20260825]] / [[facet-tree-retest-result-20260825]] / [[facet-tree-preset-vs-emergence-20260825]] / [[a2-power-analysis-20260824]]