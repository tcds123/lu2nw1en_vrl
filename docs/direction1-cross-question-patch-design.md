# Direction 1 设计: 跨题可迁移候选生成 (Cross-Question Generalization Anchoring)

**日期**: 2026-10-21
**状态**: 🟡 设计就绪, 待 [16384 基线确认 (PID 88093)](../../memory/) 收敛后启动实现
**作者**: Claude (研究主线回归自 [[aime25-heldout-iter2-nogeneralize-20261021]] 的纯过拟合 verdict)

## 0. 缘起与判定依据

AIME25 held-out 评测 (30 题, k=8, strict, seeds=3): run#6d iter2 in-sample **+0.0417** → held-out **−0.0347** (CI [−0.0681,−0.0042], 0/3 正向)。|Δ| 对称翻转 = 经典 probe-set 过拟合。F7 (max_tokens 4096 截断) 已被证伪非混淆因子 (见 `f7-confounder` 撤回文件: 近端桶 ≥th 处处 ≤0, 远端桶 <th 处处 >0, 截断反而护住 iter2; 方向与原 verdict 一致)。

**根因定位**: 候选生成层 `generate_candidates` 逐题/单轮诊断驱动 → patch 拟合单一失败题或 <30 题 probe 的噪声, 不形成跨题可迁移的全局决策程序。

AIME 任务层的结构性困难 (佐证 [[aime-fromscratch-deepanalysis-20260805]]): 30 题 probe + 高难度 → 单 lineage 极易 overfit; 每轮 pool CI (30 题) 功效不足以区分"真泛化"与"噪声拟合单题"。

## 1. 现状: 为什么候选会过拟合 (代码实证)

`rl/train/verl1/skill_evolution/skill_evolver_aime_v2.py`:

| 环节 | 位置 | 现状 | 过拟合机制 |
|---|---|---|---|
| 候选生成输入 | `generate_candidates` (L1243) | `make_diagnostics(failures)` 仅喂**本轮失败题**的 excerpt; `focus` 单一失败类型 | patch 只看到 1-8 道失败题, 无"哪些题已经对了、别弄坏"的负例 → 拟合单题/少数题的表面特征 |
| 多样性格子 | `angle_hints` (L1283/L1292) | 固定 4-6 条泛角度 | 不引导"朝跨题共有结构靠", 无跨题线索 |
| selection | 主循环 promote gate (L3287-3299) | `ci_pos=[c for c in sv_nodes if c.val_ci>0]` (L3287) → `max(ci_pos, key=val_delta)` (L3289), 只对 AIME24 30 题算 CI, 且**只朝正方向推** (admission 选 delta 高) | 正方向 gate 鼓励"多做对 30 题里已有的"→ 加深对 probe 的拟合; 无"惩罚只对单题有效的 patch" |
| held-out | 主循环里**完全不存在** AIME25 | — | held-out 只在事后手评 (本次), 非每轮压力 |

> **注**: `bootstrap_ci_lower(per_item_deltas, n_boot=1000)` 定义在 `tf_global_v2_math.py:530` (import 来的, 非本文件), 入参是 `per_item_deltas={qid:[d1,d2...]}` (每题均值后 bootstrap, 返回双侧 95% CI 下界 2.5th percentile), **不是 `probe_items`**。L3239/L3654 是 `bootstrap_ci_lower` 的调用点, 不是门控判定本身; 真 promote gate 在 L3287-3299。`PATCHER_SYSTEM` L150 ✅, `generate_candidates` L1243 ✅, `classify_failure_type` L193 ✅ (均已逐行复核)。

**关键结论**: selection `bootstrap_ci_lower` 只做**正方向单目标**, 无任何 cross-question 一致性约束 → winner's curse (probe 归一化/前沿过拟合), 与 `subset_admit`/`facet_group_admit` 同病根 (都作用同一 30 题, 不触碰 held-out)。

## 2. 设计目标 (非新增架构, 三处最小改造)

不引入新 eval 后端、不新建模块、不改 scoring 协议。**复用现有的 `probe_items` (AIME24 30 题) + `eval_paired` + `bootstrap_ci_lower`**, 在三个已有环节注入跨题信号。

三条改造对应三个瓶颈点:

### 改造 A — 候选生成: 给 patcher 喂"跨题锚 + 负例" (主攻)

**落点**: `generate_candidates` (L1243) 的 `PATCH_TEMPLATE` / `PATCH_TEMPLATE_COVERAGE` 输入侧。

**现状 (代码实证, 三元素拆解)**:

| 元素 | 现状 | 落点 |
|---|---|---|
| 跨题锚 (正例, 从已稳题蒸馏可复用决策程序) | ✅ **已实现** `make_wins` (L1186, dir-5) — 渲染 skill 解对的难题, prompt 明写 "mine the reusable decision procedure here, 而非抄具体解" | `--learn_positive` flag 控制, 默认 off |
| meta-contrast 同构对照 (无结论文字, "数据决非关键词") | ✅ **已实现** `make_meta_contrast_diagnostics` (L1201, 路径B) — 渲染同构 MISSED/SOLVED 对照块, 不写结论, 归纳留给 patcher | `--meta_contrast` flag 控制, 默认 off |
| 题级 staleness 负例 ("不得破坏这些具体已解题") | 🟡 **部分新** — 现存 `make_coverage_diagnostics` (L1150) 的 tally 表是**类型级** (COVERED/UNCOVERED 经 `_coverage_sig` signature 匹配), 不是**题级** (渲染具体已解题 excerpt 作"别弄坏"负例) | 需新写一小函数 |

三者已接 `generate_candidates` (L1254-1277) + 主循环调用点 (L4781), 由 `--learn_positive` / `--meta_contrast` 控制, **默认 off**。

**决定性事实**: run6d (产出过拟合 iter2 的运行) `run6d_launch.log` banner = `ablation=FULL | K_eff=4 | failure_driven=True | consolidate=True | antideg=True`, **无 `🔧 opt-flags ON:` 行** → 两个 flag 均 OFF。即: 改造 A 是**真实未试的角度** (不是已被证伪的重复), 有探索价值; 但**无任何证据它有效** — run6d 没开它, 它和 iter2 过拟合的关系是空白, 不是因果。**不能宣称"开改造 A 就能缓解过拟合"**, 只能说"这是 run6d 未启用的现存机制, 值得对照测"。

**改造 (分两步, 先验证再补新件)**:

**Step 1 (MVC, 优先)**: 不写新函数, 只打开现存 `--learn_positive --meta_contrast`, 与 OFF 各跑 1 轮候选生成对照 (见 §4)。两块机制已存在且已接线, 打开 flag 即生效 — 零新代码、零回归风险。这一步回答的核心问题是: "run6d 没启用的现存机制, 打开后候选是否更可能跨题 (AIME25 held-out 方向一致率 > 随机)"。

**Step 2 (仅当 Step 1 正向)**: 新写题级 staleness 负例。现状 `make_coverage_diagnostics` (L1150) 的 tally 是类型级 (按 `_coverage_sig` label 匹配 COVERED/UNCOVERED), 缺"渲染具体已解题 excerpt 作'别弄坏'负例"的题级粒度。新增 `make_staleness_negatives(stable_solved)`:
- 从 `probe_items` 中选 1-2 道当前 skill **已经成功**、且与失败面对立的题 (如: 失败面是 quantity parsing, 负例是 parsing 已正确的题)。
- 模板注入: 复用 `PATCH_TEMPLATE` 的 `{positive_section}` / `{meta_contrast_section}` 旁的槽位, 新增可选 `{staleness_section}` (注入存在则渲染, 缺失则回退现行为 — 零回归)。

**理由 (对齐 Phase C STRUCTURE 的"数据决非关键词"哲学)**: 三元素都用真实已解/已稳样本喂正例+对照+负例, 而非人工写"要泛化"的提示词。patcher 看到的是**证据**, 不是规则。前两元素已落地这套哲学 (见 `make_wins` / `make_meta_contrast_diagnostics` 的 prompt 措辞), Step 2 只是把"别弄坏"也从类型级提示细化到题级证据。

### 改造 B — selection 增益: 引入"泛化惩罚项" (对齐 review)

**落点**: 主循环候选晋升 gate (L3287-3299), 复用 `bootstrap_ci_lower` 或新增跨题一致性统计。

**现状**: promote gate (L3287-3299) `ci_pos=[c for c in sv_nodes if c.val_ci>0]; max(ci_pos, key=val_delta)` — 单目标正推 (只看 30 题 CI 下界 > 0 且 delta 最优), 无逐题方向一致性约束。

**改造**: 晋升 gate 从
```
bootstrap_ci_lower(per_item_deltas) > 0
```
改为
```
bootstrap_ci_lower(per_item_deltas) > 0  AND  cross_q_consistency(cand, probe_items) >= theta
```
其中 `cross_q_consistency` = 候选相对基线的 **逐题方向一致性** (≥θ% 的题 delta 同向 或 |正值题| 远超 |负值题|)。

**theta 的经验定标 (补)**: 文档此前只提 `>= theta` 但未给定标方法。从 step3 逐题方向诊断反推:
- step3 三 rep 逐题 delta 方向中, **仅 2/30 题三 rep 方向全一致** (文档 §2 注)。这是"噪声拟合单题"的直接观测 — 绝大多数题的 delta 方向跨 rep 不稳定, 即 patch 只在特定采样下碰巧对少数题有效。
- 故 `theta` 可设为**使"方向全一致题占比 ≤ 2/30 的 patch 不晋升"** 的阈值, 即 `theta ≈ 0.10` (3/30) 作为经验起点 — 低于此 = patch 的正 delta 几乎全靠跨 rep 不稳定的单题噪声堆出来, 非跨题信号。
- 定标步骤: (1) 跑一次 step3 现状 (learn_positive OFF) 收集每题三 rep delta 方向 → 算每题"方向全一致率"; (2) 取分布的某个分位 (如 25th percentile) 作 `theta` 初始值; (3) 在 Step 1 对照实验里同时观测该指标, 确认 `theta` 选值不致"全员不晋升"(退化为停滞)。

**目的**: 淘汰"只靠少数题 + 其余噪声抵消"的 patch。这是把 held-out 的教训**前置到主循环**, 不引入 AIME25 本身 (避免 held-out 泄漏)。

### 改造 C — held-out 不泄漏但定期哨兵 (长期)

**落点**: 主循环可选 `--heldout_watch_every N` 每 N 轮在 AIME25 (30 题, 固定) 上跑一次**单次评估**, 只记录不晋升 (不用于 gradient/selection), 作为"当前 best 是否仍泛化"的哨兵。

**违约** (仅预警非阻断): 若某轮 best 的 held-out 哨兵 < 上一哨兵兜底, 打 log + 记文件, 但不自动回滚。

**目的**: 把"事后才发现的过拟合"变成"运行中可见的 drift", 让 AIME25 成为**诊断指标**而非 **selection 目标** — 不污染 in-sample 门控 (held-out 从不进 CI/晋升)。

## 3. 改造 A 的落点细节 (最短路, 零回归)

### Step 1: 打开现存 flag (零新代码)

```bash
# 对照实验: 同一 run#6d init skill + AIME24 probe + k=8, 各跑 1 轮 (仅候选生成, 不晋升)
python -m skill_evolver_aime_v2 ... --learn_positive --meta_contrast   # ON 组
python -m skill_evolver_aime_v2 ...                                     # OFF 组 (run6d 现状)
```
两块机制 (`make_wins` L1186 / `make_meta_contrast_diagnostics` L1201) 已接线到 `generate_candidates` (L1254-1277), flag ON 即注入 `POSITIVE_SECTION` / `META_CONTRAST_SECTION` 到 `PATCH_TEMPLATE`。**不改任何代码** — 这是验证"现存未试机制"是否有效的最省路径。

### Step 2: 新写题级 staleness 负例 (仅当 Step 1 正向)

`generate_candidates` L1273 附近 (PATCH_TEMPLATE.format 注入点):
```python
# 现状
diag = make_diagnostics(failures) if failures else ...
base_user = PATCH_TEMPLATE.format(skill=..., diagnostics=diag,
                                  positive_section=pos_sec,
                                  meta_contrast_section=meta_sec, ...)

# 改造: staleness_section 可选注入
staleness_sec = ""
if use_staleness:  # 新 flag, 默认 off → 零回归
    staleness_sec = make_staleness_negatives(stable_solved, focus)
base_user = PATCH_TEMPLATE.format(skill=..., diagnostics=diag,
                                  positive_section=pos_sec,
                                  meta_contrast_section=meta_sec,
                                  staleness_section=staleness_sec, ...)
```
`PATCH_TEMPLATE` 加一个可选 `{staleness_section}` 槽 (占位为空字符串时行为字节不变)。`stable_solved` 由调用方在 `run_one`/主循环里采 (复用 `collect_failures` 反侧 — 已通过/已稳定的题)。

## 4. 验证计划 (控变量)

1. **对照 (Step 1)**: 同一 run#6d init skill + AIME24 probe + k=8, 开/关 `--learn_positive --meta_contrast` 各跑 1 轮 (候选生成), 不晋升, 仅看**候选质量**: 候选相对 init 在 AIME25 held-out 上的 delta 方向。
2. **判据**: flag ON 的候选池, held-out 方向一致率 vs OFF, 是否 > 随机 (衡量"patch 是否更可能跨题")。
3. **全链路**: 若 (1) 正向, 开 ON 重跑 evolver 1-2 iter, 看 in-sample 门控 through 后 held-out 哨兵是否不再转负。
4. **Step 2 判据**: 仅当 Step 1 正向才进入 Step 2; Step 2 再单独对照"开 staleness vs 不开"的 held-out 方向一致率增量。

## 5. 风险与失败模式

| 风险 | 缓解 |
|---|---|
| 跨题锚/负例样本选择噪声 | 从 `probe_items` 确定性采样 (split seed), 记录所选 item_id 可复现 |
| 负例让 patcher 过度保守 (不敢改) | 负例数限制 ≤2, 且只针对与 failure 对立的型 |
| held-out 哨兵每次 90 题伤预算 | 默认 off; 用 `--heldout_watch_every`, 或只 watch best 而非每候选 |
| 与现有 `subset_admit`/`facet_group_admit` 交互未测 | 改造 A 是输入侧 (候选生成), 不碰 admission; 改造 B 是加 offer 非替换, 可 flag 关 |
| Step 1 对照无正向 (现存机制本就无效) | 接受: 即 learn_positive/meta_contrast 非过拟合解药, 转 Direction 2/3; 避免写 Step 2 新函数的无效投入 |

## 6. 本次迭代最少改动单 (MVC)

1. **Step 1**: 仅 `--learn_positive --meta_contrast` flag 对照 (零代码, 零回归)
2. **Step 2 (仅当 Step 1 正向)**: `make_staleness_negatives` 纯函数 (零 API, 可单测) + `PATCH_TEMPLATE` 加 `{staleness_section}` 可选槽 + `generate_candidates` 注入 + `--staleness` flag (默认 off)
3. step (1) 单轮对照 (开/关 × AIME25 held-out 快查)

## 7. 待办依赖

- [ ] 等高优先级: 16384 基线确认 (PID 88093) 收敛 — 若 iter2 held-out 在 16384 下转正, 方向优先级可能重排
- [ ] 等设计评审 ✅ (已完成, 见 `memory/direction1-design-review-20261021.md`)
- [ ] 实现 + 对照验证后, 更新 MEMORY.md

关联 [[aime25-heldout-iter2-nogeneralize-20261021]] [[step3-iter2-rerun-result-20261020]] [[peak-diagnosis-val-to-test-negative-20260909]]；代码点 `skill_evolver_aime_v2.py:L1243/L150/L193`, promote gate `L3287-3299`, `bootstrap_ci_lower` 定义 `tf_global_v2_math.py:530`。
