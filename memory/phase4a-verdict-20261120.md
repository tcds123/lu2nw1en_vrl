# Phase 4-A 正式结论 — 2026-11-20 18:50 (5760 API 请求已花完, 本分析 0 API)

完整判读: `rl/train/verl1/outputs/crossmodel_baseline/phase4a-audit-20261120/verdict.md`
(机器产物 `verdict.formal.md`, 脚本快照 `frozen_phase4a.py` sha256 `bf7ae60a…(勘误 2026-11-23: 原记 e756abf9 有误, 以实测为准)`, rules `rules_v1.json`)。

## 三层结论 (用户要求分别回答)

| 问题 | 答案 | 关键数字 |
|---|---|---|
| ① Evolver 整体优于两个静态基线? | **是** | vs S₀ +74 (2.68 pp) CI [+1.10,+4.22] pp, p=8.0e-4; vs SkillOpt +121 (4.38 pp) CI [+2.82,+5.88] pp, p=3.3e-8 |
| ② 收益集中在哪些题型? | **class Solution** | class n=260: +87 (+33.5 pp) vs S₀, +104 (+40.0 pp) vs SkillOpt; L3.func n=373: +6/+5 CI 跨 0; L1 n=1791: −22/+1 CI 跨 0; L2 n=341: +3/+11 CI 跨 0 |
| ③ 支持跨题型泛化? | **不支持** | L1 未达 §3.2 门槛 (>0 且 ≥50% 总增益); 命中"定向修复"判读 |

Arm1 **2077**/2765 = 0.7512; S₀ 2003 = 0.7244; SkillOpt 1956 = 0.7074 (2077 已核销, 见文末)。

## 敏感性 (四项均不改变方向)

| 集 | vs S₀ g | vs SkillOpt g |
|---|---|---|
| 2765 主分析 | +74 | +121 |
| 2744 去重 | +75 | +123 |
| 2612 剔除截断 | +68 | +112 |
| 2762 剔除漂移 | +76 | +123 |

- 截断: trace length 231/5530 行 (4.2%), 命中 153 题; 72 行 item_id=None 无法归属 → 下界口径。
- C_L: L1 −0.30/+0.01, L2 +0.04/+0.09, L3 +1.26/+0.90; sum(g_L)==g_all 成立。**C 是分配, 非独立因果**。

## 对论文的影响 (必须按此措辞)

- 方法定位: **针对特定代码接口形式 (class Solution) 的定向修复**, 不是已证明的跨题型泛化。
- ✅ Arm1 全量通过 **2077**/2765 (0.7512) 已核销 (Issue #116), 与配对 net_gain (+74/+121) 并列引用无矛盾; 先前"2076"为误写。
- L1/L2/L3 是**可观测题型层** (Example 块 = 题面结构属性), 不得写成失败模式标签。
- class 层 Arm1 绝对通过率也只有 0.51 (基线 0.10–0.17) → 增益大但可能有"接口合格即通过"成分, 论文需 class 样例分析。
- 数据切分限制 (train = 前 80 题非随机 + 21 近重复) 已单列; 结论只针对 APPS-Plus 异构题库平均通过率。

## 教训

- trace 的 role 字段是 "target" 而非 "assistant", 且无 arm 字段 → 截断归属不可用, 只能保守剔除; 下轮基线应加 arm 字段 (issue #99)。
- dry-run 的分层信号 (L3 集中) 在 2765 上**完全复现** → 300 题局部 chunk 的理由是"只验证接口", 但确实能预示效应集中位置; 仍需正式集判读。

## 单臂通过率的单样本 CP 95% 区间 (2026-11-24 补记, 0 API)

用 `eva/direction_a_cp.py::uncond_cp` 现算, 供报告/论文与分层结论并列。**注**: 单臂 CP 与净增益 CI 回答不同问题 — 前者是"该臂在题库上的通过率", 后者是"配对净增益率"; 两者不可互相推导, 也不得用单臂区间不重叠来替代配对检验。

| 集合 | 臂 | pass | 通过率 | CP 95% |
|---|---|---|---|---|
| 全 2765 | Arm1 | **2077** | 0.7512 | [0.7346, 0.7672] |
| 全 2765 | S₀ | 2003 | 0.7244 | [0.7073, 0.7410] |
| 全 2765 | SkillOpt | 1956 | 0.7074 | [0.6901, 0.7243] |
| class 260 | Arm1 | 132 | 0.5077 | [0.4452, 0.5700] |
| class 260 | S₀ | 45 | 0.1731 | [0.1291, 0.2246] |
| class 260 | SkillOpt | 28 | 0.1077 | [0.0728, 0.1519] |

⚠️ ~~**2076 vs 2077 待核销 (2026-11-24 发现)**~~ → **已核销: 2077 正确, 见本节上方"旧注作废"** (下段保留作历史): 本篇上表 (Arm1 行) 与 Phase 4-A 正式产物 `paired_arm1_vs_s0.full.formal.json` 的 `total.arm_pass_rate` 写 0.7512 (=2077/2765), 而本篇正文三层结论行与 `memory/48-arm3-drop-scaffold-launch-20261119.md` 的 "g=+132, Arm1_only 309 / Arm3_only 177" 对应 **2076** (配对数 2765, 1802+274=2076)。两者差 1 题, 同一份 `test_independent_results.json` 不应有两个值 → 疑为 2077 (含 1 道 paired-invalid/agent_fail) 与 2076 (配对有效) 口径混用。**在核销前, 正式结论中一律引用配对表 +`net_gain` (+74/+121) 与 CP 区间, 不单独引 2076/2077**; 核销属 0 API 只读分析, 不影响任何统计判读。核销见 `docs/phase2_issues_202611.md` Issue #116。

## ⚠️ 本节旧注作废

下方"2076 vs 2077 待核销"一段的猜测 (含 1 道 paired-invalid) **不成立**。2026-11-24 只读复算:
Arm1 全 2765 hard=1 = 2077; 与 S₀ 配对 n=2765, both_pass 1802 / a_only 275 / b_only 201 → 2077, 净 +74, 自洽。
2076 是我误把 **Arm1 vs Arm3** 的 `Arm1_only 309` 拿去反推造成的错误, 并非真实口径。→ **唯一正确值 2077/2765 = 0.7512**,
边际通过数可直接引用。核销详见 `docs/phase2_issues_202611.md` Issue #116 的"核销"段。

## 训练实际用了哪些题 (2026-11-24 补记, 0 API 只读复核)

- train split = 790 题, 但 **capped copy 截到前 80 题** (`for_review/data/appsplus_capped_seed42/split_manifest.json`: `counts_capped {train:80, val:395, test:2765}`, note = "train truncated to 80 (first 80 of seed42 ratio train)… Built so train_size=80 hard-match assertion (trainer.py:432) passes")。
- 80 题里再按 **S₀ 是否做出来**取 fail 池: `tf-appsplus-routing-formal-20261113/train_s0_cache.json` hard=1 有 53 题, hard=0 有 **27 题** → 训练只用这 27 题 (与 [[48-formal-run-launch-20261117]] 记录一致)。
- 27 题的题面形态 (读 `data/APPS_Plus/data/v1/data_with_id.json` prompt 字段): **8 题含 `class Solution`**, 另外 19 题是 `def <随机名>():` 加 docstring 的 APPS 伪装 stdin 形态 (无 `input()`, 靠题述或 Example 块推 I/O)。27/27 都是 APPS-Plus 的 interview 难度, 其中 **9 题按 D5 截断形态分类属于 fn-call** (Arm1 `tf_appsplus_routing_result.json::coverage` 记 `fn_call_in_fail_pool=9`/`fn_call_in_train_used=9`; 与 8 题 `class Solution` 不完全重合, 具体对应关系未逐题核)。preassess 脚本曾记 0, 是 raw 形态误判 (Issue #93)。

> ⚠️ 同一节 2026-11-24 更正: 本节早期草稿写"**不含** fn-call 题", 与 Arm1 正式产物 coverage 冲突, 已按覆盖字段纠正为 9 题 fn-call。
> **逐题已核 (2026-11-24, 0 API)**: 9 道 = `1430 3324 3516 5395 5449 5987 6911 7205 7379`, 其中 **8 道题面是 `class Solution`, 仅 3324 为 stdin 伪装**; 8 道 class Solution 全部是 fn-call (反向空集)。
> 机制: raw `stdin[0]` 是 list → `literal_eval` 抛错 → 0; `truncate_test_code` 字符串化后 → list → 9。产物见 `phase4a-audit-20261120/audit_27_train_items.md`。
- 例块覆盖: 27 题中 21 题带 `Example` 块 (8 道 class + 13 道 stdin), 6 题无例块 → 训练样本偏"带例子的题面", 与 test 上的 L1/L2 分层不完全对齐, 跨层外推需谨慎。

判读: 全量 2765 上三臂单样本区间**两两重叠**, 净增益的统计证据来自配对设计而非边际率; class 260 上 Arm1 与两基线区间不重叠 (效应集中于此), 但 Arm1 绝对值仅 ~0.51 → 与"接口合格即通过"成分并存, 需 Direction A 的 RQ2 拆分 (接口重建率 vs 最终通过率)。

关联: [[phase4a-audit-tool-ready-20261120]] [[phase3b-baseline-run-progress-20261120]] [[data-split-representativeness-prereview-20261120]]
