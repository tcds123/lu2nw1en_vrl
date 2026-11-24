# for_review — Skill Self-Evolution 实验交付包（版本 2026-11-13）

## 👉 codex 请看这一段：你要做什么

**任务**：对这个包里的**实验结论、代码、评测协议、实验计划与后续方向**做一次独立技术评审，产出一份评审报告。

**必读**：`review_requests/20261113_codex/REVIEW_REQUEST_20261113.md`（你的完整清单，本包一切的入口）。

**要审的 5 件事（A–D 是技术审点，E 是对计划本身的意见）**，细节与逐条审点见请求文件 §1：

| # | 审什么 | 我们自己的当前判断（供你判断是否需要推翻） | 在哪 |
|---|---|---|---|
| A | 评测协议（cap / 截断计错 / 题级 McNemar / k=20）能不能撑住结论 | 【推断】能撑住"未显著"，但检出下限被 30 题 × k-vote 压低 | `docs/eval_protocol_paired_20261113.md` |
| B | "evolver 有 ACCEPT 但 held-out 不显著" 的归因 | 【推断】主要是测不出来（检出下限 ~+0.10 > 真实效应），非"没有产出" | `memory/aime-saturation-closeout-20261105.md` |
| C | Q1 cap 扫描：截断差异是否为题级噪声；两 seed 合并的题级检验够不够 | 【假说】s1/s2 略少截断可能只是题级噪声；**seed1 的「none=s0」已被 seed2 推翻**（none 69 vs s0 87） | `memory/cap-sweep-20261113.md` + `outputs/paired_cap_sweep_20261113/`（`per_*.json` 逐题逐样本 + `plan.json`）+ `outputs/paired_cap_sweep_20261113.cap8192.log` |
| D | APPS_Plus 路由 232/300（rescue 24 / regress 0；通道B retention 88.94% / regress 23）：是 skill 的增益，还是路由信号的选择效应 | 【事实】数字；【推断】有选择效应风险，已用通道B测量但未排除 | `memory/tf-evolver-appsplus-routing-retention-20261113.md` + `outputs/appsplus_routing_formal/`（`tf_appsplus_routing_result.json`（含 `retention_analysis`）、`test_routed_results.json`、`test_retention_results.json`、`tf_run_log.json`、3 轮 `skill_iter*.md`） |
| E | 实验计划、为什么这么做、Q1–Q4 优先级、后续方向、漏掉的风险 | 【推断】Q1→Q2→Q4→Q3 的顺序合理，但"最小验证实验"的 ROI 排序没论证过 | `plans/EXPERIMENT_PLAN_20261113.md` |

> 每条任务请回答三件事：**我们的判断成立吗 / 若不成立是哪里错 / 最小验证实验是什么**。

> **审点 D（APPS_Plus 路由 232/300）是本次最主要的争议点**。我们已在 README 下方单开一段 **「APPS_Plus 条件路由线 — codex 重点审阅段」**（A1–A6），集中说明：算法选型 rationale、结果、产出文件映射、后续实验/消融/方向、可替换代码数据集调研、本机情况与本地数据集清单（**请 codex 帮选下一步数据集**）。建议先读那段再读 §1D 原文。

**交付形式**（务必按此格式，见请求文件 §4–§5）：
- 一份 `.md` 评审报告（如改了代码，一并放进同一个 zip）；
- 每条意见必须标：**证据等级**（【事实】/【推断】/【假说】）、**是否改变论文结论**、**建议的最小验证实验**（题量 / 时长）；
- 回包放 `for_review/incoming/<YYYYMMDD>_codex/`：原 zip + `extracted/` + `NOTES.md`（md5 diff）+ `REVIEW.md`。

**不要做的**：不要重复报告 `REVIEW_REQUEST §3 已知问题` 里的内容；不要外传包内任何数据；`archive/` 里的历史包只需在你需要追溯时查，不必逐份评审。

---

> **本目录是唯一交付根**，合并了两代 review 包（`for_review` 10-15 版 / `for_review_aime_standalone` 11-10 版）。
> 历史原样存放在 `archive/`（按 `<日期>_<主题>/` 命名）。对方回包统一放 `incoming/<YYYYMMDD>_<来源>/`。
> 当前主体 = 锁定配对协议（AIME）+ Q1 cap 扫描（运行中）+ APPS_Plus 条件路由正式跑。
>
> 顶层 `for_review_2026-10-15.zip`、`for_review_10-16-gpt_review.zip`、`gpt_review_extracted/` 是**待删残留**，
> 原件完整存放在 `archive/2026-09-28_historical_zips/`（zip md5 已核）与 `archive/2026-10-16_gpt_review/`。
> 旧主体 `code/ data/ docs/ results/ README.md` 已于 11-13 删除（快照见 `archive/2026-11-10_standalone_superseded/`）。

## 从哪里开始读（reviewer 版）

| 顺序 | 文件 | 读什么 |
|---|---|---|
| 1 | `review_requests/20261113_codex/REVIEW_REQUEST_20261113.md` | **本次 review 要做的事**：A–D 四个技术审点 + E 对计划/动机/方向的意见（共 5 条） |
| 2 | `plans/EXPERIMENT_PLAN_20261113.md` | **实验计划、为什么这样做、Q1–Q4 状态、后续方向、已知风险** |
| 3 | `docs/eval_protocol_paired_20261113.md` | 评测协议（**locked**，改协议须记版本并重跑） |
| 4 | `docs/EXPERIMENT_REVIEW_20261109.md` | 上一轮复盘 + 证据等级约定（【事实】/【推断】/【假说】） |
| 5 | `memory/cap-sweep-20261113.md` | Q1 cap 扫描（边跑边记：启动参数、每臂数字） |
| 6 | `memory/aime-saturation-closeout-20261105.md` | AIME 饱和 / 检出下限结项 |
| 7 | `memory/tf-evolver-appsplus-routing-retention-20261113.md` | APPS_Plus 路由正式结果 + 通道B retention（唯一"强显著"；含 candidate preservation 诊断） |
| 8 | `docs/phase2_issues_202611.md` | 2026-11-02 起的问题记录（配对评测 / cap 扫描 / flash 网关 / 路由 / 驱动并发）；更早的历史在 `docs/phase2_issues.md` |

## 目录结构

| 路径 | 内容 | 进交付 zip |
|---|---|---|
| `README.md` | 本文件（入口） | ✅ |
| `plans/` | **实验计划总纲**（动机、时间线、Q1–Q4、风险、后续方向） | ✅ |
| `docs/` | 评测协议（locked）、设计文档、`phase2_issues_202611.md`（11 月起）+ `phase2_issues.md`（历史）、上一轮复盘、交接说明 | ✅ |
| `code/` | 当前代码（脱敏；`scripts/` 为启动脚本，路径自定位） | ✅ |
| `data/` | `aime24.jsonl` / `aime25.jsonl`；`skill_aime_initial_v2.md`（S₀，599 字）；`candidates/`（s1=demo5 预算纪律补丁、s2=demo6 宁缺毋滥补丁）；`codex_distill/`（d1 = `aime24_to_aime25_skill.md`）；`skills/`（Codex 蒸馏的 d2 = `aime-structural-transfer_SKILL.md`、d3 = `aime-transfer-final_SKILL.md`）；`appsplus_capped_seed42/` | ✅ |
| `memory/` | 关键实验节点记录（按日期，一个文件一个节点） | ✅ |
| `outputs/` | `paired_budget_s0/run1/`（cap16384 三臂 `paired_report.json` + `per_*.json`）、`paired_cap_sweep_20261113/`（`cap8192/per_*.json` + `.cap8192.log`）、`appsplus_routing_formal/`（路由结果） | ✅ |
| `review_requests/<日期>_<对象>/` | 给 reviewer 的请求文档 | ✅ |
| `incoming/<YYYYMMDD>_<来源>/` | 对方回包（原件 + 解压 + NOTES + md5 diff） | ❌ 本地归档 |
| `archive/` | 全部历史包（旧 zips、10-15 主体、10-16 回包、11-10 standalone、旧主体快照） | ❌ |

### 只读自检：包里能查到什么、查不到什么

> 路径口径：包内文档引用的 `scripts/xxx.py` 实指 `code/scripts/xxx.py`；引用以 `outputs/`、`memory/` 开头但标注"仅本地"的（如完整 trace、`outputs/trace/`）**不在包内**。若某条引用在本包中找不到，先看这条口径，再看 `docs/phase2_issues_202611.md` 对应条目里的说明。

| 你要复核的东西 | 包内路径 | 说明 |
|---|---|---|
| 每个实验题的 20 次生成（原始输出文本） | **不在包内**（体积极大：cap8192 trace 69MB / cap16384 trace 107MB） | `per_*.json` 有题级逐样本摘要：`em` / `pred`（提取的答案）/ `finish_reason` / `truncated` / `completion_tokens` / `chars` |
| 题级汇总与配对检验 | `outputs/paired_cap_sweep_20261113/`（`per_*.json`，跑完后生成 `paired_report.json`）、`paired_budget_s0/run1/paired_report.json` | 这里是权威数字，README / 计划里的表格都由它抄出 |
| 臂定义（skill 全文） | `data/skill_aime_initial_v2.md`（s0）、`data/candidates/demo{5,6}_*.md`（s1/s2）、`data/codex_distill/aime24_to_aime25_skill.md`（d1）、`data/skills/`（d2 = `aime-structural-transfer_SKILL.md`，d3 = `aime-transfer-final_SKILL.md`） | 全部入包，可直接算字符数/审措辞 |
| 题目内容 | `data/aime24.jsonl`、`data/aime25.jsonl` | 30 题 / 15 题 |
| 跑数代码 | `code/paired_five_arm_v1.py` + `code/scripts/run_{cap_sweep,paired_five_arm}_20261113.sh` | 脚本路径自定位，key 运行期从外部 yaml 读，不在包内 |
| 运行日志 | `outputs/paired_cap_sweep_20261113.cap8192.log` | 每臂一行（em/截断/k 票/err） |
| d1/d2/d3 的历史评测（非配对协议） | `outputs/flash_aime24_dskills/`（`analysis_merged.json` 为结论，`20261110_144232/` 为 cap65536 那次） | 见计划 §3 的臂来源表 |
| APPS_Plus 路由 | `outputs/appsplus_routing_formal/` | `tf_appsplus_routing_result.json` 为结论 |
| AIME 原题 | `data/aime24.jsonl`、`data/aime25.jsonl`（来自公开集 AIME 2024/2025） | 题目由人工整理，无私有数据 |

若你需要某题某次生成的原始文本，请在报告里点名（题目 id + 臂 + seed），我们单独提供。
| `dist/` | 本包打包产物 `for_review_<日期>.zip` | ❌ |

## 实验批次索引（细节见 `plans/EXPERIMENT_PLAN_20261113.md` §2）

| 日期 | 实验 | 结论 | 状态 |
|---|---|---|---|
| 10-09 | LCB / SkillOpt | Δ+4.9pp（flash），换 V3.2 −1.4pp → 依赖模型 | 完成 |
| 10-13~16 | AIME evolver 早期 | 有 ACCEPT 但 held-out 不稳 | 完成 |
| 11-05 | flash evolver 全流程 | in-sample +0.078（CI +0.011）；held-out +0.0167（CI −0.0375）；检出下限 ~+0.10 | AIME 饱和 |
| 11-09 | Codex 蒸馏 D1/D2/D3 | AIME 上无增益；d3 显著负效应 | 完成 |
| 11-10 | cap 65536 统一重跑 | d3 0.633（p=0.004）；3 条 500 = 失控长推理非截断 | 完成 |
| 11-11 | ClassEval 2-turn / APPS_Plus SkillOpt | 2-turn +12 题；APPS 整体 Δ+1pp = 噪声，按 hard 完美可分 | 完成 |
| 11-12 | gate 方向修正 | **证伪**；根因 = 组成差异 + 泛化退化 | 完成 |
| 11-12/13 | APPS_Plus 条件路由 | Phase A 228 / Phase B **231/300**（初版）/ **232/300**（rescue 24 / regress 0；通道B retention 88.94% / regress 23） | 命中判据，已用通道B补 retention |
| 11-12/13 | 配对协议定稿 | cap/k/temp/retry 锁定；题级 exact McNemar | **locked** |
| 11-13 | 配对 3 臂 @cap16384 | s0 .905 / 2.08% → s1 .9325 / 0.25% → s2 .9075 / 1.08%；题级 p=1.0 | 点估计方向一致，未显著 |
| 11-13 | **Q1 cap 扫描**（8192→4096；2048/1024 已砍） | 8192 seed1：none/s0 截断同为 74/600；s1 65、s2 64；em s0 .8333 → s1 .8433 | 🔄 运行中 |

## 四个优先级问题（状态速览）

| # | 问题 | 状态 | 下一步 |
|---|---|---|---|
| Q1 | 截断 | 🟢 定论：少数长链题 × (cap, skill 体量) 交互 | cap 扫描收尾 → 两 seed 合并出正式结论 |
| Q2 | 泛化 | 🔴 未启动 | A 档换源（HMMT / OlympiadBench / MATH-500 L5）；B 档更难子集（none 30–60%） |
| Q3 | 融合 | ⏸ 挂起 | 等 Q4 5 臂基线就位后跑 `S0 / D1 / 宏观 skill / 融合` |
| Q4 | harness | 🟢 协议 locked | 跑 5 臂 AIME24 基线（cap16384）作统一参照 |

---

## APPS_Plus 条件路由线 — codex 重点审阅段

> 本段是本次 review 的**主要技术审点 D**（"232/300 是 skill 增益还是路由选择效应"）所在。
> 下面六个小节回答 codex 要问的：为什么选这套算法、拿到了什么结果、对应哪些文件、后续该补什么实验、换什么代码数据集。
> 数据源以 `memory/tf-evolver-appsplus-routing-retention-20261113.md`（正式 milestone，含通道 B retention）和 `outputs/appsplus_routing_formal/` 为准；本段表格数字均由该 result JSON 抄出，可逐字段复核。

### A1. 为什么选当前这套算法（算法选型 rationale）

我们的目标是**在不更新任何模型参数**的前提下，自动改写给冻结 B 模型（deepseek-v4-flash）的 **skill 文本**，在真实代码基准上拿到可测的提升。选型路径是逐层**证伪**出来的，不是一开始就定的：

1. **为什么不用 SkillOpt 原版的"整体注入"优化？** —— 我们在 APPS_Plus 上跑过 SkillOpt（只读基线，3 轮整体注入）。结果是 211/300，而 S₀ baseline 是 208/300，McNemar p=0.72，即**噪声，未真超 baseline**。分层后看清楚原因：rescue 17、regress 14，双向抵消成净 +3。→ 整体注入让 skill 同时"救回"和"打破"等量题目，信号被对冲。

2. **为什么是"条件路由"而不是继续优化 skill 文本？** —— SkillOpt 的 regression 来自把候选 skill 也喂给 S₀ 已 pass 的题，导致"本来对的被打错"。条件路由（结果复用型）做一件简单的事：**turn1 = S₀ 全题复用 cache（不重调），turn2 = 只在 S₀-fail 的题上注入候选 skill，hard 优先 + soft tiebreak 偏 turn2 取 keep-the-better**。于是 S₀-pass 的题根本不碰候选 → regression **结构性 = 0**，把双向抵消洗成单向 rescue。这是最小且诚实的杠杆。

3. **为什么还要一个 evolver（`tf_skill_optimizer_appsplus.py`）？** —— 路由本身只决定"候选只在 fail 题上跑"，候选 skill 从哪来仍是问题。我们自研免训练 evolver，让它在**纯 rescue 信号**下进化（适应度 = 候选对 S₀-fail 题的 rescue 数，gate 在 val 的 routed delta 上判晋升）。Phase A 用 SkillOpt 自己产的 skill + 路由 = 228；Phase B 用我们 evolver 产的 skill + 路由 = 232（rescue 24）。→ evolver 在路由信号下产出的 skill 至少不输 SkillOpt，且有轻微优势（rescue 24 vs 17）。

4. **诚实归因（写进论文的措辞）**：路由是核心增益主体（把 211 的净 +3 洗成 +21）；evolver 是"在纯 rescue 信号下稳定产出合格 skill"的载体。**我们绝不把 232 全部记成"skill 文本增益"**——这正是 §1D 威胁所在，见 A4。

### A2. 取得了哪些成果（实验结果）

**正式跑（2026-11-13，3 轮，train_n=80 全量 S₀-fail 题，g=4，deepseek-v4-flash，cap=16384，temp=0，ts=20261113_152603）**：

| 指标 | 值 | 对照 |
|---|---|---|
| routed test hard | **232 / 300 = 0.7733** | base 208，SkillOpt best 211 |
| Δ vs base (S₀) | +24 题（+8.00pp / +11.54% 相对） | |
| rescue (0→1) | 24 | |
| regression (1→0) | **0** | 路由结构性归零 |
| McNemar vs base(208) | p = 1.19e-07 | 强显著 |
| vs best SkillOpt(211) | routed_better 26 / best_better 5，p = 1.9e-04 | routed 显著更优 |
| best_skill 写入 | 3009 chars（S₀=1796，**+1213**） | 写入健康 |
| **通道 B retention（独立诊断）** | **185/208 = 88.94%，regress 23（11.06%）** | SkillOpt 参照 93.27% / regress 14 |

**四方案横向对比（同一 300 题测试集，同一 S₀-cache，同一配置）**：

| 方案 | routed | rescue | regress | vs base p | 相对 S₀ |
|---|---|---|---|---|---|
| SkillOpt 整体注入（只读基线） | 211 | 17 | 14 | 0.72（噪声） | +1.44% |
| Phase A：SkillOpt skill + 路由 | 228 | 20 | 0 | 1.9e-06 | +9.62% |
| Phase B 冒烟：我们 evolver skill + 路由（10 题 1 轮） | 229 | 21 | 0 | 9.5e-07 | +10.10% |
| **Phase B 正式：我们 evolver skill + 路由（80 题 3 轮）** | **232** | **24** | **0** | **1.19e-07** | **+11.54%** |

> SkillOpt 211 相对 S₀ 208 的 +3 是 McNemar p=0.72 的噪声（未真超 baseline）；我们的 232 相对 208 和 211 均 p<2e-4（强显著）。但见 A4：regress=0 是路由结构性特征，**它证明的是"路由机制有效"，不是"skill 正文有等量增益"**；通道 B 才是 candidate skill 独立能力保持的测量（88.94%，详见 A4）。

### A3. 对应产出哪些文件（代码 + 数据 + 结果映射）

**代码（`code/`，路径自定位，key 运行期从外部 yaml 读不落盘）**：

| 文件 | 作用 | 入口 |
|---|---|---|
| `code/tf_skill_optimizer_appsplus.py` | 免训练 evolver 主程序（移植 + 路由 eval + 三段 S₀-cache + 通道 B `eval_retention`） | `python tf_skill_optimizer_appsplus.py --s0 ... --out_dir ...` |
| `code/validate_routing_appsplus.py` | Phase A 独立路由上界验证（turn1 复用 S₀-cache，turn2 live-call candidate on S₀-fail） | `python validate_routing_appsplus.py --config ... --baseline_results ... --best_skill ...` |
| `code/smoke_retention.py` | 通道 B 冒烟工具（5 题 S₀-pass 上验 `eval_retention` 闭合性 + infra 隔离 + valid-only McNemar） | `python smoke_retention.py` |
| `code/scripts/launch_tf_appsplus_routing.sh` | Phase B 启动垫片（读 flash key、unset AIGC_USER、cap=16384、RPM=20、cd code/ 跑 evolver） | `bash code/scripts/launch_tf_appsplus_routing.sh` |
| `code/scripts/launch_validate_routing.sh` | Phase A 启动垫片（同源 flash 口径） | `bash code/scripts/launch_validate_routing.sh` |
| `code/skillopt/configs/appsplus/default_flash.yaml` | SkillOpt 配置（validate_routing 依赖；split_dir 写包内绝对路径，换 split 需复制 yaml 改行） | `--config` 指向 |
| `code/skillopt/envs/appsplus/skills/initial.md` | S₀ skill（evolver 起点，1796 chars / 1811 bytes） | `--s0` 指向 |

**依赖（SkillOpt 只读库，不修改源码；包内已平铺一份供 import）**：`code/skillopt/`（envs/configs/utils/model...），与源 `/data/zhuldz/skillopt` 同源；sh 里把源仓库作 PYTHONPATH fallback。

**数据 + 缓存（全部随包，`data/` + `outputs/`）**：

| 文件 | 内容 |
|---|---|
| `data/appsplus_capped_seed42/` | 预建 capped split（train80/val395/test2765，ratio 2:1:7 seed42 确定性子集） |
| `outputs/appsplus_v2_cache/selection_eval_baseline/results.json` | val S₀-cache（100 题，hard=58，turn1） |
| `outputs/appsplus_v2_cache/test_eval_baseline/results.json` | test S₀-cache（300 题，hard=208，turn1 + McNemar base） |
| `outputs/appsplus_v2_cache/test_eval/results.json` | SkillOpt best test（211，McNemar 对照） |
| `outputs/appsplus_v2_cache/best_skill.md` | SkillOpt 基线 skill（validate/smoke 依赖） |

**正式跑结果（`outputs/appsplus_routing_formal/`）**：

| 文件 | 内容 |
|---|---|
| `tf_appsplus_routing_result.json` | **结论 JSON**（routed 232/rescue 24/regress 0/p 值/三轮迭代轨迹/`retention_analysis` 字段） |
| `tf_global_skill.md` | 最终 skill（3009 chars，iter1 选中；iter3 同分） |
| `test_routed_results.json` | 逐题 routed 结果（300 条，McNemar 备用） |
| `test_retention_results.json` | **通道 B** 逐题候选独立执行结果（208 条 S₀-pass，McNemar 备用） |
| `train_s0_cache.json` | S₀ train cache（80 题，evolver 用） |
| `skill_iter{1,2,3}.md` | 每轮 skill 快照（3009/3131/3162 chars） |
| `experience_iter{1,2,3}.md` | 每轮经验快照 |
| `experience_pool_tf.json` | 经验池 |
| `tf_run_log.json` | 运行日志 |

### A4. 后续需要补充的实验、消融与未来方向

**最关键的威胁（§1D，必须先回答）**：regress=0 是路由的结构性特征（S₀-pass 题从不跑候选），所以"routed 232 vs 208"只能证明路由机制有效，**不能直接证明候选 skill 正文有等量增益**。通道 B（Retention 诊断）已于 2026-11-13 18:39 补跑落盘：**retained 185/208 = 88.94%，regress 23（11.06%）**，agent_fail=0、missing=0，valid-only McNemar p=2.38e-07（both_pass=185 / regress=23 / cand_rescue=0），SkillOpt 参照 93.27% / 14。结论：**候选 skill 在 S₀ 已通过的题上确实有约 11% 的单次退化（上界；S₀ 稳定通过的确定退化 13 道），技能本身并非"无损"**；这 23 次退化在 routed 分数里结构性不可见。与 SkillOpt 的 4.33pp 差距因缺其逐题明细而无法配对检验（未配对 z 检验 p≈0.12），按待补测量处理。按优先级：

1. **通道 B：Capability Preservation 测量（✅ 已出结果：88.94% / regress 23，见上）**。在 test 208 道 S₀-pass 题上**独立重跑**最终候选 skill（不复用 cache），测 regress/retention。三概念严格分开：Capability Acquisition（rescue，通道 A）/ Capability Preservation（retention，通道 B）/ Deployment Performance（routed 232，组合系统）。通道 B 走通且 retention 不显著退化，才能说"候选 skill 在 pass 题上不破坏能力"。对应代码 `eval_retention`（已挂进 final eval）+ `smoke_retention.py`；SkillOpt 对照口径 retention=194/208、regress=14/208 写在代码常量里。**codex 审点：这个设计能不能真正回应 §1D？是否还需要更强的对照？**

2. **路由选择效应拆解**（计划 §7-4）：held-out 路由标定（train 失败信号 vs test 失败信号分开估）；或"随机路由到同规模 S₀-fail 题"作对照，看 rescue 是不是单纯来自"多给一次机会"。

3. **evolver vs SkillOpt 的 skill 头对头**：24 vs 17 rescue（+7）在 flash 噪声量级（±1~3 题）已超噪声，但单 seed / 单 split 仍需多 seed 合并做题级 McNemar 才能定结论。

4. **消融**：(a) 仅路由不进化（S₀ + 路由，看纯路由增益下界）；(b) 仅进化不路由（SkillOpt 整体注入口径，已=211）；(c) train_n 扩大（✅ 已从 27 扩到全 80 S₀-fail，rescue 23→24、routed 231→232，扩量收益边际）；(d) skill 体量 sweep（写入 +1213 chars 是不是过/欠拟合，对比旧版 +1771）。

5. **未来方向**：(a) 把条件路由从 APPS_Plus 迁到 ClassEval（repo 级、2-turn self-test 已验证可跑），看路由在 class 级多方法生成上是否同样有效；(b) 把"纯 rescue 信号进化"作为通用 recipe，在多个代码数据集上验证稳定性；(c) 论文叙事最终落点——"训练免 + 路由 + 纯 rescue 信号进化"三件套，需在 ≥2 个数据集上强显著才成立。

### A5. 还有哪些类似的代码数据集可以尝试（docker-free 调研）

我们已做过一次 docker-free 代码 benchmark 调研（`memory/benchmark-survey-no-docker-20261110.md`），硬约束是**本容器无 docker / 无 CAP_SYS_ADMIN / 无 namespace / 无 KVM**（有 CAP_SYS_CHROOT + SETUID/SETGID + libseccomp），所以凡是官方评测强制 docker build 的（SWE-bench 等）直接出局。结论按"复杂度 × 可跑性"排序：

| 数据集 | 任务粒度 | 执行机制 | docker-free 可行性 | 建议 |
|---|---|---|---|---|
| **ClassEval**（FudanSELab，ACL'24） | class 级多方法生成 | `unittest.TextTestRunner` + `importlib` 进程内执行，`@func_set_timeout(5)` 防卡死，**零 docker 零子进程** | ✅ 最高（已验证可跑，复用 mbpppro exec 模式几乎零适配） | **首选**——已接入并跑过 2-turn |
| **DevBench Python 子集**（open-compass，arXiv 2403.08604） | repo 级 5 阶段 | 标准 pytest，vene 可跑（官方主路径是 docker 预装依赖） | 🟡 需逐 repo 手工 `pip install` 依赖 | 次选——先挑 geotext/readtime 等依赖简单的 2-3 repo 扩展验证 |
| **DevEval**（seketeam，arXiv 2405.19856） | 函数级 + 每题 tests | `pass_k.py --n 20` 采样，需 HF 下 Source_Code + 逐 repo 装依赖 | 🟠 适配成本最高（不许改 repo 文件结构） | 三选——量大但要逐 repo 适配 |
| SWE-bench（princeton-nlp） | issue 修复 | v5 harness 强制 docker build | ❌ 出局 | 不可跑 |
| CrossCodeEval / RepoBench / RepoCoder | repo 级补全 | Exact Match / Edit Similarity / CodeBLEU 文本匹配，**不执行** | ❌ 无 reward 信号 | 对 skill 优化无用 |
| CommitPack | HF 训练数据 | 无评测脚本 | ❌ | 出局 |

**两个易混纠正**：DevBench（open-compass，22 repo）≠ 真正的 DevEval（seketeam，115 repo），之前记混成同一个"DevEval 117 repo"；arXiv 号 DevEval=2405.19856、DevBench=2403.08604。

### A6. 本机情况 + 本地代码数据集清单（请 codex 帮选下一步数据集）

**本机硬约束（实测 2026-11-13）**：

| 项 | 状态 |
|---|---|
| GPU | ❌ 无（`nvidia-smi` 不可用）—— 全部靠 API（deepseek-v4-flash，20 RPM），无本地推理 |
| 磁盘 | `/data` 1.0T 已用 841G，**剩 183G（83%）** |
| docker | ❌ 不可用 |
| capabilities | 有 `cap_sys_chroot` / `cap_setuid` / `cap_setgid` / `cap_mknod` / libseccomp；**无** `cap_sys_admin` |
| 网络 | 可访问 `aimpapi.midea.com`（flash 端点，OpenAI-compat，20 RPM） |

**本地已有代码/数学数据集（`/data/zhuldz/lunwen/data/`，13 个）**：

| 数据集 | 体积 | 任务类型 | 是否已接入 evolver | 备注 |
|---|---|---|---|---|
| `APPS_Plus` | 117M | 竞赛代码（stdin/stdout） | ✅ **本次主实验** | interview split，已建 capped split |
| `ClassEval` | 818M | class 级多方法生成 | ✅ 已接入 | 2-turn self-test 验证过 |
| `Code-Feedback` | 1.2G | 代码 + 反馈 | ❌ 未接 | |
| `CodeEval-Pro` | 7.1M | 代码评测 | ❌ 未接 | |
| `LCB`（LiveCodeBench） | 4.2G | 竞赛代码 | ✅ 早期基线 | SkillOpt Δ+4.9pp（flash） |
| `OpenCodeInstruct` | 33G | 代码指令 | ❌ 未接 | 训练数据，非评测 |
| `apps` | 7.9G | 竞赛代码 | ❌ 未接 | APPS 原版 |
| `codeparrot-train-v2-near-dedup` | 16G | 代码训练数据 | ❌ 未接 | 训练数据 |
| `humaneval` | 556K | 函数级代码 | ❌ 未接 | HumanEval / HumanEval+ |
| `mbpp` | 1.1M | 函数级代码 | ❌ 未接 | MBPP / MBPP+ |
| `skillsbench` / `skill_benchmarks` | 212M / 36K | skill 评测 | ❌ 未接 | |
| `aime24.jsonl` / `aime25.jsonl` | — | 数学（AIME） | ✅ 配对协议主战场 | 30 题 / 15 题，已饱和 |

**请 codex 帮我们选下一步尝试的代码数据集**，建议从这几个维度判断：
- **可执行验证**（非文本匹配，能拿到 reward 信号）—— 排除 CrossCodeEval/RepoBench/RepoCoder；
- **docker-free 可跑**（本容器约束）—— 排除 SWE-bench；
- **题量足够大、S₀ 未饱和**（AIME 30 题已饱和、检出下限 ~+0.10，需要更难或更大的题源）；
- **已本地就绪优先**（ClassEval 已接入、`apps`/`humaneval`/`mbpp` 已在盘上，零下载成本）。

我们当前倾向：**ClassEval 做路由迁移验证（repo 级、已验证可跑）** + **`apps`/`humaneval`+ / `mbpp`+ 做更大题量 / 更难子集的泛化验证**。请 codex 给出选择意见与最小验证实验设计。


## 回包约定

见 `incoming/README.md`（zip 原样保存、`extracted/`、`NOTES.md` 记 md5 diff、增量合入、合入前留 `archive/<日期>_premerge/` 快照）。
