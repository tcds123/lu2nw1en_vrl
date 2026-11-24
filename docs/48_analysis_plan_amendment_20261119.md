# #48 分析方案修订 (Analysis Plan Amendment)

**冻结日期**: 2026-11-19
**状态**: 结果揭示后修订 (post-Arm1-observation, pre-Arm3)
**适用 run**: Arm1 `tf48-arm1-full-20261117/run_20261118_091052` (已完成) + Arm3 `tf48-arm3-drop-scaffold-20261117` (未启动)

> ⚠️ **性质声明**: 本文件**不是**原始预注册。Arm1 已经跑完，其结果 (test hard 2077/2765 = 0.7512) 已被观察到。
> 本文件的价值在于：**在 Arm3 结果产生之前**冻结分析规则，防止此后继续做基于结果的选择。
> Arm3 的评测口径、配对集合定义、统计方法、δ 在此文件落定后不得再改。

---

## 1. 本实验的科学问题

本轮 #48 的核心问题是：**S₀ scaffold 对 Evolver 训练机制的贡献**。

不是：证明 Evolver 相对于 S₀ 的全量提升。

由此：

- 缺少 2765 题 S₀ 基线**不破坏**主要比较 (Arm1 vs Arm3)。
- Arm1 vs Arm3 只能识别**移除 S₀ scaffold 这一综合干预**的效果，**不能**拆分归因到四个联动机制
  (rewriter 输入 S₀ / 全局 Skill 合成 / 失败回退 / 退化判断)。

## 2. 比较结构 (冻结)

| 比较 | 定位 | 评测规模 | 状态 |
|------|------|---------|------|
| **Arm1 Full vs Arm3 Drop Scaffold** | **主要比较** | 完整 test split 2765 题，两臂共同有效 ID | 待 Arm3 |
| Arm1 vs S₀ | 参考比较 | 前 300 题，使用当前评测器重判缓存 | 已有 (定版 215) |
| Arm3 vs S₀ | 参考比较 | 前 300 题，同一缓存 | 待 Arm3 |
| 排除题后的敏感性分析 | 补充分析 | 见 §5 | 待证据审查 |

- **主测试集**: 完整 test split，**2765 题** (不是 300)。300 是 v2 run 的 `test_env_num` 截断视图，
  其 id = `test/items.json[:300]` 文件顺序前缀，**非预注册子集** (见 Issue #89)。
- **配对集合**: 两臂 `agent_ok=True` 的交集 (共同有效 ID)。两臂 infra-fail 记 `agent_ok=False`，
  不作为能力差异计入；交集之外的有效题单独列出，不进入 McNemar。
- **S₀ 参考基线**: 前 300 题，**定版文件**
  `appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.rejudged_d5_v2.json`
  (sha256 `039dd56ed00bb24b…`)，hard=**215**，`evaluator_version=runtime_evaluate_d5_rejudge_v1`。
  - 历史存储值 208 (`results.json`, sha256 `5f2ca7b0a50de994…`) **只作历史记录**，不再作为当前评测器下的有效基线。
  - 定版依据: 截断题面形态下重判 = 215，与 Arm1 run 内 Path-B 独立重建一致 (见 Issue #92)。
  - **限制**: 该基线只覆盖文件顺序前 300 题，**不能**外推为 2765 全量 S₀ 基线；本轮的 Arm1/Arm3 vs S₀
    结论均限于这 300 题，且该 300 题是文件顺序前缀而非预注册样本。

## 3. 主比较的执行规则 (冻结)

1. Arm3 使用独立 `out_dir` (`RUN_TAG=tf48-arm3-drop-scaffold-20261117`)，**不读取** Arm1 的
   skill / experience / experience_pool；共享的只有 (a) 同一份 train split，(b) 同一份 train S₀ cache，
   (c) 同一份 S₀ 初始 skill 文件，(d) 同一份 val/test 缓存。
2. 两臂训练题目集合相同 (Arm3 `drop_scaffold` 只在 rewriter/compose 处把 `s0_train` 置空，**不改变**训练题集；
   见 Issue #93)。
3. 两臂测试路径相同: `eval_independent` 在 `load_split(truncate=True)` 形态上逐题运行 best_S，
   `_evaluate_d5` 分流 (fn-call / stdin)。**不使用** `eval_on_routed` 作为任何结论依据 (Oracle-assisted，仅诊断)。
4. 若 Arm3 出现与 Arm1 不同的 infra 失败模式 (例如空 `s0_train` 导致 rewriter 全失败 abort，
   或 `n_real==0` RuntimeError)，记录为**实验中断**并单独分析，不得通过丢弃题目来"修复"。

## 4. 统计方法 (冻结)

| 项 | 方案 |
|----|------|
| 配对单位 | 单题 (同一 test id，两臂各一个 hard ∈ {0,1}) |
| 主检验 | 精确 McNemar (双侧 binomial exact，基于不一致对数) |
| 效应量 | 配对净增益 g = rescue (arm1_only_pass) − regress (arm3_only_pass)，及其比率 g / n_common |
| 置信区间 | Clopper-Pearson exact CI (净增益比率) |
| δ | **0.03**，标注为**历史设计参数**；本轮采用的**实际意义阈值**。n=2765 时 ≈ 83 题 |
| 多重比较 | 不做校正；主比较只有一个。S₀ 参考比较明确标为参考，不用于宣称优越性 |
| 报告 | 四象限计数 + p + g + CI 全部报告，无论显著与否 |

**δ 的说明**: δ=0.03 原为 300 题规模下设计 (≈9 题)。在 2765 题上它约等于 83 题，是更严格的净增益门槛。
本文件保留 0.03 作为"实际意义阈值"，**不因样本量变化而调整**，也不称为严格的实验前预注册阈值。

**结论判读规则 (冻结)**:
- 主比较净增益 ≥ δ **且** CI 下界 > 0 → "scaffold 对 Evolver 流程有可识别的正向贡献"。
- 净增益 ≤ 0 或 CI 跨 0 且点估计 < δ → "当前配置下未观察到 scaffold 带来可识别增益"。
- 其他情况 (点估计 ∈ (0, δ) 且 CI 跨 0) → "方向性提示，未达实际意义阈值"。
- **任何情况下都不以 p 值单独下结论**。

## 5. 107 题 (NameError 'Solution') 的处理 (冻结)

- **主分析保留完整 2765 题，不排除这 107 道题。**
- 逐题证据表 (题目 ID、原始题面、函数签名、测试代码、失败原因) 在最终报告中提供；
  各题需记录 (a) 是否与模型输出无关、(b) 是否存在客观可复核的排除标准、(c) 是否对所有 arm 同样适用。
- 只有同时满足三项标准的题，才在**补充敏感性分析**中排除。
- 全量结果与排除后结果**同时报告**。
- 该表**不作为 Arm3 启动阻塞项**。

**证据表已产出 (2026-11-19, Arm3 启动后, 0 API)**: `audit_0api_20261118/nameerror_107_evidence.json/.md`。
三项标准初判全部满足 (harness 对任何解都 NameError; 11 题已存储输出重评仍全 0)。见 Issue #94。

## 6. 训练预算

两臂相同: `ITERS=3 TRAIN_N=80 G=4`，共享 train S₀ cache (0 次额外 precompute)。
Arm1 实际独立测试调用 2765 次/臂；Arm3 同口径。

## 7. 结论边界 (冻结)

本实验可以主张:
- ✅ S₀ scaffold 对当前 Evolver 训练流程 (四机制联动) 的**整体**贡献。

本实验**不能**主张:
- ❌ Evolver 整体优于等预算 No-Evolution Control (需另设 Arm0)。
- ❌ 四个机制各自贡献的大小。
- ❌ Arm1 相对 S₀ 的 2765 全量提升 (S₀ 只有 300 题基线)。
- ❌ Arm1 相对 SkillOpt best (218 为旧 serde 值，且 300 题口径)。

## 8. Arm1 结论的分层定位 (冻结)

| 项 | 定位 |
|----|------|
| 训练过程 / 3 iters / iter1 ACCEPT | 保留 (探索性) |
| best_val_delta +0.1067 | 仅为原 val 协议下的训练指标，**不得**与 test hard rate 比较 |
| skill / experience 产物 | 保留 (80/80 题 train 输入 ID+cache 一致) |
| independent 2077/2765 = 0.7512 | 保留，作为主比较的一臂 |
| Arm1 vs S₀(215) 300 题: rescue 37 / regress 28 / net +9 / p=0.3211 | 参考比较；不显著 |
| Δ=+1862 (与 215 相减) | **作废** |
| McNemar 300 (file-order 前缀) | 仅探索性 |
| Retention 186/215 | 待 ID/版本一致性确认 |

## 9. 冻结检查清单 (Arm3 启动前)

- [x] S₀ 300 题参考基线定版 (215，版本化文件 + hash，见 §2)
- [x] train fail 池一致性核实 (同 split / 同 cache / 同形态 / drop_scaffold 不改题集，Issue #93)
- [x] 本分析方案修订冻结 (§1–§8)
- [x] Arm3 独立 out_dir 启动 (`run_20261119_091235`)
- [x] 107 题证据表 (2026-11-19 产出, Issue #94; 最终报告必须包含)

---

**签署**: 由用户 2026-11-19 批准 "有条件启动 Arm3"，本文件落实其五项执行顺序中的第 1–3 项。
