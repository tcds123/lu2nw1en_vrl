# LiveMath 新机制完整 Run 方案（2026-11-26，待审冻结）

> 离线设计，不动 API，不改门控。研究目标与实验原则已冻结（见 `memory/mathline-research-position-20261126.md`）。本方案待用户审阅后冻结参数，API 执行需单独授权。

## 核实完成 (2026-11-26, 用户要求的2项冻结前核对)

### 核实A: 旧增益数字的评估集归属 (用户最关注)
| 数字 | 评估集 | 来源 | 状态 |
|------|--------|------|------|
| +0.254 | **L1 select=71 (in-sample)** | multirun1 evolver_log best_val_delta iter4 | ✓ 确认 in-sample |
| +0.216 | L1 select=71 iter1 | 同上 iter1 | ✓ |
| **+16.1pp** | **L2 test=106** | 2026-07 v2 evolver, 4-seed, CI[+0.126,+0.196] | ⚠️ **原始产物已清理**, 不可复算 |
| +10.8pp | L2 test=106 | 2026-09 multirun1, 2-seed (on=0.4387 vs s0=0.3302) | ✓ 可复现最近值 |
| +18.6pp | L2 test=106 | 2026-09 onoff_shuffled, 3-seed | ✓ |

**结论**:
1. +16.1pp 确实是 **L2 test=106**, 非 in-sample。背景文档标"in-sample"是**错的, 已修**。
2. 但 +16.1pp 原始产物(2026-07 v2)已清理, 无法复算。可复现最近 L2 = +10.8pp(2-seed) / +18.6pp(3-seed shuffled)。
3. **新方案 L2 与旧 L2 可比**(同 test=106)。但旧 +16.1pp 不可复算, 作"参考效果"非"对齐基线"。可复现基线 = +10.8pp(multirun1)。
4. select=71 = train18+val53, split_manifest 闭合 177。✓ 无数据版本混用。

### 核实B: pool_filter 与 rescue 代码级关系
- **pool_filter** (admission_filter L3072): 候选入 archive **前**的准入门(质量+5峰尖峰+git-diff多样性)。拒→不入archive。
- **rescue** (_combined_rescue L3179+): 候选**未过CI晋升**后的 else 分支, 用子集/分组信号救回。在 pool_filter **之后**(只对已入archive候选生效)。
- **关系: 不互斥, 不功能重复**。pool_filter 做质量过滤, rescue 做子集救援, 不同阶段。
- **我之前说"避免冲突"是推测非代码确认 — 用户纠正对**。→ **pool_filter 改回 ON**(对齐旧evolver)。

### 机制开关表修正 (用户指出"5机制"计数问题)
完整开关:
| 开关 | 状态 | 旧evolver |
|------|------|-----------|
| facet_tree | ON | 有 |
| expand | ON | 有 |
| **pool_filter** | **ON** (改回, 对齐旧) | 有 |
| subset_admit | ON | 有 |
| paired_rescore | ON | **无(新)** |
| rescue_on_reject | ON | **无(新)** |
| veto_upper_bound + veto_skip_n1 | ON | **无(新, _veto_regression 2子开关)** |
| facet_group_admit | ON | **无(新)** |
| held_out_gate | OFF | 无 |
| emergent_facets | OFF | 无 |

**新增机制 = 4 项**(paired_rescore/rescue/_veto_regression/facet_group_admit), 非5项。veto 两子开关算1项。

---

## 冻结后的最终配置（2026-11-26，用户条件批准 + 2 项核实完成）

用户已选定 Option A（eval_k=3, iters=6, paired_rescore ON），并要求完成 2 项核实后方可冻结。核实 A/B 均已完成（见上文）。本节为冻结配置，**API 执行仍需单独授权**。

## 一、实验配置

### 1.1 冻结配置表

| 参数 | 旧 evolver (multirun1, 2026-09) | 新机制完整 run (冻结) | 对齐说明 |
|------|------|------|------|
| split | `livemathematicianbench_1-3-6_seed42` (train18/val53/test106) | **同** | 不变 |
| init_skill / S₀ | `skillopt/.../initial.md` (1018B, hash 68571cb2) | **同** | 不变 |
| model | DeepSeek-V3.2 (`aimpapi.midea.com`) | **同** | 不变 |
| iters | 4 | **6**（iter4 必保留检查点） | 略多于旧(4)，给新机制更多搜索空间 |
| K (生成候选数/facet) | 4 | **4** | 同 |
| eval_k | 3 | **3**（Option A） | 与旧 k=3 对齐，使新旧可比 |
| train_n | 35 | **18** | 旧 35 是 bug（train 仅 18 题，slice 无效=用全 18）；新显式 18=真实全量，实质相同 |
| facet_tree / expand / subset_admit | ON | **ON** | 同 |
| **pool_filter** | ON | **ON**（核实B确认不与 rescue 冲突） | 对齐旧 evolver |
| paired_rescore | OFF (旧无此机制) | **ON** | 新机制核心，smoke 已验证不崩 |
| rescue_on_reject | OFF (旧无) | **ON** | 新机制 |
| veto_upper_bound + veto_skip_n1 | OFF (旧无) | **ON** | 新机制（_veto_regression 2 子开关算 1 项） |
| facet_group_admit | OFF (旧无) | **ON** | 新机制 |
| held_out_gate | OFF | **OFF** | 保持原 split |
| emergent_facets (EFL) | OFF | **OFF** | MCQ taxonomy 未适配，ON=no-op |
| 门控参数 (eps/CI/阈值) | — | **全部不动** | 用户明确：不改已验证门控 |

### 1.2 对齐情况说明

- **eval_k=3**：与旧 evolver k=3 对齐，新旧可比。**但 k=3 的 pass_rate 量化噪声比 k=8 大**，paired_rescore 对消"同批同题共享采样噪声"，**不消除题目本身在 k=3 下的 pass_rate 离散**。不预设 paired_rescore 能补偿全部噪声——若 L2 CI 跨零但 L1 有稳定晋升，按"中等成功"处理而非直接判失败。
- **train_n=18**：旧 35 是 bug（train 仅 18），slice 无效=实际用全 18。新显式 18=真实全量，实质对齐。
- **pool_filter=ON**：核实 B 已确认 pool_filter（admission，入 archive 前）与 rescue（post-CI-rejection else 分支）在不同阶段，不互斥不重复。ON 对齐旧 evolver。**此为对齐旧实验配置，非"控制变量"措辞**。
- **新增机制 4 项**（paired_rescore / rescue / _veto_regression / facet_group_admit），非 5 项。

---

## 二、评估协议

### 2.1 三个评估层（解决"旧 in-sample vs 新 held-out 不可比"）

| 层 | 数据 | 用途 | 何时测 |
|----|------|------|--------|
| **L1 优化过程增益** | select 集 (train+val=71) | evolver 搜索过程中的 best_val_delta（含 meta，in-sample） | 每轮自动记录 |
| **L2 冻结确认增益** | **test 集 (106 题)**，与搜索过程隔离 | 最终 best_skill vs S₀ 的独立评测 | run 结束后单独跑 |
| **L3 分层增益** | test 集分 meta_89 / not_meta_17 | 看增益来自哪（记录用，非门槛） | run 结束后单独跑 |

**关键**：L1 是优化过程信号（旧 +0.254 在此层），L2 是独立确认（旧 +16.1pp test 复评在此层）。**新机制的 L1 与旧 L1 可比（同 select 集）；新 L2 与旧 L2 可比（同 test 集）。不跨层比**。

### 2.2 主要指标与成功标准

- **主指标**：L2（test=106）上 best_skill 相对 S₀ 的 pass rate 增益 Δ，bootstrap CI
- **成功标准**（预先固定，非硬性阈值）：
  - **强成功**：L2 Δ > 0 且 CI_lower > 0（独立评测显著正向）→ 新机制完整正向证据
  - **中等成功**：L2 Δ > 0 但 CI 跨零，且 L1 有稳定晋升（≥3 轮 promoted）→ 有信号待加固
  - **弱/失败**：L2 Δ ≈ 0 或负，L1 也 0 晋升 → 候选生成瓶颈，转重构 patcher
- **参考**：旧 evolver L2 = +16.1pp（强成功）。新机制达到相近 = 强证据；较小且可靠 = 同样有价值（用户原则）
- **记录**：L3 分层增益用于理解增益来源（meta vs not_meta），**不作门槛**

### 2.3 与旧机制比较方式

**不在本次 run 内做新旧直接对比**（那需统一协议重测旧 evolver，翻倍预算）。本次 run 只回答"新机制能否自主产生正向 Skill"。新旧比较留作后续控制实验。

### 2.4 复现 vs 归因的区分（用户问题 B 要求）

本次 run 同时承担两个目标，需在报告时分开：
- **效果复现**：新机制完整 run 的 L2（test=106）是否产生 over S₀ 的可重复正向增益。对标旧 L2：+10.8pp(2-seed 可复现最近值) / +16.1pp(4-seed 但产物已删，参考值)。
- **机制归因**：若有效果，增益来自哪些机制？通过 gate_trace 的 per_item_delta + operator/facet 维度区分（候选生成了哪些规则、哪些被保留）。若 L2 跨零，也需从 gate_trace 区分是"候选生成瓶颈"(无有效候选) 还是"选择瓶颈"(有效候选被门控拒)。
- **可复现基线**：旧 +16.1pp 原始产物已清理、不可复算，**作"参考效果"非"对齐基线"**。可复现对标 = +10.8pp（multirun1, 2-seed）。

---

## 三、运行日志（完整轨迹，区分生成/选择瓶颈）

### 3.1 每轮每候选记录（落盘到 run 目录）

每个候选记录以下字段（满足用户"保留完整进化轨迹"要求）：

| 字段 | 内容 | 用途 |
|------|------|------|
| `iter` | 轮次 | 定位 |
| `operator` | facet_edit/overlay/expand/crossover/strengthen | 算子来源 |
| `facet` | 哪个 facet 被改 | 作用范围 |
| `parent_hash` | 父 skill hash | 父子版本关系 |
| `cand_hash` | 候选 skill hash | 候选文本标识 |
| `cand_skill_text` | 候选完整文本 | 复现 + 看 evolver 发现了什么规则 |
| `val_delta` | 总体 paired delta (select=71) | 主指标 |
| `val_ci` | bootstrap CI lower (n=71) | 门控依据 |
| `not_meta_ci` / `meta_ci` | 子集 CI | 分层信号 |
| `gate_decision` | PROMOTE / KEEP-PARENT / RESCUE / LOST | 门控决策 |
| `gate_reason` | 拒绝/晋升原因 | 区分"没生成有效候选"vs"生成了但没保留" |
| `per_item_delta` | **逐题 delta {qid: [d]}** | ⚠️ **本次补落盘**（smoke 缺口）— 稳估 σ + 离线复算 CI |

### 3.2 终局记录

- `best_skill_evolved.md`（最终 best）
- `evolver_log.json`（逐轮汇总 + operator_counts + 总 promoted）
- `gate_trace.jsonl`（每候选一行，含 per_item_delta）— **新增**
- L2/L3 评测结果（run 后单独跑）

### 3.3 预算追踪

每轮记录实际 API calls + tokens，落盘 `budget_trace.json`，超硬性上限自动停。

---

## 四、预算（冻结口径：eval_k=3, V3.2, RPM=28）

### 4.1 按组成拆分

| 组成 | 计算 | 预计 calls |
|------|------|-----------|
| S₀ 重算 (首次) | 71 题 × k=3 × 1 次 | 213（已有 k=8 cache+manifest，但 k≠3 需重算 → **213**） |
| 每轮 facet_edit | 3 facet × K=4 候选 × 71题 × k=3 × 2(paired) | 5,112/轮 |
| 每轮 overlay | 1 × 71 × k=3 × 2 | 426/轮 |
| 每轮 expand | 1 × 71 × k=3 × 2 | 426/轮 |
| 每轮 crossover | 1 × 71 × k=3 × 2 | 426/轮 |
| 每轮 collect_failures | 71 × k=3 (单 skill) | 213/轮 |
| 每轮 LLM 生成候选 | ~10 次 | ~15/轮 (可忽略) |
| **每轮小计** | | **~6,600/轮** |

⚠️ S₀ 重算说明：smoke 的 cache 是 k=8 + manifest。冻结 k=3 后 cache 不复用，需重算 S₀(k=3) = 213 calls + 写新 manifest。这是一次性成本。

### 4.2 总预算 + retry overhead

| 项 | calls |
|----|-------|
| S₀ 重算 (1次) | 213 |
| 6 轮进化主体 | ~39,600 |
| **基线总计** | **~40k** |
| retry overhead（API 限流/超时重试，估算 +10~15%） | +4~6k |
| **含 retry 预估** | **~44~46k** |

### 4.3 硬性上限（用户：40k 不能直接成为硬上限）

- **calls 软目标**：~40k（基线，不含 retry）
- **calls 硬上限**：**60k**（含 retry 的 ~1.4× 容错，超过即停，避免 retry 雪崩）
- **墙钟上限**：36 小时（40k @ RPM=28 ≈ 24h，留 1.5× 容错）
- **超限动作**：自动停 + 落盘当前 best_skill + gate_trace + budget_trace
- **iter4 检查点**（用户要求）：iter4 结束时强制落盘 checkpoint（best_skill_iter4 + gate_trace_iter4 + budget_trace_iter4），即使后续超限/崩溃，前 4 轮结果可分析。

---

## 五、不动的门控（用户明确要求）

**门控规则全部不改**（eps=0.02, CI>0 晋升, subset/group/veto 逻辑不变）。当前最优先是在稳定选择机制下检验新 evolver 完整搜索能力。改门控 = 一旦结果变化无法归因。

---

## 六、冻结状态

| 项 | 状态 |
|----|------|
| 研究目标与实验原则 | ✅ 已冻结（`memory/mathline-research-position-20261126.md`） |
| 核实 A（旧增益评估集归属） | ✅ 完成（+0.254=L1 select in-sample, +16.1pp=L2 test, 后者产物已删） |
| 核实 B（pool_filter vs rescue） | ✅ 完成（不冲突，pool_filter→ON） |
| 配置参数（eval_k=3, iters=6, pool_filter=ON, 新机制4项ON） | ✅ 已冻结 |
| 预算（~40k 基线 / 60k 硬上限 / iter4 检查点） | ✅ 已冻结 |
| per_item_delta 落盘 | ✅ 纳入（补 smoke 缺口，无 API 成本） |
| **API 执行授权** | ⏳ **待用户单独授权** |

冻结完成。下一步 = 用户授权 API → 执行完整 run → 落盘 gate_trace → L2/L3 独立评测。
