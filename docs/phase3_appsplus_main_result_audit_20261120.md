# Phase 3 — APPS Plus 主实验结果审计: 能否宣称 Evolver 有效 (2026-11-20)

**性质**: 结果审计 (0 API, 全部基于已落盘产物 + 现有运行时评测器)。
**核心问题**: 在 APPS Plus 独立测试集上, 自动 Skill Evolver 是否优于性能参照基线 (静态 S₀ / SkillOpt),
足以宣称"训练反馈驱动的 Skill 进化带来可泛化增益"。
**判定 (一句话)**: **不能宣称。** 在可比口径下 Evolver 与两个静态基线无统计差异; 点估计为正但不达 δ;
最大的功效缺口是缺 2765 题配对静态基线。Arm0 暂缓。

---

## 1. 交付项与结论

| # | Phase 3 交付项 | 结论 | 依据 |
|---|----------------|------|------|
| 1 | 固定最终数据集与测试协议 | ✅ 已固定 | §2 |
| 2 | S₀ / SkillOpt / Evolver 公平比较口径 | ✅ 已统一 (300 题同评测器) | §2.2, §3 |
| 3 | 300 与 2765 评测规模问题 | ⚠️ 已量化, 未消除 | §4 |
| 4 | 历史结果与新评测结果可比性 | ✅ 可换算 (统一 +7 协议修正) | §2.3 |
| 5 | Evolver 是否真的优于基线 | ❌ **无统计证据** | §3 |

---

## 2. 协议与口径

### 2.1 数据集 (已固定)

- 切分: `/data/zhuldz/skillopt/data/appsplus_capped_seed42` — train 80 / val 395 / test **2765**。
- **主测试集 = 全量 2765 题。** "300" 是 v2 run 的 `test_env_num` 截断, id = `test/items.json[:300]`
  文件顺序前缀, **非预注册子集** (issue #89)。300 只能作为与静态基线的共同分母, 不能外推为 2765。
- 评测器: D5 harness (`_evaluate_d5`, `tf_skill_optimizer_appsplus.py:414`), fn-call 题走
  `_evaluate_fn_call`, 其余走 SkillOpt `appsplus_evaluate`; 题面形态 = `load_split(test, truncate=True)`。

### 2.2 三方的公平比较口径 (本轮统一)

| 对象 | 产物 | 生成方式 | 评测方式 | hard (300) |
|------|------|---------|---------|-----------|
| S₀ | `test_eval_baseline/results.rejudged_d5_v2.json` | 存储 code | D5 重判 | **215** |
| SkillOpt | `test_eval/results.rejudged_d5_v2.json` | 存储 code | D5 重判 | **218** |
| Arm1 (Evolver full) | `test_independent_results.json` (2765) | 运行期生成 | 运行期 D5 | **224** (300 子集) / **2077** (2765) |
| Arm3 (drop_scaffold) | 同上 | 运行期生成 | 运行期 D5 | **208** (300 子集) / **1945** (2765) |

- 本轮新增: SkillOpt 的 D5 重判产物 + 审计 `audit_0api_20261120/skillopt_cache_300_rejudge.json`
  (脚本 `eva/fix_48_skillopt_cache_300_v2.py`)。此前 SkillOpt 的 218 只存在于 optimizer 内存
  (`rebuild_fn_call_cache`), 磁盘无版本化证据。
- 完整性核对: 重判 id 集与原 cache 一致、code 逐条一致、`n_turns=1`、`agent_ok=True` 300/300;
  7 处变化全部 0→1 (2268/2798/3812/5149/6508/863/933), 无任何 1→0。

### 2.3 历史结果可比性

- 旧 harness 对 fn-call 题存在**系统性欠给分**。S₀ 与 SkillOpt 都恰好翻转 **7** 题, 全部 0→1,
  且 6 题相同 (S₀ 独有 5090, SkillOpt 独有 2268)。**方向与量级一致 → 协议修正对称, 不是挑臂**。
- 历史数字换算: S₀ 208 → 215; SkillOpt 211 → 218 (旧报告 0.7033 即 211, 属旧口径); Arm1/Arm3 的
  2470+7=2077 已在运行期用新评测器, 无需换算。
- 仍未统一的: SkillOpt "best" 的选取口径 (val routed delta) 与 Arm1 相同, 但 SkillOpt 是整个框架的产物,
  未含 scaffold; 这属于变量定义差异, 不是评测偏差。

---

## 3. 主判定: Evolver vs 静态基线 (300 题配对, 独立非路由)

配对 = 300 个共同有效 ID; 独立非路由 = 各臂自己的生成, 不复用任何缓存。McNemar 精确 p; 净增益率的
CI 为对不一致对比例 r=rescue/d 的 CP 区间仿射 (与 `docs/48_analysis_plan_amendment_20261119.md` §4 同法);
δ = 0.03 (≈9 题/300)。

| 比较 | 臂 (300) | 基线 (300) | 净增益 | rescue/regress | 净增益率 CI | p |
|------|---------|-----------|--------|----------------|------------|---|
| Arm1 vs S₀ | 224 | 215 | **+9** | 37 / 28 | [−0.0258, +0.0830] | 0.3211 |
| Arm1 vs SkillOpt | 224 | 218 | **+6** | 27 / 21 | [−0.0282, +0.0657] | 0.4709 |
| Arm3 vs S₀ | 208 | 215 | **−7** | 24 / 31 | [−0.0722, +0.0282] | 0.4188 |
| Arm3 vs SkillOpt | 208 | 218 | **−10** | 16 / 26 | [−0.0740, +0.0122] | 0.1641 |

**判读**: 四条比较 CI 全部跨 0 且覆盖 δ; Arm1 点估计为正 (+9/+6), Arm3 为负 (−7/−10), 但均落在
n=300 的噪声带内。**没有任何一条达到"CI 下界 > δ"或"CI 下界 > 0"。**

### 3.1 为什么不用 routed 结果 (243 vs 236)

`test_routed_results.json` 在 S₀ 通过的题上**复用 S₀ 缓存 code**, 两臂输出高度相关, 且 S₀-pass 题
本身更易通过 (Oracle 辅助)。routed 表 (Arm1 243 / Arm3 236 / SkillOpt 218 / S₀ 215) 只能作诊断,
**不能作为泛化证据**; 用它会把 +28/+25 误读为 Evolver 增益 (其中大部分是路由复用)。

### 3.2 为什么"n=300 功效不足"不是可接受的辩护

- n=300 双臂差的 MDD ≈ 11.2 pp。Arm1 vs S₀ 必须 ≥ 11.2 pp 才可能显著, 而点估计只有 3.0 pp。
- 若只看 p 值, 确实可以喊功效不足; 但 (a) 2765 题上 Arm1/Arm3 都没有对应静态基线, 这一缺口是**数据缺口**
  而非无法测; (b) +3.0 pp 点估计即使全部为真也低于 δ=3 pp 的有用阈值, 不是"效应存在但测不出"的典型形态。
- 因此本审计将其定位为 **"未证明"**, 不是 "已证伪", 也不是 "因功效不足无法判定"。

### 3.3 其他限制

- **Arm3 的 S₀ cache 参数与 Arm1 不同**: Arm3 由 `results.rejudged_d5_v2.json` (215) 启动, Arm1 用旧
  `results.json` (208); 运行期 optimizer 将控制臂映射到 215, best_hard/legacy 一致 (218/211),
  对 §3 的独立比较无影响, 但需在复现说明中注明。
- **两臂串行运行, 共享 20 RPM 账号**: 无并行争用, 但存在跨时段 API 端波动; 两臂都按 `retries=5`
  与同一 endpoint 配置运行, 视为同条件。
- **Evolver 的历史过拟合背景**: 早期 88/300 与 293/1000 轮次出现零样本外增益, 说明"训练集提升 →
  测试集泛化"不是默认成立的链路; 本轮 best skill 的 val routed delta +0.1067 (Arm1) 与 +0.0750 (Arm3)
  也未在独立集上同样幅度兑现。

---

## 4. 300 vs 2765: 规模问题量化

| 项 | 现状 | 影响 |
|----|------|------|
| 主比较 (Arm1 vs Arm3) | 2765 共同有效 ID, 完整 | 结论强度最高 (p=2.26e-09) |
| Evolver vs 静态基线 | 仅 300 前缀 | 每臂折算 +9/+6/−7/−10, CI 宽度 ±5 pp 量级 |
| 300 是否为随机子集 | 否 — 文件顺序前缀 | 可能存在难度序偏差, 无法事后随机化 |
| 2765 配对静态基线的成本 | 每基线 2765 次 flash 调用; 20 RPM 全账号共享 → 约 2.3 h 纯节流 | S₀ + SkillOpt 共 ≈ 4.6 h 账号时间, 0 训练成本 |

**建议的口径**: 若后续要继续主张 Evolver 有效, **最低要求**是补 2765 题配对静态基线 (S₀ 优先,
   预算允许再加 SkillOpt), 而不是继续在 300 上做子分析。若只用手上的 300:
   - 可用语句: "在 300 题共同子集上, Evolver 相对静态基线的净增益为 +9 (CI [−0.026,+0.083]), 未达 δ。"
   - 不可用语句: "Evolver 在 APPS Plus 上优于 S₀ / SkillOpt"。

---

## 5. 最终判定

| 命题 | 判定 | 依据 |
|------|------|------|
| S₀ scaffold 对 Evolver 训练流程有正向贡献 | ✅ 已支持 (2765 配对, 下界 > δ) | Arm1 vs Arm3, §3 / 48 归档 |
| Evolver > 静态 S₀ | ❌ 未证明 | +9, p=0.32, CI 跨 0 |
| Evolver > SkillOpt | ❌ 未证明 | +6, p=0.47, CI 跨 0 |
| Evolver > 等预算无进化对照 | ❌ 不知道 | Arm0 未启动 (用户暂缓) |
| scaffold 各成分独立贡献 | ❌ 未分解 | 设计上只识别复合效应 |

**允许的措辞**: "在 APPS Plus 独立测试集上, 训练反馈驱动的 Skill 进化相对静态 S₀/SkillOpt 的增益
为 +6~+9 题 (300 题子集), 未达到统计显著或预设实际有用阈值 δ=3pp; 因此不能宣称 Evolver 有效。
已支持的是 scaffold 对进化流程的正向作用。"

**下一步依赖**: 用户已决定先完成本审计 (已交付), Arm0 暂缓; 若后续转向"Evolver 有效性"主张,
优先补 2765 题配对静态基线, 再决定 Arm0 是否仍是归因目标。

---

## 6. 证据清单 (可复现)

- S₀ 定版: `appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.rejudged_d5_v2.json`
  (sha256 `039dd56e…`), hard 215。
- SkillOpt 定版: 同目录 `test_eval/results.rejudged_d5_v2.json` (sha256 `ed74248dd94b99a0…`), hard 218;
  原 `results.json` sha256 `84a72049736115da…` (hard 211)。
- 审计: `appsplus-skillopt-flash-20261111-v2/train/audit_0api_20261120/skillopt_cache_300_rejudge.json`
  (逐题 diff、双文件 hash、code 一致性断言)。
- Arm1/Arm3 独立结果: `tf48-arm1-full-20261117/run_20261118_091052/test_independent_results.json`,
  `tf48-arm3-drop-scaffold-20261117/run_20261119_091235/test_independent_results.json`。
- 主比较归档: `tf48-arm1-full-20261117/run_20261118_091052/audit_0api_20261118/paired_arm1_vs_arm3.{json,md}`。
- 本审计 §3 的配对计算 (0 API, 可复现):
  `eva/phase3_paired_300_0api.py` →
  `tf48-arm1-full-20261117/run_20261118_091052/audit_0api_20261120/phase3_paired_300.json`
  (+ `phase3_common_ids_300.txt`, 含全部输入文件 sha256)。
