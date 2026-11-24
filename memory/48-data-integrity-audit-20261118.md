---
name: 48-data-integrity-audit-20261118
description: #48 数据完整性审查 — manifest 定义 test=2765, 300 为 eval 时 file-order 截断; Arm1 最终测试结论口径需重定义
metadata:
  type: project
---

# #48 数据完整性审查 (2026-11-18, 纯文件/代码, 0 API)

触发: Arm1 完成核验时发现 test/items.json = 2765 题, 而 [[48-formal-run-launch-20261117]] 记为 300 题。
用户指令: 升级为 #48 数据完整性审查, **查清前不启动 Arm3**, 不改代码, 不重跑模型, 不额外 API。

## Stage 1 — Split 真实定义 (已证)

`/data/zhuldz/skillopt/data/appsplus_capped_seed42/split_manifest.json`:
- `counts_capped`: train 80 / **val 395** / **test 2765**
- note: "val/test kept full (env_num caps at eval-time via base.py build_eval_batch)"
- 源: `appsplus_2-1-7_seed42` (train 790/val 395/test 2765), seed 42
- 所有 split 文件 mtime 2026-11-11 17:05:43, 之后未改动 → **用户场景 2 成立**: manifest 定义 2765, 300 是评测时截断。

**启动记忆写错**: "train 80 / val 100 / test 300" 是 eval 截断后的视图, 不是 split 定义。val 文件实为 395。

## Stage 2 — 300 题 cache 来源 (已证)

- v2 run `config.json`: `sel_env_num=100`, `test_env_num=300` (2026-11-11 run 的评测截断参数)。
- v2 test S₀ cache (300) == **`test/items.json[:300]` 按文件顺序** (True, 非 id 排序, 无随机种子)。
- → 这 300 题**不是预注册的子集**, 是"文件前 300 条"。样本选择方式不可复现为预注册协议。

## Stage 3 — 2026-11-11 v2 run

- train.log: `train=80 val=395 test=2765 (from appsplus_capped_seed42)`; seed 42, cap 16384, temp 0。
- 载入了 2765, 但 test 评测只用了前 300 (test_env_num=300)。**训练/评测口径与 #48 正式 run 不对称**。

## Stage 4 — Arm1 执行时输入 (已证)

run: `tf48-arm1-full-20261117/run_20261118_091052`

| 文件 | 题数 | 说明 |
|------|------|------|
| s0_cache_val_d5.json | 100 | val 截断 (sel_env_num=100), mtime 09:10:53 (run 启动时) |
| s0_cache_test_d5.json | **2765** | mtime 09:10:55, 启动时生成 |
| s0_cache_best_d5.json | **2765** | mtime 15:19:08, 最终测试时生成 |
| test_independent_results.json | **2765** | 与 test/items.json 完全一致 (C==T), hard 2077, rate 0.7512 |
| test_routed_results.json | 300 | 诊断用 (Oracle-assisted) |
| test_retention_results.json | 215 | preserved 186 |

- Arm1 实际花了 **2765 次 flash 调用** 做 independent test (非预算表的 300), budget 字段亦记 independent 2765。
- 无 post-hoc 覆盖迹象: 启动时 cache 与结束时 cache 题数一致, 结果 id 集 == 全 test 集。
- 分片通过率: first300 0.7467 (224/300) vs rest2465 0.7517 (1853/2465) → 前 300 题略难, 但无系统性选择偏差的证据; **仍不能把 2765 结果与 300 题基线 (215 etc.) 直接相减**。

## 新发现: 109 道 NameError 'Solution' 结构性不可解

- test 集 109 题 hard=0 且 fail_reason 含 `Solution`: 题面为 LeetCode `class Solution: def fn(...)` 格式, 但 test_code 是 stdin 格式 → stdin harness 把类文本当输入喂进去, `Solution()` 未定义。
- 全部 hard=0 → 结构性不可解, 不是模型能力问题。其中 11 题落在前 300。
- 同类: D5 分类器只按 test_code stdin 判定 fn-call, 导致 LeetCode-stdin 混合格式题被误判为 stdin 题。
- 这 109 题让任何 2765 分母的通过率偏低约 3.9 个百分点当量 (109/2765)。

## fn-call 0 vs 9 差异 (未解)

- 预评估与当前代码重算: fail 池 27 题 fn-call=0, 80 题训练集 fn-call=0。
- Arm1 run 日志 coverage 打印 fn-in-fail=9 / fn-in-train-used=9 → 与当前代码结果不符, 未复现, 未定位原因。
- 影响范围: 若 9 属实, 预评估"Arm2/Arm4 干预为零"的结论需重新核对 (但 Arm2/4 本轮未跑, 不影响 Arm1/Arm3 主结论)。

## Arm1 结论可用性 (按用户分层, 更新后)

| 项 | 状态 |
|----|------|
| 训练过程/3 iters/iter1 ACCEPT | 暂保留 |
| best_val_delta +0.1067 | 有条件保留 (val 100 题截断口径, 非全 val 395) |
| skill/experience 产物 | 保留 (训练输入 = 80 题 train, ID/cache 一致) |
| independent 2765 结果 | **新口径**: 可作探索性/全测试集结果; 不能与 300 题基线(215/218)直接比 |
| Δ=+1862 (与 215 相减) | **作废** (分母不可比) |
| McNemar 300 / net +9 | 该 300 是 file-order 前 300, 非预注册 → 仅作探索性, 不作预注册结论 |
| Retention 186/215 | 待 ID/版本一致性确认 |

## 待用户决策 (三类结果)

1. **2765 全测试集为主口径** (已跑完, 0 额外成本) + 明确排除或修复 109 道结构性不可解题; 基线需在 2765 同口径重算 (S₀/SkillOpt 基线要重跑或用 Arm1 的 s0_cache_best 生成基线)。
2. **重建预注册 300 题子集** (需有记录证明先于结果定义; 目前证据是 file-order 前缀 → 不满足) 然后 Arm1/Arm3 都用该子集。
3. 标 Arm1 探索性, 重建可审计正式实验。

## 未做

- 未启动 Arm3 (用户封锁)。
- 未改代码, 未额外 API 调用。

## P0 0-API 重算结果 (2026-11-18, 从原始 JSON 重算)

脚本: `eva/recompute_48_baseline_0api.py` → `audit_0api_20261118/baseline_recompute.json`;
      `eva/recompute_48_s0_diag_0api.py` → `audit_0api_20261118/s0_cache_diagnostics.json`

| 项 | 结果 | 判定 |
|----|------|------|
| Dataset | test 2765 (唯一 id 2765); Arm1 结果 2765; s0_cache_test_d5 300; s0_cache_best_d5 300; v2 baseline 300 | ID 一致 (test==Arm1, 三个 300 缓存 == test[:300]) |
| S₀ 覆盖 2765 | **False** — 只有前 300 | ❌ 全量配对 Δ 不可算 |
| S₀ 300 cache 可复核 | 293 一致 / **7 条不一致** (2798,3812,5090,5149,6508,863,933; 旧 hard=**0** PartialFailure, 截断形态重判 hard=**1**) | cache 陈旧 (**方向 2026-11-19 更正**, 见下) |
| S₀ 重判 (当前 runtime 评测器, 截断形态) | hard **208 → 215** (0.7167) | **已定版**: `results.rejudged_d5_v2.json` |
| Arm1 | 2077/2765 = **0.7512** (agent_ok 全 2765) | 真 |
| 配对 Arm1 vs S₀ (300, **旧 raw 形态重判 208**) | rescue 44 / regress 28 / net +16 (0.0533), p=0.0764, CI95 [-0.0053,+0.1074] | ❌ **作废**: 用了未截断题面, 检测器全判 stdin |
| 配对 Arm1 vs S₀ (300, **定版 215**) | both_pass 187, both_fail 48, rescue 37, regress 28, **net +9 (0.03)**, p=0.3211, CI95 [-0.0258,+0.083] | ✅ **参考比较采用此组**; 与 Arm1 run.log 内 Path-B 重建一致 |
| 历史存储值 208 (`results.json`) | — | 只作历史记录 |
| 107 题 NameError | 实为 **107** (非 109); 全部 `class Solution` 题面 + stdin test_code; S₀ pass 0, Arm1 pass 0; 11 题在前 300 | 待正式证据表 |
| 291 fn-call 题 (截断形态) | question 具签名 2764/2765, `_d5_extract_sig` 成功 2765/2765; class-Solution 风格 260 题 (Arm1 132 pass), 其余 stdin 故事题 2504 (Arm1 1945 pass) | 签名存在 ≠ fn-call 判据 |

## 测量层问题 (新发现, 影响跨 arm 配对可比性)

1. **缓存 serde 与 runtime serde 不同**: Arm1 的 `test_independent_results.json` 由 runtime `run_one` 用 `load_split(truncate=True)` 题面 + `_evaluate_d5` 产生; 而 v2/S₀ 存储 artifact 的 hard 来自旧 serde 路径 (fn-call 题 "nonzero returncode=1")。→ 7 条 S₀ cache hard=1 是旧口径残留。
2. **D5 分类器形态** (2026-11-19 更正): `run()` 的 `rebuild_fn_call_cache` 收到的是 `load_split(truncate=True)` 的 item, **形态正确** (Arm3 run.log: val fn-call=21, test fn-call=30, stdin-polluted=0)。raw 形态误判只发生在独立脚本 `eva/preassess_48_0api.py` 里, 即 train "0 vs 9" 的来源 (Issue #93)。
3. 修复不耗 API: 用当前 `_evaluate_d5` 在截断形态上重判 S₀ 300 题存储 code → S₀=**215** (已落盘版本化文件)。**全量 2765 S₀ 基线仍缺失** (需 2765 次生成, 非 0-API)。

## 结论: "核验无误" 是否成立

**不成立 (截至 2026-11-18); 2026-11-19 已按用户批准的处理口径处理完毕**:
- S₀ 全量 2765 基线不存在 → 用户判定这不破坏主比较 (Arm1 vs Arm3, 2765 共同有效集); Arm1/Arm3 vs S₀ 只作 300 题参考比较。
- S₀ 300 cache 7 条陈旧 → 已重判落盘版本化文件 hard=215 (原 208 保留为历史值), 见 Issue #92。
- 存储 artifact 与 runtime 评测路径的 serde 差异会污染任何 Arm1-vs-存储基线 的配对。
- Arm1 vs Arm3 是同一 optimizer + 同一 runtime 路径 → 该配对**不受上述 serde 问题影响**(两者都 runtime 评测), 这是唯一干净的配对。

## 2026-11-19 状态: Arm3 已启动 (用户有条件批准)

- **处理口径 (用户冻结)**: 主比较 = Arm1 Full vs Arm3 Drop Scaffold, 2765 题共同有效集; Arm1/Arm3 vs S₀ = 300 题参考比较; 排除 107 题 = 补充敏感性分析。见 `docs/48_analysis_plan_amendment_20261119.md`。
- **Step1 S₀ 定版**: `eva/fix_48_s0_cache_300_v2.py` → `appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.rejudged_d5_v2.json` (300 条, hard=**215**, sha256 `039dd56ed00bb24b…`; 原 `results.json` sha256 `5f2ca7b0a50de994…` 未动); 审计 `audit_0api_20261118/s0_cache_300_rejudge.json` (7 题 old/new/response 明细)。
- **Step2 train fail 池**: 同 split / 同共享 cache (`tf-appsplus-routing-formal-20261113/train_s0_cache.json`, 80/80, hard=53) / fail 池 27 / 截断形态 fn-call 9 (raw 0 为 preassess 脚本 bug) / drop_scaffold 不改题集 (Issue #93)。
- **Step3**: 分析方案修订冻结 (见上)。
- **Step4 Arm3 启动**: `RUN_TAG=tf48-arm3-drop-scaffold-20261117`, out_dir `run_20261119_091235`; 启动日志核对: val cache 100 (D5 重建 58→65, fn-call 21, stdin-polluted=0); test cache = 定版重判文件 300 (D5 重建 215→215, fn-call 30, changed=0); train 80 (S₀-fail=27, 取前 27); `s0_train=''`, 全局 S₀ 仍 1796 chars。
- 启动脚本 `launch_tf_appsplus_routing.sh` 的 `S0_CACHE_TEST` 已改为定版文件 (原来指向 208)。

## 补充 (子代理只读审计, 2026-11-18)

**截断实现**: `skillopt/skillopt/datasets/base.py:496-510` `SplitDataLoader.build_eval_batch`:
```python
items = self.get_split_items(split)
if env_num and env_num < len(items): items = items[:env_num]   # 无 shuffle, seed 不进切片
```
300 来自 `skillopt/configs/appsplus/default_flash.yaml:62 test_env_num: 300` (注释 "paired McNemar ±3pp; 确定性子集 (可扩 2765 复测)"), 基类 `_base_/default.yaml:91` 默认 0=不截断。

**"确定性"的含义澄清**: 300 是 n 事先定好, 但**具体 id 未预注册** —— 它是 seed42 shuffle 后 split 文件的顺位前 300。rollout 落盘时按 id 字典序重排 (`appsplus/rollout.py:167`), 所以 results.json 顺序 ≠ items.json 顺序 (易误判)。原选 100 题 = `val/items.json[:100]`, 与 300 无交集。

**是否泄漏**: 无。300 是 test split 的位序前缀, 与 val 训练无关; 题 id 在全集百分位 min 0.004 / 中位 0.531 / max 0.999 → 覆盖全段, 非低 id 偏好。

**v2 时间线**: split 17:05:43 → config.json 17:21:21 → selection 17:28 → test_eval_baseline 20:04 → test_eval 20:19; 之后无改动。无 git commit 记录 (outputs 被 gitignore)。

**tf_skill_optimizer_appsplus.py 无环境/题数截断参数** —— 它直接读全 split, 所以 Arm1 跑 2765 是**设计如此**, 不是 bug; 预算表按 300 估是错的。后续若要在 300 上评测, optimizer 需要新增一个显式 `--test-n` (或 `--test-ids-file`) 参数, 不能依赖 SkillOpt 的 env_num 路径。

**DELTA_MIN=0.03 的口径**: 代码注释按 n=300 (≈9 题) 设计; 在 n=2765 上 0.03 ≈ 83 题, 阈值需随口径重新预注册 (不能在看到 Arm1 2765 结果后调)。

## 2026-11-19 Arm3 完成 + 主比较 (闭环)

- Arm3 `run_20261119_091235` 18:03 完成 (8h51m)。独立测试 2765 题 valid=2765 infra=0 → hard **1945/2765 = 0.7034**;
  Arm1 对照 2077/2765 = 0.7512。
- **主比较 (冻结 §4)**: both_pass 1768 / both_fail 511 / Arm1_only 309 / Arm3_only 177, 精确 McNemar p=2.26e-09,
  g=+132 (rate +0.0477, CP CI [+0.0321,+0.0628], δ=0.03≈83 题) → **达到预设实际有用增益**。
- 敏感性 (排除已核验的 107 题): n=2658, g=+132, rate +0.0497 CI [+0.0334,+0.0653] → 同判读。
  107 题在两臂的 hard 均 0 (any_pass=0 / both_pass=0), 排除与否不改变 discordant 计数。
- 异构性诊断: pass_rate≥0.5 口径 Arm1 2201 vs Arm3 2057, g=+144 CI[+0.0374,+0.0661] → 差异不是 near-miss 形态变化。
- 参考 (非主结论): Arm1 vs S₀(215) g+9 p0.32; Arm3 vs S₀(215) g−7 p0.42; Arm1 vs Arm3 300 题子集 g+16 p0.048 (探索性)。
- 解释边界: 该对比只识别"整套 scaffold 的复合贡献", 不能拆分到单个 scaffold 成分; 缺 Arm0 等预算对照 (issue #82 P0-1)。
