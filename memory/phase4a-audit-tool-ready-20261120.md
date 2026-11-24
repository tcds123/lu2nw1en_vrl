# Phase 4-A 审计工具就绪 — 2026-11-20 16:45 (0 API, 主 run 不暂停)

工具: `eva/phase4a_audit_0api.py` (冻结快照 `frozen_phase4a.py`, sha256 `bf7ae60a…(勘误 2026-11-23)`; 纯离线, 只读 run 产物, 不写任何实验目录之外的文件)。
(17:55 更新: 按用户 §9 复核意见加入 L3 → L3.func/L3.class 拆分 + 三层结论 `fmt_verdict` (①全体 ②题型集中 ③跨题型泛化), 合成测试 18→19 例全过;
分层 L1/L2/L3 主口径与 contribution 分母不变, 拆分仅为 L3 内部描述视图; 新增 `verdict.{tag}.md` 产物。)
产物目录: `rl/train/verl1/outputs/crossmodel_baseline/phase4a-audit-20261120/`。

## 四步协议状态

| 步 | 状态 | 证据 |
|----|------|------|
| 1. 写脚本 (固定输入, 不 API, 不改产物) | ✅ | 脚本 sha256 记录在每次产物的 `script_sha256` |
| 2. 合成数据测试 | ✅ **18/18 通过** | `synthetic_tests.json` |
| 3. 300 题接口级 dry-run | ✅ 完成, 标记 `smoke_test_nonconfirmatory` | `*.smoke_test_nonconfirmatory.{json,md}` |
| 4. 正式分析 (finalize 后 2765 + 2744 双套) | ⏳ 等 Phase 3b 主 run (19/28 chunks, 16:32) | — |

## 冻结的规则 v1 (看到任何 2765 结果之前写定)

- 分层: L1 = stdin 且有 Example 块; L2 = stdin 无 Example 块; L3 = func/class。互斥且穷尽。
  (原计划按"失败模式是否在 fail 27 出现过"分 L1/L2, 但 test 侧无失败标签、且 fail 27 的 stdin 题 11/12 都有 Example
  → 该定义无法在 test 侧执行 → 改为可观测的题面属性, 记录于 rules_v1 与本文件。)
- 失败模式分类 (只用于训练侧覆盖率表, 不反推 test): 规则顺序 whitespace → numeric_precision (相对差 ≤1e-6)
  → extra_output → need_fence → token_order (>2 token) → used_example → 多行 whitespace/token_order/need_fence
  → trailing_note → logic; `nonzero returncode=1` → runtime_error; 空 → no_failure_signal。
- fail 27 分类结果: logic 10 / runtime_error 13 / format_only 3 (6968 whitespace, 5089 numeric_precision, 1430 need_fence) / other 1 (1052)。
- 贡献 C_L = g_L/g_all, 分母 <10 或 g_all≤0 时判"无定义"。

## 合成测试覆盖 (18 例)

rescue-only 净增益与 CP 区间下界 >0; 回归-only; 双向抵消 g=0; 混合分层 (C_L1=0, C_L2=0.8, C_L3=0.2, sum g = g_all 闭合);
g_all≤0 与 |g_all|<10 → 贡献无定义不崩溃; 缺失 ID 与 invalid 不计失败且被报告; 截断剔除后重配对;
分类规则; 分层互斥穷尽。

## 300 题 dry-run 结果 (仅验证接口, **不得用于判读**)

事实层面的两个发现 (实质内容等正式 2765 分析后再下, 不得据此调分层或阈值):
- **相邻 ID 题面高度模板化**: 在已完成的 ~1800 题局部 chunk 上统计, L2 (stdin 无 Example) 净增益为 0/−6,
  而 L3 (func/class) 净增益 +59/+71 → 若在 2765 上复现, 直接指向"已知失败模式定向修复"故事。**未确认, 不作结论。**
- 数据切分形态: train = 790 题中**按 ID 排序的前 80 个**, 所以同一题族的相邻 ID 会整族进入 train/val/test,
  题目独立性比随机划分更弱 (L4 近重复只是其中一角)。

## 已备好的正式输入

- `rules_v1.json`, `near_dup_ids.json` (21 题, 全部对 train 3953), `drift_ids.json` (4770/1747/4928, 取自 drift4 的 8 题核验)。
- 正式命令 (finalize 后):
  `python3 eva/phase4a_audit_0api.py --stage formal --out $D --baseline-run <full run> --near-dup-ids $D/near_dup_ids.json --drift-ids $D/drift_ids.json --trace-dir <...>/trace`

关联: [[phase3b-baseline-run-progress-20261120]] [[data-split-representativeness-prereview-20261120]]
