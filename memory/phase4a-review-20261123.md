# Phase 4-A 正式审阅 — 2026-11-23 (0 API, 只读复核)

审阅对象: `rl/train/verl1/outputs/crossmodel_baseline/phase4a-audit-20261120/` (verdict.formal.md, 8 个 paired_*.formal.*,
coverage_matrix.formal.json, verdict.md, frozen_phase4a.py)。完整审阅结论见本会话回复; 本文件记录持久化要点。

## 审阅建议: **需要补充审计** (工序性, 非结果修订)

① 整体增益、③ 不支持跨题型泛化、② 定向修复 → 改为 **"特定接口形式的定向修复"**: 三者均**不推翻**,
但三处措辞必须收窄, 并且需要 1 项 0 API 补充分析 (class 层样例定性)。

## 证据链核对结果 (全部通过)

| 检查 | 结果 |
|---|---|
| 脚本冻结 | `frozen_phase4a.py` = `eva/phase4a_audit_0api.py`, sha256 `bf7ae60a…`, 与 coverage_matrix.script_sha256 一致 (先前记录 e756abf9 为误记, 已勘误, issue #103) |
| 同一评测器 | 三臂均 `_evaluate_d5` (optimizer sha `9954199332b8…`, mtime 11-17 → 早于全部三臂) |
| 输入一致 | arm1 输入 sha `49804ada…` = phase3b protocol `arm1_independent`; test_items `a3e1509b…` 两处一致 |
| 配对集 | 三臂各 2765 valid, 0 invalid / 0 missing / 0 extra, closure_errors [] |
| agent_fail | 0 (三臂), 没有被当作失败计入 |
| 区间方法 | `paired_net_gain_ci` = CP 精确区间条件于不一致对数 d 再仿射; d(s0)=476, d(skillopt)=477 = a_only+b_only, 可由配对表复原 |
| 贡献闭合 | 8 个文件中 sum(g_L)==g_all 均成立 |

## 需在论文/报告补写的限定 (本次复核新增)

1. **class 增益 = NameError→pass 接口转换** (issue #104): 88/88、105/105 rescue 全部为 `NameError: name 'Solution'`;
   class 层绝对通过率仅 0.51。需样例 diff 定性分析, 区分"包装修好 / 逻辑修好"。
2. **L1/L2 非"无增益", 而是双向抵消** (issue #105): 条件于 S₀ 失败, mismatch 29.8%、nonzero_rc 43.8% 被救回,
   但 L1 整体 a_only 119 / b_only 141 → 净 −22。
3. **分层口径替换** (issue #100): 计划 §3.1 的失败模式分层未执行, 现 L1/L2/L3 为可观测题型层 (exploratory)。
4. **drift 8→3** (issue #101): 规则应为"drift 且成对不一致即剔除", 执行后由规则导出 3 题。
5. **去重 21 题全在 L3** (issue #102): 去重敏感性对 L1/L2 无检验力。
6. **全体 CI 不能由分层 CI 相加得到** (各层各算自己的 d), 报告不得写"sum-derived"。

## 结论边界 (与用户 §5 表一致, 已核对无越界)

- ✅ vs S₀ 整体有统计证据; ✅ vs SkillOpt 整体有统计证据; ❌ 跨题型泛化不支持;
- 🟡 class Solution 定向修复 = 较强初步证据 (但限定为接口形式, 见上 1);
- ❓ 具体进化机制 — 未识别; Arm1 vs 静态基线 ≠ Evolver 机制优于等预算无反馈方法; Arm0 仍是独立因果识别问题。

## 状态
Phase 4-B (#11) 保持阻塞, 等用户明确意见 (通过 / 需补充审计 / 需修订研究结论)。

关联: [[phase4a-verdict-20261120]] [[phase3b-baseline-run-progress-20261120]] [[phase4a-audit-tool-ready-20261120]]

---

# 补充审计 #104 完成记录 (2026-11-23, 0 API)

交付物: `phase4a-audit-20261120/audit_104_class_diff.{py,json,md}` (只用存档 response 离线重跑 harness,
无模型调用; 仅 7403/2798 两道有 S₀ response 的题做了离线重判, 均 Timeout)。

## 抽样规则 (预先固定, 非挑成功样例)

test 260 道 class 题按 id 排序分 5 组, 组内 ceil 不等距抽取 round(j*(m-1)/(n-1)), 共 18 道, 去重已核验:

| 组 | 定义 | n | 抽中 |
|---|---|---|---|
| R_O | Arm1 pass, S₀ fail, SkillOpt fail | 81 | 19,1570,2815,3703,5951,7403 |
| R_K | Arm1 pass, S₀ pass, SkillOpt fail (仅 SkillOpt-only rescue) | 24 | 63,3152,7323 |
| R_S | Arm1 pass, S₀ fail, SkillOpt pass (仅 S₀-only rescue) | 7 | 172,2798,6424 |
| R_N | 三臂全 fail | 127 | 50,3645,7376 |
| R_B | 基线 pass, Arm1 fail (回归) | 21 | 51,3014,6730 |

## 重叠口径 (新, 必须进报告)

88 = vs-S₀ rescue = 81 (双基线均 fail) + 7 (S₀-only); 105 = 81 + 24 (SkillOpt-only);
交集 81, 并集 112, Arm1 class 总通过 132 (= 112 + 20 三臂皆 pass)。**"+88 与 +105 不可相加**。
两臂唯一 gen 不同的差异集 (7 题: 2798/3812/5090/5149/6508/863/933) 意味着 S₀-only rescue 部分反映生成噪声,
24 道 SkillOpt-only rescue 也**不是**"修好了 SkillOpt 的错误", 而是 SkillOpt 自身缺 Solution 入口。

## 代码 diff 定性结论

- 18/18 样例: 基线失败原因 (或 Arm1 成功后其基线对应) 全为 `NameError: name 'Solution'` (或 pass / 1 例 Arm1 no-parseable);
  **没有一例是算法错误**。
- 16 道无基线 response 存档: 只能读 evaluator 的 stderr 尾段, 无法排除 Arm1 同时改了算法体 → 归因 **"无法判断"**。
- 7403: 有 S₀ response, 离线重判 fail=NameError(main), 用"仅补 Solution 包装" proxy 重判 → 通过 = **接口修复 (proxy 证据)**。
- 2798: 有 S₀ response, 重判 Timeout → **无法判断** (初版因误用 best 缓存判成 pass, 已修正, issue #106)。
- class 层失败桩点: `sol = Solution()` 在 fn-call driver 第一步, 算法体是否错在"修好接口"前根本不可观测。

**可支持的最强措辞**: "Arm1 (Evolver) 使基线模型在 class Solution 题型上产出可被 harness 执行并输出正确的代码,
其中至少 81 道为双基线均因缺 `Solution` 入口而失败的题。" 「算法逻辑未变」**不能**用现有材料断言,
需 dump Arm1 response 做 diff → 归入 Phase 4-B Direction A 的材料。

## 分布与天花板

- S₀ class 失败 215: NameError(Solution) 209 (97.2%) / mismatch 4 / RuntimeError 2;
  SkillOpt 232: 225/4/3; Arm1 失败 128: NameError 107 / mismatch 11 / RuntimeError 8 / 无法解析 2。
- train 80 道中 class 题只有 8 道 (1430,3516,5395,5449,5987,6911,7205,7379); tf_run_log 只记计数不记 id,
  train_used=27 中 class 题数离线不可复原, 上界 = 8 → 反馈信号"极稀疏"可量化表述。

## 结论

#104 完成。Phase 4-A 主要统计结论不变 (①–⑤), 措辞按上收紧; 建议据此关闭 Phase 4-A, Phase 4-B (#11) 仍阻塞,
等用户决定 Direction A (接口修复稳定性, 需 Arm1 response 落盘) / Direction B (其他题型) 是否分别预注册。

关联: [[phase4a-verdict-20261120]] [[phase3b-baseline-run-progress-20261120]]  issues #106 #107

## 补记 (2026-11-23 晚): 81 题逐题 stderr 核查 + 用户裁定

- **112 道 rescue 全部有逐题记录** (不是抽样): R_O 81 / R_K 24 / R_S 7 中, 被救基线各自的 fail_reason 在
  `all_results.json` 中 100% 非空, 且 all = `NameError: name 'Solution'` 挂在 `_d5_pair_0.py` line 22
  `sol = Solution(); got = sol.<fn>(...)`。逐题清单: `phase4a-audit-20261120/rO_81_baseline_failclasses.json`。
  仍须限定的点: 该 stderr 是 **末次异常** 的 -200 字符尾部, 证明"执行停在入口", 不能证明"算法体本身正确"。
- **用户裁定 (2026-11-23)**: 正式关闭 Phase 4-A; Direction A / Direction B 拆为两独立研究问题分别预注册,
  **优先 Direction A**; 80/160/320 规模扫描**不再是**默认设计; #11 保持阻塞, 不随 Phase 4-A 关闭自动启动。
- 最终结论双层次: 统计层 (vs S₀/vs SkillOpt 配对显著, 敏感性同向) + 机制层 (收益集中于 class 接口缺失修复,
  算法主体是否改善未识别) —— 两层不矛盾, 不得写成"Evolver 改善了代码生成能力"。
- 下一步: `docs/direction_a_preregistration_draft_20261123.md` (草稿, 待用户定稿后才冻结)。

关联: issues #104–#107; [[phase4a-verdict-20261120]]
