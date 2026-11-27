---
name: history-corrections-20261127
description: 两处历史结论修正 — ①pool_filter A/B均ON(B唯一变量n_parallel) ②+10.8pp属AIME25非LiveMath须从LiveMath目标移除; LiveMath正式历史参考=theo_iter6 55.42%
metadata:
  type: project
---

# 历史结论修正 (2026-11-27, 零API核实)

用户要求更新研究记忆, 避免再次引用错误数据. 两处历史结论错误已核实修正.

## 修正1: Run A/B 的 pool_filter = ON (非 OFF)

| 项 | 旧记录 | 最新核实 |
|----|--------|---------|
| pool_filter | OFF | **A/B 均 ON** |

**核实证据**:
- Run A launch (launch_livemath_fullrun_v32.sh L31): `--pool_filter`
- Run B launch (launch_livemath_runB_v32.sh L38): `--pool_filter`
- 两者运行时日志均: `🔧 opt-flags ON: fix-①:pool_filter (质量关+5峰尖峰关+git-diff多样性关, eps=0.02)`
- Run B Theorem 的 "POOL-FILTER 拒" 来自代码 L3152 (facet_edit 的 pool_filter 质量关, Δ<−0.02 拒)

**影响**: Run B 的唯一预期行为变更是 n_parallel 1→4, **pool_filter 不是变量** (A/B 都 ON). 用户记忆的 pool_filter=OFF 可能指历史 theo_iter6 字节复现配置, 非 Run A/B. Run B 实验设计有效.

## 修正2: +10.8pp 属 AIME25, 非 LiveMath (须从 LiveMath 目标移除)

| 项 | 旧记录 | 最新核实 |
|----|--------|---------|
| 旧 Evolver +10.8pp | LiveMath 历史增益 | **属 AIME25, 严禁用于 LiveMath 对照** |

**核实证据**:
- +10.8pp 来源: memory/aime-fresh-rerun-result-20260807 — "AIME完整重跑链终局, v2 Δ=0.00, evolver Δ=-1.67pp"
- 该数字属 **AIME25** (test=30, k=3, 4-seed, skill hash 6f0e7669), 完全不同任务/skill/run
- LiveMath 的历史最佳 = theo_iter6, **55.42%**, 非 +10.8pp 增益

**影响**: 此前设定的"性能目标 ≥ 旧 +10.8pp"对 LiveMath **无效**. +10.8pp 应从 LiveMath 目标体系移除.
- 新的 LiveMath 自身算法历史参考 = **theo_iter6 55.42%** (test=106, k=3, 4-seed)
- 新的强基线 = **SkillOpt 56.05%** (同 split, northstar 同窗口 4-seed)
- 旧 Evolver vs SkillOpt = **−0.63pp, CI 跨零** (统计无定论, 非 CI-gated defeat)

## 修正3: SkillOpt 强基线口径

| 项 | 旧记录 | 最新核实 |
|----|--------|---------|
| SkillOpt 强基线 | 54.03% (test=124, 2-1-7 split) | 另有同 split (1-3-6, test=106) 历史结果 **56.05%** |

**核实**:
- 54.03% = evalskill_skill_v0002 on test=124 (2-1-7 split, baselines/skillopt_outputs/)
- 56.05% = northstar 同窗口 4-seed test=106 (1-3-6 split, 与 Run A/B 同 split)
- **对照应使用 56.05% (同 split 同口径)**, 但需在当前 V3.2 重测 (S₀ 已变, 见 [[bottleneck3-strategy-capacity-20261127]])

## LiveMath 历史参考体系 (修正后, 2026-11-27)

| 参考 | 值 | 口径 | 用途 |
|------|-----|------|------|
| S₀ (当前 V3.2) | 36.79% | test=106, k=3, 1-3-6 split | Run A 基线 |
| S₀ (历史 9月) | 29.56% | 同上, 4-seed | northstar 基线 (S₀ 已变 +7.2pp, 待逐题核查) |
| **theo_iter6 (旧 Evolver 自身最佳)** | **55.42%** | test=106, k=3, 4-seed | 自身算法历史参考 |
| **SkillOpt v0002 (强基线)** | **56.05%** | 同 split, 4-seed (同窗口) | 强基线对照 |
| 旧 Evolver − SkillOpt | −0.63pp | CI95 跨零 | 统计无定论 |
| Run A (新机制) | 40.57% | test=106, k=3 | 当前 (Δ vs S₀ +3.77pp, CI 跨零) |
| +10.8pp | — | **AIME25, 不可用于 LiveMath** | 已移除 |

**关联**: [[bottleneck3-strategy-capacity-20261127]] (第三瓶颈 + theo_iter6 文本恢复) / [[livemath-fullrun-launched-20261126]] (Run A/B 状态) / [[northstar-test-baseline-20260908]] (原始 theo_iter6 vs skillopt)
