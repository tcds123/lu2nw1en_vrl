---
name: emergence-structural-null-20260831
description: 演化设计核心 novelty=§2.2b 失败模式聚类自动开 facet 分支, 已在真实 test=106 数据上实证证伪=结构性空结果. 两大候选 cluster key 均死: Keyword(87% meta-option 单桶) 与 theorem_type(91% 多type, 12 hard-fail=11个不重复combo, 跨4seed稳定无重复簇). K=3世代复现门控下永不触发. 处置: 作为论文"设计→实现→实证证伪→根因"诚实 novelty 证据, 零 API 落盘.
metadata:
  type: project
---

# emergence (§2.2b) 实证证伪: 结构性空结果 (2026-08-31)

## 触发
用户在计划分析中给出两决定: 决定1=theorem_type(归一化后)是唯一可走 cluster key; 决定2=novelty-first 分段门控(val-verify emergence 真触发, 再花 4-seed test 预算)。本文件实证检验决定1 的本体可行性, 结论=theorem_type 与 Keyword 同重地死, 但死于不同机制。

## 核实对象
- schema `facet_tree_skill_evolution_design.md` §2.2b: 失败模式语义聚类→自动开新 facet 分支, K=3 **世代**复现门控(L263)。
- 现状: `skill_evolver_math.py:1180` emergence 硬编码 OFF(stage0), 从未兑现。

## 实证判决: 两大候选 cluster key 均死
数据源: 8238b5e8 test=106 4-seed (eval_k=3) + split data theorem_type。

### Keyword 分类器 (classify_failure_type 行130): 单桶死法
- ~87-93% 失败归类 'meta-options' (Option 轴已覆盖) → 聚类永远只出一个大桶, 无"新失败的失败簇"可开分支。

### theorem_type (设计 L83/L269 名义 cluster key): 散度死法
- **test=106 中 91% 题是多 type**(96/106; type-count 分布 = {2:96, 1:10})。
- **12 个 hard-fail (mp=0 跨4seed) = 11 个不重复 type-combo**; 唯一重复簇 (Existence+Implication) **n=2**。
- 单 seed 亦散度: seed1 28 hard=21 distinct combos, seed2 28/20, seed3 30/20, seed4 28/19 → **跨 4 seed 稳定**, 非平均化伪影。
- 单 type coarsen 亦不集中: 领先单 type 仅 Biconditional 3 / Existence 3 / Universal–Existential 2 / Implication 2 → **max n=3**。
- theorem_type 本身 dirty: `unknown` 标签 + 复合类顺序歧义 (Universal–Existential vs Existential–Universal)。

## 判决
**两种死法的关键差异**: Keyword 死于"一个稳定可判的大桶"(能当真实失败模式治理但 option 已覆盖); theorem_type 死于"每题唯一型组合"(跨题聚类在门控下永不触发)。在 §2.2b 的 **K=3 世代复现门控**下, theorem_type 聚类为**结构性空结果(structural null)**——不是"尚未算到", 而是数据本体就不产生重复簇。决定1(theorem_type as cluster key)在本体上不可行, 与 Keyword 同重, 且连"该往哪个失败模式开分支"都给不出。

## 用户处置决定 (AskUserQuestion 确认)
"emergence 结构性空结果" → **写成实证证伪证据**: 作为论文里"设计意图→实现→实证证伪→根因"的诚实 novelty 主张(比"emergence 未实现"更强更诚实), 零 API。

## 可审计产物
本文件 + 下述复现命令即 auditable 证据:
```python
# 12 hard-fail -> 11 combos, 10 unique; max cluster n=2; 单type coarsen max n=3
# 详见本文件"实证判决"节, 数据在 outputs/.../multiseed_test_eval_8238b5e8.json + split data
```

## 对后续的判据顺序 (refine 建议)
分段门控判据重排为:
(a) 零成本断言: emergence 是否在 **val** 上真触发(cluster 跨 K=3 **世代**复现, 非 K=3 题) → 不触发即停, 不烧 test 预算;
(b) 仅触发才进入 4-seed test。
关键: 把 design L263 "K=3 世代" 在实现里 spin 到位, 避免把"8 题就以为什么簇"当点火把预算浪费在噪声上。

## 关联
[[has-meta-router-recompute-20260831]] (姊妹: 唯一正向信号落盘重算) / [[facet-tree-preset-vs-emergence-20260825]] (涌现=待验证主张) / [[facet-tree-capability-ceiling-20260827]] / [[expand-operator-api-step-result-20260828]] / docs/facet_tree_empirical_verdicts_20260831.md (两判决认证合成)