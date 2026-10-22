---
name: pool-admission-8dim-spike-split-20260915
description: 候选池准入尖峰关的 8 维 FAILURE_FOCI 分峰统计 — meta-option 与内容维两级不同层，须按纯题目特性重分，最终 4 内容维+1 结构维=5 峰
metadata:
  type: project
---

# 候选池准入尖峰关的维度分峰（2026-09-15）

背景：修候选池准入（[[val-to-test-negative-transfer-remedies-20260911]] 的下一步），尖峰关要按 `FAILURE_FOCI` 的 8 个失败类型维度判「∃ 某维 per_d.ci>0」。统计每题在各维的题量，判断哪些维题太少需要合并。

## 关键教训：meta-option 与内容维是两级不同层，不能混在一个关键词梯里

`classify_failure_type` 里 `_META_Q_SIG`（"strongest"/"can be proved"/"a stronger result"）会**在跑内容关键词之前**就把含 meta 信号的题早期返回吞进 `[5]`。但 meta-option 是「题干/选项结构特征」（问"最强可证陈述"，或选项含 `one of the remaining`），其余 7 维是「题目数学内容关键词」（iff/uniqueness/quantifier/regularity/...）。这两个**层级完全不同**，同一梯里 meta 优先吞并会让 [0]~[7] 内容维全被抢占。

**实测**：val(53)+train(18)=71 题，旧口径（meta 优先返回）得 `[5]=64`（90%）、[0]=3、[1]=[4]=0，其余 1~2——几乎全挤进 meta 大桶。

## 正确口径：纯题目特性分类（关键词梯优先，meta 仅全无内容关键词时回退）

用户指令「只看题目特性分类」。重分结果（首命中互斥）：

| 维 | 标签 | 题数 |
|---|---|---|
| [0] | Implication vs biconditional (iff/one-way) | **14** |
| [1] | Existence vs uniqueness | **21** |
| [2] | Quantifier order / parameter dependence (∀∃/uniform) | **17** |
| [3] | Regularity / function-space (C^k/smooth) | **5** |
| [4] | Equality case / sharp constant | **0** |
| [5] | Meta-options（无内容关键词回退桶） | **9** |
| [6] | Domain / scope restriction | **2** |
| [7] | Classification completeness | **3** |

改成 meta 回退后，[0]/[1]/[2] 从 0~3 跳到 14/21/17（原被 meta 早期返回吞并）。

## 最终分峰方案 = 5 峰

稀疏维 [4]=0、[6]=2、[7]=3、[3]=5 单维撑不起 bootstrap CI（`subset_signal` n<2 返 `-inf`）。合并为：

1. **[0] Implication/biconditional**（14 题）
2. **[1] Existence/uniqueness**（21 题）
3. **[2] Quantifier order**（17 题）
4. **[3+4+6+7 合并] Regularity ∪ Sharp ∪ Domain ∪ Classification**（5+0+2+3=10 题，稀疏簇打包）
5. **[5] Meta-option 结构特征**（9 题，独立为「选项层/题干结构」信号）

即 **4 个数学内容维 + 1 个结构维**，各维题量 ≥9，能算 CI。与另两个不限题量的几何信号（子代↔父代 git-diff 相似度 + 关注点覆盖向量）并列，构成尖峰关 + 多样性关的三类信号。

## 后续

分峰结果已记录，待用户确认后落 `admission_filter` + `--pool_filter` 进 `skill_evolver_math.py`（默认关 = 字节复现 theo_iter6 路径，先续跑 8 iter 看正优化）。

## 关联

[[val-to-test-negative-transfer-remedies-20260911]]（三味药源头，候选池准入为其下一步）／[[yaoyi-beta-binomial-lcb-launch-20260914]]（药一单味证伪，引出候选池准入）／[[three-breaks-facet-tree-20260902]]（theo_iter6 0.5542 baseline，续跑起点）