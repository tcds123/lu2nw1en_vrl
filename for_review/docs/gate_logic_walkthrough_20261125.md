# 门控逻辑代码导读（审点 A 用）

> codex 审 CI 是否过严时，按此文件定位代码。

## 候选晋升的三道门（skill_evolver_math.py）

候选 skill 要成为 best，要过三关。**当前 iter1/iter2 的候选全栽在第二关（CI>0）。**

### 门 1：整体 delta > best_val_delta（主路径）
paired_rescore ON 时 `best_val_delta` 重置为 0.0（零假设 cand≤parent）。候选 delta>0 即过这关。
- iter1 候选 3（Option Comparison）delta=+0.023 > 0 → **过了门 1**
- iter1 候选 4/5 delta<0 → 没过门 1

### 门 2：CI_lower > 0（subset_promote_decision, L514）
```python
promote = (nm["ci"] > 0) and (nm["n"] >= 1) and (not _veto_regression(mt, eps, upper_veto, skip_n1))
```
- `nm` = not-meta 子集信号（非捷径题），`mt` = meta 子集信号（捷径题）
- 主路径：not-meta 子集 **ci_lower > 0** 且无子集显著退步才晋升
- 镜像路径：meta 子集 ci>0 且 not-meta 未显著退步
- **iter1 候选 3：过门 1 但 not-meta CI<0 → 门 2 拒（KEEP-PARENT NO-CI）**

### 门 3：rescue_on_reject（_combined_rescue, L634）
门 2 拒了之后，rescue 看 subset 是否有 ci>0 的子集可救。
- iter1：not_meta_ci=-0.037（全负）→ 无可救子集 → rescue 正确不触发

## 关键数字（审点 A2 用）

- `not_meta_n`：not-meta 子集样本数。iter1 expand 候选 not_meta_n=**10**
- paired 评测的 σ_pair ≈ **0.044**（AIME 上测得，LiveMath 移植后未单独测）
- n=10, σ=0.044 时，bootstrap CI_lower>0 需要 mean_delta ≈ 0.04+（粗估 2×SE）
- **所以 +0.023 这种信号在 n=10 上注定 CI 跨零——这是统计功效问题**

## CI 怎么算的（bootstrap_ci_lower）

`code/tf_global_v2_math.py` 的 `bootstrap_ci_lower(per_item_delta)`：1000 次重采样，取 2.5 分位。
per_item_delta 是每题的 delta 序列（cand_pass - base_pass，paired 同批），长度 = 题数。

## 候选生成长什么样（审点 B 用）

facet_edit：只改一个 slot（如 "Option Comparison" 段），其余 slot 不动。LLM 拿到 parent skill + 该 facet 的失败题，产出该 slot 的新文本。
- iter1 候选 3 的 Option Comparison slot 加了："verify that the explicit alternatives indeed fall short of the theorem's full strength..."
- 单 slot 微调 → 改动力度先天小 → delta 天花板低

expand：新增一个 slot 段（如 "Quantifier Order & Parameter Dependence"）。
- iter1 expand 候选 not_meta_ci=**0.0**（正好踩线，>0 才救）

## 代码定位速查

| 机制 | 文件 | 行号 |
|------|------|------|
| EVAL_DEADLINE_S | skill_evolver_math.py | L71 (5400, 本次从 900 改) |
| eval_paired | tf_global_v2_math.py | L510 |
| subset_promote_decision | skill_evolver_math.py | L514 |
| _combined_rescue | skill_evolver_math.py | L634 |
| group_promote_decision (N-way) | skill_evolver_math.py | L613 |
| best_val_delta 重置 (paired) | skill_evolver_math.py | L2909 |
| 3 晋升点切 eval_paired | skill_evolver_math.py | L3037/3384/3539 |
| bootstrap_ci_lower | tf_global_v2_math.py | (见函数) |
| MCQ taxonomy (不动) | skill_evolver_math.py | L74-299 |
