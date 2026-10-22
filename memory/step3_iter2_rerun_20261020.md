---
name: step3-iter2-rerun-20261020
description: step3 用硬化评分器重跑 run#6d iter2 — 池化干净口径 +0.033/CI+0.008(聚合稳健), 但逐题仅2/30方向一致(非逐题稳健); Step4 铁证评分器bug对硬评分0 hit(纯评分器效应=0)
metadata:
  type: project
---

# step3: run#6d iter2 硬化评分器重跑 (干净口径终局)

承接 [[run6d-iter2-real-promotion-20261019]] (iter2 首个真晋升 +0.042/CI+0.004). 目的: 换硬化 `_to_int` (strict, 框非纯整数) 重跑 iter2, 检验 +0.042 是否稳健、评分器 bug 贡献几何.

**Run**: `rl/train/verl1/outputs/skill_evolver_aime_v2/step3_iter2_rerun_20261020` (PID 33843, 死)
**配置** (step3_rerun_iter2.py): `scorer_version="aime_int_strict_20261020"`, `k=8`, `N_REPS=3`, split seed=42, 持久化 `eval_paired(persist_ctx=...)` ON → 每 rep 落 `paired_eval_samples.jsonl` (含**完整 raw_response**, 非 excerpt), 30题×base/cand, parse_status=parsed/abandoned.

## 终局干净口径 (纯离线复算, 只读持久化 jsonl, 零 API)

| rep | clean mean_delta | CI_lower(2000boot) | delta>0 题数 |
|---|---|---|---|
| 1 | +0.0214 | −0.0202 | 8/30 |
| 2 | +0.0607 | +0.0256 | 11/30 |
| 3 | +0.0167 | −0.0292 | 7/30 |
| 池化 | **+0.0329** | **+0.0081** | — |

逐题跨3rep方向: **仅 2/30** 三 rep 方向全非0一致; 全非负 18/30; 至少一次为负 12/30.

## 评分器 bug 贡献 (Step4 铁证)

`step3_summary.step4_decompose`: n_samples=1475, n_abandoned=47, n_parsed=1428,
- **cand_bug_hits=0, base_bug_hits=0** (旧 `_to_int_legacy` 截断的新判会被硬化 `_to_int` 拒的原始答案总数)
- **n_agree=1428** (_to_int_legacy 与 _to_int 在这批原始 parse 上完全一致, 0 处意见分歧)
- **new_delta = old_delta = 0.0416** (新旧评分器同一批 raw_response 复算得到相同 delta)

→ **纯评分器效应 = 0.00**: 硬化 `_to_int` 在 step3 这批 (k=8, 30题, DeepSeek-V3.2) 上**一处都没改变判定** (0/1428). "部分归因评分器偏差"在总量上不成立.

## 结论 (分层, 替换"七证/完全稳健"叙事)

1. **聚合层稳健**: 硬化评分器下池化 clean +0.033, CI下界+0.008>0 → 原 run#6d +0.042 硬化后约 +0.033, 方向保持正, 整体显著. 「+0.042 会因评分器修正而消失」**不成立**.
2. **逐题不稳健**: 仅 2/30 题三 rep 方向一致, 12/30 至少一次为负 → +0.033 是 **k=8 采样噪声里聚合层凑出的信号** (正负抵消后稳定), 非逐题稳健真信号. (佐证 [[run6d-iter2-real-promotion-20261019]] 里"+0.042 真解题力增量"的表述需弱化为"聚合显著".)
3. **评分器 bug 无贡献**: 0/1428 hit → 截断 bug 在这批原始答案上不制造蒙对, 也不制造虚高. 旧 _to_int 的 truncation 在硬初始化 holder 下根本不触发 (AIME 0-999 整数 + boxed 强制, 模型几乎都写纯整数).
4. **为何 summary 的 reps/pooled 与 clean 复算不同**: `step3_summary.reps[].mean_delta` 直接取 `eval_paired` 内存返回值 (0.1958/0.0208/0.10), 该函数**把 deadline-abandoned 当 0 pass 计入** → 污染口径. **判定只信 clean 复算 (剔除 abandoned 后配对)**, 见 `step3_summary.step4_decompose` 与上述离线复算. `verdict_stable=true` (基于污染 pooled 0.1056) 无效, 判据需用 pooled clean +0.0329/CI+0.008.

## 内存口径高风险 (F0z1 相关)

- rep1 内存 +0.1958 vs clean +0.0214 (base 丢 17 → 零化虚高)
- rep2 内存 +0.0208 vs clean +0.0607 (cand 丢 9>base 7 → 零化虚低)
- abandoned 在 base/cand 不对称时, 内存 delta **方向随机** — 彻底不可用, 一律用持久化口径. 已记 [[phase2-issues-20261020]] (docs/phase2_issues.md).

## 待办

- [ ] test=106 泛化评测: 现 final best 取 iter2 clean 版 (+0.042) — 但逐题不稳健, test 泛化可能不稳. 若推进, 用 iter2 skill 在 held-out test 上单跑并 rad多seed.
- [ ] 逐题一致性弱 (2/30) 暴露: 30题 × k=8 的统计功效不足以稳定检测每题. 若要论文声称"逐题稳健", 需扩 k 或扩题库.
- [ ] 更新 run6d 相关 memory 中"+0.042 稳健"表述 → "聚合显著、逐题不稳健、评分器bug无贡献".

关联 [[run6d-iter2-real-promotion-20261019]] [[run6d-results-20261019]] [[aime-grader-truncation-bug-20261020]] [[phase2-issues-20261020]].