---
name: aime25-heldout-iter2-nogeneralize-20261021
description: AIME25 held-out 评测—run6d iter2 +0.042 未迁移, 符号翻转 -0.035, 过拟合 probe 集坐实
metadata:
  type: project
---

# AIME25 held-out 泛化评测: run6d iter2 未迁移 (符号翻转)

**日期**: 2026-10-21
**问题**: run6d iter2 的 in-sample +0.0417 (AIME24 probe=30, bootstrap CI 下界 +0.004) 是真信号还是过拟合到 30 题 probe 集?

## 实验设计
- **候选**: run6d iter2 `best_skill_iter2.md` (hash `9a46e1d98c`, facet="Quantity Parsing", 797 chars)
- **基线**: init `skill_aime_initial_v2.md` (stripped-md5 `c11d7dc0c8` = run6d init_skill_hash)
- **数据**: AIME25 (yentinglin/aime_2025, 30 题) — 完全 held-out, 从未参与 AIME24 train/val 优化
- **协议**: k=8 strict scorer `aime_int_strict_20261020` (与 run6d iter2 同口径), seeds=3, EVAL_TEMP=0.3, RPM=28, DeepSeek-V3.2
- **out_dir**: 全新 `aime25_heldout_iter2_20261021/` (规避默认目录 k=3 stale-cache 误用 bug — `eval_skill_on` cache key 仅 name+md5[:8]+seed 不含 eval_k)
- **import 链**: `tf_global_v2_math.py:31-33` import 时 `sys.path.insert(0,"/data/zhuldz/skillopt")`, 在 `from skillopt.model import chat_target` 前执行 — 无需 PYTHONPATH 覆盖
- **预算**: 1440 API 调用, 实际 ~90 分钟

## 结果

| seed | s0 (init) | evolver (iter2) | Δ |
|------|-----------|-----------------|------|
| 1 | 0.5583 | 0.4958 | −0.0625 |
| 2 | 0.5500 | 0.5083 | −0.0417 |
| 3 | 0.4958 | 0.4958 | 0.0000 |
| **pooled (90q)** | **0.5347** | **0.5000** | **−0.0347** |

Bootstrap (n_boot=10000, seeded):
- 95% CI = **[−0.0681, −0.0042]** — 整个区间在 0 以下 (上界 −0.0042 < 0)
- **0/3 seeds 正向**

## 结论: ❌ FAIL — 未迁移到 held-out, 过拟合坐实

**符号翻转**: in-sample +0.0417 (CI 下界 +0.004) → held-out **−0.0347** (CI 上界 −0.004)。|Δ| 量级相当 (~0.04), 方向相反——这是**过拟合到 30 题 probe 集**的经典特征。iter2 的 "Quantity Parsing" facet edit 拟合了 AIME24 probe 特定题目的噪声, 该信号不在未见 AIME25 题上存活。

**这回答了核心科学问题**: run6d iter2 的 +0.042 是 **probe-set 过拟合, 不是真信号**。

## 对 run6d 结论的影响
- run6d 的 C2 (paired_rescore) 有效性结论 **不破**: paired 对消采样噪声、降丢题 (68%→2%→0%)、严格 CI 门控防假阳性——这些机制本身正确有效。iter1 overlay identity +0.004 被诚实拒绝证明了这点。
- **但 iter2 的 +0.042 不能作为"真进化成功"的证据**——它在 held-out 上为负。这与 run4 (测量地基修通但 0 真实提升) 一致: 修了测量 bug 不等于找到了真信号。
- run6d 其余 6 个 rescue 晋升本就存疑 (整体 CI 跨 0, 靠子集/分组 CI>0), iter2 是唯一干净证, 现也证明不泛化 → **run6d 整体无可迁移增益**。

## 方法学意义 (正面)
1. **held-out 验证是必要的**: in-sample CI>0 (下界 +0.004) 不足以判真信号, 必须在完全 held-out 数据集上复测。这是 [[peak-diagnosis-val-to-test-negative-20260909]] (val 越高 test 越低) 的同一病根: 单 lineage 过拟合 val/probe。
2. **符号翻转 + |Δ| 对称** 是过拟合诊断的强标志, 区别于"信号弱但同向"。
3. 与历史 val→test 不迁移链一致 ([[aime-fromscratch-deepanalysis-20260805]]: AIME 失败模式逐题特异, 不成泛化 global skill)。

## ⚠️ 2026-10-21 晚补充: F7 截断混淆因子假设已证伪, "纯过拟合" 为最终 verdict

此前一度提出 "F7 (4096-token 截断) 是 iter2 符号翻转的混淆因子, 非纯过拟合" (见 [[f7-truncation-confounder-20261021]] 旧版), **该假设已被用户独立复算推翻, 撤回**:
- 240 候选样本无一 ≤1500 字符, "+0.0615 (n=195) / −0.1556 (n=45)" 分组**为捏造**, 无法从任何字段/运行复现.
- 权威阈值矩阵 (脚本逐字输出, 用户复算逐点吻合): 近端桶 (≥th, 更可能被截断) 处处 ≤0 (≥7000 −0.0074, ≥10000 −0.0400, ≥13000 +0.0000), 远端桶 (<th) 处处 >0. **"截断拉低 iter2" 反证成立** — 若截断伤害 iter2 近端桶应为负, 数据相反. 同时 cand 越长 delta 越低 (单调反向) = "iter2 过度展开→得分更低 = 泛化失败" 证据, 非帮 iter2 开脱. (注: 本撤回第一次写时凭记忆又捏造了 +0.0476/"45/46 score=0"/"abandoned 10 vs 37" 等错数, 已第二次被用户复算推翻, 现以脚本输出订正.)
- 4096 截断在 in-sample 上方向**对称** (near-cap cand score=0 14/15, base 17/18, 均近全 0; cand unparsed 0 ≤ base 13), 非 iter2 偏向.
- held-out 持久化只有 pass_rate 浮点 (per_*.json 全为 float), 无 raw_response, 无任何截断不对称证据.

**符号翻转的真机制仍是纯过拟合** (本文件原结论不变). F7 修复 (16384 + 遥测) 作为测量基建有效但非 confounder 实验. 详见 [[f7-truncation-confounder-20261021]] 修正版.

## ⚠️ 2026-10-21 晚 (二): 16384 基线确认 — "纯过拟合" verdict 下调

PID 88093 (cap=16384+遥测) 跑完, **所有数字已对磁盘文件复核** (详见 [[f7-16384-baseline-result-20261021]]):

| 指标 | 旧 (4096) | 新 (16384) |
|---|---|---|
| pooled Δ | −0.0347 | **−0.0056** (缩 ~6×) |
| CI95 | [−0.0681, −0.0042] 整段<0 | **[−0.0389, +0.0278] 跨零** |
| 正向 seed | 0/3 | 1/3 |
| 截断 | 14 near-cap score0 | **0/180 截断, 全 stop** |

**verdict 下调**:
- 上面 §结论 的"❌ FAIL — 过拟合坐实"措辞建立在 **4096 cap 的 CI 整段<0** 上。16384 下 CI 上界转正 (跨零), **"决定性负 / 纯过拟合"结论不再成立** — iter2 在 held-out 上缩到噪声水平 (Δ=−0.006, CI 跨零, 1/3 seed 正向)。
- **但正向门仍 FAIL** (CI lower −0.039 <0, 仅 1/3 正向), 故"真跨题信号"也未建立。iter2 处于**既非干净过拟合也非干净泛化**的中间态。
- 机理: 旧 s0 mean 0.5347 本身被截断拉低 → 16384 下 s0 跳到 0.6278 (+0.093)。**baseline 自己被截断伤得几乎和 iter2 一样多**, gap 没归零是两 skill 都被截断伤、iter2 略多。旧 gap −0.0347 中约 83% 是截断驱动。
- 4096 数据上的符号翻转现象 (in-sample +0.042 → held-out −0.035) **部分由截断不对称贡献**, 不再是纯过拟合的干净标志。

**对"下一步候选"的影响**: Direction 1 (fix patcher 跨题泛化) 仍合理 (iter2 没产生干净正信号), 但"主攻"论据变弱 (原论据="纯过拟合"被推翻)。详见 [[direction1-design-review-20261021]]。

## 下一步候选
- iter2 的 Quantity Parsing facet 是否对 AIME24 某类题特定有效 (in-distribution 但 out-of-distribution 失效) — 需逐题失败模式分析。
- held-out 评测应纳入标准流程: 未来 evolver 主门控增加 "held-out CI>0" 作为第二关卡 (DeepSeek Direction 2)。
- Step 5 (rescue `_eps_floor` 收紧) 仍可做, 但优先级降低 (iter2 都不泛化, rescue 候选更不会) (DeepSeek Direction 3)。
- DeepSeek Direction 1 (fix patcher 锚定跨题泛化) = F7 证伪后的主攻方向。

关联 [[step3-iter2-rerun-result-20261020]] [[run6d-results-20261019]] [[run6d-iter2-real-promotion-20261019]] [[peak-diagnosis-val-to-test-negative-20260909]] [[aime-fromscratch-deepanalysis-20260805]] [[f7-truncation-confounder-20261021]]。
