# 1′-fixed rep5 重跑 — 结果 (2026-10-29 启动 / 2026-10-30 04:28 结束)

预登记: [[dir1-rep5-fixed-launch-20261029]] (判据写死于启动前, 本篇不改判据, 只报执行)。
run dir: `rl/train/verl1/outputs/skill_evolver_aime_v2/run_dir1_step1_onflags/rep5_fixed_20261029/run_20261029_182501/`
PID 69281 正常退出, 9h27m, 16 次算子尝试 (facet_edit 15 / overlay 5 / expand 5) **0 晋升**,
best_skill hash `c11d7dc0c8` → `c11d7dc0c8` (599 chars, 与 init 逐字节相同)。

## 三条判据的判定

| 判据 | 判定 | 依据 |
|---|---|---|
| ① iter1 污染被堵 | ✅ **通过 (中性路径)** | iter1 三 facet + expand **零晋升**; 且 iter2-5 也全部零晋升。rep5 唯一 PROMOTE (`12064805af` +0.1042, base-low 13/30 计 0) 未复现 |
| ② 干净信号 ≥ +0.10 且 CI>0 | ❌ **不通过** | 20 个候选最高 in-sample delta = **+0.044** (`063b0a75a3`, it5 expand), 其 boot CI [+0.007,+0.083]。全程无一候选 delta ≥ +0.10 |
| ③ identity 空白 | ⚠️ **本 run 无法执行 (无观测)** | overlay **5/5 次被 `OVERLAY-SKIP` 跳过**, 未产出 identity 候选; 见下文"偏离" |

按预登记, ② 不通过 = **"干净信号太弱"从 rep5 的观察升级为已定量结论 → Direction 1 实验侧结案, 转 generation 层**。

## 为什么 ③ 没有数据 (不是遗漏, 是修复生效)

`skill_evolver_aime_v2.py:3976-3984` 的 identity-overlay 修复 (2026-10-28, 问题 #11) 规定:
donors 空且 best_node 无 carry-forward 段时 overlay 重建的 skill 与 parent 逐字节相同 → **直接 skip**, 不发起评测。
本 run 5 轮 facet_edit 全部未过 `ci>0` → 每轮都是 donors 空 → 5/5 skip (`operator_counts.overlay.guard_rejected=5, identity_skip=5`)。
**launch 记录里"overlay 每轮产出 identity 候选"的前提在修复后已不成立**, 这是记录侧的事实错误, 不是 run 的失败。
预登记的降级方案 (用 #15 的 sd≈0.059 当尺子) **属于事后替换, 未采用**; 改用下面的独立 placebo 实验回答同一个问题。

## 替代 ③ 的证据: placebo (交叉配对) — 本次最有价值的方法结论

把每个候选 **与"其它迭代"的 base 配对** (候选本身不动, 只换基线的时段), 得到 80 组 "真候选 × 假配对":

- placebo meanΔ: median +0.002, 95% 区间 **[−0.033, +0.046]**, max **+0.048**
- 真 delta 的最高值 **+0.044** → **低于 placebo 的最高值**
- sign-flip 置换检验 (20 候选, 5000 次, 统计量 = max Δ): **p = 0.21**; 真 delta 全 20 个的均值 +0.0065

**结论: 整个 run 的最好候选与"换一段基线"同样好。这不是"信号弱", 是信号与时间漂移不可区分。**

## 时间漂移的量化 (补充 #15)

同一个 init skill、同一批 30 题、k=8, 在 5 轮里被重复测了 5 次 base 臂:

- 题级基线通过率跨轮 sd: median **0.036**, 最大单题 **0.123** (`aime24:65`: 0.62/0.28/0.34/0.50/0.50)
- 但**整池均值** 5 轮 = [0.617, 0.595, 0.593, 0.597, 0.590] → 只有 **−0.027** 的缓慢漂移
- 因此 **跨时段配对 δ 的 sd ≈ 0.06–0.10 (均值差)"但不会系统抬高 Δ"**; 各轮 Δ 的均值 = +0.006 (n=20), 与 0 不可区分 → **逐轮 Δ 是零均值噪声, 不是正偏置**
- 这与 #15 的 identity sd≈0.0587 同量级, **独立复现了 #15 的真实 SE**, 且来源是 base 臂自身, 不是 overlay

→ **1a 的 +0.05 检出下限不是"稍紧", 而是低于噪声地板; #15 的 ~+0.10 才是下限。本 run 全部候选在噪声地板之下。**

## 全部候选 (in-sample, 按 delta 降序)

| delta | CI (boot) | iter | op | hash |
|---|---|---|---|---|
| **+0.044** | [+0.007,+0.083] | 5 | expand | `063b0a75a3` |
| +0.042 | [−0.009,+0.088] | 2 | expand | `ed475cca3b` |
| +0.036 | [−0.010,+0.082] | 4 | facet | `d4c57e0e0d` |
| +0.024 | [−0.015,+0.067] | 3 | facet | `d5446526fa` |
| +0.021 / +0.021 / +0.020 / +0.016 / +0.008 / +0.003 / +0.000 / −0.001 / −0.005 / −0.005 / −0.006 / −0.006 / −0.020 / −0.023 / −0.025 | 全部含 0 | 1-5 | facet/expand | (见 agg 脚本) |

唯一 CI>0 的是 `063b0a75a3` (+0.044)。**注意: 这是 20 选 1 的 max, 未做多重检验校正; 按 placebo/置换 p=0.21 判, 不可当真实效应。**

## 附属发现

1. **iter2 facet QP: base 臂 13/30 超时被剔除 → 只剩 17 对** (`delta=−0.051 CI=−0.154`)。剔除方向正确 (不计 0), 但 43% 缺失下该分支不可用。
2. **iter1/iter2 expand 各有 7 对因 cand 侧超时被剔除**; iter1 的 expand delta 在补全后从 +0.025 变 +0.004 (排序翻转) — 部分采样时的 Δ 不稳定, 不要中途读。
3. `eval_audit.jsonl` **只记 main 侧 `eval_on` 的 8 条 abandoned**, `eval_paired` 里 drop 的 pair (iter2 的 13 + 两次 expand 的 7) **没有进审计文件**, 只有日志行。判据①依赖的 "audit 可核验" 因此只对 main 路径成立 → 记 **问题 #17**。
4. 吞吐: 实测 ~1h/iter (理论 ~4 item/min @28RPM 的 1/4), 主要是长补全 + 重试; 本 run 未改任何变量。
   ⚠️ **cap 口径事实修正 (2026-10-30, issue #18)**: launch 脚本 `export QWEN_CHAT_MAX_TOKENS=16384` 是对的, 但 **AIME evolver in-sample 评测全程实际跑在 4096** —— `tf_global_v2_aime.py:run_one` L119 硬编码 `max_completion_tokens=4096`, env 的 16384 被显式实参覆盖, F7 修复(在 `tf_global_v2_math.py:47` math 通用层)因 AIME evolver 导入 `tf_global_v2_aime` 而根本没加载。最长 response=16501 chars 卡在 4096 硬上限签名, 尾部被从句子切断。**但对本 run +0.044 结论无影响**: cand/base 长度无系统差异(7657/7666), 长样本 score=0 双向(116/146), 截断在 delta 里对称抵消 → +0.044 非截断假信号。详见 `docs/phase2_issues.md` #18 + [[f7-run-one-hardcoded-4096-20261030]]。

## 结论 / 下一步

- **Direction 1 (evolver 侧 in-sample 选择) 结案**: 1′ 修好后污染消失 (① ✅), 但干净信号自身上限 +0.044, 且在 placebo 噪声带内 → 无法过任何合理检出下限。
- **转 generation 层** (与 rep5 / B4 的结论一致): 问题在"候选质量"而非"选谁"。B4 已否证 cross_q_consistency gate; 不应再加 promote gate 类选择规则。
- **不再需要**为本次跑 AIME25 held-out (② 不通过, 按预登记跳过, 省 ~2h)。
- **方法学沉淀**: "交叉时段 placebo" (候选 × 其它轮的 base) 是比 identity 候选更易得的噪声尺子 —— identity 依赖算子恰好产出无改动候选, placebo 任何 run 都能算。建议今后默认报告。

## 关联

- [[dir1-rep5-fixed-launch-20261029]] — 预登记判据
- [[dir1-step1-rep5-onflag-launch-20261023]] — rep5 原记录 (污染源)
- [[measurement-calibration-1a-20261028]] — 1a 口径
- [[b4-result-20261029]] — B 侧同步结论 (转 generation 层)
- `docs/phase2_issues.md` #13 (1′ 修复) / #15 (SE 低估) / #16 (启动崩溃) / **#17 (eval_paired drop 不入审计)**
- 脚本: `rl/train/verl1/skill_evolution/...` 无新增; 分析脚本在 `/tmp/agg_rep5fixed.py`, `/tmp/placebo.py`, `/tmp/perm.py` (未入库, 如需复现可按本文口径重写)
