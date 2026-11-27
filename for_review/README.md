# for_review — Skill Self-Evolution 实验交付包（版本 2026-11-25）

## 👉 codex 请看这一段：你要审什么

**任务**：审我们的 skill 自进化框架里 **bootstrap CI（CI_lower > 0）作为候选 skill 唯一晋升门是否过严**，以及"LiveMath 新机制移植后两轮候选全被 CI 否决"这个结果该怎么解读。

**必读入口**：`review_requests/20261125_codex/REVIEW_REQUEST_20261125.md`（4 个审点 A–D 的完整清单 + 实验数据）。

**一句话问题**：候选 skill iter1 delta=+0.023 CI=-0.004、iter2 delta=+0.016/0.025 CI 跨零，**方向两轮一致但单轮 CI 都跨零被拒**——这是"无真实效应"还是"CI 在小样本上系统性误杀真信号"？

| 审点 | 审什么 | 我们当前判断（供反驳） | 证据在哪 |
|------|--------|----------------------|----------|
| A（核心）| CI_lower>0 唯一晋升门是否过严 | 偏严：CI 防误晋升目的对，但 n=10 时 CI>0 需 delta≈0.04+，真实 +0.02 注定跨零 | `outputs/livemath_smoke_r2/archive_iter1.json` + log |
| B | 候选生成器（单 slot 编辑）改动力度是否太小 | 是另一层根因：facet_edit 局部微调 delta 天花板低 | `code/skill_evolver_math.py` facet_edit 逻辑 |
| C | paired_rescore 移植正确性（5-tuple 解包） | 已自测通过（MCQ 后端不崩）| `code/tf_global_v2_math.py` eval_paired + `code/skill_evolver_math.py` |
| D | smoke(2轮+train_n=18) 够不够下结论 | 不够下终论，但够看机制跑通 + 信号方向 | `outputs/livemath_smoke_r2/evolver_log.json` |

**交付形式**：`.md` 评审报告放 `incoming/20261125_codex/REVIEW.md`；每条意见标**证据等级**（事实/推断/假说）+**是否改变论文结论**+**最小验证实验**。不要重复 `docs/phase2_issues_202611.md` 已知问题；`archive/` 历史包不必逐份审。

---

## 这次实验是什么

把 AIME 上验证有效的 6 个新机制（paired_rescore / rescue_on_reject / facet_group_admit / EFL / _veto_regression / F6 cache manifest）移植到 **LiveMathBench**（MCQ 数学选择题），用 **DeepSeek-V3.2** 重测，看能否松动旧饱和结论。**只移测量/门控，不碰 LiveMath MCQ taxonomy**——"LiveMath taxonomy + AIME 测量地基"是此前唯一未测的叉乘。

```
evolver 产出 skill → skill 作 system prompt → V3.2 推理 → paired 评测(pass+CI) → CI>0 才晋升 → 下一轮
```

**两轮结果**：6 个真实候选 patch，全部 delta 方向有正有负、CI 跨零，0 晋升。best_skill 保持 S₀ 不变。

---

## 从哪里开始读（reviewer 版）

1. **本 README**（你在这）
2. `docs/math_line_background_20261126.md` — **数学线完整背景与历史**（AIME × LiveMath 两条线的时间线、饱和机制、命运对照，先读建立全貌）
3. `review_requests/20261125_codex/REVIEW_REQUEST_20261125.md` — 4 审点 + 实验数据表
3. `outputs/livemath_smoke_r2/` — 原始实验产物（archive_iter1.json / evolver_log.json / log）
4. `code/skill_evolver_math.py` 的 `subset_promote_decision`（L514）— 门控逻辑核心，审点 A 的代码
5. `memory/livemath-newmech-port-exec-20261124.md` — 移植 delta 记录（审点 C）
6. `docs/phase2_issues_202611.md` — 已知问题（别重复）

---

## 包结构

```
for_review/
├── README.md                                  ← 本文件
├── review_requests/20261125_codex/            ← codex 入口（4 审点）
├── code/                                      ← 最新源码
│   ├── tf_global_v2_math.py                   ← LiveMath 评测后端（含 eval_paired）
│   ├── skill_evolver_math.py                  ← LiveMath evolver（含门控逻辑）
│   ├── tf_global_v2_aime.py                   ← AIME 后端（对照）
│   ├── skill_evolver_aime_v2.py               ← AIME evolver（移植源）
│   ├── _env.sh                                ← V3.2 后端配置（key 走 envvar，已脱敏）
│   ├── _launch_smoke_v32.sh                   ← smoke 启动脚本
│   └── _ping_v32.py                           ← API 连通性验证
├── outputs/livemath_smoke_r2/                 ← smoke 实验产物
│   ├── archive_iter1.json                     ← iter1 全候选 delta/ci
│   ├── evolver_log.json                       ← iter1 决策日志（operator_counts/promoted）
│   ├── s0_cache.json                          ← S₀ 重算结果（含 manifest）
│   └── livemath_newmech_smoke_v32_r2.log      ← 完整运行日志
├── memory/                                    ← 移植/重评估记录
├── docs/                                      ← 设计文档
└── archive/2026-11-25_livemath_newmech_pre/   ← 旧 11-13 版主体（归档，不必逐份审）
```

---

## 已知边界（别让我们重复踩）

- **捷径不消除**：新机制不删 LiveMath 的 meta-option 捷径（"a stronger result can be proven" = 100% 正确），只换测量/门控。iter1/iter2 的失败类型 100% 是 `Meta-options:8`。
- **smoke 不是终论**：2 轮 + train_n=18 只够看机制跑通 + 信号方向，不够下"能否提升"终论。
- **paired_rescore 已验证不崩**：6 机制中 5 个端到端跑通，rescue 因 iter1 无子集 CI>0 正确不触发。

---

> 历史版本归档在 `archive/`（按 `<日期>_<主题>/`）。对方回包放 `incoming/<YYYYMMDD>_<来源>/`。
> 本版替换 2026-11-13 版（旧主体快照在 `archive/2026-11-25_livemath_newmech_pre/`）。
