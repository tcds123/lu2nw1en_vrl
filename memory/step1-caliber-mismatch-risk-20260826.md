---
name: step1-caliber-mismatch-risk-20260826
description: Step1 量尺风险: 本次启动 flag 关(--probe_split 未传)→eval 落 71题 select 口径; 但判据线0.5464+种子0.5220是36题probe标的→跨口径, 0.520≈0.5220是巧合非同尺对照. 9.6h后判Step1是否破线必须同口径重标(在71题select上重标判据线或迁回36题probe), 禁直接拿71题delta比0.5464
metadata:
  type: project
---

# Step1 口径错位风险 (2026-08-26, 用户核 T+1h15m 巡检时钉死)

## 问题 (必须 9.6h 后解读 Step1 结果前解决)
本次 Step1 启动**未传 `--probe_split`** (flag 关) → `skill_evolver_math.py:1527 probe_items = select_items` = **71 题 select (train35+val36) 全量**. eval_on (line 1159/1241/1283) 全在 **71 题 select 口径**上算 delta/CI.

但外部判据标尺全部是 **36 题 probe 口径**:
- 判据线 **0.5464** (k=10) — a2 power analysis 在 36 题 probe 上标定 ([[a2-power-analysis-20260824]]).
- 种子 1dbbb821 历史 **0.5220** — Phase A 那套 36 题 probe 上测.
- Phase A 全部 8 组合 (Option/Theo/Hypo/combo) delta/CI — 36 题 probe.

→ 本次 iter 1/2 的 delta (vs s0, 71题内自洽) 拿来直接比 0.5464 判「破线」= **跨口径比较**, 无效.

## 「吻合」是巧合非同尺对照
init 锚定日志 `📊 init skill select: pass=0.520 delta=+0.294 CI_lower=+0.218` — 这是 **71 题 select** 上的数. 种子历史 0.5220 是 **36 题 probe** 上的数. 0.520 ≈ 0.5220 数值接近是**巧合级吻合**, 不是「同一量尺的对照」. 别用「吻合」措辞误导成可直接比.

## 代码里 0.5464 无硬编码 gating (确认 evolver 不会跑错)
`skill_evolver_math.py` 全文搜 0.5464/5464 = **零命中**. promote/survivor/winner 逻辑 (line 1187-1197, 1769-1777) 全是**相对判据**: vs 父叶 leaf.val_delta 提升 + CI>0 + novel(hash≠current). 不碰绝对 0.5464. → evolver 自洽地在 71 题 select 的 delta/CI 体系里做相对提升, 不因跨口径跑错或卡 gating. **0.5464 纯是我们事后判读「是否真突破」的外部标尺**, 而它跨口径.

## 注释「36题」遗留误导 (line 1374/1204, 别下次又踩)
- line 1374 注释 `eval 调用 ×N (每候选 36题×k)` / line 1204 注释 `probe 上 delta 最高且 CI>0 者` — 写「36 题」是 **probe_split 开时**的口径描述.
- 本次 flag 关 → probe_items=select_items=71 题 (line 1527 为准). 注释里的「36 题」与本次运行**不符**, 是遗留措辞. **以 line 1527 代码行为为准, 不以注释为准**.

## 日志路径 (别下次又找错地方, 用户钉)
- stdout → `outputs/skill_evolver_math/step1_launch.log` (**不在** run dir 内, 持续写, 但 stdout 重定向=block-buffered, 可能阶段性 0 字节后突发 flush).
- run dir `outputs/skill_evolver_math/run_20260826_162059/` — **iter 1 eval 完成落盘才有内容** (archive_final.json/best_skill_evolved.md). 现在 (T+1h15m) 仍空属正常.
- 判进度看 step1_launch.log (有 init 锚定 + Iter N/2 + facet diag 行); 判 iter 完成看 run dir 落盘. **两者是两回事**, 别只看 run dir 空=误判卡死.

## 9.6h 后解读 Step1 结果的动作 (同口径重标, 禁直接拿 71 题 delta 比 0.5464)
两选一, **判完前必须先定**:
1. 在 71 题 select 上**重标判据线** (重跑 a2 power analysis 的口径换算, 或直接在 select 上重算 s0 + Option 轴 delta 定新线), 再比.
2. 或把 Step1 最终 best skill **迁回 36 题 probe** (Phase A 同口径) 重测 delta/CI, 与 0.5464 比.
- 无论哪条, 解读时守 [[ci-band-not-point-est-20260826]]: 看 CI 带非单点 delta, 破线需 delta 提升且 CI>0, 非单次跑过即定论.
- 跨口径的「pass 0.5XX」数字直接比判据线 = 无效结论, 禁下.

## 关联
[[step1-launch-20260826]] (本次启动参数, 未传 --probe_split) / [[a2-power-analysis-20260824]] (判据线 0.5464 来源, 36题probe) / [[phase-a-axis-combo-result-20260826]] (种子0.5220 + Phase A 全在36题probe) / [[ci-band-not-point-est-20260826]] (解读纪律) / [[step1-nparallel-impl-20260826]] (Step1实现)
