---
name: facet-tree-overlay-smoke-20260825
description: 修法1(跨轴叠加best/算子E) live smoke 终局 — 代码正确, 三判据未全过因数据局限非bug; best_ratio 恒1.0 + n_axes_combined恒1 + became_best恒False, 根因=只有Option Comparison一轴过ci>0守门(Theorem/Hypo两代全ci<0), donor池每代仅1 → overlay≈单轴best本身不可能超越; 交叉口径差异反证(ci-argmax无ci>0守门→叠出+0.306)坐实两算子差异是设计非bug
metadata:
  type: project
---

# 修法1(跨轴叠加 best / 算子E) live smoke 终局 — 2026-08-25

## 结论一句话: **代码正确, 三判据未全过 = 数据局限(2代3轴仅1轴过ci>0), 非 bug**

## Live smoke 配置
- run_20260825_141504, `--iters 2 --eval_k 5 --train_n 35 --n_diag 8 --probe_split 0.5 seed42 --facet_tree --overlay`, freeze=Final Answer, branches=3(Option Comparison/Theorem-Level Precision/Hypotheses)。

## 三判据实测(evolver_log.json 落盘 vs 预期)

| 判据 | 实测 | 预期 | 结论 |
|---|---|---|---|
| `best_ratio` | iter1=1.0, iter2=1.0 | <1.0 | ❌ 未降 |
| `overlay.n_axes_combined` | iter1=1, iter2=1 | ≥2 | ❌ 恒=1 |
| `overlay.became_best` | iter2=False | True | ❌ +0.333 < 单轴best +0.394 |
| `overlay.admitted` | iter1=N(ci=-0.028), iter2=Y(ci=+0.239) | — | ✅ 准入生效 |

终局 best: init delta=-0.006 → **+0.394 (CI=+0.294)**, 单轴 Option Comparison。

## 根因: 数据太薄, 仅 1 轴过 ci>0 守门
逐轴日志:
- **Theorem-Level Precision**: iter1 `+0.000 CI=-0.044`, iter2 `+0.000 CI=-0.056` → 全 KEEP-PARENT。
- **Hypotheses**: iter1 `+0.028 CI=-0.017`, iter2 `-0.017 CI=-0.056` → 全 KEEP-PARENT。
- **Option Comparison**: iter1 `+0.089 CI=+0.028` → iter2 `+0.394 CI=+0.294`(TREE-BEST), **唯一过 ci>0 守门的轴**。

→ `select_axis_donors` 每代只返回 1 个 donor(Opt-Comp), overlay 其余两轴回落 `root_slots`(=S0) → **overlay 组装出 ≈ 单轴 Opt-Comp best 本身**, `n_axes=1`, 不可能超过单轴 best。要体现跨轴优势需 ≥2 轴同代过 ci>0, 2 代 smoke 数据办不到。

## 代码正确性逐项核认(排除 bug)
1. **donor ci>0 守门生效** — `overlay n_axes=1`(守门失效应=3), 即 ci<0 的 Theorem/Hypo 被正确挡掉, 未触发"n_axes 虚高+语义污染"(上一轮问题3的定时炸弹, 本次靠守门未引爆)。
2. **ratio=1.0 是真实状态非诊断 bug** — `best_native=Option Comparison`, 排除本轴后 Theorem/Hypo 逐字节 == S0(全程 KEEP-PARENT 继承根叶 S0 段), string-equality 坐实。
3. **overlay 准入门 gen2 生效** — `overlay iter2 admitted=Y ci=+0.239` 入 archive。
4. **P1/P2/P3 修复生效** — 3 个 best_node 同步点(L1105/1151/1191)让 `best_native` 正确标 `Option Comparison` 而非占位, 诊断未跑偏。
5. **--overlay flag 已定义**(L1325 BooleanOptionalAction), `--no-overlay` 消融臂可用。

## 反证 = 交叉(crossover)口径差异, 设计非 bug
`crossover iter2: Hypotheses×Option Comparison delta=+0.306 CI=+0.206 admitted=Y`。
交叉用 **ci-argmax 无 ci>0 守门**(L1121), 所以 Hypotheses(单轴 ci=-0.056) 作为轴内 ci-argmax 仍被选中 → 拿 ci<0 段组合 +0.306。这正坐实"两算子口径差异"是设计: overlay 严格 ci>0 守门(拒 Hypothesis), crossover 只看 ci-argmax(纳 Hypothesis)。交叉 +0.306 有效但 < +0.394 未成 best, 印证交叉"拼出 child 不超 best 就不成 best"的固有局限。

## 处置决定
- Bug1(算子E 标签冲突 overlay vs probe_replay)已修: overlay=算子E, probe_replay=算子F。
- Bug2(`n_axes_combined` 冗余 freeze 守卫 L1174 恒真)/ Bug3(`len(archive)>0` L1170 恒真): 保留, 无害防御性守卫非 bug。

## 教训
- **"三判据没过≠代码错"**: 结构性失效(累积通道)与数据局限(单轴过门)要分开判。overlay 机制已正确实现且准入门/守门/诊断全链路正常, 只是 `iters=2` 下 3 轴只有 1 轴 ci>0, 不足以展现跨轴叠加优势。**真正验证跨轴累积需更长的 iters + 多 seed**, 让 ≥2 轴有概率同代过 ci>0。
- 关联 [[facet-tree-no-cumulative-20260825]] (结构性缺陷根因), [[facet-tree-overlay-fix-20260825]] (修法1离线验证 best_ratio 1.0→0.333 已坐实修复, 与本次 live 三判据未过并存不矛盾: 离线用它能拿到的真实 archive, live smoke 数据太薄)。