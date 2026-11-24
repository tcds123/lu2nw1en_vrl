---
name: trunc-self-evolve-increments-20261118
description: 防截断自进化两增量落地 — budget facet EFL-ON 覆盖签名 + budget focus 块注入残文行为签名; 零 API + run2 trace 回放验证
metadata:
  type: project
---

# 防截断自进化两增量落地 (2026-11-18)

承接 issue #87 (诊断文档核查: 4 断裂点中 2 处已修/半修, 收敛到 2 真增量) + 用户核心要求:
**不能把人工经验预埋进 baseline (trunc_bank.py 旁路 / trunc_v1 手写 6 条), 必须设计奖励信号让自进化 agent 自主长出防截断 skill 段。**

## 背景: 正确载体已存在, 缺两处接线

EFL (facet_library.FacetLibrary) 已内置 `_assign_budget` (facet_library.py:597): 截断失败 `finish_reason=length` 从数据中 EMERGE 成 `is_budget=True` facet, 哨兵签名恒 UNCOVERED → 自动被 focus 选中 → patcher 自主写收尾纪律。机制骨架对, 但两处断裂:

1. **EFL-ON 路径 budget facet 永 UNCOVERED**: `get_signature` 对 budget facet 返回哨兵 → 写了纪律也判 COVERED → 每轮重选 budget 死循环 (EFL-OFF 平坦路径已在 issue #76 修好, EFL-ON 未修)。
2. **budget focus 块只喂抽象截断计数**: `_implicit_axis_block` 只渲染 "n_truncated/n_samples", 不喂 `render_behavior` → patcher 看不到残文病态 (死循环/reconsider 暴增/没写 `<answer>`), 写不出针对性纪律。

## 两增量 (均零 API)

| 增量 | 文件 | 改动 |
|------|------|------|
| A | `facet_library.py:476-495` | `get_signature` 对 `is_budget` facet 返回 `_BUDGET_COVERAGE_SIG` (commit discipline 等 6 词), import-error-safe 回退哨兵; 非预算行为 facet 仍哨兵 |
| B | `skill_evolver_aime_v2.py:1684-1698` | `_implicit_axis_block` 每条失败注入 `render_behavior(f["behavior"])`, 仅 behavior 非空 dict 时, key error-safe |

## 验证 (全零 API)

- 两源文件 `py_compile` ✅
- `facet_library.self_test()` 8 项全过 ✅; `_self_test_facet()` 全过 ✅
- 增量 A: budget facet `get_signature` → 6 词覆盖率签名 ✅; 非预算行为 facet → 哨兵不变 (零回归) ✅
- 增量 B: `_implicit_axis_block` 输出含 "NEVER EMITTED <answer>" + "tail repetition 8x" ✅
- **run2 trace 回放 (决定性)**: 12/12 budget patch prompt 含 `Trace behavior:` 行 (18 条, 6 种非重复病态), 从抽象 "3/3 samples hit the token limit" 变为 "trace_len=30550 chars, max_step=50, NEVER EMITTED <answer>" 等具体残文病态 ✅

## 结论: 设计原则 (供后续实现遵守)

1. **截断 reward 靠"计错"自然传导** (finish_reason=length 强制 em=0), 不设 λ 人工权重 (假学分可伪造)。
2. **长度走非劣约束** (`clean_pass_tokens` + pass 不降), 非单目标"越短越好" (corr(pass,len)=+0.30)。
3. **经验载体 = skill + facet 库累积计数**, 不是 bank / base_rule 预埋。trunc_bank.py 旁路方向作废。
4. **诊断信号 = 残文行为签名** (tail_loop/cycle_reps/answer_pos), 让 patcher 看具体病态而非抽象率

## 待跑 (需用户批准)

端到端 run: `--emergent_facets --budget_focus --dual_gate --trunc_gate` 同时开, 起点仍纯 S₀ (599 chars)。
核心判据: 起点无任何纪律词 → N 轮后 skill **自主**长出含 `_BUDGET_COVERAGE_SIG` 词汇的收尾纪律段, 且来自 patcher 对真实残文病态的响应 (非预埋 / 非人工 base rule)。
估算 ~2500 flash 调用。

## 复核修复 (2026-11-18, issue #88)

增量 A 落地代码经二轮审查发现 4 处 bug/隐患, 已全部核实属实并修复前 3 处:

| 修复 | 内容 |
|------|------|
| Bug 1 (🔴) | budget 标签不匹配: EFL-ON facet 名 "Budget Exhausted" ≠ EFL-OFF label "Budget-Exhausted (truncated...)" → `find_uncovered_type` 早退分支永不执行. 改用 `_is_budget_label` 前缀匹配. |
| Bug 2 (🟡) | 跨文件 lazy import + except Exception 静默回退哨兵. 改 `_budget_cov_sig` 实例属性注入, 去跨文件 import. |
| tiebreak 方向反 | `_efl_majority` 中 budget tiebreak 原为正(优先), 与 docstring "排最后" 矛盾; 改 -1 延迟. |
| Bug 4 (⚪) | self_test 缺 get_signature(budget) 覆盖; 补 3 断言. |
| Bug 3 (⚪, 已修) | `_to_int` 仍是宽松 re.search (记忆声明不符). 改 `re.fullmatch` 拒非纯整数 + `extract_int_answer` 格式守卫 + 保留 `_to_int_legacy`. `\boxed{113.9}`→None (旧版 113). |

Bug 3 已一并修复; 端到端 run 照旧等待批准。详见 `docs/phase2_issues_202611.md` issue #88。

关联 [[trunc-gate-sidechannel-step4-20261117]] [[trunc-evolve-budget-run2-20261116]] [[feedback-factor-plan-20261116]]。