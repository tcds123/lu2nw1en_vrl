---
name: gpt-review-verdict-20261016
description: GPT review 复审 AIME 线翻转算法归因, 5 项关键发现全部核实, 决定先修测量地基
metadata:
  type: project
---

# GPT Review 复审结论 + 方向决策 (2026-10-16)

**触发**: 用户解压 `for_review/for_review_10-16-gpt_review.zip`, 看 GPT review 结论。GPT 产出两份 findings, 本次会话对照源码+run产物逐条核实, **5 项关键发现全部属实**。

## 已核实的关键发现

| # | 发现 | 严重度 | 核实锚点 |
|---|------|--------|----------|
| F2 | group/subset 救援是**死代码** | 🔴P0 | `skill_evolver_aime_v2.py:3668` `if admitted:` 块内调 `_combined_rescue:3698`, `pool_ci_gate=0` → 整体CI<0 时走 else 不调救援。overlay:3496→3514 同结构。facet_edit 没接救援 |
| F3 | run#2 已有 **5 次几何定向 expand 全负 delta** | 🔴P0 | run#2 evolver_log iter2/3/6/7/10 几何section, delta −0.022~−0.111。§5"换几何就有增益"假设已被证伪 |
| F1 | 门控统计功效不足 | 🔴P0 | 30题×k=3 只可靠检出+4题; 3零分题全修好检出率仅30%; S₀自delta σ≈0.044。0 admit = 测量上限非算法失效 |
| F6 | s0_cache 每 run 重算非复用 | 🟠P1 | 三份 md5 全不同; run2 vs run3 有 13/30 题 pass_rate 不同。无 _manifest 加载器拒用重算 |
| F4/F5 | EFL 塌缩回人工种子; C 实为 B | 🟠P1 | 终态只剩 F0-F7; 全走 DeepSeek-V3.2 无 V4-Flash/Qwen3-235B |

**关键转折**: 原"merge-protection + headroom-focus"计划逻辑链有 3 环被推翻 — 几何 patch 已证伪(F3) + 救援死代码(F2) + 缓存不复用(F6)。单独跑无法验证算法有效。

## 方向决策

用户选 **修测量地基** (GPT Step1-2), 优先级:
1. **F6 缓存复用**: 实现 AIME 专用 cache manifest (记录数据哈希/S₀哈希/模型/温度/k), 加载器校验身份而非读 correct_choice。让同条件复用缓存不触发 API。
2. **F2 救援可达**: 把 `_combined_rescue` 移出 `if admitted:`, 让整体CI≤0 时也能到达救援; facet_edit 接上救援。修否决规则 (CI上界<0 而非下界<−eps)。
3. **F1 提功效**: k=3→8, 候选与S₀同批配对重测, 去掉全对题只评有信息量题; 预注册假设+多重比较校正。
4. 修完 F1/F2/F6 三个 P0 后, 才回到算法 (merge-protection 等仍可保留, 但作为地基修复后的验证)。

**Why**: F1/F2/F6 三个 P0 不修, 任何算法改动的结果都不可解读 (GPT 原话: "0 admit 是测量上限, 不是算法失效的证据")。

**How to apply**: 下一轮算法实验前必须先修测量地基。验证标准 = 同条件复用缓存不调API + 变更配置正确失效 + 救援路径在整体CI<0时可达 + k=8能可靠检出+2题。

详见 `docs/phase2_issues.md` (2026-10-16 条目, 含逐条核实锚点)。

关联 [[for-review-repackage-aime-20261015]] [[grpo-step700-status]]。
