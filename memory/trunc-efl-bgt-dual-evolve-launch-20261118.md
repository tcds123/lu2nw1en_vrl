---
name: trunc-efl-bgt-dual-evolve-launch-20261118
description: 防截断自进化端到端 run — EFL + budget_focus + dual_gate + trunc_gate 四机制联动, AIME24 S₀(599 chars), issue #87/#88 全修复后首发
metadata:
  type: project
---

# 防截断自进化端到端 run 启动 (2026-11-18)

**这是 issue #87/#88 全修复后首次端到端 run，核心判据: S₀ 无任何收尾纪律词 → N 轮后 skill 自主长出含 _BUDGET_COVERAGE_SIG 词汇的收尾纪律段。**

## 启动命令

```bash
bash rl/train/verl1/skill_evolution/launch_trunc_efl_evolve_v3.sh
```

脚本: `rl/train/verl1/skill_evolution/launch_trunc_efl_evolve_v3.sh` (本次新建)

## 后端决策: DeepSeek-V3.2 (非 flash)

**用户建议换 V3.2 以避开 flash 20 RPM 抢资源** — 采纳。V3.2 走 `aimp-deepseek/v1` 独立端点 +
独立配额 (注释 30 RPM, 稳妥设 28), 与 #48 Arm1 (appsplus flash) **可并行, 不互抢**。

**V3.2 key 记录** (已写入 `configs/models_aimp.local.yaml` providers.deepseek-v3.2, 不入库):
`${AIMP_KEY_DEEPSEEK_V3}` (脱敏，从历史 `.bak` 文件恢复: `_env.sh.bak` / `launch_aime_codeagent.sh.bak` 等均记录同一 key)

**V3.2 端点参数**:
- base_url = `https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1`
- model = `/model/DeepSeek-V3.2`
- AIGC-USER = `zhuldz` (必需, 与 flash 的 llm-openai-api 不同)
- SSE: skillopt qwen_backend 已处理 (显式 `stream:false` 避免 SSE, 见 qwen_backend.py:286-292)
- token_key = `max_tokens` (use_max_completion_tokens=false 默认)

## 配置

| 参数 | 值 |
|------|-----|
| S₀ | `/data/zhuldz/lunwen/data/skill_aime_initial_v2.md` (600 B, ≈599 chars) |
| 数据 | AIME24 30 题, `/data/zhuldz/lunwen/data/aime24.jsonl` |
| 后端 | deepseek-v3.2, aimp-deepseek 端点, aimpapi.midea.com |
| cap | 16384 (chat_target monkeypatch, 避免 AIME 硬题 CoT 人为截断 #26/#29) |
| 温度 | temp=0.0, enable_thinking=false |
| RPM | 28 (V3.2 配额 30, 稳妥) |
| seeds | k=20, 2 seeds |
| iter | 4, train_n=30, K=4, eval_k=3, n_diag=8 |
| 算子 | FULL: facet_tree+bootstrap+crossover+overlay+expand+subset_admit |
| s0_cache | "" (强制现场重算 V3.2 baseline, #28) |
| 新机制 | --emergent_facets --budget_focus --coverage_diagnosis --dual_gate --trunc_gate |

## 依赖前提

- issue #87 两增量 (A: budget 覆盖签名, B: 残文行为签名) ✅ 已落地
- issue #88 四 bug 全部修复 ✅ (Bug1 标签匹配, Bug2 跨文件 import, Bug3 _to_int fullmatch, Bug4 测试缺口)
- _efl_majority tiebreak 方向 ✅ 已修正
- self_test + _self_test_facet 全过 ✅
- py_compile 全过 ✅
- V3.2 smoke (≤5 call) ✅: skillopt chat_target 返回正常

## 调度

- **V3.2 与 flash 不冲突** → 无需等 #48 Arm1 退出, 已并行启动
- first run PID 89788 (15:28 启动) — 崩在 iter1 `scored` NameError (issue #88.5)
- **重启 run PID 9874/9876/9896 (17:34 启动)** — 修复 #88.5 后正常推进
- 预计 ~2500 calls @ 28 RPM ≈ 90 min 下限 (实际 V3.2 token bucket 较慢, 料 >2h)

## V3.2 S₀ baseline 数据点 (precompute 落盘)

| 指标 | 值 | 对比 (flash 首跑) |
|------|-----|-----|
| S₀ select pass | 0.756 | flash 首跑 0.800 |
| S₀ delta | -0.033 (CI -0.111) | flash +0.011 |
| clean_pass_tokens (ck) | **3473** | flash 2696 |
| 截断率 tr | 0.000 | 0.000 |

> V3.2 的 S₀ 答案更长 (ck 3473 vs flash 2696, +29%), 但 select 集上 pass 略低 (0.756 vs 0.800)。
> 这是 V3.2 vs flash 的模型差异, 作为后续 cross-model 对比的 anchor。

## 判据 (零人工预埋)

1. 起点 S₀ (599 chars): 无 commit/discipline/checkpoint/token_budget 等 _BUDGET_COVERAGE_SIG 词汇.
2. patcher 对 EFL classify→budget facet focus + 残文病态签名 做出响应 → skill 中**自主**出现收尾纪律段.
3. 纪律段 BUDGET_COVERAGE_SIG 词汇命中 → get_signature 返回覆盖签名 → budget facet 判 COVERED → 下一轮 focus 让出给其他轴.
4. 截断率应下降 (finish_reason=length 减少), 即便 pass 不显著上升 (截断题原 0 分 → 写完可能也对错).

## 最终结果 (2026-11-18 22:00)

**run 成功完成 4 轮** (~4.6h, PID 82061), 但 **skill 零进化**:

| 判据 | 结果 |
|------|------|
| best_skill 是否变化 | ❌ 无: 599→599 chars, hash c11d7dc0c8 不变 |
| focus 是否落 Budget | ❌ 无: budget facet count=0 → never selected |
| 是否有截断信号 | ❌ 无: S₀ baseline tr=0.000 (16384 cap 下不截断) |
| expand 是否通过门控 | ❌ 无: 4轮 CI 全负 (lost-or-ci<=0) |
| best_skill 自含 BUDGET_COVERAGE_SIG 词汇 | ❌ 无 (skill 未变) |

**根因**: S₀ 在 cap=16384 + V3.2 下截断率 0% → budget facet 无 failure 累积 → `find_uncovered_type` 永不见 budget key → 防截断自进化无目标信号。

## 2026-11-19 决议: V3.2 降 cap 到 8192 做对比实验

**用户澄清**（我此前误判为"换 flash"）: 用户要求的是 **V3.2 + 较小 cap**，看 V3.2 在低 cap 下是否会产生截断——不是换模型。

**执行**: `launch_trunc_efl_evolve_v3_cap8192.sh` (PID 39313, 2026-11-19 09:15).
- cap 16384→8192, 其余参数不变 (S₀/V3.2/aimp-deepseek 28 RPM/4iter).
- 核心看点: S₀ baseline 的 `tr=` 是否 >0。若 V3.2 在 8192 下截断 (tr>0) → budget facet 有 failure → 自进化有信号。
- 误启动的 flash cap=8192 run (PID 25051) 已 kill，只跑 V3.2 版。

**背景**: cap-sweep (2026-11-13) 已测 flash: cap=8192 截断 ~10-12%, cap=4096 ~20-22%。V3.2 缺对应数据，本次补齐。

### 关键结果 (precompute S₀, 2026-11-19 10:20)

| 指标 | V3.2 cap=16384 | V3.2 cap=8192 |
|------|---------------|---------------|
| S₀ select pass | 0.756 | **0.667** (降 cap 答错更多) |
| S₀ delta | -0.033 | -0.122 (CI -0.211) |
| S₀ tr (截断率) | 0.000 | **0.022 (2.2%)** ✅ |
| budget facet count | 0 | **1** (log 已见 2) ✅ |

**✅ 防截断自进化前提达成**: V3.2 降到 8192 后产生截断 (tr=2.2%)，EFL 已把截断失败分类进 budget facet (count≥1)，`find_uncovered_type` 现在可以选中 budget 轴。对比 flash cap=8192 tr~10-12%，V3.2 的 CoT 更紧凑、截断远少。

### issue #89 根因定位 (2026-11-19)

**harness 汇总漏计截断的根因**：qwen_backend 的**截断自动重试 (trunc retry)** 机制。

完整链路：
1. `chat_target` call → finish_reason=length → `_LAST_CALL.meta = {finish_reason: "length"}`
2. `_chat_messages_impl` line 541: trunc_left=1 > 0 → token 加宽到 32768 → `continue` (retry)
3. retry call → finish_reason=stop → `_LAST_CALL.meta` **被覆盖**为 `{finish_reason: "stop"}`
4. `run_one` 的 `get_last_call_meta()` → 只看到 stop → `harness["n_truncated"]` = 0
5. trace 层独立记录（`_trace_write` 在每次成功 call 后立即写，不受后续覆盖）→ 有 10 条 length

**修复**：在 launch 脚本中加 `export QWEN_CHAT_TRUNC_RETRY=0` 禁用截断重试。防截断是实验目标（不是自动重试就能解决的障碍）。

---

## V3.2 cap=8192 完整结果 (2026-11-19, ~4.4h)

| 指标 | 值 |
|------|-----|
| S₀ baseline pass | 0.667 (delta -0.122 CI -0.211) |
| S₀ tr | **0.022** (2.2%, harness 经 retry 覆盖后) |
| budget facet count | 1 |
| 4 iter 结果 | **全 KEEP-PARENT** (0 promoted) |
| best delta | -0.122 不变 |
| best_skill | 599 chars, hash c11d7dc0c8 不变 |
| BUDGET_COVERAGE_SIG | ❌ 无 |
| 每轮 Budget Exhausted | ✅ 各稳定 2 条 (Comp&Verification 分支) |

**结论**: 截断信号存在（每轮 2 条 Budget Exhausted），但 harness 漏计（trunc retry）导致 budget cnt=1 而非实际 ~6-8 次截断 → budget 不足以被 focus 选中 → 防截断进化无目标。

---

## ✅ V3.2 cap=8192 修复版 (QWEN_CHAT_TRUNC_RETRY=0) 最终结果 (2026-11-19)

### 历程总结

| run | cap | retry | 结果 |
|-----|-----|-------|------|
| run1 | 16384 | 1(默认) | 零进化 (tr=0 漏计, 实际有 10 条) |
| run2 | 8192 | 1(默认) | 零进化 (tr=2.2% 仍漏计) |
| **run3** | **8192** | **0(修复)** | **✅ 进化成功!** |

### 修复版 run 详细结果

| 指标 | 值 |
|------|-----|
| S₀ baseline | pass=0.756 delta=+0.044 CI=-0.011 tr=**0.056** (5.6%) |
| iter1 | delta=+0.044 CI=-0.011 — KEEP-PARENT |
| **iter2** | **delta=+0.100 CI=+0.011 — PROMOTED ✅** |
| iter3 | delta=+0.100 CI=+0.011 — KEEP-PARENT (继承 iter2) |
| iter4 | delta=+0.100 CI=+0.011 — KEEP-PARENT |
| best_skill | **817 chars** (599→817, +36%) |
| best_final_delta | **+0.100 CI +0.011** (显著) |
| BUDGET_COVERAGE_SIG | ❌ 无 |

### 进化内容分析

进化点在 Quantity Parsing 轴的子 bullet（通过 CI-gate）:
> After deriving the answer, re-express it explicitly in the requested form
> (e.g., "m+n = ...", "the probability is m/n, so m+n = ...") and verify
> it matches the problem's phrasing.

这是**quantity-rephrasing 检查法则**——改进了对"问题要求什么形式的回答"的语义检查，在 30 题上 delta +0.100 CI +0.011。

### 对核心判据的评估

| 判据 | 状态 |
|------|------|
| S₀ 无 BUDGET_COVERAGE_SIG 词汇 | ✅ True |
| 截断被正确记录 | ✅ True (tr=5.6%, Budget facet actionable count=8) |
| 自进化产出改进 | ✅ True (+0.100 显著) |
| 改进方向=防截断纪律段 | ❌ False (改进是 quantity rephrasing) |

因果链：截断信号修复 ✅ → 进化生效 ✅ → 但进化方向=答对更多题而非防截断。原因：Budget Exhausted 虽然 actionable (count=8)，但 Qty Parsing 的 answer-form 错误 (count=6) 也被 actionable，且 **quantity-rephrasing 改进绕过了 Qty Parsing 的 budget 失败**（用"检查表述形式"而非"commit 纪律"来提升 pass）。

**关键判断**: 防截断自进化的逻辑链路被修复了一个环节（trunc retry 漏计），进化生效了，但**进化方向仍是精度导向而非截断导向**。要让 patcher 产出防截断纪律段，可能需要更明确的残文病态签名来引导 patcher 的响应——这回到 issue #87 增量 B 的原始设计：渲染实际截断过程而非抽象计数。

### 下一步方向

1. 验证 Quantity Parsing improvement 的跨 seed/held-out 泛化 (AIME25 / fresh seeds)
2. 若泛化成立：修复版进化链路 = 有效基线
3. 若需防截断纪律段自动涌现：增强 patcher 的截断渲染（#87增量B 当前是「仅 n_truncated/n_samples token」，可改为逐条展示原始截断 finish_reason 行 + 残缺文本）

## V3.2 cap=8192 + retry=0 修复后 run (2026-11-19, run_20261119_140908, PID 39989, 已完成 4 轮 3h31m)

`launch_trunc_efl_evolve_v3_cap8192.sh` 加 `QWEN_CHAT_TRUNC_RETRY=0` (issue #89 L1 修复)。**首个具备有效截断信号的真正实验。**

### L1 修复验证 ✅ 完全属实
- trace 90 条 max_tokens 全 8192, **0 条 16384** (retry=0 生效, 不再 2x 翻倍), 5 条 length
- **tr = 5/90 = 0.0556 = 5.6%**, 与报告一致; trace=harness 零漏计 ✅

### ⚠️ 推翻 #90/#91 "L2 断接" 论断 (evolver_log expand 字段铁证)

**budget focus 在 facet_tree 路径真实命中, 4 轮命中 3 次** — 详见 phase2_issues_202611.md #91续:
- 经 `find_uncovered_type_for_expand` (L1307, 在 facet_tree L4221 被调) 接入, **非**扁平路径的 `find_uncovered_type`
- F11 Budget Exhausted: is_meta=false, is_budget=true, status=actionable, failure_count=5, promoted
- `get_signature` 因 `_budget_cov_sig` 注入 (L4769, 先于 facet_tree L5040) → 返回 _BUDGET_COVERAGE_SIG → 判 UNCOVERED → 可选 ✅
- 传导链全环打通 (length→harness→EFL.classify→_assign_budget→counts→expand 选中→生成→评估)

**budget focus 命中史 (4 轮 expand)**:

| iter | expand target | budget 候选 section | delta | ci | admitted |
|------|--------------|---------------------|-------|-----|----------|
| 1 | Budget Exhausted | Rapid-Exit Heuristics | +0.011 | -0.056 | ❌ |
| 2 | Missing Final Answer | Forced Final-Answer Emission | +0.011 | -0.078 | ❌ |
| 3 | Budget Exhausted | Early Commitment to a Single Counting Frame | **+0.056** | -0.033 | ❌ |
| 4 | Budget Exhausted | Budget-Exhausted Escape Hatch | -0.022 | -0.111 | ❌ |

budget 候选 delta iter3 达峰 +0.056 (CI -0.033, 最接近门控 CI>0), iter4 退回 -0.022 → 30 题样本 CI 跨 0, **被 dual_gate 诚实拦下, 非机制断接**。

### iter2 首次非零进化 ✅ (守到终局)
- Quantity Parsing: PROMOTE delta=+0.100 CI=+0.011 (parent +0.044)
- best_skill hash c11d7dc0c8 → **e00ed96a47**, best_val_delta 0.044→**0.100**, ci -0.011→**+0.011**
- iter3/iter4 均 KEEP-PARENT (+0.100 守住, 未被超越)
- **五轮实验以来 best_skill 首次显著进化** (前几轮全 KEEP-PARENT)
- 注意: 本次进化是 Quantity Parsing (数量解析) 维度, 非 budget 维度

### 最终结论 (run 完成后)
1. budget focus 机制在 facet_tree **完全连通且持续命中** (4轮3次), 推翻 #90 "断接"
2. budget 候选未过门控 = 诚实统计守门 (delta 振荡 +0.011→+0.056→-0.022, 30题 CI 跨0), 非失效
3. 真实进化来自 Quantity Parsing 维度 (非 budget): +0.100 CI+0.011 守到终局
4. "防截断自进化"目标本次未达成 (best_skill 不含收尾纪律段), 但**机制有效性已验证** — 传导链全环打通, 只差 budget 候选拿出统计显著正收益
5. #90 三个修法方向 (A/B/C) **均无必要** — 机制已通过 expand 接入; 真正瓶颈是 budget 候选 efficacy (生成质量+样本量)

### operator_counts 全局
| 算子 | attempted | admitted | promoted |
|------|-----------|----------|----------|
| facet_edit | 12 | 8 | **1** (iter2 Quantity Parsing) |
| overlay | 4 | 0 | 0 |
| expand | 4 | 0 | 0 |
总 2493 API calls, 8.96M tokens, 3h31m。

### 教训
- **"未过门控"≠"机制断接"**: admitted=false reason=lost-or-ci<=0 是诚实统计守门, 不是 bug
- **evolver_log expand 字段是判断 budget focus 命中的金标准**, 远比 grep 代码路径可靠
- #90 用 awk 取函数体搜无后缀名 `find_uncovered_type`, 漏了 facet_tree 走的 `find_uncovered_type_for_expand` (差 `_for_expand` 后缀)

## 关联

[[trunc-self-evolve-increments-20261118]] [[trunc-evolve-budget-run2-20261116]] [[trunc_shortcut_phase1_20261117]] [[48-formal-run-launch-20261117]]
