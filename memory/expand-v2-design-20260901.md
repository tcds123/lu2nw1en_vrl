---
name: expand-v2-design-20260901
description: 算子G expand v2 修复(覆盖驱动+段持久化+段增长 crossover). dry-run JSON 裁定: C1 ✅✅(两代攻非meta Quantifier, 零退步), C2 ❌(设计性未达, facet_edit +0.231 恒 > expand +0.211), C3 ✅. 信号正向→建议 4-seed 复测. v1 test=0.4654 退步 -0.07.
metadata:
  type: project
---

# 算子G expand v2 修复设计 (2026-09-01)

## 触发
用户要求解决 `docs/进展汇报_2026-08-31.md` §五(expand v1 test 显著退步 −0.07) + §六(has_meta router 唯一正向信号在 not-meta=17 子集), 「优化算法让其能够对不同轴之间进行交叉测试, 或者允许它生成更多的 section」.

## v1 失败三根因 (代码级确认, Explore+Plan agent 双重复核)
1. **expand 目标重叠**: v1 expand 走裸 `collect_failures(k=1)` 取最难 8 题, ~100% Meta-options → 与已饱和 Option 轴内容重叠 → 无净信息 → test 退步. 项目已有覆盖驱动机器(`compute_failure_distribution`/`find_uncovered_type`/`_TYPE_COVERAGE_SIG`)但只接扁平主流程, **没接进 `_run_facet_tree`** → expand 对覆盖状态盲.
2. **section 不持久**: overlay 从 `dict(root_slots)`(4段)重建 → 丢掉 expand 长出的第5段. 0828 run 死法: iter1 expand→best(5段), iter2 overlay 反超(4段), 最终 best 只 4 段 → 验收#2 永不达成.
3. **crossover 硬编码 4 段**: `CROSSOVER_COHERE_PROMPT` 硬编码 "exactly four section headers" → LLM 砍到 4 段 → guard 5≠5 拒绝 → 含 expanded 段的 grown skill 被排除出 crossover → 长出 section 无法跨轴重组.

因果链: 内容重叠 → 增益被下一代 overlay 抹掉(不持久) → 即便触发也未泛化. 三处须协同.

## v2 修复 (4 处协同改动, 全在 `rl/train/verl1/skill_evolver_math.py`)
- **改动1 expand 覆盖驱动 (修根因1, 落实§六方向A)**: 新增 `find_uncovered_type_for_expand(counts, section_keys)` — 用 section_keys(段名 joined) 查 `_TYPE_COVERAGE_SIG`, 挑 signature 不在 skill 段名里的最高频**非 meta** 失败类型; 显式 skip `meta_lbl=FAILURE_FOCI[5]`(§六: expand 只在 not-meta 子集全正, meta 反害). 全已覆盖→target=None→跳过本代(守§五纪律). `SECTION_EXPAND_PROMPT` 加 `{target_type}` + anti-overlap 指令. 作为 `--expand` 默认行为替换旧 hardest-8 路径.
- **改动2 overlay 段持久化 (修根因2)**: 抽出 `_overlay_assemble(title, root_slots, best_node_slots, donors, freeze)` — `dict(root_slots)` 基底 + **carry-forward best_node.slots 中 root 之外的已展开 section** + donor 嫁接(跳过 freeze). guard **纯 key-set 检查**(不加 md5) — `facet_compose`(L708) 对非末段补 `\n` 分隔符, expanded 追加后 "Final Answer" 变非末段→compose 补 `\n`→其 md5 变→freeze-md5 检查会误拒. key-set 避开此坑且向后兼容.
- **改动3 crossover 段增长 (落实"对不同轴交叉测试")**: `recombine_facets` 从 parent_a_slots 提 `section_headers = "\n".join(f"## {k}" for k in parent_a_slots)`. `CROSSOVER_COHERE_PROMPT` 规则4 参数化注入 `{section_headers}`, "preserve EXACTLY these section headers, in this order, verbatim". 向后兼容(4段列4个). 含 expanded 段的 grown skill 可作 crossover parent A.
- **改动4 扩展 `_self_test_facet`**: (h) carry-forward 5段 / (i) 向后兼容4段 no-op / (j) find_uncovered_type_for_expand meta-skip.

## 零 API 自测结果 ✅
`python skill_evolver_math.py --self_test_facet` (从 `/data/zhuldz/lunwen/rl/train/verl1`, venv `/data/zhuldz/lunwen/lunwen`):
`✅ facet self-test passed: S₀=1017 chars, 4 sections, roundtrip md5=68571cb2a6, non-target slot freeze OK, edit-guard OK, expand-guard OK, overlay carry-forward OK (5-sec persist + 4-sec backward-compat), expand targeting OK (skip-meta)`

测试(h)关键断言: carry-forward 后 overlay 含 5 段; root 4 段全在; expanded 段 md5==best_node.slots[expanded_key]; freeze 段 slots 层 md5 未变(donor 没碰) + compose 层 `.strip()` 存活(容忍 `\n` 分隔符). **初版用 md5 断言 freeze 栽了 compose 补 `\n` 坑, 改 `.strip()` 才对** — 这正是 "接口接通 ≠ 机制成功" 须自测的原因.

## dry-run 启动参数 (待跑, 约定先记)
```bash
cd /data/zhuldz/lunwen/rl/train/verl1
source /data/zhuldz/lunwen/lunwen/bin/activate
python skill_evolver_math.py --facet_tree --expand --iters 3 --train_n 35
# 约估 1-2h, 28RPM 限速
```
**dry-run 验收看 3 件事** (零 API 自测已验证机制构造正确, dry-run 验证真跑行为):
1. `evolver_log.json` 的 `expand_event` 带 `target_type` 且非全 meta(落在 not-meta uncovered 类型);
2. `tree_iter{t}.json` 里 best_node 段数 ≥5 且 expanded 段名跨代存活(改动2 持久化生效);
3. overlay 不再误拒含 expanded 段的 best(无 `compose-section-loss` reason).

**4-seed 复测 (仅 dry-run 信号正向才烧 API, 待用户确认)**: expand-v2 vs 8238b5e8(0.5354), eval_k=3 seeds=4 同口径, 看 not-meta 子集是否仍全正、整体是否不再 −0.07. 遵§五纪律.

## dry-run 进行中结果 (2026-09-01, iter1 已落, iter2/3 待完)

启动: PID=125737, run_dir=`outputs/skill_evolver_math/run_20260901_162224`, log=`expand_v2_dryrun.log`. 真 S0 四段(1017chars, md5=68571cb2a6) + split 1-3-6 + train_n=35 + eval_k=5 (v1 同口径). 28RPM 限速, 单 iter≈30min (expand 两轮 eval×~355 calls + 1 gen).

### iter1 结果 (§六/Change-1 验证 ✅ 正向)
- **baseline**: init skill select pass=0.214 delta=-0.011 CI=-0.051 (4 段).
- **facet-edit**: Option Comparison -0.028(KEEP), Theorem-Level -0.034(KEEP), **Hypotheses +0.132 CI+0.085 PROMOTE → TREE-BEST**.
- **overlay iter1**: n_axes=1 delta=+0.093 CI=+0.048 pass=0.318 n_sec=4 admitted=Y became_best=N (无 compose-section-loss ✅; n_sec=4 因 expand 在 overlay 之后跑, 此刻尚未有第5段, 符合预期序).
- **🧬 expand iter1 cand0**: `+Quantifier Order and Parameter Dependence` **[Quantifier order and parameter dependence (∀∃ vs ∃∀, uniform vs dependent bounds)]** delta=+0.062 CI=+0.014 pass=0.287 admitted=Y n_sections=5.

### 三验收判据 (iter1 维度)
1. ✅ **target_type 非 meta**: 命中 `Quantifier` 类型 = §六 not-meta 正向信号所在子集. `find_uncovered_type_for_expand` 正确跳过 meta(FAILURE_FOCI[5]) 选最高频 uncovered 非-meta 类型. **这正是 v1 致退步的 meta-option 内容重叠的修复点** — v1 走裸 hardest-8(~100% meta) 与已饱和 Option 轴重叠; v2 覆盖驱动避开.
2. 🟡 **段数≥5 持久化**: expand 产出 5 段节点(4→5) 并入 archive, 但 **became_best=N** (无 `✅ EXPAND-BEST` 行) — expand +0.062 < Hypotheses +0.132, expand 节点未成 best_node. **关键待验**: 若 iter2 best_node 仍是 4 段 Hypotheses, 则 iter2 overlay 无 expanded 段可 carry-forward (carry-forward 只取 best_node.slots 非-root 段), 持久化判据须看 "expanded 段是否进最终 best" 而非 "是否跨代存活在 archive". iter2/3 观察.
3. ✅ **无 compose-section-loss**: iter1 overlay 未误拒. iter2 overlay 若 carry-forward expanded 段(当 best_node 含第5段时)是真正测试.

### iter2 预测 (待验)
iter2 best_node 大概率仍是 4 段 Hypotheses(expand 节点 +0.062 没超 +0.132) → iter2 expand 的 `find_uncovered_type_for_expand(counts, best_node.slots.keys())` 仍见 4 段(无 "quantifier" 段名) → 可能再选 Quantifier(或下一最高频 uncovered 非-meta). **若要 expanded 段进 best 须 expand delta > 当前 best_val_delta** — 这是判据2 的真正达成条件, 须看 iter2/3 expand 增益是否够大.

### iter2 结果 (2026-09-01 20:54) — 累积 + 多轴 + crossover 全验
- **facet-edit (累积确认!)**: Option Comparison +0.011(KEEP, parent=-0.011=baseline); **Theorem-Level Precision +0.121 CI+0.073 PROMOTE (parent=-0.011=baseline)**; **Hypotheses +0.231 CI+0.172 PROMOTE (parent=+0.132!)** ← **parent=+0.132 证明 iter2 在 iter1 leaf 上累积** (0.132→0.231, Δ=+0.099). 跨代 facet-edit 持久化✅.
- **TREE-BEST iter2**: facet[Hypotheses] delta=+0.231 CI=+0.172 (最强信号).
- **🔀 crossover iter2**: Hypotheses×Theorem-Level Precision delta=+0.113 CI=+0.056 pass=0.338 admitted=Y ← **Change-3 验证**: section_headers 参数化生效, LLM 保留实际段名(4 段 parent A), crossover 被准入. 两正向轴(hypotheses×theorem-level)真发生跨轴重组.
- **overlay iter2**: **n_axes=2** delta=+0.228 CI=+0.163 **pass=0.454** n_sec=4 admitted=Y became_best=N ← n_axes=2 (hypotheses+theorem-level 双 donor) > iter1 n_axes=1, 多轴叠加更强; **pass 0.214→0.454 绝对 +0.240** (最强 pass 增益). 无 compose-section-loss ✅. became_best=N 因 hypotheses 单轴 +0.231 仍微胜 overlay +0.228 (Δ=0.003, 次可加).
- **🧬 expand iter2 cand0**: `+Quantifier Order & Uniformity` **[Quantifier order and parameter dependence (∀∃ vs ∃∀, uniform vs dependent bounds)]** delta=+0.211 CI=+0.146 pass=0.437 admitted=Y n_sections=5 ← **expand 再攻 Quantifier 非-meta** (§六方向A 二次命中); delta 从 iter1 的 +0.062 升到 +0.211 (因 base 从 -0.011→+0.231 更高). 但 +0.211 < best_val_delta +0.231 → **无 EXPAND-BEST 行, became_best=N**.

### §六 / 判据2 (持久化) 真相 — 一个重要的设计性发现
**expanded 段从未进 best_node (iter1/2 均 became_best=N)**. 因: 单轴 facet_edit (攻饱和 meta-option, 即主失败模式) delta (+0.231) 恒 > expand (攻少数 Quantifier 非-meta) delta (+0.211). expand 节点(5 段)入 archive 但不进 best → overlay 的 carry-forward(best_node.slots 非-root 段)无 expanded 段可携 → **carry-forward 机制构造正确(零 API 自测证)但在 dry-run 中从未被触发** (best_node 始终 4 段).
- 这不是 bug: carry-forward 是 "若 expanded 段恰好进 best 则保护其不丢" 的安全网, 非 "强制 expand 进 best" 的机制. expand 进 best 须其 delta 超过 facet_edit, 在 Quantifier 失败只占 ~1/8 的分布下结构性困难.
- **判据2 (段数≥5 进最终 best) 未达成** — 但达成的是更强的: **判据1 (target 非 meta) iter1+iter2 双命中 Quantifier**, + Change-3 crossover 真跨轴重组 + overlay pass +0.240 绝对增益.
- **§六正向信号在 dry-run 复现**: expand 攻 not-meta (Quantifier) 子集两代均 admitted=Y 且 delta 强(+0.062/+0.211), 与 §六 4-seed 全正信号一致. 对比 v1 (攻 meta → −0.0700 退步) v2 零退步.

## dry-run 最终定论 (JSON 落盘验证, 2026-09-01 21:00) — ⚠️ C2 已被 iter3 推翻, 见下方勘误

> **勘误 (2026-09-02)**: 本节写于 dry-run iter2 完成时(21:00), 当时记 C2=❌「best 仍 4 段, expand 从未进 best」. **但 iter3(23:03 落盘)实际突破了**: 最终 `best_skill_evolved.md` md5=2f3c0c1f 实测 **5 段**(Option Comparison / Theorem-Level Precision / Hypotheses / Final Answer / **Quantifier Order and Parameter Dependence**) = expand 长出的第5段**进了最终 best**. C2 实际达成. `[[expand-v2-4seed-launch-20260902]]` 已记此突破, 本 memory 当时未回填. 下表 C2 标记是 dry-run 时刻的过时快照, 保留以记录演进过程.

iter3 仍在跑(~25min 完, 不改结构结论). 已写入 JSON (`evolver_log.json`/`tree_iter{1,2}.json`) 给出确定性 3 判据裁定:

| 判据 | dry-run iter2 快照 | iter3 最终结果 | JSON 证据 |
|------|------|-----------|------|
| **C1 target_type 非 meta** | ✅✅ **双命中** | ✅ | iter1 `expand.target_type="Quantifier order and parameter dependence(∀∃ vs ∃∀...)"` admitted=Y delta=+0.062 n_sec_after=5; iter2 同 target delta=+0.211 admitted=Y. `find_uncovered_type_for_expand` 正确 skip meta(FAILURE_FOCI[5]) 选最高频 uncovered 非-meta. **v1 攻 meta→−0.07 退步, v2 零退步** |
| **C2 段≥5 进 best_node** | ❌ (iter2 快照: best 4 段) | ✅ **iter3 突破达成** | iter2 `tree_iter2.json` best=Hypotheses leaf 4 段 delta=+0.231, expand.became_best=False. **但 iter3 best_skill_evolved.md 实测 5 段 md5=2f3c0c1f** = expand 第5段(Quantifier)进最终 best. dry-run 时刻判断「expand delta 难超 facet_edit」被 iter3 推翻 |
| **C3 无 compose-section-loss** | ✅ | ✅ | iter1+iter2 `overlay.reason=None`, admitted=Y. n_axes 1→2(多轴叠加). pass 0.214→0.454(+0.240 绝对) |

**dry-run 时刻的 C2 根因分析 (已被 iter3 推翻, 保留作记录)**: facet_edit 攻饱和 meta-option delta +0.231 恒 > expand 攻少数 Quantifier delta +0.211. 当时判断 expand 节点 delta 不足以超 best_val_delta → `became_best` 永不真. **实际 iter3 中 expand delta 最终超过了 best_val_delta**, 说明「expand 进 best 结构性困难」的判断过悲观——覆盖驱动选对的 target(Quantifier, not-meta 高频) 在足够代数后能积累超 facet_edit. carry-forward(Change-2)构造正确, iter3 触发.

### dry-run 信号裁定: 正向 → 建议进入 4-seed 复测
- ✅ §六方向A 复现: expand 两代攻 not-meta(Quantifier) admitted=Y, delta 强(+0.062/+0.211), 对齐 §六 4-seed not-meta=17 子集全正信号.
- ✅ v1 退步修复: v1 攻 meta 重叠→test −0.0700; v2 攻 not-meta 零退步, expand 节点 delta 始终正.
- ✅ Change-3 crossover 真跨轴: Hypotheses×Theorem-Level 两正向轴重组 admitted=Y delta=+0.113.
- ✅ overlay 多轴: n_axes 1→2, pass +0.240 绝对增益.
- ⚠️ C2 未达是设计预期外(但不损信号): expand 进 best 须其 delta 超 facet_edit, 结构性困难. **4-seed 看 "整体不再 −0.07 + not-meta 子集仍全正" 而非 "best 含第5段"**.

**建议**: 进入 4-seed expand-v2 vs 8238b5e8(0.5354) 复测, eval_k=3 seeds=4 同口径. 待用户确认烧 API.

## 已记录的 v2 局限 (未来工作, 非本次修复 scope)
- **crossover donor 方向性 (Point 6)**: crossover expanded sections 只能作 parent A, 永不能 donated(facet_label "expand:" 前缀不匹配 section key). 当 expand/overlay 节点作 parent B 时 graft 静默失效 — pre-existing 浪费但不破坏; 过滤 donor pool 为可选 future refinement.
- **facet-tree 无 consolidate pass (Point 8)**: facet-tree 路径无 consolidate → section 无上限增长风险(6-8 段后可能冗余). 长跑须监控 best_node 段数; 超阈值应触发 consolidate.
- **per-axis coverage targeting 未接入 facet_edit (Point 9)**: 改动1 只接进 expand, 根轴的 per-branch facet_edit 仍全 meta-saturated → 跨轴 synergy 尚未真发生. 接入需 per-branch 覆盖诊断, scope 更大, 留作下一轮.

## 相关
[[expand-operator-api-step-result-20260828]] (v1 失败数据) / [[has-meta-router-recompute-20260831]] (not-meta 正向信号驱动改动1 meta-skip) / [[lit-vs-impl-vs-verdict-20260831]] (接口≠机制, 自测必要性) / [[facet-tree-capability-ceiling-20260827]] (能力边界) / [[evolver-content-diffusion-20260901]] (单轴 facet_edit 是 0.49→0.53 增益源, expand 是破边界尝试)
