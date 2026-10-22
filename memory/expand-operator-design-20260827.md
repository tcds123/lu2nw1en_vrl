---
name: expand-operator-design-20260827
description: 治本改动决策记录. 8-seed坐实evolver显著落后skillopt -2.9pp后, 代码级诊断证明根因=facet_edit算子结构上禁止新增section(单slot单bullet). 用户选"加展开型算子(治本)": 给 skill_evolver_math.py 加算子G(section-expansion), 突破约束生成决策程序/领域特化section. 设计要点: 独立guard绕过4道禁新增护栏, 镜像overlay接入主循环, --expand默认关保持向后兼容.
metadata:
  type: project
---

# 展开型算子 (算子G) 设计决策 (2026-08-27)

## 触发链 (为什么改)
1. 8-seed 同窗口 h2h 坐实 gap: evolver(8238b5e8) 0.5212 vs skillopt(cfae156d) 0.5503, Δ=-0.0291, 两判据CI上界均<0. 显著落后. 见 [[h2h-8seed-final-result-20260827]].
2. gap 坐实后诊断门合法打开. 代码级正向推导(非gap倒推): facet-tree 核心算子 `facet_edit` 结构上只能单slot内精炼一个bullet, **物理禁止新增 section**. 见 [[facet-tree-capability-ceiling-20260827]].
3. skillopt 的优势恰是 8238b5e8 生成不了的 6 个展开型 section:
   - `## Meta-Option Decision Procedure` (5步决策程序: identify→validity→maximality→precision→doubt)
   - `## Strength Discrimination` (数学示例: s_j 等式vs不等式, stably-unbounded)
   - `## Domain and Range Precision` (domain/range 比较: p>2 vs p≥1, cap(E,Ω) vs cap(E,D))
   - `## Meta-Option Strategy for 'Strongest Provable'` (另一5步程序, 包在 SLOW_UPDATE 标记)
   - `## Domain-Specific Heuristics` (6领域→失败映射: PDE/模型论/群作用/图论/集合论/capacity)
   - `## Final Decision Rule` (短 tie-break)
4. 用户最终选择: **"加展开型算子(治本)"** — 直接对准能力缺口, 不重跑判别实验(实验二已证循环parity作废).

## 设计要点 (实现锚点)
**目标文件**: `rl/train/verl1/skill_evolver_math.py` (1908行)

**必须绕过的4道禁新增section护栏** (新算子用独立 guard, 不走这些):
1. `facet_edit_guard` (L790-791): child_slots>1段 → 拒
2. `recombine_facets` (L853): set(child_slots)!=set(parent_slots) → 拒
3. `CROSSOVER_COHERE_PROMPT` (L816): 明令"keep EXACTLY four headers, do NOT add"
4. overlay 结构检查 (L1282): set相等才过, 超集拒

**新增代码块**:
- `EXPAND_CATALOG` + `SECTION_EXPAND_SYSTEM` + `SECTION_EXPAND_PROMPT` (常量, ~L773后)
  - prompt 取 `{section_name}`, `{section_purpose}`, `{existing_sections}`, `{diagnostics}`; LLM 只输出新 `## {section_name}\n{body}` 块
- `section_expansion_guard(parent_slots, expected_key, child_text)` → `(new_slots, ok)`
  - copy parent_slots, 仅新增 expected_key; 校验现有slot字节不变(构造性保证, 只add不replace); 拒空body/同名/多section
- 主循环算子块 (镜像 overlay L1269-1313, 插在 overlay 块之后 L1313)
  - gate: `getattr(args,"expand",False) and len(archive)>0`
  - 作用对象: `best_node` (非per-branch leaf, 避免结构漂移)
  - 节点 facet_label = `expand:{section_name}` (select_axis_donors L886 过滤 branch_axes, 自动排除 expand 节点不被当单轴改进移植)
  - 准入 (镜像overlay最严): `(op>=base_pass-0.06) and (cl>0)`; 全局best: `cd>best_val_delta and cl>0`
  - max-sections cap (~8-10) 防膨胀
- argparse: `--expand` (BooleanOptionalAction, **default=False** 向后兼容), `--expand_every`(freq default 1), `--expand_candidates`(default 1) — 控API成本
- `_self_test_facet` (L701-744) 增加 expansion-guard 断言: 干净单新section接受/同名或跨section拒绝. (注: 该self-test读硬编码S0 L703, 测S0完整性非evolved skill, 故4段断言 L706-708 不受 expansion 影响)

**持久化**: `log["iters"]` dict (L1341-1348) 加 `expand_event` 键镜像 `overlay_event`; tree_iter{t}.json/evolver_log.json/archive_final.json 不变.

## 边角 (待实现时处理)
- 尾换行粘接: facet_compose L684-685 给无\n结尾的body补\n, 新section body须遵守同样契约不破坏下一段
- 冻结段 `Final Answer` 保护: 新section只add不改, 冻结段值永不触碰
- LLM改名/吐多section: guard 用 facet_decompose 拆 child_text, 必须 exactly 1 新section 且 name==expected_key
- expand 与 overlay 共存: overlay 从 root_slots(4段)重建, 若 best 是 expanded(5+段) skill — 需确认 overlay 是否仍只看 root_slots(不冲突, overlay 独立路径), 或 expansion 基于当前 best_node.slots(已含扩展段)

## 验证计划
1. **离线零API**: `python skill_evolver_math.py --self_test_facet` — expansion-guard 断言通过
2. **API Step** (用户确认实现后): `--expand` 跑一轮, 检 tree_iter*.json 的 expand_event / evolver_log.json / best_skill_evolved.md 是否出现新 ## section 且 admitted. 结果记 memory/.

## 约束
- `--expand` 默认关, 既有 run 零影响 (8238b5e8 复现性不受损)
- 不启动API run 直到用户确认实现方案
- API key 不外泄 (代码内硬编码 msk-bf2324..., 仅 DeepSeek/Qwen3-235B 授权)

## 关联
[[h2h-8seed-final-result-20260827]] (gap坐实, 本改动前提) / [[facet-tree-capability-ceiling-20260827]] (代码级能力边界诊断, 本改动直接对准) / [[h2h-gap-diagnosis-20260827]] (诊断+三校准) / [[evolver-lineage-three-experiments-20260827]] (三场血统, 实验二parity作废)
