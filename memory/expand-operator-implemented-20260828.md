---
name: expand-operator-implemented-20260828
description: 算子G(section-expansion)已实现+离线验证通过. 在 skill_evolver_math.py 新增 EXPAND_SYSTEM/SECTION_EXPAND_PROMPT 常量、section_expansion_guard(只add不改)、主循环展开块(镜像overlay,gated by --expand默认off)、--expand/--expand_every/--expand_candidates argparse、self-test (g) 断言、expand_event 持久化. --self_test_facet 全绿. 下一步: API Step 验证(待用户确认), 从真S0四段+仅失败诊断驱动.
metadata:
  type: project
---

# 算子G (section-expansion) 实现完成 + 离线验证 (2026-08-28)

## 状态: 代码完成, 离线零API自测全绿; API Step 未启动 (等用户确认)

## 落地改动 (target `rl/train/verl1/skill_evolver_math.py`)
1. **常量** (`FACET_EDIT_PROMPT` 后) — `EXPAND_SYSTEM` + `SECTION_EXPAND_PROMPT`.
   prompt 取 `{existing_sections}`(现有段名清单) + `{diagnostics}`(失败诊断), LLM 自己起名并只吐一个
   `## {section_name}\n{body}` 块, 明令「不重述现有段、新增现有段没有覆盖的操作性内容」.
2. **`section_expansion_guard(parent_slots, expected_key, child_text, max_sections=10)`** — 构造性「只 add 不改」:
   - 名字撞现有段 → 拒; `facet_decompose` 拆出 != 恰好1段 → 拒; 实际段名 != expected_key → 拒; 空 body → 拒;
     超 `max_sections`(默认10) → 拒.
   - 通过则 copy parent_slots + 追加新段 (现有 slot 逐字节不变, 天然满足); body 契约对齐 facet_compose
     (顶格无前导换行, 以单个 `\n` 结尾).
3. **主循环展开块** (overlay 块之后, gated by `getattr(args,"expand",False)` 且 `len(archive)>0` 且 `t % args.expand_every == 0`):
   - sections cap=10 达上限跳过.
   - 诊断源复用 `collect_failures(diag_items, best_skill, ...)` 纯函数 + `make_diagnostics`.
   - 每候选: LLM 生成 → clean → 拆唯一段名 (名字由LLM自起) → guard → compose → eval → `bootstrap_ci_lower`.
   - 准入镜像 overlay 最严: `(ep >= base_pass-0.06) and (el>0)`; 全局 best: `ed>best_val_delta and el>0`;
     采纳即 `break` (同代不重复膨胀). 节点 `facet_label=f"expand:{exp_key}"` → select_axis_donors 自动排除
     (不被当单轴改进移植).
   - `except` LLM 失败 → `expand_event={"admitted":False,"reason":"llm-error"}` break.
4. **argparse** (`--probe_replay` 后): `--expand`(BooleanOptionalAction, **default=False** 向后兼容)、
   `--expand_every`(default 1)、`--expand_candidates`(default 1) — 控 API 成本.
5. **self-test (g)** (`_self_test_facet` 内, 断言(a)-(f) 后追加): 干净单新段接受(len 4→5、现有4段md5不变)、
   同名/异名/多段/空 四种拒绝路径全断言.
6. **持久化**: `log["iters"]` dict 加 `expand` 键 (与 crossover/overlay 并列); tree_iter/evolver_log/archive_final
   结构不变.

## 离线验证 (零API)
```
python rl/train/verl1/skill_evolver_math.py --self_test_facet
✅ facet self-test passed: S₀=1017 chars, 4 sections, roundtrip md5=68571cb2a6,
   non-target slot freeze OK, edit-guard OK, expand-guard OK
```
`ast.parse` 语法 OK; `--help` 显示 `[--expand | --no-expand] [--expand_every] [--expand_candidates]` 全接线.

## 设计承诺兑现 (用户两决策)
- ✅ 从真 S0 四段起 (best_node 起步 = 真 S0 系进化后的当前 best, 无 skillopt 嵌套; 新 section 不参考 skillopt).
- ✅ 仅失败诊断驱动 (`SECTION_EXPAND_PROMPT` 只吃 `{diagnostics}`, 无 human seed/prior prompt).
- ✅ 血统干净: 新 section 是凭空由失败诊断引导展开, 非搬运 skillopt 的 6 个 section.

## 边角处理记录 (对齐设计文档)
- 尾换行粘接: body 契约对齐 facet_compose (顶格+单\n结尾), 不破坏下一段.
- 冻结段 Final Answer 保护: guard 只 add 不改, 冻结段值永不触碰.
- LLM 改名/吐多 section: guard 用 facet_decompose 拆 child_text, 必须恰好1新段且名==expected_key.
- expand 与 overlay 共存: expand 从 `best_node.slots`(已含扩展段) 起步, overlay 仍从 root_slots(4段) 独立重建,
  两条路径互不冲突; best 命中 expand 后 facet_label="expand:{name}" 不被 select_axis_donors 单轴提取.

## 下一步 (未启动, 等用户确认)
- **API Step**: `--expand` 跑一轮 (真 S0 四段起), 检 tree_iter*.json / evolver_log.json 的 expand_event /
  best_skill_evolved.md 是否出现新 `## ` section 且 admitted. 结果记 memory/.
- 确认 gap 是否收窄 (vs skillopt -0.0291 硬基线); stats 用同窗口 8-seed CI-band 口径.
- 若 expand 增益不显著 → 不要为造 section 强开 (守 CI 纪律, 只判方向).

## 约束 (仍有效)
- `--expand` 默认关, 既有 run (8238b5e8 复现) 零影响.
- 不启动 API run 直到用户确认.
- API key 不外泄 (硬编码 msk-bf2324..., 仅 DeepSeek/Qwen3-235B 授权).

## 关联
[[expand-operator-design-20260827]] (设计决策, 本文件落地) / [[facet-tree-capability-ceiling-20260827]] (能力边界诊断, 算子G直接对准) / [[h2h-8seed-final-result-20260827]] (gap 硬结论, 改动前提) / [[evolver-lineage-three-experiments-20260827]] (血统, 实验二循环parity作废)