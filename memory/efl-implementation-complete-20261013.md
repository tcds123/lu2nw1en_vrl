---
name: efl-implementation-complete-20261013
description: Emergent Facet Library (EFL) 实现完成并通过零回归+集成验证 — 9步全绿, 几何 facet 可从失败数据涌现
metadata:
  type: project
---

## 完成状态 (2026-10-13)

承接 [[efl-implementation-started-20261013]] 的 9 步计划, 全部实现并通过验证。这是回应 [[skillopt-evolver-rootcause-20261013]] 根因的主攻方向落地: **给 evolver skillopt 的好半边(语义发现), 保留自己的好半边(LCB gate)**。

### 核心成果: 几何 facet 可从失败数据涌现 (约束#3 满足)

集成测试 (mock LLM 注入) 证明:
1. 两个几何失败(triangle/circumradius/hexagon)→ LLM EMERGE → F8 Geometry facet 从无到有出现
2. 第 2 个**不同题**的几何失败 → failure_count 累积到 2, problems_seen 跨题=2 → 触发 promote candidate→actionable
3. promote 后 `_efl_round_robin(9)` 路由到 Geometry (第 9 个 actionable facet) — focus 选择能命中它
4. `_coverage_sig('Geometry')` 返回题面词签名 (triangle/angle/circle/hexagon/circumradius) — 覆盖检查能用
5. `_is_meta_label('Geometry')==False`, `_is_meta_label(foci[5])==True` — meta 从位置索引 5 解耦为显式 flag
6. persist roundtrip 保留 Geometry actionable + count=2

**这是 evolver 此前完全做不到的**: 原关键词路由器把所有几何题误路由到 foci[5](答案格式), 现在能正确识别为几何并累积。

### 零回归保证 (已验证, 非声称)

`--emergent_facets` 默认 OFF。6 项静态+单测全绿:
- `_EFL = None` (flag off) 时, 所有 6 个 guard helper (`_coverage_sig`/`_is_meta_label`/`_meta_label_name`/`_efl_round_robin`/`_efl_majority`/`_efl_end_of_iter`) 逐字节返回原路径结果
- `classify_failure_type` OFF path = 原关键词路由器 (geometry→foci[5] 的原 bug 行为完整保留, 即"未开启 EFL = 与改前不可区分")
- `select_strengthen_target` OFF/ON vocab 在 seed 时刻 = 同一 39 token 集合 (seed_from_foci 逐字拷贝 _TYPE_COVERAGE_SIG)
- 模块级 `_EFL` 仅在 main() 的 `if args.emergent_facets:` 块内赋值, 耦合点无一处无 guard 的 `_EFL.` 属性访问

### 验证清单

| # | 验证项 | 结果 |
|---|--------|------|
| 1 | `ast.parse` 语法 | ✅ OK (修复了 Step 3 引入的重复 `def find_uncovered_type` IndentationError) |
| 2 | 模块 import, `_EFL=None` | ✅ |
| 3 | 6 guard helper OFF path 逐字节 = 原行为 | ✅ |
| 4 | `classify_failure_type` OFF path = 原关键词路由器 | ✅ |
| 5 | `facet_library.self_test()` 独立 + `--self_test_efl` CLI | ✅ |
| 6 | 全部 `_EFL` 引用有 guard (无无 guard 属性访问) | ✅ |
| 7 | `select_strengthen_target` OFF/ON seed 时 vocab 同 | ✅ |
| 8 | EFL ON path 集成: 几何涌现+累积+promote+guard 路由+roundtrip | ✅ |
| 9 | 两处 `_efl_end_of_iter` 调用 (flat 4297 + facet_tree 3619) + `_EFL._path` 迁移 | ✅ |

### 修改的文件

- `rl/train/verl1/skill_evolution/facet_library.py` (新增, ~567 行) — EFL 模块, 自带 self_test
- `rl/train/verl1/skill_evolution/skill_evolver_aime_v2.py` — Steps 2-7 耦合点全部接通 (guard helpers + classify top guard + 6 耦合点 + SECTION_EXPAND_PROMPT 参数化 + persist/promote 调用 + _path 迁移)

### 不在本次范围 (待真跑确认)

- **真 end-to-end run 未执行**: 完整 `--emergent_facets` 训练需数小时 API eval 调用, 集成测试用 mock LLM 证明的是**机制正确性**(涌现/累积/promote/guard 路由), 非真实 AIME pass-rate 增益。pass-rate delta 需待真跑。
- **ATE 二级验证**: 可选离线因果检查, AIME ~18 train 题样本可能不够, 保留为可选, 不在 emergence gate 里。
- **spike-peak admission**: `classify_item_dim`/`_DIM_TO_PEAK` 仍 index-coupled, 但 `--pool_filter` 默认 OFF (thesis config 不触发), 已知限制。

## 下一步

- 真跑 `python skill_evolver_aime_v2.py --emergent_facets --coverage_diagnosis --iters N --select_lcb` → 看 facet_library.json 是否长出 Geometry + flat path focus 是否命中它 + pass-rate vs flag-off 基线是否提升
- 真跑 `--emergent_facets --facet_tree --expand` → 看 best_skill 里 `## Geometry` 段是否物理生长

## 关联

[[efl-implementation-started-20261013]] [[skillopt-evolver-rootcause-20261013]]
