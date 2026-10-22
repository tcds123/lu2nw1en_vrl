---
name: phaseB-shuffle-unify-fix-20260918
description: Phase B 代码修复完成 — 统一 shuffle 入口 + 防泄漏检查 + cache manifest + CI 双侧 95% + API 失败统一 + 主指标改非 meta，落地 tf_global_v2_math.py + skill_evolver_math.py，全 smoke 通过
metadata:
  type: project
---

# Phase B 代码修复完成（2026-09-18）

## 范围

回应外部 review（`CODE_DOCUMENT_REVIEW_2026-09-17.md` F01-F10 + C6 + RF/CF 两份 review）的 6 类代码缺陷，全部落地到 `tf_global_v2_math.py`（demo 演化器）+ `skill_evolver_math.py`（生产演化器）。review 与 Phase A shuffle verdict 互相印证（P0↔F01 同 `load_split` 根因）。

## 落地的 6 项修复

| 项 | review 编号 | 修复 | 文件 |
|---|---|---|---|
| 统一 shuffle 入口 | F01/P0 | `load_split(split, shuffle_seed)` 注入 `_item_shuffle_seed` 确定性打乱 + remap gold | 两 evolver |
| API 失败统一计 0 | F04 | `eval_on` 中 `pass_rate<0` 进全分母按 0 计，打印 n_api_fail | tf_global |
| 防泄漏检查 | F06/§1.3 | `check_skill_leakage` 8-gram 题面命中 + 位置暗示正则（14 用例全过）；composed 候选 fail→回退 S₀ | tf_global(main gate) + skill_evolver(init/s0 预检) |
| cache manifest | F07 | `build_cache_manifest` 6 字段（k/seed/skill_hash/split_hash/choice_arrangement_hash/n_items），`load_s0_cache_with_manifest` 校验不匹配则重算 | 两 evolver |
| CI 双侧 95% | C6 | `bootstrap_ci_lower` 5%→2.5%（`int(0.025*n_boot)`）；同 import 全生效 | tf_global（两 evolver 共用） |
| 主指标非 meta | — | `tf_global_v2_math.main()` 接受门控改 `(ci_lower>0) and (other_delta>0) and (other_delta>best)` | tf_global(main) |

## 关键设计取舍

- **生产 evolver 的 13 处 facet-tree promotion gate 暂不动**：改接受门控 = 改实验定义 = 使既有 on/off 实验作废。Phase B 只修评测口径（让既有数字更可信），不修实验定义（留 Phase C）。
- **起 skill 防泄漏预检 fail 仅警告不阻断**：init_skill 是外部给定（非本工具产出），阻断会卡死 resume；演化期 composed 候选才强制门控（fail→回退 S₀）。
- **`shuffle_seed=None` 默认向后兼容**：字节复现未打乱口径（历史 0.462/on=0.587 仍可复跑），review 后新跑须显式传 `--shuffle_seed 0`。

## 验证（全 smoke 通过）

- `py_compile` 两文件通过
- 运行时 import 链：manifest loader / leakage check / meta classifier 从生产 evolver 可达
- `--help` 渲染 `--shuffle_seed`
- val shuffle smoke（生产 1-3-6 split）：raw gold 53/53 恒 A → seed=0 打散 {A:12,B:11,C:10,D:10,E:10}，seed=0 字节可复现，seed 0≠1，gold text 重映射无 desync
- 位置暗示正则 14 测试用例全过（含 meta 文本不误判、负向"dont choose A"不误报）

## 仍待办（Phase C/D 范畴，本阶段不做）

- 生产 evolver facet-tree gate 改非 meta delta（改实验定义，需重跑）
- 多 run 量运行间方差（Phase A verdict: 4-seed sd≤0.024 只刻画评测噪声，非算法运行间方差 ~0.15）
- pool_filter 与父本配对、subset vs best、算子计数、无效参数报错（Phase D）

## 关联

[[response-to-review-20260917]]（review 逐条回应）/ [[gpt-review-p0-dataset-shortcut-20260917]]（P0/P1 根因）/ [[phaseA-shuffle-verdict-20260917]]（shuffle 4-seed verdict）/ [[pool-filter-on-off-verdict-20260917]]（被推翻的 +15.6pp）
