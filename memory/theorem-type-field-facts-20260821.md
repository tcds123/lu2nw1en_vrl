---
name: theorem-type-field-facts-20260821
description: theorem_type 字段实测事实——字段名/路径正确(evolver已用item.get('theorem_type','?')行357/411),但值是字符串list非单值; test值域13类含unknown脏标签(test=3/val=5)+复合类Existential–Universal等; 纠正设计文档虚构的简化值域, failure_modes涌现分簇须先处理unknown+复合类
metadata:
  type: reference
---

# theorem_type 字段实测事实 (2026-08-21)

## 结论
`theorem_type` 字段**真实存在且字段名正确**，但**值是字符串 list（一题可多类，非单值）**，test 值域 **13 类**含 `unknown` 脏标签。设计文档 v3 原写的简化值域「Existence/Implication/Inequality/Universal」**不精确**——漏长尾、漏 `unknown`、把复合类误简化。`failure_modes` 涌现分簇不能直接按字面值做。

## 实测数据（实读 items.json）
- **路径**：`/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42/{train,val,test}/items.json`。manifest: train18/val53/test106 (ratio 1:3:6, seed=42)。
- **结构**：`theorem_type` 是**字符串 list**，实值如 `['Existential–Universal', 'Inequality or Bound']`。
- **test(n=106) 值域 13 类**（主成分计数）：Existence 35 / Implication 34 / `Inequality or Bound` 31 / Universal 30 / `Existential–Universal` 18 / `Biconditional or Equivalence` 14 / `Classification or Bijection` 14 / `Asymptotic or Limit` 8 / `Universal–Existential` 7 / Nonexistence 4 / **unknown 3** / Uniqueness 2 / Equivalence 2。
- **val(n=53)** 另有 Independence or Consistency 等，`unknown`=5。
- **evolver 已用**：`skill_evolver_math.py:357/411` `it.get('theorem_type','?')` 在诊断 prompt 里打印。

## 对 FAILURE_MODES 涌现分簇的含义（开工清单 step1 必验项）
字段名对，但分簇前须处理两件事，否则簇信号不稳定/失效：
1. **`unknown` 脏标签**：test=3、val=5，须剔除或归入回退簇。
2. **复合类**：`Existential–Universal` / `Universal–Existential`（存在性模态混排），须决定是否按「模态顺序」规范化合并（Universal–Existential ↔ Existential–Universal 是否等价）。

## 与既有诊断的关系
`theorem_type` 是 §2.2(b) 涌现机制度的**分簇信号源**（v3 把 evolver 现有 `classify_failure_type` 隐式分簇显式化）。值域错了 → 分簇静默失效 → `failure_modes` 涌现验证（终局 C 组第二条判据）崩。此为问题3订正，正文已落 `facet_tree_skill_evolution_design.md` §2.2(a) + §九-7b。

Related: [[facet-tree-s0-startpoint-20260821]], [[facet-tree-skill-evolution-20260819]], [[eval-signal-debiasing-diagnosis-20260820]]