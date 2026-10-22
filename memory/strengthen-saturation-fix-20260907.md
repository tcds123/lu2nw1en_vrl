---
name: strengthen-saturation-fix-20260907
description: strengthen 饱和判定收紧 (代码层修复, 零 API)。修 expand "更强变体抢先" 设计缺口 — 新增 type_best_delta 状态, 仅严格更弱变体判饱和交 strengthen, 更强/持平继续 expand。已实现+自测全绿。详见 docs/phase2_issues.md L1715-1739。
metadata:
  type: project
---

# strengthen 饱和判定收紧 (2026-09-07, 代码层, 零 API)

## 问题 (用户任务 #1, status-snapshot-20260905 点名的设计缺口)

算子H strengthen run (9/5) 全程未触发。根因: expand admit 一个类型的段后无"已有专段及最佳 delta"状态追踪 → 新变体即使比旧段弱也照常 expand admit (delta 持续涨 iter4 +0.127→iter6 +0.258), 抢先于 strengthen。设计前提 (expand 饱和→target=None→strengthen 接管) 在有限 iter 内不可达。

## 修复 (`skill_evolver_math.py`, 已实现+自测全绿)

1. `type_best_delta` 状态 dict (L1865, `_run_facet_tree` 局部, 非 resume 持久化): key=失败类型, val=该类型历次 expand 采纳段最佳 val_delta。
2. expand 候选评估加饱和判定 (L2284-2296): `if target in type_best_delta and ed < type_best_delta[target]` → 判饱和, break, 交 strengthen; `ed >=` (更强/持平) 不饱和, 正常 admit。
3. admit 后取 max 刷新 (L2304): `type_best_delta[target] = max(get(target,-inf), ed)` — 强变体刷新阈值不降, 弱采纳不封顶未来强变体。
4. `_run_strengthen` helper 抽取 (L1701, ~80行): 两饱和入口 (全类型覆盖 / 类型饱和) 共用。
5. 自测 test (p) (L1147-1167): 5 子例 (弱饱和/持平不饱和/更强不饱和/max刷新/首见不饱和)。

## 关键设计决策

- **严格小于 `<` 而非 `<=`**: 持平不饱和 (允许同等强度重试可能撞更强), 更强不饱和 (继续探索上限)。落用户锚点"坚持探索不半途而废, 仅严格更弱才交 strengthen"。
- **resume 不持久化是 feature**: resume 后空起步 → 每个已覆盖类型获一次 expand 重试再饱和, 保守偏探索, 无正确性风险, 无伪饱和, 无绕过 strengthen。
- **向后兼容 (字节复现)**: 饱和判定 always-on 但产物是诊断事件, 仅 `--strengthen` flag 开时 hand off 给 `_run_strengthen`; flag 关时 falls through 到 skip, 与既有无 strengthen run 行为一致。

## 验证

`python3 skill_evolver_math.py --self_test_facet` 全绿: "✅ facet self-test passed: ... 算子H OK (... saturation-decision: weaker-variant saturates, stronger/equal/first-time explore)"。

## 教训

"X 算子未触发"先查触发前提结构可达性: strengthen 前提是 expand 饱和, 饱和判定依赖段名签名匹配 (脆, `_TYPE_COVERAGE_SIG` 缺词→永不覆盖→永不饱和)。修法不在 `select_strengthen_target` (它只在 target 已 None 后才被调), 在 expand admit 侧加"该类型已有采纳段且新变体回落"的饱和闸。

## 完整记录

见 `docs/phase2_issues.md` L1715-1739 (含根因/方案/向后兼容/教训全细节, 按 CLAUDE.md 硬规则记录)。

## 关联

[[strengthen-operator-run-20260904]] (本次修复的设计缺口来源) / [[status-snapshot-20260905]] (下一步①=本条) / [[same-window-skillopt-paired-ci-20260907]] (同日 task②)
