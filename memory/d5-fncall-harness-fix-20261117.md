---
name: d5-fncall-harness-fix-20261117
description: D5 接口契约修复完成 — 30 道函数签名题建 function-call harness, 7/30 recovered, McNemar 口径统一 215 vs 218
metadata:
  type: project
---

# D5 fn-call harness 修复完成 (2026-11-17)

codex P0 (task #51) 全链路实现 + 验收 + 落盘。承接 [[codex 三 P0 隔离]]。
唯一改动: `rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py` (SkillOpt evaluator 只读 import 未改, 分母 300 不变)。

## bug 根因 (一句话)

300 题 test 集里 30 道 `test_code.stdin` 是函数签名题 (`class Solution`, stdin[0] 是 list)。
SkillOpt `parse_test_pairs` + `truncate_test_code` 对 list 元素 `str()` 化 → `stdin[0]` 变成字符串 `"[13]"`
→ 旧 evaluator 把 Python list-repr 当 stdin 喂 → 这 30 道 S₀ 全 hard=0 (假失败)。影响 test 30 / val 21 / train 9。

## 实现 (路径 B: 0 flash 调用, 重用存储 response)

- `_evaluate_fn_call` (L195-435): per-pair 子进程 `from _d5_solution import *`, timeout=6s/pair, `_d5_extract_sig`/`_d5_coerce_arg`/`_d5_build_call`/`_d5_normalize_strict` (严格保序, True≠1)。
- `_evaluate_d5` dispatch: `ast.literal_eval(stdin[0])` 是 list → fn-call harness; 否则原 `appsplus_evaluate` (stdin 不变)。
- `run_one` (L441) 单行替换调用点。
- `rebuild_fn_call_cache` (L705): Path-B 重评, 写 `out_dir/s0_cache_{val,test}_d5.json` (不覆盖 V2_RUN 原文件), 幂等, hash+`evaluator_version` 绑定。
- main 211 对照 (L1058): SkillOpt 211 也 Path-B D5 重建 → McNemar 两臂同口径。

检测器必须用 `ast.literal_eval` 而非 `isinstance`: `run_one` 拿到的是 `truncate_test_code` 后的 stringified 形态 (`stdin[0]` 是 str), `ast.literal_eval` 恢复 list; 零误报 (2474 stdin 题 0 道解析成 list)。

## 关键结果

| 闸 | 结果 |
|----|------|
| 闸1 NO_SIG 显式 | `fail_reason="no_signature"`, `error_type="NoSignature"`, 不静默 |
| 闸2 normalize 严格保序 | 普通 list 保序不排序, True≠1; 30 道 raw_solution 严格保序 30/30 过 |
| 闸3 子进程隔离 | 每对独立 subprocess, 一对崩溃继续, 无全局状态 |
| 闸4 changed⊆fn-call | test/val/BR 三段全部 stdin-polluted=0 (零 stdin 条目被改) |

| cache | total | fn-call | recovered(0→1) | hard legacy→D5 |
|-------|-------|---------|----------------|----------------|
| test S₀ | 300 | 30 | 7 | 208→215 |
| val S₀ | 100 | 21 | 7 | 58→65 |
| test BR(211) | 300 | 30 | 7 | 211→218 |

**7/30 recovered 非 30**: 30 = raw_solution 上界 (0-API preflight 30/30); 7 = flash 实际生成代码真实恢复。
23 真失败 = 21 `NameError: name 'Solution' is not defined` (flash 写 standalone, 题目要 class Solution) + 2 逻辑错 (2268 pr 0.667, 3458 mismatch)。无真实 Timeout (全 ≤0.19s)。
含义: D5 修正评测让 7 道本就对 (被旧 harness 错杀) 的题恢复; 23 道是真实模型缺陷, 改评测修不了。

## McNemar 口径统一 (step 5)

| 口径 | S₀ | BR(211) | disc | p |
|------|-----|---------|------|---|
| legacy_harness | 208 | 211 | 31 | 0.720 |
| function_call_harness_v1 | 215 | 218 | 33 | 0.728 |

- **+3 增益结构保持** (211−208=3 → 218−215=3): D5 是测量层**对称修正**, 两臂同等 +7, 不制造算法优势 → 三 P0 隔离 (D5 任务测量 vs D2 算法执行) 成立。
- McNemar 仍非显著 (p≈0.73): guardrail "238 只是上界" + "#48 positive ≠ algorithm superior" 成立。
- **历史 231/232 不得任意取**: 都是 legacy_harness, 与 D5 后 218/215 不同口径, 不可连续比较。92-fail 集已重建。

## 口径标记 (不混用)

- D5 前 = `legacy_harness` (208 / 211 / 231-232)。
- D5 后 = `function_call_harness_v1` (215 / 218)。
- cache 条目带 `evaluator_version` 字段可区分。

## 三 P0 隔离 (本次未越界)

D2/#50 (算法执行 Rewriter) ❌ 未动 / D5/#51 (任务测量) ✅ 仅本次 / D1 (归因 Step0) ❌ 未动。
Step0 和 Rewriter 未改 → 修复后结果变化可干净归因到"评测接口修正"。

## 下一步

D5 解锁 #48 (训练时消融, 验证 Evolver 训练机制因果)。
