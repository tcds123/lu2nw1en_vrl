---
name: d2-rewriter-fix-verified-20261116
description: task #50 D2 P0 bug 修复并通过 5 点闭合验收 — rollout_group rewriter 真执行+温度真生效+临界区还原; 231/232 确认为两次独立 run (基础设施修复非研究结果)
metadata:
  type: project
---

# D2 P0 修复闭合验收 (task #50, 2026-11-16)

## 修复了什么

`tf_skill_optimizer_appsplus.py` 的两个独立子 bug (codex D2, issue #74):
1. **TypeError 静默回退**: `rollout_group` 把 `dict(...)` 当第 3 位置参传给 keyword-only `llm` → TypeError → `except Exception: skills[i]=s0` → g 个变体全 = S₀, rewriter 0 次执行。
2. **温度通道断裂**: `llm` 体不传 `temperature` 给 `chat_target`, `variant_temps` 0.7-0.9 静默丢弃, 全跑全局 temp=0。

## 修复方案 (单文件 4 处)

- `llm` 函数体改为**温度临界区 (方案 C)**: `saved_temp=TARGET_CONFIG.temperature` → `with _TEMP_MUTEX: try: _update_config(...); resp,_=chat_target(...); return resp; finally: _update_config(...,saved_temp)`。`threading.Lock` 串行化, `finally` 异常还原。
- `rollout_group` rewriter 循环从 `ThreadPoolExecutor` 改**串行** (逐 variant 顺序调 llm), 保证临界区无并发 chat_target。
- 失败显式计数 (`skill_real`/`fail_reasons`), 0 真实候选抛 `RuntimeError` 中止 (不静默回退 S₀)。
- variant dict 键 `pass` (非 `pass_rate`, run_one 返回 pass_rate, 包成 variant 时改名)。

## 方案 C 并发安全前置 (用户强制条件, 已核实为真)

(1) 临界区无并发访问 TARGET_CONFIG ✓ (2) 改温+调用+还原完整临界区 ✓ (3) try/finally 异常还原 ✓ (4) 保存/还原原值非假设 0.0 ✓ (5) `_update_config` 原地改 config.temperature 并写 env, TARGET_CONFIG 模块级单例 import 冻结 ✓ → 方案 C 成立, 无需 subprocess 隔离。

## 验收 (smoke g=2 qid=6937, deepseek-v4-flash, 3 轮, 最终 clean exit 0 ALL PASS)

| 点 | 结果 |
|---|---|
| 1 真实 API | ✅ n_real=2/2, skill_len 3931/4072 vs S₀ 1796, is_s0=False |
| 2 温度匹配 | ✅ applied 0.7/0.8, saved 0.0, restored 0.0 (D2-debug 逐条) |
| 3 validity | ✅ all_real + all_nonempty |
| 4 下游 eval | ✅ 候选进入 B-eval, pass/hard 有值 (run3 pass=[0,0] 属 flash B-eval ±1 噪声, 关键是候选**进入了** eval 非 S₀ 覆盖) |
| 5 临界区还原 | ✅ TARGET_CONFIG.temperature==0.0 |

## 证据 (供未来回追 11-13 历史 run)

- 修复前 sha256 = `fd95a877cf0787a8307e14cc603d0a825ac4bc4a6061620bf60ab1e7158cd650`
- 修复后 sha256 = `864cd0245e0c0d260604299d66c54d6f4296dd2d1af74fe4a37719474046e8aa` (py_compile OK)
- smoke sha256 = `fbdd1c43...`; launch shim sha256 = `2588724a...`
- 请求日志 `_smoke_d2_out/smoke_d2_request.log` (温度轨迹); 候选记录 `_smoke_d2_out/smoke_d2_candidates.json`

## 两道 even-if 护栏 (用户强制)

1. **#50 修复成功 ≠ 11-13 历史 Evolver 有效**: 只证明今天若跑 #48 候选生成会真执行。历史 run 源码不可回追 (untracked + mtime 在 run 后 ~3h), codex §9 "证据不足, 不追认旧运行" 不变。
2. **#48 若出正结果 ≠ 算法优越**: 须与等预算 S₀ 重试 + 独立隐藏测试选择机制对照, 不因"修好 + delta 正"断言 evolver 优越。

## 231 vs 232 确认 (issue #77, 落盘于本任务)

两数都是正确的, 是**两次独立真实 Evolver run** (非抄录笔误):
- formal run `run_20261113_101637`: routed **231**/300, rescue 23, 无 retention_analysis, best_skill 3567 字符 — codex 评审引用此 run。
- retention run `run_20261113_152603`: routed **232**/300, rescue 24, 有 retention_analysis (preserved 185/regress 23/retention 88.94%), best_skill 3009 字符 — 11-16 reframe 引用此 run。
- 差 1 题 = flash temp=0 ±1 噪声; 两 run best skill 不同 → 两轮独立进化非同一 skill 两次评测。
- 撤回"全量 232→231 改写"计划 — 强行统一会抹掉"两轮独立运行"事实。论文须标明引用哪次 run, 不在两数间随意取一。

## 定位与下一步

task #50 = **基础设施修复, 非研究结果**。验收通过 → 解锁 **D5 接口契约 harness 修复** (30 道函数签名题建 function-call harness, 保留 stdin/stdout 题, 重跑 S₀ 重建 pass/fail 集, 不改分母 300→270) → 再解锁 **#48 训练时消融**。

关联: [[appsplus-codex-review-crossref-20261116]] [[appsplus-routing-oracle-leakage-reframe-20261116]] [[appsplus-attribution-decompose-26vs5-20261116]]。issue: `docs/phase2_issues_202611.md` #74-续 / #77。
