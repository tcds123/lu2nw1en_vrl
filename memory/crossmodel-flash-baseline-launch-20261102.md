---
name: crossmodel-flash-baseline-launch-20261102
description: flash 跨模型 AIME24 基线评测启动 (S0 k=8, 30题, 隔离 max_tokens=16384)
metadata:
  type: project
---

# flash 跨模型 AIME24 基线 — 启动记录 (2026-11-02)

前置: [[cross-model-configs-aimp-20261030]] (flash 接入 = 纯 env 替换, dry-run 已通)。
脚本: `eva/crossmodel_flash_baseline.py` (新增)。
产物: `rl/train/verl1/outputs/crossmodel_baseline/run_<ts>/`。

## 目的 (口径, 已与用户确认)

产出 **flash 在 AIME24 30 题、`skill_aime_initial_v2.md` (599 chars, md5 `c11d7dc0c8`) 下的 pass@k 基线**,
作为对照: 后续把各次 run 演化出的候选 skill 拿到 flash 上 paired 测, 看 skill 收益是否**跨模型泛化**。

⚠️ **跑的不是 evolver 演化过程**。用户消息里写的"3 轮迭代"与 `tf_global_v2_aime.main()` 的实际流程不符
(那是 rw/exp/rewrite 的演化线, 会改写 skill); 用户括号里同时写了 "S0 precompute + val 评测, 纯 EM",
与"基线"一字一致 → 按**基线**执行, 不演化。此偏离已向用户报备。

## 启动参数

| 项 | 值 | 说明 |
|---|---|---|
| provider | `deepseek-v4-flash` | openai-compat 网关 |
| eval_k | **8** | 与既有 rep5 / 测量口径一致 (非 main 默认 3), 检出下限才能对得上 +0.10 |
| n_items | 30 | AIME24 全量, 不切 split |
| EVAL_TEMP | 0.3 | 复用 `tf_global_v2_aime.EVAL_TEMP` |
| max_rpm | 20 | flash rate limit 未知, 保守 |
| **max_tokens** | **16384 (隔离)** | 见下 |
| cand_skill | 无 | 纯 S₀ 基线 |

规模 ~240 请求 (30×8), 按 dry-run 25.9s/请求 @20RPM 估 ~8–20 min。

## 本轮唯一的变量 = 模型; max_tokens 从本轮起隔离为 16384

10-30 的 F7 调查遗留未决: rep5-fixed in-sample 实际 cap 是 4096 还是 16384 **未定**
(`run_one` L119 显式传 4096 字面量; `tf_global_v2_math` 的 setdefault 16384 只影响默认值),
证据需 `finish_reason`/usage completion_tokens, 见 `docs/phase2_issues.md` #18。

→ 本轮**不改 load 路径的代码**, 而是 `QWEN_CHAT_MAX_TOKENS=16384` 让 `config.max_tokens=16384`, 并使新入口脚本
`@run_one` 只替换 `__code__.co_consts` 里的 4096 字面量为 16384 (进程内, 不改文件, 不影响任何别的 run),
于是输出长度天花板由题目决定, 不受 cap 混淆。
**代价**: 与 rep5-fixed 的 V3.2 数字**不是同一 cap 口径**, 跨模型对比时需注明。
后续用 trace 里的 completion_tokens 即可顺带回答 #18。

## 关联

- [[cross-model-configs-aimp-20261030]] — 接入点与 dry-run
- `docs/phase2_issues.md` #18 — F7 in-sample cap 遗留
