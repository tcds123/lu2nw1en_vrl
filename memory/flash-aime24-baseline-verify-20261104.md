---
name: flash-aime24-baseline-verify-20261104
description: 核实 11-02 flash AIME24 基线: 数字全对, 但 0 分题根因 = run_one 假成功 (4096 截断且不计 n_error); 16384 下 63 号仍 0 分因 truncation
metadata:
  type: project
---

# flash AIME24 基线 — 核实 + 0 分题根因 (2026-11-04)

被核对象: [[crossmodel-flash-baseline-result-20261102]] / launch 记录。run: `run_20261102_170450/`。

## 核实结果 (全部 ✅, 无捏造)

- PID 106121 已退出 (ps 空)、run.log 37.8 min / 240 请求 / 零 traceback、文件 mtime 均 11-02 17:42。
- `baseline_flash.json`: all 0.854167 / train 0.895833 / val 0.791667, md5 `c11d7dc0c8` — 与汇报逐字一致。
- 独立重算 `s0_cache_aime.json`: n=30, full=21, zero=2 (aime24:63, aime24:81), mean=0.854167 ✅。
- `eva/crossmodel_flash_baseline.py` 确含 `--isolate_max_tokens/--no_isolate_max_tokens`; run.log 有 "cap 隔离: run_one 内 1 处 4096 → 16384"。
- `docs/phase2_issues.md:5293` 修订节原文对得上。

## 新发现 (汇报里没说、必须记)

**AIME `run_one` (tf_global_v2_aime.py:95) 的 truncation 是"假成功"**:
- `chat_target` 直接返回 `text, usage`, 不解析 `finish_reason`; 截断输出只是"没提答案" → EM=0。
- try 块未抛异常 → `k_err` 不增 → `n_error=0`。
- 1′ 修复的"k 次全异常排除"逻辑 **对 truncation 完全失效**, 截断被静默计 0 分。

**实测 (flash, aime24:63, k=1, EVAL_TEMP=0.3)**:

| cap | completion_tokens | chars | pass | n_error |
|---|---|---|---|---|
| 4096 | **4096 (打满)** | 9312 | 0.0 | **0** |
| 16384 | 5350 (未打满) | 13603 | 0.0 | **0** |

→ 4096 下确凿截断; 16384 下未被 cap 截, 但仍 0 分 (flash 超长推理, 16384 内没写出答案)。
→ **AIME24 上 16384 隔离本身也会引入截断偏置**, 不只是噪声。

**推论**: 若直接用 `eval_aime25_multiseed.py` 跑 flash, 默认 `--eval_k 3` (`run_one` 显式 4096, 无隔离) 
→ flash 会被大面积截断 → 出现"flash 在 AIME25 崩了"的**假饱和/假结论**。必须同 16384 隔离口径。

## 待办 (本轮进行中)

- [x] 实现 `.verify_tmp/flash_aime25_s0.py` (co_consts 打补丁 + 置空被绑定的旧 run_one + 逐次调用 usage 遥测), 复用 `eval_aime25_multiseed.eval_skill_on` 跑 flash S₀@AIME25 k=8 cap=16384 × 2 seeds (后台运行中)
- [ ] 63/81 各 2 轮 k=8 在 16384 下重测 (后台任务) → 区分"cap 导致"还是"真不会"
- [ ] 正式修法 (已写成 `docs/phase2_issues.md` #20): `run_one` 读 env + 解析 finish_reason 记 `n_truncated`, 截断样本单独报
- [ ] 注: `eval_aime24_multiseed` 在 import 时绑定 math 版 `run_one`，修补必须同时置空 `tf_global_v2_aime.run_one` / `tf_global_v2_math.run_one`，`eval_skill_on` 才会走 aime 版

## 关联

[[crossmodel-flash-baseline-result-20261102]] · [[crossmodel-flash-baseline-launch-20261102]] · `docs/phase2_issues.md` #18
