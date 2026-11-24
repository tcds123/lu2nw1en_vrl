---
name: flash-aime24-dskills-launch-20261109
description: flash 上 AIME24 5 臂 (none/s0/d1/d2/d3) 截断清零评测启动 — 解 §6 harness-模型混淆
metadata:
  type: project
---

# flash × AIME24 × 5 skill 臂 — 启动记录 (2026-11-09)

前置: `for_review_aime_standalone/docs/EXPERIMENT_REVIEW_20261109.md` §4.5 (截断清零) + §6 (harness vs 模型贡献混淆)。
用户口径 (2026-11-09): "d123 不是仅仅是一段文本吗，文本直接注入呗" —— 复用现有评测脚本，**不新写 harness**。

## 目的

1. **§4.5 首要**: `max_tokens=8192` 的截断 (13%–27%) 清零后，D1/D2/D3 与 none/S0 是否仍有差异。
   预注册: INCOMPLETE (仍有截断) / CEILING (无区分度 → 换 HMMT/Olympiad) / OK (显著配对差)。
2. **§6**: 换学生 (V3.2 → flash, 单次 API 同口径) 后臂排序是否保持 —— 如果换学生后排序翻转,
   则"skill 排名"里混着模型贡献, 之前"D3 差"可能是 V3.2 特有的。

## 实现 (零新代码, 纯 env 替换)

| 项 | 值 |
|---|---|
| 脚本 | `for_review_aime_standalone/scripts/eval_d_skills_on_aime24.py` (原样, 仅修 preflight `/chat` 拼接 bug) |
| 启动 shim | `rl/train/verl1/skill_evolution/launch_flash_aime24_dskills.sh` |
| 模型 | deepseek-v4-flash @ `https://aimpapi.midea.com/t-aigc/llm-openai-api` |
| key | 运行时从 `configs/models_aimp.local.yaml` 读, **不落盘明文** |
| cap | **32768** (--max-tokens 显式 + QWEN_CHAT_MAX_TOKENS 同值, 避免 min() 钳回 16384 默认) |
| temperature | 0.0 |
| RPM / 并发 | 20 / 4 |
| AIGC-USER | 不发送 (flash openai-compat 端点不需要) |
| 臂 | none / s0 (init v2) / d1 (AIME24→25 蒸馏) / d2 (Astra) / d3 (transfer-final) |
| 题 | AIME24 全量 30 |

冒烟 (2 题 × none,d2) 通过: 端点 200, 无截断, `completion_tokens` 1858 / 2084。单题 ping 确认远端应答 `ping`。

## 预期与已知风险

- **flash 在 AIME24 上基线饱和** (k=8 时 21/30 满分, 见 [[crossmodel-flash-baseline-result-20261102]]) →
  CEILING 的概率不低。若 CEILING, 结论是 "AIME24 对 flash 无分辨力", 不能据此否证 skill。
- `model_returned` 恒为 null (网关不回填 model 字段), 只能靠 preflight 的 resolved 行确认模型。

## 产物

- `rl/train/verl1/outputs/flash_aime24_dskills/<ts>/results.json` + `analysis.json` + `run.log`
