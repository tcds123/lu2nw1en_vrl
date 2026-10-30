---
name: cross-model-configs-aimp-20261030
description: 跨模型评测的 3 个 AIMP 模型配置已迁入 lunwen/configs/ (key 脱敏入库, 明文本地保留)
---

# 跨模型评测 AIMP 模型配置迁入 (2026-10-30)

来源: `SmartPad/evaluation/cc_switch_compare/config/models_aimp*.yaml` (Midea 智能体项目)
迁入 `lunwen/configs/` 用于 A→B→C 线跨模型 skill 评测。

## 三个模型 (均经 Midea AIGC 网关 aimpapi.midea.com)

| 模型 | 端点类型 | base_url | 用途 |
|---|---|---|---|
| deepseek-v4-flash | openai-compat | `llm-openai-api` (或专有 `aimp-deepseek-v4-flash/v1`) | 候选 B/C |
| deepseek-v4-pro | openai-compat | `llm-openai-api` | 候选 B/C |
| glm-5.2 | AIMP 专有 (需 AIGC-USER header) | `aimp-glm/v1` | 候选 B/C |

三个模型各有独立 key (`AIMP_KEY_FLASH` / `AIMP_KEY_PRO` / `AIMP_KEY_GLM`), 互不相同。

## 文件结构 (key 安全: 脱敏入库 + 明文本地)

| 文件 | key 状态 | git | 用途 |
|---|---|---|---|
| `configs/models_aimp.yaml` | 脱敏 `${AIMP_KEY_*}` | ✅ 入库 | 主配置 (flash 走 openai-compat) |
| `configs/models_aimp_flash_newgw.yaml` | 脱敏 | ✅ 入库 | flash 改用专有网关兜底版 |
| `configs/.env.example` | 占位 | ✅ 入库 | 三个 key 的 env 说明 |
| `configs/models_aimp.local.yaml` | **明文真实 key** | ❌ gitignore | 本地跑评测 |
| `configs/models_aimp_flash_newgw.local.yaml` | **明文真实 key** | ❌ gitignore | 本地跑评测 (专有网关) |

`.gitignore` 规则: `configs/*.local.yaml` + `configs/.env` 排除。`git check-ignore` 实测通过: 脱敏版 tracked / 明文版 ignored。入库文件 msk-key count=0。

## 与之前找到的旧端点对比

`judge/qwen_235B.py:255-266` 保存的旧端点 (注释, key 已 `<REDACTED>`):
- DeepSeek-V3.2 (`aimp-deepseek/v1`) — 当前 B 模型, key = `DEEPSEEK_API_KEY`/`QWEN_CHAT_API_KEY` (同 msk)
- Qwen3-235B-A22B (`aimp-qwen3-235b-a22b/v1`)
- DeepSeek-R1 (`aimp-deepseek-r1/v1`)
- Qwen3-Coder-480B (`f-devops-qwen3-coder-480b/v1`)
- Claude (`mip-chat-app/claude/official`)

**注意**: SmartPad 配置里的 deepseek-v4-flash/pro/glm-5.2 是**更新一代**模型 (V4, 不是旧 Qwen3-235B), key 也与旧 msk 不同。跨模型评测可用模型池 = 新 3 个 + 旧 5 个端点 (旧的需要确认 key 是否仍有效)。

## 授权范围 (CLAUDE.md)

CLAUDE.md 写明"只有 DeepSeek 和 Qwen3-235B 授权"。SmartPad 的 3 个新 key 来自另一项目配置, **用前需确认授权** (用户已指明用这些 → 默认授权成立, 但跨模型评测实际切换模型前仍要问用户)。

## 关联

- [[skillopt-baseline-migration-20261030]] — 同期 baseline 迁移
- [[c-model-and-grpo-reward-status-20261030]] — C 模型状态 (DeepSeek-V4-Flash)
- `configs/models_aimp*.yaml` — 迁入后的配置
- `judge/qwen_235B.py:255-266` — 旧端点切换器
