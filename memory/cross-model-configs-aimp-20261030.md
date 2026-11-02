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

## Smoke test (2026-11-02)

`eva/smoke_flash.py` 验证 deepseek-v4-flash key 可用:
- 端点 `openai-completions` / `base_url=.../llm-openai-api` / `model=deepseek-v4-flash` / `extra_body.thinking={type:disabled}`
- openai 2.7.1 SDK, key 从 `configs/models_aimp.local.yaml` 读 (不硬编码)
- 结果: HTTP 200, 17.4s, content='PONG' ✅
- 用户已选定 flash 作为跨模型评测模型 (2026-11-02)

**glm-5.2 smoke 未做** (api=aimp 专有, 需 AIGC-USER header, 本脚本只测 openai-completions)。
**pro smoke 未做** (用户只选了 flash)。

## 接入点 (跨模型评测真正切 B 模型的地方) — 2026-11-02 摸清

**关键结论: flash 接入 = 纯 env 替换, 零代码改动。**

后端实现在 `baselines/skillopt/skillopt/model/qwen_backend.py`:
- 裸 `requests.post` + `Authorization: Bearer` + 可选 `AIGC-USER` header (L241-247)
- URL 构造 `{base_url}/chat/completions` (L175-178), 通用 OpenAI 兼容, 对 `aimp-deepseek/v1` 无特殊假设
- `deployment` (传给 API 的 `model` 字段) 来自 `QWEN_CHAT_MODEL` env (L95-96)
- DeepSeek-V3.2 默认 env 在 `tf_global_v2_math.py:38-50` setdefault

**flash 接入 env** (替换 DeepSeek-V3.2 的值):
```
QWEN_CHAT_API_KEY=<flash key from configs/models_aimp.local.yaml>
QWEN_CHAT_BASE_URL=https://aimpapi.midea.com/t-aigc/llm-openai-api   # V3.2 是 aimp-deepseek/v1
QWEN_CHAT_MODEL=deepseek-v4-flash                                    # V3.2 是 /model/DeepSeek-V3.2
```

**评测入口**: `rl/train/verl1/skill_evolution/tf_global_v2_aime.py main()`
- S0=`data/skill_aime_initial.md`(L47, ⚠️ 与 memory 记录 v2 用 `skill_aime_initial_v2.md` 不一致, 跑前需核实)
- data=`data/aime24.jsonl`, train_n=18(val 12), iters=3, g=4, num_exp=3, eval_k=3
- EM pass_rate, 不走 C 模型

**全量规模估算** (smoke 测 flash 17.4s/请求):
- S₀ precompute: 30题×3采样=90请求≈26min
- 3轮迭代每轮 ~58请求≈17min ×3=51min
- 全量 ~280 请求 ≈ 1.5-2 小时

**未验证风险 (跑全量前需 dry-run 确认)**:
1. flash rate limit 未知。SmartPad cc_switch_compare `concurrency:1` (串行, "避免压垮 AIMP 网关"), 保守应低 RPM。qwen_backend 默认 28 RPM (V3.2 的 30 上限), flash 上限未知。
2. `thinking` 字段: smoke 用 `extra_body.thinking={type:disabled}`, qwen_backend 发的是 `chat_template_kwargs.enable_thinking=false` (L294-295), flash 是否认这个字段、是否需要改 → dry-run 验证。
3. 17.4s/请求是简单 "PONG"; AIME 长 prompt + 16384 max_tokens 会更慢, 可能触发 timeout (qwen_backend 默认 300s)。
4. AIGC-USER header: V3.2 设了 `zhuldz`, flash 走 openai-compat 端点可能不需要/不认。smoke (OpenAI SDK) 没传也 200 了, 说明 flash openai-compat 不需此 header。

**dry-run 建议**: 1 题 k=1, 走完整 `run_one` (build_system/user → chat_target → evaluate_aime EM 解析), 确认链路通 + 无字段报错, 再谈全量。

## 关联

- [[skillopt-baseline-migration-20261030]] — 同期 baseline 迁移
- [[c-model-and-grpo-reward-status-20261030]] — C 模型状态 (当前主线不走 C, 纯 EM)
- `configs/models_aimp*.yaml` — 迁入后的配置
- `eva/smoke_flash.py` — flash key 冒烟脚本
- `judge/qwen_235B.py:255-266` — 旧端点切换器
