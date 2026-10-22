---
name: work-summary-20260722
description: 2026-07-22 工作总结 - GRPO优化skill方法阶段0端到端打通(本机1卡0.6B), 修5阻塞bug, 待8卡正式训练
metadata:
  type: project
---

# 2026-07-22 工作总结

## 核心成果: GRPO 优化 Skill 方法 阶段 0 端到端打通 🔥

本机 1×L40 + Qwen3-0.6B, verl GRPO 1-step 完整跑通:
`✅ Step 1 COMPLETE` + `🏁 [Iter 0] n=20 mean_reward=2.35 delta:+8/0:11/-0 mean_delta=0.327`

整条链路打通: **A(GRPO改写skill) → B(DeepSeek-V3.2生成代码) → C(LCB stdin/stdout单测) → delta pass-rate reward → GRPO更新A**。

方法有效性初证: hard 题 abc380_g, S_0 全错(s0=0), A 改写的 skill 有 2 条让 B 全对 (skill_pass=1.0, delta=1.0, reward=10.0) — **改写 skill 真能让 B 做出 S_0 做不出的题**。

详见 [[grpo-skill-smoke-passed-20260722]]。

## 修复的 5 个阻塞 bug (按定位顺序)
1. **S_0 预计算 backend env**: `precompute_s0_baseline.py` 只设 TARGET_BACKEND 未设 QWEN_CHAT_* → 61题全 Azure 报错。补全 setdefault。重跑 61/61 agent_ok=True, 平均 pass_rate=0.6407。
2. **reward override key**: hydra 应为 `custom_reward_function.path` (顶层), 非 `reward_model.custom_reward_function.path`。改 2 个 train 脚本。
3. **parquet prompt 爆 max_length**: LCB 网格题 80MB test_code 进 A 的 input → sequence_length=105243>4096。`_format_test_code_for_prompt` 改按大小选最小4对+每对截断800字符。
4. **test_code 截断 + delta 可比性** (最关键): py-spy 定位 "Training Progress 0/10 卡住" 实为 reward 卡在 evaluator 跑 80MB test_code (单题最坏246s)。`truncate_test_code(max_pairs=8, max_case_bytes=4096)` 丢弃大用例取最小8对 (80MB→4KB), S_0 预计算 import 同款函数保 delta 可比。
5. **reward 返回类型**: verl `return_dict=True` 期望 dict, 旧代码返回 tuple → `'tuple' has no .get`。改返回 dict。

## 新增/修改文件
- 新增 `memory/grpo-skill-smoke-passed-20260722.md` (里程碑)
- 新增 `memory/gpu-strategy-1card-smoke-8card-prod-20260722.md` (资源策略)
- 新增 `memory/work-summary-20260722.md` (本文件)
- 新增 `rl/train/verl1/check_delta_health.py` (delta 信号健康度检查脚本)
- 新增 `rl/train/verl1/train_lcb_skillopt_1card_smoke.sh` (1卡0.6B冒烟脚本)
- 新增 `docs/lcb_8card_training_guide.md` (8卡训练操作指南)
- 改 `data/OpenCodeInstruct/json2parquet_skillopt.py` (prompt截断 + test_code截断)
- 改 `rl/train/verl1/precompute_s0_baseline.py` (backend env补全 + test_code截断保可比)
- 改 `rl/train/verl1/custom_reward_lcb.py` (return_dict 支持)
- 改 `rl/train/verl1/train_lcb_skillopt.sh` (reward key修正 + GPU监控 + progress抓reward行)
- 改 `rl/train/verl1/train_monitor.py` (清空旧 MANUAL_LOG_FILE, 自动找最新)
- 重生成 `data/OpenCodeInstruct/dataparquet/train_lcb_skillopt.parquet` (截断版)
- 重生成 `rl/train/verl1/outputs/s0_baseline_lcb.json` (截断版, delta可比)
- `docs/phase2_issues.md` 追加 4 条 (80MB test_code / 1卡冒烟卡点 / 真因更正 / S0 backend)
- 更新 `memory/our-grpo-skill-method-design-20260721.md` (阶段0标记完成)

## 清理的过时文件
- `rl/train/verl1/outputs/s0_baseline_lcb.json.fulltc_bak` (旧未截断S0缓存, delta不可比, 删)
- `/tmp/smoke_v*.log`, `/tmp/smoke_1card.log` (冒烟调试临时日志, 删)
- 修正 `memory/gpu-training-platform-20260721.md` "本机无GPU" 过时陈述 (本机现1×L40)
- 修正 MEMORY.md LCB设计条目 "待跑S_0预计算" 过时措辞 (已完成)

## S_0 分层 pass_rate (截断后, delta可比性保证)
easy 0.964 (锚点) / medium 0.705 (主信号源) / hard 0.403 (stretch)。

## 下一步
阶段 1: 申请托管 8×L40 + Qwen3-4B 正式训练。用 `train_lcb_skillopt.sh` (DP=8, max_iter=5, 全量41题天然分布)。操作见 `docs/lcb_8card_training_guide.md`。建议先 max_iter=1 冒烟确认4B+8卡不OOM, 再放大5。训练后 `eval_lcb_skillopt.py` 评 test=143 对位 baseline (noskill 0.4266 / S_0 0.4406 / SkillOpt 0.4336)。

Related: [[grpo-skill-smoke-passed-20260722]], [[gpu-strategy-1card-smoke-8card-prod-20260722]], [[our-grpo-skill-method-design-20260721]], [[gpu-training-platform-20260721]]
