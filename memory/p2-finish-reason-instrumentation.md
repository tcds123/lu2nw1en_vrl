---
name: p2-finish-reason-instrumentation
description: 复审§13.10优先级#2——补齐response/finish_reason/parse_status的旁路采集函数与分母口径修正
metadata:
  type: project
---

2026-09-24：完成复审 §13.10 优先级 #2 的**代码侧**落地（尚需真实数据跑数）。

**问题**：旧 `run_one` 丢弃 `chat_target` 返回的 usage_info（completion_tokens），无 finish_reason（chat_target 不暴露），无 parse_status（parse_choice_label 的 first-token 回退会把 "A or C" 判 A），k 次投票只留末次 response。旧 `eval_on` 的 mean_pass 分母丢弃 API 全失败题（pass_rate=-3.0，见 [[review-2026-09-24-downgrade]] item8 / §7.1）。历史逐请求 JSON 无法回填（仅存成功比例）。

**解决（旁路新增，不破坏历史复现）**，全部在 `for_review/code/tf_global_v2_math.py`：
- `classify_parse_status(prediction_text, choices, predicted_label)` → `clean_exact` / `ambiguous_first_token`（first-token 回退，"A or C"→A 假阳性源）/ `no_valid_label`。复现 `evaluator.py:parse_choice_label` 的 4 步分支。
- `run_one_logged(...)` → 改用 `chat_target_messages(messages, return_message=True)`，取 `msg.metadata["finish_reason"]` + `msg.content` + `usage_info["completion_tokens"]`，保留 k 次逐采样。
- `eval_on_logged(...)` → **修正分母**：区分两类 pass_rate<0——(a) deadline 超时未跑 delta=-s0p；(b) API 全失败已跑（原 eval_on 静默丢弃）记 0 分入分母、delta=-s0p。落盘逐题完整 JSON 到 out_path。

**校验**：AST OK；模块导入 OK；`chat_target_messages` 签名 + `CompatAssistantMessage.metadata["finish_reason"]` 端到端确认（`qwen_backend.py:229-232`）；mock 三题（正常pass/正常fail/API全失败）单测 mean_pass=1/3、mean_delta=-1/6、n=3 通过——API 失败题已正确入分母。

**How to apply**：后续新 run 改用 `eval_on_logged` 采集完整状态，可统计无标签率/截断率（finish_reason=length 占比）/解析回退率（ambiguous_first_token 占比）。旧 run 不可回填。真实跑数需配 API key + 真实 s0_cache + 选择集，尚未执行。

**Why（不破坏历史复现）**：旧 `run_one`/`eval_on` 是 9/2 0.5542 等历史 run 的复现路径，改逻辑会破坏复现。新函数旁路，旧路径只加标注（见 §6 item8 注释）。分母口径修正只在 `eval_on_logged` 生效。

详见 `docs/phase2_issues.md`（新执行优先序 #2 段，2026-09-24 晚）。问题类 bug 记录同处。
