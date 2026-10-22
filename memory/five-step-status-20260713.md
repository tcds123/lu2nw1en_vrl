---
name: five-step-status-20260713
description: 5 项改造 Step 1-4 实际完成与验证状态 (2026-07-13) - 取代 step1-impl-notes 的陈旧声明
metadata:
  type: project
---

# 5 项改造实现状态 (2026-07-13, 已验证)

本文件记录 5 项改造 (skill-architecture-redesign.md 的 4 步执行计划) 的**实际**代码状态与集成测试结果。取代 `step1-impl-notes.md` 中未经验证的声明 (该文件称 "score_delta 从全 0 变为有值" 与 status-review 矛盾, 实际 402 skill ATE 仍为 0)。

## 总览

| Step | 内容 | 状态 | 验证 |
|------|------|------|------|
| 1a | 检索机制修复 (触发词 k=5 -> 全量 compact 注入) | ✅ 代码完成 | 集成测试: 6754 chars, 50 skill 注入 |
| 1b | Meta-Skill 注入 (选择性忽略引导语) | ✅ 代码完成 | 集成测试: "suggestions"/"ignore" 存在 |
| 2 | Skill 评分闭环 (per-gen 回写) | ✅ 代码完成 | EMA 0.0->0.08 正确, score_delta 未污染 |
| 3 | ADD/UPDATE/DELETE 四操作闭环 | ✅ 代码完成 | pool_compact + parse_decision 测试通过 |
| 4 | Trace 丰富化 (self-debug 多步轨迹) | ✅ 代码完成 | debug_trace 写入 case_info (代码审计) |

**注**: 代码全部落地+单元/集成测试通过, 但**尚未跑端到端实验验证效果** (需 BigCodeBench n=57 + GRPO 重启)。

## Step 1: 检索修复 + Meta-Skill (4 文件对齐)

核心: 从"用触发词替 B 选 5 条 skill"变为"把所有 active skill 的 compact 短描述给 B, 让它自己判断" (对标 Trace2Skill 全量注入 + SkillLens meta-skill)。

- `agent/b_agent/skills.py`: `format_all_active_for_context()` 返回 `Tuple[str, List[Skill]]` (注入文本 + 实际选中子集, 非 402 全集)。YAML trigger 序列化 bug 已修 (`"\n".join(f"  - {t}" ...)`)。
- `agent/b_agent/context.py`: `_default_rules()` = `_meta_skill_guidance()` + `_rules()`。Meta-Skill 引导语: "suggestions not hard rules / MUST IGNORE if irrelevant / prioritize correct code"。
- `agent/b_agent/core.py`: `run()` / `generate_code_with_skills()` / `generate_code_with_self_debug(inject_skills=True)` 全量注入; `skills_used` 记录实际子集。
- `rl/train/verl1/b_agent_interface.py`: 新增 `_build_full_system(generated_prompt)` helper, 三条生成路径全部对齐 (修复了 [[phase2_issues]] 记录的 "Step 1 只在 BAgent 库完成未传导到 GRPO 路径" gap)。

## Step 2: Skill 评分闭环 (观测性 EMA, 非因果)

- `skills.py`: 新增 `online_success_rate` 字段 + `record_outcome(skill_ids, pass_rate, alpha=0.1)` (EMA) + `flush()` (批量落盘)。线程安全 (`_skill_lock`)。
- `custom_reward.py`: 每次 case 评分后调 `B_MODEL.skill_lib.record_outcome(skills_ids_used, pass_rate)`, return 前 `flush()`。
- **关键设计**: `record_outcome` 刻意不碰 `score_delta` (因果 ATE 由 offline `skill_loop`/`prune_eval` 写), 也不递增 `usage_count` (避免污染 prune 门控, 见 [[phase2_issues]] "record_outcome 与 update_score 共用 usage_count" 条目)。
- **方法论提醒**: 全量注入下同一次生成的 co-injected skill 共享 pass_rate, 故 `online_success_rate` 是观测性信号 (success-rate-when-injected), 非因果 ATE。因果结论仍需 offline counterfactual。

## Step 3: ADD/UPDATE/DELETE/NONE 四操作闭环

- 新建 `experience/updater.py` (对标 Trace2Skill ExperienceUpdater)。
- `SkillUpdater.update_pool(candidates)`: 每条候选 vs 现有池 compact -> LLM (DeepSeek-V3.2) 决策 ADD/UPDATE/DELETE/NONE; 批量调和多候选 UPDATE 同一 target; 修改前自动备份 skills 目录; 支持 `--dry_run`。
- CLI: `python -m experience.updater --candidates new.json` 或 `--from_pool experience/experience_pool.json` (增量去重导入)。
- 测试: 402 skill 加载, pool_compact 生成, parse_decision 容错 (valid/fenced/garbage/delete 全过)。**未做真实 LLM 端到端 dry_run** (省 API, 待首次实际蒸馏后跑)。

## Step 4: Trace 丰富化

- `custom_reward.py`: `case_info` 新增 `debug_trace` 字段 (`(self_debug_info or {}).get("debug_trace", [])`), 记录 self-debug 每轮 code/exec_output/action, 供经验蒸馏提取更丰富信号。

## 发现并修复的 2 个 bug (详见 [[phase2_issues]])

1. **Self-Debug 消融 skill 静默未注入**: `generate_code_with_self_debug` 用 `system_prompt.replace("## Rules", ...)` 注入, 但 ablation 的裸 system_prompt 无 "## Rules" anchor -> replace 命中 0 次 -> skill 未注入。**影响**: 7/1~7/3 Self-Debug 消融实际是 "self-debug 无 skill"。已修 (ablation 新增 `_build_full_injection_system` helper 预组装, 传 `inject_skills=False`)。
2. **usage_count 共用污染**: record_outcome 与 update_score 都递增 usage_count -> prune 门控失真。已修 (record_outcome 不再递增)。

## 下一步 (实验验证)

代码已就绪, 待跑实验确认效果:

1. **Step 1 验证**: `python agent/b_agent/ablation_test.py --dataset bigcodebench --n 57` - 对比 Curated(全量注入+Meta-Skill) vs Baseline, 期望 p<0.05 显著优于 baseline (旧 retrieve 路径已知 70% 有害)。
2. **GRPO 重启**: 训练路径现已对齐 Step 1, 重启后 trace 中的 `skills_used` 应为实际注入子集, `online_success_rate` 开始累积。
3. **Step 3 首跑**: 首轮 GRPO 后从新 trace 蒸馏候选, 跑 `updater.py --dry_run` 预览池变化。

## 相关文件

- [[status-review-20260713]] - 诚实评估 (402 skill ATE=0, 70% 有害的根因)
- [[skill-architecture-redesign]] - 5 项改造原始方案
- [[trace2skill-diff-analysis]] - Trace2Skill 对标分析
- [[phase2_issues]] - 本轮 2 个 bug 详细记录
