---
name: p0-focused-injection-20260714
description: P0 focused-injection 消融实验 - 实现/冒烟/启动状态 (2026-07-14)。判别性测试: 注入数量 vs 池质量哪个是绑定约束
metadata:
  type: project
---

# P0 Focused-Injection 消融实验 (2026-07-14)

> 对应导师建议 #1 (SkillsBench 方向) 的前置判别实验 + [[research-direction-20260714]] P0。
> 根因分析见 [[trace2skill-diff-analysis]] (2026-07-14 核验修正块)。

## 实验设计 (判别性测试)

4 组, 单次 `run_ablation` 内 per-task 并行跑, paired t-test:
- **Baseline**: A 的 system prompt, 无 skill
- **Random-3**: 随机 3 条 skill
- **Curated(Full-50)**: 全量 compact 注入 (max 50 条) + Meta-Skill (Step 1 路径)
- **Focused-3**: **B 模型按题语义选 top-3** skill 注入, 格式与 Full-50 完全一致 (Meta-Skill+compact+Rules anchor) -> 只隔离"注入数量/选择"一个变量

**判别逻辑**:
- 若 Focused-3 > Full-50 > Baseline -> GraSP "少量聚焦 > 全量" 成立 -> 注入数量是约束 -> 修注入
- 若 Focused-3 ≈ Full-50 ≈ Baseline (全 null) -> **池质量是绑定约束** -> 任何注入方案无效 -> 转向 SkillsBench target A (用验证过的好 skill 集测我们的**方法**) + 修池质量 (信息丰富化 / 反馈环)
- **预测** (Step 1 全量注入已 null + 根因#1 信息贫瘠是任务结构上限): 大概率 null

## 实现 (agent/b_agent/ablation_test.py)

- `_select_focused_skills_llm(agent, query, skill_lib, k)`: 用 `agent._call_llm` 让 B 模型从 395 active skill 的 compact 目录里选 top-k, 解析 JSON ID 列表; 失败回退 `retrieve(top_k=k)`.
- `_build_focused_injection_system(base, selected)`: 镜像 `_build_full_injection_system` 但只注入 selected.
- **关键 bug 修复**: `Skill.compact` 初始为空, 由 `format_all_active_for_context()` 就地生成 (从 strategy 抽 "What to do")。选择器先调它填 compact, 否则 catalog 退化为 "[Gxxx] Gxxx" 无信息量。
- CLI: `--focused-injection` flag; 结果字段 `focused_*` 写入 JSON + analyze_results 的 P0 FOCUSED-INJECTION 统计块。

## 状态

- ✅ py_compile 通过; B+C API 可达 (B-model PONG; C-model CModelJudge 导入成功)
- ✅ 冒烟 n=3: Focused-3 选择生效 (`LLM picked ['G207','G387','G18'] from 395 active`), 注入+打分+JSON 保存正常, EXIT=0
- ✅ **n=100 mbpp_pro 完成** (2026-07-14, EXIT=0, 运行较预期快)
  - 输出: `agent/b_agent/ablation_results/p0_focused_mbpp_pro_n100.json`
  - 日志: `agent/b_agent/ablation_results/p0_focused_mbpp_pro_n100.log`
  - 命令: `python -m agent.b_agent.ablation_test --dataset mbpp_pro --n 100 --top_k 3 --random_k 3 --no-self-debug --focused-injection --output <above>`
  - ETA ~2.5-4h (~900 API calls, 4 组 × 100 任务)
  - B=DeepSeek-V3.2, C=DeepSeek-V4-Flash, 同 aimpapi.midea.com 端点

## 结果 (2026-07-14 完成, n=100 mbpp_pro)

| 组 | Composite | Pass Rate | C-Judge |
|---|---|---|---|
| Baseline | 7.31 ± 3.25 | 0.71 | 7.86 |
| Random-3 | 7.35 ± 3.20 | 0.71 | 7.98 |
| Curated(Full-50) | 6.97 ± 3.47 | 0.65 | 8.12 |
| Focused-3 | 7.27 ± 3.42 | 0.70 | 7.80 |

**Paired t-tests (composite)**:
- Focused vs Baseline: **p=0.8281 (not sig)** ← Focused-3 ≈ Baseline
- Focused vs Curated(Full-50): p=0.1456 (not sig, trend Focused > Curated)
- Focused vs Random: p=0.6907 (not sig)
- Curated(Full-50) vs Baseline: p=0.0511 (borderline, Full-50 略有害)
- Random vs Baseline: p=0.8554 (not sig)

Win rates: Focused > Baseline 55/100; Focused > Curated 39/100 (Curated 多小胜、Focused 少而大胜 -> 均值反超, paired t 才是正确比较)

### 判别结论: **池质量是绑定约束, 非注入数量/选择**

- **预测命中**: Focused-3 (LLM 语义精选 top-3) 仍 ≈ Baseline (p=0.83), 与 Full-50 / Random 全 null。
- 弱信号: Focused-3 (pass 0.70) 略好于 Full-50 (pass 0.65) -> "少注入"的 harm 小于"全注入", 但**都不优于 baseline** (零正向收益)。
- 即: 即使用最佳语义选择, 注入我们的 skill 也无收益。**绑定约束 = 池质量** (信息贫瘠 + 无反馈环, 见 [[trace2skill-diff-analysis]] 根因#1/#3)。
- 与 Step 1 一致 (全量注入 null p=0.819), 互为印证: 注入侧 (数量/选择/全量/检索) 已穷尽, 均无效。

### 下一步 (按判别框架)
- 注入侧已证伪, 不再调注入参数 (k 值/检索/全量都试过)。
- **转 SkillsBench target A**: 用验证过的好 skill 集 (别人的) 测我们的**方法** -> 隔离"方法是否有效" vs "我们的池是否太差"。对应导师建议 #1 (最优先)。
- 中期: 修池质量 (信息丰富化: 多步轨迹; 反馈环: UPDATE/DELETE, 见 [[skill-architecture-redesign]])。

## 相关
- [[trace2skill-diff-analysis]] - 根因 + 核验修正 + 判别性测试框架
- [[step1-validation-result-20260713]] - Step 1 全量注入 null (p=0.819), 证伪"注入机制是绑定约束"
- [[research-direction-20260714]] - P0->P1->P2->P3 因果链
