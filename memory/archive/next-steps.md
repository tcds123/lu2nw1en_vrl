---
name: next-steps
description: 当前阶段的后续研究方向与待办事项 (2026-07-01 重大更新: Self-Debug 突破)
metadata:
  type: project
---

## 当前状态

- GRPO 训练: 停止在 step 637 (6/29), 8 GPU 已被释放, 仅 1 GPU (L40) 被 finetune 占用
- 经验池: 402 条结构化 skill (WHEN/WHAT/Example/WHY, ~1000-1200字/条), G0.md 已补齐 (402/402)
- Skill ATE 评分: 仅 13 个 skill 有非零 score_delta (数据严重不足)
- BigCodeBench Lite Pro: ✅ 已完成 (2026-07-03, n=57 全量)
- 触发词: 1949 个独特短语, LLM 模式覆盖 364/401 条 skill

## 消融实验完整结果

### Phase 1 (原始: OpenCodeInstruct, 无 self-debug 改进)

| 实验 | n | baseline | curated | self_debug | random | p vs baseline | 结论 |
|------|---|----------|---------|------------|--------|---------------|------|
| n=100 k=5 | 100 | 51.3% | 50.8% | — | — | 0.866 | ❌ 无效, 78/100 ties |
| n=50 k=5 | 50 | 57.6% | 64.6% | — | — | 0.059 | ⚠️ 边际显著, 39/50 ties |
| n=30 k=5 | 30 | 5.48 | 6.36 | — | 6.37 | 0.040 | ✅ 显著, 但 curated≈random |
| n=20 k=5 | 20 | 5.57 | 5.27 | — | 5.14 | 0.385 | ❌ curated 比 baseline 差 |
| n=15 k=5 v2 | 15 | 5.69 | 5.93 | 6.05 | 5.64 | 0.065 (sd) | ⚠️ avg_rounds=1.13, 仅1个task有fix |

### Phase 3 (改进后: CodeEval-Pro, LLM 触发词, Self-Debug V2) 🎉

| 数据集 | n | baseline | curated | self_debug | curated p | self_debug p | self_debug 修复 |
|--------|---|----------|---------|------------|-----------|--------------|-----------------|
| **MBPP Pro** | 100 | 7.31 | 7.26 | **8.16** | 0.73 ❌ | **0.0009** ✅✅✅ | 35/100 tasks, avg_rounds=1.6 |
| **Humaneval Pro** | 100 | 6.57 | 6.34 | **7.32** | 0.14 ❌ | **0.0048** ✅✅ | 43/100 tasks, avg_rounds=1.77 |
| APPS_Plus hard | 100 | 0.59 | 0.58 | 0.48 | 0.69 | 0.92 ❌ | 0.39 ❌ | avg_rounds=2.9, fixes=96/100 |

### Phase 4 (BigCodeBench Lite Pro, 2026-07-03) 🔥

| 数据集 | n | Baseline | Curated | Self-Debug | Curated p | SD p | 备注 |
|--------|---|----------|---------|------------|-----------|------|------|
| **BigCodeBench Lite** | 57 | 2.32 | **2.86** | **3.05** | **0.0118** ✅ | **0.0121** ✅ | Curated 首次显著! avg_rounds=2.8, fixes=52/57 |

### 关键发现 (v3, 含 BigCodeBench)

1. **Self-Debug 在 3/4 数据集上显著**: MBPP p=0.0009, Humaneval p=0.0048, **BigCodeBench p=0.0121** — 仅 APPS_Plus Hard 不显著
2. **Curated Skills 首次在 BigCodeBench 上显著 (p=0.0118)** 🔥 — 困难任务中 skill 指南的边际价值更大
3. **Self-Debug 修复参与率**: 35-91% 随数据集难度升高而升高
4. **改进的关键因素**:
   a. 逐条详细测试反馈 (per-test PASS/FAIL + 错误原因)
   b. 正确检测 assertion failures (不是只看 Traceback)
   c. 更明确的修复指令 (分析失败原因 → 修复 → 重新输出)
   d. 切到更难的数据集 (CodeEval-Pro 的改名函数变体提供了真实的调试信号)
5. **难度梯度模式清晰**:
   - 中等 (MBPP/Humaneval): SelfDebug >> Curated ≈ Baseline
   - 困难 (BigCodeBench): SelfDebug ≈ Curated > Baseline
   - 极端 (APPS_Plus Hard): 全部不显著
6. **Self-Debug 天花板**: 无法修复结构性问题 (函数签名重写、算法级重构), 只能修复局部逻辑 bug

## 后续方向 (按优先级, 2026-07-03 更新)

### 方向 0 (最高优先): Self-Debug 子组件消融 🔥
- **Why**: Self-Debug 在 3/4 数据集上显著，需要确定哪些子组件贡献最大
- **Plan**: 见 `memory/selfdebug-ablation-plan.md`
- **What**: 依次关闭 detailed test output / assertion detection / fix prompt
- **数据集**: Humaneval Pro n=50

### 方向 1 (高优先): 论文撰写
- **Why**: 4 数据集结果已足够完整，实验结果足以支撑论文
- **What**: 按照 `paper-outline-v1.md` 结构开始撰写
- **核心卖点**:
  a. Self-Debug 在 3 数据集上 p<0.05 (含一个 p<0.001)
  b. Curated Skills 首次在 BigCodeBench 上显著
  c. 从参数空间到上下文空间的研究范式转移

### 方向 2: 提升 Curated Skills 质量
- **Why**: BigCodeBench 上 Curated 首次显著（p=0.0118），但在 MBPP/Humaneval 上仍不显著 → 需要针对性改进
- **Options**:
  a. 用 SkillOpt 方法 (验证门控 + 文本学习率) 重新蒸馏 experience_pool.json
  b. 用 SkillLens 的 meta-skill 指导提取，减少负迁移
  c. 按 ATE 评分剪枝低效 skill，只保留 top-100
  d. 参考 ReSkill 将 skill 创建嵌入 GRPO 训练循环

### 方向 3: Self-Debug 深度分析
- ✅ BigCodeBench Lite Pro 已完成 (2026-07-03)
- ⏳ 对比不同 max_debug_rounds (1/2/3/5) 的效果
- ⏳ 分析哪些任务类型最能从 self-debug 受益
- ⏳ 统计 B 模型 API 调用增量 (成本效益分析)



### 方向 5: 整合回 GRPO 训练
- 将 self-debug 结果作为 custom_reward.py 的额外 reward signal
- 在 GRPO rollout 中启用 self-debug mode

## 已完成的改进 (2026-07-01)

- [x] YAML 触发词格式修复
- [x] LLM 模式批量生成 364 条 skill 专属触发词 (1949 个独特短语)
- [x] ablation_test.py 支持 6 种数据集 (opencode, humaneval_pro, mbpp_pro, bigcodebench, apps_plus, apps_plus_hard)
- [x] Self-debug: 逐条详细测试反馈 (_run_unit_tests_detailed)
- [x] Self-debug: 正确检测 assertion failures
- [x] Self-debug: 更明确的修复指令

## 参考

- 详细改进方案: `docs/improvement_plan_task1_task5.md`
- 设计文档: `docs/training_free_grpo_fusion_design.md`
- 12 种 Skill 优化方法: `agent/b_agent/reference.md`
- BAgent 架构: `agent/b_agent/ARCHITECTURE.md`
