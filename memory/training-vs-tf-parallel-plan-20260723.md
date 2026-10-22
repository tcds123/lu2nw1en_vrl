---
name: training-vs-tf-parallel-plan-20260723
description: 训练vs免训练双线并行方案 + 免训练线(tf_skill_optimizer)已实现冒烟通, 阻塞=DeepSeek API key 401失效
metadata:
  type: project
---

# 训练 vs 免训练 Skill 优化双线并行 (2026-07-23)

## 决策
8卡4B GRPO 排队中 + "小模型GRPO是否有效"存疑 → 立**双线并行**:
- **免训练线 (TF-GRPO)**: 零GPU, rewriter=DeepSeek API, 现在本机就能跑。论文里是 named baseline (对标 SePO/Skills-Coach)。
- **训练线 (GRPO-4B)**: 创新点, 等8卡。
- 两线**只差"谁改写skill"**, B/C/evaluator/S_0/s0_cache/split/对位baseline 全共用 → 结果可直接同表比较。

方案文档: `docs/training_vs_training_free_parallel_plan.md` (含算法/复用清单/对位表/三种结局可发表)。

## 免训练线实现 (已完成, 冒烟通过)
- 新增 `rl/train/verl1/tf_skill_optimizer.py` (~280行, 复用 experience/extractor 群组蒸馏 + skillopt chat_target/evaluator + construct_training_prompt):
  - 每题采样 G 个 skill 变体 → B 生成 → LCB单测 → delta=pass-s0_pass
  - 组内好坏对比蒸馏经验 (GROUP_ADVANTAGE_TEMPLATE) → 跨题合并 (CROSS_PROBLEM_MERGE_TEMPLATE) → 合成全局 skill
  - **val 门控** (学 RSEA/GRASP): val delta≥0 才接受新 skill, 防回归
  - 产出 tf_global_skill.md + experience_pool_tf.json + 每轮快照
- 改 `eval_lcb_skillopt.py`: 加 `--skill_source {checkpoint,s0,tf_global}` 开关, tf_global 不需 GPU
- 冒烟 (T=1,g=2,n=2) 跑通: rewriter 产出合理 SePO 风格 skill, 链路全通

## 🚨 阻塞: DeepSeek API key 401 失效
- key `msk-bf2324...` 服务端 "key认证失败", curl 直连两 base_url 均 401
- 非代码 bug (冒烟首轮 key 还有效, 产出过正常 skill)
- 详见 docs/phase2_issues.md 2026-07-23 条
- **key 恢复后**直接重跑: `python tf_skill_optimizer.py --iters 3 --g 4 --train_n 41`

## 下一步 (key 恢复后)
1. TF 线正式跑 T=3 g=4 train_n=41(hard优先)
2. `eval_lcb_skillopt.py --skill_source tf_global` 评 test=143
3. 填并行方案 §5 对位表 (TF vs S_0/SkillOpt/noskill)
4. 8卡到位补训练线最后一行
5. 两者同表比较定论文叙事 (三种结局都可发表)

Related: [[our-grpo-skill-method-design-20260721]], [[sepo-mbpp-reference-20260721]], [[skillcomposer-sepo-verified-20260721]], [[grpo-skill-smoke-passed-20260722]], [[api-concurrency-deepseek-20260716]]
