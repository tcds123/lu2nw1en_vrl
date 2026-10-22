# 训练 vs 免训练：Skill 优化双线并行方案

> 建立于 2026-07-23。背景：8 卡 4B GRPO 任务仍在排队，且「小模型 GRPO 是否有效」存疑（长路径噪声奖励 + 41 题过拟合风险 + 免训练基线 SePO 已证 MBPP +5pp）。
> 本文档定义**两条线并行**：免训练线现在本机就跑（零 GPU），训练线等 8 卡到位再跑。两者**共用同一套 B/C/evaluator/数据/对位 baseline**，结果可直接同表比较。
> 相关：[[our-grpo-skill-method-design-20260721]]、[[sepo-mbpp-reference-20260721]]、[[skillcomposer-sepo-verified-20260721]]、[[grpo-skill-smoke-passed-20260722]]、`docs/training_free_grpo_fusion_design.md`。

---

## 0. 为什么要并行（一句话）

- **免训练线**回答「不更新参数、只用 LLM 群组对比 + 执行反馈改写 skill，能不能超过 S_0」——这是你论文无论如何都要跑的 named baseline（SePO/Skills-Coach），且零 GPU，排队期间就能出结果。
- **训练线**回答「GRPO 把 skill 改写内化进 4B 参数，能不能再上一个台阶」——这是创新点，但依赖 8 卡。
- 两条线**对位同一个 test=143、同一份 S_0、同一套执行器**，谁的 delta 大、谁在 hard 题上有收益，一目了然。三种结局都可发表（见 §6）。

关键：**免训练线不是训练线的退化版，而是独立的方法 + 强 baseline**。SePO 用进化搜索 + 强模型在 MBPP 已 +5pp，4B+GRPO 必须明显超过它才值得讲"训练"。

---

## 1. 两条线的共享层与差异层

### 1.1 共享（两条线完全一致，保证可比）

| 组件 | 内容 | 文件 |
|------|------|------|
| 初始 skill S_0 | `skillopt/envs/lcb/skills/initial.md` | `/data/zhuldz/skillopt/.../initial.md` |
| 数据 split | LCB 2-1-7 seed42（train 41 / val / test 143，无污染 NEW） | `_generated_splits/lcb_2-1-7_seed42` |
| B 模型 | DeepSeek-V3.2 via aimpapi（skillopt chat_target，单次生成） | `skillopt.model.chat_target` |
| 执行器 | LCB stdin/stdout 子进程单测 | `skillopt.envs.lcb.evaluator.evaluate` |
| 测试用例截断 | `truncate_test_code(max_pairs=8, max_case_bytes=4096)` 保 delta 可比 | `json2parquet_skillopt.py` |
| S_0 baseline | 离线预计算每题 s0_pass_rate（61题缓存） | `outputs/s0_baseline_lcb.json` |
| 评测脚本 | test=143 分层 pass-rate | `rl/train/verl1/eval_lcb_skillopt.py` |
| A 模型输入 prompt 构造 | `construct_training_prompt(S_0, question, test_code)` | `json2parquet_skillopt.py:39` |
| 对位 baseline | S_0=0.4406(hard0.193) / SkillOpt=0.4336 / noskill=0.4266 | `eval_lcb_skillopt.py` baselines |
| API 并发约束 | ≤28 RPM，token bucket + 429 退避 | `[[api-concurrency-deepseek-20260716]]` |

**核心 reward 信号（两条线都用）**：`delta = skill_pass_rate − s0_pass_rate`（有 skill vs 无 skill 的 pass-rate 差，隔离 skill 贡献，呼应综述 D2Skill 双 rollout）。

### 1.2 差异（只有"谁改写 skill + 怎么改写"这一步不同）

| 维度 | 训练线 (GRPO) | 免训练线 (TF-GRPO) |
|------|--------------|---------------------|
| 改写主体 | Qwen3-4B A 模型（GRPO 更新参数） | DeepSeek-V3.2 当 rewriter（冻结，仅 API） |
| 改写机制 | 模型前向 + RL 梯度回传 | LLM 群组相对对比（Group Advantage）+ 执行反馈 |
| 优化信号 | delta reward → GRPO advantage | 好坏轨迹对比 → 蒸馏成经验 G# → 注入 rewriter prompt |
| 是否更新参数 | ✅ 是（A 模型权重） | ❌ 否（只改 S_0 文本 + 经验池） |
| GPU 需求 | 8×L40（A 模型训练 + 评测 vLLM） | 0（全 API） |
| 产出物 | A 模型 checkpoint | 改写后的 skill 文件 + experience_pool 增量 |
| 论文角色 | 创新点（我们的方法） | 强 baseline（对标 SePO/Skills-Coach） |

---

## 2. 免训练线设计（TF-GRPO for Skill）

### 2.1 算法（复用 `experience/extractor.py` 的群组蒸馏思想 + `training_free_grpo_fusion_design.md`）

```
输入: S_0, train set Q_train (41题)
S ← S_0                       # 当前 skill，从 S_0 起步
for iter t = 1..T:            # 离线迭代，无梯度
  for each q in Q_train:
    生成 G 个 skill 变体 {S_q^(g)} ← Rewriter(S, q, 经验池)   # skillopt optimizer rollout 风格
    对每个变体: B 模型生成代码 → 单测 → skill_pass(S_q^(g), q)
    delta_q^(g) = skill_pass - s0_pass[q]
  组内排序: 高 delta vs 低 delta 轨迹对比
  → LLM 蒸馏本轮经验 G# (Group Advantage Template, experience/templates.py)
  跨题合并去重 → 更新 experience_pool
  S ← Rewriter(S_0, 经验池) 合成新一轮全局 skill   # 或 per-task online 模式
  # 门控: held-out val 上 delta ≥ 0 才接受 S（学 RSEA/GRASP 防回归）
评测: test=143 跑 eval_lcb_skillopt.py（skill_source 改为 TF 产出的 skill）
```

两种 skill 产出模式（对照 SkillComposer 发现「per-task online 才对代码题有效，通用 library 无效」）：
- **(A) Online per-task**：每道 test 题单独 Rewriter(S_0, q, 经验池) → 该题专属 skill。这是 SkillComposer 在 LCB 唯一有效的模式。
- **(B) Global evolved S**：把蒸馏经验合并进 S_0 得到一个全局改写 skill，所有题共用。对标 SePO evolved prompt。
- 两者都跑，比较哪个强——这本身是一个消融。

### 2.2 复用现有模块（几乎零新代码）

| 现有模块 | 免训练线用途 | 是否需改 |
|----------|-------------|---------|
| `experience/extractor.py` TraceBasedExtractor | 组内对比 + 跨题合并蒸馏经验 | 复用，trace 来源从 GRPO log 换成本线 rollout log |
| `experience/templates.py` GROUP_ADVANTAGE_TEMPLATE | 好坏轨迹对比蒸馏 | 直接复用 |
| `skillopt.model.chat_target` | rewriter（OPTIMIZER_BACKEND）+ B 模型（TARGET_BACKEND） | 复用 |
| `skillopt.envs.lcb.rollout._build_system/_build_user` | skill→system prompt | 复用 |
| `skillopt.envs.lcb.evaluator.evaluate` | 单测 | 复用 |
| `precompute_s0_baseline.py` | s0_pass cache（已有 61 题） | 复用缓存 |
| `eval_lcb_skillopt.py` | test=143 评测 | 加一个 `--skill_source {s0,tf_global,tf_pertask,checkpoint}` 开关 |

**唯一新代码**：一个 `tf_skill_optimizer.py` 编排器（~200 行），把上面模块串起来。见 §4。

### 2.3 与 SePO 的差异（论文 baseline 立得住）

| 维度 | SePO | 我们的 TF 线 |
|------|------|-------------|
| 优化算法 | 开放式进化搜索 + candidate archive | 群组相对对比（GRPO 风格 advantage）+ 执行反馈蒸馏 |
| 信号 | LLM judge 判任务相似度 + 准确率 | delta pass-rate（有/无 skill 差）+ 单测 |
| 自指 | prompt agent 优化自己的 prompt | rewriter 自身 prompt 固定，但经验池跨轮累积 |
| 数据 | MBPP 474/500 + 多任务池 | LCB 无污染 NEW |

差异化：SePO 用进化搜索，我们用 GRPO 风格群组对比；SePO 在 MBPP（baseline 91 天花板），我们在 LCB（baseline 0.44，headroom 大且无污染）。

---

## 3. 训练线设计（已有，仅存档对位）

详见 [[our-grpo-skill-method-design-20260721]] 和 [[grpo-skill-smoke-passed-20260722]]。阶段 0 端到端已打通（1卡0.6B 1-step）。

- 训练：`train_lcb_skillopt.sh`（8×L40, Qwen3-4B, max_iter=5, val 监控过拟合），操作见 `docs/lcb_8card_training_guide.md`。
- 评测：`eval_lcb_skillopt.py --a_model <ckpt>`。
- reward：`custom_reward_lcb.py`（delta pass-rate + 退化检测 floor）。
- **过拟合护栏**：train 仅 41 题，每 iter 监控 val delta；val 涨 test 不涨即停（SkillOpt 前车之鉴）。

---

## 4. 免训练线实现清单（本次要做的）

新增文件 `rl/train/verl1/tf_skill_optimizer.py`：

```
main():
  load S_0, s0_cache, train items (41), val items
  experience_pool = [] (或读现有 experience_pool.json 增量)
  S_global = S_0
  for t in 1..T:
    # 1. rollout: 每题 G 个变体
    variants = rewriter.generate(S_global, q, experience_pool, G=4)
    deltas = [evaluate(skill=v, q) - s0_pass[q] for v in variants]
    # 2. 组内蒸馏经验 (GROUP_ADVANTAGE_TEMPLATE)
    new_exp = extractor.group_advantage(q, rollouts)
    experience_pool.merge(new_exp)
    # 3. 合成全局 skill (CROSS_PROBLEM_MERGE_TEMPLATE)
    S_global_new = rewriter.compose(S_0, experience_pool)
    # 4. 门控: val 上 delta >= 0 才接受 (学 RSEA/GRASP)
    if val_delta(S_global_new) >= val_delta(S_global): S_global = S_global_new
    save S_global_t, experience_pool_t
  # 产出: tf_global_skill.md, experience_pool_tf.json, per-task skills (online 模式)
```

`eval_lcb_skillopt.py` 加开关：
- `--skill_source s0`：基线（已有）
- `--skill_source tf_global`：用 TF 全局 skill
- `--skill_source tf_pertask`：每题用 TF online skill（rewriter 现场生成）
- `--skill_source checkpoint`：训练线 A 模型（已有 `--a_model`）

---

## 5. 对位比较表（论文主表雏形）

test=143 上，同一 split、同一执行器：

| 方法 | 类型 | GPU | overall pass-rate | hard | Δ vs S_0 |
|------|------|-----|-------------------|------|----------|
| noskill | 下界 | 0 | 0.4266 | 0.181 | −0.014 |
| S_0 (initial) | 基线 | 0 | 0.4406 | 0.193 | 0 |
| SkillOpt (别人) | 免训练 baseline | 0 | 0.4336 | 0.193 | −0.007 |
| **SePO 风格** (可选复现) | 免训练 baseline | 0 | ? | ? | ? |
| **TF-GRPO global** (本线 B) | 免训练（我们） | 0 | ? | ? | ? |
| **TF-GRPO per-task** (本线 A) | 免训练（我们） | 0 | ? | ? | ? |
| **GRPO-4B** (训练线) | 训练（创新点） | 8×L40 | ? | ? | ? |

成功判据（按优先级）：
1. TF 线 overall > S_0 且 hard > 0.193（免训练方法本身成立）
2. GRPO-4B > TF 线（训练值得，创新点成立）
3. 两者都在 hard 题上有正 delta（hard 是 headroom 所在，baseline 0.193）

---

## 6. 三种结局都可发表

| 结局 | 故事 | 论文定位 |
|------|------|---------|
| GRPO-4B ≫ TF | RL 内化 skill 改写 > 进化搜索，尤其 hard 题 | 「训练值得」强故事 |
| TF ≈ GRPO-4B | 免训练已足够，小模型 RL 冗余 | 「免训练 skill 优化」实用方法 + 诚实结论 |
| TF > GRPO-4B | 4B 太小/41 题过拟合，免训练更稳 | 关于"小模型 RL 用于 prompt 优化"的 negative-but-honest 结果 |

不管哪种，都有完整、可辩护、不依赖 8 卡的结果。

---

## 7. 执行顺序

1. ✅ 写本文档（本步）
2. ⏳ 实现 `tf_skill_optimizer.py`（复用 extractor + chat_target + evaluator）
3. ⏳ 本机跑 TF 线：先 T=1, G=4, train 41 题冒烟，看 delta 信号
4. ⏳ 跑 TF 线 full（T=3~5），产出 global + per-task skill
5. ⏳ `eval_lcb_skillopt.py --skill_source tf_*` 评 test=143，填 §5 表
6. ⏳ 写 `memory/tf-grpo-skill-result-<date>.md` 记录结果（遵守 CLAUDE.md 硬性规则）
7. ⏳ 8 卡到位后跑训练线，补 §5 表最后一行
8. ⏳ 两者同表比较，定论文叙事
