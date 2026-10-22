---
name: our-grpo-skill-method-design-20260721
description: 我们的方法设计 - GRPO优化初始skill S_0 (而非从零生成), delta pass-rate奖励隔离skill贡献, 对标SkillOpt/SePO
metadata:
  type: project
---

# 我们的方法设计: GRPO 优化初始 Skill (2026-07-21)

## 创新点定位
[[code-skill-method-fix-task-20260721]] [[skillcomposer-sepo-verified-20260721]] 已确认: 论文创新点 = **优化 skill 的方法**。别人方法 (SkillOpt/SkillComposer/SePO) 都是 baseline。我们用 **GRPO** (第三条路, 差异化)。

## 核心改造: 从"从零生成 system prompt"改为"优化初始 skill S_0"

### 现状 (rl/train/verl1/a_model_full.py:136 _build_prompt_template)
- A 模型输入: ICL 示例 + 问题 + 参考解 → **从零生成** system prompt
- 问题: 无起点, 优化空间发散; 与"优化已有 skill"的论文叙事不符

### 我们的方法
- A 模型输入: **初始 skill S_0** (skillopt/envs/lcb/skills/initial.md) + 问题 (+参考解) + 失败模式反馈
- A 模型输出: **改写后的 skill** (在 S_0 基础上改进, 针对失败模式加程序性指令, 学 SePO evolved prompt)
- B 模型用改写后 skill 生成代码 → C+单测 → reward
- **关键: reward = delta pass rate = pass(skill_改写) - pass(S_0)**, 隔离 skill 贡献 (呼应对综述 D2Skill 双 rollout, [[skillcomposer-sepo-verified-20260721]] SkillComposer 也用 delta pass rate)

## 为什么 delta pass rate 是关键
- SePO/SkillComposer 都用"有 skill vs 无 skill"或"改写前后"的差作信号, 而非绝对 pass rate
- 绝对 pass rate 混杂 B 模型能力, 信号噪声高 (见 [[sepo-mbpp-reference-20260721]] §4.1)
- delta 隔离 skill 贡献: 只有当改写 skill 比 S_0 真的更好时才给正奖励
- 这也避免 A 模型走捷径 (如生成与 S_0 无关的通用 prompt 拿基础分)

## 方法对照 (我们 vs baseline)
| 方法 | 优化算法 | 信号 | skill 起点 | per-task? |
|------|---------|------|-----------|-----------|
| SkillOpt (已跑, 负) | LLM 启发式 rollout-reflect | 绝对 pass | S_0 | 否(通用) |
| SePO | 进化搜索+archive | LLM judge+准确率 | seed prompt | 否(通用)+多任务池 |
| SkillComposer | rejection sampling 训模型 | delta pass rate | per-task create | 是(online) |
| **我们** | **GRPO** | **delta pass rate** | **S_0** | **可per-task** |

差异化: GRPO (RL) + delta pass rate + 从 S_0 优化。与 SkillComposer (rejection sampling, 非RL) 和 SePO (进化搜索, 非RL) 都不同。

## 实现计划 (最小可行, 接现有 GRPO 链路)

### 进度 (2026-07-21)
- ✅ **改造1 数据脚本完成**: `data/OpenCodeInstruct/json2parquet_skillopt.py`。产出 `train_lcb_skillopt.parquet` (61题: train41+val20, hard25/medium22/easy14)。schema 与现有 exp parquet 对齐 (input/raw_input/output/id + difficulty, 已验证 ndarray of dict 与 verl 兼容)。LCB raw_solution 全空 → prompt 里用 test_code (前4对 I/O) 替代"参考解"角色。S_0 = lcb initial.md (1796 chars)。
- 🚨 **改造1 发现执行器不匹配 (关键障碍, 见 docs/phase2_issues.md 2026-07-21 条)**: custom_reward.py 的 `_unsafe_exec_worker` 是函数式 assert/exec 执行器, **无法跑 LCB stdin/stdout 竞赛题**。改造2 不能只加 delta, 必须替换执行器。**复用 skillopt lcb evaluator** (`skillopt/envs/lcb/evaluator.py`), 不重写。
- ✅ **改造2 reward 完成 (新建 LCB 专用文件, 用户确认架构)**: `rl/train/verl1/custom_reward_lcb.py`。核心: A输出改写skill → `_build_system(skill)` 包成 system → skillopt `chat_target` 调 B(DeepSeek-V3.2, **不经 BAgent skill curation**, 避免 402 skill 污染 per-task 信号) → `lcb_evaluate` (stdin/stdout) → skill_pass。reward = delta = skill_pass - s0_pass。退化检测 sim>S_0→floor。已验证: 模块加载/qid反查/退化检测/delta映射/clamp 全通过。
- ✅ **S_0 离线预计算脚本完成**: `rl/train/verl1/precompute_s0_baseline.py`。用 S_0 跑全 train+val 题, 存 `outputs/s0_baseline_lcb.json`。消除在线 API 翻倍 (待决策项1: 先离线)。
- ✅ **训练入口脚本完成**: `rl/train/verl1/train_lcb_skillopt.sh`。指向新 parquet + 新 reward (config override `reward_model.custom_reward_function.path`)。batch=8/n=5 (LCB 41题量级), 设好 skillopt API env + `TARGET_BACKEND=qwen_chat`。
- ✅ **改造3 评测脚本完成**: `rl/train/verl1/eval_lcb_skillopt.py`。训练后 A 模型(vLLM) 对 test=143 每题生成改写 skill → skillopt B+单测 → 分层统计, 对位 baseline 0.4406/SkillOpt 0.4336/noskill 0.4266。
- ⚠️ **关键坑 (已修)**: skillopt `chat_target` 默认走 Azure OpenAI backend (import 时读 `TARGET_BACKEND` env, 默认 openai_chat)。必须在 import skillopt.model 前设 `TARGET_BACKEND=qwen_chat` (与 run_code.sh 一致)。已在 3 个脚本 + train_lcb_skillopt.sh 顶部 `os.environ.setdefault` 修复。首次预计算因未设此 env 全部 agent_ok=False, 修复后待重跑。
- ⏳ **待跑**: (1) 重跑 precompute_s0_baseline.py 生成正确 cache; (2) 前置核实 (verl 链路 1 step 冒烟, 训练机 API 可达); (3) 上 8×L40 跑 train_lcb_skillopt.sh; (4) 跑 eval_lcb_skillopt.py 对位。

### 进度更新 (2026-07-22)
- ✅ **S_0 预计算重跑成功**: 修复 precompute_s0_baseline.py 的 backend env bug (只设 TARGET_BACKEND 未设 QWEN_CHAT_* → 全报 Azure 错误; 已补全 setdefault)。重跑 61/61 全 agent_ok=True, 0 失败。平均 pass_rate=0.6353 (注意这是 train+val 61题, 非 test=143 的 0.4406, train/val 分布偏易)。分层: easy=1.0(14题天花板无信号) / medium=0.69(22题主信号) / hard=0.38(25题stretch)。delta reward 信号可行性确认: medium 是 GRPO 主学习区间, easy≈0 锚点, hard headroom 大但稀疏。
- ✅ **reward 对齐链路验证**: qid 反查 (_find_qid_from_input) ✅, parquet raw_input/output/id 结构 ✅, TASK_POOL 61题 / S0_BASELINE 61题加载 ✅。轻微隐患: PROMPT_TO_QID 60 keys (61题有1-2题 question 前100字符归一化撞key), 不影响冒烟。
- ✅ **GPU 资源策略澄清** ([[gpu-strategy-1card-smoke-8card-prod-20260722]]): 本机现可用 1×L40 (Qwen3-0.6B 冒烟), 托管 8×L40 (Qwen3-4B 正式)。S0 预计算与 A 模型无关, 两规模共用同一份 s0 cache。
- ✅ **1 卡冒烟脚本创建**: train_lcb_skillopt_1card_smoke.sh (n_gpus=1, dp=1, batch=4, n=5, max_iter=1)。
- ✅ **0.2 reward 端到端冒烟通过**: 手造改写 skill 喂 _generate_and_eval (abc397_d), B API + LCB stdin/stdout 单测 + delta + 退化检测全算通 (skill_pass=0.977 s0=0.977 delta=0 reward=0)。qid 反查 3 种 verl 传参结构均 ✅。
- ✅ **0.3 verl 1-step 冒烟通过 (本机 1卡 0.6B)**: `Step 1 COMPLETE` + `🏁 [Iter 0] n=20 mean_reward=2.35 delta:+8/0:11/-0 mean_delta=0.327`。全链路打通 (A改写skill→B生成代码→LCB单测→delta reward→GRPO更新)。hard题 delta>0 信号清晰 (abc380_g S_0=0→改写skill让B全对 reward=10)。修4阻塞: (a) reward override key `custom_reward_function.path` 顶层; (b) prompt 截断大用例 (曾 sequence_length=105243>4096); (c) test_code 截断 max_pairs=8/max_case=4KB + S0预计算同款截断保 delta 可比 (曾致 reward 卡 246s); (d) return_dict 返回 dict (曾 'tuple' has no .get)。详见 [[grpo-skill-smoke-passed-20260722]]。
- ✅ **阶段 0 端到端打通完成**。下一步: 阶段 1 — 申请托管 8×L40 + Qwen3-4B 正式训练 (train_lcb_skillopt.sh, DP=8, max_iter=5, 全量41题天然分布, val 监控过拟合)。

## 实现计划 (最小可行, 接现有 GRPO 链路)
1. **A 模型 prompt 改造** (_build_prompt_template): 把 ICL-only 改为 "S_0 + 问题 + 改写指令", 让 A 输出改写后 skill 而非从零生成
2. **reward 改造** (custom_reward.py compute_custom_reward):
   - 对每个 group: 先用 S_0 跑 B 模型得 baseline_pass (可缓存, 同题只算一次)
   - 用 A 改写的 skill 跑 B 模型得 skill_pass
   - reward = skill_pass - baseline_pass (delta)
   - 保留质量检测 (退化 prompt 惩罚)
3. **数据**: LCB NEW (无污染, train=41), 对位 [[lcb-new-optimize-20260721]] baseline 0.4406
4. **训练**: 8×L40 托管平台, verl GRPO, Qwen3-4B-Instruct-2507
5. **评估**: 训练后 A 模型在 LCB test=143 上生成 skill → B 生成代码 → 对位 baseline 0.4406 / SkillOpt 0.4336

## 实现蓝图 (2026-07-21 定稿, 用户确认: 不CPU冒烟, 直接上8×L40调试)

### 系统现状核实 (已查清)
- 训练数据: train_testexample_bestmax_exp.parquet (22712题, OpenCodeInstruct), input字段=user message ("System Prompt Engineer" + User Query + Reference Solution + PROVEN GUIDELINES经验规则), output=JSON{ground_truth,unit_tests,raw_input}
- A模型输入是 parquet 预处理好的 prompt (不走 _build_prompt_template, 那是独立推理脚本 a_model_full.py 用)
- verl 训练: train_8l40.sh (8 GPU, DP=8, batch=40, max_iter=5), 用 train_testexample_bestmax_exp.parquet
- reward: custom_reward.py compute_custom_reward → process_single_sample_batch, group=3题, 每题 B生成+C评判+单测, case_score = obj(单测)×0.7 + sub(C评判)×0.3 + 质量惩罚, avg_score = group平均
- parquet 生成: data/OpenCodeInstruct/json2parquet.py (基础) → experience/inject_experiences.py (注入G规则, 产_exp.parquet)

### 改造点 (3处)

**改造1: 数据层 — 注入 S_0, 任务改为"改写 S_0"**

> ⚠️ **数据策略已定 (2026-07-21 用户决策): 先 LCB 同域小数据, 后 OpenCodeInstruct 多任务池迁移。**
> - **阶段 A (优先尝试)**: LCB NEW window train=41 同域数据。数据少, 但同域信号干净, 理论上最容易出效果。**主要风险 = 过拟合** (train=41/val=20 是 [[lcb-new-optimize-20260721]] 判死的同一坑, SkillOpt 启发式栽在这)。
>   - 对抗过拟合手段: GRPO 的 KL 约束 (相对 ref policy) + 小步数 + val-based early stop + 严格看 test 而非 val。
>   - 判据: 若 val 涨但 test 不动/下降 → 确认过拟合 → 转阶段 B。
> - **阶段 B (备选, 过拟合才转)**: OpenCodeInstruct (22712题) 多任务池迁移, 学 SePO "跨任务池预训练学改 skill 能力再迁移 LCB"。或再找其他无污染代码数据集做迁移。
>   - 注意: OpenCodeInstruct 是函数生成题, 与 LCB 的 stdin/stdout 竞赛题域不同 → S_0 也需换成函数生成域的通用 skill, 不能直接套 LCB initial.md (见改造1矛盾点)。
>   - 风险: 这条本质是 SkillComposer offline library 路线, [[skillcomposer-sepo-verified-20260721]] 指出 offline library 在 LCB 是负的 (MemP 47.4 < No Skill 56.6); 通用 skill 注入无效, per-task online 才有效。所以阶段 B 必须配合 per-task 在线细化, 不能只做通用库训练。
> - 阶段 A 与 B 的 S_0 不能混用: A 用 LCB initial.md (竞赛题 heuristics), B 用函数生成域 skill。两者的 parquet 构造脚本分开。

新写脚本 `data/OpenCodeInstruct/json2parquet_skillopt.py` (基于 json2parquet.py) — 阶段 A 版本:
- construct_training_prompt 改为: 给定 S_0 (skillopt/envs/lcb/skills/initial.md) + User Query + Reference Solution, 让 A 输出**改写后**的 system prompt (在 S_0 基础上改进)
- 指令强调: "Below is an INITIAL system prompt. Improve it to better guide code generation for this specific task. Keep what works, add task-specific procedural instructions targeting likely failure modes."
- 学 SePO evolved prompt 风格 (强制 dry-run / anti-looping / 具体失败模式)
- 保留 PROVEN GUIDELINES 注入 (inject_experiences.py 仍可用)
- 产出 train_testexample_bestmax_exp_skillopt.parquet (阶段 A: LCB 数据源)
- ⚠️ S_0 域匹配: LCB initial.md 是 stdin/stdout 竞赛题 heuristics, 仅适用阶段 A (LCB 数据); 阶段 B (OpenCodeInstruct 函数生成题) 必须换域匹配的 S_0, 否则 S_0 与题目语义错位, A 学不到有意义的改写

**改造2: reward 层 — delta pass rate**
在 custom_reward.py 加 S_0 baseline 缓存 + delta:
- 新增全局 `_S0_BASELINE_CACHE: Dict[query_hash, float]` (进程级 dict, 题不变则 S_0_pass 不变; 跨 rollout 复用; 进程重启清空; query_hash 用 test_query+ut 的 hash)
- `_compute_s0_pass(test_query, test_ut)`: 用 S_0 (initial.md) 跑 B 模型+单测, 返回 pass_rate, 缓存
- 在 process_single_sample_batch 每题算完 skill_pass 后: `delta = skill_pass - s0_pass`
- reward 改为 delta 驱动: case_score = f(delta) 而非 f(absolute pass_rate)
  - delta > 0 (改写 skill 比 S_0 好) → 正奖励
  - delta ≤ 0 → 0 或小负
  - 保留 obj/sub 细粒度但乘以 delta 符号/幅度
- 退化 prompt (复制 S_0 或空) → delta≈0 → 无奖励
- ⚠️ **退化检测需新增 (现有 check_prompt_quality 不覆盖)**: 现有质量检测只判空/崩溃 prompt, 不会判"原样复制 S_0"。A 模型最稳的捷径就是原样输出 S_0 (无退化惩罚 + delta=0 无负奖励)。必须在 reward 里加: `similarity(clean_sys_p, S_0) > τ` (或完全相等) → 直接 floor (MIN_REWARD_FLOOR)。τ 建议用字符级 Jaccard 或编辑距离比, 先取 0.95 严格档。
- ⚠️ **成本修正 (重要)**: S_0 baseline **不是 +33%, 是 B 模型 API 调用 ×2**。现 group 每题已调 1 次 B (skill_pass), delta 需每题额外调 1 次 B (s0_pass) → 双调用。结合 [[grpo-readiness-20260713]] "API 是真实瓶颈不是 GPU" (每 step 已 200 次 B+C、~1-2min GPU 空闲), 翻倍后每 step API 等待翻倍, 8×L40 空闲更严重。
  - 缓解: S_0_pass 跨 rollout 全局缓存命中后只算一次; 但首轮每题都要算 s0_pass, 成本仍 ×2。
  - 备选: S_0_pass 离线预计算 (训练前用 S_0 跑一遍 LCB train/val 全题, 存 json, 训练时读缓存零 API) — 推荐采用, 可消除 S_0 的在线 API 成本, 降回 ×1。代价是 S_0 固定不可中途换。
- ⚠️ **与现有 group 逻辑兼容性 (前置核实项, 必做)**: custom_reward.py:841 的 group 用 `TRAINING_POOL_LIST` (parquet 的 OpenCodeInstruct 题) 索引, 不是 LCB。阶段 A 若改用 LCB 数据, TRAINING_POOL_LIST 的填充路径 (从 parquet 读 q/gt/ut) 要相应换成 LCB 数据源, group 索引逻辑不能假定 OpenCodeInstruct。需先核实 `_find_index_brute_force` / `TRAINING_POOL_LIST` 在换数据后是否仍正确对齐。

**改造3: 评测层 — LCB 对位**
训练后 A 模型 (checkpoint) 在 LCB test=143 上:
- 每题: A 输入 S_0+问题 → 输出改写 skill → B 生成代码 → 单测
- 对位 baseline: S_0 直接用 (0.4406), SkillOpt best (0.4336), noskill 下界 (0.4266)
- 分层 easy/medium/hard
- ⚠️ **采样偏向 (reward 可行性命门)**: [[lcb-stratified-baseline-20260721]] 给 NEW test 分布: easy=0.97(34题天花板无信号)/medium=0.50(26题)/hard=0.18(83题)。delta 在 easy≈0 (S_0 已对)、hard 可能也≈0 (SkillOpt hard 0.193→0.181 零收益), 只在 medium 有少量信号。GRPO group=3 题, 若组内 delta 全 0 → reward 无区分度, A 学不动。
  - 对策: 训练数据采样**偏向 medium** (有信号且非天花板), hard 作 stretch goal (headroom 最大但最难, 留作 ablation 看是否能撬动)。easy 可少量保留作锚点但不指望它出信号。
  - 评测仍按全分层报告 (与 baseline 对位), 但训练/选模型看 medium 提升为主指标。

### 风险与对策
- S_0 baseline API 成本: 实为 B 模型调用 ×2 (非 +33%), 首轮每题双调用; 用 S_0_pass 离线预计算缓存消除在线成本 (推荐) 或跨 rollout 缓存命中摊薄
- delta 信号稀疏 (多数题 S_0 已对/已错, 改写无差): 用 obj 细粒度 (部分通过率差) 而非二元, 放大微小 delta; 训练采样偏向 medium 题 (有信号且非天花板)
- A 学不会改写退化为复制 S_0: 新增相似度退化检测 (sim>S_0 → floor) + delta=0 无奖励 + 训练数据 prompt 强调"improve"
- 过拟合 (阶段 A 最大风险): LCB train=41/val=20 是历史反复栽的坑 (mbpp/appsplus/lcb 全负); 靠 GRPO KL 约束 + 小步数 + val-based early stop + 只信 test 不信 val; 判死后转阶段 B

### 实现顺序
1. 先写改造1 (数据脚本) — 不依赖 GPU, 可本机验证 parquet 格式正确
2. 再写改造2 (reward delta) — 改 custom_reward.py
3. 写改造3 (评测脚本) — 基于 skillopt LCB rollout
4. 用户上 8×L40 跑 train_8l40.sh (指向新 parquet)
5. 训练后跑评测对位

## 前置核实 (实现前必做, 需可执行验收标准)
- [ ] 确认托管训练平台当前可用 + 8×L40 可申请 → 验收: 平台提交空跑任务返回资源就绪
- [ ] verl GRPO 链路当前能跑通 ([[grpo-readiness-20260713]] 7-13 冒烟通过, 但 7-21 未复查) → 验收: `bash train_8l40.sh` 跑 1 step 不报错 (用现有 exp parquet 冒烟)
- [ ] B 模型 API (DeepSeek-V3.2) 在训练平台网络可达 → 验收: 训练机上 curl aimpapi.midea.com 200
- [ ] custom_reward.py 的 S_0 注入点 + delta 计算不破坏现有 group 逻辑 → 验收: 核实 `_find_index_brute_force` / `TRAINING_POOL_LIST` 在换 LCB 数据源后仍正确对齐 (custom_reward.py:837-846); dry-run 1 个 group 看 s0_pass 缓存命中正常
- [ ] 退化检测相似度阈值 τ 标定 → 验收: 取几条"复制 S_0"和"轻微改写"样本, 确认 τ=0.95 能区分

## 待决策项 (2026-07-21 用户已定方向)
1. **S_0_pass 在线 vs 离线预计算** → **先离线** (×1 API, S_0 固定), 效果不好再转在线 (×2 API, S_0 可中途换)。
2. **训练采样配比** → 探索性超参, 先设一个**普遍默认值**起步, 后续效果不好再调。**LCB train=41 真实难度分布已查 (2026-07-21)**: easy 8 / medium 13 / hard 20 (hard 占近半); test=143 对位: easy 34 / medium 26 / hard 83 (与 [[lcb-stratified-baseline-20260721]] 一致 ✅)。
   - 信号可行性分析 (结合 baseline test 通过率 easy 0.97 / medium 0.50 / hard 0.18):
     - **easy (8题)**: 天花板, delta≈0, 无信号 → 不指望, 少量保留作锚点
     - **medium (13题)**: 0.50 一半对一半错, delta 可正可负, **区分度最好, 主信号源**, 但题量最少
     - **hard (20题)**: 0.18 headroom 最大但 delta 最稀疏 (S_0 多数全错 pass≈0, 改写难撬动), 偶尔 0→部分通过出 delta>0
   - **起步默认值建议**: 先用**全量 41 题 (天然分布) 不做重采样**作 baseline run — 因为题量本就小 (41), 重采样会进一步稀释信号或制造假多样性。先看天然分布下 GRPO 能否在 medium/hard 出 delta>0; 若 medium 信号被 hard 的稀疏 delta 淹没 (group=3 组内 delta 全 0), 再转**medium 上采样** (如 medium×2 / hard×1 / easy×0.5) 提升主信号密度。这是探索过程, 不预设最优, 实验驱动调整。
3. **SkillComposer online 对照** → **不复现**。用其论文数字 (LCB v6 online +2.5, Hard 67.2) 直接对比, 文字注明 executor 差异 (SkillComposer 用 Qwen3.5-4B, 我们用 DeepSeek-V3.2, 数字不可直接等比但趋势可参照)。
4. **阶段 A early stop 判据** → 探索性超参, 目前无法确定, 需在实验中不断微调。先设一个起步值 (如 val 连续 N 步不涨即停), 跑起来后据 test 分层表现调整。

> 注: 2 和 4 是探索性超参, 性质同学习率——先设默认起步值, 实验驱动调整, 不预先理论锁定。

## 风险
- delta pass rate 需双 rollout (S_0 + 改写), B 模型 API 调用翻倍; 离线预计算 S_0_pass 可消除在线翻倍 (见改造2)
- LCB train=41 偏小, 仍可能过拟合 (但 GRPO 有 KL 约束, 比 SkillOpt 启发式强); 过拟合则转阶段 B
- A 模型可能学不到"针对失败模式改写", 退化为复制 S_0 → delta=0, 需相似度退化检测 + 足够 reward 区分度

## 成功标准 (论文关键)
- A 模型改写的 skill 在 LCB test=143 上 **medium 显著 > 0.577** (baseline medium) 且整体 > 0.4406 (baseline) / > 0.4336 (SkillOpt)
- 分层看 hard 题 (>0.193, SkillOpt hard 零收益处) 作 stretch goal; medium 作主指标 (信号最干净)
- delta > 0 持续, 非 noise
- ⚠️ 对照完整性: 仅比 SkillOpt (已知最差启发式 baseline) 说服力弱。SkillComposer online 在 LCB v6 是 +2.5 正结果 (Hard 67.2), 应纳入对照——至少用其论文数字作 reference baseline 标注, 或说明不复现理由 (executor 不同: SkillComposer 用 Qwen3.5-4B, 我们用 DeepSeek-V3.2, 数字不可直接比但趋势可参照)。审稿人必问"为何不和正结果方法比", 需预备答复。

## 📦 实现清单 (2026-07-21 代码完成, 待端到端跑通)

### 新增文件
| 文件 | 用途 | 状态 |
|------|------|------|
| `data/OpenCodeInstruct/json2parquet_skillopt.py` | 改造1: 构造 LCB 同域训练 parquet (S_0+问题+测试用例→改写指令) | ✅ 验证通过 |
| `data/OpenCodeInstruct/dataparquet/train_lcb_skillopt.parquet` | 训练数据 (61题: train41+val20, hard25/medium22/easy14) | ✅ schema 对齐 verl |
| `rl/train/verl1/custom_reward_lcb.py` | 改造2: LCB 专用 reward (delta pass-rate + stdin/stdout 执行器 + 退化检测) | ✅ 单元验证, 待端到端 |
| `rl/train/verl1/precompute_s0_baseline.py` | S_0 baseline 离线预计算 (消除在线 API 翻倍) | ✅ 代码完成, 待重跑 |
| `rl/train/verl1/train_lcb_skillopt.sh` | 训练入口 (config override 指向新 reward + skillopt API env) | ✅ 代码完成, 待冒烟 |
| `rl/train/verl1/eval_lcb_skillopt.py` | 改造3: 训练后 A 模型(vLLM) 在 test=143 评测对位 | ✅ 代码完成, 待 GPU |
| `rl/train/verl1/outputs/s0_baseline_lcb.json` | S_0 baseline 缓存 (qid→pass_rate) | ⏳ 待重跑生成 |

### 关键设计决策 (实现中固化)
1. **执行器**: 复用 skillopt `lcb/evaluator.py` (stdin/stdout 子进程), 不重写。原 custom_reward.py 的函数式 assert 执行器与 LCB 不兼容 (见 docs/phase2_issues.md)。
2. **B 模型**: 用 skillopt `chat_target` 直接以 A 改写的 skill 作 system prompt, **不经 BAgent skill curation** (避免 402 skill 污染 per-task skill 信号, 保证 delta 只反映 A 改写的 skill 贡献)。
3. **reward 新建独立文件** (用户确认): `custom_reward_lcb.py` 与原 `custom_reward.py` (OpenCodeInstruct 函数题) 分离, 通过 config `custom_reward_function.path` 切换, 互不污染。
4. **per-task 评测**: 每条样本只评 center task (自己那题), 不做 aux group。符合 [[skillcomposer-sepo-verified-20260721]] "per-task online specification 才对代码题有效" 的启示。
5. **delta 计算**: skill_pass/s0_pass 都经 `_clamp_pass_rate` 把执行错误(-3)/超时(-1)哨兵压成 0, 使 delta 只反映真实通过率差。
6. **backend 配置**: skillopt `chat_target` 默认走 Azure (import 时读 `TARGET_BACKEND` env), 必须在 import 前设 `TARGET_BACKEND=qwen_chat`。已在 4 文件顶部 `os.environ.setdefault` 修复。

### reward 分数映射 (custom_reward_lcb.py)
- 退化: 空/崩溃 → -15 (floor), <100chars → -5, 复制 S_0 (Jaccard>0.95) → -3, 重复子串 → -15
- delta = skill_pass - s0_pass: delta>0 → 线性到 +10 (DELTA_POS_SCALE); delta<0 → 线性到 -5 (DELTA_NEG_SCALE); delta=0 → 0
- 叠加轻度质量惩罚后 clamp 到 [-10, 10]

## 🗺️ 未来规划 (实验执行路线图)

### 阶段 0: 端到端打通 (本机 + 训练机, 阻塞中)
- [ ] **0.1 重跑 S_0 预计算** (backend 已修): `precompute_s0_baseline.py` → `s0_baseline_lcb.json`。预期 61 题多数 agent_ok=True, S_0 平均 pass_rate ≈ 0.44 (与 [[lcb-new-optimize-20260721]] baseline 0.4406 一致则验证链路正确)。
- [ ] **0.2 reward 端到端冒烟**: 手造 1 条 A 输出 (改写 skill) 喂 `custom_reward_lcb.compute_custom_reward`, 确认能调通 B API + 单测 + delta 计算, reward 落在合理区间。
- [ ] **0.3 verl 1-step 冒烟** (训练机): `train_lcb_skillopt.sh` 跑 1 step, 确认 parquet 加载 + reward 调用 + GRPO 更新不报错。
- [ ] **0.4 前置核实** (蓝图已列): 训练机 API 可达、8×L40 可申请、verl 链路 7-21 复查。

### 阶段 1: 阶段 A 训练 (LCB 同域, 8×L40)
- [ ] **1.1 起步训练**: batch=8/n=5/max_iter=5, 全量 41 题天然分布 (待决策项2 起步值)。
- [ ] **1.2 过拟合监控**: 每 step 看 val (20题) + 训练 reward 曲线。判据 (待决策项4 起步值): val 连续 N 步不涨即停。**核心警惕**: val 涨但 test 不动 = 过拟合 (历史反复栽的坑)。
- [ ] **1.3 评测**: `eval_lcb_skillopt.py` 对位 test=143。看 medium (主指标) / hard (stretch) / overall vs 0.4406。

### 阶段 2: 判读与分支
- **若 test medium > 0.577 且 overall > 0.4406** → 阶段 A 成功, 代码域 skill 方向复活。深耕: 调采样配比/步数/delta 尺度做消融, 写论文主实验。
- **若 val 涨 test 不动 (过拟合)** → 转阶段 B (OpenCodeInstruct 多任务池迁移, 见数据策略)。注意 S_0 要换函数生成域 skill。
- **若 delta 全 0 (A 退化复制 S_0 或 skill 撬不动)** → 诊断: 看 trace 里 skill_pass vs s0_pass 分布。若 s0_pass 已 0/1 两极 (无中间态) → delta 信号天然稀疏, 需改用部分通过率细粒度或换 medium 上采样。

### 阶段 3: 论文实验矩阵 (成功后)
- [ ] 消融: (a) delta vs 绝对 pass-rate reward; (b) 从零生成 vs 改写 S_0; (c) 采样配比; (d) GRPO vs SkillOpt 启发式 (已有 baseline)
- [ ] 对照表: noskill(0.4266) / S_0(0.4406) / SkillOpt(0.4336) / 我们 GRPO / SkillComposer(论文数字 reference)
- [ ] 分层: easy/medium/hard + delta 分布 + 退化率
- [ ] 泛化: 阶段 B 迁移数据是否带来额外提升 (若做)

### 待决策项回顾 (2026-07-21 已定起步方向)
1. S_0_pass: 先离线 (实现已按此) → 效果不好转在线
2. 采样配比: 全量天然分布起步 → medium 信号被淹没则 medium 上采样
3. SkillComposer 对照: 不复现, 用论文数字 reference
4. early stop: val 连续 N 步不涨即停 (N 待微调)

Related: [[code-skill-method-fix-task-20260721]], [[sepo-mbpp-reference-20260721]], [[skillcomposer-sepo-verified-20260721]], [[lcb-new-optimize-20260721]], [[gpu-training-platform-20260721]], [[grpo-readiness-20260713]]
