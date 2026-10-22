---
name: aime-v2-build-launch-20260930
description: AIME v2 实验启动 — Fork B(扩 skill_evolver_aime.py 加 facet_tree/expand/pool_filter 结构)+ skillopt Path A(写 aime env adapter 跑其 trainer)。两 track 并行构建(无API成本)→smoke→checkpoint→全量跑。ON=pool_filter+facet_tree+expand, OFF=无。数据 AIME24/25。2026-09-30
metadata:
  type: project
---

# AIME v2 实验启动 (2026-09-30)

承接 [[aime-port-mcq-coupling-finding-20260930]]（决策：Fork B + skillopt Path A + AIME24/25）。本文件记录实验启动参数与执行序列。用户指令："试一下我们的算法在不同的AIME数据集上的表现，并且也用skillopt进行对比试试"。

## 实验目标

测**唯一真正未测的组合**：facet_tree/expand/pool_filter 机制结构 × AIME-native taxonomy。
- 08-07 阴性仅测旧 M1-M4 evolver（v2 Δ=0.00pp, evolver Δ=−1.67pp，见 [[aime-fresh-rerun-result-20260807]]）。
- Step6 证 ON 增益=MCQ 标签捷径（机制**内容** MCQ 耦合；gain_MCQ=+0.1682 p=0.00025, gain_freeform≈0，见 [[step6-answer-form-verdict-20260929]]）。
- 本实验隔离"机制结构" vs "MCQ 内容"——结构是否在无 MCQ 时仍独立有效。

## 两 track 并行构建（无 API 成本）

### Track A: skill_evolver_aime_v2.py
- 基底: `rl/train/verl1/skill_evolution/skill_evolver_aime.py` (410行, AIME-native, M1-M4 + `FAILURE_FOCI_AIME` 8 数值类型)
- 叠加: math evolver (`for_review/code/skill_evolver_math.py`, 4017行) 的 **schema-agnostic 机制结构** (facet_tree GA / expand / strengthen / pool_filter 门控 / 蒸馏循环 / 交叉 / 选择CI)
- 替换: MCQ-coupled **内容** → AIME-native (classify_* 改读 `item["problem"]`/`item["answer"]`; `FAILURE_FOCI_AIME` 作 facet 轴/峰; has_meta→no-op; 诊断渲染 "Answer: pred N / correct N"; PATCHER_SYSTEM 改 AIME 数值)
- 评测后端: 复用 `tf_global_v2_aime.py` (已 AIME-native, **不改**) + `eval_aime24_multiseed.py`
- 新 argparse: `--pool_filter`/`--facet_tree`/`--expand`/`--strengthen` (BooleanOptionalAction default False) + `--probe_seed` (非 --shuffle_seed; shuffle diag-items/交叉池非选项位置)
- **ON = `--pool_filter --facet_tree --expand`; OFF = 无**（AIME 无选项, on/off 退化为机制 ON/OFF 单变量）

### Track B: skillopt aime env adapter (~9 文件)
- 复制 `skillopt/envs/livemathematicianbench/` 结构 → MCQ→数值, 复用 `eval_aime24_multiseed.py` 的 `extract_int_answer`/`evaluate_aime`
- 文件: adapter/dataloader/rollout/evaluator + prompts/rollout_system.md + skills/initial.md (拷 `data/skill_aime_initial.md`) + `configs/aime/default.yaml` + 1 行 env 注册
- ⚠️ rollout **必须**写 per-item `conversation.json` + `target_system_prompt.txt` + `target_user_prompt.txt` (否则 reflect 返 None, skill 不进化——最易静默失败)
- 30 题小 → batch_size=6, num_epochs=8-10, split 7:1:2 (mitigate val=3 高方差)
- 后端: **qwen_chat/DeepSeek-V3.2** (复用 `tf_global_v2_math.py` L38-47 的 `QWEN_CHAT_*` env vars: api_key=msk-4601b27a…, base_url=aimpapi.midea.com, model=/model/DeepSeek-V3.2)。**非** skillopt 默认 azure_openai/gpt-5.5 (无权限)。trainer 不导入 tf_global_v2_math，故启动脚本须先 export 这些 env vars，或 AIME YAML 显式设 target_backend=qwen_chat。

## 执行序列
1. 构建（无 API 成本）← **当前**
2. smoke（低成本, 1-3 题/track 验证链路通）
3. smoke 结果作自然检查点（落盘 + 汇报）
4. 才启动昂贵全量跑（AIME24 + AIME25, k=3, EVAL_TEMP=0.3, multi-seed，对齐既有协议）

## smoke 门控（全过才全量）
- Track A: `skill_evolver_aime_v2.py` 语法+import 通；`--iters 1 --train_n 3` 跑通一轮（低 API）
- Track B: adapter import 通；dataloader 加载 aime24.jsonl；evaluator 评分单题；trainer 1 epoch 3 题（低 API）

## 判据
- **正**（ON > AIME s0 显著）→ 破泛化边界, 机制结构独立有效（须辨"算法有效" vs "手工好 taxonomy"，Fork B 已知混淆风险）
- **负**（ON ≈ OFF ≈ s0）→ 坐实算法形态级能力不及 skillopt（与 Step6 收敛：捷径不仅在产物在机制）
- skillopt 跑出 AIME skill vs 我们 → 公平能力对比（skillopt trainer 经查零 MCQ 偏置, 全 MCQ 逻辑锁在 livemath 目录）

关联：[[aime-port-mcq-coupling-finding-20260930]]、[[aime-fresh-rerun-result-20260807]]、[[step6-answer-form-verdict-20260929]]、[[dataset-pivot-aime-onoff-20260923]]。
