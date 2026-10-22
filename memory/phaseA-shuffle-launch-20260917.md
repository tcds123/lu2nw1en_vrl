---
name: phaseA-shuffle-launch-20260917
description: 阶段 A shuffle 重测启动记录 — 注入 dataloader 打乱 choices+remap gold，去"答案恒 A"位置捷径，分层报告 meta_46/other_60
metadata:
  type: project
---

# 阶段 A shuffle 重测启动（2026-09-17 晚，PID 94756）

> 用户选定阶段 A（定性先行）：打乱 MCQ 选项使 gold 不再恒 A、meta 选项位置随机化，直接回答"去位置捷径后还剩多少真实提升"。此为对 [[gpt-review-p0-dataset-shortcut-20260917]] P0 发现（答案全 A + meta 100% 正确）的正面回应实验。verdict（判读）待跑完后写 [[phaseA-shuffle-verdict-20260917]]。

## 启动参数

```bash
cd /data/zhuldz/lunwen
OUT=rl/train/verl1/outputs/skill_evolver_math_poolfilter/phaseA_shuffle_20260917
nohup python3 -u rl/train/verl1/skill_evolution/eval_test_multiseed_shuffle.py \
  --skills s0,on,off,skillopt,rule_meta_guess \
  --eval_k 3 --seeds 4 --out_dir $OUT > $OUT/run.log 2>&1 &
# PID 94756
```

- **5 臂**：s0 / on / off / skillopt / rule_meta_guess（4 LLM + 1 零 API 规则基线）
- **k=3, seeds=4**，与历史口径匹配
- **split**：`/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathmaticianbench_1-3-6_seed42`（test=106 题，46 meta / 60 other）
- **theo_iter6 已剔除**：outputs 目录被清 + git 无 track + hash 671135ff 全盘查无，不可恢复；用户决策"改用 4 skill"。

## 5 臂 hash（启动日志已确认）

| 臂 | hash | chars |
|----|------|-------|
| s0 | 68571cb2 | 1017 |
| on | fa63335e | 2609 |
| off | 02a7f019 | 3502 |
| skillopt | cfae156d | 9821 |
| rule_meta_guess | — | zero-API |

ON 用原样 `best_skill_evolved.md`（run_20260916_181857），**不删任何段**——核查已证 L16-20 定理在 test 0 次出现（非 test 泄漏）、L4 meta 规则 shuffle-invariant（合法算法产物）。shuffle 本身即去捷径。

## 机制确认（带行号证据，本会话独立核实）

1. **shuffle 注入点**：`eval_test_multiseed_shuffle.py` 在 `eval_skill_on` 调用前对每个 item 调 dataloader 的 `_shuffle_item_choices(dict(it), seed)`（深拷贝后打乱），remap `correct_choice.label` 到新位置。
2. **dataloader shuffle 逻辑**（`skillopt/envs/livemathematicianbench/dataloader.py:195-226`）：`_item_shuffle_seed(item_id, seed)` = `sha256(f"{seed}:{item_id}")[:16]` → int → `random.Random` 种子 → `rng.shuffle(choices)` → `_CHOICE_LABELS[idx]` remap gold。确定性 per (qid, seed)，每 seed 不同。
3. **evolver 原本绕过 shuffle**（`tf_global_v2_math.py:177 load_split` 直接 `json.load(items.json)`）→ eval gold 恒 A。本实验在外层包装注入 shuffle，**不动 run_one/evaluator/dataloader**。
4. **evaluator 按 label 判分**（`evaluator.py:41-43` `predicted_label == correct_label`）：remap 后 gold 不再恒 A，"选 A"位置捷径失效。
5. **meta 检测信号**：选项文本含 `"one of the remaining options is correct, but a stronger result can be proven"`（小写 substring）即 meta 题。test 共 46 道，全对。

## 冒烟测试通过证据

run.log 启动头：
```
📋 test items: 106 | meta: 46 | other: 60
   k=3 | seeds=4 | shuffle=ON (per qid,seed 确定性)
  s0: 1017 chars (hash 68571cb2)
  on: 2609 chars (hash fa63335e)
  off: 3502 chars (hash 02a7f019)
  skillopt: 9821 chars (hash cfae156d)
  rule_meta_guess: zero-API rule baseline
[shuffle check] qid=202602:29 seed1 gold=C (原始恒 A, 打乱后应非恒 A)
```

- gold 已从 A remap 到 C → shuffle 生效。
- 5 臂 hash 全对。
- rule_meta_guess 在 meta_46 上应 ≈1.0、other_60 上应 ≈0.2，作为"无算法下限"参照线。

## 判读规则（Step 3 核心）

- **若 rule_meta_guess ≥ 所有演化 skill 在 other_60 上** → 其余 60 题无真实算法增益，**主结论=数据集捷径，叙事需重写**。
- **若演化 skill 在 other_60 上 > s0 且跨 seed 稳定** → **这部分才是真贡献**，可量化（Xpp on 60 non-meta questions）。
- **若 shuffle 后所有 skill（含 ON）都掉到 ~0.547 附近** → ON 的 0.5873 全是 meta+A 捷径，算法无真实增益。

## 成本与时长

- 4 LLM skill × 106 题 × 4 seed × 3 calls = ~5088 API calls（DeepSeek-V3.2）
- rule_meta_guess 零 API，瞬时
- 预估 ~2.9h（按历史吞吐）
- cache key 后缀 `_shuf`（`per_{name}_{md5(txt)[:8]}_seed{seed}_shuf.json`），避免与未打乱旧 cache 混淆

## Seed 1 完整画面（2026-09-17 15:52，5 臂全落盘，待 seed2-4 确认稳定性）

🔑 **关键节点（milestone）**：5 臂 seed1 全部落盘，shuffle 后画面清晰。这是 Phase A 的核心定性结果（单 seed，待 4-seed 坐实稳定性，但与零 API P0 分析完全一致）。

| 臂 (seed1, shuffled) | mean | meta_46 | other_60 | vs rule |
|----------------------|------|---------|----------|---------|
| s0 | 0.3176 | 0.188 | 0.417 | 输 (0.318<0.519) |
| on | 0.4371 | 0.507 | 0.383 | 输 (0.437<0.519) |
| off | 0.4340 | 0.471 | 0.406 | 输 (0.434<0.519) |
| **skillopt** | **0.5755** | 0.783 | **0.417** | **赢 (0.576>0.519)** |
| rule_meta_guess | 0.5189 | 1.000 | 0.150 | (基准) |

**四个核心发现**：
1. **rule 基线 sanity 通过**：meta_46=1.000（含 meta 文本→选 meta，shuffle 后仍 100% 对）、other_60=0.150（≈预测 0.2，60 题 1/5 确定性猜的小样本方差）→ 规则实现正确，参照线可信。
2. **🚨 rule_meta_guess (0.5189) 击败 on (0.4371) 和 off (0.4340)**：零 API 平凡规则"见 meta 文本就选否则乱猜"在 shuffle 后**优于两条演化臂**。ON 的"+0.2925 算法效应"在 on/off 这两条 run 上**不成立**——平凡规则就能严格占优（meta 更准 1.0>0.5 + other 不差 0.15≈0.38-0.41 的差距小于 meta 差距）。**注意**：rule 在 other_60 上 0.150 < on/off 的 0.38-0.41，但 meta_46 上 1.0 >> on/off 的 0.49，meta 子集的压倒优势抵消了 other 的劣势。
3. **skillopt 是唯一存活者**：唯一击败 rule 基线的臂（0.5755>0.519），因为 meta 规则学得**更好**（0.783 vs on/off 0.49）**且**保留真数学能力（other_60=0.417 = s0 水平）。skillopt 不是 on/off pool_filter run 的产物，是独立 SkillOpt 演化。
4. **干净分解**：skillopt other_60=0.417 = s0 other_60=0.417（精确相等）→ **skillopt = s0 +（学到的 meta 规则）**，非 meta 能力没增也没减。而 on/off **退化**了非 meta（0.38-0.41 < 0.417）同时 meta 学得不完美（0.49 非 1.0）→ 被平凡规则严格占优。

**对主结论的修正（seed1 定性，待 4-seed 坐实）**：
- ✅ **用户正确**：meta 规则是 shuffle-invariant 的合法算法产物（非位置伪影、非泄漏），on/skillopt shuffle 后 meta_46 仍 >0.5。
- 🚨 **+15.6pp pool_filter 效应死亡**：on/off shuffle 后 0.4371 vs 0.4340 统计不可分（Δ=0.003），未打乱的 +0.1564 差距归零。独立坐实 P1。
- 🚨 **ON 的 +0.2925 算法效应在 on/off run 上不成立**：on 被平凡 rule 占优。但 ON 学到的 meta 规则本身是真实的（shuffle 后仍 +0.319 on meta_46）——只是 on/off 这两条 run 学得不够好（0.49 vs rule 1.0）。
- ✅ **skillopt 保留真实价值**：独立演化产物击败 rule 基线，是唯一有真贡献的 artifact。
- **叙事重写方向**：从"pool_filter +15.6pp / ON 算法有效 +0.2925"→"演化算法会收敛到 meta-option 启发式（合法但 on/off 学得不如平凡规则）；skillopt 是唯一击败零模型基线的 artifact，其增益 = 学到的 meta 规则（shuffle-invariant），非 meta 题无增益"。

## Seed 2 完整画面（2026-09-17 16:46，5 臂全落盘 — seed1 所有发现稳定性坐实）

| 臂 (seed2, shuffled) | mean | meta_46 | other_60 | vs seed1 |
|----------------------|------|---------|----------|----------|
| s0 | 0.3428 | 0.239 | 0.422 | ✅ 0.318 |
| on | 0.4497 | 0.551 | 0.372 | ✅ 0.437 |
| off | 0.4277 | 0.435 | 0.422 | ✅ 0.434 |
| **skillopt** | **0.5849** | 0.833 | 0.394 | ✅ 0.576 |
| rule_meta_guess | 0.5189 | 1.000 | 0.150 | ✅ 0.519 (确定性) |

**seed1→seed2 稳定性坐实**：
1. **on≈off collapse 跨 seed 成立**：on 0.443±0.006 vs off 0.431±0.003，Δ=0.012（seed2）/ 0.003（seed1），均远小于 on-off 未打乱 0.1564。+15.6pp pool_filter 效应死亡 **跨 2 seed 坐实**。
2. **skillopt 唯一存活者跨 seed 成立**：0.576/0.585 均 > rule 0.519，且 meta_46 0.783/0.833（学得比 on/off 好）、other_60 0.417/0.394（≈s0 水平）。
3. **rule 基线击败 on/off 跨 seed 成立**：rule 0.519 > on 0.443 / off 0.431。
4. rule_meta_guess seed1=seed2=0.5189（确定性，预期）→ 实现无误。

## Seed 3 完整画面（2026-09-17 17:41，5 臂全落盘 — 3-seed 跨 seed 全部坐实）

| 臂 (seed3, shuffled) | mean | meta_46 | other_60 |
|----------------------|------|---------|----------|
| s0 | 0.3333 | 0.210 | 0.428 |
| on | 0.4528 | 0.551 | 0.378 |
| off | 0.4245 | 0.420 | 0.428 |
| skillopt | 0.5660 | 0.783 | 0.400 |
| rule_meta_guess | 0.5283 | 1.000 | 0.167 |

**3-seed 跨 seed 稳定性汇总（mean / sd）**：

| 臂 | seed1 | seed2 | seed3 | 3-seed mean | sd |
|----|-------|-------|-------|-------------|-----|
| s0 | 0.3176 | 0.3428 | 0.3333 | 0.3312 | 0.0103 |
| on | 0.4371 | 0.4497 | 0.4528 | 0.4465 | 0.0065 |
| off | 0.4340 | 0.4277 | 0.4245 | 0.4287 | 0.0039 |
| skillopt | 0.5755 | 0.5849 | 0.5660 | 0.5755 | 0.0077 |
| rule | 0.5189 | 0.5189 | 0.5283 | 0.5220 | 0.0047 |

**核心结论跨 3 seed 全部坐实**：
1. **on≈off collapse 跨 3 seed**：Δ(on−off) = 0.003 / 0.022 / 0.028 → mean 0.018，3-seed 均 ≪ 未打乱 0.1564 → **pool_filter +15.6pp 效应死亡跨 3 seed 坐实**（独立坐实 P1 的 GPT review 发现）。
2. **skillopt 唯一存活者跨 3 seed**：0.576/0.585/0.566 均 > rule 0.519/0.519/0.528（3/3 seed 一致击败规则基线）。
3. **rule 击败 on/off 跨 3 seed**：rule 3-seed mean 0.522 > on 0.447 / off 0.429（平凡规则严格占优 on/off 两条演化臂）。
4. **用户正确跨 3 seed**：on/skillopt shuffle 后 meta_46 仍 >0.5（on 0.507/0.551/0.551、skillopt 0.783/0.833/0.783），证明 meta 规则是 shuffle-invariant 的真语义信号（非位置伪影非泄漏）。

**进度**：seed3 完成（5/5 臂落盘），seed4 启动（s0 seed4 10/106）。PID 94756 2:44:18 elapsed。剩余 seed4 ≈ 40min。

## Seed 4 完整画面 + 4-seed 最终汇总（2026-09-17 18:33，5 臂全落盘 — PID 94756 干净退出，20 个 cache + 汇总 JSON）

| 臂 (seed4, shuffled) | mean | meta_46 | other_60 |
|----------------------|------|---------|----------|
| s0 | 0.2925 | 0.183 | 0.378 |
| on | 0.4434 | 0.543 | 0.367 |
| off | 0.4245 | 0.464 | 0.394 |
| skillopt | 0.5189 | 0.739 | 0.356 |
| rule_meta_guess | 0.5094 | 1.000 | 0.083 |

**4-seed 跨 seed 稳定性汇总（mean / sd）**：

| 臂 | seed1 | seed2 | seed3 | seed4 | 4-seed mean | sd |
|----|-------|-------|-------|-------|-------------|-----|
| s0 | 0.3176 | 0.3428 | 0.3333 | 0.2925 | 0.3215 | 0.0190 |
| on | 0.4371 | 0.4497 | 0.4528 | 0.4434 | 0.4458 | 0.0058 |
| off | 0.4340 | 0.4277 | 0.4245 | 0.4245 | 0.4277 | 0.0039 |
| skillopt | 0.5755 | 0.5849 | 0.5660 | 0.5189 | 0.5613 | 0.0240 |
| rule | 0.5189 | 0.5189 | 0.5283 | 0.5094 | 0.5189 | 0.0060 |

**核心结论跨 4 seed 全部坐实（4-seed sd 全 ≤0.024，远小于臂间差距）**：
1. **on≈off collapse 跨 4 seed**：Δ(on−off) = 0.003 / 0.022 / 0.028 / 0.019 → mean **+0.0181**，4/4 seed 均 ≪ 未打乱 0.1564 → **pool_filter +15.6pp 效应死亡跨 4 seed 坐实**（独立坐实 P1）。
2. **skillopt 唯一存活者跨 4 seed**：0.576/0.585/0.566/0.519 > rule 0.519/0.519/0.528/0.509（4/4 seed 击败规则基线；**caveat**：seed4 margin 收窄到 +0.0095 vs seed1-3 的 0.0566/0.066/0.0377，但仍正）。
3. **rule 击败 on/off 跨 4 seed**：rule 4-seed mean 0.519 > on 0.446 / off 0.428（平凡规则严格占优 on/off 两条演化臂）。
4. **用户正确跨 4 seed**：on/skillopt shuffle 后 meta_46 仍 >0.5（on 0.507/0.551/0.551/0.543、skillopt 0.783/0.833/0.783/0.739），证明 meta 规则是 shuffle-invariant 的真语义信号（非位置伪影非泄漏）。

**最终 verdict 已落盘 [[phaseA-shuffle-verdict-20260917]]**（含完整 4-seed 分层表、Step 3 三规则判读、对主结论的修正、四条教训、seed4 caveat）。PID 94756 已退出，run.log + multiseed_test_eval.json（2497 行）齐全，20 个 per_*_shuf.json cache 落盘。

## 关联

[[gpt-review-p0-dataset-shortcut-20260917]]（P0 发现，本实验回应对象）/ [[livemath-meta-option-trap-20260910]]（9/10 已记 meta 陷阱，未意识到答案全 A 全数据集伪影）/ [[pool-filter-on-off-verdict-20260917]]（被 P0 动摇的 +15.6pp 结论，verdict 后加 caveat）
