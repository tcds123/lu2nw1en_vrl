---
name: appsplus-flash-skillopt-launch-20261111
description: APPS_Plus flash SkillOpt 启动 — train80/val100/test300 三重 cap, 8 步 ~2h; gated slow_update (非 V3.2 force-accept); 验证 flash+大数据集能否让 skill-text Δ 跳出 ClassEval val=10 噪声
metadata:
  type: project
---

# APPS_Plus flash SkillOpt 启动 (2026-11-11)

## 一句话
用户 "换大数据集" (task #29). 在 APPS_Plus interview 上跑 flash SkillOpt, 验证 flash + gated slow_update
+ 大 val/test 能否让 skill-text 优化 Δ 跳出 ClassEval val=10 饱和噪声.
三重 cap (train80/val100/test300) 把 V3.2 的 80步30h 压到 8步~2h, 且规避 V3.2 小样本(n=200)假阳性陷阱.

## 背景澄清 (重要 — 推翻 summary 的 reframe 误读)
- 上一段 summary 曾据 `appsplus-generalize-20260717.md` 断言 "V3.2 appsplus 是 +6pp 正结果, 计划文档的负结论 framing 是错的". **此 reframe 本身是误读** — 该 memory 是 07-17 的 test=200 小样本结果.
- `appsplus-full-test-reversal-20260718.md` (07-18, 次日) 已用**全量 test=2765** 反转该结论: noskill 0.6564 > seed 0.6499 > best 0.6341, best **−2.2pp 退化** (apples-to-apples 2755 共同题 z=−1.746 负向).
- 全量 eval log 实测确认 (本 session `appsplus_full_{noskill,seed,best}.eval.log`): noskill 0.6564 / seed 0.6499 / best 0.6341.
- generalize memory 作为 "小样本假阳性教训" 历史保留 (已链接 "被本文反转"). **无需 surface reframe — 负结论是真的, 计划文档 framing 正确.**

## vs V3.2 判死 run (appsplus_full_optimize, 2026-07, 21步未跑完) 三处差异
1. **model**: V3.2(DeepSeek-V3.2) → flash(deepseek-v4-flash). B1 LCB 已证强模型翻转负结论 (Δ−1.4pp→+4.9pp, [[b1-flash-skillopt-lcb-result-20261109]]).
2. **cap**: 4096 → 16384 (同 B1 flash 口径).
3. **slow_update_gate_with_selection: false → true** ⭐ THE FIX.
   V3.2 用 force-accept 把 slow guidance 无条件堆进 skill → 膨胀到 11639 chars (同 ClassEval force-accept 膨胀模式, final −4.3pp).
   gated 让 slow candidate 也过 val gate (evaluate_gate), 只有真超 current 才注入, 防膨胀退化.

## 采样设计 (避免双重陷阱)
APPS_Plus interview=3950, ratio 2:1:7 seed42 → train=790/val=395/test=2765 (同 V3.2 split).
V3.2 train=790 → ceil(790/40)=20 steps/epoch × 4ep = 80 步 × 395-item selection eval/step = 30h 不可行.
本 run 三重 cap (ratio split seed42 确定性, base.py:504-505 `env_num < len → items[:env_num]` 取前 N):
- **train_size=80** → ceil(80/40)=2 steps/epoch × 4ep = **8 步** (对齐 B1 收敛步数; B1 收敛于 step 6).
- **sel_env_num=100** → 10× ClassEval val=10, 跳出 0.9 饱和; val=100 → hard 粒度 0.01 (vs ClassEval val=10 的 0.1), paired-gate 能区分 skill 变体.
- **test_env_num=300** → paired McNemar 检 ±3pp; ratio+seed42 确定性 → 前 300 是 2765 的稳定子集, 后续 test_env_num=0 可扩全量复测对位.

大数据集意义在 **eval 信号分辨率** (val/test 大), 非 train 算力; train 8 步够优化 (B1 证).

## V3.2 小样本陷阱为何不复现
- V3.2 的 +6pp 假阳性是 **unpaired 外部样本** (07-17 test=200 与全量 test=2765 不同 items, baseline 被小样本低估 0.575 vs 全量 0.656).
- 本 run baseline-vs-final 是 **paired** (同 items 同 harness, SkillOpt 内部 test_eval) → McNemar, 远强于 unpaired t-test. test=300 paired 检 ±3pp.
- 即: 即使 flash 也得负 Δ, 也是干净 paired 结论, 非 V3.2 那种 "换样本就翻" 的脆弱信号.

## 配置验证 (本 session dry-run 通过)
- `configs/appsplus/default_flash.yaml` flatten 后与 B1 LCB 完全同构 (optimizer_backend=target_backend=qwen_chat, model 名由 QWEN_CHAT_MODEL env 驱动, 非 cfg).
- step 数学: train_size=80 / batch 40 / accum 1 → 2 steps/epoch × 4ep = 8 步 ✅
- gate_metric=hard, slow_update_gate_with_selection=true, sel_env_num=100, test_env_num=300 ✅
- launch shim: `rl/train/verl1/skill_evolution/launch_skillopt_flash_appsplus.sh` (仿 launch_skillopt_flash_lcb.sh: 读 flash key from yaml, unset AIGC_USER, RPM=20, TARGET/OPTIMIZER_BACKEND=qwen_chat).

## ⚠️ 修正: split_dir capped split (首启 exit=1, 见 issue #54)
- 首启失败: `train_size=80` 是**硬匹配断言** (trainer.py:432), 非 cap. APPS_Plus 3950 ratio split → train=790 ≠ 80 → raise. (summary 曾误判 train_size 可 cap, 实际不能.)
- 修复 = B1 范式: 从 seed42 ratio split (确定性) 复制 capped split → **train 截前 80**, val/test 全量 (eval 时 env_num cap). 落盘 `/data/zhuldz/skillopt/data/appsplus_capped_seed42/{train,val,test}/items.json`.
- config 改 `split_mode: split_dir` + `split_dir: /data/zhuldz/skillopt/data/appsplus_capped_seed42` (原 ratio 模式废).
- dry-run `adapter.setup()` 验证: get_train_size=80 (断言过), val 395→cap100 (first_id=2009), test 2765→cap300 (first_id=2801), 8 steps ✅. first_ids 与 ratio split 一致 (确定性保持).

## 判据 (对照 V3.2 全量负结论 + B1 LCB 正信号)
| test Δ (final vs baseline, paired n=300) | 判定 |
|-------------------------------------------|------|
| ≥ +3pp | flash + gated 在大数据集拿到正 skill 杠杆, 证伪 V3.2 force-accept 膨胀退化根因 ✅ |
| ±2pp 内 | skill 优化 Δ 仍噪声, 即使大 val 也区分不出 → 代码竞赛题 skill-text 优化到顶 (与 ClassEval 同型) |
| ≤ −2pp | 退化 (即使 gated 也过拟合 val) → force-accept 不是唯一根因, skill 表示本身有问题 |

## 运行参数 (备查)
- `RUN_TAG=appsplus-skillopt-flash-20261111-v2 bash launch_skillopt_flash_appsplus.sh` (v1 首启 exit=1 train_size 断言; v2 用 split_dir capped split 修复)
- ep4 seed42 ratio 2:1:7 (train=790/val=395/test=2765), cap train80/val100/test300
- single-shot (max_turns=1; 2-turn 是 ClassEval doctest 专属, 不开)
- flash, 16384 cap, 20 RPM, 预计 ~2h
- 对照 baseline: V3.2 全量 noskill 0.6564 / best 0.6341 (−2.2pp)

## 关联
- [[appsplus-full-test-reversal-20260718]] (V3.2 全量负结论, 本 run 的判死对照)
- [[appsplus-generalize-20260717]] (07-17 小样本假阳性, 已被反转 — 勿据其 reframe)
- [[b1-flash-skillopt-lcb-result-20261109]] (flash 翻转 LCB 负 +4.9pp, 强模型假设)
- [[classeval-a2-2turn-result-20261111]] (ClassEval skill-text 到顶, 转 APPS_Plus 的动因)
- [[code-skill-negative-rootcause-20260721]] (代码域负根因: 单文件竞赛题 skill 杠杆受限)
