---
name: measure-foundation-layerA-f6-done-20261016
description: F6 s0_cache 复用地基完成+自测通过 (AIME manifest build/load), Layer B/C 待做
metadata:
  type: project
---

# 修测量地基 — Layer A (F6 cache 复用) 完成 (2026-10-16)

承接 [[gpt-review-verdict-20261016]]: 用户选「修测量地基」(GPT Step1-2)。批准计划在 `/root/.claude/plans/wise-bubbling-crayon.md`，三层: F6 缓存复用 / F2 救援可达 / F1 提功效。本文件记 Layer A 完成。

## Layer A (F6) — AIME cache manifest 复用: ✅ 完成并自测全过

**改了什么** (`rl/train/verl1/skill_evolution/`):
- `tf_global_v2_aime.py`:
  - `precompute_s0` 加 `write_manifest=False` 默认 + kwargs (shuffle_seed/split_dir/split_name)。`write_manifest=True` 时落盘 AIME 专用 `_manifest`。默认 False → 独立 `main()` (本文件 :333 位置参数调用) 字节不变 (零回归)。
  - **新增** `build_cache_manifest_aime`: 用 `it["answer"]` 作 gold (AIME 数值题 0-999, 无 choices/correct_choice → math 版 `build_cache_manifest` 读 `correct_choice["label"]` 会 KeyError)。`answer_arrangement_hash` 替代 math 版 `choice_arrangement_hash`，记 `{id, answer}` 对。loader 按键名判别 AIME vs math manifest。
  - **新增** `load_s0_cache_with_manifest_aime`: 校验 6 键 [k/eval_temp/shuffle_seed/skill_hash/split_hash/answer_arrangement_hash]。无 manifest / 不匹配 → None (现场重算)。拒绝混用 math 格式 cache (有 choice_arrangement_hash 无 answer_arrangement_hash → None)。
- `skill_evolver_aime_v2.py`:
  - import 块 (L57-61) 加 `build_cache_manifest_aime, load_s0_cache_with_manifest_aime`。
  - s0_cache 加载块 (L4245+): loader 换 `load_s0_cache_with_manifest_aime`; 3 个 `precompute_s0` 调用 (trainval 全集重算 / partial 补算 / 无 s0_cache 全集重算) 全加 `write_manifest=True` + shuffle_seed/split_dir/split_name="train+val"。partial 补算后 `extra` 的 manifest 只反映 missing 子集 → 用 `build_cache_manifest_aime` 重建全集 manifest 再 dump。

**修的 bug**: F6 = 双重断裂。AIME `precompute_s0` 写不了 manifest (math 版能写但读 correct_choice 会 KeyError on AIME) → loader 拿不到 manifest → 返回 None → 每 run 现场重算 S₀。三次 run 拿到三个不同 baseline (0.544/0.589/0.600)，σ≈0.044 采样噪声，candidate-vs-baseline delta 被 baseline drift 污染。

**自测** (`/tmp/test_f6_manifest.py`, 纯零 API, 6 组 11 断言全过):
- build 无 KeyError，answer_arrangement_hash 在，choice_arrangement_hash 不在，skill_hash/k/n_items/split_dir 正确。
- 匹配 cache → 返回 cache; 错 skill_hash → None; 无 manifest → None; math 格式 → None; 错 k → None。
- (初版 1 FAIL 是 harness 的 md5 helper 未截断 `[:10]`，非代码 bug — module 的 `md5` 截断到 10 字符。)

**零回归保证**: `write_manifest` 默认 False; AIME loader 只拒绝 manifest 缺失/不匹配的 cache (与 math 版行为一致的正确行为); 无新 flag。flags 全 OFF 时 else 分支 = 仅诊断 (今日行为)。

## 待做 (Layer B/C)
- **Layer B (F2 救援可达)** — 全部 default-OFF flag: B1 CLI flags (`--rescue_on_reject`/`--veto_upper_bound`/`--veto_skip_n1`); B2 subset_signal/group_signal 加 `ci_upper`; B3 veto 参数 + `_is_regression` helper; B4 `_combined_rescue` 透传 veto; B5 overlay/expand/crossover/strengthen else 分支 + facet_edit promote 接 rescue。
- **Layer C (F1 提功效)**: C1 用 `--eval_k 8` (无代码); C2 paired_rescore 推迟到 F6+F2 验证后。
- 验证: 零回归 (flags OFF 字节同 run#3) + F2 救援可达自测 + 端到端 run#4 (F6+F2 ON, k=8)。

关键位置 (Layer B 目标，行号随 import +1 后): subset_signal 513-538 / group_signal 569-593 / subset_promote_decision 540-565 (veto:559) / group_promote_decision 595-616 (n≥2:609, veto:613) / `_combined_rescue` 619-640 / facet_edit promote 3214-3220 (citem 经 survivors tuple 恢复) / overlay else 3540 / expand else 3724 / crossover-mp else ~3308 / crossover-2p else ~3447 / strengthen else ~2933。

关联 [[gpt-review-verdict-20261016]] [[group-admit-gate-implemented-20261014]]。
