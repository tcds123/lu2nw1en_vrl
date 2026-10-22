# Plan: 修复 evolver 使其真正修改 skill (产物身份校验 + k 投票 + 保底下界)

## 背景与诊断 (已记录于 docs/phase2_issues.md 末条 + memory/evolver-unchanged-skill-bug-20260729.md)
evolver run_20260727_163814 全 8 轮 `best_skill_evolved.md` 与 `init_skill` md5 全同 (`911597f7df`), 从未修改 skill. 根因:
1. `val_passes=1` 单次 0/1 采样 → 同 skill 每次评出不同 delta, 噪声 ±0.06~0.13 足以伪造 improved.
2. `generate_candidates` 末尾 `cands.append(best_skill)` 保底 → winner 常是 current_skill 的重采样, `improved=True` 但 `best_skill` 重赋值为**同一字符串**, 8 轮原地踏步.
3. evolver `run_one` 无 k 参数(单次 0/1), 且**没 import v2 已修好的 `set_temp`/`EVAL_TEMP`** → 评测在默认 temp=0 网关非确定. 用旧 `s0_cache_math_full.json`(k=1, 0/1 值).

v2 (`tf_global_v2_math.py`) 已修好同类问题: `run_one(k)` k 投票连续 pass_rate + `set_temp(EVAL_TEMP=0.3)` + baseline 用 s0_cache 固定快照不现场重跑. **复用 v2 的成熟机制, 不重新发明.**

## 决策 (已与用户确认)
- split 统一为 **1-3-6_seed42** (train18/val53/test106), select=train+val=71, 复用已存在的 `s0_cache_math_1-3-6_k5.json` (n=71, k=5 连续值).
- 起点 `V2_SKILL_DEFAULT` 改指 `run_20260728_131934/tf_global_skill_v2_math.md` (与 multiseed 的 v2 baseline 同源).

## 改动 (单文件: rl/train/verl1/skill_evolver_math.py)

### 改动1: 复用 v2 的评测机制
顶部从 `tf_global_v2_math` import: `run_one`(带 k), `eval_on`(带 k, **返回 4 元组**), `precompute_s0`(带 k), `bootstrap_ci_lower`, `set_temp`, `EVAL_TEMP`, `GEN_TEMP`, `load_split`, `md5`.
删除 evolver 本地的: `run_one`(L108-124), `eval_on`(L134-149), `bootstrap_ci_lower`(L152-159), `precompute_s0`(L162-170), `md5`(L101-102).
保留本地 `llm`, `collect_failures`(内部 run_one 调用加 k 参数 + 评测前 set_temp(EVAL_TEMP)), `generate_candidates`, `consolidate`, `clean_skill`, 模板常量.

**调用点签名适配** (v2 eval_on 返回 4 元组 `(mean_pass, mean_delta, n, per_item)`):
- L330: `base_pass, base_delta, _ = eval_on(...)` → `base_pass, base_delta, _, base_per = eval_on(...)`; `base_ci = bootstrap_ci_lower(base_per)`.
- L366: `cp, cd, citem = eval_on(...)` → `cp, cd, _, citem = eval_on(...)`.

### 改动2: CLI flag
- 删 `--val_passes`; 加 `--eval_k`(默认5).
- `SPLIT_DEFAULT` → 1-3-6_seed42 路径.
- `V2_SKILL_DEFAULT` → run_20260728_131934 的 skill.
- `--s0_cache` 默认 → `rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json`.

### 改动3: 产物身份校验 + 保底下界 (核心)
主循环 L358-395:
- **保底不进候选竞争**: 删 `cands.append(best_skill)`. 保留 `current_skill = best_skill` 作下界.
- `generate_candidates` 前 `set_temp(GEN_TEMP=0.8)` (patch 真采样); `consolidate` 前 `set_temp(GEN_TEMP)`.
- 评候选前 `set_temp(EVAL_TEMP)` (eval_on 内部已设, 但显式保险).
- **hash 校验**: 统计 `n_novel = sum(1 for c in cands if md5(c) != md5(current_skill))`. winner 选择: 在 `scored` 中**仅 hash != current** 的候选里选 delta 最高且 ci>0; 若无 novel 候选或全部 ci<=0, winner=None.
- `improved = winner is not None and winner["delta"] > best_val_delta and winner["ci"] > 0 and winner["pass"] >= base_pass - 0.06`.
- 若 winner is None 或未 improved → no_improve++; 记录原因(no novel / ci<=0 / delta 不超 / LOST).
- 接受时 `best_skill = winner["skill"]`, 断言 `md5(best_skill) != md5(current_skill)`.

### 改动4: 日志可复查
evolver_log 每轮增加: `current_hash`, `n_novel_candidates`, `winner_hash`(或 null), `reject_reason`. main 末尾: 若 `md5(best_skill)==md5(init_skill)` 打印 ⚠️ "产物未改动" 警告(但仍落盘).

## 不改
- 4 机制设计(GEPA/失败patch/慢更新/防退化)、PATCH_TEMPLATE/CONSOLIDATE_TEMPLATE/FAILURE_FOCI 保留. 消融待真增益坐实后再做.

## 验证 (修复后)
1. 冒烟: `--iters 2 --eval_k 3 --train_n 5` 跑通, skill_iter1.md hash != init.
2. 产物身份: best_skill_evolved.md md5 != init_skill md5.
3. 全量重跑 (iters=10, eval_k=5, split=1-3-6).
4. `eval_test_multiseed.py` 同条件复评 evolver best vs v2 vs S0 (k=3/2seed). 旧 +15.7pp 作废, 以新数为准. 写 memory.

## memory 已更新 (✅ 本次会话已完成)
- 新建 memory/evolver-unchanged-skill-bug-20260729.md
- 推翻标注: evolver-reeval-test-20260729.md, next-steps-20260729.md
- MEMORY.md 索引修订
- docs/phase2_issues.md 末条追加
