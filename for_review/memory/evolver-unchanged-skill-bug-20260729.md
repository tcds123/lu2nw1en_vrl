---
name: evolver-unchanged-skill-bug-20260729
description: evolver run_20260727_163814 全8轮未改skill(best==init), +15.7pp实为两版v2对比非evolver产物, 结论推翻
metadata: type: project
---

# evolver 产物未改动 bug (2026-07-29) 🔴🔴🔴 推翻 evolver 主结果

## 核心发现
复审 run_20260727_163814 发现:
- `best_skill_evolved.md` 与 8 个 `skill_iter{1..8}.md` md5 全相同 = `911597f7df...`
- 该 md5 与 evolver 的 **init_skill** (`tf_skill_math_v2/run_20260727_100047/...md`) **完全相同** → 8 轮迭代从未修改 skill
- evolver_log 的 best_val_delta 0.2264→0.3585 标 improved=True 是 `val_passes=1` 单次采样噪声 (同 skill 多次评出不同 delta)

## 推翻的结论
- ❌ [[evolver-reeval-test-20260729]] 的 "evolver +15.7pp 远强于 v2" **不成立**
- 原因: multiseed 里 "evolver" 指向的是未改动的 init skill (md5 911597f7df), "v2" 基线是 `run_20260728_131934` 的**另一版** v2 skill (md5 b98a9abd, 3655chars). +15.7pp 实为 **v2_0727 vs v2_0728** 的对比, 与 evolver 机制无关
- ❌ [[next-steps-20260729]] task1 "evolver 机制消融" 前提不存在, evolver 主结果候选**撤回**
- ✅ v2 +6.6pp (run_20260728_131934 vs S0) 独立成立, 不受影响

## 根因 (skill_evolver_math.py)
1. `val_passes=1` 单次采样 + `cands.append(best_skill)` 保底: winner 常是 current_skill 的重采样, `improved=True` 但 `best_skill` 重赋值为同一字符串 → 8 轮原地踏步
2. LOST/CI 门控被单次采样噪声绕过
3. 评测对象错配: eval_test_multiseed.py EVOLVER_BEST 指向未改动 init, "v2" 是另一版

## 修复方向
1. evolver 评测改 k 投票连续 pass_rate (复用 v2 eval_k+EVAL_TEMP), 删 val_passes; candidate 与 baseline 同条件
2. winner 必须 delta 显著>current 且 skill_hash!=current_hash 才接受; 保底 current 只作下界不进候选池
3. 修复后重跑, 确认 best_skill md5 != init, 再 multiseed 重评测
4. 详见 docs/phase2_issues.md 末条

## ✅ 修复已完成 (2026-07-29, 代码改完待跑)
`rl/train/verl1/skill_evolver_math.py` 已按上述方向改完:
1. **复用 v2 评测机制**: 顶部 `from tf_global_v2_math import run_one, eval_on, precompute_s0, bootstrap_ci_lower, set_temp, EVAL_TEMP, GEN_TEMP, load_split, md5`. 删除本地单次采样的 `run_one`/`eval_on`/`precompute_s0`/`bootstrap_ci_lower`/`md5`. (注意 v2 `eval_on` 返回 **4 元组** `(mean_pass, mean_delta, n, per_item)`, 调用点已适配)
2. **产物身份校验**: 删 `cands.append(best_skill)` 保底竞争; winner 仅在 `md5(cand)!=current_hash 且 ci>0` 的 novel 候选里选; `improved` 须 `winner_hash!=current_hash`; accept 时 `assert md5(winner)!=current_hash`
3. **CLI**: 删 `--val_passes`, 加 `--eval_k`(默认5); `SPLIT_DEFAULT`→1-3-6_seed42; `V2_SKILL_DEFAULT`→run_20260728_131934(与 multiseed v2 同源); `--s0_cache` 默认→`s0_cache_math_1-3-6_k5.json`(k=5 连续值, 覆盖71题, 已验证)
4. **温度**: `generate_candidates`/`consolidate` 前 `set_temp(GEN_TEMP=0.8)` 真采样 (llm 的 temperature 形参已删, 同 v2 bug); 候选多样性改靠 per-candidate 文字角度提示 (angle_hints)
5. **日志**: 每轮记 `current_hash`/`winner_hash`/`n_novel_candidates`/`reject_reason`; main 末尾断言 `md5(best)!=md5(init)`, 未改动则打印 ⚠️
- 已通过 `py_compile` + import 测试 (符号/签名均匹配)
- ⏳ **未跑**: 冒烟测试 (iters=2/eval_k=3/train_n=5) 因与同机其他任务争 28RPM 配额而取消. 待空闲后跑: 先确认 `best_skill_evolved.md` md5≠init, 再 `eval_test_multiseed.py --skills s0,evolver` 复评

## 🔴 07-30 冒烟暴露第二个 bug: generate_candidates 签名错误致 0 novel

07-30 跑冒烟 (PID 39002), 基线 init pass=0.270 delta=+0.032 CI=-0.023 (select 58题). Iter 1 报 `generated 4 candidates, 0 novel` — 4 候选全 hash==current (b98a9abd). cand0 评测 `delta=+0.067 CI=+0.006 novel=False` (current 重评测噪声), **novel_positive 过滤正确将其排除**, 证明 07-29 产物身份校验逻辑生效.

根因 (非 LLM 不改, 是调用签名 bug): `generate_candidates` 把 `dict(max_tokens=4096, stage="evolver_patch")` 当**第三个位置参**传给 `llm()`, 而 `llm(system,user,*,max_tokens,stage)` 的 max_tokens/stage 是 **keyword-only** → `TypeError: llm() takes 2 positional arguments but 3 were given` → 被静默 `except Exception` 吞 → 4 候选全 fallback current_skill → 永远 0 novel. 诊断: 单跑 `llm(...,max_tokens=,stage=)` (kwargs) 4/4 novel (len 4016-4169), 唯一区别是位置 vs kwargs.

**修复 (07-30, ✅代码改完待重跑)**: `ex.submit(llm, PATCHER_SYSTEM, user, max_tokens=4096, stage="evolver_patch")` (dict 解包为 kwargs); except 改为打印异常不再静默吞. `consolidate` 已用正确 kwargs 无需改. 验证: 修复后 generate_candidates 4/4 novel (hash 各异). py_compile 过.

教训追加: `except Exception: fallback` 静默吞异常让"签名错误"伪装成"LLM没改", 优化器候选生成失败必须打印异常; ThreadPoolExecutor 传 kwargs 用 `ex.submit(fn,*args,**kwargs)` 不能把 dict 当位置参.

## 教训
- **产物身份校验第一关**: 优化器产物必须先验证 hash(output)!=hash(input)
- val_passes=1 单次 0/1 评测在 temp>0 下 delta 抖动 ±0.06~0.13 足以伪造"improved"
- 保底回退 (append current_skill) 若进候选竞争必导致原地踏步被判改进

Related: [[evolver-reeval-test-20260729]], [[multiseed-test-replicate-20260729]], [[next-steps-20260729]]
