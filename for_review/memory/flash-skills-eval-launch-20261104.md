---
name: flash-skills-eval-launch-20261104
description: flash 跨模型阶段①(paired 推理泛化)+阶段②(evolver/SkillOpt 重跑) 启动记录, 2026-11-04
metadata:
  type: project
---

# flash 跨模型 skill 评测 — 启动记录 (2026-11-04)

## 背景
用户要求用新模型 deepseek-v4-flash 跑两件事 (V3.2→flash 跨模型):
1. **阶段① 推理泛化**: V3.2 上演化的 evolver_best + skillopt_best 拿到 flash 上做 paired 评测 (cand vs S₀), 看跨模型后还有无收益. 数据集 AIME24(in-sample)+AIME25(held-out) 都跑.
2. **阶段② 完整演化**: flash 上重跑 evolver + SkillOpt 全流程, 看新模型是否产出不同 skill. 训练仍用 AIME24 (与 V3.2 同训练集, 控制变量).

## 关键 bug 修复 (本轮发现并落盘 docs/phase2_issues.md)
- **#28**: s0_cache manifest 无 model 字段 → flash 会静默复用 V3.2 baseline (0.597 而非 flash 0.854). 缓解: evolver `--s0_cache ""` 强制现场重算. 阶段① crossmodel 脚本用本地已有的 flash cache (已验证是 flash 跑的), 不受影响.
- **#29**: flash cap 必须 16384 (非 4096/8192). flash 把 CoT 写进 content, 硬题需 ~13207 completion tok; cap 4096→finish=length→`<answer>`未达→pass=0 (answer 路径 #26) + 空 patch→evolver 静默卡住 (optimizer 路径, #29 新发现). thinking=disabled 非有效兜底 (实测关了输出仍长仍截断). 修复: 两套算法 cap 均提到 16384.
  - evolver: monkeypatch `skillopt.model.chat_target` (router, 单点覆盖 answer+optimizer 全路径), 已 dry-run 验证三处绑定均 patched.
  - SkillOpt: yaml 两处 cap (`model.qwen_chat_max_tokens` + `env.max_completion_tokens`) + env `QWEN_CHAT_MAX_TOKENS` 全改 16384.
  - 阶段① crossmodel: `co_consts` 字节码补丁 run_one 内 4096→16384 (无 optimizer 调用, 够用).

## 候选 skill (V3.2 产物, 阶段①评测对象)
| skill | 路径 | chars | md5 |
|---|---|---|---|
| S₀ | data/skill_aime_initial_v2.md | 599 | c11d7dc0c8 |
| evolver_best | .../skill_evolver_aime_v2/full_off_20260930/run_20260930_161552/best_skill_evolved.md | 2408 | f4837ca80f |
| skillopt_best | .../tf_skill_aime_v2/skillopt/aime-skillopt-20261008/train/best_skill.md | 4618 | 5a63ddb6af |

## S₀ cache (flash 实测, 阶段①复用, 嵌套 {id:{pass_rate}} 格式)
| 数据集 | 路径 | 题数 | mean |
|---|---|---|---|
| AIME24 | rl/train/verl1/outputs/crossmodel_baseline/run_20261102_170450/s0_cache_aime.json | 30 | 0.8542 (train 0.896 / val 0.792) |
| AIME25 | .../crossmodel_baseline/flash_aime25_s0_20261104_100708/s0_cache_aime25.json | 30 | 0.8375 (16/30 满分) |

## 执行顺序 (全部串行 — flash 20RPM 共享配额)
1. **1a** AIME24 paired: `eva/_run_flash_aime24_paired.sh` → PID 1943, 启动 2026-11-04 18:32, log `eva/flash_aime24_paired_20261104.log`
2. **1b** AIME25 paired: `eva/_run_flash_aime25_paired.sh` → PID 122332, 启动 2026-11-04 20:16, ✅ 完成 21:46
3. **2b** SkillOpt flash: `rl/train/verl1/skill_evolution/launch_skillopt_flash.sh` (cap 16384, ~3.5-5h) — ✅ 启动 21:52 (PID 104403), OUT_ROOT `aime-skillopt-flash-20261104-215251`
4. **2a** evolver flash: `rl/train/verl1/skill_evolution/launch_evolver_flash.sh` (cap 16384 monkeypatch, --s0_cache "", 6 flags FULL, ~9h) — 待 2b 完成后起

## 阶段② 参数 (与 V3.2 run_20260930_161552 完全一致, 仅模型变)
- evolver: iters=10 K=4 eval_k=3 n_parallel=1 n_diag=8 train_n=35 consolidate_every=3 patience=3 shuffle_seed=0; flags=facet_tree+bootstrap+crossover+overlay+expand+subset_admit (ablation=FULL)
- SkillOpt: default_flash.yaml, ratio split 6:2:2, num_epochs=8, batch_size=6, lr=4, slow_update

## 判读标准
- 阶段①: delta (cand−base) + CI_lower95. CI_lower>0 → 跨模型泛化成立; delta≈0 CI跨0 → 无收益 (与 V3.2 结论一致); delta<0 → 过拟合 V3.2.
- 阶段②: flash best_skill vs V3.2 best_skill md5/chars 是否不同; flash evolver skill 回阶段①测 held-out 是否优于跨模型迁移.

## 阶段② SkillOpt 中间里程碑 (2026-11-04 22:07 检查)
**关键发现: flash 产出了与 V3.2 不同的 skill, 且 gate 分数更高.**

| 维度 | V3.2 best_skill | flash best_skill (step_0004) |
|---|---|---|
| 路径 | `.../aime-skillopt-20261008/train/best_skill.md` | `.../aime-skillopt-flash-20261104-215251/train/best_skill.md` |
| chars | 4618 | **1284** (精简 ~3.6×) |
| md5 | 5a63ddb6af | **29ae3bbce4** |
| gate score | 0.8333 (initial_skill, best_step=0) | **1.0** (step_0004, accept_new_best) |
| diff | — | 34 行不同 |

- step_0004: `candidate_gate_score=1.0`, `action=accept_new_best`, `best_origin=step_0004`. 之前 step_0001/0002 cand_gate=0.833(=baseline), step_0003 cand_gate=0.667(回退). 第 4 步 patch 突破到满分.
- step_0004 patch: n_edits_merged=5, n_edits_ranked=4, edit_apply 4/4 applied; timing rollout 99s+reflect 4s+eval 37s=144s/step.
- 新 skill 结构更紧凑: 用 case enumeration(不重叠覆盖)+ feasibility bound(discriminant≥0)+ magnitude sanity check 替代 V3.2 的冗长指令. 是更强的推理模型(flash)对"skill 该长什么样"的不同收敛.
- **无 429/timeout/truncation** (train.log 全 0) → cap 16384 生效, flash 推理健康.
- 进程 PID 104403 仍活 (etime 14:39), ~2.4min/step, 8 epoch×3 step≈24 step 总, 还剩 ~19 step → 预计 ~22:50 完成. best 已锁 1.0, 后续 step 无法再升 (1.0 是上限).

**结论 (SkillOpt, 中间)**: 新模型不仅产出了不同 skill, 且收敛更快 (step_0004 即满分 vs V3.2 step_0000 即封顶在 0.833). 但注意 gate=1.0 是 train/val split 内的 6 题 gate 评测, 非全部 30 题; 完整流程结束后需回阶段① paired 脚本测 held-out (AIME25) 才能确认泛化收益.

## 阶段② SkillOpt 最终结果 (✅ 完成, PID 104403, 21:52→23:09, ~1h17m)

### 训练轨迹 (24 step / 8 epoch, gate=val split 6 题)
| step | epoch | gate | action |
|---|---|---|---|
| 1-3 | 1 | 0.833/0.833/0.667 | reject |
| **4** | **2** | **1.000** | **accept_new_best** ← best 锁定 |
| 5 | 2 | 0.833 | reject |
| 6,7 | 2-3 | 1.000 | reject (持平, 不升) |
| 8-9 | 3 | 0.833 | reject |
| 10,13,16-18,20,22,24 | 4-8 | 1.000 | reject (反复到顶, 无法再升) |
- step_0004 之后再无 accept (1.0 已是上限, 后续 patch 全部 reject). best_origin=step_0004.
- 全程无 429/timeout/truncation → cap 16384 生效, flash 推理健康.

### test split (held-out 6 题) — 关键: val 提升未迁移到 test
| skill | val(gate) | test_eval | test_eval_final |
|---|---|---|---|
| flash best_skill (1284c, 29ae3bbce4) | **1.000** | 0.667 | 0.667 |
| flash baseline (initial) | 0.833 | 0.667 | 0.667 |
| **V3.2 best_skill** (4618c, 5a63ddb6af) | 0.833 | **0.667** | — |
| **V3.2 baseline** (initial) | — | **0.833** | — |
| V3.2 final_selection_eval | — | — | 0.500 |

**判读**:
- flash: val 1.0 但 test=baseline=0.667 → **val 提升未迁移到 test, 但也无 regression** (flat).
- V3.2: best test 0.667 < baseline 0.833 → V3.2 SkillOpt 在 test split 上 **regression** (skill 反而拉低 0.166). flash 无此 regression.
- 即: flash 上 SkillOpt 产出的 skill 虽然 held-out test 无增益, 但至少 **不损害** test; V3.2 的 skill 会损害 test. 这是新模型的一个质性差异.
- 但注意 test split 只有 6 题 (6:2:2 切分), 统计功效极低, 0.667 vs 0.833 = 1 题之差. 仍需回阶段① paired 脚本 (k=8, 全 30 题) 测 AIME25 held-out 才有统计意义.

### 阶段② SkillOpt 结论
1. **新模型产出不同 skill**: ✅ (1284c vs 4618c, md5 不同, 结构更精简).
2. **val 收敛更快更高**: ✅ (step_0004 即 1.0, V3.2 step_0006 才 0.833).
3. **held-out test 无显著迁移**: gate/test split 上无增益 (flat), 与 V3.2 一致 (V3.2 甚至 regression).
4. **待办**: flash best_skill 回阶段① paired 脚本测 AIME25 held-out (k=8, 30 题), 确认泛化.

## 阶段①结果 (实时更新)

### 1a AIME24 paired (in-sample) — ✅ 完成 (run_20261104_183227, 18:32→20:14, ~1h42m)
| candidate | delta | CI_lower(95%) | mean_pass | n_pairs | 结论 |
|---|---|---|---|---|---|
| evolver_best (f4837ca80f, 2408c) | **-0.0042** | -0.0333 | 0.871 | 30 | CI 跨 0 → 无显著收益, 与 V3.2 held-out Δ≈0 一致 |
| skillopt_best (d0b56b9330*, 4613c) | **+0.0458** | +0.0000 | 0.892 | 30 | CI_lower=0 边际显著 → flash 上收益 **大于** V3.2(+0.011) |

S₀ baseline: all=0.8542 (train 0.896 / val 0.792), 21/30 满分.

> *md5 注记: 脚本 `md5(cand)` 对内容 rstrip 后算, 得 `d0b56b9330...`; 原始字节(含尾换行)md5=`5a63ddb6af...`. 同一文件, 无歧义.

判读:
- **evolver_best** 拿到 flash AIME24 delta≈0 (−0.0042), CI 跨 0 → 跨模型迁移无正收益. flash 基线已 0.854 (21/30 满分), evolver skill 无 headroom 可挖. 与 V3.2 结论一致 (evolver held-out Δ≈0).
- **skillopt_best** 拿到 flash AIME24 **+0.046** (CI_lower=0 边际显著), mean_pass 0.892 → **比 V3.2 held-out(+0.011) 收益更大且更接近显著**. 是跨模型泛化的更有意思信号, 待 AIME25 held-out 验证.

### 1b AIME25 paired (held-out) — 🔄 进行中 (PID 122332, 20:16 起)
S₀ cache all=0.8375 (train 0.778 / val 0.927), 16/30 满分. 信号点更多 (14 题非满分 vs AIME24 的 9 题).

| candidate | delta | CI_lower(95%) | mean_pass | n_pairs | 耗时 | 结论 |
|---|---|---|---|---|---|---|
| evolver_best (f4837ca80f) | **+0.0250** | -0.0208 | 0.854 | 30 | 2656.0s | CI 跨 0 但 delta 正向; AIME25 表现 **优于** AIME24(-0.004) |
| skillopt_best (d0b56b9330*) | **+0.0208** | -0.0375 | 0.829 | 30 | 2575.1s | CI 跨 0, delta 正向但小于 AIME24(+0.046) |

S₀ baseline (AIME25): all=0.8375 (train 0.778 / val 0.927), 16/30 满分.

## 阶段①完整汇总 (推理泛化: V3.2 skill → flash 评测)

| skill | AIME24(in-sample) Δ | AIME24 CI_low | AIME25(held-out) Δ | AIME25 CI_low | 趋势 |
|---|---|---|---|---|---|
| evolver_best | **-0.0042** | -0.0333 | **+0.0250** | -0.0208 | held-out > in-sample |
| skillopt_best | **+0.0458** | +0.0000 | **+0.0208** | -0.0375 | in-sample > held-out |

**结论 (阶段①)**:
- 4 个 delta 中 3 个为正, 1 个 ≈0; **无负迁移** (没有 delta 显著为负).
- 但 **4 个 CI_lower 全部 ≤0** → 没有一个达到统计显著 (CI 跨 0).
- 最强信号: skillopt_best 在 AIME24 (in-sample) Δ=+0.046, CI_lower=0.000 — **边际显著** (恰触 0), 是 4 组里最接近显著的.
- evolver_best: in-sample(AIME24)饱和无 headroom (−0.004), held-out(AIME25)反而正向 (+0.025) — 说明收益不在训练分布内.
- skillopt_best: 与 evolver 相反, in-sample 更强 (+0.046), held-out 弱 (+0.021).
- **对照 V3.2 历史**: V3.2 evolver held-out Δ≈0, V3.2 SkillOpt held-out Δ=+0.011 (不显著). flash 上两个 skill 的 held-out delta (+0.025 / +0.021) 均 **大于** V3.2 对应值 — 更强模型 (flash) 上 skill 收益有放大, 但仍未达显著.

详见 [[flash-skills-eval-result]] (完成后落盘). 相关 bug: [[crossmodel-configs-aimp-20261030]].

## 阶段② 2a evolver 重启 (2026-11-04 23:20, #31 修复后)

**首次启动失败 (#31)**: PID 69300, ~2min "跑完", skill 599→599c hash `c11d7dc0c8` 未变. 日志 30/30 `eval_on` errored + 30/30 facet_edit 失败, 全 `ValueError('Azure OpenAI endpoint is not configured')`, 0 edit promoted. 根因 = `backend_config.py:15-16` import 时冻结 `TARGET_BACKEND` env 成模块常量, shim 的 `import skillopt.model` 早于 `tf_global_v2_math` setdefault → 冻结成默认 `openai_chat`(Azure). 详见 docs/phase2_issues.md #31. 失败 run 已移至 `..._failed_31` 保留现场.

**修复 (已应用 launch_evolver_flash.sh)**:
1. shell export `TARGET_BACKEND=qwen_chat` + `OPTIMIZER_BACKEND=qwen_chat` (早于任何 python import).
2. Python block 显式 `set_target_backend("qwen_chat")` 双保险.

**重启验证 (PID 76672, 2026-11-04 23:20)**:
- T+126s / T+4.5min / T+5.5min: **0 Azure 错误** (对照失败 run 满 log 都是). ✅
- python 76672 持有 `socket:[3449921414]` 活跃网络连接 → 真在发 flash 请求 ✅
- 进程 State=S (sleeping on IO, 等 flash HTTP 响应), 31 threads, 0.3% CPU ✅
- 卡在 `⏳ precomputing S₀ on select set (k=3)` 阶段 — select 30题 × k=3 = 90 顺序请求 @20RPM, AIME 硬题单次 20-60s, 预计 30-90min 完成 precompute 后进 GA 主循环.
- **对比失败 run**: 失败 run 2min 内 errored 完毕无 socket; 重启 run 5min+ 仍 active + socket = 真实 flash 调用确认.

**参数 (与 V3.2 run_20260930_161552 完全一致, 仅模型+out_dir 变)**: iters=10 K=4 eval_k=3 n_parallel=1 n_diag=8 train_n=35 consolidate_every=3 patience=3 shuffle_seed=0; flags=FULL(facet_tree+bootstrap+crossover+overlay+expand+subset_admit); cap=16384; s0_cache="" (#28 强制现场重算 flash baseline). 预计总耗时 ~9h (4000-5500 flash 请求).

**产物路径**: `rl/train/verl1/outputs/skill_evolver_aime_v2/flash_20261104/run_flash_20261104_232046/run_20261104_232047/`
- 日志: `rl/train/verl1/skill_evolution/evolver_flash_flash_20261104_232046.log`
- 监控: `tail -f` 该 log; 完成判据 = `best_skill_evolved.md` md5 ≠ `c11d7dc0c8` (skill 真被改) + log 出现 `✅ Facet-Tree done` 且 `best val delta` 非 +0.000.

**完成后待做**: flash evolved skill 回阶段①脚本测 AIME25 held-out paired, 对比"flash 自演化" vs "V3.2 跨模型迁移" skill.

## 阶段② 2a evolver 最终结果 (✅ 完成, 2026-11-05 05:37)

**运行**: 2026-11-04 23:20 启动 → 11-05 05:37 完成, 实际 ~6h (10 iters × ~2000-2840s/iter, 总 ~21745s). 比 V3.2 的 ~9h 快 (flash 单次生成比 V3.2 快, + cap 16384 无截断重试).

**核心结论: flash evolver 成功, V3.2 evolver 失败. 模型变强 → error-driven skill 优化才生效.**

| 指标 | V3.2 evolver (run_20260930_161552) | flash evolver (run_flash_20261104_232046) |
|---|---|---|
| best val delta | **-0.0222** (负, 回归) | **+0.0778** (正) |
| best val CI | -0.1222 (远不显著) | **+0.0111 (CI>0, 显著)** |
| 10 iters promoted 总数 | **0** (全 run 0 promote, best=父=退化的 S₀) | 1 (iter3 expand admitted→became_best; iter9 overlay 重申) |
| skill chars | 2408 | 2028 |
| skill md5 | `f4837ca80f` | `eaad9fda3e` (不同) |
| 最佳算子 | 无 (全 KEEP-PARENT) | `expand` "Number-Theoretic Case Analysis: Exhaustive Residues & Bounded Ranges" (iter3) |
| skill 内容 | 扩写的 S₀ (4 节更啰嗦, 无新结构) | 4 基础节 + **新增第 5 节 "Number-Theoretic Case Analysis: Exhaustive Residues & Bounded Ranges"** (含具体决策过程: 枚举所有 residue class / CRT 合并 / bound range / sanity-check count / 几何转整数方程) |

**逐 iter 轨迹 (flash)**: iter1-2 best=+0.033(ci<0) → **iter3 expand 跃升 +0.078(ci=+0.022, admitted→became_best)** → iter4-8 保持 +0.078 (overlay 多次试, 混合 CI 跨0, 部分 admitted 不 became_best) → iter9 overlay admitted→became_best (+0.078 ci=+0.011, 最显著) → iter10 crossover(Computation×NumberTheoretic admitted)+overlay(admitted). 最佳落地 = iter9 overlay 版.

**为什么 V3.2 失败而 flash 成功 (推断)**: evolver 靠 collect_failures 找 S₀ 错题 → patch 修. V3.2 上 S₀ 基线低 (0.597), 错题多且错因杂, 单点 patch 难显著救一题, bootstrap CI 门全拒 → 0 promote → best 退化成负 delta. flash 上 S₀ 基线高 (0.833), 错题少 (val 8 题里 S₀ 错的聚焦在 "Number-theoretic case analysis" 一个失败类型), patch 针对性强 → iter3 一次 expand 就显著救题 (+0.078). **强模型的错误更"可学习" (集中、可定位), error-driven 优化才有效; 弱模型错误太分散, patch 无的放矢.**

**对照阶段① (跨模型迁移)**: V3.2 上演化的 evolver_best 迁到 flash 测, AIME25 held-out Δ=+0.025 (CI 跨0, 不显著) — 即 V3.2 自己没学到东西 (val -0.022), 迁到 flash 也没显著收益. 而 flash 自演化的 skill 在 flash val 上 +0.078 显著. **自演化 > 跨模型迁移**, 符合预期.

**产物**:
- `rl/train/verl1/outputs/skill_evolver_aime_v2/flash_20261104/run_flash_20261104_232046/run_20261104_232047/best_skill_evolved.md` (2028c, md5 eaad9fda3e)
- `evolver_log.json` (10 iter 完整轨迹), `tree_iter1..10.json`, `archive_iter1..10.json`, `best_skill_iter1..10.md`
- 日志: `rl/train/verl1/skill_evolution/evolver_flash_flash_20261104_232046.log`

## 阶段② 2a 补测: flash 自演化 skill AIME25 held-out (✅ 完成, 2026-11-05 10:40)

**运行**: `eva/crossmodel_flash_baseline.py --provider deepseek-v4-flash --data aime25.jsonl --s0_cache <已转格式 AIME25 cache> --cand_skill <flash evolved best_skill_evolved.md> --eval_k 8 --max_tokens 16384`. PID 36543, 总 4694s (~78min, 比 1b 的 ~44min 长 — item 14 单题 flash 长 CoT 近 16384 cap 拖慢). 复用阶段①1b 已转好的 flash AIME25 S₀ cache (mean=0.8375, 16/30 满分, 14 信号点). S₀ 不重跑.

**结果**:
- **Δ=+0.0167, CI_lower(95%)=-0.0375** (CI 跨0, **不显著**), mean_pass=0.8458 vs S₀ 0.8375.
- per-item: 14/30 有信号 (8 正 6 负). 最大正 item23 (+0.500, 0.50→1.00); 最大负 item6/13/27/28 (各 -0.250). 信号抵消 → 均值微正.
- 产物: `rl/train/verl1/outputs/crossmodel_baseline/run_20261105_092706/baseline_flash.json` + `paired_eval_samples.jsonl` (480 records = 30题×8seed×2arm, base 虽有 cache 仍重跑取逐样本配对诊断).

**核心判读 — in-sample 有效但 held-out 不泛化**:

| 评测口径 | flash 自演化 evolver skill | 显著性 |
|---|---|---|
| **in-sample (AIME24 val)** | +0.078 | CI=+0.011 ✅ 显著 |
| **held-out (AIME25)** | +0.0167 | CI=-0.038 ❌ 不显著 |

→ 与历史完全同型: skill 在训练分布内有效, 迁到 held-out 回归噪声水平. 这是本线反复出现的结论 (run6d iter2 in-sample +0.042 显著 → AIME25 held-out -0.035/16384 -0.006; Direction1 rep5 同型).

**自演化 vs 跨模型迁移 — held-out 闭环对比**:

| 来源 | AIME25 held-out Δ | CI_lower | 显著? |
|---|---|---|---|
| V3.2-evolver → flash (迁移, 阶段①1b) | +0.0250 | -0.0208 | ❌ |
| **flash 自演化 (本补测)** | +0.0167 | -0.0375 | ❌ |

→ 两者 held-out 均**不显著**, 自演化 (+0.017) 甚至略低于迁移 (+0.025). **held-out 口径下 "自演化 > 跨模型迁移" 不成立** — 阶段②2a 在 in-sample 得到的 "自演化更优" 结论**未迁移到 held-out**. flash 自演化 skill 的 in-sample 增益 (+0.078) 同样有过拟合 held-out 的特征, 与 V3.2 历史一致.

**结论 (整体阶段②补测)**: 模型变强使 error-driven 优化在 **in-sample** 上从无效 (V3.2 -0.022) 变有效 (flash +0.078 显著), 但这份增益**不跨分布泛化** (held-out 不显著). 换模型未改变 "skill 优化过拟合训练分布" 这一结构性现象.

相关 bug: [[crossmodel-configs-aimp-20261030]], #31 (docs/phase2_issues.md).
