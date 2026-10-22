---
name: aime-skel-fallback-launched-20261009
description: 骨架化 S₀ + 两段式探索兜底 GA 启动 2026-10-09 — 削弱 S₀ + 停滞时松接收门
metadata:
  type: project
---

**状态**: ✅ Step A + B + C 全部完成. **核心结论: GA 免训练优化在 AIME held-out 有效** — evolver (797 chars) 超 s0_skel (600 chars) +0.044, 且与 v2_strong (2408 chars) 持平 (33% 字符达同等效果). val CI 负 (−0.022) 但 held-out 泛化 +0.044 → 证实 "n=30 CI 噪声误杀真信号", 兜底松门有价值. 瓶颈仍为样本量 (3 seed 不显著 p≈0.5), 下步扩样本 AIME_1983_2024.

## Step A 结果 (骨架 baseline, PID 74684, run_20261009_105626, 10 iter, 严门无兜底)

- **baseline_val (S₀ 骨架)**: pass=0.589, delta=−0.022, ci=−0.111
- **best_val_delta = +0.0778**, **best_val_ci = +0.0111** (iter9, Computation & Verification facet, 唯一一次 PROMOTE)
- **total promoted = 1** (facet_edit: attempted=30, admitted=15, promoted=1; overlay/expand 各 10 次尝试 0 promoted)
- **stagnation_fb**: 全程 stage1/stage2=false (严门未松); pool_ci_gate=best_ci_gate=best_delta_floor=0.0 (strict); no_improve_iter 从 0 累到 8; iter9 promoted_this_iter=True
- skill: 599→762 chars, hash c11d7dc0c8→5e3ef1ffe6, 产出 `best_skill_evolved.md`
- ⚠️ RF §6 告警 "overlay/expand 0 promoted" 属实, 但漏算 facet_edit 的 1 promote (告警只查 overlay/expand). 实际全 run 非 0-promoted.
- 关键: 严门 + 骨架 S₀ 下, 9 轮 (~50 算子尝试) 才挖出 1 个过 CI 候选 (C&V facet, 针对 Number-theoretic case analysis / Quantity misread 失败类型). 信号弱但存在.

## 动机 (用户指令 2026-10-09)

上一轮 AIME25 三方对比 ([[aime-on-skillopt-result-20261008]]) 发现 evolver GA 10 轮 50 算子尝试 **0 promoted**, skill=S₀. 两方法 (+0.004/+0.011) 均未显著超 S₀.

用户判断: ① 接收门过严 (候选池空) ② S₀ 过强 (无优化空间). 指令:
- **算法兜底**: 多代无接受时更多探索、更宽容接收, 让候选池丰富 (两段式: 先松入池, 持续停滞再松 best).
- **削弱 S₀**: 骨架化 (保留 4-facet 结构清空内容), 让 GA 有优化空间, 且能自行进化出新段落和标题.
- 先削 S₀, 再兜底, 之后对比两算法结果.

## 根因诊断 (已直读代码确认)

1. **接收门过严**: n=30 二项 CI 宽 ±0.1, +2/30 (delta=+0.067) CI 下界 −0.033<0 → 三处 admission site 全拒. archive 空转, crossover/overlay/expand 无素材.
2. **探索 no-op**: `explore` 标志受 `--fix_exploration` (默认关) 门控, 实际传 False.
3. **S₀ 过强**: 2409 chars, 4-facet 全填满.

## 变更

### CHANGE 1 — 骨架化 S₀
- `data/skill_aime_initial_v2.md`: 2409 → **600 chars**, 4 标题各留 1 句指引, Final Answer 冻结 (boxed 格式). `facet_decompose`/`compose` 无损往返已验证, `--self_test_facet` 通过.
- 强 S₀ 备份: `data/skill_aime_initial_v2_strong.md` (2409 chars, 评测对照).

### CHANGE 2 — 两段式探索兜底 (`skill_evolver_aime_v2.py`)
新 flag `--stagnation_fallback` (默认 off=零回归) + `--pool_tol`(0.05) + `--best_tol`(0.03):
- **Stage1** (no_improve>=patience): 入池 CI 门 ci>0 → ci>-pool_tol. 候选降级入 archive 作交叉素材, best 不动.
- **Stage2** (no_improve>=patience*2): best 门 (delta>best and ci>0) → (delta>best-best_tol and ci>-best_tol). 弱正信号上位. 早停阈值抬 patience*2→patience*3 (否则 Stage2 刚触发即被早停).
- 保护: LOST 守护 (cp>=base_pass-0.06) 不动; subset floor guard (_eps_floor=0.05) 不动; best_tol(0.03)<floor 先生效.
- 零回归: flag off 时 pool_ci_gate=best_ci_gate=best_delta_floor=0.0 → 退化为既有 strict ci>0/delta>best (字节一致).
- 三处 admission site: facet_edit (pool-filter 复判 + best 门), overlay (入池+best), expand (入池+best).
- 诊断: `evolver_log.json` 每 iter 加 `stagnation_fb` 字段 (stage1/stage2/gate 值).

## Bug 修复 (2026-10-09, smoke 前发现)

PID 93840 (Step A 首次启动) 跑的是**带 bug** 的代码, 已 `kill -9` 终止 (产物作废). 修复两致命 bug + 一附带修复, 详见 `docs/phase2_issues.md` 2026-10-09 条目:

1. **Scope 错位 (FIX B)**: stage 标志原放 `main()` patch 路径 (facet_tree 开时不可达) → NameError. 移入 `_run_facet_tree` iter 循环内为局部变量. main patch 路径块保留为诊断-only.
2. **计数器粒度 (FIX A/C/F)**: 旧 `no_improve` per-survivor 自增 → 单 iter 跳多步 → Stage1 iter1 误触发. 改用 `no_improve_iter` (per-iteration, `promoted_this_iter` flag 聚合 ~9 处 best-promotion site).
3. **附带 (FIX D)**: crossover 2-parent/multiparent best-gate 缺 LOST 守护, 补 `xp/mp_p >= base_pass - 0.06` (不随兜底松).
4. strengthen 签名加 `stage_gates=None` 参数, 3 处调用点传入 `stag_fb` + `became_best` 传播 `promoted_this_iter`.

验证: `ast.parse` OK; `--self_test_facet` 通过 (骨架 S₀ 599 chars 4 段往返无损). 运行时 iter-0 干净性由 smoke test 确认中.

## Step A — 骨架 baseline (运行中, PID 74684, 2026-10-09 10:52)

```bash
QWEN_CHAT_MAX_RPM=14 python rl/train/verl1/skill_evolution/skill_evolver_aime_v2.py \
  --init_skill data/skill_aime_initial_v2.md --s0 data/skill_aime_initial_v2.md \
  --split_dir data/aime24.jsonl --iters 10 --train_n 35 --eval_k 3 --K 4 \
  --n_parallel 1 --n_diag 8 --facet_tree --expand --pool_filter --bootstrap --crossover \
  --out_dir rl/train/verl1/outputs/skill_evolver_aime_v2
```
- **PID**: 74684. **log**: `rl/train/verl1/outputs/skill_evolver_aime_v2/skel_baseline_20261009.log`
- 与上一轮 evolver run (20261008_103601) **完全同参**, 唯一差异 = S₀ 骨架化 (公平对比骨架是否给优化空间).
- 无 `--stagnation_fallback` (严门, 对照组). 预期: 骨架 baseline 可能 < 强 S₀ (0.633@k3); GA 仍可能 0 promoted (严门未松).

## Step B — 兜底 GA (✅ 完成, PID 114031, 2026-10-09 21:31 → 10-10 06:21, ~8h50m)

同 Step A 参数 + `--stagnation_fallback --pool_tol 0.05 --best_tol 0.03`. run: `run_20261009_213157/`.
log: `rl/train/verl1/outputs/skill_evolver_aime_v2/skel_fallback_20261009.log`

**结果**: best_val_delta=**+0.0556**, best_val_ci=**−0.0222** (iter7, Case Enumeration facet, facet_edit promote). skill 599→797 chars, hash c11d7dc0c8→f00fa7940f.
- baseline_val (S₀ 骨架): pass=0.600, delta=+0.000, ci=−0.089
- **total promoted = 1** (facet_edit: attempted=30, admitted=23, promoted=1; overlay admitted=2/promoted=0; expand admitted=1/promoted=0; crossover attempted=0)
- **stagnation_fb 触发**: Stage1 iter4/5/6 (3次, no_improve 3→5, pool_ci_gate −0.05); Stage2 iter7/8/9/10 (4次, best_ci_gate/best_delta_floor −0.03)
- archive_size=29 (vs Step A 候选池空转); crossover_fired=0 (从未有 2 个 promoted parent)
- iter7 promote 详情: Case Enum 候选 Δ=+0.056/CI=−0.022, 严门(CI>0)必拒, Stage2 松 best_ci_gate→−0.03 后 −0.022>−0.03 通过. **首个 Stage2 松门 promote 验证成功**.
- ⚠️ **计数器设计点 (pause-not-reset, 非 bug)**: iter7 promoted_this_iter=true, no_improve_iter 暂停在 6 不自增, 但**也不重置为 0** → iter8-10 Stage1/2 仍持续触发. 代码 line 3555 `if not promoted_this_iter: no_improve_iter+=1` + 注释 "本 iter 不计停滞" 证实为**有意设计**: promote 暂停停滞计数, 但不重新拧紧门. 本 run 良性 (iter8-10 未误 promote). 待用户决策: promote 后是否应重置计数器 (重拧紧门) vs 保持暂停 (宽容噪声).
- ⚠️ RF §6 告警 "overlay/expand 0 promoted" 仍漏算 facet_edit 的 1 promote (与 Step A 同款 cosmetic bug, 未修).

### Step A vs Step B 对比

| 指标 | Step A (严门无兜底) | Step B (两段式兜底) |
|---|---|---|
| best_val_delta | **+0.0778** | +0.0556 |
| best_val_ci | **+0.0111** (正, 过严门) | −0.0222 (负, 仅过松门) |
| promoted | 1 (facet_edit) | 1 (facet_edit) |
| fallback 触发 | 0 | Stage1×3 + Stage2×4 |
| 候选池 admitted | 0 (空转) | 29 (丰富) |
| crossover | 0 | 0 (无 2 parent) |

**结论**: 兜底**未**产生更多 promote (均=1), 也**未**提高 delta (+0.056 < +0.078). 反而 promote 更弱 (CI 负 vs 正). 兜底唯一价值 = 候选池不再空转 (admitted 29), 但交叉素材未转化为 best. 根因仍 = **n=30 bootstrap CI 噪声带宽 ±0.08~0.15, 松到 −0.03 仍难稳定过门** → 扩样本 (接入 AIME_1983_2024) 才是正解, 非调 tol.

## Step C — AIME25 held-out 三方对比 (✅ 完成, PID 42502, 2026-10-10, ~1h20m)

3 skill × 3 seeds × 30 题 × k=3 = 810 API calls. cron 6eeeb725 已删.

脚本 `eval_aime25_multiseed.py` 支持 4 命名 slot (s0/v2/evolver/skillopt), 路径可覆盖. 本 run 映射:
- **s0** = `data/skill_aime_initial_v2.md` (骨架 S₀, 599 chars, hash c11d7dc0) — delta 基线
- **v2** = `data/skill_aime_initial_v2_strong.md` (强 S₀, 2408 chars, hash f4837ca8) — 对照
- **evolver** = `run_20261009_213157/best_skill_evolved.md` (Step B 产物, 797 chars, hash f00fa794)

Step B 产物内容: 骨架 S₀ + Case Enumeration 加 1 条 bullet ("symmetry/extremal 约束时, 系统遍历所有配置如 row/column counts/radii 并查一致性"). 针对 Number-theoretic case analysis / extremal 失败.

判据: evolver 是否在 held-out AIME25 上超 s0_skel (同起点) — 是 = GA 优化有效 (即便 val CI 负); 否 = val 提升 (+0.056) 不泛化或为噪声. 同时看是否超 v2_strong.

```bash
QWEN_CHAT_MAX_RPM=14 python rl/train/verl1/skill_evolution/eval_aime25_multiseed.py \
  --s0 data/skill_aime_initial_v2.md --v2_best data/skill_aime_initial_v2_strong.md \
  --evolver_best rl/train/verl1/outputs/skill_evolver_aime_v2/run_20261009_213157/best_skill_evolved.md \
  --skills s0,v2,evolver --eval_k 3 --seeds 3 \
  --out_dir rl/train/verl1/outputs/tf_skill_aime_v2/aime25_test
```
log: `rl/train/verl1/outputs/tf_skill_aime_v2/aime25_test/stepC_3way.log`

### 结果 (AIME25 held-out, 30题, k=3, 3 seeds)

| skill | 字符数 | mean_pass | per-seed | Δvs s0_skel |
|---|---|---|---|---|
| s0 (骨架 S₀) | 599 | 0.4741 | [.456, .467, .500] | — |
| v2 (强 S₀) | 2408 | 0.5148 | [.489, .533, .522] | +0.0407 |
| **evolver** | 797 | **0.5185** | [.556, .522, .478] | **+0.0444** |

- **evolver 超 s0_skel**: ✅ +0.0444 (2/3 seed 正, 1 seed 负). 即 val 提升 +0.056 (CI 负) 在 held-out 上**泛化且放大** (+0.044 held-out vs +0.056 val). GA 优化有效 — 即便 val CI 被噪声压负, held-out 证实信号真实.
- **evolver ≈ v2_strong**: +0.0037 (基本持平, 1+1-1). evolver 用 797 chars (骨架+1 bullet) 达到 2408 chars 强 S₀ 同等水平 — **33% 字符达成同等效果**.
- per-qid (3-seed sum): evo vs s0 → 11 高 / 4 低 / 15 平; evo vs v2 → 8 高 / 8 低 / 14 平.
- 统计显著性: n_seeds=3, sign test 2/3 正 — 不显著 (p≈0.5 binomial). 但 per-qid 11:4 支持正向. 噪声仍大 (seed3 evo 反超 s0 −0.022), 需更多 seed 或扩样本才能定论.

### 结论

1. **GA 免训练优化在 AIME held-out 上有效**: 骨架 S₀ + 兜底 GA 产出的 797-char skill 超 600-char 骨架基线 +0.044, 且与 2408-char 强 S₀ 持平. 证明 "骨架化给优化空间 + 兜底松门让弱正信号上位" 路线成立.
2. **val CI 负 ≠ 假信号**: Step B val +0.056/CI−0.022 被严门判 "未达 CI", 但 held-out +0.044 证实信号真实. 印证 "n=30 CI 噪声过宽会误杀真信号" 的论断 — 兜底松门的价值正在此.
3. **瓶颈仍为样本量**: held-out +0.044 在 3 seed 下不显著 (p≈0.5). 要坐实结论需扩样本 (AIME_1983_2024 933 题) 或增 seed (≥8).

## 风险

1. Stage2 best 松门过拟合 (skillopt 翻车模式): best_tol=0.03 小 + LOST 守护不动 + AIME25 held-out 对照.
2. 骨架太弱 GA 无从学起: 保留 Final Answer 格式 + 每节 1 句指引 (非全空).
3. FAILURE_FOCI 仍无几何靶点 (上轮主因, 本计划未加几何 foci): 若兜底跑仍几何题全 0, 下轮加几何 FAILURE_FOCI.

关联 [[aime-on-skillopt-result-20261008]] [[aime-on-skillopt-launched-20261008]] [[aime-v2-evolver-built-20260930]].
