# 关键发现: 两种 skill 进化算法实际写入 skill 的字符都极少 — 这可能就是"分数与 baseline 无差别"的直接原因 (2026-10-30)

## 结论 (一句话)

evolver 与 skillopt **都只往 skill 里写了 1k 字符量级以内的内容** (evolver 最终 0 改动; skillopt 2 条 edit / +2148 字符),
即 **skill 实际上几乎没被优化**。因此 test 端与 baseline 无差别不奇怪 —— 缺少的不是"选哪条", 而是"写进去的东西太少/太小"。

## 实测数据

### skillopt (两次 AIME run)

| | run1 `aime-skillopt-20261008` | run2 `aime-skillopt-skel-20261010-095515` |
|---|---|---|
| 起始 skill | 2409 字符 (≈602 tok) | 600 字符 (≈150 tok) |
| 24 步生成 edit 数 | 68 条 (`insert_after` 为主) | 94 条 |
| 生成内容总量 | 61,038 字符 (均 897/条) | 57,777 字符 (均 614/条) |
| **实际合并进 skill** | 仅 step6: 2 条 `insert_after`, **+2148 字符 → 2409→4614 (+89%), ≈+537 tok** | **0 字符** (24 步全 reject, `best_skill` == v0 逐字节) |
| 候选 skill 长度 churn | 4103→…→2361… 峰值 11,158, 均值 ~8.4k 字符 | 600→8696 (峰值 10,934), 均值 ~5.7k 字符 |
| 全程 API token | **2,237,198** (rollout 1,339k ≈60%, analyst 820k, merge 50k, ranking 28k) | **2,081,699** (rollout 1,207k, analyst 738k, merge 91k, ranking 46k) |
| selection_hard | 0.6667 → 0.8333 | 0.8333 不变 |
| **test_hard** | 0.8333 → 0.6667 → final **0.5** (picked skill 更差) | 0.6667 不变 |

- 落地率 = 2148 / 61,038 ≈ **3.5%**; 两次 run 合计 **~4.32M tokens** 只换来 2 条 edit。
- token 主开销在 rollout (评测候选), 不在生成 patch —— 和我们 evolver 的账是同构的。

### evolver (rep5-fixed 等)

| run | 算子尝试 | 晋升 | best skill 尺寸 |
|---|---|---|---|
| rep5_fixed_20261029 (5 轮, 1′ 已修) | 25 (facet 15 / overlay 5 / expand 5) | **0** | 599 → 599 (逐字节 = init) |
| dir1_step1_onflags (历史最佳) | — | 1 (overlay identity) | 599 → 705 (**+106 字符 ≈ +26 tok**) |
| rep5_20261023 (污染) | — | 1 (iter1 污染 +0.1042) | 599 → 794 (+195 字符) |

- overlay = 重建 4 个 section, in-sample Δ ∈ [−0.058, +0.083] 且事后证明是 identity;
- expand = 只加 1 个 section, Δ ∈ [−0.0375, +0.054];
- B 模型代码产出长度不受影响: paired_eval_samples 中位 completion base vs cand = 7973/7975, 7969/8099, 8000/7891, 8070/8176 字符。

## 对研究方向的含义

- 之前 Direction 1 (选择规则) / B4 (cross-Q gate) 的结论"转 generation 层"是**方向对但理由要补强**:
  更直接的证据是 **写入量本身就处在噪声量级以下** — 改 100~2000 字符的 skill 想撬动 pass@k, 要求单字符效应极强;
  实测 Δ 全在噪声带内, 与"改动太小 ⇒ 无可测效应"完全自洽。
- **skillopt 的失败模式**与 evolver 不同但同因: evolver 是"算子产出的候选 = 父 skill 或其微小变体 → 全被门拒";
  skillopt 是"候选不断膨胀到 8k–12k 字符 (3–5 倍 v0), 但 gate 每步 reject → 只有一步落地, 且落地后 test 变差"。
  两者共同点: **真正进入 skill 的内容 < 2k 字符, 且没有任何一处的效应 > 噪声地板 (~+0.10)**。
- 下一步 generation 层如果只做"更好的重写/更细的编辑操作", 很可能重复同样结局。真正要回答的是:
  **skill 容量/改动幅度与可测效应之间是否存在阈值** (即改多少字符才能把 pass@k 抬过 +0.10)。
  建议今后任何 skill 进化实验都**默认报"写入字符数 + 写入 token 数"作为一等指标**, 与 Δ 并列。

## 测量口径 (必须遵守)

- **`train/skills/skill_vNNNN.md` 存的是 patch 原文, 不是实际生效的 skill**。真实写入看 `steps/step_XXXX/candidate_skill.md` 与 `train/best_skill.md`。直接用 `skill_vNNNN.md` 长度会把"提议量"当"写入量", 虚高 10 倍。
- skillopt 每步字段: `step_record.json` 的 `tokens` / `n_patches` / `edit_budget` / `candidate_skill_len` / `edit_apply_summary` / `action`; patch 在 `steps/step_XXXX/patches/minibatch_{fail,succ}_NNN.json` 的 `patch.edits[].content`。
- evolver 侧"是否改动"用 `.strip()` 后 md5 对比 (identity `c11d7dc0c8` = `data/skill_aime_initial_v2.md` 的 strip 摘要)。

## 数据来源

- `rl/train/verl1/outputs/tf_skill_aime_v2/skillopt/{aime-skillopt-20261008, aime-skillopt-skel-20261010-095515}/train/`
- `rl/train/verl1/outputs/skill_evolver_aime_v2/run_dir1_step1_onflags/{rep5_fixed_20261029, rep5_20261023, run_20261022_091146}/`

## 关联

- [[dir1-rep5-fixed-result-20261029]] — evolver 0 晋升 + placebo
- [[b4-result-20261029]] — B 侧同步结论
- [[dir1-executable-checklist-20261029]] — 转 generation 层的重启条件
