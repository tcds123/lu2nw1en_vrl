# F7 截断 bug — AIME evolver in-sample 路径全程跑在 4096 (2026-10-30 查实)

承接 [[dir1-rep5-fixed-result-20261029]] D1 结案后, 用户要求核实 "F7 修复是否真在 rep5-fixed 生效"。
**结论: 没生效, 但不影响 D1 +0.044 结论。** 详见 `docs/phase2_issues.md` 问题 #18。

## 核心发现 (与既有 F7 记录的区别)

已有 F7 记录 [[f7-16384-baseline-result-20261021]] / [[f7-truncation-confounder-20261021]] 讲的是
**held-out AIME25 评测** (走 `eval_aime24_multiseed.py:evaluate_aime`, F7 把 max_tokens 提到 16384
**在那条路径上生效了**)。

本次发现的是**另一条代码路径 = in-sample evolver 评测**:
- `tf_global_v2_aime.py:run_one` L119 **硬编码 `max_completion_tokens=4096`**
- `eval_on` (L298) / `eval_paired` / `precompute_s0` 全部经 `run_one` → evolver 主路径逐字 4096
- F7 修复写在 `tf_global_v2_math.py:47` (math 通用层 setdefault 16384), 但 AIME evolver
  `skill_evolver_aime_v2.py:57` 导入 `tf_global_v2_aime`, 后者只复用 `llm/set_temp/md5/...`,
  **不触发 math 层的 setdefault** → F7 的 env 层修复对 AIME 路径根本没加载
- env 层 (`rep5_fixed_launch.sh` export 16384 + 不 source `_env.sh`) 本身是对的,
  但被 L119 显式实参 `4096` 覆盖 (chat_target 收到显式实参时 env 默认值不生效)
- `evaluate_aime` 里的 16384 修复也空转: run_one 先用 4096 截断拿 response, 再交 evaluate_aime
  做纯解析 (不再请求), 给已截断的 response 设上限无意义

**口径**: held-out = 16384 (F7 生效), in-sample = 4096 (F7 未生效)。rep5-fixed 的 +0.044 是 4096 下算的。

## 为什么不影响 D1 结论 (+0.044 不是被截断压出来的假信号)

查 `paired_eval_samples.jsonl` 9607 条 (rep5-fixed run):
1. 99.8% `parse_status=parsed`; no_int 仅 21 条全是空 response (API 失败, 非截断)
2. max chars=16501 (≈4096tok×4 硬上限签名), 尾部被从句子中间切断 (`...So`), score=0.0 — 截断客观存在
3. **cand vs base 长度无系统差异** (cand mean=7657 / base mean=7666, 差 ~9 chars)
4. 长样本 (>14000, 270 条) score=0 在 cand/base 两侧 = 116/146, **双向非单边**
5. 长输出集中在 2-3 道题 (aime24:73/89/81), 同题 cand+base 都长 → "长"是题目属性非 skill 造成
→ delta = cand_pass − base_pass 里截断项两臂**对称抵消**, 不单向抬高/压低 +0.044

## 隐患 (为何仍须修, 即便本次不偏)

若未来某候选 skill **系统性改变输出长度** (prompt 让模型更啰嗦/更简短), 对称性破裂 → 截断变方向偏置。
修复应**优先于**引入任何改变输出长度的 skill 变体。

## 修复方向 (未实施)

1. `tf_global_v2_aime.py:run_one` L119 硬编码 4096 → 读 env 或传 16384 (**唯一真正生效的改动点**)
2. 删 `skill_evolver_aime_v2.py:38` 误导性 setdefault("4096")
3. `evaluate_aime` 的 16384 修复保留无害, 但不应记为 "F7 已修复截断"
4. `paired_eval_samples.jsonl` 应加 `finish_reason` 字段 (chat_target_messages 已支持), 关联 issue #17

## 教训

- "改了 setdefault" ≠ "修复生效": setdefault 在 env 已 export / 被显式实参覆盖时都是 no-op
- 修复要顺**实际调用链**改, 不能只改最靠近 env 那层; F7 改了 math setdefault + evaluate_aime,
  但 AIME evolver 走 run_one 这条没被碰的路径 → 追根因时先画调用链图再改, 否则改死代码
- "bug 存在" ≠ "混淆了结论": F7 客观存在且必须修, 但 cand/base 对称截断使 delta 中性

## 关联

- `docs/phase2_issues.md` 问题 #18 (本条是它的 memory 侧镜像)
- [[dir1-rep5-fixed-result-20261029]] — D1 结案 +0.044 (结论不受影响)
- [[dir1-rep5-fixed-launch-20261029]] — cap 16384 口径 (env 层对, 但被 L119 覆盖)
- [[f7-16384-baseline-result-20261021]] — held-out 侧 F7 确生效 (与本条 in-sample 4096 互补)
- [[measurement-calibration-1a-20261028]] — 检出下限 +0.10
