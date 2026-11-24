# 防截断反馈回传 — Step 0/1 修复 2026-11-17

## 起因 (用户判定)
run2 4 轮全 KEEP PREV 后, 我先前诊断为"统计效能不足"。**用户否掉了这个诊断**:
> "现在的关键问题是目前的算法设计无法得到带有减少截断目标的 skill, 无法将设计的防截断奖励因子回传到进化算法之中。"

这个判定是对的。统计效能是"量不够", 但先坏了的是"管路": 截断信号在到达晋升门之前就被丢掉了。
若按我原先的建议去提 k / 放宽门, 只会让**准确率**这条腿更容易过门, 防截断的那条腿依然不存在 —— 结论会虚涨。
**撤回提效能的建议。**

## 机制现状 (修复前, 三级断裂)
```
run_one        采集 harness{finish_reason, n_truncated, n_samples, n_out_tokens}
   ↓ 但注释写 "只记录诊断, 不改 pass 判定"
eval_on        只算 d = pr − s0.pass_rate, harness 被丢弃
   ↓
scored / eval_items/*.json   只有 {id, deltas, mean_delta, problem}
   ↓
晋升门         improved = delta>best ∧ ci>0  → 世界里没有"截断"这件事
```
叠加先验缺陷 (review A1): `extract_int_answer` 的「全文最后一个整数」兜底会在截断时抽到推理中间的
小整数, 偶尔撞真值 ⇒ **截断却答对**。方向是反的: 截断本应稳拿 0 分。

## 改了什么 (零 API, 均向后兼容)
| 步骤 | 文件 | 内容 |
|---|---|---|
| 0 | `eval_aime24_multiseed.py` | `evaluate_aime(..., finish_reason=)`; `length` → em 强制 0; 原值留 `em_untruncated`; 新增 `_extract_explicit()` (禁兜底); 开关 `TRUNC_SCORES_ZERO=0` 可回退 |
| 0 | `tf_global_v2_aime.py::run_one` | 传 `finish_reason` 进判分 |
| 0 | `paired_five_arm_v1.py` / `paired_budget_s0_v1.py` | 同上, 且 `call_b` 重试序列含截断也计截断; 落 `explicit_at_cut` |
| 1 | `tf_global_v2_aime.py` | `run_one` 追加 `n_clean`/`clean_tok_sum`/`clean_pass_tokens` (**仅 `finish_reason=='stop'` 样本**); `aggregate_harness()`; 模块级 `LAST_HARNESS_AGG`; `eval_on` 签名不变 (10 处解包 4 元组) |
| 1 | `skill_evolver_aime_v2.py` | 候选 `scored` 增 5 个 harness 字段; `dump_eval_items(harness=)` 落盘; 日志打 `tr=/ck=` |

## 验证 (run2 真实 trace 作 fixture)
- 111 条截断样本 → 新口径 em **全 0** (断言 0 例外)。
- 被扣掉的旧口径假阳性: **8/111** (7.2%)。截断前已显式写出答案的 (A 态) 只有 8~10 条。
- 口径差: `evolver_val` −0.56pp / `evolver_s0_trainval` −1.11pp / `evolver_diag` −0.28pp / `evolver_base` 0。
  **量级小, 但方向是对的** —— 说明 run2 的 delta 读数本身没有被这个 bug 大幅污染,
  所以 run2 "4 轮 delta 全正、CI 全部不过零" 的结论**仍然成立**, 不需要因为口径而推翻。
- 非截断路径逐字不变; 回退开关生效; 全链路回放通过; 6 文件 compile + 冒烟通过。

## 走过的弯路 (记下来避免重犯)
先改的是 `tf_global_v2_math.py::eval_on`。但 evolver 的 `run_one/eval_on` 是从 **`tf_global_v2_aime`** 导入的,
math 版对 AIME item 会 `KeyError: 'question'`。math 侧改动已**完整撤回**, 实现只留一处。
教训: 动手前先 `print(x.eval_on is y.eval_on)` 确认生效实现。

## 现在的状态 / 下一步
- 信号已能回传: 候选分数里 `n_truncated_rate` 与 `clean_pass_tokens` 与 `delta` 同处一个对象, 且落盘。
- **但晋级门仍未使用它们** (Step 3 未做)。即: 信号现在"看得见", 还不会"影响选择"。
- **Step 2 (未执行)**: 用新口径重跑 run2 (~1250 次调用), 确认 4 轮正 delta 是否仍成立。
  预期变化很小 (口径差 ≤1.1pp), 主要价值是让重跑成为后续 Step 3 的干净基线。
- **Step 3 (待用户定)**: 双门 + 方案六预登记阈值 (准确率 ≤ −2pp **且** token ≥ −20%)。
  风险: 硬性"截断率不许升"会诱导"写少", 故长度信号取 `clean_pass_tokens` 而非全部样本。

**关联**: issue #80 (本次修复) / #79 (统计效能) / #69 (门不含长度轴) / A1-A3 (review)
/ `memory/trunc-evolve-budget-run2-20261116.md`。
