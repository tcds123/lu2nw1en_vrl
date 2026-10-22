---
name: aime-fresh-rerun-launch-20260807
description: AIME24/25 跨任务验证全链(fresh重跑)启动 — 全新EVAL_OUT避免与08-05阴性缓存冲突; 用户选"完整重跑"; 10:28 启动, 预计~2.5h
metadata:
  type: project
---

# AIME24/25 跨任务验证 — 完整重跑链 启动 (2026-08-07 10:28)

## 背景 (为什么重跑)
- 08-04/05 的 AIME 从头链返回**阴性**: v2+evolver 全 KEEP PREV, best skill == S0 不变, 所有 skill hash `6f0e7669`. 详见 [[aime-from-scratch-result-20260804]].
- 用户明确选择 **"完整重跑 AIME 链"**, 而非复用旧的 v2/evolver 产物.
- Run F (math live 题, 4-seed 0.5597) 现已是最强主结果; **AIME (不同题源) 是其终极泛化判据**.

## 配置 (与 run_aime_chain.sh 同参数, 唯一差异=全新 EVAL_OUT)
| 参数 | 值 |
|------|-----|
| v2 iters | 2 |
| v2 g | 4 |
| evolver iters | 4 |
| evolver K | 4 |
| eval_k (投票) | 3 |
| AIME25 test seeds | 4 |
| train_n | 18 |
| 技能对比 | s0, v2, evolver |

## 关键决策: 全新 EVAL_OUT (cache 清洁性) 🔥
- 旧 run_aime_chain.sh 用固定 `EVAL_OUT=tf_skill_aime_v2/aime25_test`, 该处有 08-05 阴性结果的 per-qid 缓存 (skill hash `6f0e7669`).
- 若复用, 若新链 v2/evolver 再次产出相同阴性 skill, 会**意外复用旧缓存** → 看似"新结果"实为旧阴性, 污染"完整重跑".
- 解法: 新脚本 `run_aime_chain_fresh.sh`, `EVAL_OUT=tf_skill_aime_v2/aime25_test_fresh_<TS>` = **全新空目录**, 零缓存冲突.
- v2/evolver 步骤自身写新 `run_<ts>` 目录 (清洁), `s0_cache` guard 未传 → 不跳过, 真跑.

## 启动
- PID 78083 (STAT=Ss, nohup setsid 独立会话抗 SIGHUP — Run F 死亡教训)
- 命令: `bash rl/train/verl1/run_aime_chain_fresh.sh`
- 日志: `rl/train/verl1/outputs/aime_chain_fresh_<TS>.log`
- 10:28 启动, Step 1/3 (v2_aime) precompute S0 in progress
- 预计总耗时 ~2.5h (v2 ~40min + evolver ~60min + AIME25 4-seed eval ~60min)
- ⚠️ 28 RPM 共享: Run F seed 扩样 (10:24) 已结束, 无竞争者, 可独占配额

## 判据 (参考 08-05 阴性基线)
- 08-05 阴性: v2/evolver 全 KEEP PREV, best==S0, hash 6f0e7669 全部相同 → 增益 0.
- 本链若 v2 或 evolver 产出 hash != 6f0e7669 的新 skill 且 AIME25 test 增益显著 (paired t, p<0.05) → **Run F 增益泛化到 AIME (跨任务泛化成功, 论文加分)**.
- 若再次阴性 (全 KEEP PREV) → 确认 AIME 是泛化边界, Run F 增益**不**跨题源泛化 → 论文需限定 claim (仅 live 数学题内).

## 产物路径
- v2: `outputs/tf_skill_aime_v2/run_<ts>/tf_global_skill_v2_aime.md`
- evolver: `outputs/skill_evolver_aime/run_<ts>/best_skill_evolved.md`
- AIME25 test: `outputs/tf_skill_aime_v2/aime25_test_fresh_<TS>/aime25_eval.json`

## 关联
- [[aime-from-scratch-result-20260804]] (08-05 阴性结果)
- [[runf-test-result-20260807]] (Run F 4-seed 最强主结果, 待其泛化验证)