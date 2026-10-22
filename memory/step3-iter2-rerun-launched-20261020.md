---
name: step3-iter2-rerun-launched-20261020
description: F0x Step 3 启动配置 — 硬化评分器 + 持久化下重跑 iter2 cand-vs-init paired eval (3× k=8 pooled) + Step 4 旧/新评分器离线分解 (零采样混淆)
metadata:
  type: project
---

# F0x Step 3 启动配置: iter2 重跑 + 评分器分解 (2026-10-20)

承接 [[aime-grader-truncation-bug-20261020]] 四步审计. Step 1 (`_to_int` 严格化) + Step 2 (`eval_paired` 持久化) 已落地零回归. 本文件 = Step 3+4 启动前记录 (CLAUDE.md 硬规则).

## 实验目标 (用户 verbatim)

> "重新运行 iter2 实验, 在硬化评分器 + 持久化的条件下, 验证 +0.042 是否稳定, 并记录逐题配对结果. 如果新评分器下 +0.042 仍 CI>0 → 口径稳健, bug 影响可忽略, 原结论巩固; 如果 +0.042 下降或不再显著 → 部分归因评分器偏差, 需进一步分析."

## 隔离变量纪律 (用户强调)

"除修复必要基础设施外, 尽量保持其他实验条件不变." 故**不重跑整个 evolver** (会重新生成候选 → 引入采样混淆). 只重测 run#6d iter2 的**那一个决定性配对评测** (cand=iter2 best_skill vs parent=init), 仅改必要基础设施 (评分器硬化 + 持久化).

## 固定量 (isolation)

| 维度 | 值 | 来源 |
|---|---|---|
| cand_skill | run#6d `best_skill_iter2.md` (init + 1 Quantity Parsing bullet) | 读 run6d dir, **非重新生成** |
| base_skill (parent) | `data/skill_aime_initial_v2.md` (init) | diff 确认 iter2 best = init + 1 bullet; iter1 QP 候选 +0.029 未晋 → 分支 leaf 仍=init |
| items | `load_aime_split(aime24.jsonl, train_n=18, seed=42)` → train+val=30 | 确定性, 与 run#6d 同 |
| k | 8 | run#6d `eval_k=8` |
| EVAL_TEMP / GEN_TEMP | 0.3 / 0.8 | tf_global_v2_math 模块常量 |
| deadline | 900s/batch | run#6d `EVAL_DEADLINE_S=900` |
| 两批串行 | eval_paired 内置 (base 30 tasks → cand 30 tasks, 各 40 worker) | run#6d 修复后设计 |
| backend | DeepSeek-V3.2 via aimpapi.midea.com | TARGET_BACKEND=qwen_chat (模块 setdefault) |
| RPM | 28 (模块默认, **不 source _env.sh** 避免 RPM=14 偏移) | run#6d 启动块未 source _env.sh → RPM=28 生效 |

## 改变量 (仅必要基础设施)

1. **评分器**: strict `_to_int` (Step 1, `SCORER_VERSION="aime_int_strict_20261020"`). 旧 `_to_int_legacy` 只读保留.
2. **持久化**: `eval_paired(persist_ctx=...)` ON (Step 2). 落 `paired_eval_samples.jsonl` 11+ 字段含 raw_response + scorer_version.

## 统计功效设计 (关键)

原 +0.042 CI 下界=+0.004 → SE≈0.019. 单次重跑仅 ~60% 功效显示 CI>0 (即便真效应属实). 故**跑 3 次独立 paired eval** (同 k=8, 自然 API 非确定性提供方差), 报告每次 + 池化. 池化 SE≈0.019/√3≈0.011 → 若真 delta=+0.042, 池化 CI 下界≈+0.020>0, 功效升至 ~85%. 3 次是"验证稳定性"的合理最小数, 非改条件.

## 判定标准 (用户 verbatim)

- 池化 CI 下界 > 0 (且 3 次方向一致正) → 口径稳健, bug 影响可忽略, 原 iter2 +0.042 结论巩固.
- 池化 CI ≤ 0 或 delta 显著下降 → 部分归因评分器偏差, 需进一步分析 (进 Step 4 量化 + 考虑 Step 5).

## Step 4 (零采样混淆分解, 内联做)

Step 3 持久化的 raw_response 上, 分别用:
- 新评分器 `_to_int` (strict fullmatch)
- 旧评分器 `_to_int_legacy` (re.search 截断)

对**同一批**输出重新评分 → old-scorer delta vs new-scorer delta. 差值 = 纯评分器效应 (零采样混淆, 因输出不变). 量化: 多少 sample 触发 bug (boxed{N.M}/boxed{N/M} 旧接受→新拒绝), cand vs base 分布是否对称. 这直接回答"bug 贡献多少给 +0.042".

## Step 5 (单独一轮, 不混改)

收紧 rescue `_eps_floor` (0.05→更小). 本轮**不碰**, 隔离变量.

## 启动命令

```bash
source /data/zhuldz/lunwen/lunwen/bin/activate
cd /data/zhuldz/lunwen/rl/train/verl1/skill_evolution
# 注意: 不 source _env.sh (保持 RPM=28 与 run#6d 一致; API key 由模块 setdefault 提供, 同值)
python step3_rerun_iter2.py
```

输出目录: `rl/train/verl1/outputs/skill_evolver_aime_v2/step3_iter2_rerun_20261020/`

## 风险

- 白天 15:00 启动, run#6d iter8 白天端点过载 11/30 丢题. 两批串行 (40 worker=baseline 级负载) 缓解, 但非零风险. 丢题记录在持久化 (parse_status=abandoned), 若 >20% 标记混淆, 考虑夜间补跑. paired 丢题无系统性偏置 (cand/base 独立提交), 只增方差不偏 delta.
- 3× paired eval ≈ 1440 API req, 预计 ~1.5-2h (latency-bound).

## 待办

- [ ] 启动 Step 3+4 脚本, 后台运行
- [ ] 完成后: 结果 + 判定写 `memory/step3-iter2-rerun-result-20261020.md` + 更新 MEMORY.md 索引 + 追加 phase2_issues.md
- [ ] 若丢题 >20%: 夜间补跑

关联 [[aime-grader-truncation-bug-20261020]] [[run6d-iter2-real-promotion-20261019]] [[run6d-results-20261019]].
