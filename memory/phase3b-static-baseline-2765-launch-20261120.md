# Phase 3b — 补 2765 题静态基线 (S₀ 先, SkillOpt 随后) 启动记录 — 2026-11-20

## 用户决策 (2026-11-20)

Phase 3 审计 (300 题, `memory/phase3-appsplus-main-result-audit-20261120.md`) 结论"不能宣称 Evolver 有效"后,
用户批准: **补齐 2765 题配对静态基线, 先补 S₀, 再补 SkillOpt; Arm0 暂缓**。
定性: **结果揭示后的基线补测** (post-revelation), 不是原始预注册确证实验。

## 冻结协议 (首次 API 调用前)

`docs/phase3b_static_baseline_2765_protocol.md` (2026-11-20 写定)。要点:

| 项 | 值 |
|----|----|
| 测试集 | `appsplus_capped_seed42/test/items.json` 全部 2765, sha `a3e1509b…c1e71c` |
| 基线 skill | S₀ `initial.md` 1811B sha `40c8886c…671f4d2`; SkillOpt `best_skill.md` 3000B sha `a4f8518c…a8ef79bea` |
| 调用 | qwen_chat / aimpapi / `deepseek-v4-flash` (可变别名) / T=0.0 / max_tokens=16384 / 非流式 / 不传 thinking |
| 重试 | retries=5, timeout=300s, 429 退避 30–62s, trunc retry ×1 (≤32768) |
| 评测 | `_evaluate_d5`; 生成路径复刻 `run_one` (`_build_system`/`_build_user`) |
| 调度 | 8 workers, `(S₀ id, SkillOpt id)` 交替提交, 20 RPM bucket, 100 题/块, 块级断点续跑 |
| 产物 | `rl/train/verl1/outputs/crossmodel_baseline/phase3b-static-baseline-2765-20261120/run_<ts>/` |

脚本: `rl/train/verl1/skill_evolution/phase3b_static_baseline_2765.py`。
key 运行期从 `configs/models_aimp.local.yaml` 读, 不打印不落盘。

## 漂移检查 (启动前, 8 题)

- 题: items.json 前 8 个 ID, 以 S₀ 重生成, 与 `test_eval_baseline/results.rejudged_d5_v2.json` 比较。
- 判读: ≥7/8 一致 → 未见明显漂移; ≤6/8 → 报告显式记为漂移风险 + 敏感性分析 (不中止)。

## 执行状态

- [ ] 漂移检查 8 题
- [ ] 全量 2765 × 2 (预计 ≈4.6h 纯节流, 实际含时延约 6–8h)
- [ ] `--finalize` → `all_results.json` / `run_summary.json`
- [ ] Phase 4 配对分析 (Arm1 vs S₀ / SkillOpt / Arm3), 0 API

## 已知局限 (报告必须写入)

结果揭示后补测; 臂 (11-18/19) 与基线 (11-20) 时序差; 模型为可变别名无法确认版本;
单题单样本; Arm0 仍缺 → 即使 Evolver > 静态基线也不能把增益完全归因于利用训练反馈。

关联: [[phase3-appsplus-main-result-audit-20261120]] [[48-arm3-drop-scaffold-launch-20261119]]
