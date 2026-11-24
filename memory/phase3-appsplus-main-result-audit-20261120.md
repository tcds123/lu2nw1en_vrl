# Phase 3 — APPS Plus 主结果审计: 能否宣称 Evolver 有效 — 2026-11-20

完整文档: `docs/phase3_appsplus_main_result_audit_20261120.md`。全部 0 API。

## 一句话结论

**不能宣称 Evolver 有效。** 在 300 题同口径配对下, Arm1 (Evolver) 相对静态 S₀ / SkillOpt 的
增益为 +9 / +6 (p=0.32 / 0.47), CI 跨 0 且覆盖 δ=0.03; Arm3 为 −7 / −10。Arm0 用户暂缓。

## 口径统一 (本轮完成)

| 对象 | 定版产物 | hard (300) | 旧值 |
|------|----------|-----------|------|
| S₀ | `test_eval_baseline/results.rejudged_d5_v2.json` | 215 | 208 |
| SkillOpt | `test_eval/results.rejudged_d5_v2.json` (本轮新写) | 218 | 211 |
| Arm1 | `test_independent_results.json` 300 子集 | 224 | 2077/2765 |
| Arm3 | 同上 | 208 | 1945/2765 |

- SkillOpt 新产物 + 审计: `audit_0api_20261120/skillopt_cache_300_rejudge.json`,
  脚本 `eva/fix_48_skillopt_cache_300_v2.py`。id/code 与原 cache 逐条一致; 7 处翻转全 0→1。
- 两基线各翻 7 题、6 题重叠 → 旧 harness 对 fn-call 题的系统性欠给分, **协议修正对称, 非挑臂**。
- 历史数字换算: S₀ +7 (208→215), SkillOpt +7 (211→218)。

## 配对表 (300, 独立非路由, McNemar 精确 p, 净增益率 CP 仿射 CI)

| 比较 | 净增益 | rescue/regress | CI | p |
|------|--------|----------------|----|---|
| Arm1 vs S₀ | +9 | 37/28 | [−0.0258, +0.0830] | 0.3211 |
| Arm1 vs SkillOpt | +6 | 27/21 | [−0.0282, +0.0657] | 0.4709 |
| Arm3 vs S₀ | −7 | 24/31 | [−0.0722, +0.0282] | 0.4188 |
| Arm3 vs SkillOpt | −10 | 16/26 | [−0.0740, +0.0122] | 0.1641 |

产物: `audit_0api_20261120/phase3_paired_300.json` (`eva/phase3_paired_300_0api.py`)。

## 关键判读

- routed 结果 (Arm1 243 / Arm3 236) **不能**作为泛化证据: S₀-pass 题复用 S₀ 缓存 code, 相关且 Oracle 辅助。
- n=300 的 MDD≈11.2pp; 不把"功效不足"当主辩护: 点估计 3.0pp 本身就低于 δ, 且 2765 配对基线是**可补的数据缺口**。
- 最终判定: S₀ scaffold 正向贡献 ✅ (2765, 下界>δ); Evolver>静态 S₀ ❌; Evolver>SkillOpt ❌;
  Evolver>等预算对照 ❌ (Arm0 未启动); scaffold 成分分解 ❌。

## 注意点 (审计口径)

- Arm3 启动用的 test S₀ cache 是 215 定版, Arm1 用旧 208; 运行期映射到 215, best 218/legacy 211 一致,
  不影响独立比较, 复现说明须注明。
- 两臂串行、共享 20 RPM; 历史 88/300 与 293/1000 轮次零样本外增益, 提示训练提升→测试泛化不默认成立。

## 若后续要主张 Evolver 有效

最低要求: 补 2765 题配对静态基线 (S₀ 优先, 每基线 2765 次 flash ≈ 2.3h 纯节流), 不在 300 上继续做子分析。
补完后若仍无差异, 再讨论 Arm0 的归因价值。

关联: [[48-arm0-control-launch-20261120]] [[48-arm3-drop-scaffold-launch-20261119]]
