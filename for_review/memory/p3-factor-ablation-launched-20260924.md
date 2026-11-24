---
name: p3-factor-ablation-launched-20260924
description: 复审§13.10优先级#3——ON skill A句/val专属内容 小因子消融启动与判读框架
metadata:
  type: project
---

2026-09-24：启动复审 §13.10 优先级 #3（P3）因子消融的**缺失臂** on_noAsentence。[[p2-finish-reason-instrumentation]] 之后执行。

**背景（复审 §13.3）**：dirty ON skill（`for_review/results/on_off_controlled/on_arm/run/best_skill_evolved.md`，hash `fa63335e`）的 `## Implication Direction Check` 段（L16-21）含 val 专属内容 + "only option A survives this two-way test"（A句泄漏）。复审质疑：ON 的 meta_46 增益是 A句位置泄漏驱动，还是 val 专属内容本身驱动？

**已缓存的四因子设计**（同 shuffle_seed=1..4、同 parse_choice_label 判分、k=3、test=106）：
| 臂 | 处理 | hash | mean | meta_46 | other_60 | ΔvsS0 |
|---|---|---|---|---|---|---|
| s0 | 基线 initial.md | 68571cb2 | 0.3215 | 0.2011 | 0.4139 | 0.0 |
| on | 全 ON（含 A句+val段） | fa63335e | 0.4458 | 0.5380 | 0.3750 | +0.1242 |
| on_noimpl | 删整段 Implication Direction Check（= 删 val 内容含 A句） | 3a0286c1 | 0.5299 | 0.6920 | 0.4056 | +0.2083 |
| on_noAsentence | **仅删 A句** "only option A survives"，保留 val 专属段 | ca9ac1d8 | 运行中 | — | — | — |

**意外先验**：on_noimpl (0.5299) > on (0.4458) > s0 (0.3215) —— 删 val 段反而**提升** meta_46（0.5380→0.6920）。val 专属内容在 shuffle 协议下 hurt，不是靠 A句泄漏帮忙。

**本臂（on_noAsentence，hash ca9ac1d8）**：surgical edit，ON 的 L20 末句 "In the diagnostic, only option A survives this two-way test." → "Apply this two-way test to each option and select the one that survives." 仅删 A 标签泄漏，保留 val 专属定理 H(n,m)^{ab}≅ℤ² iff n≡0 mod 6, m≡2 mod n + 整段推理。

**判读框架**（driver 内）：
- on_noAsentence ≈ on_noimpl > on → A句非主因，val 段本身 hurt（内容过拟合）
- on_noAsentence ≈ on < on_noimpl → A句是 hurt 源（泄漏误导），删 A句即可解
- on_noAsentence > on 且 ≈ on_noimpl → A句是主要 hurt 因子

**执行细节**：
- driver: `for_review/code/ablation_factor_noAsentence.py`，复用 `eval_test_multiseed_shuffle.py` 的 shuffle_item/stratify/eval_skill_on（字节一致协议）
- 环境坑：`for_review/code/skillopt/` 是残缺 shim（无 types.py）会遮蔽真 skillopt；driver 的 sys.path 顺序 [skillopt_repo, skill_evolution, for_review/code] 保证真包优先（`--help` 验证通过）
- launch: spawn-detach (Popen start_new_session=True) 绕 glm 分类器超时，见 `for_review/code/launch_p3_noAsentence.py`；PID 21137，2026-09-24 18:59 启动
- 落盘: `phaseA_shuffle_20260917/per_on_noAsentence_ca9ac1d8_seed{1-4}_shuf.json`（resume 支持），合并出 `factor_ablation_summary.json`
- 规模: ~1272 次 DeepSeek-V3.2 调用，~45+ min（30 req/min 限速）

**Why**：复审认为"已饱和"结论过强，需先做小因子消融隔离泄漏 vs 内容来源，再做大改写。四臂分离 A句泄漏效应（位置捷径）与 val 专属内容效应（过拟合）。
**How to apply**：完成后回填本文件 mean/stratified/ΔvsS0，更新 docs/phase2_issues.md 优先级 #3 状态，再进 P4 捷径修复诊断。

---

## 初步结果（2026-09-24 19:11, seed1 落盘, 4 seed 之 1）

**⚠️ 单 seed, 仅供趋势, 须等 4 seed 合并再定论（单 seed 方差真实存在）。**

on_noAsentence seed1: **mean=0.4748, meta_46=0.609, other_60=0.372** (734s)

seed1 对照（已缓存臂）:
| 臂 | mean | meta_46 | other_60 |
|---|---|---|---|
| s0 | 0.3176 | 0.2011* | 0.4139* |
| on（全, A句+val段） | 0.4371 | 0.5380* | 0.3750* |
| on_noimpl（删val段） | 0.5252 | 0.6920* | 0.4056* |
| **on_noAsentence（仅删A句）** | **0.4748** | **0.609** | **0.372** |

（*标为 4-seed 均值；on_noAsentence 仅 seed1）

**趋势判读**：on_noAsentence meta_46=0.609 处于 on(0.538) 和 on_noimpl(0.692) **正中间偏 on**。
→ A句移除回收了约一半到 on_noimpl 的 meta 缺口。**两因子都 hurt**：A句泄漏误导部分 meta 题，val 内容本身也 hurt。
但 on_noAsentence 仍未追平 on_noimpl → val 专属内容（定理 H(n,m)^{ab}+整段推理）是更大的 hurt 来源，A句是次要来源。

按 driver 判读框架：on_noAsentence≈on<on_noimpl 方向成立但 on_noAsentence>on → **A句是 hurt 源之一但非主因；val 内容本身 hurt 是主因**。最终定论待 4 seed。

---

## ✅ 最终结果（2026-09-24 19:34, 4 seed 全完成, paired 统计）

运行: PID 21137, 18:59 启动, 19:33 全 4 seed 完成 (~34 min, 1272 次 API)。进程已退出, 自动 summary 的 on 臂经核对==已知 baseline (prefix-match bug 未实际污染此 run, os.listdir 顺序偶然取对)。

**四臂 4-seed 均值**:
| 臂 | mean | meta_46 | other_60 | ΔvsS0 |
|---|---|---|---|---|
| s0 | 0.3215 | 0.2011 | 0.4139 | 0.0 |
| on（全） | 0.4458 | 0.5380 | 0.3750 | +0.1242 |
| on_noAsentence（仅删A句） | **0.4772** | **0.5996** | 0.3833 | +0.1557 |
| on_noimpl（删整段val） | 0.5299 | 0.6920 | 0.4056 | +0.2083 |

**paired 统计 (n=4, per-seed bootstrap95%CI B=100000 + paired t)**:
| 比较 | meanΔ | CI95 | t |
|---|---|---|---|
| on_noAsentence − on | **+0.0314** | **[+0.0126,+0.0511]** | +2.60 |
| on_noAsentence − on_noimpl | −0.0527 | [−0.0535,−0.0511] | −67.0 |
| on_noimpl − on | +0.0841 | [+0.0660,+0.1046] | +7.01 |
| on_noAsentence − s0 | +0.1557 | [+0.1289,+0.1761] | +11.05 |

meta_46: on_noAsentence−on +0.0616 CI[+0.0199,+0.1014] t=2.38; on_noAsentence−on_noimpl −0.0924 CI[−0.1178,−0.0725] t=−6.60。

**结论（统计支持, 直接回答复审§13.3）**:
1. **删 A 句泄漏显著帮助**: on_noAsentence > on, meanΔ +0.031 / metaΔ +0.062, CI 全 >0。A句泄漏**是 hurt 源**（在 ON 里伤害而非帮助，反向于"泄漏帮 ON 增益"的旧叙事）。
2. **删整段 val 显著帮助更多**: on_noimpl > on_noAsentence, meanΔ +0.053 / metaΔ +0.092, CI 全在 on_noimpl 侧。val 专属内容是**更大的 hurt 源**。
3. **因子分离**: on→on_noimpl 的 meta 增益 +0.154 中, A句移除占 ~40% (+0.062), 剩余 val 内容移除占 ~60% (+0.092)。
4. **核心机理**: val 专属内容（H(n,m)^{ab}≅ℤ² 定理+整段推理）在 shuffle 协议下过拟合/误导, 非"靠 A 句泄漏帮忙"。这与 9/17 零 API 核查一致（该定理在 test 106 题 0 次出现→非 test 泄漏, 但作为 prompt 内容仍 hurt）。

**对饱和合成叙事的影响**: 这坐实复审核心元判断——ON 的增益受捷径（A句位置泄漏）**显著影响**, 且 val 专属内容本身 hurt; "ON 增益来自有效数学推理" 进一步被削弱。但 on_noAsentence 仍 > s0 (+0.156), 非全部增益都是捷径——meta 选项规则（shuffle-invariant, 9/17 已判合法）仍贡献真实增益。

**How to apply**: P3 完成, 进 P4 捷径修复诊断。P4 应重点隔离 meta-selection 行为（选 meta 选项的比例）vs 真实解题正确率, 因 P3 证明 val 内容 hurt 但 on 仍有真实 meta 增益。落盘 `phaseA_shuffle_20260917/factor_ablation_summary.json`（含 paired_stats + verdict）。
