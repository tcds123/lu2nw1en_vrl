---
name: b1-flash-skillopt-lcb-result-20261109
description: B1 完成 - flash SkillOpt on LCB Δ+4.9pp 翻转 V3.2 负结论(Δ-1.4pp), hard-strat +6pp, 同方法同数据只换强模型 skill 优化由负转正
metadata:
  type: project
---

# B1 结果: flash SkillOpt on LCB (2026-11-09 完成)

## 一句话结论
**同方法(SkillOpt)/同数据(LCB)/只换强模型(V3.2→flash), skill 优化从负(Δ-1.4pp)翻转为正(Δ+4.9pp).**
这是 V3.2 判死前提被推翻后的关键验证: 换强模型, skill 优化才有正杠杆.

## 完整指标 (summary.json, 2026-11-09 18:54 落地, wall=12026s≈3.3h, 963 calls, 8.0M tokens)

### 顶层 hard pass-rate (test=143, SkillOpt 内部 eval)
| 项 | 值 | 对照 |
|---|---|---|
| baseline (S₀ init) | 0.4685 | V3.2 baseline 0.4406 |
| test (best step 6) | 0.5315 | — |
| final (best skill) | 0.5175 | V3.2 final 0.4266 |
| **Δ final-baseline** | **+0.0490** | V3.2 Δ **-0.014** → 符号反转 ✅ |
| best_step / accepts / rejects | 6 / 3 / 5 | V3.2: best_step=3 |

### 难度分层 hard pass-rate (baseline → final) — 收益集中在 hard
| 难度 | n | base | final | Δ | S₀基准 | V3.2判死 |
|---|---|---|---|---|---|---|
| easy | 34 | 0.9412 | 0.9706 | +0.0294 | 0.9706 | 0.9410 |
| medium | 26 | 0.6154 | 0.6538 | +0.0385 | 0.7692 | 0.5770 |
| **hard** | 83 | 0.2289 | 0.2892 | **+0.0602** | 0.5542 | 0.1930 |

- hard 段 +6pp 是最大收益 — 符合 "skill 在难题有杠杆" 的假设.
- 但 flash hard 绝对值 0.2892 仍远低于 S₀ 0.5542 → headroom 仍大 (见下 caveat).

### epoch trajectory (4 epochs, 8 steps)
- ep1: acc1/rej1, best=0.75; ep2: rej2 (停滞); ep3: acc2/rej0, best=0.85 (step6 接受);
  ep4: rej2 (收敛, 不再提分).
- 收敛于 step 6, 后续 step7-8 被拒 → 优化已饱和, 非 early-stop 问题.

## ⚠️ 关键 caveat: S₀ 独立 cache (0.6923) ≠ SkillOpt 内部 baseline (0.4685)
- S₀ standalone (flash-lcb-s0, 2026-11-06): test hard=**0.6923**, hard-strat=0.5542.
- SkillOpt 内部 S₀-init re-eval: test hard=**0.4685**, hard-strat=0.2289.
- 同 skill (initial.md md5=22dc2c3e 已验), 同 split (split_dir 复用, test=143 hard=83), 同模型 flash, 同 cap 16384.
- 22pp / 32pp(hard-strat) 差距 = **eval harness 不同**, 非数据/模型差异. 最可能: S₀ 走 pass@k(多采样取最优), SkillOpt 内部 eval 走 pass@1(单次) + 不同 code 提取/格式.
- **正确判读**: Δ+4.9pp 是 SkillOpt 内部同口径比较 (baseline→final 同 harness), 是与 V3.2 (Δ-0.014 同 harness) 对比的干净信号. S₀ 0.6923 是不同口径的校准点, 证明模型有 headroom, 但不作 delta 锚点.
- 待查 (→ docs/phase2_issues.md): SkillOpt LCB evaluator 的 k/temperature/code-extract 与 S₀ standalone 的差异.

## 对照全景
| run | model | baseline | final | Δ | 判定 |
|---|---|---|---|---|---|
| V3.2 判死 | V3.2 | 0.4406 | 0.4266 | -0.014 | ❌ 过拟合, 判死 |
| **B1** | **flash** | **0.4685** | **0.5175** | **+0.049** | ✅ 正向, 翻转 |
| S₀ standalone | flash | 0.6923 | — | — | (不同口径, headroom 校准) |

## 产物路径
- out_root: `rl/train/verl1/outputs/crossmodel_baseline/lcb-skillopt-flash-20261109-152902/train/`
- summary.json, best_skill.md, history.json, test_eval_{baseline,final}/results.json
- 分析脚本: `eva/b1_flash_lcb_analyze.py` (已跑, 输出与上表一致)

## 下一步含义
1. B1 证明 "强模型 + skill 优化 = 正收益". 但 hard 绝对值仍低 (0.289) → 复杂任务上 skill 杠杆未充分释放.
2. 用户已要求转向复杂多文件数据集 (DevEval/repo 级) + 沙箱执行反馈. B1 的正向信号支持继续投入复杂任务 (skill 在难题才显杠杆).
3. S₀ vs SkillOpt baseline 口径差需查清, 否则跨 run 对比会误读.

Related: [[b1-flash-skillopt-lcb-launch-20261106]] [[flash-lcb-s0-result-20261106]]
[[lcb-new-optimize-20260721]] [[code-skill-negative-rootcause-20260721]]
