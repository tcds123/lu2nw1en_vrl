---
name: response-to-review-20260917
description: for_review 回应文档落盘 — 逐条回应 RF/CF 两份 review，挂 Phase A shuffle 4-seed verdict，完成度状态分清，叙事重写方向已定，启动 Phase B
metadata:
  type: project
---

# Review 回应文档落盘（2026-09-17 晚）

## 交付物

`for_review/RESPONSE_TO_REVIEW_2026-09-17.md` — 逐条回应两份外部 review：
- RF = `REVIEW_FINDINGS_2026-09-17.md`（P0/P1 + §1-§5 + 6 步建议）
- CF = `CODE_DOCUMENT_REVIEW_2026-09-17.md`（F01-F10 + 阶段 A-D 序）

## 核心判断

两份 review 互相印证（P0↔F01 同根因 load_split:177；P1↔F02/F03 同 filter 无作用路径机制；§2.2↔F10 同统计单位错；§1.3↔F06 同 val 内容入 skill 证据），且核心结论已被 [[phaseA-shuffle-verdict-20260917]] 独立坐实。

## 逐条判定

| Review 项 | 判定 | 要点 |
|---|---|---|
| P0/F01 答案位置捷径 | 接受 | Phase A shuffle 坐实（s0+4pp/on-14.1pp） |
| P1/F02/F03 pool_filter 归因 | 接受 | shuffle 把 +15.6pp 压到 +1.8pp（统计不可分） |
| F06/§1.3 val 内容入 skill | 部分接受 | 事实成立，但 L16-20 定理在 test 出现 0 次→非 test 泄漏，对 test 贡献=0；L4 meta 规则 shuffle-invariant 合法→不删 ON skill 任何内容 |
| §2.2/F10 统计单位 | 接受 | 4-seed sd≤0.024 只刻画评测噪声，Phase C 量运行间方差 |
| F04 API 失败 | 接受 | Phase B 统一 |
| F07 缓存身份 | 接受 | Phase B 加 manifest |
| F05 子集救援 | 接受 | 潜在缺陷非当次原因，Phase D 修 |
| F08 算子开关/温度/resume | 接受 | Phase D 逐项修 |
| F09 打包缺模块 | 接受 | Phase B 补 |
| F10/§4 文档不符 | 全接受 | S₀=1017 chars、涌现段不成立、流程图改 B 自我改写、CI 改双侧、去 GRPO/Training-Free |
| 演化方向整体否定 | 不接受 | CF 自身未做此断言（"不足以据此否定整个 facet-tree 方向"） |

## 关键补充（用户修正，review 未完全区分）

meta 规则是 shuffle-invariant 真语义信号，非位置伪影非泄漏——on/skillopt shuffle 后 meta_46 仍 >0.5。合法算法产物。但 on/off run 学得不如平凡规则（0.538/0.447 < rule 1.0），被零 API 规则占优。学得更好的是 skillopt（0.788）。

## 完成度

- ✅ 已完成：P0/P1 零 API 复现、Phase A shuffle 4-seed、+15.6pp 归因推翻、L16-20 非 test 泄漏核查、meta shuffle-invariant 验证、旧 verdict caveat 框
- 🔄 Phase B 待启：load_split 注入打乱、防泄漏检查、缓存 manifest、CI 双侧 95%、API 失败统一、主指标改 other_60
- ⏳ Phase C：量运行间方差、多 run 检验、产物规则看 test 前写死、打开 probe_split
- ⏳ Phase D：pool_filter 与父本配对、subset vs best、分类器词边界、算子计数、无效参数报错、补模块

## 文档更新到最新版（2026-09-23）

`for_review/RESPONSE_TO_REVIEW_2026-09-17.md` 已从 457 行扩到 **576 行**，新增 **§11（四小节）**，把一周来的发现全部并入回应文档：

- **§11.1 演化选择侧第三证**：`shuffle_seed` None→0 把 ON−OFF 符号从 +0.197 翻转为 −0.094（ON +0.2864 vs OFF +0.3803），两臂均无 anchor-A，R4 治本机制闭环；并记录 `--resume best_val_delta` 降级 bug 修复（运行时实证修复生效，否则 best 退化结论翻转）。
- **§11.2 干净 test=106 评测**：OFF 54.1% / ON 52.2% / s0 33.7% / rule_meta_guess 52.2%；OFF−ON paired +0.0189（t=2.882, n=3 边际）；**提升 100% 来自 meta_46（+0.493），other_60 实际 −0.017**；60.2% lucky-end 不可复现表。
- **§11.3 判据 = shuffle 不变性**：四臂 A-vs-BCDE 表（evolver +0.1392 位置捷径 / skillopt −0.0338 最稳）；三条被推翻的 claim；content-binding（合法）vs label-binding（数据集伪影）。
- **§11.4 三证闭环表 + 最终判定**：pool_filter +15.6pp 归因「接受（三证闭环）」——评测侧（Phase A）+ 运行间方差（§10.1）+ 演化选择侧（§11.1）三证一致，pool_filter 在干净因果条件下不再是正向效应。

§0 表 pool_filter 行改为「接受（三证闭环）」，§4 完成度表加 4 行已完成项，§5 叙事加 2026-09-23 精化。

## 关联

[[phaseA-shuffle-verdict-20260917]]（核心证据）/ [[gpt-review-p0-dataset-shortcut-20260917]]（P0 发现）/ [[pool-filter-on-off-verdict-20260917]]（被推翻的旧结论，已加 caveat）/ [[best-algorithm-open-questions-20260917]]（Q1 负迁移）/ [[extension-validation-candidates-20260917]]（C-B 对象改 skillopt）/ [[onoff-shuffled-final-verdict-20260923]]（§11.1 数据源）/ [[onoff-shuffled-test-eval-20260923]]（§11.2 数据源）/ [[criterion-shuffle-invariance-not-natural-evolution-20260922]]（§11.3 数据源）
