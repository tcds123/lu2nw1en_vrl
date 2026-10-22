---
name: for-review-repackage-1310-closed-20260925
description: 复审 for_review_2026-09-24.zip 重打包(P2/P3/P4/P6 全纳入, API key 已 redact)+§13.10 优先序全闭环
metadata:
  type: project
---

2026-09-25：复审包重打包 + §13.10 执行优先序全闭环。[[p3-factor-ablation-launched-20260924]] / [[p4-shortcut-fix-diagnosed-20260924]] / [[p6-routing-saturation-20260925]] 之后的收尾。

## 重打包

旧包 `for_review_2026-09-24.zip`(711KB, Sep 24 17:43) 早于 P2/P3/P4/P6, 缺新代码与结果。重打为同路径新包:
- **245 files / 14.85 MB**（Python `zipfile` 写, `zip` CLI 未装）。
- code/ 新增 7 文件: `ablation_factor_noAsentence.py`(P3 driver) / `launch_p3_noAsentence.py` / `p4_shortcut_diagnosis.py`(P4 driver) / `launch_p4_shortcut.py` / `p4_paired_stats.py`(纯 Python paired 统计) / `p6_complementarity.py`(互补矩阵+split-half) / `eval_test_multiseed_shuffle.py`(shuffle_item/meta_qids 源, 328 行, 区别于旧 `eval_test_multiseed.py` 186 行无 shuffle)。
- results/phaseA_shuffle_20260917/: 48 P4 per-cell cache(`p4_{skill}_{cond}_seed{1-4}_k3.json`, 含 response/predicted_label/parse_status/finish_reason/samples) + 28 P3/P6 shuf cache(`per_{skill}_{hash}_seed{1-4}_shuf.json`) + `factor_ablation_summary.json`(P3) + `p4_shortcut_summary.json`(P4) + `p4_paired_stats_report.txt` + `p6_routing_report.txt` + `README.md`(数据/协议/结论说明)。
- 排除: `__pycache__`/`*.pyc`/旧 `*.zip`。

## API key 安全

launcher 两文件硬编码了 `QWEN_CHAT_API_KEY=msk-bf2324d5...`。**重打前 redact 两 launcher 为 `<REDACTED_QWEN_CHAT_API_KEY>`**（drivers 本就从 env 取 key, 未硬编码）。zip 全文 grep `msk-bf2324d5` 零命中。遵守 CLAUDE.md "API key 不要提交到公开仓库"。

## §13.10 全序闭环

复审 EXPERIMENT_REVIEW §13.10 新执行优先序(从小到大)全部 ✅:
1. ✅ 文档/口径修正
2. ✅ P2 补响应/finish_reason 数据 + `run_one_logged`/`classify_parse_status`/`eval_on_logged`(code 侧)
3. ✅ P3 因子消融(A句~40% / val内容~60% 均 hurt; on_noAs 仍 > s0 +0.156)
4. ✅ 收益抵消矩阵(干净 shuffle 互补, P6 [A]/[C]/[D])
5. ✅ P4 捷径修复诊断(4seed, meta_false bite 0.338 vs s0 0.043, z=14.23; on_noAs vs on t=0.78 不显著→删 A 句未降捷径)
6. ✅ P6 路由/饱和(routed +0.0083 CI[−0.047,+0.042]含0, permutation 内; per-item Δ r≤0.21; 无可学信号)

## 对复审元判断的最终回应

复审 §13.10 核心元判断: "饱和结论过强——混淆了(1)数据有捷径 (2)当前搜索停滞 (3)排除改进可能(未坐实)"。本批实验分支坐实:
- **(1)数据有捷径 = 成立**: P4 meta_false 文本触发 bite 0.338(meta_46"超越"=捷径非能力)。
- **(2)搜索停滞/单全局 skill 饱和 = 在 other_60 成立**: P6 四 skill 持平 0.375~0.414, 路由增益≈0(CI 含0), 互补=采样噪声。
- **(3)排除改进可能 = 仍过强, 未坐实**: 饱和仅限"当前 skill 集在 LiveMath(other_60)"; 堵捷径后 meta_46 重测、换表示层、换目标(AIME)仍开放。**不该泛化为"无改进可能"**。

## 下一步方向(非路由)

- 堵捷径后重测 meta_46(看真实能力是否仍 > s0)。
- 换表示层 / 换目标(AIME)测泛化边界。
- **不**建 per-topic 路由库(P6 证无信号)。
