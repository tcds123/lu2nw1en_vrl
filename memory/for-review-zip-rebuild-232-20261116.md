---
name: for-review-zip-rebuild-232-20261116
description: for_review zip 重建同步 232/retention 结果 (322 files, 零密钥)
metadata:
  type: project
---

`dist/for_review_20261113.zip` 重建：把 232 run + 通道 B retention 结果同步进复审包。

**起因**: 09:10 的旧 zip 已含更新后的 `outputs/appsplus_routing_formal/` (result.json=232 + retention_analysis + test_retention_results.json 208 条)，但 **README 仍是 231**（README/REVIEW_REQUEST/CHANGELOG 的 232 编辑在 09:24 才落盘，晚于 09:10 打包），且 **retention milestone 文件缺失**（README/REVIEW_REQUEST/CHANGELOG 都引用 `memory/tf-evolver-appsplus-routing-retention-20261113.md`，旧 zip 无此文件 → 审阅者点链接 404）。

**重建** (11-16 09:38):
- manifest = 旧 321 文件 + 新增 retention milestone = **322 文件**，6.9MB。
- 从工作树重新压缩（Python `zipfile`，本机无 `zip` 二进制）。
- 顶层结构不变 (code 139 / memory 83 / outputs 67 / docs 15 / data 12 / plans 2 / review_requests 1 / 3 根 .md)。

**验证通过**:
- result.json: routed=232 / rescue=24 / retention_analysis=88.94% (n_regress=23) / tag=🟡中风险 ✅
- README: 4× `232/300`，1× `231/300`（后者为"旧 run 已被取代"历史引用，保留正确）✅
- REVIEW_REQUEST / CHANGELOG: 232 ✅
- retention milestone 在 zip 内 ✅
- test_retention_results.json = 208 条逐题 ✅
- 密钥安全: 无 `configs/*.local.yaml`；全文 `msk-` + ≥20 字符扫描 0 命中；3 个 skillopt 配置模板只有空串占位符 (`api_key: ""`) ✅

**概念三分 (包内已对齐)**: Acquisition(rescue=24) / Preservation(retention=88.94%, 退化显著 p=2.4e-7) / Deployment(routed=232)。regress=0 是路由结构性特征（S₀-pass 题复用 cache，候选不执行），非零测量掩盖 —— 通道 B 补上了独立退化测量，见 [[tf-evolver-appsplus-routing-retention-20261113]]。

任务 #45 完成。
