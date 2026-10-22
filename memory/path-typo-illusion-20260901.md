---
name: path-typo-illusion-20260901
description: "livemathmaticianbench 拼写幻觉" 根因定位 — 代码里从无 typo, 四常量 (S0/V2/SPLIT/S0_CACHE_DEFAULT) 全拼写正确且盘上存在; 幻觉只出现在转写/复述链路 (摘要→grep→手打 ls), 反复"感觉出现"因目录名 livemathematicianbench 反直觉 (math·matic 毗邻触发预读纠错) + 跨 8+ 文件高频出现. 根治法: 校验路径只走 python import 查盘, 不手打字符串.
metadata:
  type: feedback
---

# 路径拼写幻觉根因 (2026-09-01)

## 症状
多次会话中"感觉"`livemathmaticianbench`(少 e) 这个 typo 反复出现, 怀疑代码里 path 常量拼错.

## 根因 (零转写校验坐实)
用 `python -c "import skill_evolver_math as m; import os; print(os.path.exists(m.S0_DEFAULT))"` 查: 四常量 (S0_DEFAULT/V2_SKILL_DEFAULT/SPLIT_DEFAULT/S0_CACHE_DEFAULT) **全拼写正确** (`livemathematicianbench`, 带 e), 全部盘上 exists=True. grep "livemathmaticianbench" (少 e) 退出码 1 (不存在). **代码里从无 typo.**

幻觉只出现在**转写/复述链路**:
- 摘要 (二手转写) 传"有 typo" → 下一会话信以为真 → grep/Edit/ls 时**手打**长字符串 → 自己注入新 typo → `ls livemathmaticianbench` 失败 → 误判"代码有 typo".
- 我在 9/01 这次会话先信摘要去 Edit, 失败 (字符串不存在) 才证伪.

## 为什么"感觉反复出现"
1. **目录名反直觉**: `livemathematicianbench` = live+mathematician+bench 全小写连写, `math` 与 `matic` 毗邻 → 人/模型预读到 `livemath` 断词 → 自动纠错补成 `livemathmatician`(少 e). 预读错误类 "the the" 读成一个.
2. **高频出现放大**: 该目录名在 8+ 文件 + 所有 run 的 args 记录 + 汇报文档里反复出现 → 每次会话提到都可能被转写成 typo → 下次看到 typo 版就觉得"又出现".

## 根治机制 (不靠记性)
**Why:** 长相似目录名 + 转写链路放大同一幻觉, 靠"注意拼写"治标不治本.
**How to apply:**
- 校验路径常量: 永远 `python -c "import <module> as m; import os; print(os.path.exists(m.<CONST>))"`, **不手打路径字符串**.
- 报告/文档里引用路径只用符号名 `S0_DEFAULT`, 不展开成字符串字面量.
- (可选) 长路径常量集中进 `paths.py`, 全脚本 `from paths import *`, 断掉转写链路.

## 相关
[[expand-v2-design-20260901]] (本次校验是为 expand v2 dry-run 启动) / [[lit-vs-impl-vs-verdict-20260831]] (接口≠机制: 同样强调用构造性验证而非二手转述)
