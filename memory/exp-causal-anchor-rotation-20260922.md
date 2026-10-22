---
name: exp-causal-anchor-rotation-20260922
description: §8.1 因果验证完成 — 改 ON skill L20 "option A"→"option C"(单变量)重跑五位置，偏置峰从 A(+0.1392)精确迁到 C(+0.1344)，命中判读规则①，L20 anchor→位置偏置因果坐实
metadata:
  type: project
---

# §8.1 位置偏置因果验证实验（2026-09-22 启动）

## 动机

§8 五位置轮换发现 ON 臂 A-vs-BCDE=**+0.1392**(A=0.6038 独高),是 s0/off/skillopt(均 −0.02~−0.03)的 6 倍量级、反方向。§2.3 核查定位嫌疑源为 ON skill L20 末句锚句 "only option A survives this two-way test"(诊断范例 H(n,m) 结论,非通则)。

但该归因是**相关性证据**(ON 有 anchor 有偏置 / OFF 无 anchor 无偏置),非因果。用户质疑"软性推断是瞎猜/模型幻觉"。本实验一锤定音:单变量改 L20 的 "option A"→"option C",若偏置迁移到 C 位则因果坐实。

## 实验设计

**单变量对照**:
- 原 on = anchor-A(L20 "only option A survives"),hash fa63335e(已有五位置 cache,在 `phaseA_shuffle_20260917/`)
- on_C = anchor-C(L20 "only option C survives"),hash 57167151(新跑)

两份 skill **除 L20 一个词外逐字节一致**(diff 确认只 1 行差异,`candidate condition C` 变量名未动,0 处残留 option A)。

**脚本**: `rl/train/verl1/skill_evolution/eval_test_5pos_rotation.py`(复用,无改动)
- `rotate_to_position(item,pos_idx)` 把 gold 强制放固定位置 p∈{A..E}
- 5 pos × 106 题 × k=3 = 1590 调用(只 on_C 新算)

**调用**: `--skills s0,on --on <on_C路径> --out_dir phaseA_shuffle_causal_20260922 --eval_k 3`
- on_C 文件: `phaseA_shuffle_causal_20260922/best_skill_on_C.md`
- 独立 out_dir `phaseA_shuffle_causal_20260922/`(不污染原 §8 结果)
- s0 五位置 cache 已拷入新目录 → s0 跑 0 API,全部 CACHED 命中

## 判读规则（决定 L20 归因对错）

对比 on_C 与原 on 的**每位置 overall accuracy**:

| on_C 结果 | 含义 | 判定 |
|---|---|---|
| **pos2(C) 独高**,A 位回落到 B-E 水平 | 偏置从 A 迁移到 C | ✅ **L20 anchor 是因果源**,坐实"软性 anchor→位置先验",我对 |
| 五位置趋平(像 s0/off,无独高位) | anchor 强度不够迁移,偏置消失 | 🟡 anchor 仍可能因果但作用是"造 A 偏好",改 C 后偏好没了 |
| **pos0(A) 仍独高**(像原 on +0.1392) | L20 不是源,偏置另有出处 | ❌ **我上轮归因错,撤回** L20→偏置 的因果链 |

## 预期成本

1590 调用,~1h10m(原 §8 跑 5×4×106×3=6360 调用 4h40m,本次 1/4 量)。后台 PID 4273,日志 `causal_run.log`。

## 状态

- ✅ **完成**(2026-09-22 启动,当日完成,exit 0)
- 冒烟验证通过(`--limit 2`:on_C 正确调 API 落盘,hash 57167151,s0 cache 命中)

## 结果与判读(命中规则① → 因果坐实)

on_C 五位置 overall(k=3, test=106, 单 seed 但位置内 3×106 取均):

| 位置 | 原 on (anchor-A) | on_C (anchor-C) |
|---|---|---|
| A | **0.6038 独高** | 0.4874 |
| B | 0.4780 | 0.5157 |
| **C** | 0.4528 | **0.6101 独高** |
| D | 0.4560 | 0.4182 |
| E | 0.4717 | 0.4811 |
| A-vs-BCDE | **+0.1392** | **−0.0189** |
| C-vs-其他 | −0.0496 | **+0.1344** |

**判读**:单变量改 L20 一个词("option A"→"option C"),其余 20 行逐字节不变 → 偏置峰从 A 精确迁到 C,量级几乎一致(+0.1392→+0.1344,差 0.005 在噪声内)。

**命中预设规则①**:"on_C pos2(C) 独高、A 回落到 B-E 水平 → **L20 anchor 是因果源**,坐实软性 anchor→位置先验"。

→ **相关升级为因果**:§8 的"+0.1392 偏置来自 L20 anchor"归因被单变量实验验证,**非瞎猜非幻觉**。

## 对用户两层质疑的校准回应

1. 用户上轮质疑"软性推断是瞎猜/模型幻觉,review 完全错了"——**部分对**:质疑时(§8 阶段)确是相关证据,但本实验补上了相关→因果的跨越,review 指出的位置偏置及源定位**被验证为真**。
2. 用户"算法根本没问题"——**不成立**:L20 anchor 是 ON skill 的一条**缺陷**(诱导位置捷径),非能力证明。恰恰相反,ON 比 OFF 高的 15.6pp 里含这一整块 +0.1392 位置偏置成分,是污染不是增益。

## 局限

- 单 seed(k=3 但位置内取均,非跨 seed)。+0.1392→+0.1344 的迁移量级在原 §8 sd 范围内,但跨 seed 复测会更稳。若要反驳"单 seed 运气",可补 seed。
- 证明"偏置源=L20 anchor",**未**证明(也不打算证明)"ON 算法做数学有效"——后者由 [[phaseA-shuffle-verdict-20260917]] 的 other_60 三臂全退步已否定。

## 关联

[[exp-deleakage-rotation-20260920]](§7 删段消融 + §8 五位置轮换,原 on +0.1392 出处) / [[criterion-shuffle-invariance-not-natural-evolution-20260922]](判据澄清) / [[gpt-review-p0-dataset-shortcut-20260917]](P0 捷径背景)
