---
name: step3-facet-subset-design-20260929
description: 第3步 facet 子集选择实验设计 — LOO+空集+full 7臂 base5 复用P4缓存, split-half 按题选择检验 (reviewer §12.2) 2026-09-29
metadata:
  type: project
---

2026-09-29：第3步 facet 子集选择（reviewer §12.2 / §13.10 优先级表第3步）设计与启动。承接 [[step2-2x2-negtransfer-verdict-20260928]]（第2步闭环：ON 全部演化内容在 other_60 上=捷径+冗余，双删≈双留 p=0.240 不显著）。本步检验 §12.2 两条基线：(a) 固定最佳子集是否优于 full（删有害 facet 是否有帮助）；(b) 按题子集选择是否超过最佳固定子集（continue-signal）。

## 设计

### ON skill 的 facet 结构（5 facet，2^5=32 子集）

ON = `# Live Mathematical MCQ Heuristics` 标题 + 5 个 `## ` 段（已用 byte-range 解析器验证 byte-exact 重建 full==ON）：

| idx | facet header | chars | 含捷径/泄漏? |
|---|---|---|---|
| F1 | `## Option Comparison` | 563 | 含 meta-trigger 粗体 bullet（捷径轴，[[step2-nometatrigger-verdict-20260928]] 坐实）|
| F2 | `## Theorem-Level Precision` | 309 | — |
| F3 | `## Hypotheses` | 280 | — |
| F4 | `## Final Answer` | 75 | 纯格式指令 |
| F5 | `## Implication Direction Check` | 1347 | 含 `H(n,m)^{ab}≅ℤ²` val 泄漏（[[step2-2x2-negtransfer-verdict-20260928]] 证无 solving 信号 p=0.706）|

### 注入协议（复用 P4）
- skill markdown 经 `rollout_system.md` 的 `{skill_section}` 占位注入：`## Skill\n{content}\n\n`（rollout.py:14 `_build_system`）。
- 子集 skill 内容 = 标题 + 选中 facet 拼接（byte-range 切片，保证 LOO-F5 == 已 cached `on_noimpl`）。
- 评测：P4 `base5` 干净条件（不注入第6选项，纯解题力），other_60 × 4 seed × k=3，n=720/cell。

### 臂集（budget 7 臂，非穷举 32 子集；reviewer §12.2「限制模块数、开发预算」）

5 个 LOO（留一）+ 空 + full：

| 臂 | facet集 | 与已缓存关系 | 需新跑? |
|---|---|---|---|
| FULL | {F1,F2,F3,F4,F5} | = ON | ❌ cached `on` base5×4seed |
| LOO-F5 | {F1,F2,F3,F4} | = on_noimpl | ❌ cached `on_noimpl` base5×4seed |
| LOO-F1 | {F2,F3,F4,F5} | 新 | ✅ 4 cell |
| LOO-F2 | {F1,F3,F4,F5} | 新 | ✅ 4 cell |
| LOO-F3 | {F1,F2,F4,F5} | 新 | ✅ 4 cell |
| LOO-F4 | {F1,F2,F3,F5} | 新 | ✅ 4 cell |
| EMPTY | {}（真无 skill, 空串注入） | **≠ s0**（见修正） | ✅ 4 cell |

**重要修正 (2026-09-29)**：`initial.md`(s0, 1018c) **非空 skill** ——它是 `{F1-去mt, F2, F3, F4}`（ON 删 F5 段 + 删 F1 meta-trigger bullet 后的预演化初始版），与 EMPTY（空串）语义不同（`_build_system("")` 返回空 skill section, vs s0 注入 1018c skill）。故 EMPTY 臂 **须新跑**，不能复用 s0 cache。已验证 `_build_system("")==_build_system(s0)` 为 False。

新跑 = (4 LOO + 1 EMPTY) × 4 seed = 20 cell × 180 calls = 3600 calls ≈ 2.5h（28 workers）。FULL/LOO-F5 复用 P4 cache = 0 calls。

**额外参考点（非 LOO 臂, 作 §12.2 全景对照, 0 新 calls）**：
- `s0`（{F1-nomt,F2,F3,F4}, 预演化初始, cached）
- `on_noMetaTrigger`（{F1-nomt,F2,F3,F4,F5}, 删 F1 mt bullet 但留 F5, cached）
- `on_clean`（{F2,F3,F4}, 双删 mt+F5, cached）

### 两基线 + continue-signal（reviewer §12.7 第3步信号定义）
1. **(a) 固定最佳子集**：max over 7 臂 of other_60 eval-半均分。问：删有害 facet 是否 > full？
2. **(b) 按题子集选择**：split-half（SEL=[1,2] 选最优臂 / EVAL=[3,4] 评测），复用 P6 的 `routed_gain(SEL,EVAL)` 脚手架，bootstrap B=2000 + permutation null 2000×。
3. **continue-signal**：按题选择 eval-半 gain 显著 > 0 且 > 最佳固定子集。否则 = 无子集选择信号（与 P6 routing 饱和一致）。

### §13.8 caveat（split-half，P6 已做）
P6 已用 SEL=[1,2]/EVAL=[3,4] split-half，避免「同半选同半评」过拟合。本步复用同一切分，7 臂替换 P6 的 4 skill。

## 预测（零API prior，待 API 坐实）

承 [[step2-2x2-negtransfer-verdict-20260928]] 结论（双删≈双留，演化内容=冗余）：
- **预测 P3-A（无子集信号）**：固定最佳子集 ≈ full（删任何 facet 无显著 gain），按题选择 gain CI 含 0。→ 坐实单全局 skill 在 other_60 饱和，§13.10 第3步关闭，进第4步（ON 因子消融细化）。
- **预测 P3-B（有有害 facet）**：某 LOO 显著 > full，固定子集优于 full。→ 说明存在可删有害 facet，需进一步定位（可能 F5 val 泄漏对 base5 也有轻 hurt，但 step2 on_noimpl vs ON p=0.706 已不显著，故 B 可能性低）。
- **预测 P3-C（按题选择有信号）**：continue-signal 成立。→ 最有价值的正结果，需扩大子集空间（pairwise）。

## 工程
- driver `for_review/code/p7_facet_subset.py`（复用 P4 的 `run_one_logged`/`shuffle_item`/`eval_condition`，importlib 加载 for_review 副本避同名覆盖）。
- 缓存目录同 P4：`phaseA_shuffle_20260917/p4_shortcut/p7_{arm}_base5_seed{seed}_k3.json`（新臂独立前缀 `p7_`，不污染 P4 cache）。
- launcher `launch_p7_facet.py` 从 `train_lcb_skillopt.sh` source 真 key，两段式 smoke→full。
- 分析 `p7_facet_stats.py`：7 臂 other_60 表 + split-half routed_gain + bootstrap/permutation。

关联：[[step2-2x2-negtransfer-verdict-20260928]]、[[step2-nometatrigger-verdict-20260928]]、[[p6-routing-saturation-20260925]]、[[response-to-review-0924-directions-20260928]]。
