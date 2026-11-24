---
name: step3-facet-subset-verdict-20260929
description: 第3步 facet 子集选择判决 — 7臂LOO+空+full, FULL最佳, 按题选择routed gain显著为负(−0.033), continue-signal不成立, 与P6饱和一致 (reviewer §12.2) 2026-09-29
metadata:
  type: project
---

2026-09-29：第3步 facet 子集选择实验闭环（reviewer §12.2 / §13.10 优先级表第3步）。承接设计 [[step3-facet-subset-design-20260929]]，结论 = **预测 P3-A 成立（无子集选择信号）**，与 [[p6-routing-saturation-20260925]] / [[step2-2x2-negtransfer-verdict-20260928]] 的"干净协议下单全局 skill 在 other_60 已饱和"一致。§13.10 第3步关闭，进第4步。

## 实验协议

- 7 臂 budget 枚举（非穷举 32 子集，reviewer §12.2「限制模块数、开发预算」）：FULL{F1..F5}/LOO_F1..F5/EMPTY{}。
- base5 干净协议（不注入第6选项，纯解题力），other_60 × 4 seed × k=3，n=720/cell。
- skill 注入 byte-range facet 切片重建（验证 LOO_F5 byte-exact==on_noimpl，FULL==on）。
- FULL/LOO_F5 复用 P4 cache（0 calls）；新跑 20 cell（LOO_F1..F4 + EMPTY × 4 seed），~480s/cell。

## 结果

### [A] 各臂 other_60 base5 解题力

| 臂 | facet集 | other_60 | meta_46 |
|---|---|---|---|
| FULL | {F1,F2,F3,F4,F5} | **0.4097** | 0.0000 |
| LOO_F5 | {F1,F2,F3,F4} | 0.4042 | 0.0000 |
| LOO_F4 | {F1,F2,F3,F5} | 0.3903 | 0.0000 |
| EMPTY | {} | 0.3931 | 0.0000 |
| LOO_F1 | {F2,F3,F4,F5} | 0.3806 | 0.0000 |
| LOO_F3 | {F1,F2,F4,F5} | 0.3806 | 0.0000 |
| LOO_F2 | {F1,F3,F4,F5} | 0.3736 | 0.0000 |

排序：FULL > LOO_F5 > EMPTY ≈ LOO_F4 > LOO_F1=LOO_F3 > LOO_F2。删任何 facet 都不升 → **无有害 facet，FULL 已最佳**。

**意外发现**：EMPTY(0.3931) 反而高于 LOO_F1(0.3806)/LOO_F2(0.3736)/LOO_F3(0.3806) —— 部分子集比"无 skill"还差，说明 5 个 facet 非加性交互（删一个破坏整体 coherent 性），但全集 coherent 有益（FULL 比 EMPTY +1.66pp）。

### [B] 固定最佳子集（§12.2 基线 a）

best_fixed = FULL，delta = +0.0000 → **删有害 facet 无帮助，full 已最佳**。✅ 预测 P3-A。

### [C] Split-half 按题子集选择（§12.2 基线 b / §12.7 continue-signal）

SEL=[1,2] 选最优臂 / EVAL=[3,4] 评测（P6 split-half，避免同半选同半评过拟合）：

- best fixed arm on eval half: FULL acc=0.4139
- routed（sel-half best）eval-half acc: 0.3806
- oracle（eval-half best, 不可部署）: 0.5750
- **routed gain over best-fixed: −0.0333（显著为负）**
- oracle gain: +0.1611
- sel-best==eval-best agreement: 0.267（7臂随机=0.143，略高但很弱）

→ SEL 半选的臂在 EVAL 半不泛化，按题选择不仅无 gain 还微负。

### [D] 统计检验（routed gain，other_60，eval half）

- observed routed gain: **−0.0333**
- bootstrap 95% CI（item resample, B=2000）: **[−0.0972, −0.0111]**，完全在 0 以下 → routed gain **显著 < 0**（非仅"含 0"）。
- permutation null（random arm）mean=−0.0140，95%=[−0.0472, +0.0222]
- observed −0.0333 vs null 97.5th=+0.0222 → WITHIN null（即"按题选择"不比"随机选臂"更能命中 EVAL 最优臂）。

## 判读

**continue-signal（按题选择显著>最佳固定子集）: ❌ 不成立**。

两条基线都阴性：
- (a) 固定最佳子集 = FULL，删任何 facet 无 gain（无有害 facet）。
- (b) 按题子集选择 routed gain = −0.0333，CI 完全 < 0，不比最佳固定子集（FULL）好，甚至显著更差；且不比随机选臂更好（permutation 内）。

→ 与 P6 routing 饱和一致：**单全局 skill 在 other_60 已饱和，无子集选择 headroom**。坐实"演化内容对干净子集无真解题信号"（[[step2-2x2-negtransfer-verdict-20260928]] p=0.240）的另一面——不仅"删冗余内容无影响"，连"按题选子集"也无信号，全集 coherent 是最优固定配置。

## 关键教训（分析脚本 false-positive bug）

**⚠️ 首次跑 `p7_facet_stats.py` 得到 continue-signal ✅ 成立是 BUG**，源于 5 处 `bf_acc = mean([PR[bf][s][k] ... if k in PR[bf]])` 漏了 `[s]`：
- `PR[bf]` 是 `{seed: {qid:rate}}`，`k in PR[bf]` 检查 qid 是否是 seed 键（1/2/3/4）→ 恒 False → 过滤掉一切 → mean=0。
- `best_fixed` 选得对（max 用带 `[s]` 的过滤），但重算 acc 时全错成 0 → routed_gain = routed − 0 = +0.38 假阳性。
- 修正后 routed_gain = 0.3806 − 0.4139 = **−0.0333**（真阴性）。

**教训**：split-half routed_gain 脚手架复用 P6 时，`PR[bf]`（无 `[s]`）与 `PR[a][s]`（有 `[s]`）一行之差，max-key 过滤对、acc 重算过滤错 → 选对臂但算错分。这类 typo 会让"选最优"类指标假阳性（best=0 时任何 routed 都 >0）。**复用统计脚手架必须逐行核对过滤条件，尤其 max 选臂与 acc 重算的两处过滤要一致**。落盘 `for_review/code/p7_facet_stats.py`（5 处 typo 已修）。

## 落盘

- driver `for_review/code/p7_facet_subset.py` + launcher `launch_p7_facet.py` + 分析 `p7_facet_stats.py`（已修）。
- cache: `phaseA_shuffle_20260917/p4_shortcut/p7_{arm}_base5_seed{seed}_k3.json`（20 新 cell）+ 复用 P4 `p4_on_*`/`p4_on_noimpl_*`（8 cell）。
- verdict: `phaseA_shuffle_20260917/p4_shortcut/p7_facet_verdict.json`。
- RESPONSE_TO_REVIEW_2026-09-24.md §12.x 追加本步结论。

关联：[[step3-facet-subset-design-20260929]]、[[step2-2x2-negtransfer-verdict-20260928]]、[[step2-nometatrigger-verdict-20260928]]、[[p6-routing-saturation-20260925]]、[[response-to-review-0924-directions-20260928]]。
