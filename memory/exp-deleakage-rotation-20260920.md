---
name: exp-deleakage-rotation-20260920
description: Exp1 de-leakage (ON minus Implication Direction Check) + Exp2 五位置轮换 verification — review-response 诊断实验
metadata:
  type: project
---

# Exp 1 (de-leakage) + Exp 2 (五位置轮换) — 2026-09-20

响应外部 review（RF Step 0-4 / CF F01），跑两项廉价诊断（合计 ~5h API），设计见 [[response-to-review-20260917]] §7/§8。

## Exp 1: de-leakage retest（ON minus `## Implication Direction Check`）

- **产物**: `best_skill_on_noimpl.md` = ON skill 删 L16-21（1262 chars，hash `3a0286c1`，vs on 2609/`fa63335e`）。
- **协议**: `eval_test_multiseed_shuffle.py --skills s0,on_noimpl`，4-seed shuffle，k=3。s0/on 命中 Phase A cache（同 hash），仅 on_noimpl 新算（1272 调用 ~50min）。
- **假设**: §2.3 核查称 val 题 `202602:12` 定理（`H(n,m)^{ab} ≅ ℤ²`）在 test 0 次出现 → 该段对 test 贡献 = 0，删后 on_noimpl mean ≈ on (0.4458, sd 0.006)，±2sd=±0.012。
- **seed1 早期信号（⚠ 与假设相反）**: on_noimpl seed1 = **0.5252** vs on seed1 = 0.4371，**Δ=+0.0881**。删段后 test 分数不降反升 +8.8pp。可能机制：该段虽定理 0-occurrence，但 (a) 稀释 L4 meta 规则（2609→1262 chars，信噪比↑）或 (b) 引导模型套用不相关定理检查程序反而干扰 meta 规则执行。**须等 4-seed 完成才能定性**（on_noimpl 自身 per-seed 方差未测）。

## Exp 1 最终结果（2026-09-20 完成）

4-seed shuffle（k=3，1272 调用，~60min）：

| arm | 4-seed mean | sd | per-seed | meta_46 | other_60 |
|-----|------------|-----|----------|---------|----------|
| s0 | 0.3215 | — | [0.3176,0.3428,0.3333,0.2925] | 0.2011 | 0.4139 |
| on | 0.4458 | 0.006 | [0.4371,0.4497,0.4528,0.4434] | 0.5380 | 0.3750 |
| **on_noimpl** | **0.5299** | **0.0233** | [0.5252,0.5126,0.5692,0.5126] | **0.6920** | 0.4056 |

**判定：§2.3「该段对 test 贡献=0」假设被推翻。** Δ(on_noimpl−on)=**+0.0841（+8.4pp）**，删段后升非降。两分布完全不重叠（on max=0.4528 < on_noimpl min=0.5126，gap=0.0598）。增益主要来自 meta_46（+15.4pp，0.6920 vs 0.5380），other_60 仅 +3.1pp。

**机制**：该段对 test **直接定理贡献=0**（定理 0-occurrence，§2.3 此点成立），但有**间接负向影响 −8.4pp**——通过 (a) prompt 稀释（2609→1262 chars，L4 meta 规则 SNR↑）或 (b) 误导性定理检查程序干扰 meta 规则执行。删段后模型更易触发并执行 meta 规则 → meta_46 大幅提升。

**对原结论影响**：去泄漏风险消除（更强）——该段非但不是泄漏增益，反而是 test 的负向负担。ON skill 真实算法信号（不含该负担）高于 0.4458。不改变 §6 主结论（pool_filter 仍被 shuffle 推翻，skillopt 仍唯一超 rule）。已更新 RESPONSE §7.3/§7.4/§7.5 + §2.3 归因修正 + §4.2 表。

## Exp 2: 五位置轮换敏感性（CF F01）

- **脚本**: `eval_test_5pos_rotation.py`，`rotate_to_position(item,pos_idx)` 把 gold 强制放固定位置 p∈{A..E}，distractors 按原序填充，remap A-E（确定性，对照 shuffle 的 per-seed 随机化）。
- **零 API 自检通过**（`--skills rule_meta_guess`）:
  - gold remap 正确：5 位置 × 106 题，每位置 gold 全在该位置，46 meta gold 全对位 ✅
  - rule meta_46 = 1.000 × 5 位置 ✅（meta 按文本检测，位置无关）
  - rule other_60 range=0.057 sd=0.020 —— **这是 hash 索引偏置的确定性产物，非模型位置偏差**：rule 用 `sha256(seed:qid)%5` 选 index（qid-based，位置无关），hash 索引分布恒为 `{0:16,1:12,2:10,3:11,4:11}`；位置 p 的命中数 = gold 落在该索引的题数。pos0(A)=16/60=0.267（hash 落 index0 最多+gold 全在 A），pos2(C)=10/60=0.167。
  - **教训**: rule arm 的 other_60 variation 是结构性的，不能当位置偏差信号读；真信号看模型臂（s0/on/off/skillopt）的 drift——它们做语义预测，drift 反映真实位置敏感性。
- **成本**: 5 pos × 4 模型臂 × 106 × k=3 = 6360 调用 ~4h（rule 臂已零 API 跑完）。
- **状态**: ⏳ 待 Exp 1 完成后启动（process-local 28 RPM TokenBucket 不跨进程，并发=2×28=56 RPM 超 DeepSeek 30 RPM cap → 429）。

## 关键约束: rate limiter 作用域

`qwen_backend.py` TokenBucket = module-level singleton + `threading.Lock`，**仅同步单进程内线程，不跨进程**。故 API 实验须串行（Exp1 → s0_regen → Exp2），不能并发。详见 [[phase2-issues]] 2026-09-20 条。

## 待办

- [x] Exp1 4-seed 完成 → on_noimpl=0.5299 vs on=0.4458，Δ=+8.4pp，假设推翻（删段升非降），已更新 [[response-to-review-20260917]] §7.3/§7.4/§7.5 + §2.3 + §4.2
- [x] s0_cache 重生（`precompute_s0_chunked.py` v2，分块子进程，带 manifest）—— ✅ **完成**（2026-09-20，1219s）。6 块全部 attempt0 成功，71 题 + manifest 落盘。v1 因 worker stdout 解析崩 (rc=0 但空 stdout → json.loads 崩)，v2 改 worker 直接写 partial 文件 + parent 校验文件合法性 + 断点续跑。wedge 是瞬时 keep-alive 丢连接（gateway 本身健康，fresh probe 0.3s），retry 已验证恢复。manifest: k=3/eval_temp=0.3/shuffle_seed=None/skill_hash=68571cb2/split_name=train+val/n_items=71，mean=0.239 nonzero=31/71（k=1→3 翻倍正常）。
- [x] Exp2 rule 臂自检完成（零 API）：`analyze_rotation_5pos.py` → meta_46=1.000×5 位置 ✅（位置无关，gold 正确 remap）；other_60 rule drift range=0.10（A=0.267…E=0.183，sha256 索引结构性偏置，非模型位置偏差）。
- [x] Exp2 模型臂全跑（5×4×106×k=3 调用，~4h40m）→ ✅ **完成**（2026-09-20，PID 17313 已 exit）。20/20 模型臂 + 5 rule 臂落盘，`analyze_rotation_5pos.py` 出 drift 表 → §8.3.2 已填。
  - **核心结论：on 是唯一存在位置捷径的 arm**。on 的 A-vs-BCDE = **+0.1392**（A=0.6038 vs B-E 均值 0.4646），是 s0（−0.0251）/off（−0.0231）/skillopt（−0.0338）的 **6 倍量级、反方向**。on 的高分部分来自位置捷径（gold 全在 A 时模型偏向选 A），非纯算法信号。
  - **skillopt 位置最稳定、无捷径**：A-vs-BCDE=−0.0338（A 反而最低），sd=0.0192（4 模型臂最低）。强化 skillopt 唯一性（高分 + 位置无关）。
  - on 的捷径**同时污染 meta_46 与 other_60**（两子集 A 位均显著高于 B-E），说明非仅 meta 文本检测所致，含真实位置偏置。
  - resume 机制验证通过：s0 pos0 在 819s 落盘 pass=0.3302，与 Phase A s0=0.3215 一致。
  - rule 臂 other_60 drift 是 sha256 索引结构性偏置（pos0=0.267…pos2=0.167），非模型位置偏差，已排除。
- [ ] 三条核心实验（§10.1/10.2/10.3）设计完成，pending API budget，不跑
