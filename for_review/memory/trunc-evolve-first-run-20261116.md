# 首版防截断 skill 自主进化 — run1 结果 (2026-11-16)

## 目的
用户: "先出第一版我们算法的带上防截断 skill, 在 AIME 数据集上有个初步的尝试看结果"。
要回答: **进化算法能否在起点 skill 已有防截断诉求时, 继续把长度/循环压下去而不掉准确率。**

## 设计
- 起点 `for_review/data/skills_v2/trunc_v1.md` (2273 字符) = v2 S₀ + 自拟 "Budget & Commit Discipline" 6 条
  (搜索预算 / sweep 上界 / 验而不重推 / 反循环 / 60% 硬交卷 / 单一 `<answer>`)。
  **刻意不复用 codex 蒸馏 demo5**: 用户此前异议点即"对照应是自进化产物, 非 gpt 蒸馏"。
- 启动 `for_review/code/scripts/run_evolve_trunc_v1.sh run1`; 产物
  `for_review/outputs/skill_evolver_trunc_20261116/run1/run_20261116_113802/`, 日志
  `for_review/logs/evolve_trunc_run1_20261116_113801.log`
- B=deepseek-v4-flash, cap=16384, retry_on_trunc=0, temp=0.3, iters=4, K=3, eval_k=3, n_parallel=6,
  AIME24 train18/val12 (select=30), 无 EFL/facet_tree/expand。
- 晋升口径不变: 单目标 (`delta>best_val_delta and ci>0 and hash!=cur`, LOST guard `pass < base−0.06`);
  **长度/截断不进晋升门** (issue #69) —— 本轮是"起点注入预算纪律 + 单目标进化"的探底, 不是反馈因子方案验证。

## 结果: 4 iter / 12 候选, 只过门 1 个

| Iter | focus | 候选 (delta / CI / len) | 门 |
|---|---|---|---|
| 1 | 误读所求量 | +0.078/+0.022/4137 · +0.044/−0.044/3756 · +0.044/−0.044/3278 | KEEP PREV |
| 2 | 边界 off-by-one | +0.078/0.000/3735 · +0.056/−0.044/3680 · +0.078/−0.011/4121 | KEEP PREV |
| 3 | 组合计数完整性 | +0.067/−0.022/3503 · +0.044/−0.044/3187 · +0.033/−0.078/3244 | KEEP PREV |
| 4 | 容斥/重复计数 | +0.033/−0.067/3258 · +0.022/−0.067/3837 · **+0.100/+0.033/3907** | **ACCEPT** |

- ACCEPT: `8282f99052 → d1a59af67a`, val pass 0.900 → 0.922, delta +0.100, CI lower +0.033。
- 最终 `best_skill_evolved.md` 3907 字符 (+72% vs 起点 2273); 新增段 = "Counting by Inclusion-Exclusion
  (double-count / symmetry test)", 其余 facet 保持一行式。
- 12 个候选**全部**长于起点 (3187~4137): 演化方向系统性偏"加内容", 单目标门对此完全无感 —— issue #69 实锤。

## 实测: 每个 skill 90 条 (30 题 × k=3), 均来自 trace; tok = `usage.completion_tokens`, 缺失时按 (reasoning+response)/4 估

| skill | 侧 | 截断率 | 平均输出 tok | 干净 trace tok |
|---|---|---|---|---|
| v2 S₀ | `evolver_s0_trainval` | 11.1% (10/90) | 4508 | 3023 |
| trunc_v1 (起点) | `evolver_base` | 6.7% (6/90) | 3137 | 2191 |
| evolved d1a59af67a | `evolver_val` 末端块 | 6.7% (6/90) | 3264 | 2327 |
| iter1 cand1 2b8ff7405a | `evolver_val` | 3.3% (3/90) | 2931 | 2468 |

- **判读**: 手写预算纪律段贡献了"截断 11.1%→6.7%、均值 −30%、干净 tok −27%"几乎全部收益;
  进化的接受步反而把均值 +4%、干净 tok +6% 推回去 (3907 字符 skill 更长, output 也略长)。
  → **"先出第一版"的答案是: 第一版收益来自设计, 不来自进化; 进化在这一轮净贡献 ≈ 0 甚至微负。**
- iter1 cand1 长度最短且截断最低, 但 delta +0.078 未严格超过 baseline +0.078 (必要非充分 → 拒), 
  低截断无晋升路径的典型例子。
- 全部 n=30/k=3, CI 均跨 0; iter4 的 CI=+0.033 上界也薄。**不可作显著性主张**, 只作探索性方向。

## 产物一致性复核 (已闭环, 勿重复排查)
`best_skill_evolved.md` 与 `evolver_val.jsonl` 中实际送入 B 的 skill 正文**逐字节一致** (同为 d1a59af67a, 3907 字符)。
之前"评估文本含 Name-the-target-noun / Noun-swap 等冗长 bullet 而文件没有"的怀疑**是错的**:
`evolver_val.jsonl` 是**累积日志**, 含 12 个候选各 90 条; 我早先比对的是**第一块 base(trunc_v1, 8282f99052)**,
不是末端 winner 块。教训: 按 skill hash 分组后再比对, 不要取"最后一行"或"第一行"当代表。
`eval_aime24_multiseed.py --skills s0,<best_skill_evolved.md>` 复评测的确实是过门产物, 可放心用。

## 后续
1. 四臂 paired 复评 `none / s0 / trunc_v1 / evolved` (k=20, seeds=2, 6000 调用 ≈ 5h):
   一次拿到"起点贡献 vs 进化增量"的配对差。需用户批准 API 预算后再跑。
2. 单目标门看不见长度轴已由本轮实锤 (12/12 变长且全过不了门/最后一个靠 pass 过门) → 反馈因子双门
   (δ_acc ≥ −1pp 且 clean-pass 长度 ≤ −15%) 有必要, 待 §8 四个决策点拍板。
3. 判读一律带 A/B/C 三态截断口径 (issue #71), 不用 sample_em 单指标。
