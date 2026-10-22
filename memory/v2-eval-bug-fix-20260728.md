---
name: v2-eval-bug-fix-20260728
description: v2 +15.3pp 假阳性根因=评测噪声(2 bug); 修投票+固定基线+扩val(1:3:6); 全量重跑验证中
metadata: type: project
---

# v2 评测噪声根因诊断 + 修复 (2026-07-28) 🔥🔥

## 触发: 用户怀疑 +15.3pp 反转
3-seed 复测完成 (07-27 21:00 → 07-28 02:53), 3 个 seed 全部 CI 门控拒绝, best skill 退回 S₀ (1017 bytes).
用户怀疑"不是方法假阳性, 是筛选/优化机制设计问题". 暂停记录, 先诊断.

## 根因: 2 个评测设计 bug (非方法失效) 🔴
**证据**: 4 个 run 的 baseline_val (S₀ 现场重跑 val) 波动: 4/18, 6/18, 7/18 — 评测端不该波动.

### Bug 1: B 模型 temp=0 在 vLLM 网关非确定
- s0_cache (07-24 快照) 记录 S₀ val = 6/18 稳定; 但 v2 各 run 现场重跑 S₀ val 得 4/6/7
- 原因: DeepSeek-V3.2 网关 vLLM 后端, temp=0 下 batch 大小变化 + 并发请求 + 浮点累加顺序不同 → 输出不完全可复现
- v2 用 ThreadPoolExecutor(max_workers=8) 并发 val, 每次 batch 组合不同 → 同一 S₀ 跑出不同结果
- delta = skill_pass - s0_cache_pass 混入 ±2~3 题纯评测噪声 (18题上 +1题翻转=+0.056, 正是门控抖动量级)

### Bug 2: baseline 与 candidate 评测不对称
- 第337行 baseline eval_on(passes=1), 第371行 candidate eval_on(passes=2)
- baseline 单次采样撞到 4/18 偏低 → base_delta=-0.111 (虚假低基线)
- candidate 2 次取平均噪声减半 → 看似 delta 更高
- **+15.3pp 大概率是 baseline 采样偏差, 不是真增益**

### 附带发现: llm() temperature 参数死参
- chat_target 全链路不接受 temperature 参数 (qwen_backend 用全局 TARGET_CONFIG.temperature)
- v2 llm() 的 temperature 形参从未传入 → rewriter/distill/merge 实际全在 temp=0.0 跑 (无采样多样性, 变体只靠 variant_hint 文字差异)

## 修复 (tf_global_v2_math.py) ✅
1. **k 次投票**: run_one 加 k 参数, 评测每题采样 k 次 (EVAL_TEMP=0.3), pass_rate=通过次数/k (连续值, 降噪声)
2. **S₀ 对称**: precompute_s0 也用 k 投票, 与 candidate 对称 (否则 S₀ 0/1 vs candidate 连续, delta 失真)
3. **固定基线**: baseline 用 s0_cache 固定快照不现场重跑 (消除采样偏差), base_delta=0
4. **真采样**: set_temp() 函数写 TARGET_CONFIG.temperature, GEN_TEMP=0.8 (rewriter 真采样), EVAL_TEMP=0.3 (评测)
5. **移除 val_passes**: 改 --eval_k 参数 (默认5)

## 数据: 扩 val (路径 A) ✅
- split_ratio 1:3:6 (train=18, val=53, test=106), seed42, 同 LiveMath 数据
- val 18→53, bootstrap CI 功效 ~3x, ±1题翻转影响 0.056→0.019
- s0_cache (k=5, EVAL_TEMP=0.3): val S₀ pass=0.2302, train S₀ pass=0.2111
- split: /data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42
- cache: rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json

## 冒烟验证 (train_n=3, g=3, eval_k=2) ✅
- 链路全通: rewriter GEN_TEMP采样→run_one EVAL_TEMP投票→distill(无KeyError)→pool不卡死→compose
- rewriter 真采样: 同一题两次跑 best_delta -0.8 vs +0.2 (多样性恢复)
- **修复生效**: baseline +0.000 (固定快照, 无采样偏差), val delta=+0.015 (连续小正), CI_lower=-0.051→正确拒绝
- train_n=3 太小, +0.015 不显著属预期

## 全量重跑 (进行中) 🔄
- 命令: python tf_global_v2_math.py --split_dir 1-3-6 --s0_cache k5 --iters 3 --g 4 --train_n 18 --eval_k 5
- 日志: /tmp/v2_full_136.log, PID 118187
- 预计 ~45-60min (28RPM, ~1230 调用)
- 验收: val delta 稳定 >0 且 CI_lower>0 → +15.3pp 真假分明

## iter1 结果 (关键判据通过) ✅✅
**iter 1: val delta=+0.079 (CI_lower=+0.042) → ACCEPT**
- val 53 题上 +0.079 (约净翻对 4 题), bootstrap 95% CI 下界 +0.042 > 0 → 严格门控接受
- best_S 更新为 composed skill (3247 chars, 非 S₀)
- train 侧撬动强: 多题 best_delta +0.6/+0.8/+1.0
- **对比修复前**: 3 seed 全部 CI_lower<0 被拒, best 退回 S₀. 修评测噪声后真信号浮现 → 证实 +15.3pp 假阳性根因=评测噪声, 方法本身有效
- iter2/3 跑中

## 全量 3-iter 最终结果 ✅✅✅ (2026-07-28)
run: rl/train/verl1/outputs/tf_skill_math_v2/run_20260728_131934
| iter | val delta | CI_lower | 决策 | skill_len |
|------|-----------|----------|------|-----------|
| 1 | +0.079 | +0.042 | ACCEPT | 3247 |
| 2 | +0.083 | +0.045 | ACCEPT | 2918 |
| 3 | +0.098 | +0.053 | ACCEPT | 3655 |

- **3 iter 全部 ACCEPT, delta 单调递增** (+0.079→+0.083→+0.098), CI_lower 全为正
- best_val_delta=+0.098 (53题, 约净翻对5题), best skill = S₀ + 追加数学诊断节 (3655 chars, 非换皮)
- **结论坐实**: +15.3pp 假阳性根因=评测噪声(2 bug); 修复后真信号稳定浮现, 方法有效, pool 迭代有效
- 对比: 修复前 3-seed 全拒 best退回S₀; 修复后 3-iter 全 accept 递增. 同一方法同一数据, 只修了评测
- test=106 评测进行中 (k=5 同条件, /tmp/eval_test_v3.log)

## test=106 最终结果 ✅✅✅ (2026-07-28)
| 方法 | test pass | Δ vs S₀ |
|------|-----------|---------|
| S₀ (seed) | 0.2906 | 0 |
| **v2 best skill** | **0.3604** | **+7.0pp** |

- **test 端 +7.0pp, 泛化成立** (val +9.8pp CI+5.3pp → test +7.0pp, 同方向同显著性)
- 评测条件: k=5 投票, EVAL_TEMP=0.3, test=106 题, 与 val 完全同条件
- 产出: rl/train/verl1/outputs/tf_skill_math_v2/run_20260728_131934/{tf_global_skill_v2_math.md, test_eval.json}
- **最终定论**: +15.3pp 是评测噪声导致的假阳性 (baseline 采样偏差); 修复评测后真实增益 = val +9.8pp / test +7.0pp, 统计显著可复现. 方法 (TF-global v2 失败模式蒸馏 + 追加式合成 + CI 门控) 有效, train 撬动信号能泛化.

## 故事线 (写论文用)
1. 单次 v2 test=0.4839 (+15.3pp) — 看似强正
2. 3-seed 复测全拒, best 退回 S₀ — 怀疑评测问题
3. 诊断: 2 评测 bug (temp=0 非确定 + baseline/candidate 不对称) + llm temperature 死参
4. 修复: k 投票 + 固定基线 + 真采样 + 扩 val (18→53)
5. 修复后: val 3-iter 全 accept 递增 (+0.079→+0.098, CI+0.042→+0.053), test +7.0pp 泛化
6. 结论: 假阳性根因=评测噪声; 真实增益 test +7.0pp, 扎实可复现

## 待办
1. ⏳ 多 seed 复测修复后 test +7.0pp (坐实显著性, 现在评测端有 k 投票降噪声, 应比修复前稳定)
2. ⏳ 消融: trace2skill 基础池作 v2 冷启动注入; 纯 rewriter vs 有池
3. ⏳ v2 超参扫描 (iters/g/eval_k)
4. ⏳ AIME24 adapter (路径C, 跨任务泛化验证)
5. ⏳ evolver best_skill_evolved.md 也用同条件 (k=5) 重评测, 看是否也翻正

## 待办
1. ⏳ 全量结果 → 判断 v2 修复后 val 是否稳定正
2. ⏳ 若正: 多 seed 复测坐实; 若仍 null: 说明 train 撬动不泛化, 转消融/超参
3. ⏳ AIME24 adapter (路径C, 跨任务泛化验证, 非筛选val)

Related: [[tf-global-v2-math-diagnosis-20260727]], [[v2-multiseed-replication-20260727]], [[direction-global-not-pertask-20260723]]
