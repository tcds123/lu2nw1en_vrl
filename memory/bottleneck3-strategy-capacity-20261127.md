---
name: bottleneck3-strategy-capacity-20261127
description: 第三个瓶颈定位 — Run A facet_tree 6轮 best_n_sec 始终=4, expand 6次全0晋升, 策略容量被锁在 S₀ 的 4-section 结构内; theo_iter6/SkillOpt 靠新增 section 获大幅增益
metadata:
  type: project
---

# 第三个瓶颈: 策略容量 (2026-11-27, 零API离线分析)

## 触发 (用户洞察)
用户指出: 旧 Evolver theo_iter6 (LiveMath 55.42%) 距 SkillOpt (56.05%) 仅 0.63pp, 新机制 Run A (40.57%) 反而跌回历史水平之下. 三个 Skill 长度对比暴露瓶颈:
- S₀ 1018c / Run A best 1624c / theo_iter6 7608c / SkillOpt v0002 9922c
- "我们的 Evolver 是否过度限制了候选的表达空间和策略容量?"

## 决定性证据 (Run A evolver_log operator_counts + best_n_sec 轨迹 + 9月对照)

```
Run A (11月) operator_counts:
  facet_edit : attempted=18  promoted=2  ← 唯一产生晋升的算子
  expand     : attempted=6   evaluated=6  admitted=0  promoted=0  ← 6次全0晋升
  overlay    : attempted=6   promoted=0
  crossover  : attempted=1   promoted=0
  strengthen : attempted=0   (未触发)
  best_n_sec 轨迹: iter1=4 iter2=4 ... iter6=4  ← 全程=4, 从不增长
  archive section 分布: {5: 19}  ← 19候选全5段, 无一新段

9月 run_20260904 对照 (同 --facet_tree --expand):
  archive 42候选 section 分布: {5:29, 6:4, 7:2, 8:1, 9:3, 10:3}  ← 结构增长丰富
  expand 候选成功入 archive:
    - expand:Existence vs Un → 7sec/4502c delta=+0.155 ci=+0.099 ✅
    - expand:Implication Dir → 9sec/8012c delta=+0.127 ci=+0.070 ✅
  最长候选 10211c (Run A 最长仅 1970c)
```

**结论**: 同样的 facet_tree + expand 框架, 9月能长出新段并晋升, 11月 Run A 结构锁死. **不是代码 bug, 是 v2 覆盖策略与失败分布的交互** (见根因).

## 机制根因 (精确, 2026-11-27 核实)
**expand 饿死的机制链条**:
1. Run A 全程 18 次诊断, meta-option 失败平均 7.8/8 (**98% 是 meta**). Run B 同样 (iter1 诊断全 meta).
2. `find_uncovered_type_for_expand` (L747-768) **显式跳过 meta-option 类型** (L763: `if lbl==meta_lbl: continue`) — 这是 v2 有意设计 (修 v1 expand 攻 meta 致 -0.07 退步; L752: "meta 交由 facet_edit 轴; expand 攻 not-meta 未覆盖类型").
3. S₀ 4 section 段名签名可能已"覆盖"多数非-meta 类型 → `target=None` → expand 饱和跳过/转 strengthen.
4. 偶尔找到 target, 生成的候选 eval 后没赢过 incumbent → 0 晋升, 不入 archive.
→ 当 meta 主导失败时, expand 被饿死; facet_edit 又只能在固定 4 段内编辑 → 结构增长停滞.

**9月 run_20260904 为何能**: 失败类型分散 (Existence/Uniqueness、Implication Direction 等非-meta 有足够占比) → expand 有 target 可攻 → 成功长段. 与 11月 Run A 的失败分布差异, 不是代码差异.

**expand 无 gate_trace 记录** (grep 证实): operator_counts=6 但 gate_trace=0, 因 expand 调用链未加 gate_trace.record. 这是 P0-3 修复仍遗漏的覆盖缺口 (audit 用).

## 三方 Skill section 结构对比 (核心)

| Skill | 总长 | section数 | 关键结构 |
|-------|------|----------|---------|
| S₀ (initial.md) | 1018c | 4 | 基础 4 段 |
| **Run A best** | 1624c | **4 (同S₀)** | 只 Option Comparison 314→921c 加长, 无新段 |
| theo_iter6 (671135ff, run_20260902) | 7608c | ~9 | facet_tree 产物 (args.facet_tree=True), 含 bidirectional verification 等新段 |
| run_20260904 iter6 (823b4a68, 同算法同源) | 8394c | **9** | Implication Direction Check 1158c + Existence vs Uniqueness 1993c + Existence/Uniqueness/Quantifier 2344c (可读文本已恢复) |
| SkillOpt v0002 | 9922c | **10** | Meta-Option Decision Procedure 2564c + Meta-Option Strategy 2143c (meta-option 专2段占48%) + 5 新段 |

## 修正: theo_iter6 是 facet_tree 产物, 非旧无约束 evolver
此前判断"旧 evolver 无 facet_tree 约束能自由长段"**错误**. 核实: theo_iter6 来自 run_20260902_205708, `args.facet_tree=True` (memory/pertype-strengthen-run-20260908 记载). run_20260904 同期同算法, iter6 9sec/8394c 含 Implication Direction Check. → 9月 facet_tree 能 expand, 11月 Run A 不能, 差异在**失败分布 + v2 覆盖策略**, 非代码框架. 代码 L1654-1666 已把 "Implication Direction & Biconditional Check" 段固化进 _self_test_facet 测试, 证明这是 facet_tree 时代曾经成功 expand 的有效结构.

## 消歧义 (用户要求, 不能合并)
- **theo_iter6 (LiveMath 55.42%)**: run_20260902_205708, 671135ff, 7608c, test=106 k=3 4-seed. 干净S₀路径6轮.
- **+10.8pp**: AIME25 数字 (2026-08-07, v2 skill 6f0e7669, test=30). **不同任务/不同skill/不同run, 严禁与 theo_iter6 LiveMath 结果合并**.
- northstar (0908): theo_iter6 55.42% vs skillopt 56.05% (gap −0.63pp, CI跨0, 同窗口4-seed)

## S₀ 变化 (H1-H4, 待逐题核查)
northstar(0908) S₀=0.2956 vs Run A(1126) S₀=0.3679, 同k=3同split同S₀skill 差7.2pp. 候选解释:
- H1 V3.2 服务端版本变化 (用户确认历史也V3.2, 但版本可能更新)
- H2 evaluator/shuffle/答案提取逻辑变化 (import链同源, 但版本可能变)
- H3 采样随机波动 (7pp 远超 k=3 投票的合理波动, 可能性低)
- H4 缓存/路由/生成参数差异 (待查)
**下一步**: 复用历史 per_s0 + Run A s0_cache 逐题对比, meta(46) vs not_meta(60) 分层. 若增长主要在 meta → 基础模型变强削弱了 meta 增益; 若在 not_meta → 需其他解释.

## 对 Run E 的指向 (用户 P2)
自适应编辑范围与策略容量: 核心是**让 expand 重新能攻 meta-option** (解除 L763 "expand 跳过 meta" 限制, 或给 meta 一个独立 expand 路径如 SkillOpt 的 Meta-Option Decision Procedure). max_tokens=2048 **非瓶颈** (当前未触及), 不需先调. 需 (a) expand 获独立搜索预算不被 facet_edit 局部改进压制, (b) 解除 expand 跳过 meta 的 v2 限制 (或为 meta 设计专属结构扩展), (c) 动态 slot registry 让新段进可继承全局结构. 这是 B(广度)/D(累积) 之外的第三条实验线.

## POOL-FILTER 配置核查结论
用户担心 B 的 pool_filter=OFF 冻结被破坏. 核查: **Run A 和 Run B 都开了 --pool_filter** (launch 脚本 L31/L38, 日志 "opt-flags ON: pool_filter"). Run B Theorem 的 "POOL-FILTER 拒" 来自代码 L3152 (facet_edit 的 pool_filter 关). **B 与 A 配置一致, B 唯一变量是 n_parallel 1→4. 不是配置差异, 是日志如实反映了 pool_filter=ON 的行为.** (用户记忆的 pool_filter=OFF 可能指历史 theo_iter6 的字节复现配置, 非 Run A/B.)

**关联**: [[livemath-fullrun-launched-20261126]] (Run A 结果) / [[northstar-test-baseline-20260908]] (theo_iter6 vs skillopt) / [[bidirectional-test-convergence-20260904]] (theo_iter6 7608c 特征) / [[cross-ax-inherit-design-20261127]] (Run D 累积机制)

## S₀ 逐题差异分析 (2026-11-27, 零API, **select=71题** 9月 vs 11月; ⚠️口径: 非test=106, meta/not_meta=39/32非46/60)
复用 9月 run_20260904 s0_cache + 11月 Run A s0_cache (同 S₀=initial.md, 同 select=train+val=71, 同 k=3, 同 split).

| 分层 | n | 9月 pass | 11月 pass | Δ | 改善/退化 |
|------|---|---------|----------|---|----------|
| 全体 select | 71 | 0.216 | 0.254 | +3.76pp | 16/15 |
| **meta** | 39 | 0.120 | 0.086 | **−3.42pp** | 3/8 |
| **not_meta** | 32 | 0.333 | 0.458 | **+12.50pp** | 13/7 |

### 关键发现 (颠覆预期)
1. **S₀ 增长几乎全来自 not_meta (+12.50pp), meta 反而退化 (−3.42pp)**. 不是"基础模型变强削弱 meta 增益", 而是 meta S₀ 本就更弱了.
2. **meta 题 S₀ 仅 8.6% pass (几乎全错)** → 这是最大可优化空间, 也是 theo_iter6/SkillOpt 通过新增 meta-option 决策程序获大幅增益的地方.
3. **not_meta S₀ 升至 45.8%** → not_meta 可优化空间被压缩 (SkillOpt/theo_iter6 在 not_meta 上的 headroom 变小).
4. 逐题: not_meta 多题 0.33→1.0 (基础模型直接做对); meta 题 202512:8 1.0→0.33 (退化), 202601:45 0.67→0 (退化).

### 对策略容量假设的闭环支持
meta S₀=0.086 (几乎全错) = 最大优化空间. 但 Run A expand 被 v2 策略禁止攻 meta (L763), facet_edit 只能在固定4段内编辑 → 无法建立 meta-option 决策程序 → Run A 在 meta 题上无法复制 theo_iter6/SkillOpt 增益. **策略容量瓶颈 + S₀ 变化共同解释了 Run A 退化**: meta headroom 最大但机制够不着, not_meta headroom 被基础模型压缩.

### H1-H4 排除
- H3 (采样波动): 3.76pp 远超 k=3 投票合理波动, 且 meta/not_meta 方向相反 → 排除纯随机.
- H1/H2/H4 (模型版本/evaluator/路由): import链同源已证, 但 select 集 meta 退 +not_meta 涨的方向性变化更像是模型能力迁移 (V3.2 版本更新让 not_meta 更强但 meta 某些题退化), 非纯 evaluator 变化. 待 test=106 同口径确认 (需重测 S₀ test).
