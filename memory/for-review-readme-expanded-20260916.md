---
name: for-review-readme-expanded-20260916
description: for_review/README.md 大幅扩充 — 加算法发展史(Phase1 GRPO→Phase2 402skills→转折诊断→facet-tree→三break修复→pool_filter)+多维度review问题(7维度28点)+诚实局限
metadata:
  type: project
---

# for_review README 扩充完成（2026-09-16）

> 用户要求「写得详细些，把整个算法的发展历史和目前的最新结果都写好，并尽可能从多角度让 review 提建议」。原 README 127 行 → 扩充至 ~280 行 9 节。

## 新增/扩充章节

1. §2 **算法发展史**（新增）— 6 个阶段：
   - 2.1 Phase1 参数空间优化（GRPO，6-16 休眠）
   - 2.2 Phase2 402 skills（YAML bug + 70%有害 + 消融不显著 p=0.87）
   - 2.3 转折诊断（Task3/4 REFUTED 0.3703 + Task5 probe-split 0.4654 天花板 + eval 去偏）
   - 2.4 facet-tree 设计（DIVE/DarwinX/EvoAgent 缝隙 + 诚实标注预设vs涌现）
   - 2.5 三 break 修复（①复合节点段进遗传池 ②crossover反协同 ③子集救援）+ 9/2 结果
   - 2.6 pool_filter 三关（9/15 0.5645 + +0.0103 非受控净效应诚实披露）

2. §3 **结果**扩充 — 加 CI/p 值/seed 明细 + evolver_log 逐字段核实 + 与 SkillOpt 同窗口无定论

3. §5 **算法机制详解**（扩充）— 演化循环主线图 + 函数索引 + **fitness 来源诚实标注**（演化线纯 EM 判卷，70/30 LLM 评判在 GRPO 链路非演化线）

4. §7 **Review 问题**（大幅扩充）— **7 维度 28 个具体可批判点**：
   - A 实验设计（val→test 负迁移/select 穷尽/4-seed/cache 对位）
   - B 统计方法（bootstrap 配对/5峰多检验/CI判据混用/SkillOpt同窗口）
   - C 代码实现（cache复用/sim分母/expand确定性/救援逻辑）
   - D 叙事框架（GRPO定位/涌现声称/双重佐证强度/SkillOpt叙事）
   - E 相关工作（novelty边界/DIVE区分/collapse风险）
   - F 可复现性（数据公开/API确定/代码版本漂移）
   - G 局限未来（数据天花板/meta结构/单一fitness/pool_filter未闭合）

5. §8 **已知局限与诚实披露**（新增）— 7 条 caveat 全列

## 数字核实

全部结果数字已对 `results/3break_trajectory/evolver_log.json` 逐字段核实：init_skill_hash 68571cb2a6、baseline 0.0235、8 iter 0.127→0.357、archive 42、xo 3。test 数字（0.2814/0.5542/0.5472/0.5645/0.5629）来自 three-breaks memory。

## 关联

[[for-review-package-built-20260916]]（包构建）/ [[three-breaks-facet-tree-20260902]]（算法史素材源）/ [[same-window-skillopt-paired-ci-20260907]]（SkillOpt 无定论）

---

## 二次扩充（2026-09-16 晚）— 证伪分支表 + 失败模式综合 ★

> 用户细化要求「主要是要把已经尝试过的算法和现在的算法说清楚，让他 review 一下可能的 bug 和方法中的缺点」。上一版 §2 算法史是 6 段散文，缺"尝试过被证伪分支"的对照——这正是 reviewer 找 bug 的真正抓手。

### 新增 §2.7「尝试过但被证伪/放弃的算法分支」（★ review 重点）

六条独立失败分支的对照表（test 数字均 4-seed eval_k=3 test=106，对比基线=当时最强公平 artifact）：

| # | 分支 | 日期 | test | 判定 | 被证伪根因 |
|---|------|------|------|------|-----------|
| 1 | expand-v1（算子G首版） | 8/28 | 0.4654 | 🔴 REFUTED→修v2 | expand 盲走最难题 / 新段不持久被 overlay 抹掉 / crossover 硬编码 4 段排除 grown skill |
| 2 | 药一（Beta-Binomial LCB） | 9/14 | 0.4064（最差） | 🔴 REFUTED | LCB 强收缩压扁 `val_ci>0` 门，进化早期连正常精英都进不了池 |
| 3 | 药三（剔除 train18） | 9/14 | 0.5141 | 🔴 REFUTED | train18 非纯泄露而承载有效信号，缩选择集反而加重赢家诅咒 |
| 4 | 方向E（pop_mutation） | 9/10 | 0.4693 | 🔴 REFUTED | val 加权采样=显式往 val 过拟合推；n_parallel=1 下 ci 加权退化为单 lineage |
| 5 | 路径B（meta_contrast） | 9/11 | 0.4701 | 🟡 机制成立但 ceiling-bound | meta 规则自主归纳成确定性决策段（科学闭环），但收益被 meta 子集 test 占比 43.4% 封顶 |
| 6 | warm-strengthen | 9/7 | 0.5417（回归） | 🟡 未越北星 | strengthen 全程不 fire（饱和判定过严），select 改善未迁移 test |

**四个反复出现的失败模式**（= 方法真实缺点，写入 §2.7 末尾，§7 A/B/C 锚定）：
1. **val→test 负迁移**（4/6 中招）— 贯穿唯一瓶颈，从未根治，只被强信号压过噪声
2. **~0.57 天花板** — 4 个 artifact 全撞 0.55-0.57，n4 vs n6 持平
3. **选择压力无意偏向 val** — pop_mutation 的 val 加权 + expand-v1 攻 val 最难题，`--pool_filter` 5 峰尖峰关是否同风险
4. **算子静默失效** — strengthen 不 fire / pop 退化为单 lineage，表现为"不 crash 不报错"最难发现

### §7 Review 问题锚定改造

A1/A2 提升为 ★ 最高优先级，直接指向 §2.7 模式 1（val→test 负迁移）+ 模式 2（~0.57 天花板）。B1 补 has_meta router 伪精细教训。B2 关联模式 3（5 峰 val 偏向）。C1 提升为 ★ 算子静默失效（最高优先级 bug 候选）。每个 review 点现在锚定到具体行号/机制/失败分支，而非抽象泛问。

### on/off 受控对照状态更新（live）

§2.6/§3/§8/§9 全部标注：on/off 受控对照 2026-09-16 18:18 启动，ON 臂（`--pool_filter` 三关）+ OFF 臂（唯一变量=无 flag）从同一 S₀ + 同一现场重生成 s0_cache 并行起跑。s0_cache 已现场重生成（71/71，双峰分布 0.0×55 / [0.7,1]×16，mean 0.225，无 None 无负值），消除 cache 采样偏差。两臂各 ~8h → 4-seed test → ON−OFF 配对 bootstrap CI。

### 数字来源

六条分支的 test 数字（0.4654/0.4064/0.5141/0.4693/0.4701/0.5417）逐条对 MEMORY.md 顶部方向实验条目核实（[[yaoyi]]/[[yaosan]]/[[direction-e-pop-test-verdict]]/[[meta-rule-pathB-test-verdict]]/[[strengthen-warm-theo6-run]]/[[expand-v2-4seed-result]]），无记错。
