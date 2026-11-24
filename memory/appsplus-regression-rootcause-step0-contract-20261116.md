---
name: appsplus-regression-rootcause-step0-contract-20261116
description: APPS_Plus 23道退化题机制归因 — Step0假设已被4组消融证伪(推理级); 退化根因未定(闪存随机噪声是噪声签名非因果结论); 232路由已撤下(oracle泄漏)
metadata:
  type: project
---

# APPS_Plus 23 道退化题机制归因 (2026-11-16)

> ## ⚠️ 2026-11-16 更新: Step 0 假设已被 4 组消融证伪 (推理级敏感性), 232 路由已撤下
>
> 本文件原提出的机制假设 ("候选 Step 0 双接口分叉 → 部分 S₀-pass 退化 23") **已被 task #47 四组配对消融反驳**。删 Step 0 (C 组) 退化=22 ≈ B 的 23; 保守化 Step 0 (D 组) 退化=25 ≥ B。C/D vs B 配对全不显著 (p=0.65–1.00), 退化集合低重叠 (B∩C=9, B∩D=12 of 23), C/D 各引入 13 道新退化 → **在推理级 (产物级), 删/改 Step 0 文本段不降低退化**。完整结果见 [[step0-ablation-result-20261116]]。
>
> **范围界定 (2026-11-16 用户方法论修正后)**: task #47 是**推理级敏感性分析**, 不是训练机制因果检验。它只支持 "Step 0 文本段不是 23 道退化的推理级主因" + "退化跨组低重叠, 与生成不确定性一致"。**不支持**: "Step 0 (作为训练产物) 没价值" / "Evolver 训练机制无效" / "**退化主因 = flash 生成随机性**" (后者是噪声签名观察, 非因果结论; 退化的训练因果根因未定, 须训练时消融 task #48 验证)。
>
> **232/300 路由已撤下** (测试集 oracle 泄漏): 路由 `Y_i = S₀(X_i) if S₀ 在 test 通过 else Evolver(X_i)` 用 test S₀ 结果当 oracle, 候选不承担 S₀-pass 退化风险 → 232 是 Oracle-assisted 组合系统, 非 Evolver 能力。Evolver 独立 209 ≈ SkillOpt 211 (p=0.88 不显著)。详见 [[appsplus-routing-oracle-leakage-reframe-20261116]] 和 `docs/phase2_issues_202611.md` #73。
>
> **以下原文保留作为假设记录; 机制假设各节均应读作"已被推理级消融证伪的假设"。** 根因状态: Step 0 推理级已证伪, 训练因果根因未定 (待 task #48)。

## 背景

用户 Part II 点出"23 道退化题是很有价值的研究对象,可判断 skill 优化是扩展能力还是改变行为分布换局部收益"。
本文件给出**纯离线机制证据链** (零 flash 调用): 候选 skill 退化的**当时假设**根因是引入了接口契约分叉指令 (⚠️ 此假设已被消融证伪, 见顶部更新)。

## 数据 (纯离线, 三个 skill 正文都在盘)

| skill | 文件 | 字符数 | 接口处理 |
|-------|------|--------|---------|
| S₀ (我们的初始=SkillOpt初始) | `appsplus-skillopt-flash-.../skills/skill_v0000.md` | 1811 | **纯 stdin/stdout** (标题即 "Heuristics (stdin/stdout)") |
| 我们的候选 best_S | `tf-appsplus-routing-.../tf_global_skill.md` | 3038 | **双接口分叉** (新增 Step 0: 函数式 vs stdin 二选一) |
| SkillOpt best | `appsplus-skillopt-flash-.../best_skill.md` | 3000 | **纯 stdin/stdout** (保形强化, 无接口分叉) |

## 23 道退化题画像 (离线从 test_retention_results.json + S0 baseline 算出)

1. **无已记录 infra 异常**: 23 道 `agent_ok=False` = 0, 全部 `agent_ok=True` → 无记录到 Agent/API 层面异常。**⚠️ 降级表述**: agent_ok=True 排除已知基础设施故障, 但**不能排除模型随机性/生成波动/评测执行差异**。准确表述: "23 道回归样本均未记录到 Agent/API 异常, 且均发生在 S₀ 原本满分通过的题上, 说明回归有明确的输出/执行表现层面表现, 而非已知基础设施故障所致"。
2. **边界题假设证伪**: 23 道 S₀ 时 `soft=1.0 & pass_rate=1.0` 全部满分通过, `soft<1` 的边界题 = **0/23** → 回归发生在原本稳定通过的样本上, 不是摇摇欲坠的题。
3. **两簇失败表现** (⚠️ 失败类型≠根因类型): 5 道 `nonzero returncode=1` (执行崩溃) + 18 道 `stdout mismatch` (答案错误, 含 got 3 want 999999998 这类差异极大者)。crash 与接口契约不匹配有**一致性**但可能由其他代码错误导致; WA 可能来自接口误判/算法策略变化/边界处理错误/其他新规则干扰。**不能把失败类型直接等同根因**。
4. **与 SkillOpt 退化重叠**: 共同 6 (2209/4118/5808/6357/7135/7281, 两 skill 都破坏的内在难题) / 仅我们 17 / 仅 SkillOpt 8。

## 机制假设 (非已证因果): 候选的 Step 0 制造了 S₀ 没有的决策分叉

**⚠️ 降级**: 本节是**机制假设**, 不是已证因果链。当前证据支持"Step 0 是有解释力的假设", 但缺逐题证据 (候选实际生成什么代码、是否违反接口契约、是否由Step0引起)。验证见 4 组消融 (task #47)。

**S₀ 隐式接口一致**: 全文假设 stdin/stdout, 靠模型默认行为处理函数题 (题面有签名就用函数式)。一致但隐式。

**候选 Step 0 (新增首段)**:
> "If it shows a function/class signature (`class Solution`, type hints, `-> T`), implement exactly that method and **forbid** input()/print()/sys.stdin. If it is genuinely stdin/stdout, follow the parsing/output rules below."

→ 把隐式一致的接口判断**显式化为主动决策点**。**假设**这个分叉:
- **在 S₀-fail 题** (模型原本没认出接口→失败): 教会认函数接口 → 救回 24 (**Acquisition↑**)
- **在部分 S₀-pass 题** (模型原本靠默认行为稳过): 可能诱导重新判断接口 → 可能判错 → crash/wrong-answer → 退化 23 (**Preservation↓**)

**SkillOpt 对照**: best skill 标题仍 stdin/stdout, **保形强化** (sys.stdin.buffer.read().split()/solve()包裹/accumulate-and-emit/sentinel-1), **不碰接口分叉** → 扰动小 → 退化更少 (14 vs 23)。

## 因果链 (待验证假设, 非已闭合)

```
S₀ (纯stdin, 隐式接口一致)
   ├─ SkillOpt路径: 保形强化(同框架+实现稳健) → 救17/退化14, 扰动小
   └─ 我们路径: 契约重塑(加Step0双接口分叉) [假设]
        ├─ S₀-fail: 教会认函数接口 → 救回24 (Acq↑) [假设]
        └─ S₀-pass(部分): 默认行为稳过→Step0诱导重判接口 [假设]
              ├─ 5道判错接口→crash [假设, crash可能其他原因]
              └─ 18道判错路线→wrong-answer [假设, WA可能其他原因]
              → 退化23 (Pres↓), 全满分题, 无记录infra异常
```

**路由的角色**: 候选不跑 S₀-pass → preservation 代价被隔离 → 部署分数(232)只享 acquisition 收益不见 pres 代价。
→ 232 的显著性**完全来自路由架构** (26:5 = pass子集14:0路由保护 + fail子集12:5不显著候选差异), 候选skill纯能力与SkillOpt统计持平 (209≈211, p=0.88)。详见 [[appsplus-attribution-decompose-26vs5-20261116]]。

## 回答用户核心问题

> "Skill 优化是在扩展模型能力, 还是通过改变整体行为分布换取局部收益?"

**离线观察指向后者, 但机制尚未完成逐题因果验证。** 候选用一条更强但带分叉的契约指令 (Step 0) — **假设**它重塑了输出分布: S₀-fail 上救回更多 (acquisition), 代价是部分 S₀-pass 上引入新失败模式 (preservation)。不是单纯加能力, 更像契约重塑带来的分布偏移 (待消融验证)。

**最有价值的发现** (用户 Part V 判断): 不是"Evolver 更易退化" (配对 p=0.11 不显著, 未统计证明), 而是发现"**新增能力的同时引入显式规则 (Step 0 接口分叉) 可能扰动原本稳定的生成行为**"。这一观察可发展为 Regression-aware / Conditional Skill Evolution 方法 — 候选只在失败子集生效、不触碰已通过子集, 即路由架构已部分实现的设计。推进该方法前, 先做 4 组消融 (task #47) 验证 Step 0 假设。

## 可写/不可写 (论文口径)

- ✅ 可写: 23 道退化题均未记录 infra 异常 (agent_ok=True, 但不排除模型随机性/生成波动), 均发生在 S₀ 满分通过的题上 (非边界题), 失败表现为 5 crash + 18 wrong-answer (失败**类型**, 非根因类型)。
- ✅ 可写 (假设级): 机制假设 — 候选 Step 0 双接口分叉**可能**在 S₀-fail 上提升 acquisition、在部分 S₀-pass 上引入 preservation 代价; SkillOpt 保形强化不碰接口分叉故扰动更小 (14 vs 23 退化, 但 p=0.11 不显著)。
- ✅ 可写: 232 的显著性**完全来自路由架构** (26:5 = pass子集14:0路由保护 + fail子集12:5不显著候选差异); 候选 skill 纯能力与 SkillOpt 统计持平 (209≈211, p=0.88, 见 [[appsplus-attribution-decompose-26vs5-20261116]])。路由隔离 preservation 代价 → 部署分数只享 acquisition 收益 (路由设计目的, 也是 +21 的唯一来源)。
- ❌ 不可写: "退化全是接口错判 + 路线换错" (失败类型 ≠ 根因类型, crash/WA 可能由其他代码错误导致, 未逐题验证)。
- ❌ 不可写: "我们比 SkillOpt 显著更易退化" (配对 p=0.11 不显著, 见 [[appsplus-rescue-retention-paired-vs-skillopt-20261116]])。
- ❌ 不可写: "Rescue 显著优于 SkillOpt" (p=0.14) / "我们的 evolver 产出更强候选 skill" (候选纯能力持平 209≈211)。
- ⚠️ 待做: 逐题 code diff 证实接口错判 + 4 组消融 (删 Step 0 / 保守版 Step 0) 验证因果。**机制证据链尚未闭合**, 当前是"假设已形成并得多项离线观察支持, 但未完成逐题因果验证"。

## 数据源复跑路径

- 我们 retention: `tf-appsplus-routing-retention-20261113/run_20261113_152603/test_retention_results.json` (208 条, 7字段无code)
- S₀ baseline: `appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.json` (300, 含code/response)
- S₀ skill: `同/skills/skill_v0000.md` (1811字)
- 我们候选: `tf-appsplus-routing-.../tf_global_skill.md` (3038字)
- SkillOpt候选: `同/best_skill.md` (3000字)

关联 [[appsplus-rescue-retention-paired-vs-skillopt-20261116]] [[tf-evolver-appsplus-routing-retention-20261113]] [[step0-ablation-result-20261116]] [[appsplus-attribution-decompose-26vs5-20261116]]。

---

## 2026-11-16 消融终判 (task #47): Step 0 假设证伪, 根因修订

四组配对消融 (A=S₀复用 / B=当前候选复用 / C=删Step0新300调用 / D=保守证据驱动Step0新300调用, Holm-Bonferroni m=6) 结果:

- **Step-0-隔离比较 (C/D vs B) 全不显著**: Preservation C 13:14 p=1.00 / D 13:11 p=0.84; Acquisition C 8:10 p=0.81 / D 8:11 p=0.65。删/保守化 Step 0 不降退化 (C 22 / D 25 vs B 23)。
- **退化集合低重叠**: B∩C=9, B∩D=12 of 23; C/D 各"治好"B 的 11~14 道却"新弄坏"13 道别的 → 非重叠逐题翻转, flash temp=0 随机性指纹。
- **主终点 vs A=S₀**: 三组都显著异于 S₀ (Holm adj_p<0.001), 但 C (无Step0) rescue=26 ≥ B 24 → rescue 是"候选跑题"的一般效应, 非 Step 0 教接口识别所致。

**根因状态修订** (覆盖原 "机制假设" 各节, 2026-11-16 用户方法论修正后):
1. **Step 0 双接口分叉不是 23 道退化的推理级主因** (消融证伪 — 但仅推理级, 非训练因果)。
2. **退化根因未定**。flash 随机性 (45 道在 ≥1 组翻转, 仅 6 道三组共翻; agent_fail=0 全组) 是**噪声签名观察**, 解释"为何删 Step 0 不降退化" (退化跨变体低重叠), 但**不能据此断言"退化主因 = flash 随机噪声"** — 那是把相关性当因果。退化是否源自训练机制 (Evolver 学到的其他规则 / 训练数据 / 进化过程) 须由**训练时消融 (task #48)** 验证。
3. **Step 0 也不贡献 rescue** (C 无 Step 0 rescue=26 ≥ B)。
4. 候选 skill 纯能力 ≈ S₀/SkillOpt (209/212/210 vs 208/211, p=0.88), skill 内容在噪声内。**232/300 部署增益已撤下** (oracle 路由泄漏, 见 [[appsplus-routing-oracle-leakage-reframe-20261116]]), 非"完全归因路由架构" (旧表述隐含 232 是合法 Evolver 部署物)。

**论文口径修订** (覆盖原 "可写/不可写", 2026-11-16 用户方法论修正后):
- ✅ 可写: 消融 (4组×2终点, Holm校正) 表明 Step 0 接口分叉**不是**退化的推理级主因 (删/保守化不降退化, 配对对称)。退化跨变体低重叠, 与生成不确定性一致。
- ✅ 可写: 候选 skill 纯能力 (含/不含 Step 0) 统计上不区分 S₀/SkillOpt (209/212/210 vs 208/211, p=0.88)。
- ✅ 可写: Oracle-assisted routing 232/300 可作"组合系统上界"单独报告, **须明示用了 test oracle, 非 Evolver 能力**。
- ❌ 不可写: "Step 0 分叉导致 preservation 退化" (推理级证伪) / "保守证据驱动策略减少退化" (D=25≥B=23, 不显著) / "我们比 SkillOpt 更易退化" (p=0.11 不显著) / "232 是 Evolver 性能" (oracle 泄漏, 已撤) / "Evolver 显著优于 SkillOpt" (p=0.88 不显著)。
- ❌ 不可写: "**退化主因 = flash 生成随机性**" (噪声签名非因果结论, 训练因果根因未定)。
- ⚠️ 小效应不排除: ±2-3 的 Step 0 效应在噪声内不可检测, 但绝非 23 道退化的解释。
- ⚠️ 训练因果待验: 退化是否源自 Evolver 训练机制须 task #48 训练时消融验证, 推理级消融不能回答。

**机制证据链状态**: Step 0 推理级因果**已闭合 (负结果)**; 退化训练因果根因**未闭合 (待 task #48)**。原"假设已形成待逐题验证" → 现"4组推理级消融已跑, Step 0 推理级假设被证伪; 退化训练根因未定, flash 随机性是噪声签名观察非因果结论"。逐题 code diff 不再做 (推理级消融已否定 Step 0 因果, 逐题验证 Step 0 错判无意义)。
