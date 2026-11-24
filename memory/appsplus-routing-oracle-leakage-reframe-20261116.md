---
name: appsplus-routing-oracle-leakage-reframe-20261116
description: APPS_Plus 232/300 路由=测试集oracle泄漏, 撤下核心结论; Evolver独立209≈SkillOpt211(p=0.88不显著); 转训练时消融
metadata:
  type: project
---

# APPS_Plus 路由评测重构 (2026-11-16, 用户方法论复盘)

## 触发: 用户的批评 (核心成立)

用户指出 232/300 路由方案有结构性评测设计问题, 可能构成"测试集信息泄漏式的评测投机":

路由决策 `Y_i = S₀(X_i) if S₀(X_i) 在 test 集通过 else Evolver(X_i)` — **判断条件依赖 test 标签对应的实际通过结果, 不是推理时可获得的信息**。

- turn1: S₀ 跑完全部 test 题 (208 pass / 92 fail, 用 **test 集** S₀ 真实结果)
- turn2: 仅在 S₀-fail 92 题跑候选
- → Evolver 不承担 S₀-pass 题上任何回归风险 → 结构性零退化

这是 Oracle-assisted 组合系统, **不是 Evolver 独立解题**。代码已逐行复核: `tf_skill_optimizer_appsplus.py:eval_on_routed` L284-317。

## 审计: 泄漏在哪一层 (子代理逐行核实 710 行)

✅ **训练/选择侧无泄漏**:
- gate (`accepted = new_delta >= best_val_delta` L544) 用 `eval_on_routed(VAL, S_new, s0_cache_VAL)` L543 — **val cache only**
- 训练反馈循环 (L527-536) 用 `s0_cache_train` (train 80 与 val/test 不相交)
- skill 生成 LLM (`make_rewriter_user` L111) **不收** fail_reason/code/error, 只收 question+sample I/O+experience pool
- 三段 cache 源自三个不相交 split

❌ **部署评测口径泄漏**: `eval_on_routed(TEST, best_S, s0_cache_test)` L572 把 test S₀ 结果当路由 oracle → 232 含 test oracle 贡献。**test cache 本身只用于 post-selection 最终评测 (L572/578/606), 问题是这个"最终评测"的评测函数本身就是 oracle 路由**。

## 结论修正 (用户建议已采纳)

| 数字 | 性质 | 能否作 Evolver 结论 |
|---|---|---|
| 232/300 (routed) | Oracle-assisted 组合系统 | ❌ **撤下**, 改 "Oracle-assisted routing" 单独说明 |
| 209/300 (Evolver 独立) | 候选 skill 全题独立跑 | ✅ **Evolver 独立能力** |
| 211/300 (SkillOpt 独立) | SkillOpt best skill 全题独立跑 | ✅ SkillOpt 独立能力 |
| 209 vs 211 | 配对 p=0.878 不显著 | ❌ **撤销** "Evolver 显著优于 SkillOpt" |
| 23 regress (通道B) | 候选在 208 S₀-pass 独立退化 | ✅ 真实 preservation 代价 (诊断口径, 非路由) |

**撤销的旧结论**:
- ❌ "Evolver 达到 232/300" → 撤, 改 Oracle-assisted routing
- ❌ "Evolver 显著优于 SkillOpt" (vs best 211 p=1.8e-4) → 撤 (那是 oracle 组合 vs 独立, 架构不对等)
- ❌ "232 显著性完全来自路由架构" (旧归因, [[appsplus-attribution-decompose-26vs5-20261116]]) → 修正: 路由架构优势 14:0 **本身是 oracle 保护** (S₀-pass 题复用 S₀ 答案, 候选不跑), 这正是泄漏点, 不是"架构优势"而是"评测不要求候选承担风险"

**保留的结论**:
- ✅ 候选 skill 在 208 S₀-pass 上独立退化 23 (通道B retention, 诊断口径, agent_fail=0, McNemar p=2.4e-7) — 真实 preservation 代价
- ✅ 候选 skill 在 92 S₀-fail 上独立救回 24 (通道A acquisition, 诊断口径)
- ✅ 候选 vs SkillOpt 纯能力 209 vs 211, p=0.878 — 统计持平, 点估计 SkillOpt 略优 2 题
- ✅ task #47: Step 0 文本段不是退化的推理级主因 (降级为指令敏感性, 非训练因果)

## 评测原则 (确立)

**test 集只评不决策**:
- 训练: 可用 train 数据上 S₀ 成功/失败轨迹指导 skill evolution
- 验证: 可选 skill / 调提示 / 定超参
- 测试: 冻结最终 skill, 每题独立运行, **不根据 test 标签/历史通过/test 结果缓存切换输出**

未来研究动态路由: 须用 **val 集**训练/确定路由策略, test 上只执行**推理时可获得信息**的路由 (如题面函数签名路由, 非 test S₀ 通过结果)。

## 转向: 训练时消融 (task #48)

用户核心方向: **关键不是删某条规则, 是验证进化过程产生的规则是否有效**。最有价值 = 训练阶段消融。

Evolver skill-gen 流程 (子代理已核实, `tf_skill_optimizer_appsplus.py`):
```
train_items = S₀-fail 题 (L502-504) 
  → rollout_group: make_rewriter_user(s0, question, sample_io, experiences) (L111/192) 生成 g 个 skill 变体
  → run_one 每个 (L152) → delta = pass - s0_pass (L212)
  → distill_experiences (L218/533): 高delta vs 低delta 变体对比 → 抽取 experience
  → merge_experiences (L242/538): 跨题合并 experience_pool
  → compose_global_skill(s0, experience_pool) (L258/270): LLM 重写 → S_new
  → gate: eval_on_routed(VAL, S_new, s0_cache_VAL) (L543) → best_S
```

**Step 0 来源 (待修复后重验, 见 issue #74 / codex D2)**: LLM 自发归纳, 非硬编码/模板 — 这一点成立。train 中含函数签名题 (LeetCode 式), S₀ (stdin/stdout 偏置) 让模型写 input()/print() → 全测失败。⚠️ 但"涌现路径"现不确定, 因 codex D2 (本会话独立确认): 现包 `rollout_group` rewriter 调用 L191-193 传 dict 作第3位置参 → TypeError → 静默回退 → **g 个变体全 = S₀, rewriter 从未执行** (见 [[appsplus-codex-review-crossref-20261116]] + issue #74)。两种可能: (a) 若历史运行用了正确代码: rewriter 变体自发加接口识别 → delta 升 → distill 抽出 G0 (原故事); (b) 若历史运行用了 bug 代码: 变体全 S₀, delta 是噪声, G0 来自 distill LLM 读到题面 `class Solution` 后**主动建议**接口识别, 非 delta 驱动选择。**文件 untracked + mtime 在运行后 ~3h → 无法回追历史实际代码** (codex §9: "证据不足, 不追认旧运行")。须 task #50 修复后小规模闭合验证, 确认 G0 真实涌现路径, 再定 Step 0 来源叙事。

**训练时消融杠杆点** (file:line):
| 目标 | 拦截点 | 改动 |
|---|---|---|
| 去接口错误反馈 (隐藏函数签名题) | L502 | fail_items_all 过滤掉含 `class Solution`/`def `/`-> ` 签名的题 |
| 去成功案例 (隐藏 S₀ scaffold) | L192 call + L111 fn | make_rewriter_user 传 `""` 作 s0 |
| 去成功案例 (compose 层) | L540 call + L258 fn | compose_global_skill 传 `""` 作 s0 |
| 抑制接口 experience | L533-534 | distill 输出后过滤含 "interface"/"signature" 的 experience |
| 抑制 (merge 层) | L538 | merge 后过滤 experience_pool |
| 改失败类型喂入 | L533 | distill 前按 fail_reason 过滤/置零 delta |

各组**独立生成 skill**, 然后独立 (非路由) 评测全 300 题 — 这才是训练机制因果检验。

## 论文口径 (修正后可写)

- ✅ Evolver 独立能力 209/300, SkillOpt 独立 211/300, 配对 p=0.878 统计持平 (无证据 Evolver 优)
- ✅ 通道B 诊断: 候选在 S₀-pass 上独立退化 23 (preservation 代价真实存在)
- ✅ Step 0 是训练中从函数签名题失败**涌现**的规则 (非注入), 但删它不降退化 (task #47 推理级)
- ✅ Oracle-assisted routing 232/300 可作"组合系统上界"单独报告, 须明示用了 test oracle, 非 Evolver 能力
- ❌ 不可写 "Evolver 优于 SkillOpt" / "232 是 Evolver 性能" / "Step 0 是退化主因"

关联 [[step0-ablation-result-20261116]] [[appsplus-attribution-decompose-26vs5-20261116]] [[appsplus-rescue-retention-paired-vs-skillopt-20261116]] [[tf-evolver-appsplus-routing-retention-20261113]]。详见 `docs/phase2_issues_202611.md` #73。
