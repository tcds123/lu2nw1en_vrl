---
name: appsplus-attribution-decompose-26vs5-20261116
description: 26:5拆解+候选vs候选纯能力209≈211 — 232显著性来自路由架构非候选能力 (修正归因)
metadata:
  type: project
---

# APPS_Plus 归因核对: 26:5 拆解 + 候选纯能力持平 (2026-11-16)

## 起因: 外部审阅指出 12:5 vs 26:5 不一致

若两系统都路由(都复用S0-pass、都只在fail跑候选), 全量分歧应=fail子集分歧. 但 12≠26.
→ 比较对象不是"两个路由系统", 而是架构不对等的比较.

## 真相: 232=路由系统, 211=非路由候选系统 (架构不对等)

| 系统 | pass子集(208) | fail子集(92) | regress | 总分 |
|------|--------------|-------------|---------|------|
| 我们 232 | S0 cache(全pass) | 我们候选独立 | **0**(路由保护) | 232 |
| SkillOpt 211 | **SkillOpt候选独立** | SkillOpt候选独立 | **14** | 211 |

SkillOpt的211是候选独立系统(regress=14≠0), 我们的232是路由系统(regress=0).
→ 26:5是"路由系统 vs 非路由候选系统", 非"两候选系统比".

## 26:5 逐题拆解 (已闭合)

```
全量 26 vs 5  =  pass子集 14 vs 0  +  fail子集 12 vs 5
                  ↑路由保护贡献        ↑候选rescue贡献
```

- **pass子集 14 vs 0**: 我们路由保住S0全208pass, SkillOpt候选退化14 → 净赢14. 纯架构差异, 不反映候选能力.
- **fail子集 12 vs 5**: 候选vs候选真能力(12:5, p=0.14, 不显著).

核对: ours_only 14+12=26 ✅, SO_only 0+5=5 ✅.

## 新检验: 候选 vs 候选纯能力 (全300, 都不路由) ★

这是此前没做的关键检验. 把我们候选也还原成非路由独立系统
(pass用retention候选独立 + fail用routed候选), 与SkillOpt候选比:

| 比较 | 分数 | discordant | p | 含义 |
|------|------|-----------|---|------|
| 候选vs候选纯能力(都不路由) | 209 vs 211 | 20 vs 22 | **0.878 ❌不显著** | 候选能力持平, SO点估计略优2题 |
| 都路由(架构对等,归约fail子集) | — | 12 vs 5 | 0.143 ❌不显著 | =Rescue配对 |
| 路由系统vs非路由候选(架构不对等) | 232 vs 211 | 26 vs 5 | 1.9e-4 ✅显著 | 含架构差异 |

## 修正归因 (核心结论)

**232的显著性完全来自路由架构, 不是候选skill的能力优势.**
- 26:5中: 路由保护14:0(纯架构) + 候选rescue 12:5(能力但p=0.14不显著).
- 候选纯能力本身: 209≈211 (p=0.88) 持平, SkillOpt点估计略优.

**诚实口径修正**:
- ❌ 旧表述"232显著性主体来自路由"(模糊, 暗示候选也有贡献) → 改为:
- ✅ 新表述"232的显著性完全来自路由架构(14:0架构优势 + 12:5不显著的能力差异); 候选skill纯能力与SkillOpt统计持平(209≈211, p=0.88)".
- ✅ 可写: 路由是把"候选能力持平"变成"部署分数+21"的唯一因素.
- ⚠️ 不可写: "我们的evolver产出更强候选skill" (候选纯能力持平, 209≈211).

## B组复用核验 (为4组消融做准备, 已通过)

A/B复用前提: B组208 retention + 92 routed-fail 须来自同一skill/配置.
- 覆盖: retention 208/208 pass, routed 92/92 fail ✅
- 路由保护确认: routed pass子集与S0 0不一致 (B组pass确是候选独立非复用) ✅
- 字段结构一致: retention/routed 同7字段 ✅
- B组重建: 185保持 + 24rescue = 209/300, 与归因脚本吻合 ✅

## 功效启示

候选vs候选纯能力 p=0.88 是"真不显著"(方向都偏SkillOpt), 非功效不足.
Rescue/Retention配对 p=0.14/0.11 是"方向有利/不利但功效不足"(discordant太少).
三者性质不同, 不可混为一谈.

## 数据源 + 脚本

- 归因分解脚本: `eva/appsplus_attribution_decompose.py`
- 数据: `paired_vs_skillopt_analysis.json` (在OURS run目录)
- A组=S0: `appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.json`
- B组=候选独立: retention(208) + routed-fail(92) 拼装

关联 [[appsplus-rescue-retention-paired-vs-skillopt-20261116]] [[appsplus-regression-rootcause-step0-contract-20261116]] [[tf-evolver-appsplus-routing-retention-20261113]]。
