# 峰值诊断: floor-regression 假设推翻, 真根因=val→test 负迁移 (2026-09-09)

## 实验
评测峰值 skill `best_skill_iter3.md` (cb901c5c, val_delta +0.319) 于 test=106 4-seed k=3, 与 iter4 (07bc48c6, val +0.286) + skillopt 对比。

## 结果 (反直觉)
| skill | val_delta(select) | test mean(4seed) | 迁移比(test_gain/val_gain) |
|-------|-------------------|------------------|---------------------------|
| 峰值 cb901c5c | +0.319 (高) | **0.5157** (低) | 0.2201/0.319=0.69 |
| iter4 07bc48c6 | +0.286 (低) | **0.5393** (高) | 0.2437/0.286=0.85 |

**val 越高 → test 越低 = 负相关/负迁移。** 选峰值会从 0.5393 掉到 0.5157 (更差)。

## 两个 paired CI 坐实
1. **峰值 vs skillopt (北星)**: Δ=−0.0448, CI95=[−0.0912,+0.0008], 下界<0 跨0 → **未越 ❌** (上界贴0, 比 iter4 +0.0275 更差)。
2. **峰值 vs iter4 (floor 假设检验)**: Δ(峰−iter4)=−0.0236, CI95=[−0.0605,+0.0134], 跨0 不可区分, 但**点估计峰值更低** (正/负=31/47)。

## 结论: floor-regression 假设被推翻
- floor 救援的 iter4 (val 低) 在 test 上**显著不弱于、点估计更强于**峰值 → floor 不是病根。
- **修 floor 救不了北星**: 即使输出 archive 峰值 (cb901c5c), test=0.5157 < iter4 0.5393 < skillopt 0.5605, 仍没越。
- 之前 [[northstar-direction-b-verdict-20260909]] 把 floor-regression 列主因 → **修正: floor 是次因, 真根因是 val→test 负迁移**。

## 真根因: 单 lineage 过拟合 val (坐实)
- val(select 口径, probe=train+val=71) 高分不迁移到 test=106, 甚至负相关。
- 迁移比随 val 升高而降 (峰值 0.69 < iter4 0.85) = 过拟合 val 的典型特征: val 越拟合, 泛化越差。
- train 仅 18 题, 单 lineage keep-best 在小集上过拟合 val 的能力 > 泛化能力。
- 历史佐证: 8/27 evolver −2.9pp, 9/9 方案1 −1.65pp, 9/9 方向B −2.12pp, 全未越 = 同一根因反复。

## 决策: 方向 E (archive 驱动 GA) 升为主方向
floor 修复不再是优先项 (治不了根)。真根因 = 单 lineage 过拟合 val → 正是 [[direction-e-archive-ga-crossgen-20260909]] (用户提的跨代遗传) 治的本。
但方向 E 须配防过拟合机制 (DIVE 多种群 bootstrap / fitness sharing), 否则小 train 上 GA 会更过拟合 val。
- 落点优先级调整: ①输出环节改 archive max (仍做, 防再被 floor 拖累, 但非主杠杆) ②**突变来源=archive 精英种群采样** (主杠杆, 打破单 lineage) ③交叉池跨代重组。
- 方向 C (ancestor-chain) 治标层面也弱于 E (C 仍单 lineage 回看, E 是种群级), 暂缓。

## 教训
1. **反直觉但自洽**: val 高≠test 高, floor 救的低 val 产物反而更泛化 → select 口径本身有偏 (train+val=71 含 train 18, 过拟合放大)。
2. **诊断要 paired CI**: 单看点估计会误判 (峰值 0.5157 vs iter4 0.5393 差 2.4pp 看似明显, 但 CI 跨0 不可区分 → 不能判"峰值显著更弱", 只能说"floor 没帮上也没明显害")。
3. **守 CI 纪律**: 峰值 vs skillopt 上界 +0.0008 贴 0 → 极接近但未跨, 不报"追平" (下界仍<0)。

**关联**: [[direction-e-archive-ga-crossgen-20260909]] (用户方向, 升主) / [[northstar-direction-b-verdict-20260909]] (floor 主因判断被修正) / [[direction-b-run-trajectory-20260909]] (峰值+0.319 来源) / `docs/phase2_issues.md`
