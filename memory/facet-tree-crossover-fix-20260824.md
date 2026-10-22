---
name: facet-tree-crossover-fix-20260824
description: 交叉准入被 guard 拒的根因(prompt↔guard矛盾)+ 修复 + iters=2 smoke 验证 crossover_fired=1
metadata:
  type: project
---

# Facet-Tree 交叉准入修复 (defect #2 终结) — 2026-08-24

## 问题
`recombine_facets` (skill_evolver_math.py) 在 153302 smoke 的 gen2 交叉**触发但准入被拒**: `crossover_fired: 0`, `reason: "guard/cohere-fail"`。函数被调用了 (bug#2 字面已修) 但产出被自身 guard 系统性拒绝 = 死算子。

## 根因 (静态确认)
`CROSSOVER_COHERE_PROMPT` 与 L850 structure guard **直接矛盾**:
- prompt L811 (旧) 指令 LLM "把冗余/重叠的段落合并成最强版本" → 诱导跨段合并 → 丢/改 `## ` 段头 → **FAIL-③** (`set(child_slots.keys()) != set(parent_a_slots.keys())`)。
- L850 guard 要求 4 个 `## ` 段标题集合严格不变。
- 本 run 特定触发: gen1 两交叉父本 (Hypotheses 叶 + Option Comparison 叶) 各自独立演化出近乎相同的 "stronger result can be proven" meta-option 指引, 移植后组装出跨两段重复指引 = prompt 要合并的典型态 → LLM 合并即触发 FAIL-③。次可能 = FAIL-④ (LLM 复述 Final Answer 改写措辞 → 冻结 md5 变)。

## 修复 (两步, robust 覆盖 ③④)
1. **重写 `CROSSOVER_COHERE_PROMPT`**: 删"合并冗余段"指令 (它诱导丢段头); 改为只消解**真冲突** (两指引推向相反答案时留更精确者); 跨段**重复但并行**指引保留在各自段不合并 (facet-tree 本就需要每 facet 独立可演化槽); 逐字列出 4 个 `## ` 段标题要求保留 + `## Final Answer` 逐字不动。
2. **`recombine_facets` 冻结槽强制覆写**: coherency 后 `child_slots[freeze] = parent_a_slots[freeze]` 直接覆盖, 构造性消除 FAIL-④。
- **关键判断: 改 prompt 对齐 guard, 而非放宽 guard** —— 放宽 L850 (允许丢段) 会瓦解 facet-tree 语义 (合并掉一个 facet 段 = 丢一个 facet 专家槽, 该分支再无法独立演化)。

## 验证 (双重通过)
1. **离线 repro** (`/tmp/crossover_repro.py`, import 修复后的逻辑, 单次 LLM 调用): 喂 gen1 两叶 slots → `[SUCCESS]`, child slot set==parent, freeze md5 保留, 与父本不同。FAIL-③④ 均不触发。
2. **修复后 iters=2 全 on smoke** (`run_20260824_183940`, exit 0, PID 9904): 入场券 5 判据全过:
   - **`crossover_fired: 1`** ✅ (crossover 字段 `admitted: True`, delta=+0.189, ci=+0.117)
   - 每分支 PROMOTE/KEEP+🔎 ✅; diag disjoint [12,12,11] ✅; 无崩 ✅; 全产物落盘 ✅
   - 终局: init delta=+0.006 → best delta=+0.194 (CI=+0.122), 1017→1235 chars, hash 68571cb2a6→2c33e2f5fc, archive_size=10 (6→10 含交叉子代)。
   - 交叉子代 delta=+0.189 < best +0.194 (gen1 Opt-Comp PROMOTE), 是有效次优解非噪声, 符合 (μ+λ) 选择。

## 状态
- defect #2 "recombine_facets 从未被调用" **彻底修复**: 既被调用又通过准入。
- Stage 0 入场券 smoke **全过** (4/5→5/5, 核心 (a) crossover_fired>0 转绿)。
- 下一步: test=106 多 seed 终评待人工执行 (命令见 evolver_log 末尾), 判据 = **闭合到 0.5464 判据线**
  (见 [[a2-power-analysis-20260824]] §2/§3; 0.4654 是 Stage 0 破天花板线, 已达成退役, seed 维度已从判据退役)。多代归因代数由收敛曲线定, 不提前锁。
