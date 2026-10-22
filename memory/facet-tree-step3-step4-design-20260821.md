---
name: facet-tree-step3-step4-design-20260821
description: Facet-Tree Stage 0 Step3(纯函数)/Step4(护栏+tree) 实现决策与验收事实
metadata:
  type: project
---

# Facet-Tree Stage 0 实现决策 (Step3/Step4, 2026-08-21)

进化对象 = Line A 全局 math skill (见 [[facet-tree-skill-evolution-20260819]])。Stage 0 go/no-go 唯一判据 = test 4-seed > 0.4654。

## Step 3 交付 (已完成, 单测通过)

`rl/train/verl1/skill_evolver_math.py` 新增:

- `facet_decompose(skill) -> (title, slots)` — 按 `^## ` 切段。真 S₀ 得 4 段
  (`Option Comparison`/`Theorem-Level Precision`/`Hypotheses`/`Final Answer`) + 题名 title。
- `facet_compose(title, slots, slot_order=None)` — 逆重组, 逐字节往返 = 无损。
- `_facet_roundtrip_assert(skill)` — md5 无损往返断言 (Step4 护栏共用)。
- `_self_test_facet()` — 纯函数自检 (零 API), 由 `--self_test_facet` 触发后 return。

**切段契约 (实测确认)**: `re.split(r'^(## .+)$', skill, MULTILINE)` →
`[title_block, h1, body1, h2, body2, ...]`, body 首字符恒 `\n`, 无嵌套 `##`, 无 frontmatter。
交替 (header=奇数位, body=偶数位) → `''.join` 恒等于原文。

**单测结果**: S₀=1017 chars, 4 sections, roundtrip md5=68571cb2a6, non-target slot freeze OK。
(注: 1017 vs plan 的 1018 属 strip() 后差 1 尾字符, 非 bug。)

## Step 4 设计决策 (已实现, 2026-08-21)

greenfield 两个: `archive_tree`(树) + `recombine_facets` 2-parent 交叉。主循环树遍历作为
`--facet_tree` 门控的**并行路径** (非改动既有 single-lineage `for t`, flag 关字节复现 0.462)。

**新增符号** (全在 `skill_evolver_math.py`):
- `FACET_EDIT_SYSTEM` / `FACET_EDIT_PROMPT` — 单 slot patcher prompt (只编辑 target slot)
- `facet_edit_guard(parent_slots, target_key, child_text)` — 护栏: 剥标题; 含其他 ## 段 → 拒绝
  (物理防塌缩); 非目标 slot md5 不变否则拒绝。返回 (slots, ok)
- `recombine_facets(a,b,a_key,b_key)` — 2-parent 交叉占位 (Stage 0 未用, 终局 C 组开)
- `_ArchiveNode` (slots 存到节点含 `__slots__`) — 树节点
- `build_archive_tree(skill, axes)` — 建根
- `_run_facet_tree(...)` — Stage 0 树遍历主循环: 每代遍历活跃叶, 各按 facet_label 单 slot
  变异 + 护栏 + eval_on/probe 评估 + MAP-Elites 归档 (输家不删 stepping stone)
- 新增 flags: `--facet_tree`/`--facet_axes`/`--freeze_facet`/`--failure_modes`/`--no_emergence`

**主循环 dispatch**: `total_iters` 后加 `if args.facet_tree: _run_facet_tree(...); return`。
单 lineage 分支 (原 for t) 完全未动。

**护栏 bug 修复 (实写时发现)**: 初版 `facet_edit_guard` fallback 把整个 child skill 塞进
slot body (错), 且 `title` 变量 shadowing。改为: child_text 含其他 ## 段 → ok=False 拒绝;
只含 target 段 → 取该段正文; 纯正文 → 直接用作正文。单测 (e) 覆盖接受/拒绝两条路径。

**单测扩展 (e)**: `--self_test_facet` 现覆盖 (a) 4 段 (b) 无损往返 (c) 单 slot 替换非目标
md5 不变 (d) 空/无 ## 退化 (e) 护栏接受/拒绝。全部通过。

**验收事实**: 语法 clean; `--self_test_facet` 绿 (roundtrip md5=68571cb2a6);
`--help` 5 新 flag 全部注册; 静态符号检查全齐。

## Step 5 smoke test (2026-08-21 启动)

用户拍板「先 smoke test (--iters 1)」。后台启动 (ID b40v9p9x0) 命令:
```
--init_skill <真S0> --s0 <真S0> --iters 1 --probe_split --probe_ratio 0.5 --probe_seed 42
--eval_k 3 --facet_tree --freeze_facet "Final Answer" --no_emergence
```
预期 ~1.1h 下限 API。验证目标: 树遍历 + 护栏 + archive_tree 归档在真实 LLM 调用下无运行时错误。
stdout 全缓冲(非tty), 完成前输出文件为空属正常; baseline precompute_s0 阶段 CPU 低(等 API)属正常。
smoke 通过后 → 正式 2 代产 go/no-go 结论 (test 4-seed > 0.4654)。

## 关键事实 (Step1 核对产出, 供实现引用)

- theorem_type 值域 test(n=106)=13 类, unknown=3(脏); val(n=53) unknown=5。
- 一题多类 96/106。facet 轴不能按单题单类硬切。
- `--init_skill`(L633 起点, 默认 V2) vs `--s0`(L634 delta 基线, 默认 S0)。step5 须双同传真 S₀。
- fitness 链路 = 纯 EM pass_rate, 不调 qwen_235B/CModelJudge/c_interface → C-judge=pro 对 Stage0 是 no-op。