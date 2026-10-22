# 方向 B 实现完成: 多 parent crossover + 定向 donor (2026-09-08)

## 背景 (决策链)
方案1 (per-type strengthen) 从 theo_iter6 起步, 北星未越 (evolver 0.5440 vs skillopt 0.5605, CI95[−0.0597,+0.0259] 下界<0)。
根因: theo_iter6 是 warm-start (非 S₀), 已消费 ~+0.26 headroom, 留给进化超 skillopt 的空间≈0。
→ 按用户指令 "如果不行就换方向b", 转方向 B, 且**从 s0 起步** (FAIR 条件, 给完整 +0.265 headroom)。
详见 [[northstar-pertype-verdict-20260909]]。

## 方向 B 代码改动 (唯一文件: `rl/train/verl1/skill_evolver_math.py`)

### 4 个新函数 (插入 `recombine_facets` 之后, `select_section_donors` 之前)
1. **`_donor_coverage_types(body, counts)`** — 纯函数: donor section body 覆盖哪些「当前仍在失败」(counts[lbl]>0) 的失败类型。覆盖判定与 `_TYPE_COVERAGE_SIG` 同口径 (body 含签名关键词)。返回 set[label], 空 = 无定向价值。
2. **`select_donors_directed(archive, root_slots, freeze, counts, topk=2)`** — 定向 donor 选取: 按 section body 对当前失败分布 counts 的覆盖匹配度排序 (非纯 val_ci)。门控: val_ci>0, md5!=root, sec!=freeze, coverage>0。排序键: (覆盖类型数 desc, val_ci desc)。返回 list[(sec_key, _ArchiveNode)]。
3. **`_assemble_multi_grafts(base_slots, donors, freeze)`** — 纯函数零 API 可测: 多 donor 段嫁接组装。base 全段 + 各 donor sec_key 段覆盖/新增。门控: sec==freeze 跳过, 同 sec 只取第一个 (去重), donor 段 md5==base 同名段 (未变异) 跳过。返回 (grafted_slots, grafted_list)。
4. **`recombine_facets_multi(base_slots, donors, freeze, llm_stage, retries, allow_merge)`** — 多 parent facet 交叉: 3+ donor 不同 section 段同时嫁接进 base + 1 coherence pass (复用 CROSSOVER_COHERE_PROMPT + _xo_section_guard + freeze 回填守卫)。失败→回退纯组装 slots。返回 (new_slots, ok)。

### 2 个新 argparse flag (默认 off = 字节复现)
- `--xo_multiparent` (int, default 0): >0 = 多 parent crossover; 0 = 旧 2-parent 行为。
- `--xo_donor_directed` (bool, default False): True = 用 `select_donors_directed`; False = 用 `select_section_donors`。

### crossover 循环集成 (在旧 2-parent 分支之前)
`xo_multiparent >= 2` 且 `t>=2` 且 `len(archive)>=2` 时:
1. 聚合 `branch_failure_types[t]` (per-branch fcounts) → `_mp_counts` (crossover 点的 iter-stable 失败分布; 全 `counts` 在 L2364 之后才算, 故用 branch 聚合)。
2. `xo_donor_directed` → `select_donors_directed(archive, root_slots, freeze, _mp_counts, topk=xo_multiparent)`; 否则 `select_section_donors` 取 top-N by val_ci。
3. `recombine_facets_multi(best_node.slots, _mp_donors, freeze, ...)` → eval → admit (ci>0) → archive + best promotion + subset-rescue。
4. `xo_event` 非 None → 跳过旧 2-parent 分支 (互斥)。

### self-test (r) 系列 (零 API, 全绿)
- r1: `_donor_coverage_types` — sig kw 覆盖类型; counts<=0 排除。
- r2: `select_donors_directed` — ci>0+md5!=root+coverage>0 门控; coverage-desc 排序; val_ci<=0 排除; 空 counts→[]; 未变异 (md5==root) 排除 + 变异保留。
- r3: `_assemble_multi_grafts` — 多段嫁接 + freeze 存活 + 去重 + 空/未变异→不嫁接。

## 验证 (全绿)
- `--self_test_facet`: ✅ `S₀=1017 chars, 4 sections, roundtrip md5=68571cb2a6, ... 方向B OK`。
  - **md5=68571cb2a6 不变 = 字节复现未破坏** (新 flag 默认 off → 旧路径零影响)。
- `py_compile`: OK。
- `--help`: 显示 `--xo_multiparent` + `--xo_donor_directed` 含完整 help。
- 静态检查: 两条 crossover 路径都在 (`_xo_mp_n >= 2` + `xo_event is None`), 4 新函数已定义。

## 待跑 (下一步)
s0 起步 FAIR warm-start run (方向 B 机制生效测试):
- `--init_skill` = initial.md (s0, hash 68571cb2) — 非 theo_iter6。
- `--facet_tree` (必须, 否则走 flat path)。
- `--xo_multiparent 3 --xo_donor_directed --xo_allow_merge`。
- run 中即时判据: 日志出现 `🔀 xover-MP iter` = 多 parent crossover fire; `✅ XOVER-MP-BEST` = 夺 best。
- 完成后: 4-seed test=106 CI (复用 s0_68571cb2 + skillopt_cfae156d 缓存, 仅新评 evolver arm) → 北星判据。

## 教训
1. **counts 时序**: crossover 点没有单 `counts` 变量 (全 `counts` 在 L2364 之后算)。用 `branch_failure_types[t]` 聚合跨分支 fcounts 得 `_mp_counts` — 这是 crossover 点可用的 iter-stable 失败分布。
2. **零 API self-test**: `recombine_facets_multi` 调 `llm`, 不可在 self-test 直接测。抽出纯函数 `_assemble_multi_grafts` 测嫁接逻辑 (同 `_xo_section_guard`/`_archive_section_union` 模式)。
3. **字节复现**: 新 flag 默认 off + `xo_event is None` 守卫 → `xo_multiparent=0` 时旧 2-parent 路径字节不变 (md5 验证)。

**关联**: [[northstar-pertype-verdict-20260909]] (方案1 未越 → 转方向 B) / [[pertype-run-complete-20260909]] (方案1 run 链路) / [[northstar-test-baseline-20260825]] (theo_iter6 基线) / `docs/phase2_issues.md` (方向 B 问题记录)
