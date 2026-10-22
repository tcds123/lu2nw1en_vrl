---
name: aime-v2-evolver-built-20260930
description: Track A — skill_evolver_aime_v2.py built + self_test_facet gate green (zero API); ready for smoke
metadata:
  type: project
---

**状态**: skill_evolver_aime_v2.py 构建完成，`--self_test_facet` 全绿 (零 API)。2026-09-30。

**起点**: 从 4214 行 math evolver (skill_evolver_math.py) 复制，重写所有 MCQ 耦合层为 AIME 数值内容。

**S₀ (4-facet, 可编辑)**: `/data/zhuldz/lunwen/data/skill_aime_initial_v2.md`
- 标题 "# Competition Mathematics Reasoning Skill"
- 4 个 `##` 段: Quantity Parsing / Case Enumeration / Computation & Verification / Final Answer
- Final Answer 段含 `\boxed{N}` 答案格式规则 (freeze 目标)
- `--facet_axes` 默认 = 这 4 段; `--freeze_facet` 默认 = "Final Answer"

**重写的 MCQ→AIME 层** (skill_evolver_aime_v2.py):
- `FAILURE_FOCI` (L111-120): 8 类数值失败 (misread-quantity / off-by-one / case-enum / inclusion-exclusion / arithmetic / answer-form / number-theory / equation-setup)
- `classify_failure_type` / `classify_item_dim`: 读 `problem`+`answer` (非 question/theorem/choices), 关键词阶梯 dim0(how many)→1(between/inclusive)→2(cases)→3(inclusion)→4(compute)→6(divisor/mod)→7(equation)→fallback 5(answer-form)
- `_TYPE_COVERAGE_SIG`: 8 类的 sig 关键词 (donor 覆盖检测)
- `_BIDIRECTIONAL_KW`, `select_strengthen_target`, `_donor_coverage_types`, `select_donors_directed`, `_assemble_multi_grafts`: 全部 AIME 主题
- `--facet_axes`/`--freeze_facet` 默认改 AIME 4-facet
- `_self_test_facet` 全部 fixture (块 l/m/n/o/q/q7/r/fix-①): MCQ body (iff/uniqueness/domain-interior) → AIME 数值 (count vs value / boundary inclusive / divisor modulo)

**Schema-agnostic 核心 (verbatim 复用, 未改)**: facet_decompose/compose, facet_edit_guard, _ArchiveNode, archive, recombine_facets, _xo_section_guard, operator-counting, eval_on, bootstrap_ci_lower, per_item_counts, beta_binomial_lcb, load_s0_cache_with_manifest, check_skill_leakage, spike_peak_hit, admission_filter, git_diff_similarity.

**5-峰合并 (AIME 对齐)**: 0=quantity 1=boundary 2=cases SPARSE(3+4+6+7) FORM(5). `_DIM_TO_PEAK` 映射后 `spike_peak_hit` 返回 str(peak_key).

**ON/OFF 定义 (确认, 无 _run_flat)**: facet_tree GA 总跑; OFF = 仅 `--facet_tree --expand` (无 pool_filter); ON = `--facet_tree --expand --pool_filter`.

**验证**: `python skill_evolver_aime_v2.py --self_test_facet` → "✅ facet self-test passed: S₀=2408 chars, 4 sections, roundtrip md5=f4837ca80f, ...". 全 assert 过。

**后端**: qwen_chat / DeepSeek-V3.2 (env QWEN_CHAT_* 模块加载时读, tf_global_v2_aime.py). EVAL_TEMP=0.3 k-voting. train=AIME24 (30 题), held-out=AIME25 (30 题).

**冒烟 (✅ 通过, 2026-09-30)**: `/tmp/smoke_aime_trackA.py` — `load_aime_split(SPLIT,18,42)` → train=18/val=12 ✅; `run_one(item, S0, k=1)` 3 题, 148.3s (~50s/题)。后端链路 **实测打通**: DeepSeek-V3.2 → `evaluate_aime` (整数提取+exact-match) → `pass_rate`。
- `aime24:79` pred 321 = correct 321 ✅
- `aime24:74` pred 480 = correct 480 ✅
- `aime24:70` pred 31528 ≠ correct 104 ❌ (agent_ok=True, 非 API 错误; 答案 31528 > 999 = foci[5] answer-form 失败类型, 正是 S₀ "Final Answer" 段要治的, 有优化空间)
- env 自带 (skill_evolver_aime_v2.py L34-41 `os.environ.setdefault` 全部 QWEN_CHAT_*), 无需外部 export。EVAL_TEMP=0.3 k-voting 生效。

**待办**: `--iters 1 --train_n 3 --K 1 --eval_k 1 --facet_tree --expand --pool_filter` 跑通一轮 evolver (低 API); 然后昂贵全量 (AIME24+25, k=3, multi-seed) + skillopt 对比。执行序列见 [[aime-v2-build-launch-20260930]]。问题记录见 docs/phase2_issues.md。
