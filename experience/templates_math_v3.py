"""数学失败模式蒸馏 v3 模板 -- 形态/内容分层 + S₀ 覆盖检查 (算法优化尝试, 2026-08-05).

== 诊断动机 (来自 LiveMath vs AIME skill 内容对比) ==
- LiveMath (定理级 MCQ) 的失败模式全是 *形态级* (option-vs-theorem 结构关系:
  等价双向/量词域/强度排序/等号vs包含/meta-statement 决策), 跨题高复用 -> global 泛化 (+16.1pp);
  且这些结构程序 S₀ 未全覆盖 -> 蒸馏真增了可泛化内容.
- AIME (自由作答 0-999) 的失败模式多为 *内容级* (主题特异战术: Power of a Point/sinusoidal/
  median), 跨题低复用 -> 不泛化 (CI 正确拒); ~4 条形态级的 (maximal vs maximum/完备性/
  at-least vs exactly) 已在 S₀ 的 Approach 节 -> 蒸馏没产出 S₀ 还没有的可泛化内容.

== v3 改进 ==
1. 蒸馏时显式分层 FORM-LEVEL (跨题型/跨主题的结构诊断程序) vs CONTENT-LEVEL (主题特异战术).
   - FORM-LEVEL -> 进 global skill;
   - CONTENT-LEVEL -> 尝试抽象为形态级; 抽象不出则拒绝 (留待 per-topic 库).
2. 加 S₀-COVERAGE 检查: 已被 S₀ 覆盖的模式直接拒绝 (避免 v2 在 AIME 上产出 S₀ 已有内容).
3. 去 MCQ 硬编码: 适用于任意数学题 (MCQ / 自由作答), 由调用方传 task_form.

== 诚实边界 ==
v3 filter 单独不足以让 AIME 涨分 (AIME 形态级已被 S₀ 覆盖, 剩余内容级不可泛化).
真正能在 AIME 上取得增益的是 per-topic skill 库 (按主题聚类, 每簇同质->簇内可泛化).
v3 模板是 per-topic 方案的蒸馏基础, 也独立用作 global 路径的防稀释 correctness 改进.

== 集成点 ==
tf_global_v2_math.py / skill_evolver_math.py 当前 `from experience.templates_math import ...`;
切到 v3: 改 import 为 templates_math_v3, 并把 S₀ 文本传给 distill (做 S₀-coverage 检查).
"""

# 占位: task_form 由调用方传, 如 "theorem-level multiple-choice" / "AIME free-response integer 0-999"
# 占位: s0_excerpt 由调用方传 (S₀ 前 ~1500 字符), 供 S₀-coverage 检查

FAILURE_MODE_TEMPLATE_V3 = """You are an expert in mathematical reasoning diagnosis for {task_form} problems.

For the SAME problem, {num_rollouts} different system prompts (skills) were tested on a strong LLM (the B model). Below are the results, including what the B model actually predicted vs the correct answer, and an excerpt of its reasoning.

## Theorem / ground-truth (if provided)
{theorem}

## Problem
{problem}

## Rollouts (each used a different skill; sorted by delta vs S_0 baseline, best first)
{rollout_summaries}

## Task: Extract GENERALIZABLE failure modes, classified by LEVEL

Analyze WHY the B model got low-scoring rollouts wrong and what helped on high-scoring ones.
Focus on the MATHEMATICAL CONTENT of the error, not prompt wording.

For EACH failure mode, use this STRUCTURED format:

```
[F#]. <Failure-mode name>

**Level**: FORM-LEVEL | CONTENT-LEVEL
   - FORM-LEVEL = a STRUCTURAL diagnostic procedure reusable across DIFFERENT topics/problem-archetypes
     (e.g. "restate whether the question asks for a count / value / sum / max", "check enumeration cases
     are disjoint and exhaustive", "test both directions of a claimed equivalence").
   - CONTENT-LEVEL = a tactic specific to ONE topic/archetype (e.g. "apply Power of a Point",
     "maximize A cos θ + B sin θ"). CONTENT-LEVEL modes do NOT generalize to a global skill.
**Symptom**: What does the B model tend to wrongly do / conflate on this kind of problem?
**Root cause**: Which mathematical criterion was mis-applied or ignored.
**Diagnostic instruction**: A concrete, reusable action to ADD to a system prompt.
**Abstraction** (only if CONTENT-LEVEL): Can this be abstracted into a FORM-LEVEL procedure that applies
   across topics? If yes, give the abstracted form (and relabel Level=FORM-LEVEL). If no, mark "discard for global".
```

## CRITICAL: Generalization rules (v3, form/content-aware)

KEEP a candidate iff it is FORM-LEVEL (or successfully abstracted to FORM-LEVEL) AND NOT already covered by S_0.
- A mode is "already covered by S_0" if S_0's existing text already tells the model to perform substantially
  the same action. The S_0 excerpt is provided below for this check.
- DO NOT emit a mode that merely restates what S_0 already says.

REJECT a candidate if it is:
- CONTENT-LEVEL and cannot be abstracted to FORM-LEVEL (route to a per-topic library, not the global skill).
- Already covered by S_0 (see S_0 excerpt).
- Pure PROMPT WORDING technique ("use bold headers", "output one label") - already in S_0.
- The SPECIFIC ANSWER to this problem (any label / constant / this problem's particular data).
- A restatement of the theorem without a diagnostic action.

## S_0 excerpt (for coverage check)
{s0_excerpt}

Extract up to {num_experiences} FORM-LEVEL failure modes not covered by S_0. If the rollouts reveal no clear
FORM-LEVEL pattern (e.g. all errors are content-level tactics, or pure computation slips, or all correct),
output an empty <Experiences></Experiences> block rather than fabricating content-level advice or restating S_0.

Wrap ALL extracted failure modes in <Experiences>...</Experiences> tags.

Your analysis and failure modes:"""


FAILURE_MERGE_TEMPLATE_V3 = """You are managing a pool of generalizable failure modes for {task_form} reasoning.

Below are candidate failure modes (already classified FORM-LEVEL / CONTENT-LEVEL) from MULTIPLE problems.
Some may be:
- **Form-level, not in S_0** - a cross-topic structural diagnostic procedure (KEEP, possibly enrich)
- **Redundant** - same structural confusion phrased differently (MERGE into the strongest version)
- **Content-level** - topic-specific tactic (attempt ABSTRACTION to form-level; if impossible, REJECT for global)
- **Already covered by S_0** (REJECT - check against S_0 excerpt)
- **Problem-specific / prompt-wording** (REJECT)

## Candidate failure modes (from {num_problems} problems)
{candidate_experiences}

## Current Failure-Mode Pool (for dedup reference)
{current_pool}

## S_0 excerpt (for coverage check)
{s0_excerpt}

## Task: Merge, abstract, deduplicate - keep only FORM-LEVEL not in S_0

1. **REJECT** content-level (non-abstractable), S_0-covered, problem-specific, prompt-wording entries.
2. **ABSTRACT**: for each content-level entry, try to rewrite as a FORM-LEVEL procedure applying across topics;
   keep only if abstraction succeeds and the result is not in S_0.
3. **MERGE** semantically identical form-level modes into the strongest single entry.
4. **DEEPEN**: ensure each has concrete Symptom + Root cause + Diagnostic instruction.
5. Keep at most {max_experiences} FORM-LEVEL modes, ordered by cross-problem frequency (most common first).

For each kept mode use:
```
[F#]. <Failure-mode name>
**Level**: FORM-LEVEL
**Symptom**: <...>
**Root cause**: <...>
**Diagnostic instruction**: <a reusable cross-topic action>
```

Output format:
<Experiences>
[F0]. ...
...
</Experiences>

Your merged, abstracted failure modes:"""


COMPOSE_APPEND_TEMPLATE_V3 = """You are improving a system prompt (S_0) for a strong mathematical-reasoning LLM ({task_form}).

The base skill S_0 below is ALREADY GOOD and verified. Do NOT rewrite or weaken it.
Your ONLY job is to APPEND a new section at the very end, titled "## Common Failure Modes", that turns the
distilled FORM-LEVEL failure modes into a concise, reusable DIAGNOSTIC CHECKLIST.

Rules:
1. Do NOT modify, rephrase, or delete any existing part of S_0. Output S_0 verbatim first.
2. After S_0, append exactly one new section "## Common Failure Modes".
3. List each FORM-LEVEL failure mode as a SHORT, ACTIONABLE checklist bullet (1-2 sentences). Each must be a
   cross-topic structural diagnostic action, NOT a topic-specific tactic and NOT a restatement of one problem.
4. DROP any failure mode that is content-level, problem-specific, prompt-wording, or ALREADY COVERED by S_0.
5. If no failure modes survive filtering, output S_0 UNCHANGED with no appended section (do not pad).
6. Output ONLY the final system prompt text (S_0 + optional appended section). No meta commentary.

## INITIAL SYSTEM PROMPT (S_0) - keep verbatim
{s0}

## DISTILLED FORM-LEVEL FAILURE MODES (filter and convert to checklist bullets)
{experiences}

## YOUR OUTPUT
"""
