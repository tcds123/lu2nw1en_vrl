"""数学失败模式蒸馏 Prompt 模板 (TF-global v2).

与 experience/templates.py 的通用 prompt-写作技巧模板互补。关键区别:
- 通用模板 (GROUP_ADVANTAGE_TEMPLATE) 的 Generalization Check **拒绝**问题特定内容、只留 prompt 措辞技巧
  → 对定理级 MCQ 把有用数学信号删光, 留下 S₀ 已有的陈词滥调 (根因之一).
- 本模板反转: **拒绝纯 prompt 措辞技巧**, **保留可泛化的数学诊断程序** (锚定 quantifier/equality case/
  regularity/存在 vs 唯一 等数学判据), **拒绝单题答案**.

产出 failure-mode 形态经验, 供 compose 阶段以"追加节"方式注入 S₀ (不重写 S₀).
"""

# ============================================================
# 单题失败模式蒸馏: 从同一题的多个 rollout 提取"为什么 B 选错"
# ============================================================
FAILURE_MODE_TEMPLATE = """You are an expert in mathematical reasoning diagnosis for theorem-level multiple-choice questions (LiveMathematicianBench).

For the SAME problem, {num_rollouts} different system prompts (skills) were tested on a strong LLM (the B model). Below are the results, including what the B model actually predicted vs the correct answer, and an excerpt of its reasoning.

## Theorem (ground-truth statement)
{theorem}

## Problem
{problem}

## Choices
{choices}

## Rollouts (each used a different skill; sorted by delta vs S_0 baseline, best first)
{rollout_summaries}

## Task: Extract GENERALIZABLE MATHEMATICAL FAILURE MODES

Analyze WHY the B model picked wrong options on low-scoring rollouts and what helped on high-scoring ones.
Focus on the MATHEMATICAL CONTENT of the error, not on prompt wording.

For EACH failure mode you extract, use this STRUCTURED format:

```
[F#]. <Failure-mode name — a mathematical confusion, e.g. "Confuses existence with uniqueness">

**Symptom**: What kind of distractor does the B model tend to wrongly pick? What does it conflate?
**Root cause**: Which mathematical criterion was mis-applied or ignored? Anchor it to a concrete mathematical
   distinction (quantifier flip: 'for every' vs 'there exists'; dropped equality case; upgraded regularity;
   existence vs uniqueness; global vs local; necessary vs sufficient; scale/domain restriction).
**Diagnostic instruction**: A concrete, reusable checklist item to ADD to a system prompt that would catch this.
   Must be generalizable ACROSS problems of this type, not specific to this single problem.
   Phrase it as an action the model should perform ("For each option, check whether X holds...").
```

## CRITICAL: Generalization rules (REVERSED vs generic prompt-engineering templates)

KEEP a candidate if it is a REUSABLE MATHEMATICAL DIAGNOSTIC PROCEDURE that would help on a DIFFERENT theorem-level MCQ. Examples of KEEP:
- "For each option, identify whether it asserts mere existence of a solution or uniqueness; the theorem's hypothesis must match."
- "Check each option's equality case: a statement that drops a boundary/equality condition is a weakening, not the theorem."

REJECT a candidate if it is:
- Pure PROMPT WORDING technique ("use bold headers", "compare all options", "output a single label") — these belong to the base skill S_0 already.
- The SPECIFIC ANSWER to THIS problem (any reference to which label is correct, or to this problem's particular constants).
- A restatement of the theorem without a diagnostic action.

REJECT a candidate that mentions this problem's specific data/numbers as if they were general.

Extract up to {num_experiences} generalizable failure modes. If the rollouts reveal no clear mathematical failure pattern (e.g. all correct, or errors are pure computation slips unlikely to recur), output an empty <Experiences></Experiences> block rather than fabricating generic advice.

Wrap ALL extracted failure modes in <Experiences>...</Experiences> tags.

Your analysis and failure modes:"""


# ============================================================
# 跨题合并: 把多题蒸馏的 failure modes 去重、合并、泛化
# ============================================================
FAILURE_MERGE_TEMPLATE = """You are managing a pool of generalizable mathematical failure modes for theorem-level MCQ reasoning.

Below are candidate failure modes extracted from analyzing rollouts of MULTIPLE DIFFERENT problems.
Some may be:
- **General** — a reusable mathematical diagnostic procedure (KEEP, possibly enrich)
- **Redundant** — same mathematical confusion phrased differently (MERGE into the strongest version)
- **Problem-specific** — only relevant to one problem's particular theorem/numbers (REJECT)
- **Prompt-wording** — about prompt style, not mathematical content (REJECT — these are already in the base skill S_0)

## Candidate failure modes (from {num_problems} problems)
{candidate_experiences}

## Current Failure-Mode Pool (for dedup reference)
{current_pool}

## Task: Merge, generalize, and deduplicate

1. **REJECT** problem-specific and prompt-wording entries.
2. **MERGE** semantically identical failure modes (e.g. several "existence vs uniqueness" variants → one strongest entry combining their diagnostic instructions).
3. **GENERALIZE**: if an entry is anchored to one problem's specifics, rewrite its diagnostic instruction so it applies across problems of that theorem type (quantifier/equality/regularity distinctions).
4. **DEEPEN** thin entries: ensure each has a concrete Symptom + Root cause + Diagnostic instruction.
5. Keep at most {max_experiences} failure modes, ordered by how frequently the confusion appeared across problems (most common first).

For each kept failure mode use:
```
[F#]. <Failure-mode name>

**Symptom**: <which distractor type B tends to wrongly pick / what it conflates>
**Root cause**: <the mis-applied mathematical criterion, anchored to quantifier/equality/regularity/existence-uniqueness>
**Diagnostic instruction**: <a reusable checklist action to add to a system prompt>
```

Output format:
<Experiences>
[F0]. ...
**Symptom**: ...
**Root cause**: ...
**Diagnostic instruction**: ...

[F1]. ...
...
</Experiences>

Your merged, generalized failure modes:"""


# ============================================================
# Global skill 合成: 以"追加节"方式把 failure modes 注入 S₀ (最小侵入)
# ============================================================
COMPOSE_APPEND_TEMPLATE = """You are improving a system prompt (S_0) for a strong mathematical-reasoning LLM.

The base skill S_0 below is ALREADY GOOD and is verified to work. Do NOT rewrite or weaken it.
Your ONLY job is to APPEND a new section at the very end, titled "## Common Failure Modes", that turns the distilled failure modes into a concise, reusable DIAGNOSTIC CHECKLIST the model should run through before committing to an answer.

Rules:
1. Do NOT modify, rephrase, or delete any existing part of S_0. Output S_0 verbatim first.
2. After S_0, append exactly one new section "## Common Failure Modes".
3. In that section, list each failure mode as a SHORT, ACTIONABLE checklist bullet (1-2 sentences each), grouped if natural. Each bullet must be a generalizable mathematical diagnostic action, not a restatement of one problem.
4. Drop any failure mode that is problem-specific, prompt-wording, or already covered by S_0.
5. If no failure modes survive filtering, output S_0 UNCHANGED with no appended section (do not pad).
6. Output ONLY the final system prompt text (S_0 + optional appended section). No meta commentary.

## INITIAL SYSTEM PROMPT (S_0) — keep verbatim
{s0}

## DISTILLED FAILURE MODES (filter and convert to checklist bullets)
{experiences}

## YOUR OUTPUT
"""
