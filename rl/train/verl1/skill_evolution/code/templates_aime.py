"""AIME 数值题失败模式蒸馏 Prompt 模板 (TF-global v2-aime).

与 templates_math.py (定理级 MCQ) 的关键区别:
- MCQ 模板分析"为什么 B 选错选项 (distractor)", 锚定 quantifier/equality/regularity 等数学判据.
- AIME 模板分析"为什么 B 算错数值", 锚定计算/setup/case/边界/答案形式等数值题特有失败模式.
- 共同保留: 反转泛化规则 (拒绝纯 prompt 措辞, 保留可泛化数学诊断程序; 拒绝单题答案/常数).

产出 failure-mode 形态经验, 供 compose 阶段以"追加节"方式注入 S₀(AIME) (不重写 S₀).
"""

# ============================================================
# 单题失败模式蒸馏: 从同一题的多个 rollout 提取"为什么 B 算错"
# ============================================================
FAILURE_MODE_TEMPLATE_AIME = """You are an expert in mathematical reasoning diagnosis for AIME-style competition problems (numerical answers, integers 0-999).

For the SAME problem, {num_rollouts} different system prompts (skills) were tested on a strong LLM (the B model). Below are the results, including what the B model actually computed vs the correct answer, and an excerpt of its reasoning.

## Problem
{problem}

## Correct Answer
{correct_answer}

## Rollouts (each used a different skill; sorted by delta vs S_0 baseline, best first)
{rollout_summaries}

## Task: Extract GENERALIZABLE MATHEMATICAL FAILURE MODES

Analyze WHY the B model arrived at wrong numerical answers on low-scoring rollouts and what helped on high-scoring ones.
Focus on the MATHEMATICAL CONTENT of the error, not on prompt wording.

For EACH failure mode you extract, use this STRUCTURED format:

```
[F#]. <Failure-mode name - a mathematical/computational confusion, e.g. "Miscounts overlapping cases in inclusion-exclusion">

**Symptom**: What kind of wrong answer does the B model tend to produce? What does it conflate or drop?
**Root cause**: Which step broke - case enumeration, equation setup, arithmetic, boundary handling, or answer-form conversion? Anchor it to a concrete, reusable mathematical distinction
   (count vs value vs sum vs max; inclusive vs exclusive; missing case; overcounting; unreduced fraction; wrong residue class; sign/orientation).
**Diagnostic instruction**: A concrete, reusable checklist item to ADD to a system prompt that would catch this.
   Must be generalizable ACROSS competition problems of this type, not specific to this single problem.
   Phrase it as an action the model should perform ("Before counting, verify cases are disjoint by ...").
```

## CRITICAL: Generalization rules (REVERSED vs generic prompt-engineering templates)

KEEP a candidate if it is a REUSABLE MATHEMATICAL DIAGNOSTIC PROCEDURE that would help on a DIFFERENT competition problem. Examples of KEEP:
- "For 'number of' questions, verify the count is over distinct objects - check whether two listed cases share an element (inclusion-exclusion)."
- "For residue/remainder problems, enumerate all residue classes in the bounded range, not just the obvious one."
- "For 'sum of all' vs 'how many', re-read the question and confirm which quantity is requested before computing."

REJECT a candidate if it is:
- Pure PROMPT WORDING technique ("show your work", "be careful", "use clear steps") - these belong to the base skill S_0 already.
- The SPECIFIC ANSWER to THIS problem (any reference to this problem's particular numbers or the final integer).
- A restatement of the problem without a diagnostic action.

REJECT a candidate that mentions this problem's specific data/numbers as if they were general.

Extract up to {num_experiences} generalizable failure modes. If the rollouts reveal no clear mathematical failure pattern (e.g. all correct, or errors are pure one-off arithmetic slips unlikely to recur), output an empty <Experiences></Experiences> block rather than fabricating generic advice.

Wrap ALL extracted failure modes in <Experiences>...</Experiences> tags.

Your analysis and failure modes:"""


# ============================================================
# 跨题合并: 把多题蒸馏的 failure modes 去重、合并、泛化
# ============================================================
FAILURE_MERGE_TEMPLATE_AIME = """You are managing a pool of generalizable mathematical failure modes for AIME-style numerical competition reasoning.

Below are candidate failure modes extracted from analyzing rollouts of MULTIPLE DIFFERENT problems.
Some may be:
- **General** - a reusable mathematical diagnostic procedure (KEEP, possibly enrich)
- **Redundant** - same mathematical confusion phrased differently (MERGE into the strongest version)
- **Problem-specific** - only relevant to one problem's particular numbers (REJECT)
- **Prompt-wording** - about prompt style, not mathematical content (REJECT - these are already in the base skill S_0)

## Candidate failure modes (from {num_problems} problems)
{candidate_experiences}

## Current Failure-Mode Pool (for dedup reference)
{current_pool}

## Task: Merge, generalize, and deduplicate

1. **REJECT** problem-specific and prompt-wording entries.
2. **MERGE** semantically identical failure modes (e.g. several "off-by-one in counting" variants -> one strongest entry combining their diagnostic instructions).
3. **GENERALIZE**: if an entry is anchored to one problem's specifics, rewrite its diagnostic instruction so it applies across competition problems of that type (counting/setup/boundary/residue distinctions).
4. **DEEPEN** thin entries: ensure each has a concrete Symptom + Root cause + Diagnostic instruction.
5. Keep at most {max_experiences} failure modes, ordered by how frequently the confusion appeared across problems (most common first).

For each kept failure mode use:
```
[F#]. <Failure-mode name>

**Symptom**: <which wrong-answer type B tends to produce / what it conflates or drops>
**Root cause**: <the broken step, anchored to count-vs-value / case-completeness / boundary / residue / answer-form>
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
# Global skill 合成: 以"追加节"方式把 failure modes 注入 S₀(AIME) (最小侵入)
# ============================================================
COMPOSE_APPEND_TEMPLATE_AIME = """You are improving a system prompt (S_0) for a strong mathematical-reasoning LLM solving AIME-style competition problems.

The base skill S_0 below is ALREADY GOOD and is verified to work. Do NOT rewrite or weaken it.
Your ONLY job is to APPEND a new section at the very end, titled "## Common Failure Modes", that turns the distilled failure modes into a concise, reusable DIAGNOSTIC CHECKLIST the model should run through before committing to an answer.

Rules:
1. Do NOT modify, rephrase, or delete any existing part of S_0. Output S_0 verbatim first.
2. After S_0, append exactly one new section "## Common Failure Modes".
3. In that section, list each failure mode as a SHORT, ACTIONABLE checklist bullet (1-2 sentences each), grouped if natural. Each bullet must be a generalizable mathematical diagnostic action, not a restatement of one problem.
4. Drop any failure mode that is problem-specific, prompt-wording, or already covered by S_0.
5. If no failure modes survive filtering, output S_0 UNCHANGED with no appended section (do not pad).
6. Output ONLY the final system prompt text (S_0 + optional appended section). No meta commentary.

## INITIAL SYSTEM PROMPT (S_0) - keep verbatim
{s0}

## DISTILLED FAILURE MODES (filter and convert to checklist bullets)
{experiences}

## YOUR OUTPUT
"""
