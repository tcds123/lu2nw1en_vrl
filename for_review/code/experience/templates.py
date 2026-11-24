"""
经验蒸馏 Prompt 模板
参考 Trace2Skill (arXiv:2603.25158) 方法论 — 从广泛轨迹蒸馏结构化综合指南

v4: 修复 Trace2Skill 方法论 Gap
  G3: LLM 驱动语义泛化，替代正则关键词过滤
  G4: 结构化 skill (WHEN/WHAT/HOW/WHY)，替代一句话摘要
  G5: 增强轨迹上下文（含单测结果、函数签名等代码结构化信息）
"""

# ============================================================
# Step 1: 单轨迹总结 (v4: 增强上下文 — 代码结构 + 单测细节)
# ============================================================
SINGLE_ROLLOUT_SUMMARY = """You are an expert in analyzing system prompt generation for code generation tasks.

Below is a single rollout trajectory from a prompt optimization system:
- A model generates a SYSTEM PROMPT
- B model (code generator) uses that prompt to write CODE
- The code is evaluated against UNIT TESTS (pass rate) and a JUDGE model (quality score)

Please analyze this trajectory in detail:

<trajectory>
System Prompt (A model output):
{system_prompt}

Generated Code (B model output):
{code}

Unit Test Pass Rate: {pass_rate:.1%}
Judge Score: {judge_score:.1f}

Ground Truth (Reference Solution):
{ground_truth}

Original Problem:
{problem}
</trajectory>

## Analysis Required

Please provide a structured analysis:

1. **Code Quality Assessment**: What did the generated code do well? What bugs or issues does it have?
   - Check: function signature correctness, edge case handling, algorithmic correctness, code style
   - If pass_rate < 100%, identify which test cases likely failed and why

2. **Prompt Effectiveness**: How did the system prompt influence the code quality?
   - What specific instructions in the prompt were HELPFUL (e.g., clear type constraints, explicit edge case instructions)?
   - What was MISSING from the prompt that contributed to failures?
   - Was the prompt too vague, too restrictive, or well-balanced?

3. **General Principle**: Extract ONE general principle about prompt writing technique that this trajectory illustrates.
   - Label: [General] or [Specific] — be honest about whether this applies to code generation tasks broadly.
   - A [General] principle should apply to string processing, graph algorithms, class design, and API calls alike.

Your analysis:"""


# ============================================================
# Step 2: 组内优势对比 — 从同一问题的多个 rollout 中提取经验 (v4: 结构化输出)
# ============================================================
GROUP_ADVANTAGE_TEMPLATE = """You are an expert in prompt engineering for code generation.

For the SAME coding problem, {num_rollouts} different system prompts were generated and tested.
Below are the results, ranked from BEST to WORST by overall score (combining unit test pass rate + code quality):

## Problem
{problem}

## Rollouts (sorted by quality)
{rollout_summaries}

## Learning Objective
{learning_objective}

## Task: Extract Structured Skills

Analyze the differences between HIGH-scoring and LOW-scoring prompts.
What specific prompt-writing techniques caused the score difference?

For EACH skill you extract, provide a STRUCTURED format (not a one-liner):

```
[G#]. <Title (5-10 words summarizing the tactic)>

**When to apply**: <What task characteristics trigger this skill? What signals indicate it's needed?>
**What to do**: <The specific technique — be concrete and actionable, 2-4 sentences>
**Example**: <A concrete example of what the system prompt should say or avoid saying>
**Why it works**: <What failure mode does this prevent? What does the B model tend to get wrong without this instruction?>
```

## CRITICAL: Generalization Check
For EACH skill you propose, apply this test:
- Would this skill be useful for a DIFFERENT type of coding problem (e.g., string processing vs. graph algorithms vs. class design)?
- If the skill mentions specific data structures, algorithm names, or this problem's domain — REJECT it as overfitting.
- Only keep skills about PROMPT WRITING TECHNIQUE (structure, clarity, constraint specification, error handling, output format), NOT about solving specific problems.

A good skill looks like:
```
[G0]. Explicit Function Signature Requirements

**When to apply**: When the task involves implementing a specific function whose signature (parameter names, types, return type) is given in the problem description.
**What to do**: In the system prompt, explicitly instruct the B model to match the exact function signature including parameter names, types, and return annotation. List the required signature verbatim as a constraint.
**Example**: "You MUST implement the function exactly as specified: `def process_data(items: List[int], threshold: float) -> Dict[str, List[int]]`. Do not rename parameters or change types."
**Why it works**: Without explicit signature constraints, code generation models often rename parameters (e.g., `items` → `arr`) or omit type annotations, causing unit tests that use keyword arguments or type checking to fail.
```

A bad (overfitted) skill looks like:
```
[G#]. Use binary search on the sorted array  ← Problem-specific, not general!
```

Extract exactly {num_experiences} structured skills. Wrap ALL in <Experiences>...</Experiences> tags.

Your analysis and structured skills:"""


# ============================================================
# Step 3: 跨题合并去重 — 混合模式核心 (v4: 结构化 + 深化)
# ============================================================
CROSS_PROBLEM_MERGE_TEMPLATE = """You are managing a pool of prompt engineering best practices for code generation.

Below are candidate skills extracted from analyzing rollouts of MULTIPLE DIFFERENT coding problems.
Some skills may be:
- **General** — applicable across problem types (KEEP and ENRICH)
- **Redundant** — same idea phrased differently (MERGE into the best version)
- **Problem-specific** — only relevant to one problem type (REJECT)
- **Too thin** — stated as a one-liner without enough detail (DEEPEN)

## Candidate Skills (from {num_problems} problems)
{candidate_experiences}

## Current Experience Pool (for dedup reference)
{current_pool}

## Task: Merge, Deepen, and Generalize

1. **Reject** problem-specific skills that don't generalize across problem types.
2. **Merge** semantically identical skills into the best single version — combine their WHEN/WHAT/HOW/WHY content.
3. **Deepen** thin or vague skills by adding specific WHEN conditions and concrete EXAMPLE text.
4. **Resolve contradictions** between skills — pick the correct version and explain why.
5. **Keep the pool concise** — output at most {max_experiences} skills, ordered by importance/impact.

For each skill you KEEP, use this STRUCTURED format:
```
[G#]. <Title (5-10 words)>

**When to apply**: <specific triggering conditions>
**What to do**: <concrete, actionable instructions (2-4 sentences)>
**Example**: <concrete example prompt text>
**Why it works**: <what failure mode this prevents>
```

Also add a `// General: <one-sentence justification of cross-problem applicability>` comment.

Output format:
<Experiences>
[G0]. <Title>
**When to apply**: ...
**What to do**: ...
**Example**: ...
**Why it works**: ...
// General: <justification>

[G1]. <Title>
...
</Experiences>

Your merged, generalized, deepened skills:"""


# ============================================================
# Step 4: 批量经验更新 — ADD / UPDATE / DELETE / KEEP (保留兼容)
# ============================================================
EXPERIENCE_UPDATE_TEMPLATE = """You are managing a pool of prompt engineering best practices.

## Current Experience Pool
{current_experiences}

## New Candidate Experience (extracted from recent rollouts)
{new_experience}

## Task
Decide what to do with the new experience. Choose ONE action:
- ADD: if it's genuinely new and not covered by existing ones
- KEEP: if it's already well-covered by existing ones
- DELETE: if it contradicts or degrades existing best practices

IMPORTANT: Before ADD, verify this is a GENERAL prompt-writing principle, not a problem-specific trick.

Output a JSON decision:
{{"action": "ADD|KEEP|DELETE", "target_id": null|"G#", "reasoning": "one-sentence justification"}}

Your decision:"""


# ============================================================
# Step 5: 批量合并 — 调和冲突操作 (保留兼容)
# ============================================================
BATCH_MERGE_TEMPLATE = """You are managing a pool of prompt engineering best practices.

## Current Experience Pool
{current_experiences}

## Proposed Operations (from analyzing multiple problem rollouts)
{operations}

## Task
Merge these operations into a final, non-redundant experience pool.
Rules:
- If multiple operations UPDATE the same ID, synthesize the best version
- If ADD conflicts with existing ones, resolve the conflict
- Remove truly redundant entries
- Reject any guideline that is problem-specific rather than general
- Keep the pool concise ({max_experiences} max)

Output a JSON array of final experiences:
[{{"id": "G0", "text": "..."}}, {{"id": "G1", "text": "..."}}, ...]

Your final pool:"""


# ============================================================
# Step 6: 经验注入模板 (保留兼容)
# ============================================================
EXPERIENCE_INJECTION_TEMPLATE = """When generating system prompts for coding tasks, you MUST follow these proven guidelines:

{experiences}

Now, generate a system prompt for the following problem:

{problem_input}"""


# ============================================================
# Step 7: Skill 批量增强模板 (v4 新增) — 将现有 thin skills 升级为结构化版本
# ============================================================
SKILL_ENHANCEMENT_TEMPLATE = """You are an expert in prompt engineering for code generation.

Below are multiple short skill guidelines from an existing skill library.
Each skill is currently just a one-line tip. Your job is to DEEPEN each skill
into a structured, actionable guide that a code generation model can actually use.

## Skills to Enhance
{skills_batch}

## Task
For EACH skill above, produce an enhanced version in this structured format:

```
[<original_id>]. <Enhanced Title (5-10 words)>

**When to apply**: <What task characteristics or signals indicate this skill is needed? Be specific — name concrete scenarios.>
**What to do**: <The specific technique, expanded to 2-4 concrete sentences. Don't just repeat the title.>
**Example**: <A concrete example of what the system prompt should say, in quotes. This is the most important part — show, don't just tell.>
**Why it works**: <What specific failure mode does this prevent? What do code generation models tend to get wrong without this instruction?>
```

IMPORTANT:
- The **Example** section must contain actual prompt text in quotes — this is what makes a skill actionable.
- Keep each skill self-contained — a BAgent should be able to use it without reading any other skill.
- Generalize beyond specific algorithms/data structures. Skills should apply to string processing, math, graph problems, class design alike.
- Output ALL enhanced skills together, each in the format above, separated by `---`.

Enhanced skills:"""
