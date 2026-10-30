# Competition Mathematics Reasoning Skill

You are solving AIME-style competition mathematics problems. Your final answer must be an integer from 0 to 999.

## Approach

1. **Read the problem twice.** Identify exactly what is asked: a count, a sum, a maximum/minimum, a specific value, or a remainder. A large fraction of errors come from answering a slightly different question than the one asked (e.g. computing a value when a count is requested, or a sum when a maximum is requested).

2. **Restate the quantities.** Before computing, name every quantity in play and the relation asked for. Distinguish "the number of X" from "the sum of X" from "the largest X". Distinguish "for how many" (a count) from "what is" (a value).

3. **Choose a representation suited to the problem type.**
   - Combinatorics: decide between casework, complementary counting, and inclusion-exclusion early; pick the one with the fewest cases.
   - Number theory: translate divisibility/congruence conditions into residue classes and bound the search range explicitly.
   - Algebra: introduce variables for the unknowns, write the governing equations, and count equations vs unknowns before solving.
   - Geometry: set up coordinates or identify a symmetry; keep track of orientation and sign.

4. **Enumerate cases systematically.** When cases are needed, list them in a fixed order and verify they are mutually exclusive and exhaustive. The most common combinatorial error is a case that overlaps another or a case that is silently dropped.

5. **Carry the computation carefully.** Write intermediate values down rather than doing long chains mentally. After each arithmetic step, do a quick magnitude/parity sanity check.

6. **Verify the answer form.** Reduce fractions fully. If the problem asks for a remainder, take it. If the answer must be an integer 0-999 and yours is not, re-examine your setup, not just your arithmetic.

## Self-check before committing

- Did I answer the question that was actually asked (count vs value vs sum vs max)?
- Are my cases disjoint and complete?
- Is the magnitude plausible (AIME answers are 0-999)?
- Did I reduce / take the mod / round as required?

Report the final integer inside `<answer>\boxed{N}</answer>`.
