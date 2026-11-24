# Competitive Programming Heuristics (stdin/stdout)

## Input Parsing
- Read the problem carefully: how many test cases per run? Is the first token `n` (size) or `t` (number of queries)? Misreading this is the #1 cause of wrong-answer.
- Parse multi-line / multi-token input robustly: read all tokens once into a list, then index, rather than interleaving `input()` calls that assume line boundaries.
- Decide input types up front: integers vs floats vs strings. Cast once, reuse.

## Output Format
- Match the exact output spec: one answer per line vs space-separated; "YES"/"NO" vs "Yes"/"No" vs 1/0; trailing spaces; decimal precision.
- When the spec is ambiguous, follow the sample I/O literally — it is ground truth for format.

## Correctness Discipline
- Work through the sample I/O by hand before coding; re-run your solution on it mentally after writing.
- Check edge cases: empty input, n=1, n=2, all-equal values, negative numbers, maximum n, already-sorted input.
- Verify the algorithm's correctness argument (greedy choice / invariant / monotonicity) — not just that it "feels right".
- Confirm complexity fits the constraints: n ≤ 10^5 ⇒ O(n log n) is safe, O(n^2) is not; n ≤ 20 ⇒ exponential is fine.

## Common Pitfalls
- Off-by-one in loops and ranges: `range(n)` vs `range(1, n+1)`; half-open vs closed intervals.
- Using `==` on floats; scale to integers or use an epsilon when the problem allows.
- Mutating a shared structure across test cases inside one run — reinitialise per case.
- Forgetting to flush or printing debug output to stdout (it corrupts the answer stream). Remove all debug prints.

## Final Check
- Before submitting, trace the smallest non-trivial case and one boundary case through your code, and confirm the output matches the expected format exactly.
