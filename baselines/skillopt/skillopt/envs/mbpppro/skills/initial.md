# MBPP-Pro Code Generation Heuristics

## Edge Cases
- Handle empty containers (`[]`, `""`, `{}`) and single-element inputs explicitly before the main loop.
- Check boundary indices off-by-one: ranges, slicing, and `for` bounds. Prefer `range(len(x))` clarity over clever indexing.
- For numeric inputs, decide the sign of zero and integer vs float division deliberately.

## Correctness Discipline
- Re-read the function signature: argument names, default values, and return type. Match the expected return shape (tuple vs list, int vs bool) exactly.
- Verify the return value against every clause of the spec, not just the example.
- When the spec says "find the X that ...", confirm there is exactly one; tie-break per the spec or return the first match.

## Common Pitfalls
- Mutating an input list in place when the caller expects a new object — copy first.
- Using `==` on floats; compare with a tolerance or scale to integers when the problem allows.
- Forgetting that `sorted()` returns a new list while `.sort()` returns `None`.

## Final Check
- Before outputting, mentally run the smallest non-trivial example and the empty example through your code.
