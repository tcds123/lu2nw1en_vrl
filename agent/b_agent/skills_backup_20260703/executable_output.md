---
id: G0
name: Executable Output Enforcement
version: 1
score_delta: 0.0
usage_count: 0
triggers:
  - code generation
  - function implementation
  - write code
  - implement
---

Always output complete, runnable code as the final answer. Never output descriptions, meta-instructions, or partial code snippets as the final answer. The code must be directly executable without modification.

Bad: "You could implement this by..."
Good: "```python\ndef solve():\n    ...\n```"
