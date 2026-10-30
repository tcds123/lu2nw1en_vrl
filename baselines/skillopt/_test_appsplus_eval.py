from skillopt.envs.appsplus.evaluator import evaluate
import json

good = """import collections
n = int(input())
A = list(map(int, input().split()))
S = [0]*n
S[0] = A[0]
for i in range(1, n):
    S[i] = S[i-1] + A[i]
C = collections.Counter(S)
print(n - max(C.values()))
"""
tc = json.dumps({"stdin": ["3\n5 0 -5\n", "1\n0\n"], "stdout": ["1\n", "0\n"]})

r = evaluate("```python\n" + good + "\n```", tc)
print("GOOD:", r["hard"], r["soft"], r["error_type"], "| fail:", r["fail_reason"][:150])

r2 = evaluate("```python\nn=int(input());A=list(map(int,input().split()));print(0)\n```", tc)
print("BAD:", r2["hard"], r2["soft"], r2["error_type"], "| fail:", r2["fail_reason"][:100])

r3 = evaluate("```python\nwhile True:\n    pass\n```", tc)
print("TIMEOUT:", r3["hard"], r3["soft"], r3["error_type"])
