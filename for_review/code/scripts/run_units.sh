#!/usr/bin/env bash
# 离线单元自检 (零 API, 秒级): 评分器 / 数据划分 / 防泄漏 / EFL 库 / imports.
# 改代码后先跑这个, 再跑 run_smoke.sh.
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PKG/code"
# 允许在没有 key 的情况下 import
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-offline-dummy}"
python - <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
import _env

# 1) 评分器
from eval_aime24_multiseed import extract_int_answer, evaluate_aime
cases = {
    r"so $\boxed{113}$. <answer>\boxed{113}</answer>": 113,
    r"<answer>\boxed{0}</answer>": 0,
    r"reasoning 12, 34, then \boxed{007}": 7,
    "no numbers here": None,
}
for t, want in cases.items():
    got = extract_int_answer(t)
    assert got == want, (t, got, want)
assert evaluate_aime(r"\boxed{999}", "999")["em"] == 1.0
assert evaluate_aime(r"\boxed{998}", "999")["em"] == 0.0
print("✅ scorer")

# 2) AIME24 划分确定性 train18/val12
from tf_global_v2_aime import load_aime_split
tr, va = load_aime_split(_env.data_path("aime24.jsonl"), train_n=18, seed=42)
assert len(tr) == 18 and len(va) == 12, (len(tr), len(va))
tr2, va2 = load_aime_split(_env.data_path("aime24.jsonl"), train_n=18, seed=42)
assert [x["id"] for x in tr] == [x["id"] for x in tr2]
assert not ({x["id"] for x in tr} & {x["id"] for x in va})
print("✅ split 18/12 deterministic")

# 3) 防泄漏检查可调用
from tf_global_v2_math import check_skill_leakage
s0 = open(_env.data_path("skill_aime_initial_v2.md"), encoding="utf-8").read()
ok, reasons = check_skill_leakage(s0, tr)
print(f"✅ leakage check on S0: {'OK' if ok else 'WARN ' + str(reasons)}")

# 3b) 纯 Python Beta 分位 (_beta_ppf, 免 scipy) 与 scipy 对照; 没装 scipy 则只做范围/单调性检查
from tf_global_v2_math import _beta_ppf, beta_binomial_lcb
try:
    import scipy.stats as _st
    for a, b in [(0.5, 0.5), (0.5, 3.5), (9.5, 4.0), (10.5, 18.5), (30.5, 0.5)]:
        for q in (0.025, 0.25, 0.5):
            d = abs(_beta_ppf(q, a, b) - _st.beta.ppf(q, a, b))
            assert d < 1e-6, (a, b, q, d)
    print("✅ beta_ppf vs scipy (<1e-6)")
except ImportError:
    prev = -1.0
    for i in range(11):
        v = _beta_ppf(i / 10, 0.5 + 3, 0.5 + 7)
        assert 0.0 <= v <= 1.0 and v >= prev, (i, v)
        prev = v
    print("✅ beta_ppf monotone in [0,1] (scipy not installed)")
# 全对的题 posterior Beta(1.5,0.5), 25% 分位 = 0.5971500783...
assert abs(beta_binomial_lcb({"q": (1, 1)})[0] - 0.5971500783855161) < 1e-9
print("✅ beta_binomial_lcb")
PYEOF

# 4) facet_library 自检
python skill_evolver_aime_v2.py --self_test_efl

# 5) 评测/进化脚本 import 闭环
python -c "import sys; sys.path.insert(0,'.'); import eval_aime24_multiseed, eval_aime25_multiseed"
echo "✅ imports ok"
