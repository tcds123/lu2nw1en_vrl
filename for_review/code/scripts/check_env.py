#!/usr/bin/env python3
"""环境自检: 路径 / 数据 / 依赖 / key / 一次真实 API 调用 (1 题 1 次, 成本可忽略).

用法:
    python scripts/check_env.py          # 完整检查 (含 1 次 API ping)
    python scripts/check_env.py --offline  # 不调 API, 只检查离线部分
"""
import argparse
import json
import os
import sys
import time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(PKG, "code")
sys.path.insert(0, CODE)

import _env  # noqa: E402  (导入即加载 .env)

ok = True


def check(name, fn):
    global ok
    t0 = time.time()
    try:
        msg = fn()
        print(f"  ✅ {name}: {msg}  ({time.time()-t0:.2f}s)")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  ❌ {name}: {e!r}")


def _py():
    return sys.version.split()[0]


def _data():
    out = []
    for n in ("aime24.jsonl", "aime25.jsonl", "skill_aime_initial_v2.md"):
        p = _env.data_path(n)
        assert os.path.isfile(p), f"缺少数据文件 {p}"
        out.append(n)
    with open(_env.data_path("aime24.jsonl"), encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    assert len(items) == 30, f"aime24.jsonl 应为 30 题, 实际 {len(items)}"
    s0 = open(_env.data_path("skill_aime_initial_v2.md"), encoding="utf-8").read().strip()
    assert s0, "S₀ skill 为空"
    return f"{len(items)} AIME24 items, S₀ {len(s0)} chars"


def _deps():
    import numpy  # noqa: F401
    return f"numpy {numpy.__version__}"


def _imports():
    import tf_global_v2_aime  # noqa: F401
    import skill_evolver_aime_v2  # noqa: F401
    import eval_aime25_multiseed  # noqa: F401
    return "evolver + evaluator + AIME backend"


def _key():
    return "set" if _env.require_key() else "missing"


def _api():
    from skillopt.model import chat_target
    resp, usage = chat_target(
        system="You are a calculator. Answer with only the number.",
        user="What is 17*23?",
        max_completion_tokens=2048,
        retries=2,
        stage="check_env",
        timeout=300,
    )
    snippet = (resp or "").strip().replace("\n", " ")[:120]
    if not snippet:
        raise RuntimeError("API 返回空内容 (检查 key / model 名 / base_url)")
    return f"reply={snippet!r} usage={usage}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    print("== 配置 ==")
    _env.print_config()
    print("\n== 检查 ==")
    check("python", _py)
    check("data", _data)
    check("deps", _deps)
    check("imports", _imports)
    check("api_key", _key)
    if not args.offline:
        check("api ping (1 call)", _api)
    print()
    if ok:
        print("✅ 全部通过 — 可以运行 scripts/run_evolve.sh")
    else:
        print("❌ 有检查未通过 — 按上面的报错修复; 常见问题见 README §8 排错")
        sys.exit(1)


if __name__ == "__main__":
    main()
