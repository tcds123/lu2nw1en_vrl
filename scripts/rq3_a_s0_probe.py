#!/usr/bin/env python3
"""RQ3-A: 单题 S₀ 探针 (deepseek-v4-pro) — 1 次调用, 回答"能否调用 + 实际生效了什么"。

用法 (授权后):
    set -a; source configs/.env; set +a          # AIMP_KEY
    python3 scripts/rq3_a_s0_probe.py --report outputs/rq3/rq3a_probe.json

输出 JSON: 请求值 (model / thinking / max_tokens / temperature 实际序列化),
响应 usage, 以及延迟/attempt。**1 次调用, 不是 100 次抽样。**
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SKILLOPT = "/data/zhuldz/skillopt"

PROBE_SYSTEM = "You are a careful assistant. Answer concisely."
PROBE_USER = "Return only the integer 7."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="outputs/rq3/rq3a_probe.json")
    ap.add_argument("--max-tokens", type=int, default=16384)
    args = ap.parse_args()

    key = os.environ.get("QWEN_CHAT_API_KEY") or os.environ.get("AIMP_KEY")
    if not key:
        print("[probe] BLOCKED: 未设置 QWEN_CHAT_API_KEY / AIMP_KEY", file=sys.stderr)
        return 2

    # 冻结 backend 选择 (与 launch_tf_appsplus_routing.sh 同口径), 仅换 model 与 base_url。
    os.environ["QWEN_CHAT_API_KEY"] = key
    os.environ["QWEN_CHAT_BASE_URL"] = "https://aimpapi.midea.com/t-aigc/llm-openai-api"
    os.environ["QWEN_CHAT_MODEL"] = "deepseek-v4-pro"
    os.environ.pop("QWEN_CHAT_AIGC_USER", None)  # OpenAI-compat 端点: 与 flash 同端点
    os.environ["QWEN_CHAT_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["QWEN_CHAT_TEMPERATURE"] = "0.0"
    os.environ["QWEN_CHAT_ENABLE_THINKING"] = "false"
    os.environ["QWEN_CHAT_MAX_RPM"] = "20"
    os.environ["QWEN_CHAT_BURST"] = "1"
    os.environ["QWEN_CHAT_TIMEOUT_SECONDS"] = "600"
    os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
    os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")

    sys.path.insert(0, SKILLOPT)
    from skillopt.model import chat_target  # noqa: E402
    from skillopt.model import qwen_backend  # noqa: E402

    # TARGET_CONFIG 在 import 时按 env 构建; 若此前已被别的模块 import 过, 需显式重建。
    qwen_backend.TARGET_CONFIG = qwen_backend._initial_config("target")
    cfg = qwen_backend.TARGET_CONFIG
    # 请求值 = 代码里真正序列化进 payload 的字段 (逐项复刻 _chat 的拼装逻辑)
    token_key = "max_completion_tokens" if cfg.use_max_completion_tokens else "max_tokens"
    serialized = {
        "model": cfg.deployment,
        "base_url": cfg.base_url,
        token_key: min(args.max_tokens, cfg.max_tokens),
        "stream": False,
        "temperature": cfg.temperature,
        "thinking_field": {"chat_template_kwargs": {"enable_thinking": True}} if cfg.enable_thinking else None,
        "aigc_user_header": bool(os.environ.get("QWEN_CHAT_AIGC_USER")),
    }
    config_values = {
        "QWEN_CHAT_ENABLE_THINKING": os.environ.get("QWEN_CHAT_ENABLE_THINKING"),
        "QWEN_CHAT_MAX_TOKENS": os.environ.get("QWEN_CHAT_MAX_TOKENS"),
        "QWEN_CHAT_TEMPERATURE": os.environ.get("QWEN_CHAT_TEMPERATURE"),
    }

    t0 = time.time()
    text, usage = chat_target(system=PROBE_SYSTEM, user=PROBE_USER,
                              max_completion_tokens=args.max_tokens, retries=1,
                              stage="rq3a_probe")
    dt = time.time() - t0

    report = {
        "status": "PASS",
        "calls": 1,
        "config_values": config_values,
        "serialized_request": serialized,
        "reply": text[:200],
        "usage": usage,
        "latency_s": round(dt, 2),
        "note": "服务端是否照此执行仍需人工读 reply/usage 判断; attempt=1 表示无重发被观测",
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
