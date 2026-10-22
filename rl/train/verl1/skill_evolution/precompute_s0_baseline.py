#!/usr/bin/env python3
"""
离线预计算 S_0 baseline pass-rate (待决策项1: 先离线)。

用初始 skill S_0 (skillopt/envs/lcb/skills/initial.md) 对 LCB train+val 全题跑 B 模型
(DeepSeek-V3.2 via skillopt chat_target) + stdin/stdout 单测, 得每题 s0_pass_rate。
存成 JSON 缓存, 训练时 custom_reward_lcb.py 读取, 消除在线 S_0 的 API 翻倍成本 (见
[[our-grpo-skill-method-design-20260721]] 改造2)。

缓存 key = LCB question_id, value = pass_rate ∈ [0,1] (-3.0=执行错误, -1.0=超时)。

用法:
  cd /data/zhuldz/skillopt && python /data/zhuldz/lunwen/rl/train/verl1/precompute_s0_baseline.py \
    --split_dir /data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42 \
    --s0 /data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md \
    --output /data/zhuldz/lunwen/rl/train/verl1/outputs/s0_baseline_lcb.json

环境变量 (与 run_code.sh 一致):
  export QWEN_CHAT_API_KEY="msk-bf2324..."
  export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/v1"
  export QWEN_CHAT_MAX_RPM=28
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 让 skillopt 可 import
_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)

# 必须在 import skillopt.model 前设: backend_config 在 import 时读 env, 默认 openai_chat(走 Azure 报错)
# 与 train_lcb_skillopt.sh / run_code.sh 一致: target/optimizer backend = qwen_chat (DeepSeek-V3.2 via aimpapi)
# qwen_chat backend 在 import 时还需要以下 env, 否则报 "Azure OpenAI endpoint is not configured"。
# 全部用 setdefault: 允许外部 shell 已 export 的值优先 (托管平台 / fresh shell 均自包含)。
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_API_KEY", "")  # 由环境变量提供 key
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1")
os.environ.setdefault("QWEN_CHAT_MODEL", "/model/DeepSeek-V3.2")
os.environ.setdefault("QWEN_CHAT_AIGC_USER", "zhuldz")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "4096")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "28")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

from skillopt.envs.lcb.rollout import _build_system, _build_user  # noqa: E402
from skillopt.envs.lcb.evaluator import evaluate  # noqa: E402
from skillopt.model import chat_target  # noqa: E402


def run_one(item: dict, s0_content: str, max_completion_tokens: int = 4096) -> tuple[str, dict]:
    """对单题用 S_0 跑 B 模型 + 单测, 返回 (qid, {pass_rate, hard, soft, fail_reason, agent_ok})."""
    qid = str(item["id"])
    try:
        system = _build_system(s0_content)
        user = _build_user(item)
        response, _ = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="s0_baseline",
        )
        res = evaluate(response, item["test_code"])
        res["agent_ok"] = True
        res["response_len"] = len(response)
        return qid, res
    except Exception as e:
        return qid, {
            "hard": 0, "soft": 0.0, "pass_rate": -3.0,
            "fail_reason": f"exception: {e!r}", "error_type": "Exception",
            "agent_ok": False, "response_len": 0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_dir",
        default="/data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42",
    )
    parser.add_argument("--s0", default="/data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md")
    parser.add_argument(
        "--output",
        default="/data/zhuldz/lunwen/rl/train/verl1/outputs/s0_baseline_lcb.json",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_completion_tokens", type=int, default=4096)
    args = parser.parse_args()

    with open(args.s0, encoding="utf-8") as f:
        s0_content = f.read().strip()
    print(f"📄 S_0: {len(s0_content)} chars")

    items = []
    for s in args.splits:
        path = os.path.join(args.split_dir, s, "items.json")
        if not os.path.exists(path):
            continue
        items.extend(json.load(open(path, encoding="utf-8")))
    print(f"📊 共 {len(items)} 题 (splits={args.splits})")

    # ⚠️ 必须与 parquet (json2parquet_skillopt.py:truncate_test_code) 用同款截断,
    # 否则 S_0_pass (这里算) 与 skill_pass (reward 用 parquet test_code) 的 test_code 不同,
    # delta = skill_pass - s0_pass 不可比。import 同一个函数保证一致。
    sys.path.insert(0, "/data/zhuldz/lunwen/data/OpenCodeInstruct")
    from json2parquet_skillopt import truncate_test_code  # noqa: E402
    for it in items:
        it["test_code"] = truncate_test_code(it.get("test_code", ""))
    print(f"✂️ test_code 已截断 (与 parquet 同款, max_pairs=8/max_case=4KB), delta 可比性保证")

    cache = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, it, s0_content, args.max_completion_tokens): it for it in items}
        done = 0
        for fut in as_completed(futures):
            qid, res = fut.result()
            cache[qid] = res
            done += 1
            if done % 10 == 0 or done == len(items):
                pr = res.get("pass_rate", 0)
                print(f"  [{done}/{len(items)}] {qid}: pass_rate={pr:.3f} agent_ok={res.get('agent_ok')}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # 统计
    ok = [r for r in cache.values() if r.get("agent_ok")]
    prs = [r.get("pass_rate", 0) for r in ok]
    hard = [r.get("hard", 0) for r in ok]
    print(f"\n✅ 保存: {args.output}")
    print(f"   agent_ok: {len(ok)}/{len(cache)}")
    if prs:
        print(f"   平均 pass_rate: {sum(prs)/len(prs):.4f}")
        print(f"   hard pass (全对): {sum(hard)}/{len(ok)} = {sum(hard)/len(ok):.4f}")
    print(f"   耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
