#!/usr/bin/env python
# eva/dryrun_flash.py — deepseek-v4-flash 跨模型评测 dry-run (1 题 k=1, 走完整 run_one 链路)
#
# 目的: 全量跨模型评测前的链路验证。低成本排掉 2 个风险:
#   1. thinking 字段兼容性 (qwen_backend 发 chat_template_kwargs.enable_thinking, flash 是否认)
#   2. 长 prompt + 16384 max_tokens 是否触发 300s timeout (smoke 的 17.4s 是简单 "PONG")
# 链路: _build_system/_build_user → chat_target → evaluate_aime (EM 解析)
#       完整复用 tf_global_v2_aime.run_one, 与全量评测同一条路径。
#
# flash 接入 = 纯 env 替换 (qwen_backend 通用 OpenAI 兼容 POST, 零代码改动):
#   QWEN_CHAT_MODEL=deepseek-v4-flash, QWEN_CHAT_BASE_URL=.../llm-openai-api, QWEN_CHAT_API_KEY=<flash>
#
# 用法:
#   source lunwen/bin/activate
#   python eva/dryrun_flash.py [--qid aime24:74] [--k 1]
#
# 注意: env 必须在 import tf_global_v2_aime 之前设置 (tf_global_v2_math 的 setdefault 才不会覆盖)
import argparse
import os
import sys
import time
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---- 1. 读 flash key from local.yaml (在 import evolver 之前) ----
def load_flash_key(cfg_path: Path) -> str:
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    p = (cfg.get("providers") or {}).get("deepseek-v4-flash")
    if not p:
        sys.exit(f"[err] deepseek-v4-flash 不在 {cfg_path}")
    return p["api_key"], p["base_url"], p.get("model", "deepseek-v4-flash")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="aime24:74", help="AIME24 题 id (默认 aime24:74, answer=480, V3.2 历史答对)")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--config", default=str(REPO / "configs" / "models_aimp.local.yaml"))
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"[err] 本地明文配置不存在: {cfg_path}\n      需要 configs/models_aimp.local.yaml (含真实 key)")

    key, base_url, model = load_flash_key(cfg_path)

    # ---- 2. 在 import evolver 之前覆盖 env (setdefault 不覆盖已设值) ----
    os.environ["QWEN_CHAT_API_KEY"] = key
    os.environ["QWEN_CHAT_BASE_URL"] = base_url   # .../llm-openai-api (V3.2 是 aimp-deepseek/v1)
    os.environ["QWEN_CHAT_MODEL"] = model          # deepseek-v4-flash (V3.2 是 /model/DeepSeek-V3.2)
    # flash openai-compat 端点 smoke 验证不需要 AIGC-USER; 保留旧的也无害 (header 多余但 openai-compat 忽略)
    # 不设 QWEN_CHAT_AIGC_USER → qwen_backend L244-247 不加该 header (flash 不需要)
    os.environ.pop("QWEN_CHAT_AIGC_USER", None)
    # 保守限流: flash rate limit 未知, 降到 20 RPM (V3.2 默认 28)
    os.environ.setdefault("QWEN_CHAT_MAX_RPM", "20")
    os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "16384")

    print(f"[dryrun] model={model}  base_url={base_url}  key={key[:12]}...{key[-4:]}")
    print(f"[dryrun] qid={args.qid}  k={args.k}")

    # ---- 3. import evolver (此时 env 已生效) ----
    sys.path.insert(0, str(REPO / "rl" / "train" / "verl1" / "skill_evolution"))
    from tf_global_v2_aime import run_one, set_temp, EVAL_TEMP, S0_DEFAULT
    print(f"[dryrun] S0={S0_DEFAULT}  EVAL_TEMP={EVAL_TEMP}")

    # ---- 4. 读题 ----
    with open(REPO / "data" / "aime24.jsonl", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    item = next((it for it in items if str(it["id"]) == args.qid), None)
    if item is None:
        sys.exit(f"[err] 找不到 qid={args.qid}, 可用: {[it['id'] for it in items[:5]]}...")
    print(f"[dryrun] problem[:80]={item['problem'][:80]!r}  answer={item['answer']}")

    # ---- 5. 读 S0 skill + 跑 run_one ----
    s0 = open(S0_DEFAULT, encoding="utf-8").read().strip()
    set_temp(EVAL_TEMP)
    t0 = time.time()
    try:
        res = run_one(item, s0, stage="dryrun_flash", k=args.k)
    except Exception as e:
        dt = time.time() - t0
        print(f"\n[FAIL] run_one 异常 ({dt:.1f}s): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

    dt = time.time() - t0
    print(f"\n[done] run_one 完成, {dt:.1f}s")
    print(f"  pass_rate   = {res['pass_rate']}")
    print(f"  agent_ok     = {res['agent_ok']}")
    print(f"  n_pass/n_samples = {res['n_pass']}/{res['n_samples']}")
    print(f"  n_error      = {res.get('n_error', '?')}")
    print(f"  predicted    = {res['predicted_label']!r}")
    print(f"  correct      = {res['correct_label']!r}")
    print(f"  fail_reason  = {res.get('fail_reason', '')!r}")
    print(f"  response[:400] = {res.get('response', '')[:400]!r}")

    # 判定
    if res["agent_ok"] and res["pass_rate"] >= 0:
        print("\n[verdict] ✅ flash 链路通: 正常返回 + EM 解析成功 + 无字段报错。可进行全量。")
    elif res.get("n_error", 0) > 0 and res["pass_rate"] == 0:
        print(f"\n[verdict] ⚠️ 链路通但全部采样异常 (n_error={res.get('n_error')}): 可能字段不兼容/timeout, 见 fail_reason")
    else:
        print(f"\n[verdict] ⚠️ 非预期状态, 见上方字段")


if __name__ == "__main__":
    main()
