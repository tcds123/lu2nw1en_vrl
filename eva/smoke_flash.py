#!/usr/bin/env python
# eva/smoke_flash.py — deepseek-v4-flash key 冒烟测试 (1 条请求, 验证 200 + 正常回复)
#
# 用途: 跨模型评测切到 flash 前, 先验证 key/端点是否可用。
# key 从 configs/models_aimp.local.yaml 读取 (明文本地文件, gitignored), 不硬编码。
#
# 用法:
#   source lunwen/bin/activate
#   python eva/smoke_flash.py [--config configs/models_aimp.local.yaml] [--model deepseek-v4-flash]
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_provider(cfg_path: Path, model_key: str) -> dict:
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    providers = cfg.get("providers") or {}
    if model_key not in providers:
        sys.exit(f"[err] model '{model_key}' 不在 {cfg_path} 的 providers 里。可选: {list(providers)}")
    return providers[model_key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "models_aimp.local.yaml"))
    ap.add_argument("--model", default="deepseek-v4-flash")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"[err] 本地明文配置不存在: {cfg_path}\n"
                 f"      跨模型评测需要 configs/models_aimp.local.yaml (含真实 key)。")

    p = load_provider(cfg_path, args.model)
    api = p.get("api")
    key = p.get("api_key", "")
    base_url = p.get("base_url", "")
    model_name = p.get("model", args.model)

    print(f"[smoke] model={args.model}  api={api}  base_url={base_url}  model_name={model_name}")
    print(f"[smoke] key={key[:12]}...{key[-4:]}  (len={len(key)})")

    if api != "openai-completions":
        sys.exit(f"[err] 本脚本只测 openai-completions 类型, 该 model api={api}。glm-5.2 需 AIGC-USER header, 另测。")

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=base_url)

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
            max_tokens=16,
            extra_body=p.get("extra_body") or {},
        )
    except Exception as e:
        dt = time.time() - t0
        print(f"[FAIL] 异常 ({dt:.1f}s): {type(e).__name__}: {e}")
        sys.exit(2)

    dt = time.time() - t0
    content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    print(f"[ok]   HTTP 200, {dt:.1f}s, content={content!r}")
    print("[done] flash key 可用, 可进行跨模型评测。")


if __name__ == "__main__":
    main()
