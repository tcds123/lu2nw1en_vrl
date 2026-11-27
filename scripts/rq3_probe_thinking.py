#!/usr/bin/env python3
"""RQ3 诊断: 同题分别带/不带 chat_template_kwargs.enable_thinking, 观察
延迟 / completion_tokens / finish_reason, 以判定服务端 thinking 实际状态。

只发 1~4 次请求; key 从 configs/models_aimp.local.yaml 读, 不打印。
"""
import json, sys, time, urllib.request, importlib.util
import yaml

SPLIT = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
S0 = "/data/zhuldz/skillopt/skillopt/envs/appsplus/skills/initial.md"
URL = "https://aimpapi.midea.com/t-aigc/llm-openai-api/chat/completions"

item_id = sys.argv[1] if len(sys.argv) > 1 else "6482"
modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["no_think", "serverdef"]

spec = importlib.util.spec_from_file_location(
    "ev", "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
tr = {str(i["id"]): i for i in m.load_split(SPLIT, "train", truncate=True)[:80]}
it = tr[item_id]
s0 = open(S0, encoding="utf-8").read()
key = yaml.safe_load(open("/data/zhuldz/lunwen/configs/models_aimp.local.yaml"))["providers"]["deepseek-v4-pro"]["api_key"]

for mode in modes:
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if mode == "no_think" else {}
    mt = 512 if mode == "trunc512" else 16384
    body = {"model": "deepseek-v4-pro",
            "messages": [{"role": "system", "content": m._build_system(s0)},
                         {"role": "user", "content": m._build_user(it)}],
            "max_tokens": mt, "stream": False, "temperature": 0.0, **extra}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"}, method="POST")
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            j = json.loads(r.read())
        ch = j["choices"][0]; msg = ch.get("message") or {}
        print(json.dumps({"item": item_id, "mode": mode, "status": "OK",
                          "latency_s": round(time.time() - t, 1), "usage": j.get("usage"),
                          "finish_reason": ch.get("finish_reason"),
                          "content_chars": len(msg.get("content") or ""),
                          "reasoning_chars": len(str(msg.get("reasoning_content") or ""))}, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"item": item_id, "mode": mode, "status": "FAIL",
                          "latency_s": round(time.time() - t, 1), "err": repr(e)[:300]}, ensure_ascii=False))
    sys.stdout.flush()
