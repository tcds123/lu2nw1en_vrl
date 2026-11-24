#!/usr/bin/env python3
# 一次性探测: flash 在 cap 4096 / 16384 + 带/不带 thinking 字段时的 content 空否.
# 验证 #26 (hard 题 thinking 填满 4096 → content 空 → pass=0) + thinking disabled 是否为有效兜底.
import json, os, sys, urllib.request

import yaml
cfg = yaml.safe_load(open("/data/zhuldz/lunwen/configs/models_aimp.local.yaml"))
p = cfg["providers"]["deepseek-v4-flash"]
KEY, URL, MODEL = p["api_key"], p["base_url"] + "/chat/completions", p.get("model", "deepseek-v4-flash")

# 真实 AIME24:62 (曾在 #26 smoke 全错)
items = [json.loads(l) for l in open("/data/zhuldz/lunwen/data/aime24.jsonl")]
prob = [x for x in items if str(x["id"]) == "aime24:62"][0]["problem"]
SYS = "You are a mathematics expert. Solve the problem step by step, then put the final integer answer in <answer>\\boxed{N}</answer>."

def call(max_tok, thinking=None):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prob}],
            "max_tokens": max_tok, "temperature": 0.0}
    if thinking is not None:
        body["extra_body"] = {"thinking": thinking}
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking.get("type") == "enabled")}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        ch = r.get("choices", [{}])[0]; m = ch.get("message", {}); u = r.get("usage", {})
        det = u.get("completion_tokens_details", {})
        return (f"finish={ch.get('finish_reason')} content_len={len(m.get('content') or '')} "
                f"reasoning_len={len(m.get('reasoning_content') or '')} "
                f"completion_tok={u.get('completion_tokens')} reasoning_tok={det.get('reasoning_tokens')}")
    except Exception as e:
        return f"ERR {e}"

print(f"AIME24:62 (hard)  problem_len={len(prob)}")
print(f"  cap4096  default      : {call(4096)}")
print(f"  cap16384 default      : {call(16384)}")
print(f"  cap16384 thinking=dis : {call(16384, {'type':'disabled'})}")
