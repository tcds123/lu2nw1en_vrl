#!/usr/bin/env python
"""补全 appsplus_full_best 缺失的 548 题。

best eval 在 2026-07-18 07:24 启动后被中断, 只完成 2217/2765 (缺 548)。
本脚本:
  1. 复用 noskill 已生成的确定性 split (seed42, 2:1:7), 取 test 全 2765 题
  2. 找出 best/predictions 中缺失的 item id
  3. 用相同 best_skill.md + 相同 backend 配置 (DeepSeek-V3.2, temp=0, 28RPM)
     只跑缺失题, 写入 best/predictions/<id>/conversation.json
  4. 合并全部 2765 题, 重算 eval_summary.json (hard/soft/n)

不重跑已完成的题 (省 API). 复用 rollout.process_one 保证 prompt/eval 格式与原 run 一致.
"""
from __future__ import annotations
import json, os, sys, time

SKILLOPT = "/data/zhuldz/skillopt"
sys.path.insert(0, SKILLOPT)
os.chdir(SKILLOPT)

# ---- config backend (与 run_code.sh evalskill 一致) ----
os.environ.setdefault("QWEN_CHAT_API_KEY", os.environ.get("QWEN_CHAT_API_KEY", ""))  # key 从环境变量读取, 不写入仓库
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1")
os.environ.setdefault("QWEN_CHAT_MODEL", "/model/DeepSeek-V3.2")
os.environ.setdefault("QWEN_CHAT_AIGC_USER", "zhuldz")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "4096")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "28")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

from skillopt.model import qwen_backend as qwen_backend
from skillopt.model.backend_config import set_target_backend
from skillopt.envs.appsplus.rollout import process_one

# ---- configure qwen chat (mirrors configure_qwen_chat in eval_only) ----
qwen_backend.configure_qwen_chat(
    base_url=os.environ["QWEN_CHAT_BASE_URL"],
    api_key=os.environ["QWEN_CHAT_API_KEY"],
    temperature=float(os.environ["QWEN_CHAT_TEMPERATURE"]),
    timeout_seconds=300,
    max_tokens=int(os.environ["QWEN_CHAT_MAX_TOKENS"]),
    enable_thinking=False,
    target_base_url=os.environ["QWEN_CHAT_BASE_URL"],
    target_api_key=os.environ["QWEN_CHAT_API_KEY"],
    target_temperature=float(os.environ["QWEN_CHAT_TEMPERATURE"]),
    target_timeout_seconds=300,
    target_max_tokens=int(os.environ["QWEN_CHAT_MAX_TOKENS"]),
    target_enable_thinking=False,
)
set_target_backend("qwen_chat")
qwen_backend.set_target_deployment(os.environ["QWEN_CHAT_MODEL"])

# ---- paths ----
BEST_OUT = "outputs/appsplus_full_best"
BEST_SKILL = "outputs/appsplus_full_optimize/best_skill.md"
SPLIT_DIR = f"{BEST_OUT}/_generated_splits/appsplus_2-1-7_seed42"
TEST_ITEMS = json.load(open(f"{SPLIT_DIR}/test/items.json"))

with open(BEST_SKILL) as f:
    skill_content = f.read()
print(f"[skill] {BEST_SKILL} ({len(skill_content)} chars)")

# ---- find missing ----
existing = set(os.listdir(f"{BEST_OUT}/predictions"))
items_by_id = {str(it["id"]): it for it in TEST_ITEMS}
missing_ids = sorted(
    [i for i in items_by_id if i not in existing],
    key=lambda x: int(x) if x.isdigit() else 99999
)
print(f"[missing] {len(missing_ids)} / {len(TEST_ITEMS)} test items need completion")
print(f"[missing] first 10: {missing_ids[:10]}")

if not missing_ids:
    print("[done] nothing to complete")
else:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {
            pool.submit(process_one, items_by_id[i], BEST_OUT, skill_content,
                        max_turns=1, max_completion_tokens=4096, exec_timeout=300): i
            for i in missing_ids
        }
        for fut in as_completed(futs):
            iid = futs[fut]
            try:
                r = fut.result()
                done += 1
                if done % 25 == 0 or done == len(missing_ids):
                    el = time.time() - t0
                    print(f"  [{done}/{len(missing_ids)}] id={iid} hard={r['hard']} soft={r['soft']:.3f} "
                          f"agent_ok={r['agent_ok']} ({el:.0f}s elapsed)")
            except Exception as e:
                print(f"  !! id={iid} exception: {e!r}")

# ---- recompute summary over ALL 2765 ----
import re
hard_pass = 0; soft_sum = 0.0; n_eval = 0; n_total = 0
results = []
for it in TEST_ITEMS:
    iid = str(it["id"])
    n_total += 1
    cpath = f"{BEST_OUT}/predictions/{iid}/conversation.json"
    if not os.path.exists(cpath):
        continue
    try:
        conv = json.load(open(cpath))
    except Exception:
        continue
    for m in conv:
        c = str(m.get("content", ""))
        mt = re.search(r"\[EVALUATION RESULT\](.*)", c, re.S)
        if mt:
            block = mt.group(1)
            hp = re.search(r"hard \(all pass\):\s*([01])", block)
            sp = re.search(r"pass_rate:\s*([0-9.]+)", block)
            if hp:
                h = int(hp.group(1)); s = float(sp.group(1)) if sp else 0.0
                n_eval += 1
                hard_pass += h
                soft_sum += s
                results.append({"id": iid, "hard": h, "soft": s})
            break

hard = hard_pass / n_eval if n_eval else 0.0
soft = soft_sum / n_eval if n_eval else 0.0
summary = {
    "skill": os.path.abspath(BEST_SKILL),
    "split": "test",
    "n_items": n_eval,
    "n_total": n_total,
    "hard": hard,
    "soft": soft,
    "note": "completed 2026-07-20: 548 missing items filled in, full 2765 recomputed",
}
with open(f"{BEST_OUT}/eval_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
with open(f"{BEST_OUT}/results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"[FINAL] n_eval={n_eval}/{n_total}  hard={hard:.4f}  soft={soft:.4f}")
print(f"[FINAL] saved -> {BEST_OUT}/eval_summary.json")
print(f"{'='*60}")
