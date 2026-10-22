#!/usr/bin/env python3
"""s0_cache 重生 — 分块 + 每块子进程超时 (抗连接 wedge), v2.

v1 问题: worker 通过 stdout 返回 JSON, 但 worker 偶发异常时 rc=0 但 stdout 为空,
parent json.loads("") 崩溃 (而非重试). 且 stdout 大对象易截断.

v2 对策: worker 直接把结果写到 partial 文件 (路径经 spec 传入), parent 只校验
文件存在 + 合法 JSON. stdout/stderr 仅作日志. 单块 wedge/异常 → 文件不存在或非法
→ parent 重试 (最多 3 次). 断点续跑: 已存在的合法 partial 跳过.

口径与 precompute_s0_live.py / evolver 一致:
  split_dir = livemathematicianbench_1-3-6_seed42, select=train+val=71,
  eval_k=3, shuffle_seed=None, split_name="train+val", EVAL_TEMP=0.3.
"""
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tf_global_v2_math import build_cache_manifest, md5  # noqa: E402

S0 = "/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md"
SPLIT_DIR = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42"
OUT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json"
PARTIAL_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2/_s0_chunks"
EVAL_K = 3
CHUNK = 12  # 每块题数 (12×3=36 调用 ≈ 77s, 远低于 ~7 min wedge 点)
SUBPROC_TIMEOUT = 600  # 单块子进程硬超时 10 min (含可能的 429 退避)
MAX_ATTEMPTS = 3  # 单块重试上限

# 单块 worker: 被 subprocess 调起, 跑指定 qid 子集, 结果直接写 partial 文件.
# stdout 仅打进度日志; 结果 JSON 落盘到 spec["out"].
_WORKER = '''
import json, os, sys, time
sys.path.insert(0, "{here}")
from tf_global_v2_math import precompute_s0, load_split
SD = "{sd}"
S0 = "{s0}"
K = {k}
spec = json.loads(sys.argv[1])  # {{"qids":[...], "idx":i, "out":"..."}}
t0 = time.time()
print("[chunk{{}}] start {{}} items k={{}}".format(spec["idx"], len(spec["qids"]), K), flush=True)
all_train = load_split(SD, "train")
all_val = load_split(SD, "val")
all_items = all_train + all_val
by_id = {{str(it["id"]): it for it in all_items}}
sub = [by_id[q] for q in spec["qids"]]
s0 = open(S0, encoding="utf-8").read().strip()
c = precompute_s0(sub, s0, K, stage="tf_v2_s0_chunk"+str(spec["idx"]),
                  shuffle_seed=None, split_dir=SD, split_name="train+val")
c.pop("_manifest", None)  # 块级 manifest 无意义, 合并时统一重建
# 原子写: 先写 .tmp 再 rename, 防 parent 在写中途读到半截 JSON
tmp = spec["out"] + ".tmp"
json.dump(c, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
os.replace(tmp, spec["out"])
print("[chunk{{}}] OK {{}} items in {{:.0f}}s -> {{}}".format(
    spec["idx"], len(c), time.time()-t0, spec["out"]), flush=True)
'''.format(here=_HERE, sd=SPLIT_DIR, s0=S0, k=EVAL_K)


def _load_partial(path):
    """返回 (ok, data_or_err). 校验文件存在 + 合法 JSON + 非空 dict."""
    if not os.path.exists(path):
        return (False, "missing")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return (False, f"bad-json: {e}")
    if not isinstance(d, dict) or len(d) == 0:
        return (False, f"empty-dict")
    return (True, d)


def main():
    from tf_global_v2_math import load_split
    s0 = open(S0, encoding="utf-8").read().strip()
    train = load_split(SPLIT_DIR, "train")
    val = load_split(SPLIT_DIR, "val")
    select = train + val
    qids = [str(it["id"]) for it in select]
    print(f"📋 S0: {len(s0)} chars (hash {md5(s0)[:8]}) | select: {len(qids)} "
          f"(train {len(train)} + val {len(val)}) | k={EVAL_K} | chunk={CHUNK}", flush=True)

    os.makedirs(PARTIAL_DIR, exist_ok=True)
    chunks = [qids[i:i + CHUNK] for i in range(0, len(qids), CHUNK)]
    print(f"   {len(chunks)} chunks × {CHUNK} items (timeout {SUBPROC_TIMEOUT}s/chunk, "
          f"max {MAX_ATTEMPTS} attempts)", flush=True)

    merged = {}
    t0 = time.time()
    failed_chunks = []
    for ci, qid_sub in enumerate(chunks):
        partial_path = os.path.join(PARTIAL_DIR, f"partial_{ci:02d}.json")
        # 断点续跑: 已存在的合法 partial 直接复用
        ok, data = _load_partial(partial_path)
        if ok:
            merged.update(data)
            print(f"  chunk {ci:02d} CACHED ({len(data)} items) | total {len(merged)}/{len(qids)} "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
            continue

        done = False
        last_err = "init"
        for attempt in range(MAX_ATTEMPTS):
            # 清掉可能残留的半截/非法文件
            if os.path.exists(partial_path):
                try:
                    os.remove(partial_path)
                except OSError:
                    pass
            spec = json.dumps({"qids": qid_sub, "idx": ci, "out": partial_path})
            try:
                r = subprocess.run(
                    [sys.executable, "-c", _WORKER, spec],
                    capture_output=True, text=True, timeout=SUBPROC_TIMEOUT,
                    env={**os.environ, "PYTHONPATH": "/data/zhuldz/skillopt:" + _HERE})
                # worker 把日志打到 stdout, 错误打到 stderr
                if r.stdout.strip():
                    for line in r.stdout.strip().splitlines()[-3:]:
                        print(f"    c{ci:02d}a{attempt}: {line}", flush=True)
                if r.returncode != 0:
                    last_err = f"rc={r.returncode} stderr={r.stderr[-200:]}"
                    print(f"  chunk {ci:02d} attempt{attempt} FAIL {last_err}", flush=True)
                    time.sleep(5)
                    continue
                # 校验 worker 是否真的写出了合法文件
                ok, data = _load_partial(partial_path)
                if not ok:
                    last_err = f"worker-rc0-but-no-valid-file ({data})"
                    print(f"  chunk {ci:02d} attempt{attempt} WARN {last_err}", flush=True)
                    time.sleep(5)
                    continue
                merged.update(data)
                print(f"  chunk {ci:02d} attempt{attempt} OK ({len(data)} items) | "
                      f"total {len(merged)}/{len(qids)} elapsed {time.time()-t0:.0f}s", flush=True)
                done = True
                break
            except subprocess.TimeoutExpired:
                last_err = "TIMEOUT (wedge)"
                print(f"  chunk {ci:02d} attempt{attempt} {last_err} → retry", flush=True)
                time.sleep(5)
        if not done:
            print(f"  ❌ chunk {ci:02d} FAILED {MAX_ATTEMPTS} attempts ({last_err}) — aborting",
                  flush=True)
            failed_chunks.append(ci)
            # 继续后续块而非整体 abort (尽量多落盘), 最后再判失败
            continue

    if failed_chunks:
        print(f"\n❌ {len(failed_chunks)} chunk(s) failed: {failed_chunks}; "
              f"recovered {len(merged)}/{len(qids)} items. Aborting merge.", flush=True)
        sys.exit(1)

    # 全部完成 → 合并 + 加 manifest
    merged["_manifest"] = build_cache_manifest(
        s0, EVAL_K, 0.3, None, select, SPLIT_DIR, "train+val")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    json.dump(merged, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    os.replace(tmp, OUT)
    n = len(merged) - 1
    m = merged["_manifest"]
    print(f"✅ wrote {OUT} ({n} items + manifest) in {time.time()-t0:.0f}s", flush=True)
    print(f"   manifest: k={m.get('k')} eval_temp={m.get('eval_temp')} "
          f"shuffle_seed={m.get('shuffle_seed')} skill_hash={str(m.get('skill_hash',''))[:8]} "
          f"split_name={m.get('split_name')} n_items={m.get('n_items')}", flush=True)


if __name__ == "__main__":
    main()
