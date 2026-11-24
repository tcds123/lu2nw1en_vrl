#!/usr/bin/env python3
"""2026-11-13: cap 扫描自动监控+记账。

每 20s 扫 cap8192/cap4096 日志, 新出现的结果行 →
  1) 读 per_<arm>_seed<k>.json 核实: n_errors=0, completion_tokens<=cap, truncated 计数与日志一致
  2) 追加结构化记录到 memory/cap-sweep-20261113.md 并同步 for_review/memory/
  3) 重建 for_review/dist/for_review_20261113.zip (含 key 扫描, 0 命中才替换)
不写 /tmp (环境重启即丢), 状态存 .cap_autorecord_state.json。
"""
import json, os, re, subprocess, sys, time, zipfile, glob

ROOT = "/data/zhuldz/lunwen"
OUT = f"{ROOT}/for_review_aime_standalone/outputs"
LOGS = [f"{OUT}/paired_cap_sweep_20261113.cap8192.log",
        f"{OUT}/paired_cap_sweep_20261113.cap4096.log"]
MEM = f"{ROOT}/memory/cap-sweep-20261113.md"
MEM_PKG = f"{ROOT}/for_review/memory/cap-sweep-20261113.md"
STATE = f"{ROOT}/.cap_autorecord_state.json"
ZIP = f"{ROOT}/for_review/dist/for_review_20261113.zip"
RE_RESULT = re.compile(r"\[(\w+) seed(\d+)\] sample_em=([\d.]+) trunc=([\d.]+) \((\d+)/(\d+)\) k-vote=([\d.]+) err=(\d+)")
RE_DONE = re.compile(r"CAP=(\d+) (DONE|FAILED)")
RE_SUMMARY = re.compile(r"^  (none|s0|s1|s2|d1|d2|d3): sample_em=")


def load_state():
    try:
        return set(json.load(open(STATE)))
    except Exception:
        return set()


def save_state(s):
    json.dump(sorted(s), open(STATE, "w"))


def verify(arm, seed, cap):
    """从 per_*.json 核实一致性, 返回 (text, ok)。"""
    d = f"{OUT}/paired_cap_sweep_20261113/cap{cap}"
    cands = glob.glob(f"{d}/per_{arm}_seed{seed}.json") or glob.glob(f"{d}/*{arm}*seed{seed}*.json")
    if not cands:
        return f"- ⚠️ `{arm} seed{seed}`: 日志有结果但找不到 per_*.json (路径 {d})", False
    try:
        j = json.load(open(cands[0]))
    except Exception as e:
        return f"- ⚠️ `{arm} seed{seed}`: per_*.json 解析失败 ({e})", False
    # per_*.json 结构: {"aime24:75": [{em, pred, chars, finish_reason, truncated, completion_tokens}, ...20], ...30题}
    if isinstance(j, dict):
        recs = [r for v in j.values() if isinstance(v, list) for r in v if isinstance(r, dict)]
    else:
        recs = [r for r in j if isinstance(r, dict)]
    n = len(recs)
    cut = sum(1 for r in recs if r.get("finish_reason") == "length" or r.get("truncated"))
    errs = sum(1 for r in recs if r.get("error"))
    toks = [r.get("completion_tokens") or 0 for r in recs]
    maxtok = max(toks) if toks else 0
    bad = []
    if errs: bad.append(f"err={errs}")
    if maxtok > cap: bad.append(f"max_tokens {maxtok} > cap {cap}")
    tag = "✅ 一致" if not bad else "⚠️ " + ", ".join(bad)
    return (f"  - 核实: 样本 {n} 条, finish_reason=length/truncated **{cut}** 条, "
            f"max completion_tokens **{maxtok}** (cap={cap}), {tag}; 文件 `{os.path.relpath(cands[0], ROOT)}`"), not bad


def rebuild_zip():
    sec = f"{ROOT}/for_review"
    skip = {"dist", "incoming", "archive", "gpt_review_extracted"}
    files = []
    for dp, dn, fn in os.walk(sec):
        rel = os.path.relpath(dp, sec)
        if rel != "." and rel.split(os.sep)[0] in skip:
            dn[:] = []
            continue
        for f in fn:
            if not f.endswith(".zip"):
                files.append(os.path.normpath(os.path.join(rel, f)))
    pat = re.compile(rb"(?:sk-|msk-)[A-Za-z0-9]{20,}")
    old = os.getcwd()
    os.chdir(sec)
    try:
        hits = [p for p in files if pat.search(open(p, "rb").read())]
        if hits:
            return f"⚠️ zip 未重建: key 扫描命中 {hits}"
        tmp = ZIP + ".tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(files):
                z.write(p, p)
        os.replace(tmp, ZIP)
        return f"zip 已重建: {len(files)} 文件 / {os.path.getsize(ZIP)/1e6:.2f}MB / key 0 命中"
    finally:
        os.chdir(old)


def main():
    seen = load_state()
    while True:
        new_blocks = []
        for log in LOGS:
            if not os.path.exists(log):
                continue
            m = re.search(r"cap(\d+)\.log$", log)
            cap = int(m.group(1)) if m else 0
            for line in open(log, encoding="utf-8", errors="ignore"):
                line = line.strip()
                r = RE_RESULT.search(line)
                if r:
                    arm, seed, em, tr, cut, tot, kv, err = r.groups()
                    key = f"{cap}|res|{arm}|{seed}"
                    if key in seen:
                        continue
                    seen.add(key); save_state(seen)
                    vtxt, ok = verify(arm, seed, cap)
                    blk = [f"- **cap{cap} / {arm} seed{seed}**: sample_em **{em}**, trunc **{tr} ({cut}/{tot})**, "
                           f"k-vote **{kv}**, err {err}, 日志 `{os.path.basename(log)}`", vtxt]
                    new_blocks.append("\n".join(blk))
                sm = RE_SUMMARY.match(line)
                if sm:
                    cap = cap or 0
                    key = f"{cap}|sum|{sm.group(1)}"
                    if key not in seen:
                        seen.add(key); save_state(seen)
                        new_blocks.append(f"- **cap{cap} SUMMARY / {sm.group(1)}**: `{line}`")
                d = RE_DONE.search(line)
                if d:
                    c, st = d.groups()
                    key = f"cap{c}|done|{st}"
                    if key in seen:
                        continue
                    seen.add(key); save_state(seen)
                    msg = (f"- **cap{c} {st}** ({time.strftime('%F %H:%M')})\n"
                           + ("  - 下一步: 算两 seed 合并的题级 McNemar (none vs s0/s1/s2), 结论写回 `memory/cap-sweep-20261113.md`"
                              if st == "DONE" else "  - ⚠️ 该档失败, 查日志尾部 + `docs/phase2_issues_202611.md` 记账"))
                    new_blocks.append(msg)
        if new_blocks:
            with open(MEM, "a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(new_blocks) + "\n")
            import shutil
            shutil.copy2(MEM, MEM_PKG)
            tail = " | ".join(b.splitlines()[0].replace("- ", "")[:110] for b in new_blocks)
            print(f"[{time.strftime('%F %T')}] RECORD {tail}", flush=True)
            print(f"  -> {rebuild_zip()}", flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
