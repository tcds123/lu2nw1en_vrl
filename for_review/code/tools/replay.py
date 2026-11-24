"""零 API 回放: 从 5 臂 per_*.json 复算三态双轴 (pass / trunc / 干净答对长度)."""
import json, glob, os, statistics as st, sys

def replay(d, strict=True):
    """d: {qid: [sample]} -> 三态指标."""
    clean_pass_len, clean_fail_len, n_pass_loose, n_pass_strict = [], [], 0, 0
    n_s = n_tr = 0
    for qid, s in d.items():
        for x in s:
            n_s += 1
            tr = bool(x.get("truncated"))
            n_tr += tr
            if x.get("em"):
                n_pass_loose += 1
                if not (strict and tr):
                    n_pass_strict += 1
            if not tr and x.get("completion_tokens"):
                (clean_pass_len if x["em"] else clean_fail_len).append(x["completion_tokens"])
    return dict(n=n_s, trunc=n_tr, trunc_rate=n_tr / n_s,
                pass_loose=n_pass_loose / n_s,
                pass_strict=n_pass_strict / n_s,
                clean_pass_len=st.mean(clean_pass_len) if clean_pass_len else 0.0,
                clean_fail_len=st.mean(clean_fail_len) if clean_fail_len else 0.0)

def main(root):
    arms = {}
    for p in sorted(glob.glob(os.path.join(root, "per_*.json"))):
        arms[os.path.basename(p)[4:-5]] = json.load(open(p))
    print("%-12s %5s %7s %8s %8s %9s %10s" % ("arm", "n", "trunc%", "pass_loose", "pass_strict", "len_passtok", "len_failtok"))
    rows = {}
    for a, d in arms.items():
        r = replay(d); rows[a] = r
        print("%-12s %5d %6.2f%% %8.4f %8.4f %9.0f %10.0f" % (a, r["n"], 100 * r["trunc_rate"], r["pass_loose"], r["pass_strict"], r["clean_pass_len"], r["clean_fail_len"]))
    return rows

if __name__ == "__main__":
    main(sys.argv[1])
