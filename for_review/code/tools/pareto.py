"""逐题配对: 干净答对长度 delta (相对 none) 与严格 pass delta 的联合分布 + TR/FL/IP 三分类."""
import json, glob, os, statistics as st, random, sys

def load(root):
    A = {}
    for p in sorted(glob.glob(os.path.join(root, "per_*.json"))):
        A[os.path.basename(p)[4:-5]] = json.load(open(p))
    return A

def item_stats(s, strict=True):
    n = len(s)
    pas = [x for x in s if x["em"] and not (strict and x.get("truncated"))]
    fl = [x for x in s if not x["em"] and not x.get("truncated")]
    tr = sum(1 for x in s if x.get("truncated"))
    return dict(n=n, passr=len(pas) / n, trunc=tr,
                L=st.median([x["completion_tokens"] for x in pas]) if pas else None,
                LF=st.median([x["completion_tokens"] for x in fl]) if fl else None)

def boot(per_item, B=4000, seed=0):
    """per_item: list of (dL_pass, dPass) dict rows; bootstrap over items."""
    rng = random.Random(seed); idx = list(range(len(per_item)))
    outL, outP, outW = [], [], []
    for _ in range(B):
        smp = [per_item[rng.choice(idx)] for _ in idx]
        Ls = [r["dL"] for r in smp if r["dL"] is not None]
        outL.append(st.mean(Ls) if Ls else 0.0)
        outP.append(st.mean([r["dP"] for r in smp]))
        outW.append(sum(1 for r in smp if r["cls"] == "win_eff") / len(smp) - sum(1 for r in smp if r["cls"] == "loss_eff") / len(smp))
    def ci(v):
        v = sorted(v); return v[int(.025 * B)], v[int(.975 * B)]
    return ci(outL), ci(outP), ci(outW)

def main(root, base="none_seed1"):
    A = load(root); qids = sorted(A[base])
    for arm in A:
        rows, cls = [], {"TR_pass": 0, "TR_fail": 0, "FL": 0, "IP": 0}
        for q in qids:
            b, c = item_stats(A[base][q]), item_stats(A[arm][q])
            dP = c["passr"] - b["passr"]
            dL = (c["L"] - b["L"]) if (c["L"] and b["L"]) else None
            # 三分类: 纯截断变化(Δpass=0 但 Δtrunc≠0) / 长度-纯(Δpass=0) / 成败相关
            if dP == 0 and c["trunc"] != b["trunc"]:
                k = "TR_pass" if c["trunc"] < b["trunc"] else "TR_fail"
            elif dP == 0:
                k = "FL"
            else:
                k = "IP"
            cls[k] += 1
            # 有效赢: 严格 pass 不降 且 干净答对长度降 >=15%
            win = dP >= 0 and dL is not None and dL <= -0.15 * b["L"]
            loss = dP < 0 and dL is not None and dL >= 0.15 * b["L"]
            rows.append(dict(q=q, dL=dL, dP=dP, cls=k, dtr=c["trunc"] - b["trunc"],
                             win_eff=win, loss_eff=loss, baseL=b["L"], candL=c["L"]))
        Lc, Pc, _ = boot([{**r, "cls": "win_eff" if r["win_eff"] else ("loss_eff" if r["loss_eff"] else "")} for r in rows])
        nw = sum(r["win_eff"] for r in rows); nl = sum(r["loss_eff"] for r in rows)
        dPm = st.mean([r["dP"] for r in rows])
        Lv = [r["dL"] for r in rows if r["dL"] is not None]
        print("%-12s  pass %+.4f [%+.4f,%+.4f]  dL(clean-pass, tok) %+.0f [%+.0f,%+.0f] over n=%d  effwin=%d effloss=%d  || %s" % (
            arm, dPm, Pc[0], Pc[1], st.mean(Lv) if Lv else 0, Lc[0], Lc[1], len(Lv), nw, nl,
            " ".join("%s=%d" % kv for kv in cls.items())))

if __name__ == "__main__":
    main(sys.argv[1])
