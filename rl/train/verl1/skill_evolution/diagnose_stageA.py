#!/usr/bin/env python
# 阶段A P0-2 + P0-3配 Bonferroni: Skill 累积对比 + 多重检验敏感性 (2026-11-27, 零 API)
import json, os, sys, hashlib

RUN_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/livemath_fullrun_v32_20261126_161211/run_20261126_161212"
sys.path.insert(0, "/data/zhuldz/lunwen")
sys.path.insert(0, "/data/zhuldz/skillopt")
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ============ P0-2: 三版本 Skill 文本累积对比 ============
print("=" * 72)
print("P0-2: Skill 累积对比 (S₀ → iter2 Hypotheses → iter6 Option Comparison)")
print("=" * 72)
s0 = open("/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md", encoding="utf-8").read().strip()
best = open(os.path.join(RUN_DIR, "best_skill_evolved.md"), encoding="utf-8").read().strip()

# 从 archive_iter2.json 找 iter2 Hypotheses 候选 (hash 02de0a03)
iter2_skill = None
for arch in [f"archive_iter{i}.json" for i in range(2, 7)]:
    try:
        data = json.load(open(os.path.join(RUN_DIR, arch), encoding="utf-8"))
        for node in data:
            if md5(node.get("skill", "")) == "02de0a03":
                iter2_skill = node["skill"]
                print(f"iter2 Hypotheses 找到于 {arch}")
                break
        if iter2_skill:
            break
    except Exception:
        pass

print(f"\nS₀:        {len(s0)} chars md5={md5(s0)}")
print(f"iter2 Hyp: {len(iter2_skill)} chars md5={md5(iter2_skill)}" if iter2_skill else "iter2 未找到")
print(f"iter6 best:{len(best)} chars md5={md5(best)}")

# 按 section 分解对比
def split_sections(text):
    """按 markdown ## 分段, 返回 {标题: 内容}"""
    secs = {}
    cur = "_header_"
    buf = []
    for line in text.split("\n"):
        if line.strip().startswith("##"):
            secs[cur] = "\n".join(buf).strip()
            cur = line.strip().lstrip("#").strip()
            buf = []
        else:
            buf.append(line)
    secs[cur] = "\n".join(buf).strip()
    return secs

s0_secs = split_sections(s0)
iter2_secs = split_sections(iter2_skill) if iter2_skill else {}
best_secs = split_sections(best)
print(f"\nS₀ sections: {list(s0_secs.keys())}")
print(f"iter2 sections: {list(iter2_secs.keys())}")
print(f"best sections: {list(best_secs.keys())}")

# 追踪 iter2 的有效策略是否在 iter6 保留
print(f"\n--- 策略累积追踪 ---")
if iter2_secs:
    iter2_titles = [k for k in iter2_secs if k != "_header_"]
    for title in iter2_titles:
        in_best = title in best_secs
        same_text = (iter2_secs[title] == best_secs.get(title, "")) if in_best else False
        # 内容相似度 (字符级 Jaccard 粗估)
        if in_best and not same_text:
            a, b = set(iter2_secs[title].split()), set(best_secs[title].split())
            jacc = len(a & b) / len(a | b) if (a | b) else 0
            print(f"  [{title}] 保留(改写): best 含此段, Jaccard={jacc:.2f}")
        elif same_text:
            print(f"  [{title}] 保留(字节一致): iter2 内容原样在 best")
        else:
            print(f"  [{title}] ❌ 丢失: best 无此段")
            # 看是否被合并进别的段
            frag = iter2_secs[title][:60].replace("\n", " ")
            for bt, bv in best_secs.items():
                if frag[:30] in bv:
                    print(f"       → 片段疑似并入 best[{bt}]")
                    break

# best 新增了哪些段 (iter2 没有)
print(f"\n--- iter6 相对 iter2 的新增/变化 ---")
for title in best_secs:
    if title == "_header_":
        continue
    if title not in iter2_secs:
        if title in s0_secs:
            print(f"  [{title}] 来自 S₀ (iter2 删过, iter6 恢复或保留)")
        else:
            print(f"  [{title}] ✨ iter6 新增段 (S₀/iter2 都没有)")
    elif iter2_secs[title] != best_secs[title]:
        a, b = set(iter2_secs[title].split()), set(best_secs[title].split())
        jacc = len(a & b) / len(a | b) if (a | b) else 0
        print(f"  [{title}] 改写: iter2→best Jaccard={jacc:.2f} (iter2 {len(iter2_secs[title])}c → best {len(best_secs[title])}c)")

# ============ P0-3 配: Bonferroni 敏感性 ============
print()
print("=" * 72)
print("P0-3 配: Bonferroni 多重检验敏感性 (meta + not_meta 两子集)")
print("=" * 72)
import numpy as np
from tf_global_v2_math import load_split, is_meta_item
import skill_evolver_math as se

test_items = load_split(se.SPLIT_DEFAULT, "test", shuffle_seed=0)
res = json.load(open(os.path.join(RUN_DIR, "L2_L3_test_results.json")))
pid = res["per_item_delta"]
meta_qids = {str(it["id"]) for it in test_items if is_meta_item(it)}
nm_qids = {str(it["id"]) for it in test_items if not is_meta_item(it)}

def bootstrap_ci_lower_custom(per_item, n_boot=10000, alpha=0.05, seed=42):
    """自实现 bootstrap: 题级重采样, 返回 (1-alpha) 分位下界. alpha=0.05→95%, 0.0125→98.75%(Bonferroni)"""
    rng = np.random.RandomState(seed)
    qids = list(per_item.keys())
    means = per_item  # {qid: [d]}, 每题先平均
    per_q = [sum(means[q]) / len(means[q]) for q in qids]
    n = len(per_q)
    if n == 0:
        return 0.0
    boots = np.empty(n_boot)
    arr = np.array(per_q)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        boots[i] = arr[idx].mean()
    lo = alpha / 2
    return float(np.percentile(boots, lo * 100))

meta_pid = {q: pid[q] for q in meta_qids if q in pid}
nm_pid = {q: pid[q] for q in nm_qids if q in pid}

print(f"子集: meta(n={len(meta_pid)}) not_meta(n={len(nm_pid)})")
print(f"\n名义 95% (α=0.05, 当前落盘口径):")
print(f"  meta    CI_lower={bootstrap_ci_lower_custom(meta_pid, alpha=0.05):+.4f}  (落盘={res['L3']['meta']['ci_lower']:+.4f})")
print(f"  not_meta CI_lower={bootstrap_ci_lower_custom(nm_pid, alpha=0.05):+.4f}  (落盘={res['L3']['not_meta']['ci_lower']:+.4f})")
print(f"\nBonferroni 校正 (2 子集, α=0.025, 即 98.75% 分位下界):")
meta_bonf = bootstrap_ci_lower_custom(meta_pid, alpha=0.025)
nm_bonf = bootstrap_ci_lower_custom(nm_pid, alpha=0.025)
print(f"  meta    CI_lower={meta_bonf:+.4f}  {'✅ 校正后仍>0' if meta_bonf > 0 else '🔴 校正后跨零'}")
print(f"  not_meta CI_lower={nm_bonf:+.4f}  {'✅ 校正后>0' if nm_bonf > 0 else '校正后跨零 (与名义一致)'}")
print(f"\n结论: L3-meta 名义 +0.0362 → Bonferroni {meta_bonf:+.4f}. ", end="")
if meta_bonf > 0:
    print("校正后仍显著正向 → meta 子集增益对多重检验稳健 ✅")
else:
    print(f"校正后跨零 → 名义显著性对多重检验不稳健, 论文应降级为'边际显著'")

# 同时报 L2 总体 (单检验, 无需校正)
l2_bonf = bootstrap_ci_lower_custom(pid, alpha=0.05)
print(f"\nL2 总体 (单检验, 无需 Bonferroni): CI_lower={l2_bonf:+.4f} (落盘={res['L2']['ci_lower']:+.4f})")
