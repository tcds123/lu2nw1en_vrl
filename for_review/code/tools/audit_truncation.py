"""截断样本审计 (阶段 0.5): 回答"截断是否等于一定错"这个问题, 而不是假设它.

旧证据的缺陷 (issue #71): `pair` 基线的 24/557 只看了 `em` 字段, 而 `em` 来自
`extract_int_answer`, 后者在缺 \boxed / <answer> 时**取全文最后一个整数**作兜底.
截断的本质就是"答案没写出来", 于是兜底恰好抽到推理中间的小整数 (1/2/3/14...),
偶尔撞上真值 -> 这 24 条不是"截断却答对", 而是"截断 + 兜底抽到残文里的巧合".

本脚本给三态判读, 每题每个截断样本一条:
  A  ANSWER_BEFORE_CUT  截断前已写出答案且有显式标记 (<answer> / \boxed / 尾窗 'Answer: N')
  B  NO_MARKER_AT_CUT   无任何显式标记 (含"A 被误抽"的假阳性)
  C  NO_ANSWER_AT_CUT   连隐式落地都没有 (残文里无真值)
只有 A 才构成"截断却答对"的硬证据; B 里的假阳性必须靠人工抽样复核.

用法: python3 -I audit_truncation.py <per_*.json 或 arm 目录> [--k 20] [--sample 40]
输出: 每臂三态计数 + JSON (含逐条样本的分类与复核抽样), 写 <dir>/truncation_audit.json
局限: per_*.json **不含全文**, 故本脚本以 `pred`(宽松兜底结果) 与"是否小整数"作代理;
      真正的 A/B/C 判读需整批重跑落全文 (阶段 1), 本脚本只做零 API 初筛与问题定型.
"""
import json, glob, os, sys, re, shutil, argparse
from collections import Counter

sys.path.insert(0, '/data/zhuldz/lunwen/for_review/code')
import importlib.util
_spec = importlib.util.spec_from_file_location('m', '/data/zhuldz/lunwen/for_review/code/eval_aime24_multiseed.py')
m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m)

MARK = re.compile(r'(?i)(?:final\s+answer|answer|ans)\s*[\*:_=\s]*\$?\\?(?:boxed\{)?\s*(-?\d{1,4})\b')


def truth_from_uncut(per):
    """从同题未截断答对样本的一致 pred 反推真值 (30/30 题可反推, 见 issue #71)."""
    votes = {}
    for q, samples in per.items():
        for x in samples:
            if not x.get('truncated') and x.get('em'):
                votes.setdefault(q, Counter())[x.get('pred')] += 1
    return {q: c.most_common(1)[0][0] for q, c in votes.items()}


def classify_trunc_sample(x, truth):
    """零 API 代理分类 (per_*.json 无全文)."""
    pred = x.get('pred')
    if pred is None:
        return 'C_NO_ANSWER_AT_CUT' if x.get('chars', 0) < 500 else 'B_NO_MARKER_AT_CUT'
    if truth is not None and pred == truth:
        # 宽松兜底撞上真值: 真实来源未知, 记为待复核的 A 候选
        return 'A_CANDIDATE_NEEDS_TEXT'
    return 'B_NO_MARKER_AT_CUT'


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('target', help='per_*.json 文件或含 per_*.json 的目录')
    ap.add_argument('--out', default='')
    args = ap.parse_args(argv)

    files = [args.target] if args.target.endswith('.json') else sorted(glob.glob(os.path.join(args.target, 'per_*.json')))
    if not files:
        print('没有找到 per_*.json'); return 2
    arms = {os.path.basename(f)[4:-5]: json.load(open(f)) for f in files}
    qids = sorted(next(iter(arms.values())))
    truths = truth_from_uncut(next(iter(arms.values())))
    print('真值反推: %d/%d 题 (来自数组内未截断答对样本的一致 pred)' % (len(truths), len(qids)))

    report = {'truth_source': 'uncut_consistent_pred', 'n_truth': len(truths), 'arms': {}}
    hdr = '%-12s %6s %10s %10s %10s %10s' % ('arm', 'n_tr', 'A候选', 'B无标记', 'C无落地', '小整数pred%')
    print('\n' + hdr)
    for arm, per in arms.items():
        cnt = Counter(); small = 0; tot = 0
        detail = []
        for q in qids:
            for x in per[q]:
                if not x.get('truncated'):
                    continue
                tot += 1
                k = classify_trunc_sample(x, truths.get(q))
                cnt[k] += 1
                if isinstance(x.get('pred'), int) and abs(x['pred']) <= 20:
                    small += 1
                detail.append(dict(qid=q, pred=x.get('pred'), truth=truths.get(q),
                                   tokens=x.get('completion_tokens'), chars=x.get('chars'), cls=k))
        report['arms'][arm] = dict(n_truncated=tot, **{k: cnt[k] for k in
                                ('A_CANDIDATE_NEEDS_TEXT', 'B_NO_MARKER_AT_CUT', 'C_NO_ANSWER_AT_CUT')},
                                small_int_pred=tot and round(small / tot, 3), detail=detail)
        print('%-12s %6d %10d %10d %10d %9.1f%%' % (
            arm, tot, cnt['A_CANDIDATE_NEEDS_TEXT'], cnt['B_NO_MARKER_AT_CUT'], cnt['C_NO_ANSWER_AT_CUT'], 100 * small / tot if tot else 0))

    out = args.out or (os.path.splitext(files[0])[0] + '.truncation_audit.json' if len(files) == 1
                       else os.path.join(args.target, 'truncation_audit.json'))
    json.dump(report, open(out, 'w'), ensure_ascii=False, indent=1)
    print('\n写出:', out)
    print('\n注意: 无全文时 "A候选" 无法与 "兜底撞值" 区分. 阶段 1 的 pilot 必须落全文,')
    print('      并加 extract_strict() (禁用全文最后整数兜底) 后重算, 才能给出 A/B/C 硬结论.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
