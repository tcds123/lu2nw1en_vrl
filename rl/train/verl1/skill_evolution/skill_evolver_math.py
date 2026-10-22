#!/usr/bin/env python3
"""Skill 自进化迭代器 (免训练, 目标超 SkillOpt-best 0.5161).

以 v2 当前 skill 为起点迭代 (不回 S₀)。融合 SkillOpt + GEPA + ReSkill 失败驱动, 免训练:
  - 机制1 候选种群 + 帕累托选择 (GEPA): 每轮 K 候选, val bootstrap CI 选前沿
  - 机制2 失败驱动局部 patch (SkillOpt 贪婪编辑 + ReSkill 失败归因): 只做 add/replace 小改,
        patch 锚定当前 skill 仍错题的具体失败, 产出具体决策程序非通用 checklist; 拒绝缓冲区
  - 机制3 慢更新 (SkillOpt epoch-level): 快周期局部 patch + 每 N 轮蒸馏整合去冗余
  - 机制4 防退化 + LOST 监控 + 多样性 (轮换失败类型侧重)
  - 机制5 test-free 选择, 最终 test 验证

与 v2 区别: v2 整段追加节重写 + 单候选 + 通用 checklist; 本器局部编辑 + 多候选 + 具体决策程序.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)
_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)

os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_API_KEY", "")  # 由环境变量提供 key
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1")
os.environ.setdefault("QWEN_CHAT_MODEL", "/model/DeepSeek-V3.2")
os.environ.setdefault("QWEN_CHAT_AIGC_USER", "zhuldz")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "4096")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "28")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

from skillopt.model import chat_target  # noqa: E402

# 复用 v2 已修好的评测机制 (k 投票连续 pass_rate + set_temp + 固定 s0_cache 快照基线).
# evolver 本地曾用 val_passes=1 单次 0/1 采样, 致同 skill 多次评测 delta 抖动 ±0.06~0.13,
# 叠加保底回退退化, 8 轮原地踏步被误判 improved (run_20260727_163814 best==init). 详见
# docs/phase2_issues.md 末条 + memory/evolver-unchanged-skill-bug-20260729.md.
from tf_global_v2_math import (  # noqa: E402
    run_one, eval_on, precompute_s0, bootstrap_ci_lower,
    per_item_counts, beta_binomial_lcb,
    set_temp, EVAL_TEMP, GEN_TEMP, load_split, md5,
    load_s0_cache_with_manifest, check_skill_leakage, is_meta_item,
)

S0_DEFAULT = "/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md"
# 起点改为与 multiseed 的 v2 baseline 同源 (run_20260728_131934), 统一对比基准.
V2_SKILL_DEFAULT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2/run_20260728_131934/tf_global_skill_v2_math.md"
# split 统一为 1-3-6 (train18/val53/test106), 与 v2/multiseed 对齐, select=train+val=71,
# 可复用 s0_cache_math_1-3-6_k5.json (k=5 连续值). evolver 产物可直接进 test=106 复评.
SPLIT_DEFAULT = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42"
S0_CACHE_DEFAULT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json"
OUT_DIR_DEFAULT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math"

# M3-gate 容差: consolidate 后 re-eval, 若 delta 下降超此阈值则回退原 skill (治盲压缩损质).
# 设 0.03 ≈ eval_k=3 下 delta 采样噪声的一半, 容忍噪声抖动但拦截真实退化.
GATE_MARGIN = 0.03

# eval_on 整批墙钟上限 (秒). 正常 ~24-71 题 k=3 在 28RPM 下 ~3-6min; 900s 留 3x 余量.
# 防止 urllib socket.read 无超时卡死整批 (PID 53188 死锁根因): 超时未完成 item 计 0 并告警, 不阻塞.
EVAL_DEADLINE_S = 900

# 失败类型轮换 (覆盖长尾)
FAILURE_FOCI = [
    "Implication vs biconditional (one-way vs iff)",
    "Existence vs uniqueness",
    "Quantifier order and parameter dependence (∀∃ vs ∃∀, uniform vs dependent bounds)",
    "Regularity / function-space strength (C^{k+1} ⊂ C^k, higher regularity claims)",
    "Equality case / sharp constant / exact formula vs inequality",
    "Meta-options ('a stronger result can be proven') decision procedure",
    "Domain / scope restriction (interior vs closure, restricted subfamily)",
    "Classification completeness (all divisors vs exactly two possibilities)",
]

# 算子H (strengthen) 总开关: 由 main() 的 --strengthen flag 置 True. 默认 False = 字节复现既有 run.
# 影响两处协同行为:
#   (1) classify_failure_type: 开关 ON 时 meta 检查后移 (见函数内 _STRENGTHEN_ON 分支), 让含
#       unique/exist 关键词的 meta-框架题按关键词归类为 existence/uniqueness, 而非被 meta 早期返回吞掉
#       (84% meta 的 test set 下, existence/uniqueness 失败此前全被 foci[5] 吸收, expand 显式跳过 meta).
#   (2) expand 饱和分支 (target is None 且仍有失败): 开关 ON 时触发 strengthen 算子强化现有段.
_STRENGTHEN_ON = False


def llm(system: str, user: str, *, max_tokens=4096, stage="evolver") -> str:
    """优化端生成 (patch/consolidate). 采样温度由调用方在阶段入口 set_temp(GEN_TEMP) 设定;
    chat_target 不接受 temperature 参数 (历史 temperature 形参已删, 同 v2 bug 修复)."""
    resp, _ = chat_target(
        system=system, user=user,
        max_completion_tokens=max_tokens, retries=5, stage=stage,
        reasoning_effort=None, timeout=300,
    )
    return resp


PATCHER_SYSTEM = (
    "You are an expert System Prompt Engineer for a mathematical-reasoning LLM solving theorem-level "
    "multiple-choice questions (LiveMathematicianBench). You make SMALL, SURGICAL edits to an existing "
    "skill to fix SPECIFIC failure modes — you do NOT rewrite it. You add concrete, operational "
    "DECISION PROCEDURES (not generic checklists) anchored to real errors."
)


def _format_choices(choices):
    return "\n".join(f"{c['label']}. {c['text']}" for c in choices)


def _excerpt(text: str, n: int = 700) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[:n] + " …[truncated]"


def clean_skill(s: str, fallback: str) -> str:
    s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL).strip()
    s = re.sub(r'```markdown\s*', '', s).strip()
    s = re.sub(r'```\s*$', '', s).strip()
    s = s.split("## YOUR OUTPUT")[-1].strip() if "## YOUR OUTPUT" in s else s
    return s if s else fallback


# ===================================================================
# 机制2: 失败驱动局部 patch
# ===================================================================
# 数据驱动 focus (2026-08-17, Task3): 把已诊断失败按类型分类, 选多数类型对应 focus,
# 让 patcher 对真正主导的失败模式写决策程序. 根因: collect_failures focus 无关, 总返回
# 最难 n_diag 个失败 (val 最难题 ~93% 是 meta-option); 原 round-robin focus 文本在早 iter
# 是 "Implication/Biconditional"/"Existence/Uniqueness", 让 patcher 用错方向 patch 这些
# meta-option 失败 (写蕴含/存在性决策程序而非 "strongest provable statement" 程序).
# 签名来源: memory/skillopt-headtohead-start-20260803.md:232 (union=44/53 val validated).
_META_Q_SIG = ("strongest", "can be proved", "can be proven",
               "a stronger result", "a stronger statement")
_META_OPT_SIG = ("one of the remaining",)


def classify_failure_type(failure: dict) -> str:
    """对单个失败 (collect_failures 产物) 分类, 返回对应 FAILURE_FOCI label.
    meta-option: 题面含 strongest/can be proved|proven/a stronger result|statement,
    或某选项含 'one of the remaining'. 否则按关键词映射最近 focus; 歧义回退 foci[5]
    (主导未覆盖 mass, 与诊断根因一致).

    _STRENGTHEN_ON (算子H) 模式: meta 检查后移到关键词梯之后. 84%-meta 的 test set 下, 含
    unique/exist 关键词的 meta-框架题此前被 meta 早期返回吞进 foci[5] (expand 显式跳过 meta),
    existence/uniqueness 失败永远看不到 → 永不触发第 2 对双向测试轴. 后移让这类题按关键词归
    existence/uniqueness, 使 find_uncovered_type_for_expand 与 strengthen 都能看到第 2 轴."""
    it = failure.get("item", {}) or {}
    q = (str(it.get("question", "")) + " " + str(it.get("theorem", ""))).lower()
    choices = it.get("choices") or []
    opt_text = " ".join(str(c.get("text", "")) for c in choices).lower()
    is_meta = (any(sig in q for sig in _META_Q_SIG) or any(sig in opt_text for sig in _META_OPT_SIG))

    # 非 meta: 关键词 -> 最近 focus (顺序对齐 FAILURE_FOCI). strengthen 模式下也先跑, 让关键词优先于 meta.
    if not _STRENGTHEN_ON:
        # 旧行为 (字节复现): meta 早期返回, 其后才跑关键词梯
        if is_meta:
            return FAILURE_FOCI[5]
    if any(kw in q for kw in ("iff", "biconditional", "if and only if", "characteriz")):
        return FAILURE_FOCI[0]
    if any(kw in q for kw in ("unique", "uniqueness", "at most one", "exist")):
        return FAILURE_FOCI[1]
    if any(kw in q for kw in ("for all", "for every", "∀", "∃", "uniform", "parameter")):
        return FAILURE_FOCI[2]
    if any(kw in q for kw in ("regularit", "c^", "smooth", "continu", "differentiab")):
        return FAILURE_FOCI[3]
    if any(kw in q for kw in ("sharp", "equality case", "exact formula", "best constant")):
        return FAILURE_FOCI[4]
    if any(kw in q for kw in ("domain", "interior", "closure", "restrict", "scope")):
        return FAILURE_FOCI[6]
    if any(kw in q for kw in ("divisor", "classif", "all cases", "two possibilit")):
        return FAILURE_FOCI[7]
    # 歧义回退: meta-option 是主导未覆盖 mass (93% 难题), 默认往该方向 patch.
    # strengthen 模式下: 含 meta 信号但无任何具体关键词的题仍回退 meta (避免无锚点误归).
    if is_meta:
        return FAILURE_FOCI[5]
    return FAILURE_FOCI[5]


def classify_item_meta(item):
    """has_meta 标签 (零 API, 确定性). 复用 recompute_has_meta_router.py 的签名口径.

    返回 True = meta-option 题 (问"最强可证陈述" / 选项含 'one of the remaining').
    break-③ 用此把 probe_items 切成 meta/not-meta 两子集, 让 became_best 看 per-subset 信号
    (而非整体单目标 — 整体被 meta 84% 权重淹没, not-meta 的增益被掩盖, 见 expand-v2 结果).
    """
    q = (str(item.get("question", "")) + " " + str(item.get("theorem", ""))).lower()
    choices = item.get("choices") or []
    opt_text = " ".join(str(c.get("text", "")) for c in choices).lower() if choices else ""
    return any(sig in q for sig in _META_Q_SIG) or any(sig in opt_text for sig in _META_OPT_SIG)


# ===================================================================
# 候选池准入 (fix-①, 2026-09-15) 尖峰关的维度分峰
# ===================================================================
# 分峰教训 ([[pool-admission-8dim-spike-split-20260915]]): classify_failure_type 里 meta 优先级
# 在内容关键词之前就把 meta 题吞进 [5] → [0]~[4] 全空 ([5]=90%). meta-option 是"题干/选项结构
# 特征", 其余 7 维是"数学内容关键词", 两级不同层不能混在一个梯. 故尖峰关用独立的内容优先梯:
# 内容关键词先匹配, meta 仅在无任何内容关键词命中时回退 [5] (结构维).
_CONTENT_KW = (
    (0, ("iff", "biconditional", "if and only if", "characteriz")),
    (1, ("unique", "uniqueness", "at most one", "exist")),
    (2, ("for all", "for every", "∀", "∃", "uniform", "parameter")),
    (3, ("regularit", "c^", "smooth", "continu", "differentiab")),
    (4, ("sharp", "equality case", "exact formula", "best constant")),
    (6, ("domain", "interior", "closure", "restrict", "scope")),
    (7, ("divisor", "classif", "all cases", "two possibilit")),
)


def classify_item_dim(item):
    """内容优先维度分类 (尖峰关专用), 返回 FAILURE_FOCI 的 dim index.

    docstring 语义: 内容关键词 (0,1,2,3,4,6,7) 先匹配 (首命中), meta (结构维 5) 只在
    无任何内容关键词命中时回退. 与 classify_failure_type 的区别: 不提 meta 优先级吞并.
    """
    q = (str(item.get("question", "")) + " " + str(item.get("theorem", ""))).lower()
    for dim, kws in _CONTENT_KW:
        if any(kw in q for kw in kws):
            return dim
    return 5  # 无内容关键词 → meta 结构维 (回退)


# 5-峰合并 ([[pool-admission-8dim-spike-split-20260915]]): 稀疏维 [3]=5/[4]=0/[6]=2/[7]=3 单维撑不起
# bootstrap CI, 合并为"稀疏簇"一峰. 最终 5 峰:
#   0=implication(14) 1=existence(21) 2=quantifier(17) SPARSE=3+4+6+7(10) META=5(9)
_SPARSE_DIMS = (3, 4, 6, 7)
_DIM_TO_PEAK = {0: 0, 1: 1, 2: 2, 5: "meta"}
for _d in _SPARSE_DIMS:
    _DIM_TO_PEAK[_d] = "sparse"
_FIVE_PEAKS = ((0, "implication"), (1, "existence"), (2, "quantifier"),
               ("sparse", "sparse"), ("meta", "meta"))


def git_diff_similarity(a, b):
    """多样性关: git-diff 归一化文本相似度 (去一致后只看 changed 位置).

    对齐用户要求「不是对整段做多样性, 而是对 skill 位置信息, 去除一致之后类似 git diff 再算相似度」.
    用 SequenceMatcher.get_opcodes() 找出 replace/insert/delete 行 (changed), 相似度只在 changed
    位置上算: 若几乎无 changed (文本近同) → 高相似 (趋 1) = 低多样性; 有实质 diff → 低相似 = 高多样性.
    落下: ratio = len(changed 行) / len(总行) 的互补, 或返回 opcodes 里等同行占比. 返回 [0,1], 1=完全相同.
    """
    from difflib import SequenceMatcher
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    if not a_lines and not b_lines:
        return 1.0
    sm = SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    equal_chars = 0
    total_a = sum(len(l) for l in a_lines)
    total_b = sum(len(l) for l in b_lines)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            equal_chars += sum(len(l) for l in a_lines[i1:i2])
    return equal_chars / max(1, max(total_a, total_b))


def spike_peak_hit(per_item, items, rng=None):
    """尖峰关: 5 峰中是否 ∃ 一峰 per_d CI 下界 > 0. 返回 (hit: bool, hit_peak: str|None, detail).

    分峰见 _FIVE_PEAKS / _DIM_TO_PEAK. per_item: {qid:[d]}; items: 含 id 的题集. CI=固定 seed 42
    bootstrap 2.5% 下界; n<2 的峰跳过 (算不出 CI). 稀疏维已聚合 (3+4+6+7) 到 n=10 可算.
    """
    import numpy as _np
    if rng is None:
        rng = _np.random.default_rng(42)
    qid_by_peak = {0: [], 1: [], 2: [], "sparse": [], "meta": []}
    for it in items:
        qid = str(it["id"])
        if qid not in per_item:
            continue
        qid_by_peak[_DIM_TO_PEAK[classify_item_dim(it)]].append(qid)
    detail = {}
    for peak_key, _label in _FIVE_PEAKS:
        qids = qid_by_peak.get(peak_key, [])
        vals = []
        for q in qids:
            v = per_item[q]
            vals.append(v[0] if isinstance(v, list) and v else v)
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            detail[str(peak_key)] = (0.0, len(vals))
            continue
        arr = _np.asarray(vals, dtype=float)
        samples = rng.choice(arr, size=(1000, len(arr)), replace=True).mean(axis=1)
        ci_lower = float(_np.percentile(samples, 2.5))
        detail[str(peak_key)] = (ci_lower, len(vals))
        if ci_lower > 0:
            return True, str(peak_key), detail
    return False, None, detail


def admission_filter(child_skill, parent_skill, child_delta, per_item, items, eps=0.02):
    """fix-① 候选池准入三关 (--pool_filter): 只入池两类 — 次优但文本差异大 / 某指标尖峰.
    但用户硬约束: 平均不回落 (不能为一点提升让平均明显回落).

      ① 质量关 (平均不回落): child_delta >= -eps.
      ② 尖峰关: 5 峰中 ∃ 一峰 CI 下界 > 0 (显著增益), 否则看③.
      ③ 多样性关: git_diff_similarity(parent_skill, child_skill) < 0.95 (实质 diff).
      ② 或 ③ 任一成立即过 (尖峰/多样性二选一), 但 ① 必过.
    返回 (admitted: bool, reason: str). 未开 --pool_filter 时不调用 (字节复现 theo_iter6).
    """
    if child_delta < -eps:
        return False, f"quality(Δ={child_delta:+.3f}<-{eps})"
    spike, hit_peak, _ = spike_peak_hit(per_item, items)
    if spike:
        return True, f"spike@{hit_peak}"
    sim = git_diff_similarity(parent_skill, child_skill)
    if sim < 0.95:
        return True, f"diverse(git-diff sim={sim:.3f})"
    return False, f"no-spike-no-diverse(sim={sim:.3f})"


class _ArchiveNode:
    __slots__ = ("skill", "slots", "facet_label", "parent", "val_delta", "val_ci")

    def __init__(self, skill="", slots=None, facet_label=None, val_delta=0.0, val_ci=0.0):
        self.skill = skill
        self.slots = slots or {}
        self.facet_label = facet_label
        self.parent = None
        self.val_delta = val_delta
        self.val_ci = val_ci


# ===================================================================
# 算子触发计数 (F08 / RF Q6 / RF §6 Step 3, 2026-09-18)
# ===================================================================
# review 要求: 每轮每算子记录 attempted/guard_rejected/evaluated/admitted/promoted,
#   pool_filter 分关计数, 写入 evolver_log.json; 开启的算子全程 0 次生效就告警.
# 零 API 零回归: 纯插桩 (不改任何 promotion/best/admission 语义).
# 旧版事后从 tree_iter{t}.json + iters 数组重建 (见 [[operator-trigger-count-onarm-20260918]]),
#   生产 evolver 原无运行时全算子计数 — 本次补齐.
# 算子集 = facet_edit / crossover / overlay / expand / strengthen (与 RF Step 3 五算子一致).
_OPERATOR_NAMES = ("facet_edit", "crossover", "overlay", "expand", "strengthen")


def _new_operator_counts():
    """每轮每算子的计数器. 返回 dict[operator] = {attempted, guard_rejected, evaluated, admitted, promoted}.

    语义 (对齐 RF Step 3):
      attempted       — 本算子本轮尝试发起的次数 (进入算子主体, LLM/edit 调用前)
      guard_rejected  — 结构/温度/merge guard 拒掉 (未进 eval, 非"价值不够"拒绝)
      evaluated       — 进 eval_on 算出 pass/delta/ci 的次数
      admitted        — 过准入门进 archive 的次数 (含未成 best 的 keep-losers)
      promoted        — 成为本轮/历史全局 best 的次数 (TREE-BEST / *-BEST 命中)
    pool_filter 是 gate 不是算子, 单独记 per-gate: {quality, spike, diverse} × {pass, reject}.
    """
    return {op: {"attempted": 0, "guard_rejected": 0, "evaluated": 0,
                 "admitted": 0, "promoted": 0} for op in _OPERATOR_NAMES}


def _accumulate_counts(run_total, iter_counts):
    """把本轮 iter_counts 累加进 run_total (跨轮汇总, 供终局 0-promoted 告警)."""
    for op, d in iter_counts.items():
        for k in ("attempted", "guard_rejected", "evaluated", "admitted", "promoted"):
            run_total[op][k] += d.get(k, 0)


def _zero_promoted_alert(run_total, args):
    """RF §6 Step 3: 开启的算子全程 0 promoted 就告警 (算子静默失效的运行时检测)."""
    active_ops = []
    if getattr(args, "facet_tree", False):
        active_ops.append("facet_edit")
        if getattr(args, "crossover", True):
            active_ops.append("crossover")
        if getattr(args, "overlay", True):
            active_ops.append("overlay")
        if getattr(args, "expand", False):
            active_ops.append("expand")
        if getattr(args, "strengthen", False):
            active_ops.append("strengthen")
    alerts = []
    for op in active_ops:
        d = run_total.get(op, {})
        if d.get("attempted", 0) > 0 and d.get("promoted", 0) == 0:
            alerts.append(f"{op}(attempted={d['attempted']}, promoted=0)")
    return alerts


def _new_pool_filter_gates():
    """pool_filter 三关 per-gate 计数器 (gate 非算子, 单独记). 返回 {gate: {pass, reject}}.
    gate 评估短路顺序 = quality → spike → diverse (admission_filter 内同序):
      quality 不过则短路拒 (spike/diverse 不评); spike 过则 admit (diverse 不评)."""
    return {g: {"pass": 0, "reject": 0} for g in ("quality", "spike", "diverse")}


def _tally_pool_filter(reason, admitted, gates):
    """解析 admission_filter 返回的 reason 字符串, 累计 per-gate pass/reject.
    零回归: 只读 reason 字符串, 不改 admission_filter 任何行为.
    reason 形态 (admission_filter L309-317):
      "quality(Δ=...)"        → quality:reject (短路, spike/diverse 未评)
      "spike@{peak}"          → quality:pass + spike:pass (admit, diverse 未评)
      "diverse(git-diff ...)" → quality:pass + spike:fail(reject) + diverse:pass (admit)
      "no-spike-no-diverse…"  → quality:pass + spike:reject + diverse:reject (拒)"""
    if reason.startswith("quality"):
        gates["quality"]["reject"] += 1
    elif reason.startswith("spike"):
        gates["quality"]["pass"] += 1
        gates["spike"]["pass"] += 1
    elif reason.startswith("diverse"):
        gates["quality"]["pass"] += 1
        gates["spike"]["reject"] += 1
        gates["diverse"]["pass"] += 1
    elif reason.startswith("no-spike"):
        gates["quality"]["pass"] += 1
        gates["spike"]["reject"] += 1
        gates["diverse"]["reject"] += 1


def subset_signal(per_item_delta, meta_mask):
    """把 per_item_delta {qid:[d]} 按 meta_mask 切两子集, 返回每子集的 (mean_delta, ci_lower, n).

    用于 break-③ 的 per-subset 准入判据: not-meta 子集增益被整体单目标淹没时, 子集信号能
    让"某子集增益>0 且无子集退步超 ε"的候选被接受 (对齐 expand-v2 的 not-meta +0.157 信号).
    per_item_delta: eval_on 返回的 per_item (每 qid 一个 [d] 列表).
    meta_mask: {qid: bool} 由 classify_item_meta 预算.
    """
    import numpy as _np
    meta_means = [sum(v) / len(v) for q, v in per_item_delta.items()
                  if meta_mask.get(q, False) and v]
    notmeta_means = [sum(v) / len(v) for q, v in per_item_delta.items()
                     if not meta_mask.get(q, True) and v]
    def _ci(ms):
        if len(ms) < 2:
            return float("-inf")
        rng = _np.random.default_rng(42)
        arr = _np.asarray(ms)
        samples = rng.choice(arr, size=(1000, len(arr)), replace=True).mean(axis=1)
        return float(_np.percentile(samples, 2.5))
    return {
        "meta": {"mean": float(_np.mean(meta_means)) if meta_means else 0.0,
                 "ci": _ci(meta_means), "n": len(meta_means)},
        "not_meta": {"mean": float(_np.mean(notmeta_means)) if notmeta_means else 0.0,
                     "ci": _ci(notmeta_means), "n": len(notmeta_means)},
    }


def subset_promote_decision(xitem, best_val_delta, meta_mask, eps=0.02, enabled=True):
    """break-③ per-subset 准入判据: 整体未超 best 时, 看子集信号决定是否晋升为 best.

    判据 (对齐 expand-v2 not-meta +0.157 信号):
      - 某子集 ci_lower > 0 (显著增益) 且
      - 无子集 mean_delta < -eps (无子集显著退步, eps 容忍噪声)
      → 晋升 best (即使整体 delta 未超 best_val_delta).
    enabled=False (--no-subset_admit) 时直接返回 (False, sig), 回退纯整体单目标 (旧行为).
    返回 (should_promote, sig_dict). 整体已超 best 由调用方先判 (本函数只处理"整体未超但子集有信号").
    """
    sig = subset_signal(xitem, meta_mask)
    if not enabled:
        return False, sig
    nm = sig["not_meta"]
    mt = sig["meta"]
    # 主路径: not-meta 子集显著增益 (ci>0) 且 meta 子集未显著退步 (ci >= -eps).
    # 关键: 交叉子集判据必须用 ci (置信下界), 不是 mean — mean 易被少数极端样本拉偏, 而 ci 才是
    # "是否真的有正向效应". Bug C 根因: 旧版用 mt["mean"]>=-eps / nm["mean"]>=-eps 作交叉判据,
    # 导致 not-meta ci 显著为负 (如 -0.08) 但 mean 刚好 >=-eps 的候选被 meta-mirror 路径误救.
    promote = (nm["ci"] > 0) and (nm["n"] >= 1) and (mt["ci"] >= -eps)
    # 镜像备用路径: meta 子集显著增益 且 not-meta 子集未显著退步 (ci 判据).
    # 旧 Bug C: 此处用 nm["mean"]>=-eps → not-meta ci=-0.027/-0.06/-0.08 全部误救 (5/6 误判).
    # 收紧为 nm["ci"]>=-eps: not-meta 必须置信下界不显著为负, 才允许 meta 增益带它一起晋升.
    if not promote and (mt["ci"] > 0) and (mt["n"] >= 1) and (nm["ci"] >= -eps):
        promote = True
    return promote, sig


def lcb_score_from_deltas(per_item, s0_cache, eval_k):
    """药一: 从 eval_on 返回的 per_item delta 分布重建每题 k 投票原始计数, 算 Beta-Binomial LCB delta 分数.

    药一动机 = 治 val→test 负迁移根因① (winner's curse): bootstrap_ci_lower 在 n 小 (probe 53/71 题、
    每题 k=3 投票) 时 CI 下界过窄、低估方差 → 高估 delta 的候选被选为冠军, 迁移到 test 即崩.
    Beta 后验下分位 (Jeffreys 先验) 在 n 小时自然收缩向先验 0.5, 悲观惩罚采样噪声大的候选.

    重建: pass_rate_q = n_pass_q / k, delta_q = pass_rate_q - s0_pass_q
         → n_pass_q = round((delta_q + s0_pass_q) * k)  (恢复正整数计数, 无额外 API)
    per_item: {qid: [d]} (eval_on 返回值); s0_cache: {qid: {"pass_rate": ...}}.
    返回 (lcb_delta, per_lcb_delta): lcb_delta = 全题每题 (25%% 后验通过分位 - s0_pass_q) 的均值,
    是与 val_delta 同量纲的悲观 delta 分数 (对标 val_ci 门控 `>0`, 直接替换 bootstrap CI 下界);
    per_lcb_delta = {qid: q分位 - s0_pass_q} (供诊断/子集信号)."""
    per_lcb_delta = {}
    counts = {}
    for qid, ds in per_item.items():
        if not ds:
            continue
        d = ds[0]
        s0p = s0_cache.get(qid, {}).get("pass_rate", 0.0)
        k = max(1, int(eval_k))
        n_pass = max(0, min(k, int(round((d + s0p) * k))))
        counts[qid] = (n_pass, k)
    _lcb_mean, _per = beta_binomial_lcb(counts, q=0.25)
    per_lcb_delta = {qid: float(_per[qid]) - s0_cache.get(qid, {}).get("pass_rate", 0.0)
                     for qid in _per}
    lcb_delta = float(sum(per_lcb_delta.values()) / len(per_lcb_delta)) if per_lcb_delta else 0.0
    return lcb_delta, per_lcb_delta


# ===================================================================
# Task4 (2026-08-17): 全面诊断 + 覆盖驱动 focus
# 根因精化: Task3 证实诊断机制生效 (val +0.268) 但 val→test 迁移失败 (test 0.4654).
# collect_failures 已在全部 select 题上跑 run_one, 仅截断返回最难 8 个 -> patcher 只看
# val 的 8 个特定失败, 过拟合 val 分布. D: 复用已算结果给 patcher 看全失败分布 (零额外 API),
# 并让 focus 选 "当前 skill 尚未覆盖的类型" (填洞) 而非 "多数类型" (加固).
# 每轮分布 + 覆盖状态持久化到 evolver_log, 作为 C (结构自进化) 的数据基底.
# ===================================================================
def compute_failure_distribution(failures):
    """全失败按类型 tally (复用已算 run_one 结果, 零额外 API). 返回 (counts, reps):
    counts={label:count}; reps={label:[up to 3 failures]} 按 s0_pass 升序=最难优先."""
    counts, reps = {}, {}
    for f in failures:
        lbl = classify_failure_type(f)
        counts[lbl] = counts.get(lbl, 0) + 1
        reps.setdefault(lbl, []).append(f)
    for lbl in reps:
        reps[lbl].sort(key=lambda x: x["s0_pass"])
        reps[lbl] = reps[lbl][:3]
    return counts, reps


# 覆盖检查签名: 当前 skill 文本是否已含该类型的决策程序关键词 (heuristic, 非保证;
# 仅作廉价 gap 探测, fallback-to-majority 在全已覆盖时诚实触发).
# 顺序对齐 FAILURE_FOCI, 让 find_uncovered_type 能按类型定位.
_TYPE_COVERAGE_SIG = {
    FAILURE_FOCI[0]: ("iff", "biconditional", "both direction"),
    FAILURE_FOCI[1]: ("uniqueness", "at most one"),
    FAILURE_FOCI[2]: ("quantifier", "uniform"),
    FAILURE_FOCI[3]: ("regularit", "c^{", "smooth"),
    FAILURE_FOCI[4]: ("sharp", "equality case", "exact formula"),
    FAILURE_FOCI[5]: ("meta", "stronger result", "strongest provable"),
    FAILURE_FOCI[6]: ("domain", "interior", "closure", "restrict"),
    FAILURE_FOCI[7]: ("divisor", "classif", "all cases"),
}


def find_uncovered_type(counts, skill_content):
    """在分布中出现且当前 skill 尚未覆盖 (无签名关键词) 的类型里, 选最高频的一个 (先填最大的洞).
    全已覆盖则返回 None -> 调用方回退多数类型 (Task3 行为)."""
    sc = skill_content.lower()
    for lbl in sorted(counts, key=lambda l: -counts[l]):
        sigs = _TYPE_COVERAGE_SIG.get(lbl, ())
        if sigs and not any(s in sc for s in sigs):
            return lbl
    return None


def find_uncovered_type_for_expand(counts, section_keys):
    """facet-tree expand 专用覆盖选择 (修 v1 根因1 + 落实 §六方向A).

    与 find_uncovered_type (正文关键词匹配, 供扁平主流程) 的两点关键差异:
      1) 显式跳过 meta-option 类型 (FAILURE_FOCI[5]): meta 由 facet_edit/Option 轴负责
         (0.49→0.53 增益路径); expand 攻 meta 会与现有 Option 段内容重叠致 test 退步
         (v1 -0.0700). §六正向信号全在 not-meta 子集 (expand 跨 4 seed 全正).
      2) 段名而非正文匹配 (修 Plan-agent point4): 正文关键词 ('regularity'/'domain' 常作为
         meta-option 程序里的措辞出现) 会误判 COVERED 而饿死 expand. 段名 = 该类型已有专属
         section = 真覆盖, 且与 expand 机制自洽 (expand 加段 → 该类型转 COVERED).

    返回最高频且无专属段 的非-meta 类型; 全已覆盖返回 None → 调用方跳过本代 expand
    (守 §五纪律: 不为造段烧 API)."""
    meta_lbl = FAILURE_FOCI[5]
    key_blob = " | ".join(section_keys).lower()
    for lbl in sorted(counts, key=lambda l: -counts[l]):
        if lbl == meta_lbl:
            continue  # meta 交由 facet_edit 轴; expand 不碰 (避免 v1 内容重叠退步)
        sigs = _TYPE_COVERAGE_SIG.get(lbl, ())
        if sigs and not any(s in key_blob for s in sigs):
            return lbl
    return None


def find_covered_failing_type(counts, section_keys, exclude=None):
    """方案1 (per-type strengthen 触发放宽): 找「已覆盖但仍在失败」的非-meta 类型.

    镜像 find_uncovered_type_for_expand, 但语义相反 — 该类型已有专属 section (签名出现在段名 blob)
    且 counts[lbl] > 0 (仍失败). 用于 expand 仍能找到新未覆盖类型时, 让 strengthen **同时**
    fire 加固既有专段 (而非苦等 target is None 全覆盖, 后者在有限 iter 内不可达 — 见 9/5/9/7 run
    strengthen 全程未触发的结构性根因).

    exclude (too-early-abandonment guard 最小版): 排除本 iter 刚 expand 覆盖的 target 类型 —
    给新段至少 1 iter 的「自然使用」窗口 (避免一覆盖即立刻 strengthen, 用户标注的过早放弃风险).
    传 target → 该类型本轮不参与 strengthen gate.

    判定口径与 find_uncovered_type_for_expand 完全一致 (段名签名 blob, 非正文匹配), 两者互补:
      - 未覆盖       → find_uncovered_type_for_expand 返回该类型, 本函数对它返回 None
      - 已覆盖+仍失败 → 本函数返回该类型, find_uncovered_type_for_expand 对它跳过
    保证同一类型不会同时进 expand 与 strengthen gate (互斥, 不重复消耗 API).

    返回最高频的「已覆盖但仍在失败」非-meta 类型 (排除 exclude); 无则 None."""
    meta_lbl = FAILURE_FOCI[5]
    key_blob = " | ".join(section_keys).lower()
    for lbl in sorted(counts, key=lambda l: -counts[l]):
        if lbl == meta_lbl:
            continue  # meta 不走 strengthen 路径 (与 expand 同口径)
        if lbl == exclude:
            continue  # 本 iter 刚覆盖 → 给新段 1 iter 自然使用窗口
        if counts.get(lbl, 0) <= 0:
            continue  # 该类型无失败 → 无 efficacy gap → 不需加固
        sigs = _TYPE_COVERAGE_SIG.get(lbl, ())
        if sigs and any(s in key_blob for s in sigs):
            return lbl  # 已有专属段 + 仍失败 → per-type strengthen 候选
    return None


def _archive_section_union(archive):
    """方案1 gate 修复 (Task #7): archive 所有节点 section name 的并集 (保序去重).

    背景: 方案1 per-type strengthen 的 gate `find_covered_failing_type` 原传
    `best_node.slots.keys()` 判「已覆盖」. 但 expand 产物若 `became_best=False`
    (EXPAND-SUBSET-GATE: delta < best − floor, 或被更强同 iter 候选夺 best) 则只进
    archive 不进 best_node 链 → 其专属 section (如 Existence) 对 gate 不可见 → gate
    返回 None → strengthen 全程不 fire. 这是 9/8 facet_tree run iter1/iter2 坐实的
    第三种不可达变体 (前两种: 全覆盖前提不可达; per-type 饱和死代码).

    修复语义: 「已覆盖」= 「archive 任一节点曾采纳该类型专属段」(历史组合), 而非
    「当前 best_node 持有该段」. 这落实用户方法论 — "历史中的组合可能会更好":
    archive-only (未夺 best) 的段是合法历史组合, 不应因未夺 best 就对 gate 失明.
    expand 独立性不受影响 (find_uncovered_type_for_expand 仍用 best_node — expand
    只在当前 best 上长段, 不跨历史; strengthen 才跨历史, 因它加固的是既有段).

    返回 list[section_key] (保插入序去重); 空 archive → [].
    """
    seen = []
    for node in archive:
        for k in node.slots.keys():
            if k not in seen:
                seen.append(k)
    return seen


# 算子H (strengthen) 选择: 当 expand 全非-meta 类型已覆盖 (target is None) 但仍有失败 (efficacy gap),
# 选一个现有段做"加固" — 加一条 bullet, 而非长新段. 与 expand (add-only) 互补.
# 核心动机: 干净路径 (theo_iter6) 与 seeded 路径 (evo_seed) 独立收敛到同一"双向测试"启发式, 但干净路径
# 停在 1 轴 (implication↔equivalence), evo_seed 多发展了第 2 轴 (existence↔uniqueness) 并合并进现有段.
# strengthen 让 evolver 把"已含 1 条双向测试程序的段"再补第 2 条 (互补的不同逻辑区分), 关闭这 ~0.5pp gap.
# 双向测试关键词 (section body 中): 已含这类词 = 该段已实现一条具体决策程序 = 加第 2 条的首选候选.
_BIDIRECTIONAL_KW = ("both direction", "iff", "biconditional", "necessary", "sufficient",
                     "forward", "backward", "check both", "verify")


def _section_axis_score(body_lower):
    """段正文里双向测试关键词计数. >=1 = 该段已含一条双向测试决策程序 (加固第 2 轴的首选)."""
    return sum(body_lower.count(kw) for kw in _BIDIRECTIONAL_KW)


def select_strengthen_target(counts, reps, slots, freeze):
    """expand 饱和 (全覆盖但有失败) 时, 选一个现有段做 strengthen 加固.

    classifier-independent 结构选择 (不依赖分类标签 — 饱和事件 counts 常为 {meta-options:16, implication:1},
    meta 16:1 主导, 按"最高频类型"会误选 meta-处理段, 而非第 2 双向轴应去的段):
      1) axis_score = 段正文双向测试关键词数. 已含>=1 的段 = 已有 1 条决策程序 = 加第 2 轴首选 (收敛结构).
      2) relevance_score = 段正文 vs 真实失败题文 (question+theorem) 的词汇重叠 (复用 _TYPE_COVERAGE_SIG 词表).
      选 axis_score 最高 (平则 relevance 最高); 全 0 时退化为 relevance 最高 (早 iter 无双向段时标准内容匹配).

    返回 (target_section_key, target_failures) 或 (None, None).
    target_failures = 跨所有类型的<=3 个最难 reps (展平 reps.values(), 按 s0_pass 升序取 3) —
    LLM 看真实最难失败 (不管分类器把它们埋在哪个桶), 自己能识别被 meta 吞掉的 existence/uniqueness 模式."""
    candidates = {k: v for k, v in slots.items() if k != freeze}
    if not candidates:
        return None, None
    # 真实失败题文词袋 (复用覆盖签名词表作词汇面, 避免纯字面噪声)
    all_reps = [f for reps_list in reps.values() for f in reps_list]
    if not all_reps:
        return None, None
    fail_blob_parts = []
    for f in all_reps:
        it = f.get("item", {}) or {}
        fail_blob_parts.append(str(it.get("question", "")))
        fail_blob_parts.append(str(it.get("theorem", "")))
    fail_blob = " \n ".join(fail_blob_parts).lower()
    # 覆盖签名词表作重叠词汇面
    vocab = set()
    for sigs in _TYPE_COVERAGE_SIG.values():
        for s in sigs:
            vocab.add(s)
    fail_tokens = set(t for t in vocab if t in fail_blob)

    best_key, best_axis, best_rel = None, -1, -1.0
    for sec_key, body in candidates.items():
        body_l = (body or "").lower()
        axis = _section_axis_score(body_l)
        rel = sum(1 for tok in fail_tokens if tok in body_l)
        # 选 axis 最高; 平则 relevance 最高; 再平则取字典序最小 (确定性)
        if (axis > best_axis) or (axis == best_axis and rel > best_rel) or \
           (axis == best_axis and rel == best_rel and (best_key is None or sec_key < best_key)):
            best_key, best_axis, best_rel = sec_key, axis, float(rel)
    # 跨类型最难 reps (s0_pass 升序=最难优先, 取 3)
    hardest = sorted(all_reps, key=lambda x: x.get("s0_pass", 1.0))[:3]
    return best_key, hardest


def collect_failures(val_items, skill_content, s0_cache, n_diag=8, k=1, n_win=4,
                     return_all_failures=False, meta_contrast=False):
    """在 val 上跑当前 skill (k 投票, EVAL_TEMP), 收集仍错题 (含 pred/correct/response/theorem) 做诊断样本.
    pass_rate==0 判定仍错 (k 次全错).
    dir-5 (2026-08-05): 同一 eval pass 顺带收集 wins = skill 解对且 S₀ 非满分的难题 (有正信号),
    供 --learn_positive 正向蒸馏 (从成功解法挖可推广决策程序), 不额外花 API.
    meta_contrast (路径B, 2026-09-10): 同一 eval pass 顺带收集 meta_passes = skill 解对的 meta 题
    (classify_item_meta=True, 不受 s0_pass<1 限制). 用于诊断对照对称化: 让 patcher 看"同是 meta 题,
    做对的选了 meta 选项", 自主归纳 meta⇒答案 规则. 零额外 API (复用已算 results)."""
    set_temp(EVAL_TEMP)
    results = {}
    with ThreadPoolExecutor(max_workers=40) as ex:
        futs = {ex.submit(run_one, it, skill_content, "evolver_diag", k): it for it in val_items}
        for fut in as_completed(futs):
            it = futs[fut]
            results[str(it["id"])] = (it, fut.result())
    failures = []
    wins = []
    meta_passes = []
    for qid, (it, r) in results.items():
        if not r.get("agent_ok"):
            continue
        s0p = s0_cache.get(qid, {}).get("pass_rate", 0.0)
        if r["pass_rate"] == 0:  # 仍错
            failures.append({
                "id": qid, "item": it,
                "pred": r.get("predicted_label", ""),
                "correct": r.get("correct_label", ""),
                "response_excerpt": _excerpt(r.get("response", ""), 600),
                "s0_pass": s0p,
            })
        else:
            # 路径B: skill 解对的 meta 题 (作对照), 独立于 S₀ 满分与否.
            if meta_contrast and classify_item_meta(it):
                meta_passes.append({
                    "id": qid, "item": it,
                    "pred": r.get("predicted_label", ""),
                    "correct": r.get("correct_label", ""),
                    "response_excerpt": _excerpt(r.get("response", ""), 600),
                    "s0_pass": s0p, "skill_pass": r["pass_rate"],
                })
            if r["pass_rate"] > 0 and s0p < 1.0:
                # skill 解对且 S₀ 非满分 = 有正信号 (难题 skill 胜过/补足 S₀)
                gain = r["pass_rate"] - s0p
                if s0p == 0 or gain > 0:  # S₀ 全错的硬题, 或 skill 真增益
                    wins.append({
                        "id": qid, "item": it,
                        "pred": r.get("predicted_label", ""),
                        "correct": r.get("correct_label", ""),
                        "response_excerpt": _excerpt(r.get("response", ""), 600),
                        "s0_pass": s0p, "skill_pass": r["pass_rate"], "gain": gain,
                    })
    # 取最多 n_diag 个, 优先 S0 也错的 (真 headroom)
    failures.sort(key=lambda f: f["s0_pass"])
    # wins 优先 S₀ 全错的硬题 (信息量最大), 再按增益降序
    wins.sort(key=lambda w: (w["s0_pass"], -w["gain"]))
    if return_all_failures:
        # Task4: 返回全失败 (未截断) 供 compute_failure_distribution 看全分布, 零额外 API.
        # 默认关=保主结果复现, 现有调用点仍解 4-tuple 不受影响.
        if meta_contrast:
            # 确定性: meta_passes 按 id 排序做最近邻配对 (配对策略 in make_meta_contrast_diagnostics).
            meta_passes.sort(key=lambda p: str(p["id"]))
            return failures[:n_diag], len(failures), wins[:n_win], list(failures), meta_passes
        return failures[:n_diag], len(failures), wins[:n_win], list(failures)
    return failures[:n_diag], len(failures), wins[:n_win]


PATCH_TEMPLATE = """You are improving a system prompt (skill) for a mathematical-reasoning LLM.

The CURRENT skill is below. It already works well on many problems but STILL FAILS on the diagnostic cases shown. Your job: make SMALL, SURGICAL edits to fix the SPECIFIC failure mode — add a concrete DECISION PROCEDURE (a step-by-step test the model can run), not a generic checklist.

Rules:
1. Do NOT rewrite the skill. Keep its structure. Add at most ONE new short subsection (or replace/extend one existing bullet) that addresses the failure mode.{modify_clause}
2. The new content must be a CONCRETE, OPERATIONAL decision procedure anchored to the diagnostic cases — e.g. "For implication options, test both directions: does P⇒Q and Q⇒P? If only one direction holds, it is not the characterization." NOT "check implications carefully".
3. Keep total skill under ~5000 chars. If near the limit, compress redundant existing bullets rather than appending.
4. Do NOT reference specific problem answers, labels, or this problem's constants.
5. Focus area this round: {focus}
6. Rejected patches to avoid (do not repeat these ideas): {rejected}

## CURRENT SKILL
{skill}

## DIAGNOSTIC FAILURES (current skill still gets these wrong)
{diagnostics}
{positive_section}
{meta_contrast_section}
## YOUR OUTPUT
Output ONLY the edited skill text (full skill with your small edit applied). No meta commentary. START directly with the skill.
"""

# Task4 (2026-08-17): 覆盖驱动 patch 模板. 与 PATCH_TEMPLATE 唯一差别 = 喂全失败分布
# (tally 表 + 未覆盖类型的分层代表样本) 而非仅 8 个最难; focus = 尚未覆盖类型 (填洞) 非多数.
# 治 val 过拟合: patcher 看全分布而非 val 特定 8 题, 减分布偏置. 复用已算 run_one, 零额外 API.
PATCH_TEMPLATE_COVERAGE = """You are improving a system prompt (skill) for a mathematical-reasoning LLM.

The CURRENT skill is below. It already works well on many problems but STILL FAILS on the diagnostic cases shown. Below is the FULL failure-type distribution over the select set (not just the hardest 8), with each type marked COVERED (the skill already has a decision procedure for it) or UNCOVERED (gap to fill). Your job: make SMALL, SURGICAL edits to fix an UNCOVERED failure type — add a concrete DECISION PROCEDURE (a step-by-step test the model can run), not a generic checklist.

Rules:
1. Do NOT rewrite the skill. Keep its structure. Add at most ONE new short subsection (or replace/extend one existing bullet) that addresses the UNCOVERED failure type marked FOCUS.{modify_clause}
2. The new content must be a CONCRETE, OPERATIONAL decision procedure anchored to the representative failures shown — e.g. "For implication options, test both directions: does P⇒Q and Q⇒P? If only one direction holds, it is not the characterization." NOT "check implications carefully".
3. Keep total skill under ~5000 chars. If near the limit, compress redundant existing bullets rather than appending.
4. Do NOT reference specific problem answers, labels, or this problem's constants.
5. Focus area this round (an UNCOVERED gap, not the majority): {focus}
6. Rejected patches to avoid (do not repeat these ideas): {rejected}

## CURRENT SKILL
{skill}

## FULL FAILURE-TYPE DISTRIBUTION (select set; COVERED = already addressed, UNCOVERED = gap)
{tally}

## REPRESENTATIVE FAILURES FOR UNCOVERED TYPES (current skill still gets these wrong)
{stratified}
{positive_section}
{meta_contrast_section}
## YOUR OUTPUT
Output ONLY the edited skill text (full skill with your small edit applied). No meta commentary. START directly with the skill.
"""

# dir-4 (用户提的"允许改删S₀拖后腿内容", 2026-08-05): 默认空串 = 纯追加 (保主结果复现).
# --allow_modify 时注入: 给 patcher 定向删除/替换 *有害* 既有内容的权限 (非整段重写).
# 门控兜底: winner 须 CI>0 + LOST 监控 (pass < base-0.06 即拒) + rejected 缓冲回灌,
# 故有害删除会被拒并记入 rejected, 不会污染 best_skill. 与 M3 (去冗余) 互补: M3 删冗余, dir-4 删有害.
MODIFY_CLAUSE = (
    "\n   You MAY also DELETE or REPLACE a specific existing subsection/bullet that is HARMFUL "
    "- i.e. whose presence CAUSES the diagnostic failures (misleads the model, conflicts with a "
    "correct procedure, or pushes toward the wrong answer). Each such edit must: (a) target a "
    "SPECIFIC identified harmful passage, (b) keep every concrete decision procedure that still "
    "helps, (c) stay smaller than a wholesale rewrite. Prefer targeted deletion over append WHEN "
    "existing content actively causes the failure; otherwise still append."
)

# dir-5 (用户提的"从正确题学习", 2026-08-05): 默认空串 = 只从错题学 (保主结果复现).
# --learn_positive 时注入 POSITIVE EXAMPLES 节: 把 skill 解对的难题 (S₀ 非满分) 喂给 patcher,
# 让它从 *成功解法* 抽象可推广决策程序 (正信号), 与失败模式 (负信号) 互补.
# 关键纪律: 抽象可推广程序, 不抄具体解; 聚焦难题 (S₀ 全错的才有信息量, 已在 collect_failures 排序保证).
POSITIVE_SECTION = """
## POSITIVE EXAMPLES (hard questions the skill solves CORRECTLY - mine the winning procedure)
{wins}

The skill gets these HARD questions right (where baseline S₀ fails). The reasoning excerpt shows the CORRECT path. If a GENERALIZABLE decision procedure (the reusable step that made it work) is NOT already explicit in the skill, add it as a short new subsection so similar questions benefit. Abstract the reusable procedure - do NOT copy the specific solution or reference these problems' constants. If the winning approach is already captured, do nothing here (don't pad).
"""

# A2 消融 (no-failure): 通用 patch, 不喂诊断失败. 与 PATCH_TEMPLATE 唯一差别 = 无 DIAGNOSTIC FAILURES 节 +
# 无 rejected 反面教材, 隔离"失败驱动"机制的贡献. 仍保持小改 + 具体决策程序纪律, 公平对照.
GENERIC_PATCH_TEMPLATE = """You are improving a system prompt (skill) for a mathematical-reasoning LLM.

The CURRENT skill is below. Make a SMALL, SURGICAL edit to improve it - add a concrete DECISION PROCEDURE (a step-by-step test the model can run) for the focus area, not a generic checklist.

Rules:
1. Do NOT rewrite the skill. Keep its structure. Add at most ONE new short subsection (or replace/extend one existing bullet) that addresses the focus area.
2. The new content must be a CONCRETE, OPERATIONAL decision procedure - e.g. "For implication options, test both directions: does P⇒Q and Q⇒P? If only one direction holds, it is not the characterization." NOT "check implications carefully".
3. Keep total skill under ~5000 chars. If near the limit, compress redundant existing bullets rather than appending.
4. Do NOT reference specific problem answers, labels, or this problem's constants.
5. Focus area this round: {focus}

## CURRENT SKILL
{skill}

## YOUR OUTPUT
Output ONLY the edited skill text (full skill with your small edit applied). No meta commentary. START directly with the skill.
"""


def make_diagnostics(failures):
    parts = []
    for f in failures:
        it = f["item"]
        parts.append(
            f"### Problem {f['id']} (type={it.get('theorem_type','?')})\n"
            f"Theorem: {str(it.get('theorem') or '(n/a)')[:500]}\n"
            f"Question: {str(it['question'])[:500]}\n"
            f"Choices: {_format_choices(it['choices'])[:500]}\n"
            f"Model predicted: {f['pred']!r} | Correct: {f['correct']!r}\n"
            f"Model reasoning excerpt:\n{f['response_excerpt']}"
        )
    return "\n\n".join(parts)


def make_coverage_diagnostics(reps, counts, uncovered, skill_content, max_per_type=3):
    """Task4: 渲染全失败分布 (tally 表) + 未覆盖类型的分层代表样本.
    关键: 复用已算 run_one 结果, 零额外 API; 丢弃 600-char response_excerpt (bloat 源),
    只保留 theorem/question/choices/pred/correct 各截断 ~200 char, 控制 patcher prompt 体积.
    reps 已是每类型 up-to-3 最难样本 (compute_failure_distribution 排序保证)."""
    sc = skill_content.lower()
    # tally 表: 每类型一行, 标注 count + covered/uncovered
    tally_lines = []
    for lbl in sorted(counts, key=lambda l: -counts[l]):
        sigs = _TYPE_COVERAGE_SIG.get(lbl, ())
        covered = bool(sigs) and any(s in sc for s in sigs)
        flag = "COVERED" if covered else "UNCOVERED"
        focus_mark = "  <- FOCUS" if (uncovered and lbl == uncovered) else ""
        tally_lines.append(f"  [{flag}] ({counts[lbl]:2d}x) {lbl}{focus_mark}")
    tally = "\n".join(tally_lines) if tally_lines else "  (no failures)"

    # 分层样本: 喂所有未覆盖类型 (covered 的已有内容, 不重复喂), 每类最多 max_per_type
    parts = []
    uncovered_types = []
    for lbl in sorted(counts, key=lambda l: -counts[l]):
        sigs = _TYPE_COVERAGE_SIG.get(lbl, ())
        if sigs and not any(s in sc for s in sigs) and lbl in reps:
            uncovered_types.append(lbl)
    for lbl in uncovered_types:
        for f in reps[lbl][:max_per_type]:
            it = f["item"]
            parts.append(
                f"### Problem {f['id']} (failure-type: {lbl})\n"
                f"Theorem: {str(it.get('theorem') or '(n/a)')[:200]}\n"
                f"Question: {str(it['question'])[:200]}\n"
                f"Choices: {_format_choices(it['choices'])[:200]}\n"
                f"Model predicted: {f['pred']!r} | Correct: {f['correct']!r}"
            )
    stratified = "\n\n".join(parts) if parts else "(all failure types already covered - make a diversifying exploratory edit)"
    return tally, stratified


def make_wins(wins):
    """dir-5: 把 wins (skill 解对的难题) 渲染成正向蒸馏样本.
    关键 = 给 patcher 看 *正确的解法推理*, 让它抽象出可推广决策程序, 而非抄具体解."""
    parts = []
    for w in wins:
        it = w["item"]
        parts.append(
            f"### Problem {w['id']} (type={it.get('theorem_type','?')}, S₀_pass={w['s0_pass']:.2f} skill_pass={w['skill_pass']:.2f})\n"
            f"Theorem: {str(it.get('theorem') or '(n/a)')[:400]}\n"
            f"Question: {str(it['question'])[:400]}\n"
            f"Correct: {w['correct']!r}\n"
            f"Model's CORRECT reasoning excerpt (mine the reusable decision procedure here):\n{w['response_excerpt']}"
        )
    return "\n\n".join(parts)


def make_meta_contrast_diagnostics(meta_failures, meta_passes):
    """路径B (2026-09-10): 诊断对照对称化 — 把每道 meta 错题配对一道同构 meta 对题, 并列展示.
    唯一变量 = "skill 是否选对了 meta 选项". 只摆原始事实 (Theorem/Question/Choices + pred/correct),
    不写任何结论 (不注入 "meta⇒答案" 这句话) — 归纳留给 patcher 模型自己完成.
    配对策略: 确定性最近邻 (按 item id 序), 无"保证模型必然看出"的承诺.
    meta_failures 传入的是混型 failures (含 not-meta), 此处在入口先滤出真 meta 错题 (同构配对前提)."""
    if not meta_failures:
        return ""
    meta_failures = [f for f in meta_failures if classify_item_meta(f.get("item", {}) or {})]
    if not meta_failures:
        return ""
    parts = []
    for fi, f in enumerate(meta_failures[:6]):
        p = meta_passes[fi % len(meta_passes)] if meta_passes else None
        fit, pt = f["item"], (p["item"] if p else None)
        blk = (
            f"### Contrast {fi + 1}\n"
            f"[MISSED] Problem {f['id']} — model chose {f['pred']!r}, correct is {f['correct']!r}\n"
            f"Theorem: {str(fit.get('theorem') or '(n/a)')[:400]}\n"
            f"Question: {str(fit.get('question') or '(n/a)')[:400]}\n"
            f"Choices:\n{_format_choices(fit.get('choices') or [])}\n"
        )
        if p is not None:
            blk += (
                f"\n[SOLVED] Problem {p['id']} (same question style) — model chose {p['pred']!r}, correct is {p['correct']!r}\n"
                f"Theorem: {str(pt.get('theorem') or '(n/a)')[:400]}\n"
                f"Question: {str(pt.get('question') or '(n/a)')[:400]}\n"
                f"Choices:\n{_format_choices(pt.get('choices') or [])}\n"
            )
        else:
            blk += "\n[SOLVED] (no solved meta example available this round)\n"
        parts.append(blk)
    return "\n\n".join(parts)


# 路径B (2026-09-10): meta 对照块 — 纯粹并列 MISSED/SOLVED 两个原始样本, 无结论文字.
# 挂在 {positive_section} 之后, 让 patcher 自己从"同构 meta 题, 做对的选了 meta 选项"归纳出规则.
META_CONTRAST_SECTION = """## CONTRAST: SAME QUESTION STYLE, MISSED vs SOLVED
{contrast}

Above are pairs where each MISSED problem has a SOLVED partner of the SAME question style. Study what distinguishes the solved one from the missed one — the evidence is in the two raw examples. Derive any GENERALIZABLE procedure yourself; do not reference these specific problems' labels or constants.
"""


def generate_candidates(current_skill, failures, focus, rejected, K, explore=False, failure_driven=True, allow_modify=False, wins=None, learn_positive=False, coverage_bundle=None, meta_contrast=None):
    """生成 K 个候选 patch. 多样性靠 (a) GEN_TEMP 真采样 + (b) per-candidate 文字角度提示.
    C1 修复 (2026-08-05): 原 temps 参数从未使用 (chat_target 不接受 temperature), 探索模式是 no-op;
    现改用 explore 标志, 为 True 时切到 bolder angle hints 池 (卡住时加大力度).
    dir-4 (2026-08-05): allow_modify 时往 PATCH_TEMPLATE 注入 MODIFY_CLAUSE, 允许定向删除有害内容.
    dir-5 (2026-08-05): learn_positive 且有 wins 时注入 POSITIVE_SECTION, 从成功解法蒸馏正信号.
    Task4 (2026-08-17): coverage_bundle=(reps,counts,uncovered) 非空时切 PATCH_TEMPLATE_COVERAGE,
    喂全失败分布 tally + 未覆盖类型分层样本 (复用已算 run_one, 零额外 API)."""
    set_temp(GEN_TEMP)
    if failure_driven:
        pos_sec = ""
        if learn_positive and wins:
            pos_sec = POSITIVE_SECTION.format(wins=make_wins(wins))
        meta_sec = ""
        if meta_contrast is not None:
            # 路径B: meta_contrast=(meta_failures, meta_passes) — 渲染 MISSED/SOLVED 对照块.
            meta_sec = META_CONTRAST_SECTION.format(
                contrast=make_meta_contrast_diagnostics(meta_contrast[0], meta_contrast[1]))
        if coverage_bundle is not None:
            # Task4: 覆盖驱动 - 喂全分布而非仅 8 最难, focus=未覆盖类型 (填洞).
            reps, counts, uncovered = coverage_bundle
            tally, stratified = make_coverage_diagnostics(reps, counts, uncovered, current_skill)
            base_user = PATCH_TEMPLATE_COVERAGE.format(skill=current_skill, tally=tally,
                                                      stratified=stratified,
                                                      focus=focus, rejected=rejected or "(none)",
                                                      modify_clause=MODIFY_CLAUSE if allow_modify else "",
                                                      positive_section=pos_sec,
                                                      meta_contrast_section=meta_sec)
        else:
            diag = make_diagnostics(failures) if failures else "(no diagnostic failures this round - make a diversifying exploratory edit)"
            base_user = PATCH_TEMPLATE.format(skill=current_skill, diagnostics=diag,
                                              focus=focus, rejected=rejected or "(none)",
                                              modify_clause=MODIFY_CLAUSE if allow_modify else "",
                                              positive_section=pos_sec,
                                              meta_contrast_section=meta_sec)
    else:
        base_user = GENERIC_PATCH_TEMPLATE.format(skill=current_skill, focus=focus)
    # per-candidate 角度提示: 让不同候选侧重不同诊断维度, 增大候选池多样性.
    # C1: explore=True (卡住时) 切到 bolder 池 -- 更大刀阔斧的编辑角度, 原池仅做最小改.
    if explore:
        angle_hints = [
            "Make a BOLDER edit than usual: restructure or replace a section if it helps the diagnostic cases.",
            "Try an UNCONVENTIONAL angle - look for an overlooked edge case the current skill misses.",
            "Add a crisp counter-example / negation check the skill currently lacks.",
            "If a current bullet conflicts with the diagnostic cases, REPLACE it with a sharper version.",
            "Reorder: move the most failure-relevant instruction to the top and make it a hard rule.",
            "Introduce a concrete worked micro-decision that forces the right call on the diagnostic cases.",
        ]
    else:
        angle_hints = [
            "Emphasize the most concrete step-by-step test you can run on the diagnostic cases.",
            "Focus on the failure mode that recurs across the most diagnostic cases.",
            "Prefer a minimal edit (extend one existing bullet) over adding a new subsection.",
            "Target the equality-case / sharp-constant distinction specifically.",
        ]
    cands = []
    with ThreadPoolExecutor(max_workers=min(K, 8)) as ex:
        futs = {}
        for i in range(K):
            hint = angle_hints[i % len(angle_hints)]
            user = f"Editor angle #{i+1}: {hint}\n\n" + base_user
            futs[ex.submit(llm, PATCHER_SYSTEM, user,
                           max_tokens=4096, stage="evolver_patch")] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                cands.append((i, clean_skill(fut.result(), current_skill)))
            except Exception as e:  # noqa: BLE001
                # 不再静默吞异常回退 current_skill (会致 0 novel 假象). 记录并回退, 便于诊断.
                print(f"  ⚠️ cand{i} patch 生成失败: {e!r}", flush=True)
                cands.append((i, current_skill))
    cands.sort(key=lambda x: x[0])
    return [c for _, c in cands]


# ===================================================================
# Phase C (2026-08-18): 结构自进化 — 内容重组为候选维度
# ===================================================================
# 根因 (Task4 证伪 INPUT 侧, 孤立出结构为因果变量): 同内容不同结构 = 10pp 差距.
# Task3 (0.4654) meta-option = 独立顶级 bullet 置底 (突出, 末位近因); Task4 (0.3632) 同内容
# 但嵌套为 "Meta-Statement Evaluation" 子 bullet (从属, 中段) = 退步 10pp. 即内容量近似,
# 结构差异驱动分数. 真瓶颈在结构层非内容层也非输入层.
#
# 设计: 把"如何组织已有内容"作为候选维度, 与 patch 候选同台竞争 pass_rate (让数据决, 非关键词).
# 不指定 schema (绝不用结构原型菜单 — 违反用户要求"自己探索无穷的可能性"). 多样性杠杆 =
# STRUCTURE_ANGLES stance 旋转 (每个候选一个不同的重组视角), 而非温度采样. 结构候选
# 内容重组 v2 skill 的决策程序, 不注入 SkillOpt 也不新增内容 (规则1/5 防 SkillOpt 病).
# 公平: init 必须 = v2 (b98a9abd); 结构变异是自家内容重组 v2 的决策程序, 非 SkillOpt 嵌套.
#
# 复用: make_coverage_diagnostics/make_diagnostics 渲染失败上下文 (让结构重组锚定真实失败,
# 非空泛重排); clean_skill 清洗 LLM 输出; eval_on+bootstrap_ci_lower 评判. 零新增 eval 代码.
STRUCTURE_REORG_SYSTEM = (
    "You are an expert Information Architect for a system prompt (skill) driving a "
    "mathematical-reasoning LLM on theorem-level multiple-choice questions "
    "(LiveMathematicianBench). You REORGANIZE the existing content of the skill — its "
    "section order, hierarchy, grouping, and what sits at the top vs buried at the bottom. "
    "You do NOT add new content, and you do NOT delete any concrete decision procedure."
)

STRUCTURE_REORG_TEMPLATE = """You are restructuring a system prompt (skill) for a mathematical-reasoning LLM.

The CURRENT skill is below. Its CONTENT is good — every concrete decision procedure in it is valuable. The problem is its ORGANIZATION: how the content is ordered, grouped, and prioritized does NOT optimally serve the diagnostic failures shown. Different organizations of the SAME content can change which failure modes the model actually acts on (e.g. a decision procedure buried mid-skill as a sub-bullet is far less effective than the same procedure as a prominent top-level rule near the end).

Your job: produce a FUNDAMENTALLY DIFFERENT ORGANIZATION of the same content, driven by the stance below and anchored to the diagnostic failures. Reorganize freely.

Rules:
1. STRUCTURE only. Reorder, regroup, promote, demote, merge section headers, split sections, change heading hierarchy, move the most failure-relevant decision procedures to positions of prominence (top-level, or near the end where they land in the model's recency window). You decide the new organization — there is no fixed template.
2. PRESERVE every concrete decision procedure. Keep ALL the operational step-by-step tests from the current skill verbatim or near-verbatim; you may reword transitions/headers but not weaken or drop any procedure. The skill's knowledge content must be the same set of procedures, only arranged differently.
3. Do NOT add new decision procedures, new failure modes, or new padding. Do NOT inject content from any external skill. Reorganize what is already there.
4. Do NOT reference specific problem answers, labels, or constants.
5. The diagnostic failures show which failure mode dominates this round — let that drive WHICH content you promote to prominence.{modify_clause}

## CURRENT SKILL
{skill}

## DIAGNOSTIC FAILURES (the failure mode that dominates this round — promote content that addresses it to prominence)
{diagnostics}{positive_section}
## YOUR OUTPUT
Output ONLY the reorganized skill text (full skill, restructured). No meta commentary. START directly with the skill.
"""

# 结构重组 stance 池: 每个候选一个不同的组织视角 (非原型菜单 — 是"从什么角度重排"的提示,
# 不是"选哪种固定结构"). 高 temp 仍可能 mode collapse (风险B), stance 旋转是主多样性杠杆.
# iter_idx % len 决定本轮用哪几个 stance (确定性, 可复现), 跨轮旋转覆盖不同组织假设.
STRUCTURE_ANGLES = [
    "PROMINENCE: promote the decision procedure for the dominant diagnostic failure mode to a TOP-LEVEL section near the END (recency window), as an explicit hard rule. Demote everything else below it.",
    "SINGLE-PASS PIPELINE: reorganize the whole skill as one sequential decision procedure the model runs top-to-bottom on every option — stage 1 (parse), stage 2 (per-option test), stage 3 (rank/select) — folding the existing procedures into the matching stage.",
    "FAILURE-INDEXED: reorganize around failure types. Make each concrete decision procedure a top-level named rule keyed to the failure it fixes (e.g. 'RULE — equivalence: test both directions'), ordered by how often that failure dominates in the diagnostics.",
    "STRENGTH-LADDER: order content by logical strength climb — start with the weakest-statement / equality-case checks, escalate to quantifier/scope, peak with the meta-option 'strongest provable statement' ranking as the culminating top rule.",
    "TIGHTEN-AND-FLATTEN: remove all nesting. Flatten every sub-bullet to a top-level rule, merge sections that cover the same idea, and cut every header that does not precede a concrete procedure. Minimal, flat, dense.",
    "OPTION-CENTRIC: reorganize around evaluating ONE option end-to-end — group procedures into 'when an option claims equivalence', 'when it claims existence/uniqueness', 'when it is a meta-statement about the theorem', so the model picks a rule by the option's shape.",
]


def make_structure_diagnostics(failures, coverage_bundle, skill_content):
    """渲染结构候选的失败上下文. 复用 make_coverage_diagnostics (Task4 全分布) 或
    make_diagnostics (退化), 让结构重组锚定真实失败 (非空泛重排). 与 patch 路径同一数据源,
    零额外 API. 返回 diagnostics 字符串 (注入 STRUCTURE_REORG_TEMPLATE 的 {diagnostics})."""
    if coverage_bundle is not None:
        reps, counts, uncovered = coverage_bundle
        tally, stratified = make_coverage_diagnostics(reps, counts, uncovered, skill_content)
        return (f"### FULL FAILURE-TYPE DISTRIBUTION (select set)\n{tally}\n\n"
                f"### REPRESENTATIVE FAILURES FOR UNCOVERED TYPES\n{stratified}")
    if failures:
        return make_diagnostics(failures)
    return "(no diagnostic failures this round — reorganize for general clarity and prominence)"


def generate_structure_candidates(current_skill, *, failures, focus, coverage_bundle,
                                  rejected, Kc, struct_temp, iter_idx,
                                  allow_modify=False, wins=None, learn_positive=False):
    """生成 Kc 个结构变体候选 (Phase C). 内容重组 v2 skill 的决策程序, 不注入新内容.
    多样性靠 STRUCTURE_ANGLES stance 旋转 (每个候选一个重组视角) + struct_temp 采样.
    复用 make_structure_diagnostics 渲染失败上下文; clean_skill 清洗输出; eval_on 评判.
    返回 [skill, ...] (Kc 个, 按角度索引序, 失败回退 current_skill 不进候选竞争由调用方 hash 判).
    注: rejected 参数与 generate_candidates 调用签名对齐但本函数未用 — 结构重组重排既有内容,
    不追加新 patch idea, 故无"避免重复 patch 想法"之需 (与 patch 路径的有意不对称)."""
    set_temp(struct_temp)  # 阶段入口设温 (同 generate_candidates 的 set_temp(GEN_TEMP), 阶段间不重叠无竞态)
    pos_sec = ""
    if learn_positive and wins:
        pos_sec = POSITIVE_SECTION.format(wins=make_wins(wins))
    diag = make_structure_diagnostics(failures, coverage_bundle, current_skill)
    base_user = STRUCTURE_REORG_TEMPLATE.format(
        skill=current_skill, diagnostics=diag,
        modify_clause=MODIFY_CLAUSE if allow_modify else "",
        positive_section=pos_sec)
    cands = []
    with ThreadPoolExecutor(max_workers=min(Kc, 8)) as ex:
        futs = {}
        for i in range(Kc):
            angle = STRUCTURE_ANGLES[(iter_idx + i) % len(STRUCTURE_ANGLES)]
            user = f"Reorganization stance #{i+1}: {angle}\n\nFocus area (the dominant failure mode this round): {focus}\n\n{base_user}"
            futs[ex.submit(llm, STRUCTURE_REORG_SYSTEM, user,
                           max_tokens=4096, stage="evolver_struct")] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                cands.append((i, clean_skill(fut.result(), current_skill)))
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ struct{i} 生成失败: {e!r}", flush=True)
                cands.append((i, current_skill))  # 回退 current, hash 同 -> 不进候选竞争
    cands.sort(key=lambda x: x[0])
    return [c for _, c in cands]


# ===================================================================
# 机制3: 慢更新 — 蒸馏整合去冗余
# ===================================================================
CONSOLIDATE_TEMPLATE = """You are consolidating a system prompt (skill) for a mathematical-reasoning LLM.

Over several iterations, multiple patches have been applied. The skill below may now be redundant, bloated, or internally repetitive. Consolidate it WITHOUT losing any operational decision procedure.

Rules:
1. Merge redundant/overlapping bullets into the strongest single version.
2. Remove pure padding or restatements of the same idea.
3. Keep EVERY concrete decision procedure (step-by-step test) — these are the valuable parts.
4. Preserve the section structure. Keep under ~4500 chars.
5. Do NOT weaken or remove content that addresses a specific failure mode.
6. Output ONLY the consolidated skill text. No meta commentary.

## SKILL TO CONSOLIDATE
{skill}
"""


def consolidate(skill):
    set_temp(GEN_TEMP)
    try:
        out = llm(PATCHER_SYSTEM, CONSOLIDATE_TEMPLATE.format(skill=skill),
                  max_tokens=4096, stage="evolver_consolidate")
        c = clean_skill(out, skill)
        if len(c) < len(skill) * 0.7:  # 防过度删减
            return skill
        return c
    except Exception:
        return skill


# ===================================================================
# Facet-Tree (Stage 0): facet 分解 / 2-parent facet 交叉 (greenfield, 设计 §2.2(a)/§2.5)
# ===================================================================
# 进化对象 = Line A 全局 math skill (设计 §九-7a), 表示 = slotted template.
# facet_decompose 按 `^## ` 把 skill 切成 {slot_name: text} dict, facet_compose 取父本 A 的
# a_slot 段 + 父本 B 的 b_slot 段做 2-parent facet 交叉. 两者互为无损往返 (非目标 slot md5 不变),
# 该断言既是 compose/decompose 无损性的客观单测, 也是 Step 4 facet-edit 护栏的同一验收.
#
# 切段契约 (2026-08-21 实测真 S₀ 确认):
#   re.split(r'^(## .+)$', skill, MULTILINE) → [title_block, h1, body1, h2, body2, ...].
#   首块 title_block = "# Live Mathematical MCQ Heuristics\n\n" (无 ## 的题名, 实为前置文本);
#   其后交替 (## header, body). 无嵌套 ##、无 frontmatter `---`。body 首字符恒为 '\n'。
#   因此 decompose→compose 逐字节往返 = 按序 `''.join(split_blocks)` 恒等于原文 (md5 不变)。
_FACET_TITLE_RE = re.compile(r"^(## .+)$", re.MULTILINE)


def facet_decompose(skill: str):
    """按 `^## ` slot 分解 skill 为 dict{slot_name: body_text} + title 文本.

    返回 (title, slots):
      title  = "## " 之前的全部前置文本 (题名 `# ...`, 通常只有这一行)
      slots  = {section_key: body_text}, section_key 由 `## Xxx Yyy` → 'Xxx Yyy' (去 `## ` 前缀).
    body_text 保留段首换行符与原文缩进, 以便 compose 逐字节重组。空 skill/无 ## 退化: 返回 (title, {})。
    """
    blocks = _FACET_TITLE_RE.split(skill)
    title = blocks[0] if blocks else ""
    slots = {}
    # 交替 (header, body): header 在奇数位, 其下一块为 body (偶数位)。
    i = 1
    while i < len(blocks):
        header = blocks[i]
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        key = header[len("## "):].strip()
        slots[key] = body
        i += 2
    return title, slots


def facet_compose(title: str, slots: dict, slot_order=None):
    """facet_decompose 的逆: 按序把 title + (## key + body) 重组成 skill.

    slot_order 控制 `## ` 段的输出顺序 (Stage 0 交叉后重排 facet 轴用); 默认按 slots 插入序。
    每段 = "## {key}" + body (body 已含段首换行)。title 与 body 逐字保留, 不重排段内文本 →
    与非目标 slot 的 md5 无损往返兼容。返回完整 skill 文本。
    """
    parts = [title]
    keys = list(slots.keys()) if slot_order is None else slot_order
    keys = [k for k in keys if k in slots]
    for idx, key in enumerate(keys):
        parts.append("## " + key)
        body = slots[key]
        # 保段间分隔 (两头都防粘连):
        #   (a) 前缘: body 必以 \n 起头, 否则 `## key` 与 body 首行粘成 `## key- x`(header 解析失败,
        #       key 变成 `key- x`, key-set guard 误拒整个候选). 真 S₀ 经 decompose 的 body 必以 \n 起头
        #       (## Header\nBody 的换行被保留为 body 首字符), 此分支不触发 → 无损往返不破;
        #       仅对人工/嫁接/crossover 来的无头 body 补 \n — 把"格式技术性误拒"转成接受 (break-① donor 嫁接).
        #   (b) 后缘: 非末段若 body 不以 \n 结尾, 补单个 \n, 否则下一 `## ` 头粘进本段尾。
        #       末段不补 (真实 S₀ 末段尾换行已被调用方 strip, 补了反破坏无损往返)。
        if not body.startswith("\n"):
            body = "\n" + body
        if idx < len(keys) - 1 and not body.endswith("\n"):
            body = body + "\n"
        parts.append(body)
    return "".join(parts)


def _facet_roundtrip_assert(skill: str):
    """decompose→compose 逐字节往返校验 (Step 3 md5 无损单测锚 + Step 4 facet-edit 护栏共用断言)。

    若重组文本 md5 != 原文 md5, 抛 AssertionError (切段/重组丢失了字节)。"""
    title, slots = facet_decompose(skill)
    back = facet_compose(title, slots)
    assert md5(back) == md5(skill), (
        f"facet roundtrip not lossless: {md5(skill)} -> {md5(back)}"
        f" (len {len(skill)} -> {len(back)})")


def _self_test_facet():
    """纯函数自检 (零 API, 可被 Step 3 入口调用或 --self_test_facet 触发)。无 --unittest, 直接 assert。"""
    s0 = open(S0_DEFAULT, encoding="utf-8").read().strip()
    # (a) 真 S₀ decompose 应得 4 段 (含 frozen Final Answer), 题名 title 单独拿开
    title, slots = facet_decompose(s0)
    assert set(slots) == {"Option Comparison", "Theorem-Level Precision", "Hypotheses", "Final Answer"}, \
        f"unexpected S₀ sections: {set(slots)}"
    assert len(slots) == 4
    assert title.strip().startswith("# Live Mathematical MCQ Heuristics")
    # (b) 逐字节无损往返
    _facet_roundtrip_assert(s0)
    # (c) 单 slot 替换后, 非目标 slot md5 不变 (Step 4 护栏断言预演)
    new_slots = dict(slots)
    new_slots["Hypotheses"] = slots["Hypotheses"] + "\n- (self-test) verify non-empty domain."
    for key in slots:
        if key != "Hypotheses":
            assert md5(new_slots[key]) == md5(slots[key]), f"non-target slot {key} changed"
    # (d) 空 skill / 无 ## 退化不抛
    assert facet_decompose("") == ("", {})
    assert facet_decompose("plain text no section") == ("plain text no section", {})
    # (e) facet_edit_guard 护栏: 干净单 slot 编辑接受, 跨 slot 编辑拒绝
    _, s_slots = facet_decompose(s0)
    clean_edit = "\n- (guard-test) check hypotheses.\n"  # 无 ## 纯正文 -> 接受
    ok_slots, ok = facet_edit_guard(s_slots, "Hypotheses", clean_edit)
    assert ok and "Hypotheses" in ok_slots
    full_rewrite = "## Option Comparison\n- bad\n## Hypotheses\n- bad2\n"  # 多段 -> 拒绝
    _, bad_ok = facet_edit_guard(s_slots, "Hypotheses", full_rewrite)
    assert not bad_ok, "cross-slot edit must be rejected by guard"
    # (f) 回归: edit-guard 后的 compose 必须仍产出 4 段, 含独立 `## Final Answer` 段且其正文 md5 不变
    # (smoke test 曾暴露 Final Answer 段丢失: Hypotheses 尾换行被剥 → `## Final Answer` 粘进正文)。
    body_no_trailing_nl = "- (guard-test) check hypotheses.\n- (guard-test) tail has NO trailing newline"
    g_slots, g_ok = facet_edit_guard(s_slots, "Hypotheses", body_no_trailing_nl)
    assert g_ok
    recomposed = facet_compose(title, g_slots)
    rt_title, rt_slots = facet_decompose(recomposed)
    assert set(rt_slots) == {"Option Comparison", "Theorem-Level Precision", "Hypotheses", "Final Answer"}, \
        f"recompose dropped a section: {set(rt_slots)}"
    assert "Final Answer" in rt_slots
    assert md5(rt_slots["Final Answer"]) == md5(s_slots["Final Answer"]), \
        "Final Answer body changed by Hypotheses edit"
    assert "## Final Answer" in recomposed and "tail has NO trailing newline## " not in recomposed, \
        "Hypotheses tail glued into next header (missing separator)"
    # (g) section_expansion_guard: 干净单新 section 接受, 同名/跨 section/0段 拒绝, 现有 slot 不能改.
    _, e_slots = facet_decompose(s0)
    e_base_slots = dict(e_slots)
    clean_expand = "## Meta-Option Decision Procedure\n- (expand-test) identify options.\n- (expand-test) check validity.\n"
    exp_slots, exp_ok = section_expansion_guard(e_base_slots, "Meta-Option Decision Procedure", clean_expand)
    assert exp_ok and "Meta-Option Decision Procedure" in exp_slots, \
        "clean single-section expansion must be accepted"
    assert len(exp_slots) == 5 and set(exp_slots) == set(e_base_slots) | {"Meta-Option Decision Procedure"}, \
        "expansion must add exactly one new section"
    for key in e_base_slots:  # 现有 slot 必须逐字节不变 (只 add 不 replace)
        assert md5(exp_slots[key]) == md5(e_base_slots[key]), f"expansion mutated existing slot {key}"
    same_name = "## Final Answer\n- no\n"                       # 名字撞现有段 → 拒
    _, same_ok = section_expansion_guard(e_base_slots, "Final Answer", same_name)
    assert not same_ok, "same-name expansion must be rejected"
    conflict_name = "## Other\n- no\n"                          # 吐别的名字 → 拒
    _, conflict_ok = section_expansion_guard(e_base_slots, "Meta-Option Decision Procedure", conflict_name)
    assert not conflict_ok, "name-mismatch expansion must be rejected"
    multi = "## A\n- x\n## B\n- y\n"                            # 多段 → 拒
    _, multi_ok = section_expansion_guard(e_base_slots, "A", multi)
    assert not multi_ok, "multi-section expansion must be rejected"
    empty = ""                                                   # 空 → 拒
    _, empty_ok = section_expansion_guard(e_base_slots, "Zed", empty)
    assert not empty_ok, "empty expansion must be rejected"
    # (h) v2 overlay carry-forward: best_node 含已展开 section 时, overlay 必须把它携带过来.
    # 用真 S0 的 4 段作 root_slots, 模拟 expand 长出的第5段作 best_node.slots 的非-root key.
    # 验证 _overlay_assemble (overlay 算子E 的真实组装路径, 不是镜像) 把 expanded 段带到 overlay.
    cf_title, cf_root_slots = facet_decompose(s0)
    freeze_key = "Final Answer"
    expanded_key = "Regularity Decision Procedure"   # 非-root, 模拟 expand 长出的段
    expanded_body = "\n- (carry-test) check C^{k+1} subset C^k before claiming regularity.\n"
    best_node_slots = dict(cf_root_slots)
    best_node_slots[expanded_key] = expanded_body     # best_node 现在 5 段
    # donor: 一个单轴改进过的叶 (Option Comparison 段被改), 模拟 select_axis_donors 返回
    class _DonorStub:
        __slots__ = ("slots", "val_delta", "val_ci")
        def __init__(self, slots):
            self.slots = slots; self.val_delta = 0.1; self.val_ci = 0.2
    donor_slots = dict(cf_root_slots)
    donor_slots["Option Comparison"] = cf_root_slots["Option Comparison"] + "- (carry-test) donor edit.\n"
    donors = {"Option Comparison": _DonorStub(donor_slots)}
    ov_slots, ov_skill, ov_chk, ov_ok = _overlay_assemble(
        cf_title, cf_root_slots, best_node_slots, donors, freeze_key)
    assert ov_ok, "overlay carry-forward must pass key-set guard when expanded section present"
    # root 4 段全在 + expanded 段携带过来 = 5 段
    assert set(cf_root_slots).issubset(set(ov_chk)), "overlay dropped a root section"
    assert expanded_key in ov_chk, "overlay failed to carry-forward expanded section"
    assert len(ov_chk) == 5, f"overlay must have 5 sections (root 4 + expanded 1), got {len(ov_chk)}"
    # expanded 段逐字节等于 best_node 的 (构造性 copy, 无 LLM 改写)
    assert md5(ov_chk[expanded_key]) == md5(expanded_body), "overlay mutated the carried-forward section"
    # donor 嫁接生效: Option Comparison 段被 donor 覆盖 (= donor 段, ≠ root 原文)
    assert md5(ov_chk["Option Comparison"]) == md5(donor_slots["Option Comparison"]), \
        "overlay did not graft donor axis"
    # freeze 段未被 donor 覆盖 (donors 跳过 freeze). 注意: expanded 段追加在 Final Answer 之后,
    # Final Answer 变非末段 → facet_compose L684 合法补 \n 分隔符 → md5 变 (这正是 Change 2 guard
    # 刻意 key-set-only 不查 md5 的原因). 故这里验证两件事:
    #   (1) slots 层 (pre-compose): donors 没碰 freeze (仍是 root 原文对象, md5 不变);
    #   (2) compose 层 (post-compose): freeze 内容语义不变 (.strip() 相等, 容忍分隔符 \n).
    assert md5(ov_slots["Final Answer"]) == md5(cf_root_slots["Final Answer"]), \
        "overlay donors must not touch frozen section at slots level"
    assert ov_chk["Final Answer"].strip() == cf_root_slots["Final Answer"].strip(), \
        "frozen section content must survive compose round-trip (trailing separator tolerated)"
    # (i) 向后兼容: best_node 无 expanded 段 (只有 root keys) → carry-forward 是 no-op, guard 退化为旧 key-set.
    noexp_best = dict(cf_root_slots)   # 4 段, 无 expanded key
    ov2_slots, ov2_skill, ov2_chk, ov2_ok = _overlay_assemble(
        cf_title, cf_root_slots, noexp_best, donors, freeze_key)
    assert ov2_ok, "overlay guard must reduce to old key-set check when no expanded section"
    assert set(ov2_chk) == set(cf_root_slots), "no-expanded carry-forward must not add sections"
    assert len(ov2_chk) == 4, f"backward-compat overlay must keep 4 sections, got {len(ov2_chk)}"
    assert md5(ov2_chk["Option Comparison"]) == md5(donor_slots["Option Comparison"]), \
        "backward-compat overlay must still graft donor"
    # (j) v2 find_uncovered_type_for_expand: 跳过 meta (FAILURE_FOCI[5]) + 段名而非正文匹配.
    # 构造一个失败分布: meta 最高频但应被跳过, regularity 次高频且无专属段 → 应选 regularity.
    meta_lbl = FAILURE_FOCI[5]
    reg_lbl = FAILURE_FOCI[3]
    test_counts = {meta_lbl: 10, reg_lbl: 3}
    # best_node 只有 4 个 root 段名, 无 regularity 专属段 → regularity UNCOVERED
    picked = find_uncovered_type_for_expand(test_counts, cf_root_slots.keys())
    assert picked == reg_lbl, f"expand targeting must skip meta and pick uncovered non-meta type, got {picked!r}"
    # 已有 regularity 专属段后 → regularity 转 COVERED → 全覆盖 → 返回 None
    picked_after = find_uncovered_type_for_expand(test_counts, list(cf_root_slots.keys()) + ["Regularity Decision Procedure"])
    assert picked_after is None, "all non-meta covered must return None (skip expand)"
    # (k) Bug C 回归: subset_promote_decision 的 meta-mirror 路径不得用 mean 判据误救.
    # 真实 Bug C 数据形态: not-meta mean ≈ 0 (略 > -eps 使旧 mean 判据过) 但 ci 显著负 (分布跨零).
    # 旧代码 (mt["mean"]>=-eps / nm["mean"]>=-eps) 会在 not-meta ci<0 时误救 → best 回退.
    # 新代码 (mt["ci"]>=-eps / nm["ci"]>=-eps) 必须拒绝这类候选.
    # 构造: 7 题 +0.06 + 3 题 -0.18 → not-meta mean=-0.012 (>=-0.02 旧判据过) ci≈-0.08 (新判据拒).
    bugc_nm = [+0.06] * 7 + [-0.18] * 3            # 10 题, mean≈-0.012, ci<0
    bugc_mt = [+0.17] * 14                            # meta mean==ci≈0.17 (正)
    bugc_item, bugc_mask = {}, {}
    for i, d in enumerate(bugc_nm):
        bugc_item[f"nm_{i}"] = [d]; bugc_mask[f"nm_{i}"] = False
    for i, d in enumerate(bugc_mt):
        bugc_item[f"mt_{i}"] = [d]; bugc_mask[f"mt_{i}"] = True
    bugc_sig = subset_signal(bugc_item, bugc_mask)
    assert bugc_sig["not_meta"]["ci"] < 0, "Bug C fixture: not-meta ci must be negative"
    assert bugc_sig["not_meta"]["mean"] >= -0.02, "Bug C fixture: not-meta mean must clear old -eps bar (else not a regression test)"
    assert bugc_sig["meta"]["ci"] > 0, "Bug C fixture: meta ci must be positive (mirror path)"
    bugc_promote, _ = subset_promote_decision(bugc_item, best_val_delta=0.380, meta_mask=bugc_mask, enabled=True)
    assert not bugc_promote, \
        f"Bug C regression: meta-mirror must NOT rescue when not-meta ci<0 (got promote=True, nm_ci={bugc_sig['not_meta']['ci']:+.3f})"
    # 对称: 真 not-meta 正信号 (ci>0) 必须仍被救 (iter2 EXPAND-SUBSET 的正路径)
    good_nm = [+0.067] * 10                          # mean==ci>0, 真信号
    good_item, good_mask = {}, {}
    for i, d in enumerate(good_nm):
        good_item[f"nm_{i}"] = [d]; good_mask[f"nm_{i}"] = False
    for i, d in enumerate(bugc_mt):
        good_item[f"mt_{i}"] = [d]; good_mask[f"mt_{i}"] = True
    good_promote, _ = subset_promote_decision(good_item, best_val_delta=0.277, meta_mask=good_mask, enabled=True)
    assert good_promote, "genuine not-meta ci>0 signal must still be rescued (primary path intact)"
    # disabled 开关回退纯整体 (两者都 False)
    dis_promote, _ = subset_promote_decision(bugc_item, 0.380, bugc_mask, enabled=False)
    assert not dis_promote, "--no-subset_admit must disable rescue entirely"
    # (l) 靶点1 回归: _xo_section_guard 段数守卫放宽 (allow_merge).
    # 旧守卫 `len(child)<len(parent_a)` 硬拒任何降段数, 但 cohere-prompt 允许 "MAY merge two sections"
    # → 合法 2→1 merge 被误杀 (evolver_log 4/7 crossover guard/cohere-fail 主因之一).
    # 用真 S0 的 4 段 (3 非 freeze + 1 freeze=Final Answer) 构造 child 段集.
    xo_a = dict(s_slots)                       # parent_a = S0 4 段
    xo_freeze = "Final Answer"
    # (l.1) 旧行为 (allow_merge=False): 合并两非 freeze 段 → 总段数 4→3 → 必拒.
    xo_child_merge = {k: v for k, v in xo_a.items() if k != "Hypotheses"}  # 去掉 Hypotheses → 3 段
    assert not _xo_section_guard(xo_child_merge, xo_a, xo_freeze, allow_merge=False), \
        "legacy guard must reject any section-count decrease (incl. legit merge)"
    # (l.2) 新行为 (allow_merge=True): 同上合并 → 非 freeze 段 3→2 (减少) → 仍拒 (真丢内容).
    assert not _xo_section_guard(xo_child_merge, xo_a, xo_freeze, allow_merge=True), \
        "allow_merge must still reject when non-freeze section COUNT drops (real content loss)"
    # (l.3) 关键: 合并两非 freeze 段但保留其内容 → 非 freeze 段总数不变 → allow_merge 必须放行.
    # 模拟 cohere 把 Hypotheses 内容并入 Option Comparison (改名), 非 freeze 段 3→3 不变.
    xo_child_keepnf = {k: v for k, v in xo_a.items() if k != "Hypotheses"}
    xo_child_keepnf[xo_freeze] = xo_a[xo_freeze]            # freeze 段保留 (4 key → 3 key, 但非freeze=3→2 这里)
    # 真正构造非 freeze 不减: parent_a 非 freeze=3 (Option/Theorem/Hypo), child 非 freeze 也需 ≥3.
    # 合并只能减非 freeze 数, 故 "非 freeze 不减" 的合法场景 = cohere 把 freeze 段并掉 (freeze 不计),
    # 或 child 多了桥接段抵消. 这里测: child 段数减少只因 freeze 段被并掉 (非 freeze 全留).
    xo_child_freeze_merged = dict(xo_a)                      # 4 段全留
    del xo_child_freeze_merged[xo_freeze]                    # freeze 被并掉 → 3 段, 但非 freeze 3→3 不变
    assert _xo_section_guard(xo_child_freeze_merged, xo_a, xo_freeze, allow_merge=True), \
        "allow_merge must pass when only freeze section is merged (non-freeze count unchanged)"
    assert not _xo_section_guard(xo_child_freeze_merged, xo_a, xo_freeze, allow_merge=False), \
        "legacy guard must still reject freeze-only merge (total count 4→3)"
    # (l.4) 增段 (桥接/expand) → 两种守卫都放行 (增段从不丢内容).
    xo_child_add = dict(xo_a)
    xo_child_add["Bridge Section"] = "- bridge\n"
    assert _xo_section_guard(xo_child_add, xo_a, xo_freeze, allow_merge=True), \
        "guard must pass section addition (add never loses content)"
    assert _xo_section_guard(xo_child_add, xo_a, xo_freeze, allow_merge=False), \
        "legacy guard must pass section addition"
    # (l.5) 不变 (段数相同) → 两种守卫都放行.
    assert _xo_section_guard(xo_a, xo_a, xo_freeze, allow_merge=True), \
        "guard must pass unchanged child"
    assert _xo_section_guard(xo_a, xo_a, xo_freeze, allow_merge=False)
    # (m) 算子H: facet_edit_guard 必须能改 expand 长出的段 (base 轴名之外的 key).
    # 模拟 best_node 含一个 expand 长出的 "Implication Direction & Biconditional Check" 段,
    # strengthen 算子只重写它 → guard 接受且其他段逐字节不变.
    h_base_slots = {
        "Option Comparison": "- compare options.\n",
        "Theorem-Level Precision": "- be precise.\n",
        "Implication Direction & Biconditional Check":
            "- check both directions: forward sufficiency and backward necessity.\n",
        "Final Answer": "- final.\n",
    }
    h_edit = "## Implication Direction & Biconditional Check\n- check both directions: forward sufficiency and backward necessity.\n- also distinguish existence from uniqueness: prove existence AND rule out others.\n"
    h_new, h_ok = facet_edit_guard(h_base_slots, "Implication Direction & Biconditional Check", h_edit)
    assert h_ok, "facet_edit_guard must accept single-section body replacement on expand-grown section name"
    assert ("existence from uniqueness" in h_new["Implication Direction & Biconditional Check"]), \
        "strengthen body must be applied to target section"
    for k in h_base_slots:
        if k != "Implication Direction & Biconditional Check":
            assert md5(h_new[k]) == md5(h_base_slots[k]), f"strengthen mutated non-target slot {k}"
    # (n) 算子H 分类器后移: strengthen ON 时, 含 meta 信号 + unique/exist 关键词的题应归 existence/uniqueness (foci[1]).
    global _STRENGTHEN_ON
    _saved = _STRENGTHEN_ON
    try:
        _STRENGTHEN_ON = False
        # OFF (字节复现旧行为): meta 早期返回 → foci[5]
        meta_uniq_fail = {"item": {"question": "Which statement gives the strongest provable result about the uniqueness of the solution?",
                                   "theorem": "There exists a unique fixed point.", "choices": []}}
        assert classify_failure_type(meta_uniq_fail) == FAILURE_FOCI[5], \
            "strengthen OFF: meta-question with uniqueness keyword must fall back to meta (byte-repro)"
        _STRENGTHEN_ON = True
        # ON: 关键词梯优先于 meta → foci[1] (existence/uniqueness), 不被 meta 吞
        assert classify_failure_type(meta_uniq_fail) == FAILURE_FOCI[1], \
            "strengthen ON: meta-question with uniqueness keyword must classify as existence/uniqueness, not meta"
        # ON: 纯 meta 无具体关键词 → 仍 foci[5] (无锚点不误归)
        pure_meta_fail = {"item": {"question": "Which is the strongest statement that can be proven?",
                                   "theorem": "Some theorem.", "choices": [{"label": "E", "text": "one of the remaining"}]}}
        assert classify_failure_type(pure_meta_fail) == FAILURE_FOCI[5], \
            "strengthen ON: pure meta (no concrete keyword) must still fall back to meta"
        # ON: iff 关键词 → foci[0] (implication), 不被 meta 吞
        iff_fail = {"item": {"question": "Which statement holds if and only if the function is continuous — the strongest provable result?",
                             "theorem": "Continuity iff ...", "choices": []}}
        assert classify_failure_type(iff_fail) == FAILURE_FOCI[0], \
            "strengthen ON: meta-question with iff keyword must classify as implication, not meta"
    finally:
        _STRENGTHEN_ON = _saved  # 复位 (self-test 不应改全局状态)
    # (o) select_strengthen_target: 饱和事件下, 选已含双向测试关键词的段 (axis_score 优先).
    sat_counts = {FAILURE_FOCI[5]: 16, FAILURE_FOCI[0]: 1}
    sat_reps = {FAILURE_FOCI[5]: [{"s0_pass": 0.0, "item": {"question": "strongest provable uniqueness result", "theorem": "exists unique", "choices": []}}],
                FAILURE_FOCI[0]: [{"s0_pass": 0.1, "item": {"question": "iff biconditional", "theorem": "implies", "choices": []}}]}
    sat_slots = {
        "Option Comparison": "- compare options.\n",
        "Theorem-Level Precision": "- be precise.\n",
        "Implication Direction & Biconditional Check":
            "- check both directions: forward sufficiency and backward necessity.\n",
        "Final Answer": "- final.\n",
    }
    sec_key, sec_failures = select_strengthen_target(sat_counts, sat_reps, sat_slots, "Final Answer")
    assert sec_key == "Implication Direction & Biconditional Check", \
        f"strengthen must pick the bidirectional-axis-bearing section, got {sec_key!r}"
    assert sec_failures is not None and len(sec_failures) >= 1, "strengthen must return hardest failure reps"
    # 跨类型展平取最难: foci[5] rep (s0_pass=0.0) 应在前
    assert sec_failures[0]["s0_pass"] == 0.0, "strengthen reps must be sorted hardest-first (s0_pass ascending)"
    # freeze 段不入选
    assert sec_key != "Final Answer", "strengthen must not select the frozen section"
    # 无双向段时退化为 relevance 匹配 (仍返回某非 freeze 段)
    early_slots = {"Option Comparison": "- compare.\n", "Theorem-Level Precision": "- precise.\n", "Final Answer": "- final.\n"}
    ek, ef = select_strengthen_target(sat_counts, sat_reps, early_slots, "Final Answer")
    assert ek is not None and ek != "Final Answer", \
        "strengthen must fall back to a non-frozen section when no bidirectional axis exists"
    # (p) 算子H 饱和判定 (修 strengthen run 暴露的设计缺口, 用户指定):
    #   "已有专段且新变体 delta < 旧段 delta 才判饱和, 否则 expand 抢先". 纯逻辑测试 — 不依赖 llm/eval_on,
    #   只验证判据本身: type_best_delta 已记该类型 + 新 delta 回落 = 饱和; 新 delta 持平/更强 = 不饱和 (坚持探索).
    impl_lbl = FAILURE_FOCI[0]   # implication, 模拟 strengthen run 里反复出现的类型
    type_best_delta = {impl_lbl: 0.127}   # iter4 采纳段 delta (专段已存在)
    # (p1) 新变体 delta < 旧段 (0.100 < 0.127) → 饱和 → 应交 strengthen
    new_delta_weaker = 0.100
    sat1 = (impl_lbl in type_best_delta) and (new_delta_weaker < type_best_delta[impl_lbl])
    assert sat1, "weaker new variant (delta < prev_best) must saturate (hand to strengthen)"
    # (p2) 新变体 delta == 旧段 (0.127 == 0.127) → 不饱和 (坚持探索, 持平不算回落)
    new_delta_equal = 0.127
    sat2 = (impl_lbl in type_best_delta) and (new_delta_equal < type_best_delta[impl_lbl])
    assert not sat2, "equal-delta variant must NOT saturate (explore; only strictly-weaker saturates)"
    # (p3) 新变体 delta > 旧段 (0.258 > 0.127, 如 iter6 强于 iter4) → 不饱和 → expand 正常采纳
    new_delta_stronger = 0.258
    sat3 = (impl_lbl in type_best_delta) and (new_delta_stronger < type_best_delta[impl_lbl])
    assert not sat3, "stronger new variant (delta > prev_best) must NOT saturate (expand preempts, explore)"
    # (p4) 采纳后 type_best_delta 取 max 刷新 (更强变体不降阈值, 弱采纳不封顶未来强变体)
    type_best_delta[impl_lbl] = max(type_best_delta.get(impl_lbl, float("-inf")), new_delta_stronger)
    assert type_best_delta[impl_lbl] == 0.258, "type_best_delta must update to max(stronger variant)"
    # (p5) 首次类型 (type_best_delta 无记录) → 不饱和 (新类型允许 expand 探索)
    new_type_lbl = FAILURE_FOCI[3]
    sat5 = (new_type_lbl in type_best_delta)
    assert not sat5, "first-time type (no prior section) must NOT saturate (expand explores new type)"
    # (q) 方案1 find_covered_failing_type (per-type strengthen 触发放宽): 纯函数断言, 零 API.
    #   镜像 (j) 的 find_uncovered_type_for_expand, 但语义相反 — 找「已覆盖但仍在失败」类型.
    impl_lbl = FAILURE_FOCI[0]   # sigs = ("iff","biconditional","both direction")
    exist_lbl = FAILURE_FOCI[1]  # sigs = ("uniqueness","at most one")
    reg_lbl = FAILURE_FOCI[3]    # sigs = ("regularit","c^{","smooth")
    # 段名含签名关键词 = 该类型已有专属 section = COVERED (与 find_uncovered_type_for_expand 同口径: 段名 blob).
    cov_impl_key = "Biconditional iff Test Procedure"     # 含 "iff" + "biconditional" → impl COVERED
    cov_exist_key = "Existence-Uniqueness Verification"   # 含 "uniqueness" → exist COVERED
    # (q1) counts 含已覆盖+仍失败类型 → 返回最高频该类型 (impl=5 高于 reg=2; reg 未覆盖故非候选).
    q1_counts = {impl_lbl: 5, reg_lbl: 2}
    q1_keys = list(cf_root_slots.keys()) + [cov_impl_key]   # impl 已覆盖, reg 未覆盖
    q1 = find_covered_failing_type(q1_counts, q1_keys)
    assert q1 == impl_lbl, f"q1: covered+failing highest-freq type must return impl, got {q1!r}"
    # (q2) 已覆盖类型全无失败 (counts<=0) → 返回 None (无 efficacy gap, 不需加固).
    q2_counts = {impl_lbl: 0, reg_lbl: 2}   # impl 已覆盖但无失败; reg 未覆盖
    q2 = find_covered_failing_type(q2_counts, q1_keys)
    assert q2 is None, f"q2: covered type with no failures must return None, got {q2!r}"
    # (q3) meta (FAILURE_FOCI[5]) 即使已覆盖+仍失败也跳过 (与 expand 同口径: meta 交 facet_edit 轴).
    meta_lbl = FAILURE_FOCI[5]
    cov_meta_key = "Meta-options Stronger Result Decision"  # 含 "meta" + "stronger result" → meta COVERED
    q3_counts = {meta_lbl: 10, impl_lbl: 3}   # meta 最高频且已覆盖, 但应被跳过
    q3_keys = list(cf_root_slots.keys()) + [cov_meta_key, cov_impl_key]
    q3 = find_covered_failing_type(q3_counts, q3_keys)
    assert q3 != meta_lbl, f"q3: meta type must be skipped (strengthen path mirrors expand), got meta"
    assert q3 == impl_lbl, f"q3: must fall through to next covered-failing non-meta type, got {q3!r}"
    # (q4) exclude 参数: 排除指定类型 (模拟本 iter 刚 expand 的 target, too-early-abandonment guard).
    q4_counts = {impl_lbl: 5, exist_lbl: 4}   # 都已覆盖+仍失败
    q4_keys = list(cf_root_slots.keys()) + [cov_impl_key, cov_exist_key]
    q4_no_excl = find_covered_failing_type(q4_counts, q4_keys)
    assert q4_no_excl == impl_lbl, f"q4: without exclude must return highest-freq (impl), got {q4_no_excl!r}"
    q4_excl = find_covered_failing_type(q4_counts, q4_keys, exclude=impl_lbl)
    assert q4_excl == exist_lbl, f"q4: exclude=impl must fall through to exist, got {q4_excl!r}"
    # (q5) 确定性: 同输入多次调用结果一致 (可复现, tiebreak 由 sorted 稳定性决定 — 不依赖字典序断言).
    q5_counts = {impl_lbl: 3, exist_lbl: 3}   # 频次并列, 都已覆盖+仍失败
    q5_keys = list(cf_root_slots.keys()) + [cov_impl_key, cov_exist_key]
    q5_a = find_covered_failing_type(q5_counts, q5_keys)
    q5_b = find_covered_failing_type(q5_counts, q5_keys)
    assert q5_a == q5_b, f"q5: identical input must return identical result (deterministic), got {q5_a!r} vs {q5_b!r}"
    # (q6) 与 find_uncovered_type_for_expand 互补: 同 counts+section_keys 下返回不同类型 (互斥, 不重复消耗 API).
    #   impl 已覆盖+仍失败 → covered_failing 返回 impl; reg 未覆盖 → uncovered 返回 reg. 两者不重叠.
    q6_counts = {impl_lbl: 5, reg_lbl: 3}
    q6_keys = list(cf_root_slots.keys()) + [cov_impl_key]   # impl COVERED, reg UNCOVERED
    q6_cov = find_covered_failing_type(q6_counts, q6_keys)
    q6_unc = find_uncovered_type_for_expand(q6_counts, q6_keys)
    assert q6_cov == impl_lbl, f"q6: covered-failing must return impl, got {q6_cov!r}"
    assert q6_unc == reg_lbl, f"q6: uncovered must return reg, got {q6_unc!r}"
    assert q6_cov != q6_unc, "q6: covered-failing and uncovered must be disjoint (no type in both gates)"
    # (q7) Task #7: _archive_section_union — gate 失明修复. 9/8 run 坐实 expand 产物 became_best=False
    #   时只进 archive 不进 best_node 链 → 其专属 section 对 best_node.slots gate 不可见 → 0 fire.
    #   union 必须覆盖 archive-only 节点的 section; find_covered_failing_type 用 union 后能见 Existence.
    # q7a: union 跨节点去重保序, 含 archive-only 节点的专属段.
    class _StubN:
        __slots__ = ("skill", "slots", "facet_label", "parent", "val_delta", "val_ci")
        def __init__(self, slots, delta=0.0, ci=0.0, label="stub"):
            self.skill = ""; self.slots = slots; self.facet_label = label
            self.parent = None; self.val_delta = delta; self.val_ci = ci
    base_slots = {"Option Comparison": "x", "Theorem-Level Precision": "x",
                  "Hypotheses": "x", "Final Answer": "x"}
    q7_arch = [
        _StubN(dict(base_slots), delta=0.390, ci=0.300, label="best"),       # best_node: 4 基本段, 无 Existence
        _StubN(dict(base_slots), delta=0.352, ci=0.270, label="leaf"),
        _StubN({**dict(base_slots),
                "Distinguish Existence from Uniqueness/Regularity Claims": "y"},
               delta=0.324, ci=0.251, label="expand:Existence"),              # archive-only (未夺 best)
    ]
    q7_union = _archive_section_union(q7_arch)
    assert "Distinguish Existence from Uniqueness/Regularity Claims" in q7_union, \
        "q7a: union MUST include archive-only node's section (the gate-blindness fix)"
    assert len(q7_union) == 5, f"q7a: union dedups 4 base + 1 Existence = 5, got {len(q7_union)}"
    # 保序: best_node 段在前 (archive 顺序), Existence 在后
    assert q7_union[:4] == list(base_slots.keys()), "q7a: union preserves archive insertion order"
    # q7b: 决定性复现 — gate on best_node → None (失明); gate on union → Existence (fire).
    q7b_counts = {exist_lbl: 6, reg_lbl: 1}   # Existence 仍高频失败
    q7b_gate_best = find_covered_failing_type(q7b_counts, list(base_slots.keys()), exclude=reg_lbl)
    q7b_gate_union = find_covered_failing_type(q7b_counts, q7_union, exclude=reg_lbl)
    assert q7b_gate_best is None, \
        f"q7b: gate on best_node (no Existence section) must be BLIND → None, got {q7b_gate_best!r}"
    assert q7b_gate_union == exist_lbl, \
        f"q7b: gate on archive union (sees Existence) must fire → exist, got {q7b_gate_union!r}"
    # q7c: 空 archive → [] (不抛异常, gate 传 [] blob="" → None, 安全不 fire).
    assert _archive_section_union([]) == [], "q7c: empty archive union = [] (safe, no fire)"
    # q7d: 单节点 archive → 该节点 slots keys 拷贝 (退化为旧行为, 向后兼容).
    solo = _StubN({"A": "x", "B": "y"})
    assert _archive_section_union([solo]) == ["A", "B"], "q7d: single-node union = its keys"
    # (r) 方向 B: _donor_coverage_types / select_donors_directed / recombine_facets_multi (纯函数, 零 API).
    #   定向 donor 按 section body 对当前失败分布 counts 的覆盖匹配度排序, 非纯 val_ci.
    # r1: _donor_coverage_types — body 含签名关键词且 counts[lbl]>0 → 该类型算覆盖.
    impl_lbl_r = FAILURE_FOCI[0]   # sigs=("iff","biconditional","both direction")
    exist_lbl_r = FAILURE_FOCI[1]  # sigs=("uniqueness","at most one")
    reg_lbl_r = FAILURE_FOCI[3]    # sigs=("regularit","c^{","smooth")
    r1_counts = {impl_lbl_r: 5, exist_lbl_r: 3, reg_lbl_r: 2}
    r1_body_impl = "Check the iff and biconditional direction of the implication."  # 含 iff+biconditional
    r1_body_exist = "Verify uniqueness and at most one solution."                    # 含 uniqueness+at most one
    r1_body_none = "General advice about checking your work."                        # 无签名关键词
    assert impl_lbl_r in _donor_coverage_types(r1_body_impl, r1_counts), \
        "r1: body with iff+biconditional covers impl type"
    assert exist_lbl_r in _donor_coverage_types(r1_body_exist, r1_counts), \
        "r1: body with uniqueness covers exist type"
    assert _donor_coverage_types(r1_body_none, r1_counts) == set(), \
        "r1: body with no signature keyword covers nothing"
    # r1b: counts[lbl]<=0 的类型不算覆盖 (无 efficacy gap, 定向价值 0).
    r1b_counts = {impl_lbl_r: 0, exist_lbl_r: 3}   # impl 已无失败
    assert _donor_coverage_types(r1_body_impl, r1b_counts) == set(), \
        "r1b: type with counts<=0 excluded (no gap to cover)"
    # r2: select_donors_directed — 门控 + 定向排序.
    #   构造 archive: 节点A 段含 impl 签名 (val_ci=0.3), 节点B 段含 exist 签名 (val_ci=0.2),
    #   节点C 段无签名 (val_ci=0.9, 高 ci 但定向价值 0 → 应被门控排除).
    root_slots_r = {"Option Comparison": "root oc", "Theorem-Level Precision": "root tlp",
                    "Final Answer": "root fa"}
    ndA = _StubN({"Option Comparison": "root oc",
                  "Implication iff Biconditional Test": r1_body_impl,  # impl 覆盖
                  "Final Answer": "root fa"}, delta=0.30, ci=0.30, label="A")
    ndB = _StubN({"Option Comparison": "root oc",
                  "Existence Uniqueness Verification": r1_body_exist,   # exist 覆盖
                  "Final Answer": "root fa"}, delta=0.20, ci=0.20, label="B")
    ndC = _StubN({"Option Comparison": "root oc",
                  "General Strategy": r1_body_none,                      # 无覆盖, 高 ci
                  "Final Answer": "root fa"}, delta=0.50, ci=0.90, label="C")
    # root 段 (== root_slots) 应被排除; freeze=Final Answer 应被排除; ndC 的 General Strategy 无覆盖应被排除.
    r2_donors = select_donors_directed([ndA, ndB, ndC], root_slots_r, "Final Answer",
                                        r1_counts, topk=3)
    r2_sec_keys = [k for k, _ in r2_donors]
    assert "General Strategy" not in r2_sec_keys, \
        "r2: donor with no failure-type coverage must be EXCLUDED (directed gate)"
    assert "Implication iff Biconditional Test" in r2_sec_keys, "r2: impl-covering donor included"
    assert "Existence Uniqueness Verification" in r2_sec_keys, "r2: exist-covering donor included"
    assert len(r2_donors) == 2, f"r2: only 2 directed donors survive gate, got {len(r2_donors)}"
    # r2b: 定向排序 — impl 覆盖更高频失败类型 (counts=5) 应排前于 exist (counts=3).
    #   两个 donor 各覆盖 1 个类型, impl 覆盖的类型 counts=5 > exist 的 3 → impl donor 排第一.
    assert r2_donors[0][0] == "Implication iff Biconditional Test", \
        "r2b: donor covering higher-freq failure type ranked first"
    # r2c: val_ci<=0 节点被门控排除 (与 select_section_donors 同口径).
    ndNeg = _StubN({"Option Comparison": "root oc",
                    "Implication iff Biconditional Test": r1_body_impl,
                    "Final Answer": "root fa"}, delta=-0.1, ci=-0.05, label="Neg")
    r2c_donors = select_donors_directed([ndNeg, ndB], root_slots_r, "Final Answer",
                                         r1_counts, topk=3)
    assert all(nd.facet_label != "Neg" for _, nd in r2c_donors), \
        "r2c: val_ci<=0 node excluded (ci>0 gate)"
    # r2d: 空 counts → 所有 donor 覆盖=0 → 全被门控排除 → 返回 [].
    r2d_donors = select_donors_directed([ndA, ndB], root_slots_r, "Final Answer",
                                        {}, topk=3)
    assert r2d_donors == [], "r2d: empty counts → all donors coverage=0 → excluded → []"
    # r2e: 段 md5 == root 段 md5 (未变异) 被排除. root 已有 impl 段; ndDup impl段 == root (未变异→排除),
    #   ndMut impl段 != root (变异, 仍含 iff 签名→覆盖 impl) → 保留.
    root_impl = dict(root_slots_r)
    root_impl["Implication iff Biconditional Test"] = r1_body_impl  # root 已有 impl 段
    ndMut = _StubN({"Option Comparison": "root oc",
                    "Implication iff Biconditional Test": "Check iff and biconditional MODIFIED direction.",
                    "Final Answer": "root fa"}, delta=0.30, ci=0.30, label="Mut")
    ndDup = _StubN(dict(root_impl), delta=0.40, ci=0.40, label="Dup")  # impl段 == root (未变异)
    r2e_donors = select_donors_directed([ndDup, ndMut], root_impl, "Final Answer",
                                        r1_counts, topk=3)
    assert all(nd.facet_label != "Dup" for _, nd in r2e_donors), \
        "r2e: unmutated donor (md5==root) excluded"
    assert any(nd.facet_label == "Mut" for _, nd in r2e_donors), \
        "r2e: mutated donor (md5!=root, covers impl) kept"
    # r3: _assemble_multi_grafts — 多 donor 段嫁接纯组装逻辑 (零 API, 零 LLM).
    #   测「无有效嫁接→空 grafted」「freeze 不碰」「同 sec_key 去重」「未变异不嫁接」.
    mp_base = {"Option Comparison": "oc base", "Theorem-Level Precision": "tlp base",
               "Final Answer": "fa base"}
    # r3a: 正常嫁接 — donor 贡献新段 + 替换段, freeze 不碰.
    ndDonor1 = _StubN({"Existence Uniqueness Verification": r1_body_exist,
                       "Final Answer": "fa donor"}, delta=0.20, ci=0.20, label="d1")
    ndDonor2 = _StubN({"Option Comparison": "oc donor variant",  # 替换 base 的 OC 段
                       "Final Answer": "fa donor2"}, delta=0.15, ci=0.15, label="d2")
    r3a_slots, r3a_grafted = _assemble_multi_grafts(
        mp_base, [("Existence Uniqueness Verification", ndDonor1),
                  ("Option Comparison", ndDonor2)], "Final Answer")
    assert r3a_slots["Final Answer"] == "fa base", \
        "r3a: freeze slot must stay base value (not grafted)"
    assert "Existence Uniqueness Verification" in r3a_slots, "r3a: new grafted section present"
    assert r3a_slots["Option Comparison"] == "oc donor variant", \
        "r3a: existing section overwritten by donor variant"
    assert len(r3a_grafted) == 2, f"r3a: 2 valid grafts, got {len(r3a_grafted)}"
    # r3b: 空 donors → 空 grafted, slots == base.
    r3b_slots, r3b_grafted = _assemble_multi_grafts(mp_base, [], "Final Answer")
    assert r3b_grafted == [], "r3b: empty donors → no grafts"
    assert r3b_slots == mp_base, "r3b: empty donors → return base copy unchanged"
    # r3c: donor 段 == base 段 (md5 相同, 未变异) → 该 donor 不嫁接 → grafted 空.
    ndSame = _StubN({"Option Comparison": "oc base",  # == base, 未变异
                     "Final Answer": "fa base"}, delta=0.20, ci=0.20, label="same")
    r3c_slots, r3c_grafted = _assemble_multi_grafts(mp_base, [("Option Comparison", ndSame)],
                                                    "Final Answer")
    assert r3c_grafted == [], "r3c: donor==base (unmutated) → no graft"
    assert r3c_slots == mp_base, "r3c: no graft → return base copy"
    # r3d: 同 sec_key 多 donor → 只取第一个 (去重).
    ndDx = _StubN({"Option Comparison": "oc variant X"}, delta=0.20, ci=0.20, label="X")
    ndDy = _StubN({"Option Comparison": "oc variant Y"}, delta=0.25, ci=0.25, label="Y")
    # list order: X 在前 → X 的段被采, Y 被 seen_sec 去重跳过.
    r3d_slots, r3d_grafted = _assemble_multi_grafts(
        mp_base, [("Option Comparison", ndDx), ("Option Comparison", ndDy)], "Final Answer")
    assert r3d_slots["Option Comparison"] == "oc variant X", \
        "r3d: same sec_key → first donor wins (dedup)"
    assert len(r3d_grafted) == 1, f"r3d: 1 graft after dedup, got {len(r3d_grafted)}"
    # r3e: freeze 段作 donor sec_key → 被跳过 (不嫁接 freeze).
    ndFreeze = _StubN({"Final Answer": "fa hijack"}, delta=0.30, ci=0.30, label="frz")
    r3e_slots, r3e_grafted = _assemble_multi_grafts(mp_base, [("Final Answer", ndFreeze)], "Final Answer")
    assert r3e_grafted == [], "r3e: freeze-key donor → no valid graft"
    assert r3e_slots["Final Answer"] == "fa base", "r3e: freeze slot not hijacked"
    # (n) fix-① 候选池准入自检: classify_item_dim / git_diff_similarity / spike_peak_hit / admission_filter.
    # n1: 内容优先分类 — 含 iff 的题归 dim0, 含 unique 归 dim1, 全无内容关键词归 meta(5).
    assert classify_item_dim({"question": "iff p then q", "theorem": ""}) == 0
    assert classify_item_dim({"question": "prove uniqueness and existence", "theorem": ""}) == 1
    assert classify_item_dim({"question": "what is the strongest result", "theorem": ""}) == 5, \
        "no content keyword must fall back to meta dim5"
    assert classify_item_dim({"question": "show the domain is the interior", "theorem": ""}) == 6
    # n2: git_diff_similarity — 近同文本 → 高相似 (>0.95), 实质 diff → 低相似.
    assert git_diff_similarity("a\nb\nc", "a\nb\nc") == 1.0
    assert git_diff_similarity("a\nb\nc", "a\nb\nc\n\n- completely new bullet here\n") < 0.95
    assert 0.95 <= git_diff_similarity("a\nb\nc", "a\nb\nc") <= 1.0
    # n3: spike_peak_hit — 构造单峰 ci>0 的 per_item, 必须命中该峰; 全负则不命中.
    import numpy as _np_test
    _items = [{"id": f"q{i}", "question": "prove uniqueness of the solution", "theorem": ""} for i in range(8)]
    _items += [{"id": f"q8{i}", "question": "iff p then q", "theorem": ""} for i in range(3)]
    _per = {}
    for i in range(8):
        _per[f"q{i}"] = [0.20]  # dim1 (uniqueness) 正信号
    for i in range(3):
        _per[f"q8{i}"] = [-0.10]  # dim0 (iff) 负
    _hit, _pk, _det = spike_peak_hit(_per, _items)
    assert _hit and _pk in (1, "1", "existence"), f"spike must hit uniqueness peak, got {_pk}"
    _per_neg = {k: [-0.20] for k in _per}
    _hit2, _pk2, _ = spike_peak_hit(_per_neg, _items)
    assert not _hit2, "all-negative per_item must not produce a spike"
    # n4: admission_filter — 质量关: Δ < -eps 直接拒; 尖峰关: 正 Δ 且有峰 ci>0 过; 多样性关: 实质 diff 过.
    assert not admission_filter("x", "x", -0.05, _per, _items, eps=0.02)[0], "quality gate must reject Δ<-eps"
    _per_sp = {f"q{i}": [0.20] for i in range(8)}
    _per_sp.update({f"q8{i}": [0.05] for i in range(3)})
    _adm1, _r1 = admission_filter("child-variant a", "parent a", 0.05, _per_sp, _items, eps=0.02)
    assert _adm1, f"positive Δ + spike-diverse must admit, got {_r1}"
    # 无尖峰也无差异 (child==parent 文本) 但 Δ 正 → 拒 (尖峰/多样性二选一都不过).
    _per_flat = {f"q{i}": [0.0] for i in range(8)}
    _per_flat.update({f"q8{i}": [0.0] for i in range(3)})
    _adm2, _r2 = admission_filter("same text", "same text", 0.02, _per_flat, _items, eps=0.02)
    assert not _adm2, f"no spike and no diversity must reject, got {_r2}"
    print(f"✅ facet self-test passed: S₀={len(s0)} chars, 4 sections, "
          f"roundtrip md5={md5(s0)}, non-target slot freeze OK, edit-guard OK, expand-guard OK, "
          f"overlay carry-forward OK (5-sec persist + 4-sec backward-compat), expand targeting OK (skip-meta), "
          f"Bug C regression OK (meta-mirror ci-gate blocks nm_ci<0; primary path keeps nm_ci>0), "
          f"Target-1 xo_section_guard OK (allow_merge passes legit merge, blocks real content loss), "
          f"算子H OK (edit-guard on expand-grown section; classifier meta-后移; strengthen target selection; "
          f"saturation-decision: weaker-variant saturates, stronger/equal/first-time explore), "
          f"方案1 OK (find_covered_failing_type: covered-failing returns highest-freq; no-failure→None; "
          f"skip-meta; exclude guard; deterministic; complementary to find_uncovered_type_for_expand; "
          f"Task#7 gate union-fix: archive-only section visible via _archive_section_union → fire), "
          f"方向B OK (_donor_coverage_types: sig kw ∩ counts>0; select_donors_directed: "
          f"ci>0+md5!=root+coverage>0 gate, coverage-desc ranking; _assemble_multi_grafts: "
          f"multi-graft + freeze-safe + dedup + empty/unmutated→no-graft)")


# ===================================================================
# Facet-Tree (Stage 0): facet-edit 护栏 + archive_tree + 2-parent 交叉
# ===================================================================
# 设计 §2.5/§2.6/§五护栏落地。物理防塌缩 (Phase C REFUTED 0.3703 根因 = 候选塌缩到单目标):
# 每个 facet 分支的变异只编辑该分支对应 slot, 其他 slot 逐字保留 (非目标 slot md5 不变, 否则拒候选)。
FACET_EDIT_SYSTEM = (
    "You are an expert System Prompt Engineer. You make a SMALL, SURGICAL edit to exactly ONE "
    "named section of a mathematical-reasoning skill for theorem-level multiple-choice questions. "
    "You MUST keep every other section byte-for-byte identical."
)

# 单 slot 编辑 prompt: 只重写 {facet} 段的正文, 其余段落逐字原样返回 (护栏由 facet_edit_guard 校验).
FACET_EDIT_PROMPT = """You are editing ONE named section of a mathematical-reasoning skill.

Only edit the section named `{facet}`. Give me ONLY the new body text of that section (the lines of that section, as a markdown list) — no other section, no commentary, no re-stating the header. The rest of the skill is handled separately and MUST NOT be changed.

Your edit must be a CONCRETE improvement to the decision procedures in this section, anchored to the diagnostic failures below. Keep it SMALL and SURGICAL (extend/replace at most one bullet).

## CURRENT SECTION `{facet}`
{slot_body}

## DIAGNOSTIC FAILURES (this skill still gets these wrong)
{diagnostics}

## YOUR OUTPUT
Only the new body text of section `{facet}`.
"""


# ===================================================================
# 算子G: section-expansion (展开型算子) — 治本改动, 突破 facet_edit「单slot单bullet」能力边界
# ===================================================================
# 8-seed 坐实 evolver 显著落后 skillopt (-0.0291, CI上界<0) 后, 代码级诊断证明 facet-tree 的核心算子
# facet_edit 结构上只允许单 slot 内精炼一个 bullet, 物理禁止新增 section. skillopt 的优势恰是 6 个
# 「展开型 section」(决策程序步骤 + 领域特化映射 + 数学示例), 落在该能力边界之外.
# 算子G 给 LLM 一份失败诊断, 让它生成一个**全新** section (决策程序/领域特化), 摆脱「只能改现有段」的
# 约束. 血统干净: 从真 S0(4段) 起步, 新 section 只由失败诊断驱动 (无 skillopt 内容/无人类 seed).
# 见 memory/expand-operator-design-20260827 / memory/facet-tree-capability-ceiling-20260827.
EXPAND_SYSTEM = (
    "You are an expert System Prompt Engineer. You ADD ONE new `## ` section to a "
    "mathematical-reasoning skill for theorem-level multiple-choice questions, targeted at a "
    "specific diagnosed failure the current skill still gets wrong. You MUST NOT touch any "
    "existing section."
)

# 生成新 section prompt: 只输出一个新 `## {section_name}` 块. {existing_sections} = 现有段名清单
# (让 LLM 起个不冲突的名字); {diagnostics} = 失败诊断 (决定新 section 该解决什么).
# {target_type} = 覆盖驱动选定的未覆盖失败类型 (v2 新增, 修 v1 expand 攻 meta 致内容重叠退步).
SECTION_EXPAND_PROMPT = """You are EXPANDING a mathematical-reasoning skill by adding ONE brand-new section.

The skill currently has these sections (any order): {existing_sections}. It still fails on the diagnostics below.

This new section MUST address a SPECIFIC failure type: **{target_type}**.
Do NOT address the "meta-options / strongest provable" failure mode — that is already owned by another section and the skill's existing guidance handles it. Do NOT restate existing guidance. Add a CONCRETE, OPERATIONAL decision procedure or domain-specialized heuristic for the **{target_type}** failure type only — something the current sections do not already cover.

Name the new section yourself (it must NOT duplicate any existing section name above, and should reflect the {target_type} failure type it targets). Give the full body as a markdown list. Keep it focused and under ~1200 chars.

## DIAGNOSTIC FAILURES (the new section must target these — all are {target_type} failures)
{diagnostics}

## YOUR OUTPUT
Only the new section: `## {{section_name}}` followed by its body. No commentary.
"""


# ===================================================================
# 算子H: section-strengthen (加固型算子) — 与 expand (add-only) 互补
# ===================================================================
# expand 在"全非-meta 类型已有专属段"(target is None) 时静默跳过, 即使仍有失败 (efficacy gap).
# 算子H 在该饱和分支触发: 选一个现有段, 让 LLM 在其正文里加/精炼一条 bullet (而非长新段),
# 复用 facet_edit_guard (对任意段名都工作, base 轴或 expand 长出的段均可).
# 血统干净: 强化内容仍由失败诊断驱动 (无 skillopt 内容/无人类 seed). LLM 生成全部文本;
# 模板只提供 meta-启发式 (类比 expand 的"为未覆盖类型加段"提示), 非领域内容.
# 见 memory/bidirectional-test-convergence-20260904.md (干净路径缺第 2 双向轴 = ~0.5pp gap).
STRENGTHEN_SYSTEM = (
    "You are an expert System Prompt Engineer for a mathematical-reasoning LLM solving theorem-level "
    "multiple-choice questions (LiveMathematicianBench). You STRENGTHEN ONE existing section of the "
    "skill by adding or refining a single concrete bullet — you do NOT rewrite the section, do NOT add "
    "new sections, do NOT remove existing bullets."
)

# 加固 prompt: 输出整段新正文 (现有 bullets + 新 bullet). {facet}=目标段名; {slot_body}=现正文;
# {diagnostics}=跨类型最难失败 (LLM 自行识别被分类器埋在 meta 桶里的 existence/uniqueness 模式).
# 通用化提示 (模板, 非手 craft 内容): "若该段已有一条具体决策程序处理某逻辑区分, 而下面失败语义相关但仍错,
# 该段可能需要第二条 *不同* 逻辑区分的互补决策程序 (不同方向 / 不同强度声称). 加一条具体 bullet."
SECTION_STRENGTHEN_PROMPT = """You are STRENGTHENING one existing section of a mathematical-reasoning skill.

You will edit ONLY the section named `{facet}`. Output the FULL new body text of that section (every existing bullet, plus ONE added/refined bullet) as a markdown list. Do NOT output any other section, header, or commentary. The rest of the skill is handled separately and MUST NOT change.

Your edit must be a CONCRETE, OPERATIONAL improvement to the decision procedures in this section, anchored to the diagnostic failures below. Keep it SMALL and SURGICAL (extend/replace at most one bullet).

Generalization heuristic (meta-strategy, not domain content): if this section ALREADY implements a concrete decision procedure for ONE specific logical distinction (e.g. checking one direction of an implication, or proving existence), and the failures below are semantically related but the model still gets them wrong, then this section likely needs a SECOND, DISTINCT decision procedure for a related-but-DIFFERENT logical distinction (a different direction, a different strength claim — e.g. if it already checks forward sufficiency, add backward necessity; if it already checks existence, add uniqueness). Add ONE concrete bullet that does NOT duplicate the existing procedure. Do NOT restate existing guidance; do NOT rewrite the section.

## CURRENT SECTION `{facet}`
{slot_body}

## DIAGNOSTIC FAILURES (this skill still gets these wrong — the hardest remaining ones, across all failure types)
{diagnostics}

## YOUR OUTPUT
Only the full new body text of section `{facet}` (existing bullets + your one added/refined bullet).
"""


def section_expansion_guard(parent_slots, expected_key, child_text, max_sections=10):
    """把 expand LLM 生成的新 section child_text 追加进 parent_slots, 构造性保证「只新增不改」。

    与 facet_edit_guard 的「非目标 slot md5 不变」相比, 本 guard 更严格的是「只 add 不 replace」:
      - child_text 必须恰好含 1 个 `## ` section (facet_decompose 拆出), 且名字 == expected_key;
      - 该名字必须不在 parent_slots 中 (新增, 不是改写);
      - 现有 slot 全部逐字节不变 (纯构造性 copy, 天然满足);
      - 新增后总段数 <= max_sections (防膨胀).
    返回 (new_slots, ok): ok=False 时 new_slots 未改动, 调用方拒绝该候选。
    """
    if expected_key in parent_slots:
        return dict(parent_slots), False  # 名字与现有段冲突 → 拒 (不是新增)
    _, child_slots = facet_decompose(child_text)
    if len(child_slots) != 1:
        return dict(parent_slots), False  # 空 / 0段 / 多段 → 拒
    # 拿唯一新段 (LLM 可能吐不同名字: 用 expected_key 强制改名, 但解析前先取实际段)
    new_key = next(iter(child_slots))
    if new_key != expected_key:
        return dict(parent_slots), False  # 名字不一致 → 拒
    body = child_slots[new_key].strip()
    if not body:
        return dict(parent_slots), False  # 空 body → 拒
    if len(parent_slots) + 1 > max_sections:
        return dict(parent_slots), False  # 达段数上限 → 拒 (防膨胀)
    new_slots = dict(parent_slots)
    # 契约对齐 facet_compose: body 以单个 \n 结尾 (天然分隔下一段), 顶格无前导换行.
    body = body.lstrip("\n")
    body = body.rstrip("\n") + "\n" if body.strip() else body
    new_slots[expected_key] = "\n" + body
    return new_slots, True


def facet_edit_guard(parent_slots, target_key, child_text):
    """把 patcher 对 target_key 段的新正文 child_text 替换进 parent, 校验非目标 slot md5 不变。

    child_text 是 patcher 输出的目标段新正文(整 skill 或单段都行; 拿 target_key 段读回)。
    返回 (new_slots, ok): ok=False 表示非目标 slot 被破坏 → 调用方拒绝该候选 (护栏)。
    """
    new_slots = dict(parent_slots)
    # 剥掉可能的 `## {target}` 标题 / ``` 围栏, 得到纯正文
    body = child_text.strip()
    # 若输出含 ## 段: 只有 target 段时取该段正文 (patcher 吐了 `## {target}` + 正文);
    # 含其他段则跨 slot 编辑 -> 护栏拒绝 (物理防塌缩).
    _, child_slots = facet_decompose(child_text)
    if target_key in child_slots and len(child_slots) == 1:
        body = child_slots[target_key]
    elif child_slots:
        return new_slots, False
    # 否则 child_text 无 ## (纯正文 markdown list), 直接用作目标段正文.
    # 只剥开头换行 (正文顶格), 并规范化为「以单个 \n 结尾」: 结尾换行是与下一 `## ` 段之间的
    # 天然分隔 (切段契约: body 自带结尾换行), patcher 常输出 `...statement.` 后无换行, 若这里
    # 保留剥尾会把 `## Final Answer` 粘进 Hypotheses 正文 → Final Answer 段整体丢失。
    body = body.lstrip("\n")
    body = body.rstrip("\n") + "\n" if body.strip() else body
    new_slots[target_key] = "\n" + body
    # 护栏校验: 非目标 slot 必须逐字节未变
    for k in parent_slots:
        if k != target_key and md5(new_slots[k]) != md5(parent_slots[k]):
            return new_slots, False
    return new_slots, True


CROSSOVER_COHERE_PROMPT = """You are reconciling a system prompt (skill) assembled from TWO parent skills for a mathematical-reasoning LLM.

The skill below was built by taking parent A as the base and grafting in parent B's section `{b_key}` (and keeping parent A's section `{a_key}`). The two parents evolved independently, so the grafted section may now OVERLAP or CONFLICT with the rest of the skill.

This skill uses a FACET-TREE structure: each `## ` section is an evolvable slot. Your job is to make the sections COHERENT AND COLLABORATIVE — two sections giving related guidance should reinforce each other, not contradict or redundantly repeat.

Reconcile while KEEPING operational content and ALLOWING 协同 (coordinated restructuring):
1. Resolve TRUE CONFLICTS — two decision procedures that push toward OPPOSITE answers on the same situation. For each conflict, keep the concrete step-by-step test that is more precise, and drop only the contradicting clause.
2. You MAY merge two sections when they give genuinely redundant guidance (same decision procedure, same trigger). When you merge, keep the MORE PRECISE concrete step-by-step test and drop only the duplicate clause — do NOT lose any distinct decision procedure.
3. You MAY add a short BRIDGING section when two sections interact (e.g., "when condition X (section A) holds, prefer procedure Y (section B)"). Bridging sections make the facets 协同 — that is the point of cross-axis inheritance.
4. You MAY reorder sections to group related guidance. Keep the decision flow logical: general scope → specific decision procedures → final answer.
5. Keep EVERY concrete decision procedure (step-by-step test) — these are the valuable parts. Merging is to remove DUPLICATION, never to drop a distinct procedure.
6. The `## Final Answer` section is FROZEN — keep a `## Final Answer` section and output its body byte-for-byte unchanged (it may be reordered to remain the LAST section).
7. Keep under ~5000 chars. Do NOT reference specific problem answers or constants.
8. Output ONLY the reconciled skill text. START directly with the skill title line (`# ...`).

## REFERENCE: sections present in the assembled skill (you may merge/reorder/bridge among these)
{section_headers}

## ASSEMBLED SKILL (parent A + grafted {b_key} from parent B)
{skill}
"""


def _xo_section_guard(child_slots, parent_a_slots, freeze, allow_merge=False):
    """靶点1 (2026-09-03): recombine_facets 的段数守卫 (纯函数, 零 API 可测).

    cohere-prompt (CROSSOVER_COHERE_PROMPT) 明确允许 "MAY merge two sections", 即合法的 2→1 合并
    会降低 child 段数。旧守卫 `len(child) < len(parent_a)` 硬拒任何降段数 → 合法 merge 被误杀
    (evolver_log 4/7 crossover guard/cohere-fail 主因之一)。

    两种模式:
      allow_merge=False (旧行为): child 总段数 < parent_a 总段数 即拒 (禁降段数).
      allow_merge=True  (语义守卫): 降段数本身不拒, 仅当非 freeze 段总数减少 (= 真丢内容) 才拒.
        2 段合并成 1 段 (非 freeze 段总数不变) → 放行; 丢掉一个整段 (非 freeze 段总数降) → 拒.
      freeze 段的存活单独由调用方 (recombine_facets) 的 freeze 回填守卫负责, 本函数只看段数.

    返回 True = 通过 (可继续), False = 拒绝 (回退纯组装).
    """
    if allow_merge:
        a_nfreeze = [k for k in parent_a_slots if k != freeze]
        c_nfreeze = [k for k in child_slots if k != freeze]
        # 非 freeze 段总数不得减少 (2→1 合并允许, 整段丢失拒绝).
        return len(c_nfreeze) >= len(a_nfreeze)
    else:
        # 旧行为: 总段数不得减少 (禁任何降段数, 含合法 merge).
        return len(child_slots) >= len(parent_a_slots)


def recombine_facets(parent_a_slots, parent_b_slots, a_key, b_key, freeze,
                     llm_stage="evolver_facet", retries=0, allow_merge=False):
    """2-parent facet 交叉 (design 算子C, 修 bug#2): 父本 A 全 slot 为基 + 父本 B 的 b_key 段移植进来,
    再做一次 LLM 一致性 pass (DIVE 的 conflict resolution) 消解跨 slot 冗余/冲突。

    比 DIVE (whole-skill 合成) 更细: slot 粒度组装 (facet_compose) + 单次 coherence pass。
    a_key 仅作提示上下文 (标注 A 贡献的强 facet); 实际写入只有 b_key (来自 B)。其余 slot 沿用 A。
    冻结 slot 由调用方用 facet_edit_guard 语义校验 (本函数产出的 slots → compose → guard 比对 freeze md5)。
    返回 (new_slots, ok): ok=False = 一致性 pass 失败 / 冻结 slot 被破坏 → 调用方拒绝。

    靶点1 (2026-09-03): 旧版 cohere-LLM 单次失败即弃 → evolver_log 显示 4/7 crossover 死在结构守卫
    (guard/cohere-fail), 从未进 eval 评测。两个改动:
      (1) `retries`: cohere pass 失败 (LLM 异常/clean 空/decompose 空) 时, 不立即放弃, 重试 N 次
          (换温度扰动 re-prompt), 给 LLM 多一次产出合规结构的机会。默认 0 = 旧行为。
      (2) `allow_merge`: 旧 L1154 守卫 `len(child) < len(parent_a)` 硬拒任何降段数, 但 cohere-prompt
          明确允许 "MAY merge two sections" → 合法 merge 被误杀。改为语义守卫: 降段数本身不拒,
          仅当父本 A 的非 freeze 段内容确有丢失 (非 freeze 段数 < 原非 freeze 段数) 才拒。
          即: 2 段合并成 1 段 (非 freeze 段总数不降) 可过; 丢掉一个整段 (非 freeze 段总数降) 才拒。
    """
    slots = dict(parent_a_slots)
    if b_key in parent_b_slots and b_key != freeze:
        slots[b_key] = parent_b_slots[b_key]
    # 组装后做一次 LLM coherence pass (DIVE conflict resolution). title 在主循环 compose 时自带.
    assembled = facet_compose("", slots)
    # section headers 清单从 parent_a 实际段名取 (slots dict 保序, 无需 regex). v2: 参数化
    # 硬编码的 4-header 列表 → 让含 expanded section 的 grown skill 也能作 parent A 进 crossover
    # (落实「对不同轴交叉测试」, 修 v1 4-header 屏障致 grown skill 被 guard 5≠4 拒).
    section_headers = "\n".join(f"- `## {k}`" for k in parent_a_slots)
    # 靶点1: cohere pass 失败时重试 (re-prompt, 温度扰动), 不单次即弃.
    # 重试只在"结构失败"(LLM异常/clean空/decompose空)时发生; 段数守卫/freeze守卫失败不重试
    # (那是 LLM 产出了但内容不合, 重试同 prompt 多半仍不合 — 由主循环换 parent 对重试).
    set_temp(GEN_TEMP)
    child_slots = None
    for _attempt in range(retries + 1):
        try:
            cohered = llm(PATCHER_SYSTEM, CROSSOVER_COHERE_PROMPT.format(
                a_key=a_key, b_key=b_key, skill=assembled, section_headers=section_headers),
                max_tokens=4096, stage=llm_stage)
        except Exception:  # noqa: BLE001
            if _attempt < retries:  # 还有重试机会 → 扰动温度再试
                set_temp(GEN_TEMP + 0.15 * (_attempt + 1))
                continue
            return slots, False  # 重试耗尽 → 回退纯 slot 组装 (仍可过 guard, 但无 coherence)
        c = clean_skill(cohered, assembled)
        if not c:
            if _attempt < retries:
                set_temp(GEN_TEMP + 0.15 * (_attempt + 1))
                continue
            return slots, False
        # 解析 cohere 后的 skill 回 slots; 校验结构完整 (含所有原 slot)
        _, _child = facet_decompose(c)
        if not _child:
            if _attempt < retries:
                set_temp(GEN_TEMP + 0.15 * (_attempt + 1))
                continue
            return slots, False  # 解析失败 → 回退纯组装
        child_slots = _child
        break  # 成功拿到合规结构, 跳出重试循环
    if child_slots is None:
        return slots, False
    # break-② 修复: 放宽结构守卫 (旧 set(child)==set(parent_a) 禁止任何增删并段 = 反协同).
    # 新守卫只三道: ① 段数不减少 (允许 merge/bridge 增段, 禁止丢段);
    #              ② 关键段不丢 (parent_a 的非 freeze 段必须各有内容存活, 允许改名/合并但需保留语义内容);
    #              ③ Final Answer 冻结 (存在且内容不变, 由下方 freeze 回填强制).
    # 注意: 允许 LLM 合并/重排/加桥接段 — 这是协同多编辑能力 (skillopt 协同10段的关键).
    #
    # 靶点1 (2026-09-03): 旧 L1154 `len(child) < len(parent_a)` 硬拒降段数, 与 cohere-prompt 的
    # "MAY merge two sections" 冲突 → 合法 merge 被误杀 (evolver_log 4/7 guard/cohere-fail 主因之一).
    # 抽出为 _xo_section_guard 纯函数 (零 API 可测); allow_merge=True 时改语义守卫:
    # 降段数本身不拒, 仅当非 freeze 段总数减少 (= 真丢内容) 才拒.
    if not _xo_section_guard(child_slots, parent_a_slots, freeze, allow_merge):
        # 段数/内容丢失 → 拒, 回退纯组装
        return slots, False
    # 冻结 slot 强制还原 (构造性消除 FAIL-④: LLM 复述 Final Answer 改写措辞 → md5 变被拒).
    # 冻结槽语义本就冻结, 无论 coherency 是否动它, 一律用 parent_a 的原值覆盖回填, 再过下界 md5 守卫.
    # break-②: 允许合并/重排后, freeze 段可能被 LLM 改名/合并掉 → 此时硬拒 (Final Answer 必须独立存活).
    if freeze in parent_a_slots and freeze in child_slots:
        child_slots[freeze] = parent_a_slots[freeze]
        if md5(child_slots[freeze]) != md5(parent_a_slots[freeze]):
            return child_slots, False  # 理论不可达 (刚覆写); 留作防御性守卫
    elif freeze in parent_a_slots:
        return slots, False  # freeze 段被 LLM 合并/改名丢了 → 拒, 回退纯组装
    return child_slots, True


def _donor_coverage_types(body, counts):
    """方向 B 辅助 (纯函数): donor section body 覆盖哪些「当前仍在失败」的失败类型.

    覆盖判定与 _TYPE_COVERAGE_SIG 同口径 (body 含签名关键词 = 该段提供该类型的决策程序)。
    只算 counts[lbl] > 0 的类型 (仍在失败 = 有 efficacy gap, 值得定向补)。
    返回 set[label], 空集合 = 该 donor 不覆盖任何当前失败类型 (定向价值低)。
    """
    bl = body.lower()
    covered = set()
    for lbl, cnt in counts.items():
        if cnt <= 0:
            continue
        sigs = _TYPE_COVERAGE_SIG.get(lbl, ())
        if sigs and any(s in bl for s in sigs):
            covered.add(lbl)
    return covered


def select_donors_directed(archive, root_slots, freeze, counts, topk=2):
    """方向 B: 定向 donor 选取 — 按 section body 对「当前失败分布 counts」的覆盖匹配度排序,
    而非纯 val_ci (select_section_donors 的口径).

    动机 (北星 verdict 坐实): theo_iter6 warm-start run 的 archive 内 donor 虽 val_ci 高, 但
    其覆盖的失败类型与「当前仍在失败」的类型不对齐 → 2-parent 嫁接未补上 skillopt 满分而
    evolver 丢分的题 (202602:50, 202511:28 等)。定向 donor 让多 parent 嫁接对准当前失败缺口。

    门控 (与 select_section_donors 一致, 保证 donor 可信):
      1) val_ci > 0 — ci>0 守门, 挡负 CI 噪声段.
      2) 段 md5 != root 段 md5 — 排除未真正变异的零改动段.
      3) 段 != freeze — 冻结段永不作 donor.
      4) [NEW] 段覆盖 ≥1 个当前失败类型 (_donor_coverage_types 非空) — 定向价值守门.
         不覆盖任何当前失败类型的段即使 val_ci 高也不入选 (定向语义: 对准失败缺口).

    排序键: (覆盖的当前失败类型数 desc, val_ci desc) — 先按覆盖广度 (补的洞多者优先),
    再按 val_ci (同覆盖广度取历史更可信者)。这把「覆盖未命中题的失败类型」的 donor 排到前面。

    返回 list[(sec_key, _ArchiveNode)] (按定向匹配度降序, 至多 topk 个); [] = 无定向 donor.
    与 select_section_donors 的 dict 返回不同 — 多 parent 嫁接需要的是「跨不同 section 的
    最佳 donor 列表」(每段最多取 1, 但跨段合并成平铺 pool), 而非每段 topk dict.
    """
    best_per_sec = {}   # {sec_key: [(coverage_count, val_ci, node)]}
    for nd in archive:
        if nd.val_ci <= 0:
            continue
        for sec_key, body in nd.slots.items():
            if sec_key == freeze:
                continue
            if md5(body) == md5(root_slots.get(sec_key, "")):
                continue   # 未真正变异段排除
            cov = _donor_coverage_types(body, counts)
            if not cov:
                continue   # 不覆盖任何当前失败类型 → 定向价值 0, 不入选
            best_per_sec.setdefault(sec_key, []).append((len(cov), nd.val_ci, nd))
    # 每段取定向匹配最优 1 个 (coverage desc, val_ci desc)
    pool = []
    for sec_key, cands in best_per_sec.items():
        cands.sort(key=lambda x: (-x[0], -x[1]))
        pool.append((sec_key, cands[0][2]))
    # 跨段再按定向匹配度排序 (该段 donor 的 coverage desc, val_ci desc), 取 topk
    # coverage 值用段内最优 donor 的 (段唯一, 已取 1)
    sec_cov = {}
    for sec_key, nd in pool:
        cov = _donor_coverage_types(nd.slots[sec_key], counts)
        sec_cov[sec_key] = len(cov)
    pool.sort(key=lambda pair: (-sec_cov[pair[0]], -pair[1].val_ci))
    return pool[:topk]


def _assemble_multi_grafts(base_slots, donors, freeze):
    """方向 B 辅助 (纯函数, 零 API 可测): 多 donor 段嫁接组装.

    base 全段保留 + 各 donor 的 sec_key 段覆盖/新增. 门控:
      - sec_key == freeze → 跳过 (冻结段不嫁接).
      - 同 sec_key 只取第一个 donor (去重, 避免重复覆盖).
      - donor 段 md5 == base 同名段 md5 (未变异) → 跳过 (无变化不算嫁接).
      - donor 段 md5 == base.get(sec_key,"") (base 无此段时 == "") → 变异才嫁接.

    返回 (grafted_slots, grafted_list): grafted_slots = 组装后的 slots;
    grafted_list = [(sec_key, donor_label)] 实际嫁接记录. 空 grafted_list = 无有效嫁接.
    """
    slots = dict(base_slots)
    grafted = []
    seen_sec = set()
    for sec_key, nd in donors:
        if sec_key == freeze or sec_key in seen_sec:
            continue
        if sec_key in nd.slots and md5(nd.slots[sec_key]) != md5(base_slots.get(sec_key, "")):
            slots[sec_key] = nd.slots[sec_key]
            grafted.append((sec_key, nd.facet_label))
            seen_sec.add(sec_key)
    return slots, grafted


def recombine_facets_multi(base_slots, donors, freeze,
                           llm_stage="evolver_crossover_mp", retries=0, allow_merge=True):
    """方向 B: 多 parent facet 交叉 — 把 3+ donor 的不同 section 段同时嫁接进 base,
    再一次 LLM coherence pass 消解跨段冗余/冲突/补桥接.

    vs recombine_facets (2-parent 单段嫁接): 一次嫁接多个不同 section, 覆盖更多失败类型 —
    北星 verdict 显示 2-parent 单段嫁接补不上 skillopt-perfect 而 evolver 丢分的题,
    多段同时嫁接 (定向选的 N 个 donor 各补一个不同失败类型) 才有望覆盖。

    语义约束:
      - base_slots: 基底 (通常 = 当前 best_node.slots), 全段保留作 coherence 底座.
      - donors: list[(sec_key, _ArchiveNode)] — 各 donor 贡献其 sec_key 段 (来自 donor.slots[sec_key]).
        sec_key 可能 == base 已有段 (替换该段为 donor 的更强变体) 或 base 新段 (扩轴)。
        自动去重: 同一 sec_key 只取第一个 donor (避免重复覆盖)。
      - freeze: 冻结段, 嫁接不碰, coherence 后强制回填 (构造性消除改写).

    coherence pass 复用 CROSSOVER_COHERE_PROMPT (section_headers 列出所有 donor 段名),
    guard 复用 _xo_section_guard (allow_merge) + freeze 回填守卫 (与 recombine_facets 同口径)。
    失败 (cohere 异常 / clean 空 / decompose 空 / guard 拒 / freeze 丢) → 回退纯组装 slots (无 coherence)。

    返回 (new_slots, ok): ok=False 时 new_slots = 纯嫁接组装 (无 coherence, 仍可过 guard 进 eval)。
    """
    if not donors:
        return dict(base_slots), False
    # 组装: base 全段 + 各 donor 的 sec_key 段覆盖/新增 (同 sec_key 只取第一个) — 纯函数 _assemble_multi_grafts.
    slots, grafted = _assemble_multi_grafts(base_slots, donors, freeze)
    if not grafted:
        # 无有效嫁接 (所有 donor 段 == base 或 == freeze) → 无重组, 回退
        return dict(base_slots), False
    assembled = facet_compose("", slots)
    # section_headers 列出基底 + 所有嫁接段 (让 LLM 知道哪些段来自不同 parent 需协同)
    section_headers = "\n".join(f"- `## {k}`" for k in slots)
    # 多段嫁接的 b_key 描述 (CROSSOVER_COHERE_PROMPT 的 {b_key} 占位用)
    b_key_desc = ", ".join(s for s, _ in grafted)
    a_key_desc = "base (current best)"
    set_temp(GEN_TEMP)
    child_slots = None
    for _attempt in range(retries + 1):
        try:
            cohered = llm(PATCHER_SYSTEM, CROSSOVER_COHERE_PROMPT.format(
                a_key=a_key_desc, b_key=b_key_desc, skill=assembled,
                section_headers=section_headers),
                max_tokens=4096, stage=llm_stage)
        except Exception:  # noqa: BLE001
            if _attempt < retries:
                set_temp(GEN_TEMP + 0.15 * (_attempt + 1))
                continue
            return slots, False
        c = clean_skill(cohered, assembled)
        if not c:
            if _attempt < retries:
                set_temp(GEN_TEMP + 0.15 * (_attempt + 1))
                continue
            return slots, False
        _, _child = facet_decompose(c)
        if not _child:
            if _attempt < retries:
                set_temp(GEN_TEMP + 0.15 * (_attempt + 1))
                continue
            return slots, False
        child_slots = _child
        break
    if child_slots is None:
        return slots, False
    # 段数守卫: 多段嫁接后非 freeze 段总数不得减少 (允许合并, 禁丢内容).
    # 注意: base_slots 作 parent_a 口径 (多 parent 的「基底」= parent A).
    if not _xo_section_guard(child_slots, base_slots, freeze, allow_merge=allow_merge):
        return slots, False
    # freeze 回填 (与 recombine_facets 同口径)
    if freeze in base_slots and freeze in child_slots:
        child_slots[freeze] = base_slots[freeze]
        if md5(child_slots[freeze]) != md5(base_slots[freeze]):
            return child_slots, False
    elif freeze in base_slots:
        return slots, False
    return child_slots, True


def select_section_donors(archive, root_slots, freeze, topk=1):
    """section 级 donor 选取 (修 break-①: 放宽单轴叶限制, 让 expand/xo/overlay 复合节点的新段进遗传池).

    核心改动 vs 旧 select_axis_donors:
      - 旧: `if nd.facet_label not in branch_axes: continue` 把 expand/xo/overlay 节点全挡在 donor 池外
            → expand 长出的新 section 永远进不了 overlay/crossover 的遗传池 (无法被继承).
      - 新: donor key 取"节点真正携带的 section 名"而非 facet_label. 对单轴叶 (facet_label==段名)
            语义不变; 对复合节点 (facet_label="expand:Quantifier..." / "xo:A+B" / "overlay"),
            把它携带的、已变异过的非-root section 都作为可继承 donor 暴露出来.

    门控 (与 TREE-BEST 守门口径一致: delta 主排序 + ci>0 守门 + 段确经变异):
      1) val_ci > 0 — ci>0 守门 (对齐 L1416 `cl > 0`), 挡掉负 CI 噪声段.
      2) 段 md5 != root 段 md5 — 排除未真正变异的零改动段 (含 root 叶整段 S0 原文).
         facet 往返逐字节无损 (L696 `_facet_roundtrip_assert`), md5 比对精确可靠.
      3) 段 != freeze — 冻结段永不作 donor (overlay/crossover 一律不嫁接 freeze).

    历史探索 (用户: "历史中的组合可能会更好"): topk>1 时每段保留 top-K 历史最佳 (按 val_ci 排序),
    供 crossover/overlay 随机采样而非永远只用 top-1. 单一 top-1 会把"次优但与其他轴协同更好"的
    历史节点永久排斥; top-K 让树状结构允许回退和复现路径变化.

    返回 {section_key: [_ArchiveNode, ...]} (每段最多 topk 个, 按 val_ci 降序); 空列表 = 无可信改进段.
    """
    best_per_sec = {}   # {sec_key: [(val_ci, node), ...]}
    for nd in archive:
        if nd.val_ci <= 0:
            continue
        for sec_key, body in nd.slots.items():
            if sec_key == freeze:
                continue
            if md5(body) == md5(root_slots.get(sec_key, "")):
                continue   # 该段未真正变异 (== S0 原文) → 排除
            best_per_sec.setdefault(sec_key, []).append((nd.val_ci, nd))
    donors = {}
    for sec_key, cands in best_per_sec.items():
        cands.sort(key=lambda x: -x[0])   # val_ci 降序
        donors[sec_key] = [nd for _, nd in cands[:topk]]
    return donors


def select_axis_donors(archive, branch_axes, root_slots):
    """[deprecated: 保留向后兼容] 每 axis_key 取 val_delta 最大且 val_ci>0 单轴叶作 donor.
    新逻辑见 select_section_donors (放宽 break-①). 本函数仅保留给 --no_relax_donor 回退路径与既有
    self_test 调用, 不应在新流程使用."""
    donors = {}
    for nd in archive:
        if nd.facet_label not in branch_axes:   # 只取单轴叶
            continue
        if nd.val_ci <= 0:                      # ci>0 守门 (对齐 L1416 `cl > 0`)
            continue
        if md5(nd.slots.get(nd.facet_label, "")) == md5(root_slots.get(nd.facet_label, "")):
            continue                            # 本轴段未真正变异 → 排除
        cur = donors.get(nd.facet_label)
        if cur is None or nd.val_delta > cur.val_delta:   # delta 主排序 (对齐 L1416 `cd > ...`)
            donors[nd.facet_label] = nd
    return donors


class _ArchiveNode:
    """archive_tree 节点 = {skill, facet_label, lineage(parent 指针), val_delta, val_ci, slots}.
    MAP-Elites illumination (设计 §2.3): 输家不删, 保 facet 专家。Stage 0 最小版节点即此。
    """
    __slots__ = ("skill", "slots", "facet_label", "parent", "val_delta", "val_ci")

    def __init__(self, skill, slots, facet_label, parent=None, val_delta=0.0, val_ci=0.0):
        self.skill = skill
        self.slots = slots
        self.facet_label = facet_label
        self.parent = parent
        self.val_delta = val_delta
        self.val_ci = val_ci


def _facet_non_native_s0_ratio(node, freeze, root_slots):
    """诊断指标: 节点 slots 中"非本轴非 freeze 且 = root_slots(S0) 原文"的段占比.

    修法1(纯叠加 best) 的累积进化判据:
      - 现状 best = 单轴最优 + S0 其余 → 排除本轴后其余非冻结段全 S0 → ratio ≈ 1.0 (累积失效)
      - 修法1 叠加出"多轴同改"best → 被替换段 ≠ S0 → ratio 下降 (< 1.0, 理想 → 0)

    门槛:
      - 排除 freeze (恒 = root, 会拉高占比失真)
      - 排除本轴 (node.facet_label): 本轴段本就该是改过的, 计入会稀释读数.
        best 命中 overlay 时 facet_label="overlay" 不在 branch_axes → 无本轴豁免,
        全部非冻结段计入分母 → ratio 能降到 0 (这是 overlay multi-native 的关键性质).
      - 分母为 0 边界 (极端配置) → 返回 1.0 (无信息, 视为"没累积"), 不除零.
    """
    native = node.facet_label
    denom = 0
    s0_cnt = 0
    for key, body in node.slots.items():
        if key == freeze or key == native:
            continue
        denom += 1
        if md5(body) == md5(root_slots.get(key, "")):
            s0_cnt += 1
    if denom == 0:
        return 1.0
    return s0_cnt / denom


def _overlay_assemble(title, root_slots, best_node_slots, donors, freeze):
    """overlay 算子E 的 slot 组装 (v2 抽出为独立纯函数, 供 _self_test_facet 验证 carry-forward).

    构造顺序 (保向后兼容):
      1) root_slots 为基底 (root 轴语义不变); 无 donor 轴隐式回落 S0 (旧行为);
      2) 【v2 新增】 carry-forward: best_node.slots 中超出 root_slots 的已展开 section 携带过来
         (修 v1 根因2: expand 长出的第5段被下一代 overlay 从 root_slots 重建抹掉);
      3) donor 轴嫁接: 每段历史 best 覆盖对应 root 轴. 【break-① 修复】 donor 现可含 expand/xo
         复合节点携带的 expanded-section key (与 root 轴 key-space 不再不相交). 这与 carry-forward
         合流: 新段要么靠 best_node carry-forward 携带, 要么靠 donor 嫁接 — 两条遗传路都通.
    donors 接口: {section_key: _ArchiveNode} (单节点, 已由调用方从 select_section_donors 的 top-K
      列表里取 top-1 或采样). 旧 {axis_key: node} 格式仍兼容 (同为 dict 迭代).
    返回 (overlay_slots, overlay_skill, ov_slots_chk, ok):
      ov_slots_chk = 对 overlay_skill 再 decompose 回来 (catch compose 丢段);
      ok = root 段全存活 且 compose 后段集合 == 组装基底 (key-set-only, 无 md5 检查).
    无 md5 检查是刻意的: facet_compose L684 对非末段补 \\n, expanded 段追加后 Final Answer
    变非末段 → md5 变 → 任何 freeze-md5 检查会误拒 overlay (Plan-agent 发现的关键风险).
    expanded 段完整性由构造性 copy 保证 (直接取 best_node_slots 引用), 无需 md5."""
    overlay_slots = dict(root_slots)
    # v2: carry-forward 已展开 section (root_slots 之外的 key)
    for k, v in best_node_slots.items():
        if k not in root_slots:
            overlay_slots[k] = v
    for ax_key, dn in donors.items():
        if ax_key != freeze and ax_key in dn.slots:
            overlay_slots[ax_key] = dn.slots[ax_key]
    overlay_skill = facet_compose(title, overlay_slots)
    _, ov_slots_chk = facet_decompose(overlay_skill)
    ok = (set(root_slots.keys()).issubset(set(ov_slots_chk.keys()))
          and set(ov_slots_chk.keys()) == set(overlay_slots.keys()))
    return overlay_slots, overlay_skill, ov_slots_chk, ok


def partition_diag(diag_items, n_branches, seed=42):
    """DIVE bootstrap (design 算子A): 把 diag_items 切成 n_branches 个不相交子集, 每分支诊断自己的子集。

    seeded random partition → 确定性可复现 (resume 安全)。各分支看到不同失败环境 → 演化方向分叉
    (异构环境 × 结构特化 = N=4 硬约束下"基因库丰富性"的来源之一)。尽量均分, 余数按序补到前几个
    子集 (前 len%n 个子集各多 1 题)。bootstrap 关时调用方应直接用全 diag_items (消融 E 组),
    不调本函数。返回 list[list], 长度 = n_branches; 并集 = 原 diag_items, 两两不相交。
    """
    if n_branches <= 1 or not diag_items:
        return [list(diag_items)]
    import random as _rng
    idx = list(range(len(diag_items)))
    _rng.Random(seed).shuffle(idx)
    parts = [[] for _ in range(n_branches)]
    for i, pos in enumerate(idx):
        parts[i % n_branches].append(diag_items[pos])
    return parts


def sample_mutation_parent(archive, facet, leaf, seed, t, eps=0.3):
    """方向E 落点②: 同 facet archive 精英种群加权采样作突变起点 (打破单 lineage)。

    用户本意 = "不只 keep-best, 历史好表现通过交叉遗传变异保留"。当前主循环每 branch 从单 leaf
    (incumbent) 突变 → 单 lineage, archive 的 keep-losers 只在交叉时偶用。本函数把突变起点从单 leaf
    扩到同 facet archive 精英种群 (含 leaf 及同轴历史 elite), population-based (类 PromptBreeder/DIVE)。

    采样策略:
      - ε-greedy: prob=eps 返回 leaf (incumbent 探索), 否则种群加权采样;
      - 权重 = max(0, val_ci) + 1e-6 (val_ci>0 的可信节点优先, ci<=0 者仍保留极小权重不丢多样性);
      - within-facet (非全 archive): facet_edit_guard 只编辑 slot `ax`, 其余 slot 必须是 parent 携带的
        该轴历史累积才语义自洽; 跨 facet parent 会把别的轴结构带进本轴 guard, 破坏单 slot 编辑约束。
    gen1 安全退回: 同 facet archive 仅 {根叶} (len<=1) → 返回 leaf, 等价单 lineage (与交叉 gating 一致)。
    确定性: seeded (seed*1000+t) → resume 可复现同一采样序列。
    """
    pool = [nd for nd in archive if nd.facet_label == facet]
    if len(pool) <= 1:  # gen1 无选择空间 → 退回单 lineage
        return leaf
    import random as _r
    _rng = _r.Random(seed * 1000 + t)
    if _rng.random() < eps:  # ε-greedy 探索 incumbent
        return leaf
    ws = [max(0.0, nd.val_ci) + 1e-6 for nd in pool]
    return _rng.choices(pool, weights=ws, k=1)[0]


def build_archive_tree(skill, facet_axes):
    """从 init skill 建根节点 + 每 facet 轴一个冷启动分支 (Stage 0 关涌现, 分支数固定)。"""
    title, slots = facet_decompose(skill)
    root = _ArchiveNode(skill, slots, "root")
    return title, root


# Step1 N-parallel 突变: 每分支每代并发生 N 个 facet-edit 候选的多样性提示角度.
# 设计哲学 = DIVE 异构环境下放: 同一失败信号, 让 N 个候选从不同「修补角度」出发, 扩搜索方差
# (正面打 8-25 暴露的 t=2.935 边际 / 修 bug 后 t=-5.89 退步 — 搜索方差是否够).
# 角度互斥可叠加: 一个候选可拿多个角度, 但 N 个候选组合应覆盖所有角度.
# 角度源 = 对诊断失败的「反操作」谱 (诊断说怎么错的, 角度就说从哪修), 非凭空.
_FACET_EDIT_ANGLES = [
    # 0: baseline (无角度注入) — 保 N=1 时与既有单候选字节一致
    "",
    # 1~N: 变体角度
    "ANGLE: Rank options by *provability strength* BEFORE committing. "
    "The strongest provable statement wins; if two seem provable, pick the one with fewer hidden assumptions.",
    "ANGLE: Hunt the counterexample FIRST. For each option, try to construct the minimal case where it fails; "
    "only options surviving all constructed cases survive your decision procedure.",
    "ANGLE: Eliminate by *necessary-condition violation*. List each option's required hypothesis; "
    "if the problem's givens violate any, strike it before deeper analysis.",
    "ANGLE: Distinguish 'true' from 'strongest provable'. An option can be true yet not the answer "
    "because a weaker condition already suffices — prefer the *minimal sufficient* claim.",
    "ANGLE: Check boundary/edge ordering. Re-examine options whose truth flips at equality or limit cases; "
    "these are where theorem-level precision is decided.",
    "ANGLE: Trace the proof obligation. For the *intended* answer, write the 2-step proof skeleton; "
    "if the skeleton has a gap, the answer is likely a distractor.",
]


def gen_facet_candidates(leaf, ax, failures, n, system, prompt_tmpl, gen_temp, diversity="auto"):
    """Step1: 单分支并发生 n 个 facet-edit 候选, 各过 clean_skill + facet_edit_guard.

    多样性来源 (diversity): auto=变体角度提示注入 prompt 的 {diagnostics} 前缀 (默认, 保 GEN_TEMP
    固定 → 可真并行, 不抢 set_temp 全局锁); temp=每候选 set_temp 扰动 (串行, 失并行, 仅消融对照).

    返回 list[(cand_slots, angle_idx)]: length <= n (guard 拒掉的不返回, 故无 ok 标志位).
    每个候选拿到的 {diagnostics} = base_diag 前置 [ANGLE i] 行 (i>0 时) — 同源失败信号, 不同修补角度.
    N=1 且 diversity=auto angle[0]="" → prompt 与既有单候选字节一致 (零回归).
    """
    diag = make_diagnostics(failures) if failures else "(no diagnostic failures)"
    out = []
    # temp 扰动模式: 串行 set_temp (chat_target 读全局, 抢锁失并行; 仅消融)
    if diversity == "temp":
        temps = [gen_temp] + [min(1.2, gen_temp + 0.1 * i) for i in range(1, n)]
        for i in range(n):
            set_temp(temps[i])
            ep = prompt_tmpl.format(facet=ax, slot_body=leaf.slots.get(ax, ""), diagnostics=diag)
            raw = llm(system, ep, max_tokens=2048, stage=f"evolver_facet_p{i}")
            cs = clean_skill(raw, leaf.skill)
            cslots, cok = facet_edit_guard(leaf.slots, ax, cs)
            if cok and cslots is not None:
                out.append((cslots, i))
        set_temp(gen_temp)  # 复位
        return out
    # auto (默认): 变体角度注入 — GEN_TEMP 固定, 候选可并行 (本实现先串行调, 并行需 ThreadPool,
    # 但 llm() 内部 chat_target 已有 retries+timeout; N=3 串行 ~3 次 LLM 调用, 非 eval 主瓶颈).
    set_temp(gen_temp)
    for i in range(n):
        ang = _FACET_EDIT_ANGLES[i % len(_FACET_EDIT_ANGLES)]
        diag_i = (f"{ang}\n\n{diag}") if ang else diag
        ep = prompt_tmpl.format(facet=ax, slot_body=leaf.slots.get(ax, ""), diagnostics=diag_i)
        raw = llm(system, ep, max_tokens=2048, stage=f"evolver_facet_p{i}")
        cs = clean_skill(raw, leaf.skill)
        cslots, cok = facet_edit_guard(leaf.slots, ax, cs)
        if cok and cslots is not None:
            out.append((cslots, i % len(_FACET_EDIT_ANGLES)))
    return out


def _run_strengthen(args, t, counts, reps, best_node, freeze, title,
                    s0_cache, probe_items, base_pass, meta_mask, archive,
                    best_val_delta, best_val_ci, best_skill, expand_event,
                    op_counts=None, pool_gates=None):
    """算子H strengthen — 加固现有段 (加/精炼一条 bullet), 而非长新段. 供两处饱和入口复用:

      (1) find_uncovered_type_for_expand 返回 None: 全非-meta 类型已覆盖 (target is None);
      (2) type_best_delta 判该类型饱和: 该类型已有采纳过的专段 且 本次新变体 delta < 旧段 delta
          (用户指定修复: "已有专段且新变体 delta < 旧段 delta 才判饱和, 否则 expand 抢先").

    classifier-independent 选段 (select_strengthen_target), 绕过分类器把 existence/uniqueness
    吞进 meta 的 bug, 让 LLM 看真实最难失败自识别第 2 双向轴. archive 原地 append (mutable list).
    入参 expand_event: 已由调用方设置的诊断 (None = 全覆盖; "type-saturated" = 单类型饱和). 若
    strengthen 产出候选 (无论采纳与否), expand_event 被 strengthen 事件覆盖; 若 sec_key 为 None
    (无可加固段) 则原样返回调用方的 expand_event (保既有诊断).
    返回 (did_strengthen, expand_event, best_val_delta, best_val_ci, best_skill, best_node)."""
    all_failures = [f for reps_list in reps.values() for f in reps_list]
    did_strengthen = False
    if not (_STRENGTHEN_ON and len(all_failures) > 0 and getattr(args, "strengthen", False)):
        return did_strengthen, expand_event, best_val_delta, best_val_ci, best_skill, best_node
    sec_key, sec_failures = select_strengthen_target(
        counts, reps, best_node.slots, freeze)
    if sec_key is None or not sec_failures:
        return did_strengthen, expand_event, best_val_delta, best_val_ci, best_skill, best_node
    if op_counts is not None:
        op_counts["strengthen"]["attempted"] += 1  # strengthen: 有可加固段 → 将调用 LLM (F08/Q6)
    sec_diag = make_diagnostics(sec_failures)
    slot_body = best_node.slots.get(sec_key, "")
    try:
        str_raw = llm(STRENGTHEN_SYSTEM, SECTION_STRENGTHEN_PROMPT.format(
            facet=sec_key, slot_body=slot_body, diagnostics=sec_diag),
            max_tokens=2048, stage="evolver_strengthen")
    except Exception as e:  # noqa: BLE001
        expand_event = {"admitted": False, "reason": "strengthen-llm-error",
                        "section": sec_key, "err": repr(e)}
        print(f"  ⚠️ strengthen iter{t} 失败: {e!r}", flush=True)
        return did_strengthen, expand_event, best_val_delta, best_val_ci, best_skill, best_node
    str_clean = clean_skill(str_raw, "")
    st_slots, st_ok = facet_edit_guard(best_node.slots, sec_key, str_clean)
    if not st_ok:
        expand_event = {"admitted": False, "reason": "strengthen-guard-reject",
                        "section": sec_key}
        print(f"  ✋ strengthen iter{t}: guard rejected (~{sec_key})", flush=True)
        return did_strengthen, expand_event, best_val_delta, best_val_ci, best_skill, best_node
    st_skill = facet_compose(title, st_slots)
    if op_counts is not None:
        op_counts["strengthen"]["evaluated"] += 1  # strengthen: 进 eval (guard 通过)
    sp, sd, _, sitem = eval_on(
        probe_items, st_skill, s0_cache,
        stage="evolver_strengthen_val", k=args.eval_k,
        eval_deadline_s=EVAL_DEADLINE_S)
    sl = lcb_score_from_deltas(sitem, s0_cache, args.eval_k)[0] if args.select_lcb else bootstrap_ci_lower(sitem)
    st_admitted = (sp >= base_pass - 0.06) and (sl > 0)
    if st_admitted:
        st_node = _ArchiveNode(st_skill, st_slots,
                               f"strengthen:{sec_key}",
                               parent=best_node.facet_label,
                               val_delta=sd, val_ci=sl)
        if args.pool_filter:
            _adm, _reason = admission_filter(
                st_skill, best_node.skill, sd, sitem, probe_items, eps=args.pool_eps)
            if pool_gates is not None:
                _tally_pool_filter(_reason, _adm, pool_gates)  # strengthen: pool_filter 闸门计数 (F08)
            if not _adm:
                print(f"  🔧 strengthen iter{t}: POOL-FILTER 拒 {_reason} → 不入 archive, 不触发", flush=True)
                return did_strengthen, expand_event, best_val_delta, best_val_ci, best_skill, best_node
        archive.append(st_node)
        if op_counts is not None:
            op_counts["strengthen"]["admitted"] += 1  # strengthen: LOST+ci>0 (+可选 pool_filter) 准入入 archive
        did_strengthen = True
        expand_event = {"section_name": sec_key,
                        "delta": round(sd, 4), "ci": round(sl, 4),
                        "admitted": True, "reason": "strengthen",
                        "became_best": False,
                        "n_sections_after": len(st_slots)}
        print(f"  🔧 strengthen iter{t}: ~{sec_key} delta={sd:+.3f} "
              f"CI={sl:+.3f} pass={sp:.3f} admitted=Y", flush=True)
        if sd > best_val_delta and sl > 0:
            best_val_delta, best_val_ci, best_skill = sd, sl, st_skill
            best_node = st_node
            expand_event["became_best"] = True
            print(f"  ✅ STRENGTHEN-BEST iter{t}: ~{sec_key} "
                  f"delta={sd:+.3f} CI={sl:+.3f}", flush=True)
            if op_counts is not None:
                op_counts["strengthen"]["promoted"] += 1  # strengthen: TREE-BEST 夺魁
        else:
            # break-③ 救援 (镜像 expand): 整体未超 best 看 per-subset
            _promote, _sig = subset_promote_decision(
                sitem, best_val_delta, meta_mask, enabled=args.subset_admit)
            _eps_floor = 0.05
            if _promote and sd >= best_val_delta - _eps_floor:
                best_val_delta, best_val_ci, best_skill = sd, sl, st_skill
                best_node = st_node
                expand_event["became_best"] = True
                expand_event["subset_promote"] = {
                    "not_meta_ci": round(_sig["not_meta"]["ci"], 4),
                    "not_meta_n": _sig["not_meta"]["n"]}
                print(f"  ✅ STRENGTHEN-SUBSET-BEST iter{t}: ~{sec_key} "
                      f"delta={sd:+.3f} (not-meta ci={_sig['not_meta']['ci']:+.3f})",
                      flush=True)
                if op_counts is not None:
                    op_counts["strengthen"]["promoted"] += 1  # strengthen: SUBSET-BEST 救援晋升
            elif _promote:
                expand_event["subset_gated"] = {
                    "not_meta_ci": round(_sig["not_meta"]["ci"], 4),
                    "not_meta_n": _sig["not_meta"]["n"]}
                print(f"  🔶 STRENGTHEN-SUBSET-GATE iter{t}: ~{sec_key} "
                      f"delta={sd:+.3f} < best{best_val_delta:+.3f}-floor{_eps_floor} "
                      f"(not-meta ci={_sig['not_meta']['ci']:+.3f}); archive-only",
                      flush=True)
    else:
        _sig = subset_signal(sitem, meta_mask)
        expand_event = {"section_name": sec_key,
                        "delta": round(sd, 4), "ci": round(sl, 4),
                        "admitted": False, "reason": "strengthen-lost-or-ci<=0",
                        "not_meta_ci": round(_sig["not_meta"]["ci"], 4),
                        "not_meta_n": _sig["not_meta"]["n"],
                        "meta_ci": round(_sig["meta"]["ci"], 4)}
    return did_strengthen, expand_event, best_val_delta, best_val_ci, best_skill, best_node


def _run_facet_tree(args, init_skill, s0_cache, probe_items, diag_items, base_pass, base_delta,
                    base_ci, out_dir, log, init_best_delta=None, init_best_ci=None):
    """Facet-Tree GA 树遍历 (greenfield, gated by --facet_tree).

    init_best_delta/init_best_ci: --resume 时 prior 末轮 best 的 val_delta/ci (传入避免 best 状态
    被 base_delta 覆盖降级 — 见 docs/phase2_issues.md 2026-09-23). fresh run 传 None → 默认用
    base_delta/base_ci (= 旧行为, 零回归). base_* 始终保持"原始 baseline"语义, 供 LOST-guard 用.

    真正的进化算法 (修 bug#1 随机游走 / bug#3 盲改):
      - 算子A 初始化: 4 分支 = 4 非冻结 facet 轴, 各吃到 bootstrap 切分的不相交 diag 子集 (DIVE 异构环境)
      - 算子B 选择+接受: 每分支 child 经 eval 后 → archive 追加保留 (DarwinX keep-losers) +
        活跃叶精英晋级 (child.ci>0 AND child.delta>parent.val_delta 才替换父叶, 否则父叶留)
      - 算子C 交叉: gen≥2 末, binary-tournament 从 archive 选 2 父本 → recombine_facets → eval → 准入
      - 算子D 多样性: facet_edit_guard 物理防塌缩 (已建) + keep-losers archive (跨代素材池)
    Stage 0: 关涌现 (2 代 < K=3)、4 冷启动分支、Final Answer 冻结。fitness = 纯 EM pass_rate (同既有)。
    复用 eval_on/bootstrap_ci_lower/collect_failures 不变; 新增 facet 分解/交叉/edit 护栏。
    返回 (best_skill, best_val_delta, best_val_ci)。
    """
    axes = [a.strip() for a in args.facet_axes.split(",") if a.strip()]
    freeze = args.freeze_facet.strip()
    # 冷启动分支 = 非冻结 facet 轴 (Stage 0 固定 4 轴, failure_modes 空不参与)
    branch_axes = [a for a in axes if a != freeze]
    title, root_slots = facet_decompose(init_skill)
    n_branches = len(branch_axes)

    # 算子A: DIVE bootstrap — diag_items 切成 n_branches 不相交子集, 每分支诊断自己的子集.
    # bootstrap 关 (消融 E 组) → 各分支共享全 diag_items (退回 gonogo 原简化).
    if getattr(args, "bootstrap", True) and n_branches > 1 and diag_items:
        diag_subsets = partition_diag(diag_items, n_branches, seed=args.probe_seed)
    else:
        diag_subsets = [list(diag_items)] * n_branches
    diag_sizes = [len(s) for s in diag_subsets]
    print(f"🌱 facet-tree init: branches={n_branches} | bootstrap={'ON' if getattr(args,'bootstrap',True) else 'OFF'} "
          f"| diag subsets={diag_sizes} | crossover={'ON' if getattr(args,'crossover',True) else 'OFF'} "
          f"| freeze={freeze}", flush=True)

    # 根节点 + 每轴一个叶节点(初始 = 根 slots, label 标定其变异轴). 叶节点初始全 = S₀ (结构特化靠
    # facet_edit_guard 在变异时约束, 非靠胖初始化 — N=4 硬约束下丰富性来自异构环境+结构特化+归档累积).
    leaves = []  # active_leaves: 每分支当前活跃叶 (top-1 per branch)
    # best 起点: resume 时 = prior 末轮 best delta/ci (传入), fresh run = base_delta/base_ci.
    # base_* 仍保留原始 baseline 语义 (下方 LOST-guard 用), 不被 resume best 污染.
    _best0_delta = base_delta if init_best_delta is None else init_best_delta
    _best0_ci = base_ci if init_best_ci is None else init_best_ci
    for i, ax in enumerate(branch_axes):
        nd = _ArchiveNode(init_skill, dict(root_slots), ax, parent="root",
                          val_delta=_best0_delta, val_ci=_best0_ci)
        leaves.append(nd)
    # archive: 全部产出叶 (含输家, DarwinX keep-losers). 初始含 4 个根叶作交叉素材池起点.
    archive = list(leaves)
    # per-branch diag 缓存: skill 未变则复用上轮 failures (修 bug#3, 复用 D1 cache_failures 模式).
    last_diag_hash = {ax: None for ax in branch_axes}
    last_failures = {ax: (None, 0) for ax in branch_axes}  # {ax: (failures_list, n_fail)}

    best_skill = init_skill
    best_val_delta = _best0_delta
    best_val_ci = _best0_ci
    # best_node 追踪: 诊断需 best 的 facet_label (本轴) 才能正确排除本轴算 non_native_s0_ratio.
    # 仅占位 (首个根叶); TREE-BEST/XOVER-BEST/OVERLAY-BEST 命中处同步更新. 纯追踪, 不改 best 选取逻辑.
    # ⚠️ 若 gen1 4 单轴 child 全未过 ci>0 → best_skill 仍 = init_skill, best_node 仍占位 → 诊断走 corner
    # (md5(best_skill)==md5(init_skill) → ratio=1.0). 见 _facet_non_native_s0_ratio 注释.
    best_node = leaves[0]
    no_improve = 0
    crossover_fired = 0
    # 算子H 饱和判定追踪 (修 strengthen run 暴露的设计缺口):
    #   expand 对"已覆盖但仍有失败"的类型持续生成更强变体 (Implication 段 iter4 +0.127 → iter6 +0.258),
    #   抢先于 strengthen → strengthen 永不触发. 用户指定修复: "已有专段且新变体 delta < 旧段 delta 才判
    #   饱和, 否则 expand 抢先". 即仅当某类型已有专属段 (type_best_delta 有记录) 且本次新变体的 val delta
    #   回落 (ed < 旧段最佳 delta) 时才判该类型饱和 → 交给 strengthen 加固既有段; 新变体 delta >= 旧段
    #   (更强或持平) 时 expand 正常采纳并刷新该类型最佳 delta (坚持探索不半途而废).
    # key = failure type label (FAILURE_FOCI[*]), val = 该类型历次 expand 采纳段的最佳 val_delta.
    type_best_delta = {}
    # GLM first-order 信号: {t: {ax: {failure_type: count}}} — 每分支每代 diag 子集的失败类型分布.
    # 用于验证 DIVE bootstrap 异构环境是否真发生 (弱轴是否吃到轴专有失败). 落盘到 evolver_log.
    branch_failure_types = {}
    # break-③: probe_items 的 has_meta 预标签 (确定性, 零 API). 供 per-subset 准入判据用.
    # 记忆: expand-v2 整体 −0.021 不显著, 但 not-meta 17 题 +0.157 p=0.0235 显著 — 整体单目标
    # 把 not-meta 增益淹没. per-subset 准入让"某子集增益>0 且无子集退步超 ε"的候选被接受.
    meta_mask = {str(it["id"]): classify_item_meta(it) for it in probe_items}

    # ── 算子触发计数 (F08/Q6/RF §6 Step 3, 纯插桩零回归): run_total 跨轮汇总 → 终局 0-promoted 告警 ──
    run_total = _new_operator_counts()

    for t in range(1, args.iters + 1):
        t0 = time.time()
        # 每轮每算子计数器 (attempted/guard_rejected/evaluated/admitted/promoted) + pool_filter 三关计数.
        iter_counts = _new_operator_counts()
        pool_gates = _new_pool_filter_gates()
        print(f"\n{'='*50}\nFacet-Tree Iter {t}/{args.iters} (facet axes={branch_axes}, "
              f"freeze={freeze}, emergence={'OFF(disabled)' if args.no_emergence else 'OFF(stage0)'})"
              f"\n{'='*50}")
        # 每代遍历活跃叶节点: 各按自己 facet 轴单 slot 变异 + 接受准则
        leaves_by_ax = {l.facet_label: l for l in leaves}
        new_active = []
        for bi, leaf in enumerate(leaves):
            ax = leaf.facet_label
            if ax == freeze:  # 冻结轴不变异
                new_active.append(leaf)
                continue
            # 算子A: 每分支诊断自己的 diag 子集; skill 未变则复用缓存 (修 bug#3 盲改)
            sh = md5(leaf.skill)
            if last_diag_hash[ax] != sh:
                ds = diag_subsets[bi]
                failures, n_fail, wins = collect_failures(ds, leaf.skill, s0_cache,
                                                          n_diag=args.n_diag, k=args.eval_k)
                last_diag_hash[ax] = sh
                last_failures[ax] = (failures, n_fail)
            else:
                failures, n_fail = last_failures[ax]
            # GLM first-order 信号: per-branch diag 子集的失败类型分布.
            # 验证 DIVE bootstrap 是否真产生异构环境 — 若弱轴 (Theorem/Hypotheses) 喂进
            # 的 12 题诊断出来 8/8 仍是 meta-option, 则 bootstrap 对该轴是"无效环境"
            # (信号被 meta-option 主导占满, 轴专有失败 ≈ 0), bootstrap 帮/害的前置不成立.
            try:
                fcounts, _ = compute_failure_distribution(failures)
                dist_str = ", ".join(f"{lbl}:{c}" for lbl, c in
                                     sorted(fcounts.items(), key=lambda kv: -kv[1]))
                diag_size = len(diag_subsets[bi]) if getattr(args, "bootstrap", True) else len(diag_items)
                print(f"  🔎 facet[{ax}] diag-subset n={diag_size} fail={n_fail} "
                      f"types={{{dist_str}}}", flush=True)
                branch_failure_types.setdefault(t, {})[ax] = fcounts
            except Exception:  # noqa: BLE001
                pass
            # 单 slot patcher: 只让 LLM 编辑 target slot, 非目标 md5 护栏校验 (算子D 护栏1)
            # Step1 N-parallel (n_parallel>1): 该分支并发生 N 个候选 (变体角度注入), 各过 guard+eval,
            # 取 delta 最优晋级, 其余入 archive (keep-losers). n_parallel=1 → 字节复现既有单候选路径.
            n_par = getattr(args, "n_parallel", 1) or 1
            div_mode = getattr(args, "parallel_diversity", "auto") or "auto"
            # 方向E 落点②: 突变起点 = 同 facet archive 精英种群加权采样 (打破单 lineage).
            # --pop_mutation off → sampled_parent=leaf, 字节复现既有单 lineage 路径 (零回归).
            # on → gen1 退回 leaf (同 facet archive 仅根叶); gen≥2 种群采样 (含 leaf + 同轴历史 elite).
            # 纪律不变: 去重/promotion 仍以 incumbent leaf 为基准 (只升不降), 采样仅扩搜索起点.
            sampled_parent = leaf
            if getattr(args, "pop_mutation", False):
                sampled_parent = sample_mutation_parent(archive, ax, leaf,
                                                        seed=args.probe_seed, t=t,
                                                        eps=getattr(args, "pop_eps", 0.3))
                if sampled_parent is not leaf:
                    print(f"  🧬 facet[{ax}] pop-sample: parent δ={sampled_parent.val_delta:+.3f} "
                          f"ci={sampled_parent.val_ci:+.3f} (incumbent δ={leaf.val_delta:+.3f})", flush=True)
            cand_list = []  # list[(cand_slots, angle_idx)]
            iter_counts["facet_edit"]["attempted"] += 1  # 进入算子主体 (LLM/edit 调用前)
            try:
                if n_par > 1:
                    cand_list = gen_facet_candidates(sampled_parent, ax, failures, n_par,
                                                     FACET_EDIT_SYSTEM, FACET_EDIT_PROMPT,
                                                     GEN_TEMP, diversity=div_mode)
                else:
                    set_temp(GEN_TEMP)
                    diag = make_diagnostics(failures) if failures else "(no diagnostic failures)"
                    edit_prompt = FACET_EDIT_PROMPT.format(
                        facet=ax, slot_body=sampled_parent.slots.get(ax, ""), diagnostics=diag)
                    raw = llm(FACET_EDIT_SYSTEM, edit_prompt, max_tokens=2048, stage="evolver_facet")
                    child_text = clean_skill(raw, sampled_parent.skill)
                    cslots, cok = facet_edit_guard(sampled_parent.slots, ax, child_text)
                    if cok and cslots is not None:
                        cand_list = [(cslots, 0)]
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ facet[{ax}] 编辑失败: {e!r}", flush=True)
            if not cand_list:
                iter_counts["facet_edit"]["guard_rejected"] += 1  # 结构/温度 guard 拒掉 (未进 eval)
                new_active.append(leaf)  # 护栏拒绝 → 父叶留作活跃叶
                continue
            # 过 md5 无改动 (与父叶相同 → 丢弃, 父叶留)
            evald = []  # list[(node, pass, item)] 通过 guard 且 != 父叶 的已评候选 (带 pass/citem 供 LOST & pool_filter 判)
            for cslots, ang_idx in cand_list:
                child_skill = facet_compose(title, cslots)
                if md5(child_skill) == md5(leaf.skill):
                    continue
                iter_counts["facet_edit"]["evaluated"] += 1  # 进 eval_on 算出 pass/delta/ci
                cp, cd, _, citem = eval_on(probe_items, child_skill, s0_cache,
                                          stage="evolver_facet_val", k=args.eval_k,
                                          eval_deadline_s=EVAL_DEADLINE_S)
                if args.select_lcb:
                    cl = lcb_score_from_deltas(citem, s0_cache, args.eval_k)[0]
                else:
                    cl = bootstrap_ci_lower(citem)
                child = _ArchiveNode(child_skill, cslots, ax, parent=leaf.facet_label,
                                     val_delta=cd, val_ci=cl)
                evald.append((child, cp, citem))
            if not evald:
                new_active.append(leaf)  # 全候选 == 父叶 (无改动) → 父叶留
                continue
            # LOST monitor + keep-losers: 非 LOST 候选全部入 archive (作交叉素材), LOST 拒.
            survivors = []  # list[(node, pass, item)]
            for child, cp, citem in evald:
                if cp < base_pass - 0.06:
                    print(f"  facet[{ax}] LOST: pass={cp:.3f} delta={child.val_delta:+.3f} → 拒 (不入 archive)",
                          flush=True)
                    continue
                survivors.append((child, cp, citem))
            if not survivors:
                new_active.append(leaf)  # 全候选 LOST → 父叶留
                continue
            # 全部 survivor 入 archive (keep-losers: 含未晋级者作跨代交叉素材)
            for child, _cp, citem in survivors:
                if args.pool_filter:
                    _adm, _reason = admission_filter(
                        child.skill, leaf.skill, child.val_delta, citem, probe_items, eps=args.pool_eps)
                    _tally_pool_filter(_reason, _adm, pool_gates)  # 纯读 reason, 零回归
                    if not _adm:
                        print(f"  facet[{ax}] POOL-FILTER 拒: {_reason} → 不入 archive", flush=True)
                        continue
                archive.append(child)
                iter_counts["facet_edit"]["admitted"] += 1  # 过准入门进 archive (含 keep-losers)
            # Step1 选择: 取 delta 最优且 ci>0 者晋级; 无 ci>0 者取 delta 最优但父叶留作活跃叶
            sv_nodes = [c for c, _, _ in survivors]
            promoted_node = None
            ci_pos = [c for c in sv_nodes if c.val_ci > 0]
            if ci_pos:
                promoted_node = max(ci_pos, key=lambda c: c.val_delta)
                if promoted_node.val_delta > leaf.val_delta:
                    new_active.append(promoted_node)
                    promoted = "PROMOTE"
                else:
                    new_active.append(leaf)  # 未超父叶 → 父叶留, promoted 仅入 archive
                    promoted = "KEEP-PARENT"
            else:
                promoted_node = max(sv_nodes, key=lambda c: c.val_delta)
                new_active.append(leaf)  # 全 ci<=0 → 父叶留, 候选仅入 archive
                promoted = "KEEP-PARENT(NO-CI)"
            log_line = (f"  facet[{ax}] N={len(evald)} cand: best_delta={promoted_node.val_delta:+.3f} "
                        f"CI={promoted_node.val_ci:+.3f} | {promoted} (parent={leaf.val_delta:+.3f})")
            if n_par > 1:
                dvals = ", ".join(f"{c.val_delta:+.3f}" for c in sorted(sv_nodes, key=lambda x: -x.val_delta))
                log_line += f" | survivors_delta=[{dvals}]"
            print(log_line, flush=True)
            # 全树 best 追踪 (probe 上 delta 最高且 CI>0 者) — 跨所有 survivor 取最优
            for child in sv_nodes:
                if child.val_delta > best_val_delta and child.val_ci > 0:
                    best_val_delta, best_val_ci, best_skill = child.val_delta, child.val_ci, child.skill
                    best_node = child
                    no_improve = 0
                    iter_counts["facet_edit"]["promoted"] += 1  # 成为本轮/历史全局 best
                    print(f"  ✅ TREE-BEST iter{t}: facet[{ax}] delta={child.val_delta:+.3f} CI={child.val_ci:+.3f}")
                else:
                    no_improve += 1
        leaves = new_active  # 活跃叶 = 每分支 top-1 (精英晋级或父叶保留)

        # 算子C: 交叉 (gen≥2, crossover 开, archive 有 ≥2 不同 facet 叶). Stage 0 最多触发 1 次.
        # break-① 修复: donor key 用"节点实际携带的 section 名"而非 facet_label. 旧逻辑 key=facet_label,
        # 对单轴叶 (facet_label==段名) 正常, 但对 expand 节点 (facet_label="expand:Quantifier...")
        # parent_b_slots 的 key 是裸段名 "Quantifier..." → `b_key in parent_b_slots` 恒 False →
        # 移植静默 no-op (旧"交叉成功"实则只 parent-A + coherence, 从未真嫁接 expand 段).
        xo_event = None
        # 方向 B: 多 parent crossover. xo_multiparent>0 时走多段嫁接路径 (与旧 2-parent 互斥).
        # 这一路径在 archive 有足够定向 donor 时执行; 否则回退到下方旧 2-parent 路径.
        _xo_mp_n = getattr(args, "xo_multiparent", 0) or 0
        if (getattr(args, "crossover", True) and t >= 2 and len(archive) >= 2
                and _xo_mp_n >= 2):
            # 当前 iter 的失败分布: 聚合所有分支的 per-branch fcounts (compute_failure_distribution 的
            # iter-stable 版本在 L2364 expand 块内, AFTER crossover — 此处不可用).
            # branch_failure_types[t][ax] = 每分支诊断子集的失败类型 counts (L2054 已填).
            # 聚合 = 把各分支 counts 相加 = 当前 iter 全诊断子集的失败分布 (定向 donor 的 counts 来源).
            _mp_counts = {}
            for _ax, _fc in branch_failure_types.get(t, {}).items():
                for _lbl, _c in _fc.items():
                    _mp_counts[_lbl] = _mp_counts.get(_lbl, 0) + _c
            # 若本 iter 无分支失败分布 (诊断异常), 退化为全 archive — 仍可定向但覆盖信号弱.
            if not _mp_counts and archive:
                _mp_counts = _mp_counts or {}  # 空 counts → select_donors_directed 全部 donor 覆盖=0 → 返空
            # 选 N 个定向 donor (每段最多 1, 各覆盖不同失败类型) — base = 当前 best_node.
            if getattr(args, "xo_donor_directed", False):
                _mp_donors = select_donors_directed(archive, root_slots, freeze, _mp_counts,
                                                    topk=_xo_mp_n)
            else:
                # 非定向: 用 select_section_donors 取每段 top1, 展平为 pool, 取前 N (val_ci 排序).
                _mp_sec = select_section_donors(archive, root_slots, freeze, topk=1)
                _mp_pool = [(k, nd) for k, nds in _mp_sec.items() for nd in nds]
                _mp_pool.sort(key=lambda pair: -pair[1].val_ci)
                _mp_donors = _mp_pool[:_xo_mp_n]
            _mp_allow_merge = getattr(args, "xo_allow_merge", False)
            _mp_cohere_retries = getattr(args, "xo_cohere_retries", 0) or 0
            if len(_mp_donors) >= 2:
                mp_base = best_node.slots if best_node is not None else root_slots
                iter_counts["crossover"]["attempted"] += 1  # 进 recombine (MP 路径)
                mp_slots, mp_ok = recombine_facets_multi(
                    mp_base, _mp_donors, freeze,
                    llm_stage="evolver_crossover_mp",
                    retries=_mp_cohere_retries, allow_merge=_mp_allow_merge)
                if mp_ok and md5(facet_compose(title, mp_slots)) != md5(best_skill):
                    mp_skill = facet_compose(title, mp_slots)
                    mp_parents = "+".join(k for k, _ in _mp_donors[:_xo_mp_n])
                    iter_counts["crossover"]["evaluated"] += 1  # 进 eval_on
                    mp_p, mp_d, _, mp_item = eval_on(probe_items, mp_skill, s0_cache,
                                                    stage="evolver_crossover_mp_val", k=args.eval_k,
                                                    eval_deadline_s=EVAL_DEADLINE_S)
                    mp_l = lcb_score_from_deltas(mp_item, s0_cache, args.eval_k)[0] if args.select_lcb else bootstrap_ci_lower(mp_item)
                    mp_node = _ArchiveNode(mp_skill, mp_slots, f"xomp:{mp_parents}",
                                            parent=mp_parents, val_delta=mp_d, val_ci=mp_l)
                    mp_admitted = (mp_l > 0)
                    if mp_admitted:
                        archive.append(mp_node)
                        iter_counts["crossover"]["admitted"] += 1  # 进 archive
                        crossover_fired += 1
                        if mp_d > best_val_delta and mp_l > 0:
                            best_val_delta, best_val_ci, best_skill = mp_d, mp_l, mp_skill
                            best_node = mp_node
                            iter_counts["crossover"]["promoted"] += 1  # XOVER-MP-BEST
                            print(f"  ✅ XOVER-MP-BEST iter{t}: xomp[{mp_parents}] delta={mp_d:+.3f} CI={mp_l:+.3f} "
                                  f"(n_donors={len(_mp_donors)})", flush=True)
                        else:
                            # break-③ per-subset 救援 (与 2-parent 同口径)
                            _promote, _sig = subset_promote_decision(mp_item, best_val_delta, meta_mask, enabled=args.subset_admit)
                            _eps_floor = 0.05
                            if _promote and mp_d >= best_val_delta - _eps_floor:
                                best_val_delta, best_val_ci, best_skill = mp_d, mp_l, mp_skill
                                best_node = mp_node
                                iter_counts["crossover"]["promoted"] += 1  # XOVER-MP-SUBSET-BEST (夺 best)
                                print(f"  ✅ XOVER-MP-SUBSET-BEST iter{t}: xomp[{mp_parents}] delta={mp_d:+.3f} "
                                      f"(not-meta ci={_sig['not_meta']['ci']:+.3f} n={_sig['not_meta']['n']})", flush=True)
                            elif _promote:
                                print(f"  🔶 XOVER-MP-SUBSET-GATE iter{t}: xomp[{mp_parents}] delta={mp_d:+.3f} "
                                      f"< best{best_val_delta:+.3f}-floor{_eps_floor} "
                                      f"(not-meta ci={_sig['not_meta']['ci']:+.3f}); archive-only", flush=True)
                        xo_event = {"parents": [k for k, _ in _mp_donors], "mode": "multiparent",
                                    "delta": round(mp_d, 4), "ci": round(mp_l, 4),
                                    "admitted": True, "n_donors": len(_mp_donors),
                                    "directed": getattr(args, "xo_donor_directed", False)}
                    else:
                        _sig = subset_signal(mp_item, meta_mask)
                        xo_event = {"parents": [k for k, _ in _mp_donors], "mode": "multiparent",
                                    "delta": round(mp_d, 4), "ci": round(mp_l, 4), "admitted": False,
                                    "not_meta_ci": round(_sig["not_meta"]["ci"], 4),
                                    "meta_ci": round(_sig["meta"]["ci"], 4),
                                    "n_donors": len(_mp_donors),
                                    "directed": getattr(args, "xo_donor_directed", False)}
                    print(f"  🔀 xover-MP iter{t}: [{mp_parents}] delta={mp_d:+.3f} CI={mp_l:+.3f} "
                          f"pass={mp_p:.3f} admitted={'Y' if mp_admitted else 'N(ci<=0)'} "
                          f"(n_donors={len(_mp_donors)}, directed={getattr(args,'xo_donor_directed',False)})",
                          flush=True)
                elif not mp_ok:
                    iter_counts["crossover"]["guard_rejected"] += 1  # MP guard/cohere-fail (未进 eval)
                    print(f"  🔀 xover-MP iter{t}: guard/cohere-fail "
                          f"(n_donors={len(_mp_donors)}, fallback no-cohere)",
                          flush=True)
                    xo_event = {"parents": [k for k, _ in _mp_donors], "mode": "multiparent",
                                "admitted": False, "reason": "guard/cohere-fail",
                                "n_donors": len(_mp_donors)}
            # 多 parent 路径设了 xo_event (无论成败) → 跳过下方旧 2-parent 路径;
            # 未设 (donor 不足 / counts 空) → 回退下方旧 2-parent (保既有行为).
        if (getattr(args, "crossover", True) and t >= 2 and len(archive) >= 2
                and xo_event is None):
            # section 级 donor 池: 节点携带的每个已变异 section 都暴露 (含 expand/xo/overlay 复合节点)
            donor_topk = getattr(args, "donor_topk", 1) or 1
            sec_donors_lists = select_section_donors(archive, root_slots, freeze, topk=donor_topk)
            # 展平成 [(sec_key, node)] 候选对池
            pool = []
            for sec_key, cands in sec_donors_lists.items():
                for nd in cands:
                    pool.append((sec_key, nd))
            if len(pool) >= 2:
                iter_counts["crossover"]["attempted"] += 1  # 2-parent: 有可行 pair 池 → 将调用 recombine (F08/Q6)
                import random as _xo_rng
                _xr = _xo_rng.Random(args.probe_seed + t)
                _xr.shuffle(pool)
                # 靶点1: cohere/guard 失败时换 parent 对重试 (旧版单对即弃 → 4/7 交叉死在 guard).
                # 每代最多尝试 (1 + xo_pair_retries) 对不同 parent; 任一对 recombine 成功即进 eval.
                _xo_retries = getattr(args, "xo_pair_retries", 0) or 0
                _allow_merge = getattr(args, "xo_allow_merge", False)
                _cohere_retries = getattr(args, "xo_cohere_retries", 0) or 0
                _xo_attempts = []   # 记每对的结局 (parents, outcome) 供日志诊断
                _eval_done = False
                for _pair_idx in range(min(len(pool), 2) + _xo_retries):
                    # 选两个不同的 section (避免自交) — A 贡献 a_key, B 移植 b_key
                    i_a = (2 * _pair_idx) % len(pool)
                    i_b = (2 * _pair_idx + 1) % len(pool)
                    a_key, pa = pool[i_a]
                    b_key, pb = pool[i_b]
                    # 尽量取不同父本 (若两 section 来自同节点, 找一个不同的 b)
                    for _ in range(3):
                        if pb is not pa and b_key != a_key:
                            break
                        if len(pool) > 2:
                            i_b = (i_b + 1) % len(pool)
                            b_key, pb = pool[i_b]
                        else:
                            break
                    # a_key 标 A 的强 facet (上下文), b_key = 从 B 移植的裸段名 (修复 key 不匹配)
                    # 靶点1: 传 cohere_retries (单次 LLM 失败重试) + allow_merge (放宽段数守卫)
                    x_slots, x_ok = recombine_facets(pa.slots, pb.slots, a_key, b_key, freeze,
                                                    llm_stage="evolver_crossover",
                                                    retries=_cohere_retries, allow_merge=_allow_merge)
                    if not x_ok:
                        _xo_attempts.append(([a_key, b_key], "guard/cohere-fail"))
                        if _pair_idx < min(len(pool), 2) + _xo_retries - 1:
                            continue  # 换下一对 parent 重试
                        # 重试耗尽 — 最后一对仍 guard-fail
                        xo_event = {"parents": [a_key, b_key], "admitted": False,
                                    "reason": "guard/cohere-fail",
                                    "attempts": _xo_attempts}
                        print(f"  🔀 crossover iter{t}: {a_key}×{b_key} guard/cohere-fail "
                              f"(attempts={len(_xo_attempts)}, all structural)", flush=True)
                        iter_counts["crossover"]["guard_rejected"] += 1  # 2-parent: 重试耗尽仍 guard-fail, 未进 eval
                        break
                    x_skill = facet_compose(title, x_slots)
                    if md5(x_skill) == md5(pa.skill):  # 无重组变化
                        _xo_attempts.append(([a_key, b_key], "no-change"))
                        if _pair_idx < min(len(pool), 2) + _xo_retries - 1:
                            continue
                        xo_event = {"parents": [a_key, b_key], "admitted": False, "reason": "no-change",
                                    "attempts": _xo_attempts}
                        print(f"  🔀 crossover iter{t}: {a_key}×{b_key} no-change "
                              f"(attempts={len(_xo_attempts)})", flush=True)
                        iter_counts["crossover"]["guard_rejected"] += 1  # 2-parent: 重组无变化, 未进 eval
                        break
                    iter_counts["crossover"]["evaluated"] += 1  # 2-parent: 进 eval (recombine 产出有效重组)
                    xp, xd, _, xitem = eval_on(probe_items, x_skill, s0_cache,
                                               stage="evolver_crossover_val", k=args.eval_k,
                                               eval_deadline_s=EVAL_DEADLINE_S)
                    xl = lcb_score_from_deltas(xitem, s0_cache, args.eval_k)[0] if args.select_lcb else bootstrap_ci_lower(xitem)
                    xo_node = _ArchiveNode(x_skill, x_slots, f"xo:{a_key}+{b_key}",
                                           parent=f"{a_key}+{b_key}",
                                           val_delta=xd, val_ci=xl)
                    # 准入: ci>0 且较 best 有提升才入 archive 竞争全局 best
                    admitted = (xl > 0)
                    if admitted:
                        archive.append(xo_node)
                        iter_counts["crossover"]["admitted"] += 1  # 2-parent: ci>0 入 archive
                        crossover_fired += 1
                        if xd > best_val_delta and xl > 0:
                            best_val_delta, best_val_ci, best_skill = xd, xl, x_skill
                            best_node = xo_node  # 同步追踪 (诊断需 best_node.facet_label)
                            print(f"  ✅ XOVER-BEST iter{t}: {a_key}+{b_key} delta={xd:+.3f} CI={xl:+.3f}",
                                  flush=True)
                            iter_counts["crossover"]["promoted"] += 1  # 2-parent: TREE-BEST 夺魁
                        else:
                            # break-③: 整体未超 best, 看 per-subset 信号 (not-meta 增益被整体淹没时救援)
                            _promote, _sig = subset_promote_decision(xitem, best_val_delta, meta_mask, enabled=args.subset_admit)
                            # Bug C floor guard: 子集救援晋升 best 必须有底线 — 候选 delta 不得显著低于
                            # 当前 best (eps_floor 容忍噪声). 旧版无此守卫 → 低 delta 候选 (如 +0.268)
                            # 替换高 delta best (如 +0.380) → best_node 污染 → overlay carry-forward 级联膨胀.
                            _eps_floor = 0.05
                            if _promote and xd >= best_val_delta - _eps_floor:
                                best_val_delta, best_val_ci, best_skill = xd, xl, x_skill
                                best_node = xo_node
                                print(f"  ✅ XOVER-SUBSET-BEST iter{t}: {a_key}+{b_key} delta={xd:+.3f} "
                                      f"(not-meta ci={_sig['not_meta']['ci']:+.3f} n={_sig['not_meta']['n']})",
                                      flush=True)
                                iter_counts["crossover"]["promoted"] += 1  # 2-parent: SUBSET-BEST 救援晋升
                            elif _promote:
                                # 子集信号正向但 delta 低于底线 — 入 archive 竞争, 但不夺 best (语义分离)
                                print(f"  🔶 XOVER-SUBSET-GATE iter{t}: {a_key}+{b_key} delta={xd:+.3f} "
                                      f"< best{best_val_delta:+.3f}-floor{_eps_floor} "
                                      f"(not-meta ci={_sig['not_meta']['ci']:+.3f}); archive-only",
                                      flush=True)
                        xo_event = {"parents": [a_key, b_key], "delta": round(xd, 4),
                                    "ci": round(xl, 4), "admitted": True,
                                    "attempts": _xo_attempts}
                    else:
                        # break-③ 诊断回路: ci<=0 拒绝时, 仍记录 per-subset 信号 (不静默 skip)
                        _sig = subset_signal(xitem, meta_mask)
                        xo_event = {"parents": [a_key, b_key], "delta": round(xd, 4),
                                    "ci": round(xl, 4), "admitted": False,
                                    "not_meta_ci": round(_sig["not_meta"]["ci"], 4),
                                    "meta_ci": round(_sig["meta"]["ci"], 4),
                                    "attempts": _xo_attempts}
                    print(f"  🔀 crossover iter{t}: {a_key}×{b_key} delta={xd:+.3f} CI={xl:+.3f} "
                          f"pass={xp:.3f} admitted={'Y' if admitted else 'N(ci<=0)'} "
                          f"(pair_attempts={len(_xo_attempts)})", flush=True)
                    _eval_done = True
                    break  # 成功进 eval, 不再重试 parent 对
                if not _eval_done and xo_event is None:
                    # pool 不足或循环正常结束未设 event (防御性)
                    xo_event = {"parents": [], "admitted": False, "reason": "no-pool-pair"}

        # 算子E: 跨轴叠加 (修法1, 每代末). 把每轴历史最佳段 (ci>0 且本轴段确经变异) 叠成 overlay,
        # 直接打通"单轴改进永不叠加"的结构性失效 (见 memory/facet-tree-no-cumulative-20260825).
        # overlay 只组装叠加 skill + 竞争 best, 不动 leaves/单轴变异逻辑, 不动 facet_edit_guard.
        overlay_event = None
        if getattr(args, "overlay", True) and len(archive) > 0:
            iter_counts["overlay"]["attempted"] += 1  # overlay: 算子触发 (F08/Q6)
            # break-①: donor 选取改用 select_section_donors (放宽单轴叶限制, 复合节点的新段进遗传池).
            # 旧 select_axis_donors 排除了 expand/xo/overlay 节点 → 新 section 永不进 donor 池.
            # donor_topk>1 允许历史回退 (用户: "历史中的组合可能会更好").
            donor_topk = getattr(args, "donor_topk", 1) or 1
            sec_donors_lists = select_section_donors(archive, root_slots, freeze, topk=donor_topk)
            # 从每段 top-K 里取一个 (top-1 或确定性采样 — 偶数代采次优以探索历史路径)
            donors = {}
            for sec_key, cands in sec_donors_lists.items():
                if not cands:
                    continue
                pick = cands[0] if (t % 2 == 1 or len(cands) == 1) else cands[min(1, len(cands) - 1)]
                donors[sec_key] = pick
            # v2: 组装走 _overlay_assemble (carry-forward best_node 的已展开 section).
            # 旧: dict(root_slots) 重建 → 丢 expand 长出的第5段 (v1 根因2). 新: 携带 + key-set guard.
            overlay_slots, overlay_skill, ov_slots_chk, ov_ok = _overlay_assemble(
                title, root_slots, best_node.slots, donors, freeze)
            if ov_ok:  # root 段全存活 且 compose 未丢段 (key-set-only, 无 md5 检查)
                iter_counts["overlay"]["evaluated"] += 1  # overlay: 进 eval (compose 成功)
                op, od, _, oitem = eval_on(probe_items, overlay_skill, s0_cache,
                                           stage="evolver_overlay_val", k=args.eval_k,
                                           eval_deadline_s=EVAL_DEADLINE_S)
                ol = lcb_score_from_deltas(oitem, s0_cache, args.eval_k)[0] if args.select_lcb else bootstrap_ci_lower(oitem)
                admitted = (op >= base_pass - 0.06) and (ol > 0)  # LOST guard + ci>0 准入
                became_best = False
                if admitted:
                    ov_node = _ArchiveNode(overlay_skill, overlay_slots, "overlay",
                                           parent="overlay", val_delta=od, val_ci=ol)
                    archive.append(ov_node)
                    iter_counts["overlay"]["admitted"] += 1  # overlay: LOST+ci>0 准入入 archive
                    if od > best_val_delta and ol > 0:
                        best_val_delta, best_val_ci, best_skill = od, ol, overlay_skill
                        best_node = ov_node  # 诊断需 best_node.facet_label="overlay" (无本轴豁免)
                        became_best = True
                        print(f"  ✅ OVERLAY-BEST iter{t}: n_axes={len(donors)} delta={od:+.3f} CI={ol:+.3f}",
                              flush=True)
                        iter_counts["overlay"]["promoted"] += 1  # overlay: TREE-BEST 夺魁
                    else:
                        # break-③: 整体未超 best, 看 per-subset 信号救援
                        _promote, _sig = subset_promote_decision(oitem, best_val_delta, meta_mask, enabled=args.subset_admit)
                        # Bug C floor guard: 见 crossover 同名守卫注释. 救援晋升 best 不得显著拉低 delta.
                        _eps_floor = 0.05
                        if _promote and od >= best_val_delta - _eps_floor:
                            best_val_delta, best_val_ci, best_skill = od, ol, overlay_skill
                            best_node = ov_node
                            became_best = True
                            print(f"  ✅ OVERLAY-SUBSET-BEST iter{t}: n_axes={len(donors)} delta={od:+.3f} "
                                  f"(not-meta ci={_sig['not_meta']['ci']:+.3f} n={_sig['not_meta']['n']})",
                                  flush=True)
                            iter_counts["overlay"]["promoted"] += 1  # overlay: SUBSET-BEST 救援晋升
                        elif _promote:
                            print(f"  🔶 OVERLAY-SUBSET-GATE iter{t}: n_axes={len(donors)} delta={od:+.3f} "
                                  f"< best{best_val_delta:+.3f}-floor{_eps_floor} "
                                  f"(not-meta ci={_sig['not_meta']['ci']:+.3f}); archive-only",
                                  flush=True)
                    overlay_event = {"delta": round(od, 4), "ci": round(ol, 4),
                                     "admitted": True, "became_best": became_best,
                                     "n_axes_combined": len(donors),
                                     "n_sections": len(overlay_slots),
                                     "carried_forward": len(overlay_slots) - len(root_slots)}
                    print(f"  🔗 overlay iter{t}: n_axes={len(donors)} delta={od:+.3f} CI={ol:+.3f} "
                          f"pass={op:.3f} n_sec={len(overlay_slots)} admitted={'Y' if admitted else 'N'} became_best={'Y' if became_best else 'N'}",
                          flush=True)
                else:
                    # break-③ 诊断回路: 拒绝时记录 per-subset 信号 (不静默 skip)
                    _sig = subset_signal(oitem, meta_mask)
                    overlay_event = {"delta": round(od, 4), "ci": round(ol, 4),
                                     "admitted": False, "reason": "lost-or-ci<=0",
                                     "n_axes_combined": len(donors),
                                     "n_sections": len(overlay_slots),
                                     "not_meta_ci": round(_sig["not_meta"]["ci"], 4),
                                     "meta_ci": round(_sig["meta"]["ci"], 4)}
                    print(f"  🔗 overlay iter{t}: n_axes={len(donors)} delta={od:+.3f} CI={ol:+.3f} "
                          f"admitted=N (lost={op < base_pass - 0.06}) not-meta_ci={_sig['not_meta']['ci']:+.3f}", flush=True)
            else:
                overlay_event = {"admitted": False, "reason": "compose-section-loss",
                                 "n_axes_combined": len(donors),
                                 "n_sections": len(overlay_slots)}
                print(f"  🔗 overlay iter{t}: skipped (compose 丢段; n_sec={len(overlay_slots)})", flush=True)
                iter_counts["overlay"]["guard_rejected"] += 1  # overlay: compose 丢段, 未进 eval

        # 算子G: section-expansion (展开型算子). 每 expand_every 代末, 给 best_node 追加一个全新 section,
        # 目标对准当前累积诊断的失败, 突破 facet_edit「单slot单bullet」能力边界 (见算子G注释块).
        # 从 best_node.slots (已含本代进化/叠加段) 起步, 只 add 不改 (section_expansion_guard 构造性保证).
        # gated by --expand (默认 off, 向后兼容既有 run 零影响). 血统干净: 新 section 仅由失败诊断驱动.
        expand_event = None
        if getattr(args, "expand", False) and len(archive) > 0 and t % args.expand_every == 0:
            # sections cap: 已达上限则跳过 (防膨胀) — 上限 = section_expansion_guard.max_sections
            if len(best_node.slots) < 10:
                # v2 覆盖驱动 (修 v1 根因1 + 落实 §六方向A): 用全失败分布选「无专属段 且 非 meta」的
                # 最高频类型, 而非 v1 取最难 8 题(~100% meta → 内容重叠退步). meta 交由 facet_edit 轴
                # (0.49→0.53 路径); expand 攻 not-meta 未覆盖类型 = §六正向信号所在子集.
                # k=args.eval_k (非 v1 k=1) 保证失败类型分类非硬币翻转 (Plan-agent RISK A).
                _, _, _, all_failures = collect_failures(
                    diag_items, best_skill, s0_cache,
                    n_diag=8, k=args.eval_k, n_win=4, return_all_failures=True)
                counts, reps = compute_failure_distribution(all_failures)
                target = find_uncovered_type_for_expand(counts, best_node.slots.keys())
                if target is None:
                    # 全非-meta 类型已有专属段 (或无失败) → expand 饱和.
                    # 算子H (strengthen): 若仍有失败 (efficacy gap) 且 --strengthen 开, 不静默跳过,
                    # 选一个现有段做加固 (加/精炼一条 bullet). 逻辑抽到 _run_strengthen 供两处饱和入口复用.
                    did_strengthen, expand_event, best_val_delta, best_val_ci, best_skill, best_node = \
                        _run_strengthen(args, t, counts, reps, best_node, freeze, title,
                                        s0_cache, probe_items, base_pass, meta_mask, archive,
                                        best_val_delta, best_val_ci, best_skill, expand_event,
                                        op_counts=iter_counts, pool_gates=pool_gates)
                    if not did_strengthen and expand_event is None:
                        # 无 strengthen / strengthen 未触发 / 未采纳 → 记饱和 skip (保既有诊断)
                        # (守 §五纪律: 不为造段烧 API; strengthen 未产出 CI>0 候选时诚实回退)
                        expand_event = {"admitted": False, "reason": "all-covered-no-gap",
                                        "n_fail": len(all_failures), "n_sections": len(best_node.slots),
                                        "failure_counts": {k: v for k, v in counts.items()}}
                        print(f"  ⏭️ expand iter{t}: skipped (all non-meta types covered; "
                              f"n_fail={len(all_failures)} n_sec={len(best_node.slots)})", flush=True)
                else:
                    exp_failures = reps.get(target, [])  # reps 已按最难排序, 每类≤3
                    exp_diag = (make_diagnostics(exp_failures) if exp_failures
                                else "(no failure reps for target type)")
                    existing_names = "/".join(best_node.slots.keys())
                    for cand_i in range(args.expand_candidates):
                        # 候选 section 名: guard 校验 LLM 吐的名字必须 == expected_key (不同名拒).
                        iter_counts["expand"]["attempted"] += 1  # expand: 候选进 LLM 调用 (F08/Q6)
                        try:
                            new_sec = llm(EXPAND_SYSTEM, SECTION_EXPAND_PROMPT.format(
                                existing_sections=existing_names, diagnostics=exp_diag,
                                target_type=target),
                                max_tokens=2048, stage="evolver_expand")
                        except Exception as e:  # noqa: BLE001
                            expand_event = {"admitted": False, "reason": "llm-error", "target_type": target}
                            print(f"  ⚠️ expand iter{t} cand{cand_i} 失败: {e!r}", flush=True)
                            break
                        raw = clean_skill(new_sec, "")
                        # 从 LLM 输出拆新 section 名 (名字由 LLM 自起, 用于 expected_key)
                        _, raw_slots = facet_decompose(raw)
                        if len(raw_slots) != 1:
                            expand_event = {"admitted": False, "reason": f"cand{cand_i}-not-one-section",
                                            "target_type": target}
                            iter_counts["expand"]["guard_rejected"] += 1  # expand: LLM 输出非单 section
                            continue
                        exp_key = next(iter(raw_slots))
                        if exp_key in best_node.slots:
                            expand_event = {"admitted": False, "reason": f"cand{cand_i}-name-collide",
                                            "target_type": target}
                            iter_counts["expand"]["guard_rejected"] += 1  # expand: section 名碰撞
                            continue
                        ex_slots, ex_ok = section_expansion_guard(
                            best_node.slots, exp_key, raw)
                        if not ex_ok:
                            expand_event = {"admitted": False, "reason": f"cand{cand_i}-guard-reject",
                                            "target_type": target}
                            iter_counts["expand"]["guard_rejected"] += 1  # expand: section guard 拒绝
                            continue
                        ex_skill = facet_compose(title, ex_slots)
                        iter_counts["expand"]["evaluated"] += 1  # expand: 进 eval (guard 通过)
                        ep, ed, _, eitem = eval_on(probe_items, ex_skill, s0_cache,
                                                   stage="evolver_expand_val", k=args.eval_k,
                                                   eval_deadline_s=EVAL_DEADLINE_S)
                        el = lcb_score_from_deltas(eitem, s0_cache, args.eval_k)[0] if args.select_lcb else bootstrap_ci_lower(eitem)
                        admitted = (ep >= base_pass - 0.06) and (el > 0)  # LOST guard + ci>0 准入 (镜像 overlay)
                        became_best = False
                        # 算子H 饱和判定 (修 strengthen run 暴露的设计缺口, 用户指定):
                        #   "已有专段且新变体 delta < 旧段 delta 才判饱和, 否则 expand 抢先".
                        #   target 类型已在 type_best_delta 有记录 = 该类型已有采纳过的专段. 此时若新变体
                        #   delta (ed) 回落到低于旧段最佳 delta → 判该类型饱和: 停止本代 expand (break),
                        #   记 type-saturated 诊断, 交由下方 _run_strengthen 加固既有专段而非长冗余变体.
                        #   新变体 delta >= 旧段 (更强或持平, 如 iter4 +0.127 → iter6 +0.258) → 不饱和,
                        #   expand 正常采纳并刷新 type_best_delta (坚持探索不半途而废 — 用户方法论锚点).
                        _type_saturated = False
                        if target in type_best_delta and ed < type_best_delta[target]:
                            _type_saturated = True
                        if _type_saturated:
                            expand_event = {"section_name": exp_key, "target_type": target,
                                            "delta": round(ed, 4), "ci": round(el, 4),
                                            "admitted": False, "reason": "type-saturated",
                                            "prev_best_delta": round(type_best_delta[target], 4),
                                            "n_sections": len(best_node.slots)}
                            print(f"  🕳️ expand iter{t} cand{cand_i}: [{target}] saturate "
                                  f"(new delta={ed:+.3f} < prev_best={type_best_delta[target]:+.3f}); "
                                  f"hand to strengthen", flush=True)
                            break  # 判饱和: 停 expand, 跳候选循环交 strengthen 加固
                        if admitted:
                            ex_node = _ArchiveNode(ex_skill, ex_slots, f"expand:{exp_key}",
                                                   parent=best_node.facet_label, val_delta=ed, val_ci=el)
                            archive.append(ex_node)
                            iter_counts["expand"]["admitted"] += 1  # expand: LOST+ci>0 准入入 archive
                            # 刷新该类型采纳段的最佳 delta (供后续饱和判定: 新变体回落到 < 此值时触发).
                            # 取 max 而非直接赋值: 同类型跨代多次 expand 只记历次最强变体, 避免偶发弱采纳
                            # 拉低阈值误判饱和. (坚持探索: 更强变体始终有机会通过 expand, 不被早 iter 弱段封顶.)
                            type_best_delta[target] = max(type_best_delta.get(target, float("-inf")), ed)
                            # break-①: expand 节点现可作 donor (select_section_donors 看 section 名而非
                            # facet_label), 新 section 进遗传池供后续 crossover/overlay 继承.
                            expand_event = {"section_name": exp_key, "target_type": target,
                                            "delta": round(ed, 4), "ci": round(el, 4),
                                            "admitted": True, "became_best": became_best,
                                            "type_best_delta": round(type_best_delta[target], 4),
                                            "n_sections_after": len(ex_slots)}
                            print(f"  🧬 expand iter{t} cand{cand_i}: +{exp_key} [{target}] delta={ed:+.3f} CI={el:+.3f} "
                                  f"pass={ep:.3f} admitted=Y n_sections={len(ex_slots)}", flush=True)
                            if ed > best_val_delta and el > 0:
                                best_val_delta, best_val_ci, best_skill = ed, el, ex_skill
                                best_node = ex_node  # facet_label="expand:{name}", 进遗传池
                                expand_event["became_best"] = True
                                print(f"  ✅ EXPAND-BEST iter{t}: +{exp_key} [{target}] delta={ed:+.3f} CI={el:+.3f}",
                                      flush=True)
                                iter_counts["expand"]["promoted"] += 1  # expand: TREE-BEST 夺魁
                            else:
                                # break-③: 整体未超 best, 看 per-subset 信号救援
                                # (expand-v2 的价值正落在 not-meta 子集, 整体被 meta 淹没时需救援)
                                _promote, _sig = subset_promote_decision(eitem, best_val_delta, meta_mask, enabled=args.subset_admit)
                                # Bug C floor guard: 见 crossover/overlay 同名守卫注释. 救援晋升 best 不得显著拉低 delta.
                                _eps_floor = 0.05
                                if _promote and ed >= best_val_delta - _eps_floor:
                                    best_val_delta, best_val_ci, best_skill = ed, el, ex_skill
                                    best_node = ex_node
                                    expand_event["became_best"] = True
                                    expand_event["subset_promote"] = {
                                        "not_meta_ci": round(_sig["not_meta"]["ci"], 4),
                                        "not_meta_n": _sig["not_meta"]["n"]}
                                    print(f"  ✅ EXPAND-SUBSET-BEST iter{t}: +{exp_key} [{target}] delta={ed:+.3f} "
                                          f"(not-meta ci={_sig['not_meta']['ci']:+.3f} n={_sig['not_meta']['n']})",
                                          flush=True)
                                    iter_counts["expand"]["promoted"] += 1  # expand: SUBSET-BEST 救援晋升
                                elif _promote:
                                    expand_event["subset_gated"] = {
                                        "not_meta_ci": round(_sig["not_meta"]["ci"], 4),
                                        "not_meta_n": _sig["not_meta"]["n"]}
                                    print(f"  🔶 EXPAND-SUBSET-GATE iter{t}: +{exp_key} [{target}] delta={ed:+.3f} "
                                          f"< best{best_val_delta:+.3f}-floor{_eps_floor} "
                                          f"(not-meta ci={_sig['not_meta']['ci']:+.3f}); archive-only",
                                          flush=True)
                            break  # 采纳即停 (避免同代重复膨胀)
                        else:
                            # break-③ 诊断回路: 拒绝时记录 per-subset 信号 (不静默 skip — 研究"为什么没接受")
                            _sig = subset_signal(eitem, meta_mask)
                            expand_event = {"section_name": exp_key, "target_type": target,
                                            "delta": round(ed, 4), "ci": round(el, 4),
                                            "admitted": False, "reason": "lost-or-ci<=0",
                                            "not_meta_ci": round(_sig["not_meta"]["ci"], 4),
                                            "not_meta_n": _sig["not_meta"]["n"],
                                            "meta_ci": round(_sig["meta"]["ci"], 4)}
                    # expand 候选循环结束. 若因单类型饱和而 break (expand_event.reason == "type-saturated"),
                    # 该类型已有专段且新变体 delta 回落 → 交 strengthen 加固既有专段 (复用 target is None 同一入口).
                    # (用户: "已有专段且新变体 delta < 旧段 delta 才判饱和, 否则 expand 抢先" — 此处正是判饱和后
                    #  把控制权从 expand 转给 strengthen, 关闭 strengthen run 暴露的"strengthen 永不触发"缺口.)
                    if expand_event is not None and expand_event.get("reason") == "type-saturated":
                        _ds, expand_event, best_val_delta, best_val_ci, best_skill, best_node = \
                            _run_strengthen(args, t, counts, reps, best_node, freeze, title,
                                            s0_cache, probe_items, base_pass, meta_mask, archive,
                                            best_val_delta, best_val_ci, best_skill, expand_event,
                                            op_counts=iter_counts, pool_gates=pool_gates)
                    # 方案1 (per-type strengthen 触发放宽): expand 仍能找到新未覆盖类型 (target is not None)
                    # 时, 若存在「已覆盖但仍在失败」的既有类型, 也 fire strengthen 加固其专段.
                    # 这是 ADDITIVE 路径 — 不替换 expand (expand 已在上跑完), 不 block 探索 (find_uncovered_type
                    # 不变). 关闭 9/5/9/7 run 暴露的"strengthen 永不触发"结构性缺口 (全覆盖前提 4 iter 不可达).
                    # (用户方法论: 坚持 expand 为主探索, strengthen 互补; 过早放弃风险由 exclude=target 最小 guard.)
                    if getattr(args, "strengthen_per_type", False) and _STRENGTHEN_ON:
                        # gate: 是否存在已覆盖+仍失败的既有类型 (排除本 iter 刚覆盖的 target).
                        # covered_target 仅决定是否 fire; _run_strengthen 内部 select_strengthen_target
                        # 自行按 _section_axis_score (双向测试关键词) 选加固哪段 — 不受 covered_target 约束.
                        # Task #7 修复: gate 的「已覆盖」判定改用 archive union (_archive_section_union)
                        # 而非 best_node.slots.keys(). 原因: expand 产物若 became_best=False (SUBSET-GATE
                        # 或被同 iter 更强候选夺 best) 则只进 archive 不进 best_node 链 → 其专属 section
                        # 对 gate 不可见 → 返回 None → strengthen 全程不 fire (9/8 run iter1/iter2 坐实).
                        # archive union = 历史组合并集, 落实"历史中的组合可能会更好" — 未夺 best 的段
                        # 仍是合法已覆盖类型, 不应失明. expand 不受影响 (find_uncovered_type 仍用 best_node).
                        covered_target = find_covered_failing_type(
                            counts, _archive_section_union(archive), exclude=target)
                        if covered_target is not None:
                            _ds2, expand_event, best_val_delta, best_val_ci, best_skill, best_node = \
                                _run_strengthen(args, t, counts, reps, best_node, freeze, title,
                                                s0_cache, probe_items, base_pass, meta_mask, archive,
                                                best_val_delta, best_val_ci, best_skill, expand_event,
                                                op_counts=iter_counts, pool_gates=pool_gates)
                            if _ds2:
                                print(f"  🔧 per-type strengthen iter{t}: fired on covered-failing type "
                                      f"[{covered_target}] (expand target was [{target}])", flush=True)
            else:
                expand_event = {"admitted": False, "reason": "max-sections-reached"}

        # 诊断: 累积进化判据 (每代末, 纯只读). 测 best 的 non_native_s0_ratio — 现状≈1.0 (累积失效),
        # 修法1 叠加出多轴 best 后应降 <1.0 (overlay 命中时 facet_label="overlay" 无本轴豁免 → 可降至 0).
        # ⚠️ 必须测 best 不测 leaves: 纯叠加方案下叶 ratio 恒≈1.0 (叶仍各改各的), 是预期不是失败.
        # best 仍是 init_skill 的 corner (gen1 全未过 ci>0): 直接 ratio=1.0, 不进本轴排除分支.
        if md5(best_skill) == md5(init_skill):
            best_ratio = 1.0  # best 未被替换 = 全段 S0 → 累积完全失效
        else:
            best_ratio = _facet_non_native_s0_ratio(best_node, freeze, root_slots)
        per_leaf_diag = [{"facet": l.facet_label,
                          "ratio": round(_facet_non_native_s0_ratio(l, freeze, root_slots), 4)}
                         for l in leaves]
        leaves_ratio_mean = (sum(p["ratio"] for p in per_leaf_diag) / len(per_leaf_diag)
                             if per_leaf_diag else 1.0)
        cumulative_diag = {
            "best_ratio": round(best_ratio, 4),               # 判据对象: 现状≈1.0, 修复后应降
            "best_native": best_node.facet_label,             # 单轴名 / "overlay"
            "leaves_ratio_mean": round(leaves_ratio_mean, 4),  # 现状基线: 纯叠加下恒≈1.0
            "per_leaf": per_leaf_diag,
        }

        # persistence: 树结构 json + 每叶 md (输家不删, MAP-Elites 归档)
        tree_nodes = [{"facet": l.facet_label, "parent": l.parent, "hash": md5(l.skill),
                       "val_delta": round(l.val_delta, 4), "val_ci": round(l.val_ci, 4),
                       "skill": l.skill} for l in leaves]
        json.dump(tree_nodes, open(os.path.join(out_dir, f"tree_iter{t}.json"), "w",
                                   encoding="utf-8"), ensure_ascii=False, indent=2)
        # 【Bug B 修复】每轮落盘完整 archive (含 overlay/crossover/expand 复合节点 + slots).
        # 旧版仅终局 archive_final.json, 中途 kill 丢复合 best (PID 53188 死锁时 n_sec=6 overlay
        # best 被丢). 现 per-iter 落 archive_iter{t}.json + best_skill_iter{t}.md, 中断可恢复.
        archive_dump = [{"facet": nd.facet_label, "parent": nd.parent,
                         "hash": md5(nd.skill),
                         "val_delta": round(nd.val_delta, 4),
                         "val_ci": round(nd.val_ci, 4),
                         "n_sec": len(nd.slots),
                         "slots": nd.slots, "skill": nd.skill} for nd in archive]
        json.dump(archive_dump, open(os.path.join(out_dir, f"archive_iter{t}.json"), "w",
                                     encoding="utf-8"), ensure_ascii=False, indent=2)
        # best skill 单独存 md (直接可用, 无需解析 archive)
        open(os.path.join(out_dir, f"best_skill_iter{t}.md"), "w", encoding="utf-8").write(best_skill)
        _accumulate_counts(run_total, iter_counts)  # 跨轮汇总 → 终局 0-promoted 告警 (F08/Q6/RF §6 Step 3)
        log["iters"].append({"t": t, "facet_tree": True, "secs": round(time.time() - t0, 1),
                             "n_active": len(leaves), "n_archive": len(archive),
                             "best_val_delta": round(best_val_delta, 4),
                             "best_val_ci": round(best_val_ci, 4),
                             "best_n_sec": len(best_node.slots),
                             "best_facet": best_node.facet_label,
                             "crossover": xo_event,
                             "overlay": overlay_event,
                             "expand": expand_event,
                             "cumulative_diag": cumulative_diag,
                             "branch_failure_types": branch_failure_types.get(t, {}),
                             "operator_counts": iter_counts,
                             "pool_filter_gates": pool_gates})
        json.dump(log, open(os.path.join(out_dir, "evolver_log.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    # archive 落盘 (终局 probe-replay 的素材池)
    json.dump([{"facet": nd.facet_label, "parent": nd.parent, "hash": md5(nd.skill),
                "val_delta": round(nd.val_delta, 4), "val_ci": round(nd.val_ci, 4),
                "skill": nd.skill} for nd in archive],
              open(os.path.join(out_dir, "archive_final.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log["crossover_fired"] = crossover_fired
    log["archive_size"] = len(archive)
    log["operator_counts"] = run_total  # 全 run 算子计数汇总 (供 main() 终局 0-promoted 告警; F08/Q6/RF §6 Step 3)
    return best_skill, best_val_delta, best_val_ci


def main():
    ap = argparse.ArgumentParser(description="Skill 自进化迭代器 (免训练, 超 SkillOpt-best)")
    ap.add_argument("--init_skill", default=V2_SKILL_DEFAULT, help="起点 skill (默认 v2 当前)")
    ap.add_argument("--s0", default=S0_DEFAULT)
    ap.add_argument("--split_dir", default=SPLIT_DEFAULT)
    ap.add_argument("--s0_cache", default=S0_CACHE_DEFAULT, help="S₀ cache json (同条件 k 投票连续值)")
    ap.add_argument("--shuffle_seed", type=int, default=0,
        help="Phase B (P0/F01 修复): 非 None 时 train/val 按 (qid,seed) 确定性打乱选项 (移除 '选 A' 位置捷径+meta 位置伪影). "
             "默认 0=打乱口径 (源数据 correct_choice 单存+choices 不含正确项, dataloader 排序后 gold 恒 A → 锚句 skill 在恒A val 上被正向选择, "
             "§8.1 因果实验已坐实 L20 anchor→位置偏置 +0.1392). 打乱后 gold 散在 A-E, 锚句 skill 在 val 上下降被自然淘汰. "
             "历史未打乱口径 (0.462/on=0.587) 由 9/16 run 产物存档, 无需再复现. "
             "注意: s0_cache 须为同 seed 打乱口径下生成, load 时按 manifest 校验 choice_arrangement_hash, 不匹配则弃用重算.")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--K", type=int, default=4, help="每轮候选数")
    ap.add_argument("--eval_k", type=int, default=5, help="评测每题采样投票次数 (降 temp 非确定噪声, 同 v2)")
    ap.add_argument("--n_parallel", type=int, default=1,
        help="Step1: 每分支每代并行 N 个 facet-edit 候选 (变体提示注入多样性), 各过 guard+eval, "
             "取 delta 最优晋级其余入 archive (keep-losers). 默认 1=字节复现既有单候选路径. "
             "gated by --facet_tree. 代价: eval 调用 ×N (每候选 36题×k), 共享 28RPM 预算")
    ap.add_argument("--parallel_diversity", default="auto",
        help="Step1: N 候选多样性来源. auto=变体角度提示注入 FACET_EDIT_PROMPT (默认); "
             "temp=每候选 set_temp 扰动 (串行, 失并行). 仅 --n_parallel>1 生效")
    ap.add_argument("--n_diag", type=int, default=8, help="每轮诊断失败题数")
    ap.add_argument("--train_n", type=int, default=35, help="选择集用 train 题数")
    ap.add_argument("--consolidate_every", type=int, default=3, help="每 N 轮慢更新")
    ap.add_argument("--patience", type=int, default=3, help="连续 N 轮无提升则探索/停")
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    # ── 机制消融 (leave-one-out, 互斥使用; 无 flag = FULL 完整 evolver, 与主结果路径一致) ──
    ap.add_argument("--ablate_gepa", action="store_true", help="A1: K=1 单候选, 消融多候选种群+选择 (M1)")
    ap.add_argument("--ablate_failure", action="store_true", help="A2: 通用 patch 不喂诊断失败, 消融失败驱动 patch (M2)")
    ap.add_argument("--ablate_consolidate", action="store_true", help="A3: 禁用慢更新, 消融 consolidate (M3)")
    ap.add_argument("--ablate_antideg", action="store_true", help="A4: 去 LOST/rejected/focus 轮换, 消融防退化+多样性 (M4)")
    # --- 算法优化 quick-win 开关 (2026-08-05, 默认关 = 保主结果复现) ---
    ap.add_argument("--select_by_ci", action="store_true", help="A1: winner 按 CI 下界选优 (治 val-overfit), 默认 argmax delta")
    ap.add_argument("--fix_exploration", action="store_true", help="C1: 修复探索模式死代码 (卡住时切 bolder angle hints), 默认沿用原行为")
    ap.add_argument("--cache_failures", action="store_true", help="D1: skill 未变时复用上轮 failures 缓存, 省重跑")
    ap.add_argument("--allow_modify", action="store_true", help="dir-4: 允许 patcher 定向删除/替换 S₀ 中有害内容 (默认纯追加, 保主结果)")
    ap.add_argument("--learn_positive", action="store_true", help="dir-5: 从 skill 解对的难题蒸馏正信号 (成功解法->可推广决策程序), 与失败模式互补")
    ap.add_argument("--meta_contrast", action="store_true", help="路径B (2026-09-10): 诊断对照对称化. collect_failures 额外收集 skill 解对的 meta 题 (零额外 API), 把每道 meta 错题配对一道同构 meta 对题并列展示, 让 patcher 从 '同是 meta 题, 做对的选了 meta 选项' 自主归纳规则 (不注入结论文字). 默认关=字节复现既有路径")
    ap.add_argument("--gate_consolidate", action="store_true", help="M3-gate: consolidate 后 re-eval 选择集, delta 下降超 GATE_MARGIN 则回退 (治盲压缩损质, RunA/B 均观测到 consolidate 后 failure 上扬)")
    ap.add_argument("--data_driven_focus", action="store_true", help="Task3: focus 按已诊断失败的多数类型选 (classify_failure_type), 非固定 round-robin; 默认关=保主结果复现. 根因: val 最难失败 ~93%% 是 meta-option, 原 round-robin 早 iter 用错方向 patch")
    ap.add_argument("--coverage_diagnosis", action="store_true", help="Task4: collect_failures 返回全失败分布 (零额外 API), focus 选 '尚未覆盖类型' (find_uncovered_type 填洞) 非多数类型; patcher 看全分布 tally+分层样本. 默认关=保主结果复现. 治 val 过拟合 + 为 C 结构自进化提供分布数据基底")
    ap.add_argument("--structure_evolution", action="store_true", help="Phase C: 结构自进化. 每轮额外生成 Kc 个结构变体候选 (内容重组 v2 skill, 不注入新内容), 与 K 个 patch 候选同台竞争 pass_rate. 治 Task4 根因 (同内容不同结构=10pp 差, 结构是因果变量). 默认关=字节复现既有路径; 关时 generate_structure_candidates 永不被调用, mutation_kinds 恒为 ['patch']*K")
    ap.add_argument("--Kc", type=int, default=4, help="Phase C: 每轮结构变体候选数 (仅 --structure_evolution 时生效)")
    ap.add_argument("--struct_temp", type=float, default=1.0, help="Phase C: 结构生成采样温度 (仅 --structure_evolution 时生效); 默认 1.0 高温促结构多样性 (规则5 防 SkillOpt 病: padding 泛泛建议由 validator CI>0+novel 兜底)")
    ap.add_argument("--resume", default=None, help="无损续传: 传入 prior run 目录 (含 evolver_log.json + skill_iterN.md), 恢复 best_skill+baseline+辅助状态, 跳过 baseline 重测, 从 len(iters)+1 续跑. --iters = 再跑轮数. 注意: --structure_evolution/--Kc/--struct_temp 不自动恢复, 续传时必须重传")
    # --- Task5: 诊断-选择解耦 probe 集 (2026-08-20, 默认关 = 保 0.462 主结果复现) ---
    # 根因: collect_failures(诊断) 与 eval_on(选 winner) 共用同一 select_items → 同源过拟合链
    # (patcher 针对 val 特定 8 题内容写决策程序, 恰又在这批题上被选为 winner). 解法: 诊断走 diag_items,
    # winner 选择走不相交 probe_items → diag 过拟合程序在 probe(诊断未见内容)上被自然拒绝.
    # 详见 memory/eval-signal-debiasing-diagnosis-20260820.md (0.462 非 val 失败类型偏差假天花板,
    # 真瓶颈 = val 内容过拟合; 重采样去失败类型偏差原理上部分无效, 故转诊断-选择解耦).
    ap.add_argument("--probe_split", action="store_true",
        help="Task5: 诊断-选择解耦. 将 select_items 随机切为不相交 diag_items(诊断)+probe_items(选择信号), 切断同源过拟合链. 默认关=字节复现既有路径(0.462 主结果)")
    ap.add_argument("--probe_ratio", type=float, default=0.5,
        help="Task5: probe 占比 (probe=round(N*ratio), diag=其余). 默认 0.5")
    ap.add_argument("--probe_seed", type=int, default=42,
        help="Task5: 切分随机种子 (固定=可复现/resume 安全). 默认 42")
    # --- 药三 (2026-09-11): 剔除 train 重叠 (val→test 负迁移根因③) ---
    # 根因: select_items=train+val=71, train 18 题从头到尾在候选计分/晋升集里 = 在"背过答案的题"上选冠军,
    # 天然乐观偏差 (winner 的 val delta 被 train 重叠抬高). 解法: 候选选择/晋升(eval_on 的 probe_items)
    # 只落纯 val 53 题, diag_items(诊断失败样例)仍看全 71 题(多看失败无害). 默认关=字节复现既有路径.
    ap.add_argument("--select_exclude_train", action="store_true",
        help="药三: 候选选择/晋升计分集剔除 train 18 题, 只用纯 val 53 题 (治 train⊂val 数据泄露). 默认关")
    # --- 药一 (2026-09-14): Beta-Binomial LCB 选择准则 (val→test 负迁移根因① 赢家诅咒) ---
    # 根因: bootstrap_ci_lower 在小样本 (probe 53~71 题、每题 k=3 投票) 的 CI 下界过窄,
    # 低估方差 → delta 被采样噪声高估的候选当选冠军 (winner's curse), test 迁移即崩.
    # 解法: 候选选择/晋升/门控的置信量改用 Beta-Binomial 后验 25% 分位 (Jeffreys 先验 Beta(0.5+k, 0.5+n-k)),
    # 在 n 小时自然收缩向 0.5 通过率, 悲观惩罚噪声候选。默认关=字节复现既有 bootstrap-CI 路径.
    ap.add_argument("--select_lcb", action="store_true",
        help="药一: 候选选择/晋升门控改用 Beta-Binomial LCB 25%% 后验分位 (治赢家诅咒), 替换 val_ci>bootstrap 下界. 默认关")
    # --- Facet-Tree (Stage 0): facet 分解/交叉 (greenfield, 设计 §七 Step 3-4) ---
    ap.add_argument("--self_test_facet", action="store_true",
        help="只跑 facet_decompose/compose 纯函数无损单测后退出 (零 API, Step 3 验收)")
    ap.add_argument("--facet_tree", action="store_true",
        help="Facet-Tree Stage 0: 启用树遍历 + facet 分解/交叉 (替单 lineage). 关=字节复现既有 0.462 路径")
    ap.add_argument("--facet_axes", default="Option Comparison,Theorem-Level Precision,Hypotheses,Final Answer",
        help="冷启动 facet 轴 (逗号分隔, 对齐 S₀ 段名). Stage 0 固定 4 轴")
    ap.add_argument("--freeze_facet", default="Final Answer",
        help="冻结 facet (不参与变异/交叉, 原样保留). 默认 Final Answer")
    ap.add_argument("--failure_modes", default="", help="failure_modes 空 slot 占位 (Stage 0 关涌现, 物理空)")
    ap.add_argument("--no_emergence", action="store_true",
        help="显式关 §2.2(b) 涌现 (Stage 0 < K=3 门控, 双保险)")
    # --- Facet-Tree GA 算子开关 (设计 五算子, 默认 on=完整 C 组; 关对应消融组) ---
    # 完整 GA = bootstrap(异构环境) + 交叉(slot 重组) + guardrail(facet-edit + keep-losers archive).
    # 消融矩阵: B=关交叉关bootstrap(纯树+guard) / D=关交叉开bootstrap / E=开交叉关bootstrap / C=全开.
    # BooleanOptionalAction 自动生成 --bootstrap/--no-bootstrap, --crossover/--no-crossover (Python3.9+).
    ap.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True,
        help="算子A: DIVE bootstrap — diag_items 切 n_branches 不相交子集, 每分支诊断自己的子集 (异构环境). "
             "默认 on=完整 C 组; --no-bootstrap 关 = 各分支共享全 diag (消融 E 组, 归因交叉独立效应)")
    ap.add_argument("--crossover", action=argparse.BooleanOptionalAction, default=True,
        help="算子C: gen≥2 末 2-parent facet 交叉 (binary-tournament 选 donor + recombine + coherence + 准入). "
             "默认 on=完整 C 组; --no-crossover 关 = 消融 D 组 (归因 bootstrap 独立效应, 直接答'bootstrap 帮/害')")
    ap.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True,
        help="算子E: 每代末跨轴叠加 best (修法1). 取每轴 val_delta 最大且 ci>0 且本轴段经变异的 donor, "
             "组装'全轴最优叠加 skill' eval 后竞争 global best (打通跨轴累积, 否则 best 恒'单轴最优+S0其余'). "
             "默认 on; --no-overlay 关 = 消融 (退回单轴 GA, best_ratio≈1.0 坐实累积失效). "
             "gated by --facet_tree")
    ap.add_argument("--preserve_extend", action="store_true",
        help="算子B 终局: preserve-and-extend 契约 (child 解题集 ⊇ 父本并集, 否则降级 stepping-stone). "
             "默认 off (Stage 0 数据太薄, 仅记录不门控); 终局 C 组多代开为 survivor 准入")
    ap.add_argument("--probe_replay", action="store_true",
        help="算子F 终局: 终选对 archive top-K 用更高 k re-eval, 取 CI 下界最高者 (跳出'自评自选'闭环). "
             "与每代算子E(overlay)独立, 是终局独立的 refinement 机制. "
             "默认 off (Stage 0 用现有 best_val_delta 满足 go/no-go); 终局开为 refinement")
    ap.add_argument("--pop_mutation", action="store_true",
        help="方向E 落点②: 每 branch 突变起点从单 leaf(incumbent) 扩到同 facet archive 精英种群加权采样 "
             "(val_ci 权重 + ε-greedy 留 incumbent). 打破单 lineage keep-best — 治 val→test 负迁移真根因 "
             "(见 peak-diagnosis-val-to-test-negative). gen1 退回单 lineage 安全 (同 facet archive 仅根叶). "
             "promotion 逻辑不变 (只升不降). 默认 off = 字节复现既有单 lineage 路径. gated by --facet_tree")
    ap.add_argument("--pop_eps", type=float, default=0.3,
        help="方向E: pop_mutation 的 ε-greedy 概率 (返回 incumbent leaf 的概率). 默认 0.3 = 70%% 种群采样 / "
             "30%% incumbent 探索. 仅 --pop_mutation 生效")
    ap.add_argument("--expand", action=argparse.BooleanOptionalAction, default=False,
        help="算子G(展开): 每 --expand_every 代末给 best 追加一个全新 section (决策程序/领域特化), "
             "突破 facet_edit「单slot单bullet」能力边界. 血统干净: 新 section 仅由失败诊断驱动 "
             "(无 skillopt 内容/无人类 seed). 默认 off (向后兼容既有 run 零影响); 另见 --expand_every/--expand_candidates")
    ap.add_argument("--expand_every", type=int, default=1,
        help="算子G 触发频率 (每 N 代展开一次). 默认 1 (每代末都试, 控 --expand 的 API 成本)")
    ap.add_argument("--expand_candidates", type=int, default=1,
        help="算子G 每次展开的候选数 (最多试 N 个名字, 采纳即停). 默认 1 (控 --expand 的 API 成本)")
    ap.add_argument("--strengthen", action=argparse.BooleanOptionalAction, default=False,
        help="算子H(加固): expand 饱和(全非-meta 类型已覆盖但仍有失败)时, 不静默跳过, 而是选一个现有段 "
             "加/精炼一条 bullet (复用 facet_edit_guard, 对 base 轴和 expand 长出的段均工作). 关闭干净路径 "
             "停在第 1 双向轴的结构性 gap (theo_iter6 0.5542 vs evo_seed 0.5590 缺第 2 轴 existence↔uniqueness). "
             "协同: 同时后移 classify_failure_type 的 meta 检查 (让 existence/uniqueness 不被 meta 吞), 使 "
             "find_uncovered_type_for_expand 与 strengthen 都能看到第 2 轴. gated by --expand. "
             "默认 off = 字节复现既有 run; 开 = 饱和迭代变有效迭代. 见 memory/bidirectional-test-convergence-20260904.md")
    ap.add_argument("--strengthen_per_type", action=argparse.BooleanOptionalAction, default=False,
        help="方案1: 放宽 strengthen 触发为 per-type. 即使 expand 仍能找到新未覆盖类型 (target is not None), "
             "若存在「已覆盖但仍在失败」的既有类型, 也 fire strengthen 加固其专段 (ADDITIVE, 不替换 expand). "
             "关闭 9/5/9/7 run 暴露的「strengthen 永不触发」结构性缺口 (全覆盖前提 4 iter 内不可达). "
             "too-early-abandonment guard: exclude 本 iter 刚覆盖的 target (给新段 1 iter 自然使用窗口). "
             "默认 off = 字节复现既有 run; 开需同时 --strengthen --expand. 见 docs/phase2_issues.md 2026-09-07 方案1.")
    # --- break-①②③ 算法优化开关 (2026-09-02, 默认关 = 保既有 run 字节复现) ---
    ap.add_argument("--donor_topk", type=int, default=1,
        help="break-①: section 级 donor 池每段保留 top-K 历史 (按 val_ci 降序). "
             "默认 1 (= 旧单 top-1 行为); >1 允许历史回退和路径变化 (用户: '历史中的组合可能会更好'). "
             "K>1 时偶数代采次优 donor 探索历史路径.")
    ap.add_argument("--xo_pair_retries", type=int, default=0,
        help="靶点1: crossover cohere/guard 失败时换 parent 对重试次数. "
             "旧版每代仅试 1 对 (pool[0]×pool[1]) 即弃 → evolver_log 显示 4/7 交叉死在 guard/cohere-fail "
             "(从未进 eval, 非价值拒绝). 设 N 后每代最多尝试 (1+N) 对不同 parent, 任一对重组成功即进 eval. "
             "默认 0 = 旧单对行为. 落实用户'历史中的组合可能会更好' — 多对试出可接受重组.")
    ap.add_argument("--xo_cohere_retries", type=int, default=0,
        help="靶点1: recombine_facets 内 cohere-LLM 单次失败时 re-prompt 重试次数 (温度扰动 +0.15/次). "
             "4/7 guard-fail 主因 = cohere-LLM 一次性产不出合规结构 (非重组价值不够). 重试给 LLM 多一次机会. "
             "默认 0 = 旧单次即弃; 段数/freeze 守卫失败不重试 (产出了但内容不合, 由 --xo_pair_retries 换 parent).")
    ap.add_argument("--xo_allow_merge", action=argparse.BooleanOptionalAction, default=False,
        help="靶点1: 放宽 recombine_facets 段数守卫 — 允许 cohere-LLM 合并两段 (2→1). "
             "旧守卫 `len(child)<len(parent_a)` 硬拒任何降段数, 但 cohere-prompt 明确允许 'MAY merge two sections' "
             "→ 合法 merge 被误杀. 新守卫: 降段数本身不拒, 仅当非 freeze 段总数减少 (= 真丢内容) 才拒. "
             "默认 off = 旧硬守卫; --xo_allow_merge 开 = 语义守卫.")
    ap.add_argument("--xo_multiparent", type=int, default=0,
        help="方向 B: 多 parent crossover — 一次把 N 个 donor 的不同 section 段同时嫁接进 base, "
             "再一次 coherence pass (recombine_facets_multi). 比 2-parent 单段嫁接覆盖更多失败类型. "
             "默认 0 = 旧 2-parent 行为 (recombine_facets, 字节复现); >0 = 多 parent, 每代最多嫁接 N 段. "
             "N>=2 时若 archive 中定向 donor 不足 N 个, 退化为实际可用数. 见 docs/phase2_issues.md 方向 B.")
    ap.add_argument("--xo_donor_directed", action=argparse.BooleanOptionalAction, default=False,
        help="方向 B: 定向 donor 选择 — select_donors_directed 按 section body 对当前失败分布 "
             "(counts) 的覆盖匹配度排序而非纯 val_ci. 让多 parent 嫁接对准当前仍在失败的类型缺口. "
             "默认 off = 用 select_section_donors (val_ci 排序); 开 = 用 select_donors_directed. "
             "与 --xo_multiparent 配合: directed 选出的 N 个 donor 各覆盖一个不同失败类型.")
    ap.add_argument("--subset_admit", action=argparse.BooleanOptionalAction, default=True,
        help="break-③: per-subset 准入 (has_meta 切 meta/not-meta 两子集, 零 API 确定性标签). "
             "整体未超 best 时, '某子集 ci>0 且无子集退步超 ε' 仍可晋升 best (救援 not-meta 增益). "
             "默认 on; --no-subset_admit 关 = 回退纯整体单目标 (旧行为). "
             "对齐 expand-v2: 整体 −0.021 不显著但 not-meta +0.157 p=0.0235 显著.")
    ap.add_argument("--pool_filter", action="store_true",
        help="fix-① 候选池准入 (2026-09-15): archive.append 前过三关 — ①质量关(整体 meanΔ>=-eps 平均不回落) "
             "②尖峰关(5峰中有峰 CI下界>0) ③多样性关(git-diff 归一化相似度<0.95 实质 diff). ②或③任一过(尖峰/多样性 "
             "二选一)且①必过才入池; 淘汰者不入 archive(不污染交叉/overlay 素材池). 分峰依据 5 峰合并 "
             "(见 [[pool-admission-8dim-spike-split-20260915]]). 默认 off = 字节复现 theo_iter6 路径.")
    ap.add_argument("--pool_eps", type=float, default=0.02,
        help="fix-① 质量关 eps: 平均回落的噪声容忍带宽 (~1σ per-question Δ). Δ在[-eps,0)视为噪声内波动, "
             "掉超 eps 才拒(不能为一点提升让平均明显回落). 仅 --pool_filter 时生效.")
    args = ap.parse_args()

    # 算子H 总开关落定: --strengthen 同时激活 classify_failure_type 后移 + 饱和分支 strengthen.
    # 置模块级 _STRENGTHEN_ON 让 classify_failure_type (早于 main 调用栈) 看到.
    global _STRENGTHEN_ON
    if getattr(args, "strengthen", False) and getattr(args, "expand", False):
        _STRENGTHEN_ON = True

    if args.self_test_facet:
        _self_test_facet()
        return

    # 机制消融配置 (leave-one-out; 无 flag = FULL, 与主结果路径完全一致)
    K_eff = 1 if args.ablate_gepa else args.K
    _abl = {"gepa": args.ablate_gepa, "failure": args.ablate_failure,
            "consolidate": args.ablate_consolidate, "antideg": args.ablate_antideg}
    ablation_label = "FULL" if not any(_abl.values()) else "ablate_" + "+".join(k for k, v in _abl.items() if v)
    print(f"🧪 ablation={ablation_label} | K_eff={K_eff} | failure_driven={not args.ablate_failure} | "
          f"consolidate={not args.ablate_consolidate} | antideg={not args.ablate_antideg}")
    _flags = []
    if args.select_by_ci: _flags.append("A1:ci_select")
    if args.fix_exploration: _flags.append("C1:fix_explore")
    if args.cache_failures: _flags.append("D1:cache_fail")
    if args.allow_modify: _flags.append("dir4:allow_modify")
    if args.learn_positive: _flags.append("dir5:learn_positive")
    if args.meta_contrast: _flags.append("路径B:meta_contrast")
    if args.gate_consolidate: _flags.append("M3:gate_consol")
    if args.data_driven_focus: _flags.append("Task3:data_driven_focus")
    if args.coverage_diagnosis: _flags.append("Task4:coverage_diag")
    if args.structure_evolution: _flags.append(f"PhaseC:struct_evo Kc={args.Kc} struct_temp={args.struct_temp}")
    if args.probe_split: _flags.append(f"Task5:probe_split ratio={args.probe_ratio} seed={args.probe_seed}")
    if args.select_exclude_train: _flags.append("药三:select_exclude_train")
    if args.select_lcb: _flags.append("药一:select_lcb (Beta-Binomial 25%后验分位替换 CI 下界)")
    if args.pop_mutation: _flags.append(f"方向E:pop_mutation eps={args.pop_eps} (archive种群采样突变起点)")
    if args.donor_topk and args.donor_topk > 1:
        _flags.append(f"break1:donor_topk={args.donor_topk}")
    if not args.subset_admit:
        _flags.append("break3:no_subset_admit")
    if args.xo_pair_retries or args.xo_cohere_retries or args.xo_allow_merge:
        _flags.append(
            f"靶点1:xo_pair_retries={args.xo_pair_retries} "
            f"xo_cohere_retries={args.xo_cohere_retries} "
            f"xo_allow_merge={args.xo_allow_merge}")
    if args.strengthen:
        _flags.append("算子H:strengthen (饱和分支加固现有段 + 分类器meta后移)")
    if getattr(args, "strengthen_per_type", False):
        _flags.append("方案1:strengthen_per_type (已覆盖仍失败类型亦加固)")
    if args.pool_filter:
        _flags.append(f"fix-①:pool_filter (质量关+5峰尖峰关+git-diff多样性关, eps={args.pool_eps})")
    if _flags:
        print(f"🔧 opt-flags ON: {' | '.join(_flags)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # --resume: 从 prior run 目录恢复, 跳过 baseline 重测, 续跑 (治进程中途死亡丢进度, 见
    # docs/phase2_issues.md 2026-08-06 条). prior log 必须含 iters; 末轮 skill_iterN.md = resume 起点.
    resume_from = 0
    prior = None
    skill_path = None
    if args.resume:
        prior = json.load(open(os.path.join(args.resume, "evolver_log.json"), encoding="utf-8"))
        prior_iters = prior.get("iters", [])
        resume_from = len(prior_iters)
        if resume_from == 0:
            print(f"⚠️ --resume: prior log 无 iters, 退化为 fresh run (重测 baseline)")
            prior = None
        else:
            last_t = prior_iters[-1]["t"]
            # 【Bug B 修复】facet-tree 路径写 best_skill_iter{t}.md (含复合 overlay/crossover best);
            # 旧路径写 skill_iter{t}.md (单轴). 优先取复合 best, 回落单轴, 再回落终局.
            skill_path = os.path.join(args.resume, f"best_skill_iter{last_t}.md")
            if not os.path.exists(skill_path):
                skill_path = os.path.join(args.resume, f"skill_iter{last_t}.md")
            if not os.path.exists(skill_path):
                skill_path = os.path.join(args.resume, "best_skill_evolved.md")
            print(f"🔄 --resume: 从 {args.resume} 恢复 (prior {resume_from} 轮, 末轮 iter{last_t} "
                  f"delta={prior_iters[-1]['best_val_delta']:+.3f}), 续跑 {args.iters} 轮 "
                  f"-> iter{resume_from+1}..{resume_from+args.iters}")

    # 起点 skill: resume 时取 prior 末轮 skill_iterN.md (= prior best), 否则 --init_skill
    if prior is not None:
        init_skill = open(skill_path, encoding="utf-8").read().strip()
    else:
        init_skill = open(args.init_skill, encoding="utf-8").read().strip()
    s0 = open(args.s0, encoding="utf-8").read().strip()
    val_items = load_split(args.split_dir, "val", shuffle_seed=args.shuffle_seed)
    train_items = load_split(args.split_dir, "train", shuffle_seed=args.shuffle_seed)
    n_train_actual = len(train_items)  # C11: 切片前实际 train 题数 (供 --train_n 失效告警)
    # 选择集 = train+val, 免训练优化无过拟合权重, test 作终极裁判.
    # 用 args.train_n 控制 train 用量 (默认全用).
    train_items = train_items[: args.train_n]
    select_items = train_items + val_items

    # Phase B 防泄漏护栏 (F06/§1.3): 起 skill 与 S₀ 不得直接含 train 题面 n-gram 或选项位置暗示.
    # 演化期 composed 候选的检查在各自 compose 处做 (facet-tree 路径分散, 暂不强制门控, 见 Phase C);
    # 此处至少守住"起手就泄漏"的低级情况 (如旧 on/off run 的 init_skill 经 review 怀疑带 val 定理).
    # 检查对象 = train_items (rewriter 可见面). fail 仅警告不阻断 (起 skill 是外部给定, 非本工具产出).
    _leak_ok, _leak_reasons = check_skill_leakage(init_skill, train_items)
    if not _leak_ok:
        print(f"⚠ init_skill 防泄漏检查未通过: {'; '.join(_leak_reasons)} | "
              f"非本工具产出, 仅警告不阻断; 演化期候选若同样泄漏将被拒绝.")
    _s0_ok, _s0_reasons = check_skill_leakage(s0, train_items)
    if not _s0_ok:
        print(f"⚠ S₀ 防泄漏检查未通过: {'; '.join(_s0_reasons)} | "
              f"S₀ 为冻结基线, 仅警告不阻断.")
    print(f"🔒 防泄漏预检: init_skill={'OK' if _leak_ok else 'WARN'} S₀={'OK' if _s0_ok else 'WARN'} "
          f"(检查面=train {len(train_items)} 题)", flush=True)

    # Task5: 诊断-选择解耦. flag 关时 diag=probe=select_items (字节复现既有 0.462 路径);
    # 开时按固定 seed shuffle 后不相交切分: diag_items 喂 collect_failures (patcher 诊断),
    # probe_items 喂 eval_on (baseline/candidate/gate 选 winner). 切断同源过拟合链.
    # 确定性: select_items 每次 train_n 重建, 固定 seed shuffle → 切分逐次一致 (resume 安全).
    # s0_cache 零额外算: diag∪probe=select_items ⊆ s0_cache (71 keys 超集), 下方覆盖校验走全集不变.
    if args.probe_split:
        import random as _probe_rng
        _idx = list(range(len(select_items)))
        _probe_rng.Random(args.probe_seed).shuffle(_idx)
        _np = round(len(select_items) * args.probe_ratio)
        probe_items = [select_items[i] for i in _idx[:_np]]
        diag_items = [select_items[i] for i in _idx[_np:]]
        print(f"🔬 probe_split: probe={len(probe_items)} diag={len(diag_items)} "
              f"(seed={args.probe_seed}, ratio={args.probe_ratio}) | select={len(select_items)}")
    else:
        probe_items = select_items
        diag_items = select_items

    # 药三 (2026-09-11): 候选选择/晋升计分集剔除 train 重叠. 根因③ (data leak): select=train+val=71,
    # train 18 题从头到尾在 eval_on 的 probe_items 里 (candidate/baseline/gate 全落这之上) = 在"背过答案的题"上选冠军.
    # 开 flag: probe_items 改成纯 val 53 (选择/晋升信号不再含 train), diag_items 仍看全 71 (诊断为 collect_failures 多看失败无害).
    # 与 probe_split 正交: probe_split 在 71 内 shuffle 切 probe/diag; 本 flag 把 probe 固定到 val 上.
    if args.select_exclude_train:
        probe_items = val_items
        print(f"💊 药三 select_exclude_train: probe(计分/晋升)→纯 val {len(probe_items)} 题, "
              f"diag(诊断)→仍 {len(diag_items)} 题 (train 已退出选择集)")

    if args.s0_cache and os.path.exists(args.s0_cache):
        # F07: cache 带 manifest, 按 k/shuffle_seed/skill_hash/choice_arrangement_hash 校验口径.
        # 旧 json.load 只看文件存在 → 换 k/换 seed/换 S₀ 仍静默复用旧 cache, 污染 delta.
        # manifest 不匹配 (或旧 cache 无 manifest) → 返回 None, 现场重算 select set 全集.
        s0_cache = load_s0_cache_with_manifest(
            args.s0_cache, s0, args.eval_k, args.shuffle_seed,
            select_items, args.split_dir, "train+val")
        if s0_cache is None:
            print(f"⏳ precomputing S₀ on select set (k={args.eval_k}, EVAL_TEMP={EVAL_TEMP}, "
                  f"shuffle_seed={args.shuffle_seed}) ...", flush=True)
            s0_cache = precompute_s0(select_items, s0, args.eval_k,
                                     shuffle_seed=args.shuffle_seed,
                                     split_dir=args.split_dir, split_name="train+val")
            json.dump(s0_cache, open(os.path.join(out_dir, "s0_cache.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        else:
            # manifest 已校验 k/seed/口径一致, 但仍需覆盖 select set 题 (cache 可能是子集).
            missing = [str(it["id"]) for it in select_items if str(it["id"]) not in s0_cache]
            if missing:
                print(f"⚠️ s0_cache 缺 {len(missing)} 题, 补算 ...")
                extra = precompute_s0([it for it in select_items if str(it["id"]) in missing], s0,
                                      args.eval_k, shuffle_seed=args.shuffle_seed,
                                      split_dir=args.split_dir, split_name="train+val(partial)")
                s0_cache.update(extra)
                json.dump(s0_cache, open(os.path.join(out_dir, "s0_cache.json"), "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
    else:
        print(f"⏳ precomputing S₀ on select set (k={args.eval_k}, EVAL_TEMP={EVAL_TEMP}, "
              f"shuffle_seed={args.shuffle_seed}) ...", flush=True)
        s0_cache = precompute_s0(select_items, s0, args.eval_k,
                                 shuffle_seed=args.shuffle_seed,
                                 split_dir=args.split_dir, split_name="train+val")
        json.dump(s0_cache, open(os.path.join(out_dir, "s0_cache.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print(f"📄 init_skill: {len(init_skill)} chars | select(train+val): {len(select_items)} | "
          f"S₀ cache: {len(s0_cache) - 1} 题 (含 _manifest) | eval_k={args.eval_k} | shuffle_seed={args.shuffle_seed}")

    # 基线: init_skill 在选择集上的表现 (与 candidate 同 eval_k 同条件, delta 对位 s0_cache 固定快照)
    # --resume 跳过 baseline 重测 (省 ~7min k=5), 用 prior 记录值; base_per 不持久化, resume 后不再用.
    if prior is not None:
        bv = prior["baseline_val"]
        base_pass, base_delta, base_ci = bv["pass"], bv["delta"], bv["ci"]
        print(f"📊 [resume] baseline 跳过 (prior recorded): pass={base_pass:.3f} delta={base_delta:+.3f} CI={base_ci:+.3f}")
    else:
        base_pass, base_delta, _, base_per = eval_on(probe_items, init_skill, s0_cache,
                                                     stage="evolver_base", k=args.eval_k,
                                                     eval_deadline_s=EVAL_DEADLINE_S)
        base_ci = bootstrap_ci_lower(base_per)
        print(f"📊 init skill select: pass={base_pass:.3f} delta={base_delta:+.3f} CI_lower={base_ci:+.3f}")

    best_skill = init_skill
    if prior is not None:
        # 恢复 prior 末轮状态: best_skill 已 = prior 末轮 skill_iter; delta/ci/no_improve 从 log 取.
        pi = prior["iters"][-1]
        best_val_delta = pi["best_val_delta"]
        best_val_ci = pi["best_val_ci"]
        no_improve = pi.get("no_improve", 0)
        rejected_patches = list(prior.get("rejected_patches", []))
        failure_cache = {}
        for h, v in prior.get("failure_cache", {}).items():
            # JSON list -> tuple. Task4: coverage 模式缓存 4 元 (含 all_failures), 按实际长度还原.
            failure_cache[h] = tuple(v)
    else:
        best_val_delta = base_delta
        best_val_ci = base_ci
        rejected_patches = []
        failure_cache = {}  # D1: {skill_hash: (failures, n_fail)}, skill 未变时复用
        no_improve = 0
    log = {"ts": ts, "init_skill": args.init_skill, "init_skill_hash": md5(init_skill),
           "ablation": ablation_label,
           "baseline_val": {"pass": base_pass, "delta": base_delta, "ci": base_ci},
           "iters": [], "args": vars(args),
           # Task5: 诊断-选择解耦溯源 (flag 关时 n_select=probe=diag, 0.462 复现路径)
           "probe_split": args.probe_split, "probe_n": len(probe_items),
           "diag_n": len(diag_items), "n_select": len(select_items)}
    if prior is not None:
        # 继承 prior iters 保完整轨迹; init_skill 记原始 seed (prior 的, 非 resume 点); 标 resume 来源.
        log["iters"] = list(prior["iters"])
        log["init_skill"] = prior.get("init_skill", args.init_skill)
        log["init_skill_hash"] = prior.get("init_skill_hash", md5(init_skill))
        log["resumed_from"] = {"path": args.resume, "n_prior_iters": resume_from}

    total_iters = resume_from + args.iters

    # ── Facet-Tree (Stage 0): 树遍历路径 (gated by --facet_tree), 替换单 lineage 主循环 ──
    # 关 flag = 字节复现既有 0.462 路径 (下方 for t 不动)。开 = 树遍历 + facet 分解/交叉 + edit 护栏。
    # C11: 无效参数告警 (flag, 非硬退 — 保既有 run 脚本零回归; review C11 接受 error/flag).
    _param_warnings = []
    if args.facet_tree and args.K != 4:
        _param_warnings.append(
            f"--K {args.K} 在 --facet_tree 路径下不生效 (候选数由 --n_parallel 控制); 已忽略")
    if args.train_n > n_train_actual:
        _param_warnings.append(
            f"--train_n {args.train_n} > 实际 train 题数 {n_train_actual}; slice 无效 (已用全部 train)")
    if _param_warnings:
        print("⚠️ C11 — 参数告警:", file=sys.stderr)
        for _w in _param_warnings:
            print(f"   - {_w}", file=sys.stderr)
    log["param_warnings"] = _param_warnings
    if args.facet_tree:
        best_skill, best_val_delta, best_val_ci = _run_facet_tree(
            args, init_skill, s0_cache, probe_items, diag_items,
            base_pass, base_delta, base_ci, out_dir, log,
            init_best_delta=best_val_delta if prior is not None else None,
            init_best_ci=best_val_ci if prior is not None else None)
        with open(os.path.join(out_dir, "best_skill_evolved.md"), "w", encoding="utf-8") as f:
            f.write(best_skill)
        json.dump(log, open(os.path.join(out_dir, "evolver_log.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
        final_hash = md5(best_skill)
        init_hash = md5(init_skill)
        print(f"\n{'='*50}\n✅ Facet-Tree done. init val delta={base_delta:+.3f} → "
              f"best val delta={best_val_delta:+.3f} (CI={best_val_ci:+.3f})")
        print(f"   skill: {len(init_skill)} → {len(best_skill)} chars | hash {init_hash} → {final_hash}")
        print(f"   产出: {out_dir}/best_skill_evolved.md + tree_iter*.json")
        print(f"   评测 test=106: cd /data/zhuldz/lunwen && python rl/train/verl1/eval_test_multiseed.py "
              f"--skills s0,evolver --evolver_best {out_dir}/best_skill_evolved.md --eval_k 3 --seeds 2")
        # RF §6 Step 3: 终局 0-promoted 告警 — 开启的算子全程 0 次晋升 best 就告警 (写到 stderr + log)
        _op_alerts = _zero_promoted_alert(log.get("operator_counts", {}), args)
        log["zero_promoted_alerts"] = _op_alerts
        if _op_alerts:
            print("⚠️ RF §6 Step 3 — operator(s) active but 0 promoted all run:", file=sys.stderr)
            for _a in _op_alerts:
                print(f"   - {_a}", file=sys.stderr)
        return

    for t in range(resume_from + 1, total_iters + 1):
        t0 = time.time()
        # focus 选择: 默认 round-robin (保主结果复现). --data_driven_focus / --coverage_diagnosis
        # 时延迟到 collect_failures 之后, 按已诊断失败的类型选 (Task3 多数 / Task4 未覆盖类型).
        # A4 消融固定为通用 focus.
        focus_deferred = (not args.ablate_antideg) and (args.data_driven_focus or args.coverage_diagnosis)
        focus = "General mathematical reasoning improvement" if args.ablate_antideg else (
            None if focus_deferred else FAILURE_FOCI[(t - 1) % len(FAILURE_FOCI)]
        )
        print(f"\n{'='*50}\nIter {t}/{total_iters}"
              + ("" if focus is None else f"  focus={focus}")
              + ("\n  [data_driven_focus: focus 待失败诊断后选定]" if focus_deferred and not args.coverage_diagnosis else "")
              + ("\n  [coverage_diagnosis: focus 待全失败分布诊断后选未覆盖类型]" if args.coverage_diagnosis and not args.ablate_antideg else "")
              + f"\n{'='*50}")

        current_skill = best_skill
        current_hash = md5(current_skill)

        # 收集当前 best 的失败 (k 投票). A2 消融跳过 (不诊断真实错误, 隔离失败驱动机制)
        # Task4: --coverage_diagnosis 时 collect_failures 额外返回全失败 all_failures (零额外 API).
        # 路径B: --meta_contrast 时额外返回 meta_passes (skill 解对的 meta 题, 作对照).
        all_failures = None  # 仅 coverage 模式下非 None
        meta_passes = None   # 仅 meta_contrast 模式下非 None (list)
        if args.ablate_failure:
            failures, n_fail, wins = [], 0, []
            print(f"  [ablate_failure] 跳过 collect_failures (通用 patch, 无诊断)")
        else:
            # D1: 失败集缓存. skill 未变 (hash 同) 则复用上轮 failures+wins, 省 ~355 call/轮.
            # Task4: coverage 模式缓存 4-tuple (含 all_failures). 路径B: 缓存 5-tuple (含 meta_passes).
            if args.cache_failures and current_hash in failure_cache:
                cached = failure_cache[current_hash]
                failures, n_fail, wins = cached[0], cached[1], cached[2]
                if len(cached) > 3:
                    all_failures = cached[3]
                if getattr(args, "meta_contrast", False) and len(cached) > 4:
                    meta_passes = cached[4]
                print(f"  best skill val failures: {n_fail} (diagnosing {len(failures)})"
                      + (f" | full distribution: {len(all_failures)}" if all_failures is not None else "")
                      + " [CACHED, skill 未变]"
                      + (f" | wins: {len(wins)}" if args.learn_positive else ""))
            else:
                if args.coverage_diagnosis:
                    if getattr(args, "meta_contrast", False):
                        failures, n_fail, wins, all_failures, meta_passes = collect_failures(
                            diag_items, best_skill, s0_cache,
                            n_diag=args.n_diag, k=args.eval_k, return_all_failures=True,
                            meta_contrast=True)
                    else:
                        failures, n_fail, wins, all_failures = collect_failures(
                            diag_items, best_skill, s0_cache,
                            n_diag=args.n_diag, k=args.eval_k, return_all_failures=True)
                    failure_cache[current_hash] = (failures, n_fail, wins, all_failures,
                                                   meta_passes) if meta_passes is not None else (failures, n_fail, wins, all_failures)
                    print(f"  best skill val failures: {n_fail} (diagnosing {len(failures)} | full distribution: {len(all_failures)})"
                          + (f" | wins: {len(wins)} (hard Qs skill solves, S₀ fails)" if args.learn_positive else "")
                          + (f" | meta_passes: {len(meta_passes)} (对照)" if meta_passes is not None else ""))
                else:
                    if getattr(args, "meta_contrast", False):
                        failures, n_fail, wins, all_failures, meta_passes = collect_failures(
                            diag_items, best_skill, s0_cache,
                            n_diag=args.n_diag, k=args.eval_k,
                            return_all_failures=True, meta_contrast=True)
                    else:
                        failures, n_fail, wins = collect_failures(diag_items, best_skill, s0_cache,
                                                                  n_diag=args.n_diag, k=args.eval_k)
                    failure_cache[current_hash] = (failures, n_fail, wins, all_failures,
                                                   meta_passes) if meta_passes is not None else (failures, n_fail, wins)
                    print(f"  best skill val failures: {n_fail} (diagnosing {len(failures)})"
                          + (f" | wins: {len(wins)} (hard Qs skill solves, S₀ fails)" if args.learn_positive else "")
                          + (f" | meta_passes: {len(meta_passes)} (对照)" if meta_passes is not None else ""))

        # data-driven focus override (Task3): 收集到失败后, 按多数失败类型选 focus.
        # 根因: collect_failures focus 无关, val 最难失败 ~93% 是 meta-option; 原固定
        # round-robin 在早 iter 用错方向 patch. 这里用已诊断失败的真实类型分布定向 patcher.
        # Task4: --coverage_diagnosis 模式由下方 coverage 块独占 focus (supersede), 此块跳过避免冗余.
        if focus_deferred and not args.coverage_diagnosis:
            if failures:
                counts = {}
                for f in failures:
                    lbl = classify_failure_type(f)
                    counts[lbl] = counts.get(lbl, 0) + 1
                # 多数类型; 平票按 FAILURE_FOCI 顺序 (后者优先级低, 保持稳定确定性).
                focus = max(counts, key=lambda lbl: (counts[lbl], -FAILURE_FOCI.index(lbl)))
                top_lbl, top_n = focus, counts[focus]
                print(f"  [data_driven_focus] failure-type tally: "
                      + " | ".join(f"{l[:28]}={n}" for l, n in sorted(counts.items(), key=lambda kv: -kv[1]))
                      + f"  -> pick: {focus} ({top_n}/{len(failures)})")
            else:
                focus = FAILURE_FOCI[(t - 1) % len(FAILURE_FOCI)]  # 无失败则回退 round-robin
                print(f"  [data_driven_focus] no failures to classify, fallback round-robin: {focus}")

        # coverage-driven focus override (Task4): 在全失败分布上找当前 skill 尚未覆盖的类型,
        # 选最高频未覆盖类型填洞 (非加固多数). 全已覆盖则回退多数类型 (Task3 行为).
        # 根因: Task3 仅看 val 最难 8 题过拟合 val 分布; Task4 复用已算全失败给 patcher 看全分布.
        cov_counts, cov_reps, cov_uncovered, coverage_bundle = None, None, None, None
        if args.coverage_diagnosis and not args.ablate_antideg and all_failures:
            cov_counts, cov_reps = compute_failure_distribution(all_failures)
            cov_uncovered = find_uncovered_type(cov_counts, current_skill)
            if cov_uncovered:
                focus = cov_uncovered
                focus_src = "uncovered (gap-fill)"
            elif cov_counts:
                focus = max(cov_counts, key=lambda lbl: (cov_counts[lbl], -FAILURE_FOCI.index(lbl)))
                focus_src = "majority (all covered, fallback)"
            else:
                focus = FAILURE_FOCI[(t - 1) % len(FAILURE_FOCI)]
                focus_src = "round-robin (no failures)"
            # 给 patcher 的覆盖包: reps/counts/uncovered (generate_candidates 渲染 tally+分层样本)
            coverage_bundle = (cov_reps, cov_counts, cov_uncovered)
            # 标注每类型 covered/uncovered 状态打印
            sc_lc = current_skill.lower()
            status = " | ".join(
                f"{l[:24]}={cov_counts[l]}{'✗' if (any(s in sc_lc for s in _TYPE_COVERAGE_SIG.get(l, ()))) else '✓'}"
                for l in sorted(cov_counts, key=lambda x: -cov_counts[x]))
            print(f"  [coverage_diagnosis] full failure-type distribution ({len(all_failures)} failures): "
                  + status + f"  -> pick: {focus} [{focus_src}]"
                  + ("" if cov_uncovered else "  ⚠️ no uncovered gap, fallback to majority"))
        elif args.coverage_diagnosis and not args.ablate_antideg and not all_failures:
            print(f"  [coverage_diagnosis] no all_failures (ablate_failure?), focus stays round-robin/data-driven")

        # 探索模式: 连续无提升则加大力度. C1 修复 (原 temps 参数从未使用, 探索是 no-op):
        # 现通过 explore 标志传给 generate_candidates, 切到 bolder angle hints 池.
        # A4 消融禁用探索; --fix_exploration 控制 C1 修复开关 (默认关 = 沿用原 no-op 行为以保主结果).
        explore = (not args.ablate_antideg) and no_improve >= args.patience
        if explore:
            print(f"  ⚠️ no_improve={no_improve}, EXPLORATION mode (bolder angle hints)"
                  + ("" if args.fix_exploration else " [C1 未启用, 仍原行为]"))
        # 生成 K 候选. 保底 current_skill 不进候选竞争 (只作下界), 避免原地踏步被误判改进.
        # A1 消融 K_eff=1; A2 消融 failure_driven=False; A4 消融不喂 rejected 反面教材
        rejected_feed = "" if args.ablate_antideg else "; ".join(rejected_patches[-3:])
        cands = generate_candidates(best_skill, failures, focus, rejected_feed,
                                    K_eff, explore=(explore and args.fix_exploration),
                                    failure_driven=not args.ablate_failure,
                                    allow_modify=args.allow_modify,
                                    wins=wins, learn_positive=args.learn_positive,
                                    coverage_bundle=coverage_bundle,
                                    meta_contrast=((all_failures if (all_failures is not None and getattr(args, "meta_contrast", False)) else failures), meta_passes) if meta_passes is not None else None)
        # mutation_kinds: 每个候选的变异来源. --structure_evolution 关时恒 ["patch"]*K (字节复现既有路径);
        # 开时 patch 候选标 "patch", 结构候选标 "structure" (内容重组 v2 的决策程序).
        # Phase C: 结构候选与 patch 候选同台竞争 pass_rate, 让数据决哪种变异有效 (研究主张: 结构 > 内容为瓶颈).
        mutation_kinds = ["patch"] * len(cands)
        n_novel = sum(1 for c in cands if md5(c) != current_hash)
        print(f"  generated {len(cands)} candidates, {n_novel} novel (hash != current)")

        if args.structure_evolution:
            # 生成 Kc 个结构变体候选 (内容重组 v2 skill, 不注入新内容; 失败上下文复用同一份, 零额外 API).
            # 结构候选的 angle 旋转用 iter 索引 t, 跨轮覆盖不同组织假设 (确定性可复现).
            struct_cands = generate_structure_candidates(
                best_skill, failures=failures, focus=focus,
                coverage_bundle=coverage_bundle, rejected=rejected_feed,
                Kc=args.Kc, struct_temp=args.struct_temp, iter_idx=t,
                allow_modify=args.allow_modify, wins=wins,
                learn_positive=args.learn_positive)
            struct_novel = sum(1 for c in struct_cands if md5(c) != current_hash)
            cands = cands + struct_cands
            mutation_kinds = mutation_kinds + ["structure"] * len(struct_cands)
            n_novel = sum(1 for c in cands if md5(c) != current_hash)
            print(f"  + {len(struct_cands)} structure candidates, {struct_novel} novel "
                  f"(struct_temp={args.struct_temp}) | total {len(cands)} cands, {n_novel} novel "
                  f"| mutation_kinds={mutation_kinds}")

        # 评测所有候选 (k 投票, 与 baseline 同条件)
        scored = []
        for ci_idx, cand in enumerate(cands):
            cp, cd, _, citem = eval_on(probe_items, cand, s0_cache,
                                       stage="evolver_val", k=args.eval_k,
                                       eval_deadline_s=EVAL_DEADLINE_S)
            cl = lcb_score_from_deltas(citem, s0_cache, args.eval_k)[0] if args.select_lcb else bootstrap_ci_lower(citem)
            kind = mutation_kinds[ci_idx] if args.structure_evolution else ""
            scored.append({"idx": ci_idx, "skill": cand, "hash": md5(cand),
                           "kind": kind,
                           "pass": cp, "delta": cd, "ci": cl, "len": len(cand)})
            # kind_tag 空 (flag off) 时此行与既有字节一致; 开时加 [structure]/[patch] 标注.
            kind_tag = f" [{kind}]" if kind else ""
            print(f"  cand{ci_idx}{kind_tag}: delta={cd:+.3f} CI={cl:+.3f} pass={cp:.3f} len={len(cand)} "
                  f"novel={md5(cand) != current_hash}", flush=True)

        # winner 选择: 仅在 novel (hash != current) 且 ci>0 的候选里选 delta 最高.
        # 这是产物身份校验: winner 必须是真正改过的 skill, 否则保底下界保持 current.
        novel_positive = [s for s in scored if s["hash"] != current_hash and s["ci"] > 0]
        reject_reason = None
        # A1: --select_by_ci 时按 CI 下界选优 (治 val-overfit; max-of-K 原始 delta 有向上偏,
        # CI 下界已扣除采样方差, 选最稳而非最走运). 否则沿用 argmax delta (保主结果复现).
        sel_key = (lambda s: s["ci"]) if args.select_by_ci else (lambda s: s["delta"])
        if novel_positive:
            winner = max(novel_positive, key=sel_key)
        else:
            # 无 novel+正CI 候选: 看 novel 候选里 delta 最高的, 记 reject_reason 但不接受
            novel_all = [s for s in scored if s["hash"] != current_hash]
            if novel_all:
                winner = max(novel_all, key=sel_key)
                reject_reason = (f"no novel candidate with CI>0 "
                                 f"(best novel delta={winner['delta']:+.3f} CI={winner['ci']:+.3f})")
            else:
                winner = None
                reject_reason = "no novel candidate generated (all == current)"

        # winner_kind (前向兼容字段): winner 的变异来源. --structure_evolution 关时为 "patch" (既有路径);
        # 开时为 "patch" 或 "structure". 用于诊断研究主张: 若 structure 占非平凡比例 = 结构是瓶颈.
        winner_kind = winner.get("kind", "patch") if (winner is not None and args.structure_evolution) else "patch"

        # 改进判定: winner 存在 + delta 严格超 best + ci>0 + hash 不同 + LOST 监控
        improved = False
        if winner is not None:
            improved = (winner["delta"] > best_val_delta and winner["ci"] > 0
                        and winner["hash"] != current_hash)
            if not args.ablate_antideg and winner["pass"] < base_pass - 0.06:
                print(f"  ⚠️ winner pass {winner['pass']:.3f} << base {base_pass:.3f}, REJECT (LOST risk)")
                improved = False
                reject_reason = f"LOST: pass drop to {winner['pass']:.3f}"
                rejected_patches.append(f"iter{t}: caused pass drop to {winner['pass']:.3f}")

        if improved:
            assert md5(winner["skill"]) != current_hash, "accept 需产物已改动"
            best_skill = winner["skill"]
            best_val_delta = winner["delta"]
            best_val_ci = winner["ci"]
            no_improve = 0
            kind_tag = f" [{winner_kind}]" if args.structure_evolution else ""
            print(f"  ✅ ACCEPT iter{t}: delta={winner['delta']:+.3f} CI={winner['ci']:+.3f} "
                  f"hash {current_hash}→{winner['hash']}{kind_tag}")
        else:
            no_improve += 1
            if not args.ablate_antideg and winner is not None and reject_reason and "LOST" not in reject_reason:
                rejected_patches.append(f"iter{t} focus={focus}: {reject_reason}")
            print(f"  ✗ KEEP PREV (no_improve={no_improve}) reason={reject_reason}")

        # 慢更新
        consolidated = False
        if not args.ablate_consolidate and t % args.consolidate_every == 0 and len(best_skill) > 4000:
            before = len(best_skill)
            before_hash = md5(best_skill)
            cand_skill = consolidate(best_skill)
            if md5(cand_skill) == before_hash:
                # clean_skill 回退原值, consolidate 实未改动
                consolidated = False
                print(f"  🔧 consolidated: no change (clean_skill reverted to original)")
            elif args.gate_consolidate:
                # M3-gate: re-eval 压缩后 skill, delta 下降超 GATE_MARGIN 则回退 (治盲压缩损质)
                gp, gd, _, gitem = eval_on(probe_items, cand_skill, s0_cache,
                                           stage="evolver_consol_gate", k=args.eval_k,
                                           eval_deadline_s=EVAL_DEADLINE_S)
                gcl = lcb_score_from_deltas(gitem, s0_cache, args.eval_k)[0] if args.select_lcb else bootstrap_ci_lower(gitem)
                if gd >= best_val_delta - GATE_MARGIN:
                    best_skill = cand_skill
                    consolidated = True
                    print(f"  🔧 consolidated: {before} -> {len(best_skill)} chars GATE-ACCEPT "
                          f"delta {best_val_delta:+.3f}->{gd:+.3f} (CI={gcl:+.3f}) hash {before_hash}->{md5(best_skill)}")
                else:
                    consolidated = False
                    print(f"  🔧 consolidated: REVERT (delta {best_val_delta:+.3f}->{gd:+.3f} 降超容差 {GATE_MARGIN}, "
                          f"保原 skill {before} chars) hash {before_hash}")
            else:
                best_skill = cand_skill
                consolidated = True
                print(f"  🔧 consolidated: {before} -> {len(best_skill)} chars hash {before_hash}->{md5(best_skill)}")

        # 快照
        with open(os.path.join(out_dir, f"skill_iter{t}.md"), "w", encoding="utf-8") as f:
            f.write(best_skill)
        log["iters"].append({
            "t": t, "secs": round(time.time() - t0, 1), "focus": focus,
            "n_val_failures": n_fail, "n_candidates": len(cands),
            "n_novel_candidates": n_novel, "current_hash": current_hash,
            "winner_hash": winner["hash"] if winner else None,
            "winner_kind": winner_kind,  # Phase C: 前向兼容字段, structure 占比 = 结构是瓶颈的证据
            "mutation_kinds": mutation_kinds,  # Phase C: 本轮候选变异来源分布 (诊断 mode collapse)
            "n_structure_candidates": mutation_kinds.count("structure") if args.structure_evolution else 0,
            "n_novel_structure": sum(1 for k, c in zip(mutation_kinds, cands)
                                    if k == "structure" and md5(c) != current_hash) if args.structure_evolution else 0,
            "reject_reason": reject_reason,
            "best_val_delta": round(best_val_delta, 4), "best_val_ci": round(best_val_ci, 4),
            "improved": improved, "no_improve": no_improve, "consolidated": consolidated,
            "skill_len": len(best_skill), "skill_hash": md5(best_skill),
            # Task4 (2026-08-17): 全失败类型分布 + 选定的未覆盖类型, C 阶段结构自进化的数据基座.
            "failure_distribution": cov_counts, "uncovered_type": cov_uncovered,
        })
        # 持久化辅助状态供 --resume 无损续传 (rejected_patches + failure_cache).
        # failure_cache value=(failures,n_fail,wins[,all_failures]) tuple -> list for JSON; 含 item dict (题面).
        # Task4: coverage 模式缓存 4-tuple, 这里按实际长度序列化 (非硬切 3), resume 时 len>3 检查可还原 all_failures.
        # 序列化失败 (非预期字段) 则丢弃 failure_cache 重试 -> resume 时重建, 不阻断当轮.
        log["rejected_patches"] = rejected_patches
        log["failure_cache"] = {h: list(v) for h, v in failure_cache.items()}
        _log_fp = open(os.path.join(out_dir, "evolver_log.json"), "w", encoding="utf-8")
        try:
            json.dump(log, _log_fp, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            log["failure_cache"] = {}
            _log_fp.seek(0)
            _log_fp.truncate()
            json.dump(log, _log_fp, ensure_ascii=False, indent=2)
        _log_fp.close()

        # 早停: 连续 patience*2 轮无提升
        if no_improve >= args.patience * 2:
            print(f"  ⏹ early stop: no_improve={no_improve}")
            break

    with open(os.path.join(out_dir, "best_skill_evolved.md"), "w", encoding="utf-8") as f:
        f.write(best_skill)
    json.dump(log, open(os.path.join(out_dir, "evolver_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    final_hash = md5(best_skill)
    init_hash = md5(init_skill)
    changed = final_hash != init_hash
    print(f"\n{'='*50}\n✅ Done. init val delta={base_delta:+.3f} → best val delta={best_val_delta:+.3f} (CI={best_val_ci:+.3f})")
    print(f"   skill: {len(init_skill)} → {len(best_skill)} chars | hash {init_hash} → {final_hash}")
    if not changed:
        print(f"   ⚠️ 产物未改动 (best == init)! 优化器未产出有效 patch, 需检查生成/门控.")
    else:
        print(f"   ✅ 产物已改动 (hash 不同), 可进 test 复评.")
    print(f"   产出: {out_dir}/best_skill_evolved.md")
    print(f"   评测 test=106: cd /data/zhuldz/lunwen && python rl/train/verl1/eval_test_multiseed.py "
          f"--skills s0,evolver --evolver_best {out_dir}/best_skill_evolved.md --eval_k 3 --seeds 2")


if __name__ == "__main__":
    main()
