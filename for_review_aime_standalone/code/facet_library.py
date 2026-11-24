"""
Emergent Facet Library (EFL) — 可涌现的失败类型分类法.

替换 evolver 里手写的固定 8 维 FAILURE_FOCI + 纯关键词路由器. 给 evolver skillopt 的好半边
(语义 LLM 发现), 保留 evolver 自己的好半边 (LCB gate, 在 evolver 主文件里逐字保留).

两级涌现:
  Level 1 (分类涌现, 本模块核心): 语义 LLM 分类器读 failure trace (problem+pred+correct+excerpt),
    ASSIGN 到已有 facet 或 EMERGE 新 facet. facet 跨轮累积; LLM 配对合并; 稳定性 gate 提升
    candidate→stable→actionable.
  Level 2 (skill 涌现, evolver 主文件耦合点): actionable facet 喂扁平路径 focus 选择 + facet_tree
    expand target (长出 `## Geometry` 段). skill-section 机制 label-agnostic, 零下游改动.

设计纪律 (镜像 experience/cache.py 持久化 + experience/updater.py JSON 决策模式):
  - Facet dataclass 镜像 Experience (cache.py:19); FacetLibrary 镜像 ExperienceCache (cache.py:42).
  - FACET_CLASSIFY_TEMPLATE 镜像 DECISION_TEMPLATE (updater.py:55); parse_facet_decision 镜像
    parse_decision (updater.py:116). 严格 JSON, 容错解析, 失败回退关键词路由.
  - LLM 后端由调用方注入 (llm_fn callable) — 复用 evolver 的 llm() (主文件:131), 不重复硬编码 key.
    None 时降级纯关键词路由 (facet 自带 keyword_signature), 可在无 key 环境跑 self-test.

零回归: evolver 主文件里 `--emergent_facets` 默认 OFF; 关 = _EFL=None, 原路径逐字节执行.
"""

import os
import re
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# facet 状态机: candidate (刚涌现, 1 次) → stable (跨题复现) → actionable (有 keyword 签名, 可喂 focus/expand).
STATUS_CANDIDATE = "candidate"
STATUS_STABLE = "stable"
STATUS_ACTIONABLE = "actionable"


@dataclass
class Facet:
    """单条 facet (失败类型). 镜像 Experience (cache.py:19).

    name = 分类器返回的 label (与 FAILURE_FOCI 字符串同地位 — 既作 dict key 又作段名锚).
    is_meta = 结构维 flag (替换原 FAILURE_FOCI[5] 位置硬编码): True = 答案结构特征维 (非数学内容),
        expand 显式跳过 (交由 facet_edit 轴). 首批 seed 只有 foci[5] answer-form 是 meta.
    keyword_signature = 题面 surface 词元组 (e.g. ("triangle","angle","circle")), 供覆盖检查
        (find_uncovered_type) 与无 LLM 回退路由用. 必须是会出现在题面/段名里的词.
    """
    id: str                       # F0, F1, ...
    name: str                     # label (dict key / 段名锚)
    description: str = ""         # 一句话: 该 facet 抓什么数学失败
    keyword_signature: Tuple[str, ...] = field(default_factory=tuple)
    is_meta: bool = False         # 结构维 flag (替换 FAILURE_FOCI[5] 位置硬编码)
    example_failures: List[str] = field(default_factory=list)   # ≤3 个短样本 (problem 片段)
    failure_count: int = 0        # 累计被分类到此 facet 的失败数
    problems_seen: List[str] = field(default_factory=list)      # 累计见过的 problem id (跨题复现判定)
    status: str = STATUS_ACTIONABLE   # candidate/stable/actionable
    is_behavior: bool = False     # 行为 facet (轨迹病态, 跨题型; keyword_signature = 哨兵, 恒 UNCOVERED)
    is_budget: bool = False       # 预算 facet (输出被 token 上限截断; 见 budget_signal, issue #53 追加7)
    ate_score: Optional[float] = None    # 可选: 离线 ATE 因果效应 (scorer.py), 不进涌现 gate
    created_at: float = 0.0
    promoted_at: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON 无 tuple → list; 读回时 from_dict 转回 tuple (签名要 hashable/可 in 检查)
        d["keyword_signature"] = list(self.keyword_signature)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Facet":
        d = dict(d)
        d["keyword_signature"] = tuple(d.get("keyword_signature") or [])
        d.setdefault("example_failures", [])
        d.setdefault("problems_seen", [])
        d.setdefault("status", STATUS_ACTIONABLE)
        d.setdefault("is_meta", False)
        d.setdefault("is_behavior", False)
        d.setdefault("is_budget", False)
        d.setdefault("ate_score", None)
        d.setdefault("promoted_at", None)
        return cls(**d)


# ================================================================
# LLM 决策模板 (镜像 updater.py DECISION_TEMPLATE)
# ================================================================
FACET_CLASSIFY_SYSTEM = (
    "You are a failure-taxonomy curator for a mathematical-reasoning skill optimizer (AIME-style "
    "competition problems, integer answers 0-999). You read a SINGLE failed problem trace and decide "
    "which FAILURE FACET it belongs to. A facet = a recurring mathematical failure mode (e.g. "
    "off-by-one boundary, case-enumeration gap, geometry setup). You maintain a growing library; new "
    "facets EMERGE only when a failure's mathematical content is genuinely distinct from all existing "
    "facets."
)

FACET_CLASSIFY_TEMPLATE = """## Current Facet Library (id: name — description [keywords])
{library_compact}

## Failed Problem Trace
Problem: {problem}
Correct answer: {correct}
Model predicted: {pred}
Trace behavior: {behavior}
Model reasoning excerpt:
{excerpt}

## Task
Decide which facet this failure belongs to. Respond in STRICT JSON only (no markdown fences):
{{
  "operation": "ASSIGN" | "EMERGE",
  "target_id": "<existing facet id for ASSIGN, or null>",
  "new_name": "<short facet name for EMERGE, or null>",
  "new_description": "<one sentence for EMERGE, or null>",
  "new_keywords": ["<problem-text surface term>", "..."]
}}

Rules:
- ASSIGN when the failure fits an existing facet (even partially). Prefer ASSIGN over EMERGE — be conservative.
- EMERGE only when the failure's mathematical content is genuinely NEW (none of the existing facets' descriptions or keywords cover it). Example: a geometry problem (triangle/angle/circle/polygon) when no geometry facet exists yet.
- For EMERGE, new_name must be SHORT (1-3 words, e.g. "Geometry Setup"), reflecting the failure mode.
- For EMERGE, new_keywords MUST be problem-text surface terms that would appear in the problem statement of similar failures (e.g. "triangle", "angle", "circumradius", "hexagon") — these power coverage checking. Give 2-5 terms.
- Do NOT emerge a facet for generic "computation slip" or "arithmetic error" — those are already covered.
- If Trace behavior is NOT "clean" (no final answer emitted / same block repeated / repeated final answers), the failure is a PROCESS pathology. Name it as its own facet (EMERGE, e.g. "Mid-trace Loop") rather than blaming a math topic — unless a facet for that pathology already exists (then ASSIGN to it). Leave new_keywords empty for such facets."""


FACET_MERGE_SYSTEM = (
    "You consolidate a growing failure-facet library. Some candidate facets (recently emerged) may be "
    "semantically DUPLICATES of existing facets or of each other. You decide which candidates should MERGE "
    "into an existing facet (consolidate, avoid taxonomy bloat)."
)

FACET_MERGE_TEMPLATE = """## Existing Facets (stable/actionable)
{existing_compact}

## Candidate Facets (recently emerged, status=candidate)
{candidates_compact}

## Task
For each candidate, decide if it is semantically a DUPLICATE of an existing facet (same failure mode under a different name). Respond in STRICT JSON only (no markdown fences):
{{
  "merges": [
    {{"candidate_id": "<F-id>", "merge_into_id": "<existing F-id or null if keep>"}}
  ]
}}

Rules:
- Merge only when the candidate and target describe the SAME mathematical failure mode (e.g. "Triangle Geometry" into "Geometry Setup").
- Keep a candidate (merge_into_id=null) when it is genuinely distinct.
- Be conservative: prefer KEEP over MERGE to preserve genuinely new facets."""


# ================================================================
# 解析 (镜像 updater.py parse_decision, updater.py:116)
# ================================================================
# ================================================================
# 行为归因 (2026-11-12, issue #53 追加4): 轨迹病态检测 + 行为 facet 命名
# ================================================================
# 原则: 这里只做**测量** (轨迹里有没有交卷 / 有没有原地打转 / 有没有兜圈子), 不写任何
# "该怎么解题" 的规则. 规则从 facet 库涌现: 病态 → 行为 facet → focus → 专属段.
BEHAVIOR_SIGNATURE = ("__behavior__",)   # 哨兵: 永不命中题面 → find_uncovered_type 恒判 UNCOVERED
BUDGET_FACET_NAME = "Budget Exhausted"   # 预算信号 facet 名 (确定性, 不交给 LLM 命名)
HARNESS_FAILURE_LABEL = "Harness Failure (API error / eval abandoned, not model behavior)"
_BEH_NAME_BUDGET = 1                     # 每轮最多 1 次 LLM 命名 (其余失败并入已命名 facet, 控成本)
_BEHAVIOR_NOISE = {"steps", "step", "submitted", "submission", "tags", "tag", "chars", "char",
                   "times", "kept", "second", "guessing", "progress"}
_ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
_REPEAT_WINDOW = 120
_REPEAT_MIN = 5


def _repeat_count(text: str, blk: str) -> int:
    return text.count(blk) if blk else 0


def find_tail_loop(text: str, window: int = _REPEAT_WINDOW, min_reps: int = _REPEAT_MIN) -> int:
    """结尾复读: 末尾 window 字符的块在全文重复出现次数 (>=min_reps 才算). 无则 0."""
    t = (text or "").strip()
    if len(t) < window * min_reps:
        return 0
    blk = t[-window:]
    n = t.count(blk)
    return n if n >= min_reps else 0


def find_cycle(text: str, s_len: int = 160, min_reps: int = 3) -> str:
    """全文循环: 字符滑窗统计长度为 s_len 的子串出现次数 (不受换行丢失影响), 取最频繁者并右扩到
    "所有出现处下一个字符都相同" 的最长块. 返回重复块 (未达 min_reps 返回 "")."""
    t = text or ""
    if len(t) < s_len * min_reps:
        return ""
    counts: Dict[str, int] = {}
    for i in range(0, len(t) - s_len + 1):
        piece = t[i:i + s_len]
        counts[piece] = counts.get(piece, 0) + 1
    blk, n_blk = max(counts.items(), key=lambda kv: kv[1])
    if n_blk < min_reps:
        return ""
    while True:
        offs = [m.start() for m in re.finditer(re.escape(blk), t)]
        nxt = {t[o + len(blk)] for o in offs if o + len(blk) < len(t)}
        if len(nxt) == 1 and len(offs) == n_blk:  # 扩展后出现次数不变才扩 (否则块被切碎)
            blk += nxt.pop()
        else:
            break
    return blk


def behavior_signature(text: str) -> dict:
    """一条轨迹的"过程行为"特征 (零 API). 字段: n_chars / n_steps / n_answer_tags /
    answer_pos (首个 <answer> 相对位置) / tail_loop / cycle_reps / cycle_block."""
    t = text or ""
    tags = _ANSWER_TAG_RE.findall(t)
    steps = [int(x) for x in re.findall(r"Step\s+(\d+)", t)]
    cyc = find_cycle(t)
    return {
        "n_chars": len(t),
        "n_steps": max(steps) if steps else 0,
        "n_answer_tags": len(tags),
        "answer_pos": round(t.find(tags[0]) / len(t), 2) if tags and t else 1.0,
        "tail_loop": find_tail_loop(t),
        "cycle_reps": _repeat_count(t, cyc) if cyc else -1,
        "cycle_block": cyc,
    }


def budget_signal(failure: dict) -> str:
    """预算/完成度信号 (2026-11-12, issue #53 追加7): 该失败是否由 token 预算耗尽导致 (finish_reason=length).
    与题库无关 — 换模型/换数据集只要"输出超出预算"就是这一类. 返回一句英文描述, 干净则 "".
    harness 字段由 run_one 写入 (`{"finish_reason","n_truncated","max_tokens_used","n_out_tokens",...}`)."""
    h = failure.get("harness") or {}
    if not isinstance(h, dict) or int(h.get("n_truncated") or 0) < 1:
        return ""
    return (f"token budget exhausted: {int(h.get('n_truncated') or 0)}/{int(h.get('n_samples') or 0)} "
            f"samples hit the {int(h.get('max_tokens_used') or 0)}-token limit (finish_reason=length, "
            f"output ~{int(h.get('n_out_tokens') or 0)} tokens)")


def is_budget_failure(failure: dict) -> bool:
    return bool(budget_signal(failure))


def harness_signal(failure: dict) -> str:
    """其余 harness 侧失败 (2026-11-12, issue #53 追加7): 采样全为 API 异常 / 整批放弃 (deadline).
    与题库、模型推理都无关 — 换数据集照样出现, 不能被当成数学/行为失败去学. 返回英文描述, 干净则 ""."""
    h = failure.get("harness") or {}
    if not isinstance(h, dict):
        return ""
    n_samp, n_exc = int(h.get("n_samples") or 0), int(h.get("n_exceptions") or 0)
    if n_exc >= 1 and n_exc >= max(1, n_samp):
        return f"harness failure: {n_exc} API exception(s) with no usable sample"
    return ""


def normalize_behavior(failure: dict) -> str:
    """把失败的行为特征归一成一句简短病态描述 (英文, 只描述"观察到了什么"); 轨迹干净返回 "".
    优先用 failure["behavior"] (collect_failures 预计算), 缺失时从 response 现算.
    这是行为 facet 的**信号**, 不是规则: 只回答 "这条轨迹病态在哪", 不回答 "该怎么解".
    """
    sig = failure.get("behavior")
    if not sig:
        resp = failure.get("response") or failure.get("response_excerpt") or ""
        if not resp:
            return ""
        sig = behavior_signature(resp)
    if sig.get("n_answer_tags", 0) == 0:
        return "never emitted <answer> (no final answer submitted" + (
            f", {sig.get('n_steps', 0)} steps)" if sig.get("n_steps") else ")")
    reps = sig.get("cycle_reps", -1)
    if reps >= _REPEAT_MIN:
        return f"mid-trace loop: same {_REPEAT_WINDOW}-char block repeated {reps}x without progress"
    if sig.get("tail_loop", 0) >= _REPEAT_MIN:
        return f"tail repetition: final block repeated {sig['tail_loop']}x"
    if sig.get("n_answer_tags", 0) > 1:
        return f"repeated final answers ({sig['n_answer_tags']} <answer> tags, kept second-guessing)"
    return ""


def _behavior_word_set(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower())
            if w not in {"with", "without", "that", "this", "then", "after", "before", "when",
                         "trace", "same", "char", "block", "repeated", "model", "answer", "final",
                         "emitted", "emits", "progress"}}


def _label_matches(name: str, beh: str) -> bool:
    """facet 名与某次病态描述是否同一类: 名字词集与描述词集交集 >= 1 (名字本身由同源证据命名)."""
    return bool((_behavior_word_set(name) - _BEHAVIOR_NOISE) & (_behavior_word_set(beh) - _BEHAVIOR_NOISE))


def _behavior_label(beh: str) -> str:
    """无 LLM 的机械命名 (仅库为空时用): 取病态描述的前几个实词, Title Case."""
    # 现象 -> 简短名 (仅用于库为空时的引导; 稳态下由 LLM 按证据命名). 只读同现象的其他描述词
    # (never→emitted answer 与 mid-trace loop→trace loop 分别取词), 不会把不同现象混成一个.
    # 已存在的 facet 名也走本函数 (同类判定用), 故同时接受 Title Case 名 (先 lower).
    t = (beh or "").lower()
    words = _behavior_word_set(beh) - _BEHAVIOR_NOISE
    if t.startswith("mid-trace loop") or "loop" in t:
        return "Trace Loop"
    if t.startswith("tail repetition"):
        return "Tail Repetition"
    if "second-guessing" in t:
        return "Repeated Answers"
    if "missing final answer" in t or t.startswith("never") or "no final answer" in t \
            or ("answer" in t and "never" in t):
        return "Missing Final Answer"
    return " ".join(w.capitalize() for w in sorted(words)[:2]) or "Trace Pathology"


def parse_behavior_name(response: str) -> Optional[dict]:
    """解析行为 facet 命名响应 {"name": "...", "description": "..."}; 容错同 parse_facet_decision."""
    if not response:
        return None
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        dec = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    name = str(dec.get("name") or "").strip()
    if not name:
        return None
    return {"name": name, "description": str(dec.get("description") or "")}


BEHAVIOR_NAME_SYSTEM = (
    "You maintain a failure-facet library for a mathematical-reasoning skill optimizer. A model trace "
    "showed a PROCESS pathology (not a mathematical mistake): it never submitted a final answer, "
    "looped on the same block, or kept re-deriving. Name the pathology so that a prompt editor can "
    "write a procedure that prevents it."
)

BEHAVIOR_NAME_TEMPLATE = """## Observed trace behavior
{behavior}

## Failed problem (for context only — the pathology is usually problem-agnostic)
{problem}
Model predicted: {pred}

## Task
Respond in STRICT JSON only:
{{"name": "<SHORT 1-3 word pathology name, e.g. 'Mid-trace Loop', 'Missing Final Answer'>",
  "description": "<one sentence: what the model keeps doing instead of committing>"}}"""


def parse_facet_decision(response: str) -> Optional[dict]:
    """从 LLM 响应解析 JSON 决策 (容错: 去 fence, 找第一个 {...}, json.loads, 校验 operation)."""
    if not response:
        return None
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        dec = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    op = str(dec.get("operation", "")).upper().strip()
    if op not in ("ASSIGN", "EMERGE"):
        return None
    dec["operation"] = op
    return dec


def parse_merge_decision(response: str) -> Optional[dict]:
    """解析合并决策 ({"merges":[...]}). 容错同 parse_facet_decision."""
    if not response:
        return None
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        dec = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "merges" not in dec:
        return None
    return dec


def _norm_name(name: str) -> str:
    """归一化 facet 名作精确去重 (lowercase + 单空格 + 去标点). 语义合并交 LLM merge."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower())).strip()


class FacetLibrary:
    """可涌现 facet 库. 镜像 ExperienceCache (cache.py:42): JSON 持久化 + CRUD.

    llm_fn: 调用方注入的 LLM callable, 签名 llm(system, user, *, max_tokens, stage) -> str.
        复用 evolver 的 llm() (主文件:131), 不重复硬编码 key. None = 纯关键词回退 (无 LLM, 可跑 self-test).
    """

    def __init__(self, path: str, llm_fn: Optional[Callable] = None):
        self.path = path
        self._llm_fn = llm_fn
        self._facets: Dict[str, Facet] = {}   # id -> Facet
        self._name_index: Dict[str, str] = {}  # norm_name -> id (快速精确去重)
        self._beh_name_calls = 0              # 本轮已用的行为 facet LLM 命名预算
        self._beh_deferred: List[dict] = []   # 本轮预算耗尽时暂存的行为失败, 轮末并入已命名 facet
        self._load()

    # ---- 持久化 (镜像 ExperienceCache._load/_save, 原子写) ----
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._facets = {fid: Facet.from_dict(d) for fid, d in data.items()}
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"EFL load failed ({e}); starting fresh")
                self._facets = {}
        self._rebuild_name_index()
        if self._facets:
            logger.info(f"EFL loaded {len(self._facets)} facets from {self.path}")

    def _save(self):
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({fid: fc.to_dict() for fid, fc in self._facets.items()},
                      f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def _rebuild_name_index(self):
        self._name_index = {_norm_name(fc.name): fid for fid, fc in self._facets.items()}

    def _next_id(self) -> str:
        if not self._facets:
            return "F0"
        # 取现有最大数字序号 +1 (F0..F9, F10..)
        max_n = -1
        for fid in self._facets:
            m = re.match(r"^F(\d+)$", fid)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"F{max_n + 1}"

    # ---- 查询 ----
    @property
    def all(self) -> List[Facet]:
        return sorted(self._facets.values(), key=lambda fc: fc.id)

    @property
    def actionable(self) -> List[Facet]:
        """可喂 focus/expand 的 facet (status=actionable)."""
        return [fc for fc in self.all if fc.status == STATUS_ACTIONABLE]

    @property
    def meta_facets(self) -> List[Facet]:
        return [fc for fc in self.all if fc.is_meta]

    def get_by_name(self, name: str) -> Optional[Facet]:
        fid = self._name_index.get(_norm_name(name))
        return self._facets.get(fid) if fid else None

    def is_meta_label(self, lbl: str) -> bool:
        """核心解耦: 'meta' 从位置索引 5 变成显式 flag. 替换主文件 meta_lbl = FAILURE_FOCI[5]."""
        fc = self.get_by_name(lbl)
        return bool(fc and fc.is_meta)

    def get_signature(self, lbl: str) -> Tuple[str, ...]:
        """替换主文件 _TYPE_COVERAGE_SIG.get(lbl, ()). 无该 facet → ()."""
        fc = self.get_by_name(lbl)
        return fc.keyword_signature if fc else ()

    # ---- 种子 (零回归: 从既有 FAILURE_FOCI 初始化, 无新行为) ----
    def seed_from_foci(self, foci: List[str], sig_table: Dict[str, Tuple[str, ...]],
                       meta_index: int = 5) -> int:
        """从既有 8 维 FAILURE_FOCI 初始化 F0-F7. 一次性, 仅在空库时跑.

        meta_index=5 → foci[5] (Answer-form conversion) is_meta=True, 其余 False.
        这是 5 这个魔数唯一存在处 (从 legacy 结构一次性导入); 此后 is_meta 是真 flag, 下游用
        is_meta_label() 查 flag 不查位置. 全部 status=actionable (冷启动即可用).
        返回 seeded 数 (已非空则跳过返回 0).
        """
        if self._facets:
            return 0
        now = time.time()
        for i, name in enumerate(foci):
            fid = f"F{i}"
            self._facets[fid] = Facet(
                id=fid, name=name,
                description=name,  # foci 字符串自带描述
                keyword_signature=tuple(sig_table.get(name, ())),
                is_meta=(i == meta_index),
                status=STATUS_ACTIONABLE,
                created_at=now,
            )
        self._rebuild_name_index()
        self._save()
        return len(foci)

    # ---- 记录失败到 facet (累积统计) ----
    def _record(self, fc: Facet, failure: dict):
        fc.failure_count += 1
        qid = str(failure.get("id", ""))
        if qid and qid not in fc.problems_seen:
            fc.problems_seen.append(qid)
        if len(fc.example_failures) < 3:
            prob = str((failure.get("item", {}) or {}).get("problem", "") or "")
            if prob:
                fc.example_failures.append(prob[:160])

    # ---- 分类 (核心: 语义 LLM 发现, 失败回退关键词) ----
    def classify(self, failure: dict) -> str:
        """对单失败分类, 返回 facet name (与 FAILURE_FOCI 字符串同地位的 label).

        LLM 可用: ASSIGN/EMERGE; EMERGE 时精确去重 (同名归并到既有 candidate, 累积 failure_count).
        LLM 不可用 / 解析失败: 关键词回退 (遍历 actionable facet 的 keyword_signature, 首命中;
            无命中 → meta facet name 作兜底, 镜像原 return FAILURE_FOCI[5]).

        行为归因 (2026-11-12, issue #53 追加4): 失败不一定错在数学, 也可能是 "轨迹病态"
        (从不交卷 / 中途死循环 / 只在极长轨迹末尾交卷). 这类失败**不是新 facet** ——
        它跨所有题型, 与题面无关. 故: normalize_behavior(failure) 非空 (关键词梯+LLM 双路,
        零额外 API) 时 → 先尝试 ASSIGN_BEHAVIOR 到某个行为 facet (库中已有的行为 facet, 或
        经 LLM/规则 crop 从 "这题为何失败" 里抽出的 facet 名, 如 "Mid-trace Loop"); 库为空且
        LLM 可用时 EMERGE 一个 is_behavior=True 的 candidate. 该 facet 的 keyword_signature =
        (__behavior__,) 哨兵 → find_uncovered_type 永远判 UNCOVERED, 从而被 focus 选中,
        行为 facet 自动获得专属段, 并由 patch 候选生成流程写出防护程序. 全程无需人工规定规则.
        """
        bud = budget_signal(failure)
        if bud:
            fc = self._assign_budget(bud, failure)
            if fc is not None:
                return fc.name
        hsig = harness_signal(failure)
        if hsig:
            # harness 故障 (全异常/deadline): 同样是"过程"失败, 不是数学或轨迹病态, 不建 facet、不入库,
            # 以免一条 eval 仪器故障被学成 skill 规则. 单列标签 → 覆盖诊断里可见、可被排除.
            logger.warning(f"HARNESS failure (excluded from facet learning): {hsig} :: {failure.get('id')}")
            return HARNESS_FAILURE_LABEL
        beh = normalize_behavior(failure)
        if beh:
            fc = self._assign_behavior(beh, failure)
            if fc is not None:
                return fc.name
        if self._llm_fn is not None:
            dec = self._classify_via_llm(failure)
            if dec is not None:
                return dec
        return self._classify_keyword(failure)

    # ---- 行为 facet (轨迹病态, 跨题型) ----
    def _assign_behavior(self, beh: str, failure: dict) -> Optional[Facet]:
        """行为 facet 路由: 已有行为 facet 且名字与本次病态同类 (词集相交) → ASSIGN.
        无匹配: 库为空 → 机械命名建 F0; 否则 LLM 命名 (每轮最多 _BEH_NAME_BUDGET 次), 同名归并.
        LLM 不可用且库非空 → 返回 None 走常规分类 (不污染库)."""
        label = _behavior_label(beh)
        for fc in self.all:
            # 名字同类 (词集相交) 或机械名一致 (LLM 起的名可能不带描述用词, 如 "Premature Commit")
            if fc.is_behavior and (_label_matches(fc.name, beh) or _behavior_label(fc.name) == label):
                self._record(fc, failure)
                return fc
        # 库已非空但 LLM 不可用 / 预算耗尽: 不做单向 crop (名字只会有一次样本, 永远不会
        # 达到 candidate→stable 门槛, 白白污染库) → 返回 None, 走常规分类.
        if not self.all:
            name, desc = label, f"model trace pathology: {beh}"
        elif self._llm_fn is not None and self._beh_name_calls < _BEH_NAME_BUDGET:
            self._beh_name_calls += 1
            named = self._name_behavior_via_llm(beh, failure)
            if named is None:
                return None
            name, desc = named
        else:
            self._beh_deferred.append({"beh": beh, "failure": failure})
            return None
        # 先查这个名字是否已存在 (规范化精确匹配) —— 跨轮同名 → 累积到同一 facet, 走稳定性 gate.
        existing = self.get_by_name(name)
        if existing is not None:
            self._record(existing, failure)
            return existing
        fc = Facet(id=self._next_id(), name=name, description=desc[:200],
                   keyword_signature=BEHAVIOR_SIGNATURE, is_behavior=True, is_meta=False,
                   status=STATUS_CANDIDATE, created_at=time.time())
        self._facets[fc.id] = fc
        self._rebuild_name_index()
        self._record(fc, failure)
        logger.info(f"BEHAVIOR facet emerged: {fc.id} {name} :: {beh[:160]}")
        self._save()
        return fc

    def _assign_budget(self, sig: str, failure: dict) -> Optional[Facet]:
        """预算类失败路由 (确定性, 零 LLM 命名): 库中已有预算 facet → ASSIGN; 无 → 建一个
        is_behavior=True 的预算 facet (哨兵签名 → 恒 UNCOVERED, 自动被 focus 选中).
        与 _assign_behavior 的区别: 预算耗尽与题型/推理习惯无关, 换个 max_tokens 或换模型可能会重现/消失,
        不应该被当成数学失败轴, 也不应该被 LLM 命名成"某种推理病态"字符串."""
        for fc in self.all:
            if fc.is_behavior and getattr(fc, "is_budget", False):
                self._record(fc, failure)
                return fc
        fc = Facet(id=self._next_id(), name=BUDGET_FACET_NAME,
                   description="Output exceeded the token budget before completion; raise the generation "
                               "budget and/or force a short, committed final answer instead of endless derivation.",
                   keyword_signature=BEHAVIOR_SIGNATURE, is_behavior=True, is_meta=False,
                   status=STATUS_CANDIDATE, created_at=time.time())
        fc.is_budget = True
        self._facets[fc.id] = fc
        self._rebuild_name_index()
        self._record(fc, failure)
        logger.info(f"BUDGET facet emerged: {fc.id} {fc.name} :: {sig[:160]}")
        self._save()
        return fc

    def drain_deferred(self) -> int:
        """轮末收口: 本轮预算耗尽而暂存的行为失败, 并入已存在且同类的行为 facet (累积计数).
        仍无同类 → 丢弃本轮该失败的行为归因 (走常规分类), 保证名字至少有跨轮证据再建 facet."""
        n = 0
        pending, self._beh_deferred = self._beh_deferred, []
        for item in pending:
            for fc in self.all:
                if fc.is_behavior and (_label_matches(fc.name, item["beh"])
                                       or _behavior_label(fc.name) == _behavior_label(item["beh"])):
                    self._record(fc, item["failure"])
                    n += 1
                    break
        if n:
            self._save()
        return n

    def reset_round_budget(self) -> None:
        """新一轮: 重置 LLM 命名预算 (跨轮可以再为"新出现"的病态命名一次)."""
        self._beh_name_calls = 0

    def _name_behavior_via_llm(self, beh: str, failure: dict) -> Optional[Tuple[str, str]]:
        try:
            user = BEHAVIOR_NAME_TEMPLATE.format(
                behavior=beh,
                problem=str((failure.get("item") or {}).get("problem", ""))[:400],
                pred=str(failure.get("pred", ""))[:80],
            )
            resp = self._llm_fn(BEHAVIOR_NAME_SYSTEM, user, max_tokens=256, stage="efl_behavior")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"EFL behavior-name LLM error: {e!r}")
            return None
        dec = parse_behavior_name(resp)
        if dec is None:
            return None
        return dec["name"][:60], dec.get("description") or beh

    def _classify_via_llm(self, failure: dict) -> Optional[str]:
        it = failure.get("item", {}) or {}
        lib_compact = self._library_compact()
        user = FACET_CLASSIFY_TEMPLATE.format(
            library_compact=lib_compact,
            problem=str(it.get("problem", "") or "(n/a)")[:500],
            correct=str(failure.get("correct", ""))[:100],
            pred=str(failure.get("pred", ""))[:100],
            behavior=normalize_behavior(failure) or "clean",
            excerpt=str(failure.get("response_excerpt", ""))[:600],
        )
        try:
            resp = self._llm_fn(FACET_CLASSIFY_SYSTEM, user, max_tokens=512, stage="efl_classify")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"EFL classify LLM error: {e!r}; keyword fallback")
            return None
        dec = parse_facet_decision(resp)
        if dec is None:
            return None
        if dec["operation"] == "ASSIGN":
            fid = dec.get("target_id")
            fc = self._facets.get(fid) if fid else None
            if fc is None:
                return None  # 无效 target_id → 回退
            self._record(fc, failure)
            return fc.name
        # EMERGE: 精确去重 (同名已存在 → 归并, 累积 failure_count; 否则建新 candidate)
        new_name = (dec.get("new_name") or "").strip()
        if not new_name:
            return None
        existing = self.get_by_name(new_name)
        if existing is not None:
            self._record(existing, failure)
            return existing.name
        fid = self._next_id()
        kws = [str(k).strip().lower() for k in (dec.get("new_keywords") or []) if str(k).strip()]
        # 行为 facet: LLM 自己起的行为名 (clear 轨迹病态 + 没给题面关键词) → 哨兵签名, 恒 UNCOVERED.
        is_beh = bool(normalize_behavior(failure)) and not kws
        if is_beh:
            kws = list(BEHAVIOR_SIGNATURE)
        fc = Facet(
            id=fid, name=new_name,
            description=str(dec.get("new_description") or new_name)[:200],
            keyword_signature=tuple(kws),
            is_behavior=is_beh,
            is_meta=False,
            status=STATUS_CANDIDATE,
            created_at=time.time(),
        )
        self._record(fc, failure)
        self._facets[fid] = fc
        self._name_index[_norm_name(new_name)] = fid
        return fc.name

    def _classify_keyword(self, failure: dict) -> str:
        """无 LLM 回退: 遍历 actionable facet signature 首命中; 无命中 → meta facet 兜底.
        镜像原 classify_failure_type 的 catch-all 行为 (return FAILURE_FOCI[5])."""
        it = failure.get("item", {}) or {}
        q = (str(it.get("problem", "")) + " " + str(it.get("answer", ""))).lower()
        for fc in self.actionable:
            if fc.keyword_signature and any(kw in q for kw in fc.keyword_signature):
                self._record(fc, failure)
                return fc.name
        # 兜底: meta facet (answer-form 结构维); 无 meta 则取首个 actionable
        metas = self.meta_facets
        if metas:
            self._record(metas[0], failure)
            return metas[0].name
        if self.actionable:
            self._record(self.actionable[0], failure)
            return self.actionable[0].name
        return "(unclassified)"

    def classify_all(self, failures: List[dict]) -> Dict[str, int]:
        """批量分类, 返回 counts {name: n}. 侧效: 累积每 facet 的 failure_count/problems_seen."""
        counts: Dict[str, int] = {}
        for f in failures:
            lbl = self.classify(f)
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    # ---- 合并 (LLM 配对语义去重, 每轮一次) ----
    def _library_compact(self, facets: Optional[List[Facet]] = None) -> str:
        facets = facets if facets is not None else self.all
        if not facets:
            return "(empty library)"
        lines = []
        for fc in facets:
            kws = ",".join(fc.keyword_signature) if fc.keyword_signature else "-"
            lines.append(f"{fc.id}: {fc.name} — {fc.description[:80]} [kw: {kws}] (n={fc.failure_count})")
        return "\n".join(lines)

    def merge_candidates(self):
        """LLM 配对合并 candidate facet (语义去重). 一次/轮 (classify 后).

        合并: 把 candidate 的 failure_count/problems_seen/examples 并入 target, 删 candidate.
        只合并 status=candidate 的 (actionable/stable 不动, 保稳定)."""
        if self._llm_fn is None:
            return
        cands = [fc for fc in self.all if fc.status == STATUS_CANDIDATE]
        if not cands:
            return
        existing = [fc for fc in self.all if fc.status != STATUS_CANDIDATE]
        user = FACET_MERGE_TEMPLATE.format(
            existing_compact=self._library_compact(existing),
            candidates_compact=self._library_compact(cands),
        )
        try:
            resp = self._llm_fn(FACET_MERGE_SYSTEM, user, max_tokens=1024, stage="efl_merge")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"EFL merge LLM error: {e!r}")
            return
        dec = parse_merge_decision(resp)
        if not dec:
            return
        for m in dec.get("merges", []):
            cid = m.get("candidate_id")
            tid = m.get("merge_into_id")
            if not cid or not tid or cid not in self._facets or tid not in self._facets:
                continue
            cand = self._facets[cid]
            tgt = self._facets[tid]
            tgt.failure_count += cand.failure_count
            tgt.problems_seen = list(dict.fromkeys(tgt.problems_seen + cand.problems_seen))
            tgt.example_failures = (tgt.example_failures + cand.example_failures)[:3]
            # 合并 keyword_signature (去重保序)
            merged_kw = list(dict.fromkeys(list(tgt.keyword_signature) + list(cand.keyword_signature)))
            tgt.keyword_signature = tuple(merged_kw)
            del self._facets[cid]
        self._rebuild_name_index()

    # ---- 稳定性 gate (candidate→stable→actionable) ----
    def promote_candidates(self, k_min: int = 2, p_min: int = 2):
        """稳定性 gate (约束3: 算法自动涌现, 非人注入):

        candidate→stable: failure_count ≥ k_min AND len(problems_seen) ≥ p_min (跨题复现, 非单题偶发).
        stable→actionable: keyword_signature 非空 (有题面锚词, 可覆盖检查/喂 focus/expand).
        无 keyword 的 stable facet 不喂 expand (expand 靠 signature 探覆盖), 但保留观察."""
        now = time.time()
        for fc in self._facets.values():
            if fc.status == STATUS_CANDIDATE:
                if fc.failure_count >= k_min and len(fc.problems_seen) >= p_min:
                    fc.status = STATUS_STABLE
                    fc.promoted_at = now
            if fc.status == STATUS_STABLE:
                if fc.keyword_signature:
                    fc.status = STATUS_ACTIONABLE

    def persist(self):
        self._save()


# ================================================================
# self-test (零 API, 可被 evolver --self_test_efl 触发)
# ================================================================
def self_test():
    """纯函数 + mock-LLM 自检. 验证 seed/EMERGE/累积/promote/meta-flag/roundtrip."""
    import tempfile
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "facet_library_test.json")

    # 8 维 seed (镜像 evolver FAILURE_FOCI)
    foci = [f"FOCUS_{i}" for i in range(8)]
    sig = {foci[i]: (f"kw{i}a", f"kw{i}b") for i in range(8)}
    sig[foci[5]] = ("remainder", "modulo", "fraction")  # meta 维

    # mock LLM: 第 1 次几何失败 → EMERGE Geometry; 后续 → ASSIGN Geometry
    calls = {"n": 0}

    def mock_llm(system, user, *, max_tokens=4096, stage="evolver"):
        calls["n"] += 1
        if "EMERGE" in user or "geometry" in user.lower() or "triangle" in user.lower():
            if calls["n"] <= 2:
                return '{"operation":"EMERGE","target_id":null,"new_name":"Geometry Setup","new_description":"Geometric configuration / angle / shape failures","new_keywords":["triangle","angle","circle","polygon"]}'
        return '{"operation":"ASSIGN","target_id":"F8","new_name":null,"new_description":null,"new_keywords":[]}'

    lib = FacetLibrary(path, llm_fn=mock_llm)
    n = lib.seed_from_foci(foci, sig, meta_index=5)
    assert n == 8, f"seed expected 8, got {n}"
    assert len(lib.all) == 8
    assert len(lib.actionable) == 8
    assert len(lib.meta_facets) == 1
    assert lib.is_meta_label(foci[5]) is True
    assert lib.is_meta_label(foci[0]) is False
    # signature 查询 (替换 _TYPE_COVERAGE_SIG.get)
    assert lib.get_signature(foci[0]) == ("kw0a", "kw0b")
    assert lib.get_signature("Nonexistent") == ()

    # 几何失败 #1 (题 A) → EMERGE F8 candidate
    f1 = {"id": "A", "item": {"problem": "Triangle ABC has angle ...", "answer": "42"},
          "pred": "13", "correct": "42", "response_excerpt": "assumed right angle"}
    lbl1 = lib.classify(f1)
    assert lbl1 == "Geometry Setup", f"emerge #1 expected 'Geometry Setup', got {lbl1!r}"
    f8 = lib.get_by_name("Geometry Setup")
    assert f8 is not None and f8.id == "F8"
    assert f8.status == STATUS_CANDIDATE
    assert f8.failure_count == 1
    assert len(f8.problems_seen) == 1

    # 几何失败 #2 (题 B) → ASSIGN F8 (精确去重, 累积 count)
    f2 = {"id": "B", "item": {"problem": "Hexagon with circumradius ...", "answer": "7"},
          "pred": "9", "correct": "7", "response_excerpt": "miscomputed side"}
    calls["n"] = 0  # reset 让 mock 再走 EMERGE 分支测精确去重
    lbl2 = lib.classify(f2)
    assert lbl2 == "Geometry Setup", f"emerge #2 (dedup) expected 'Geometry Setup', got {lbl2!r}"
    f8 = lib.get_by_name("Geometry Setup")
    assert f8.failure_count == 2
    assert len(f8.problems_seen) == 2
    assert f8.problems_seen == ["A", "B"]

    # promote: candidate→stable (k≥2, p≥2), stable→actionable (有 signature)
    lib.merge_candidates()
    lib.promote_candidates(k_min=2, p_min=2)
    f8 = lib.get_by_name("Geometry Setup")
    assert f8.status == STATUS_ACTIONABLE, f"after promote expected actionable, got {f8.status}"
    # 几何 facet 不是 meta
    assert lib.is_meta_label("Geometry Setup") is False

    # 持久化 roundtrip
    lib.persist()
    lib2 = FacetLibrary(path, llm_fn=mock_llm)
    assert len(lib2.all) == 9
    f8b = lib2.get_by_name("Geometry Setup")
    assert f8b is not None and f8b.status == STATUS_ACTIONABLE and f8b.failure_count == 2

    # ---- 预算信号 (issue #53 追加7) ----
    assert budget_signal({"behavior": {}}) == ""
    bf1 = {"item": {"problem": "x", "id": "p1"}, "pred": "", "correct": "1",
           "harness": {"finish_reason": "length", "n_truncated": 1, "n_samples": 1,
                       "max_tokens_used": 16384, "n_out_tokens": 16384}}
    assert is_budget_failure(bf1)
    assert "16384" in budget_signal(bf1)
    bf2 = {"item": {"problem": "y", "id": "p2"}, "pred": "", "correct": "1",
           "harness": {"finish_reason": "stop", "n_truncated": 0, "n_samples": 1, "max_tokens_used": 16384}}
    assert not is_budget_failure(bf2)
    # 预算失败优先于行为归因 (即便行为上也"没交卷"); 且复用同一个预算 facet
    bf1["behavior"] = {"n_answer_tags": 0, "n_steps": 12, "cycle_reps": -1, "tail_loop": 0}
    n_beh_before = sum(1 for fc in lib.all if fc.is_behavior)
    assert lib.classify(bf1) == BUDGET_FACET_NAME
    bud = [fc for fc in lib.all if getattr(fc, "is_budget", False)]
    assert len(bud) == 1 and bud[0].keyword_signature == BEHAVIOR_SIGNATURE and bud[0].is_behavior
    bf3 = dict(bf1, item={"problem": "z", "id": "p3"})
    assert lib.classify(bf3) == BUDGET_FACET_NAME
    assert sum(1 for fc in lib.all if fc.is_behavior) == n_beh_before + 1
    # 持久化 roundtrip: is_budget 标记不丢
    lib._save()
    lib2 = FacetLibrary(path, llm_fn=mock_llm)
    assert any(getattr(fc, "is_budget", False) and fc.name == BUDGET_FACET_NAME for fc in lib2.all)

    # 无 LLM 关键词回退
    lib3 = FacetLibrary(os.path.join(tmp, "kw.json"))
    lib3.seed_from_foci(foci, sig, meta_index=5)
    fb = {"id": "X", "item": {"problem": "find the kw0a value", "answer": "1"},
          "pred": "2", "correct": "1", "response_excerpt": ""}
    assert lib3.classify(fb) == foci[0]
    # 无命中 → meta 兜底
    fb2 = {"id": "Y", "item": {"problem": "mystery problem", "answer": "1"},
           "pred": "2", "correct": "1", "response_excerpt": ""}
    assert lib3.classify(fb2) == foci[5]

    print("✅ EFL self-test passed: seed=8 facets, EMERGE Geometry Setup (F8), "
          "dedup accumulate count=2, promote candidate→actionable, meta flag correct, "
          "persist roundtrip, keyword fallback, budget facet (signal factor).")


if __name__ == "__main__":
    self_test()
