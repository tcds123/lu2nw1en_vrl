"""
经验蒸馏器 — Phase 1 核心模块 (v3: 混合模式 + 泛化约束)

直接从 GRPO 训练产出的 readable_trace_*.log 中:
  1. 按问题分组, 取每题多个 rollout 的 (提示词, 代码, 得分)
  2. 对比高分/低分 → LLM 蒸馏候选经验 (单题)
  3. 跨题合并去重 → LLM 过滤泛化经验 (批量)
  4. 更新 experience_pool.json

混合模式: 单题提取候选 + 跨题批量合并, 平衡成本与泛化性
"""

import os
import sys
import json
import re
import time
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from experience.templates import (
    SINGLE_ROLLOUT_SUMMARY,
    GROUP_ADVANTAGE_TEMPLATE,
    CROSS_PROBLEM_MERGE_TEMPLATE,
)
from experience.cache import ExperienceCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 默认 trace 日志路径
DEFAULT_TRACE_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout"
DEFAULT_CACHE_PATH = "/data/zhuldz/lunwen/experience/experience_pool.json"


class TraceBasedExtractor:
    """从 GRPO trace 日志中提取经验 (v3: 混合模式)"""

    def __init__(
        self,
        trace_dir: str = DEFAULT_TRACE_DIR,
        cache_path: str = DEFAULT_CACHE_PATH,
        llm_model: str = "/model/DeepSeek-V3.2",
        llm_base_url: str = "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1",
        llm_api_key: str = os.environ.get("DEEPSEEK_API_KEY", ""),
    ):
        self.trace_dir = trace_dir
        self.cache = ExperienceCache(cache_path)
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key

    # ============================================================
    # Trace 日志解析
    # ============================================================
    def find_latest_trace(self) -> Optional[str]:
        """找到最新的 trace 日志文件"""
        if not os.path.isdir(self.trace_dir):
            logger.error(f"Trace directory not found: {self.trace_dir}")
            return None

        today = time.strftime("%Y%m%d")
        candidates = [
            f for f in os.listdir(self.trace_dir)
            if f.startswith(f"readable_trace_{today}") and f.endswith(".log")
        ]
        if not candidates:
            candidates = [
                f for f in os.listdir(self.trace_dir)
                if f.startswith("readable_trace_") and f.endswith(".log")
            ]

        if not candidates:
            logger.error("No trace log files found!")
            return None

        candidates.sort(key=lambda f: os.path.getmtime(os.path.join(self.trace_dir, f)), reverse=True)
        latest = os.path.join(self.trace_dir, candidates[0])
        logger.info(f"Found trace: {latest} ({os.path.getsize(latest)/1024/1024:.1f} MB)")
        return latest

    def parse_trace(self, trace_path: str) -> List[dict]:
        """解析 trace 日志, 提取每条 rollout 的关键信息"""
        rollouts = []

        with open(trace_path, 'r') as f:
            content = f.read()

        blocks = re.split(r'={30,}.*?={30,}', content)
        for block in blocks:
            block = block.strip()
            if not block or not block.startswith('{'):
                continue

            try:
                trace = json.loads(block)
            except json.JSONDecodeError:
                start = block.find('{')
                if start >= 0:
                    try:
                        trace = json.loads(block[start:])
                    except:
                        continue
                else:
                    continue

            entry = self._extract_entry(trace)
            if entry:
                rollouts.append(entry)

        logger.info(f"Parsed {len(rollouts)} rollouts from trace")
        return rollouts

    def _extract_entry(self, trace: dict) -> Optional[dict]:
        """从单条 trace 中提取 ESSENTIAL 信息"""
        try:
            a_model = trace.get("a_model", {})
            system_prompt = a_model.get("output", "")
            problem = a_model.get("input", "")

            group_tests = trace.get("group_tests", [])
            if not group_tests:
                return None

            for test in group_tests:
                code = test.get("generated_code", "")
                pass_rate = test.get("pass_rate", 0.0)
                scores = test.get("scores", {})
                obj_raw = scores.get("obj_raw", 0.0)
                sub_raw = scores.get("sub_raw", 0.0)
                final_score = test.get("final_case_score", 0.0)

                return {
                    "problem": problem,
                    "system_prompt": system_prompt,
                    "code": code,
                    "pass_rate": pass_rate,
                    "obj_score": obj_raw,
                    "sub_score": sub_raw,
                    "total_score": final_score,
                }
        except Exception:
            return None

    # ============================================================
    # 问题分组
    # ============================================================
    def group_by_problem(self, rollouts: List[dict]) -> Dict[str, List[dict]]:
        """将 rollout 按问题分组 (用 A 模型输入中包含的参考解作为唯一标识)"""
        groups = defaultdict(list)
        for r in rollouts:
            problem_text = str(r.get("problem", ""))
            match = re.search(r'Ground Truth.*?```python\s*(.*?)\s*```', problem_text, re.DOTALL)
            if match:
                key = match.group(1)[:200].strip()
            else:
                key = str(r.get("system_prompt", ""))[:200].strip()

            if key:
                groups[key].append(r)

        merged = {}
        small_group = []
        for key, group in groups.items():
            if len(group) >= 2:
                merged[key] = group
            else:
                small_group.extend(group)
        if small_group:
            merged["__misc__"] = small_group

        return merged

    # ============================================================
    # 经验蒸馏
    # ============================================================
    def _summarize_single(self, rollout: dict) -> str:
        """调用 LLM 总结单个 rollout"""
        prompt = SINGLE_ROLLOUT_SUMMARY.format(
            system_prompt=str(rollout.get("system_prompt", ""))[:2000],
            code=str(rollout.get("code", ""))[:2000],
            pass_rate=rollout.get("pass_rate", 0),
            judge_score=rollout.get("sub_score", 0),
            ground_truth="(see problem above)",
            problem=str(rollout.get("problem", ""))[:1000],
        )
        return self._call_llm(prompt, max_tokens=500)

    def _extract_from_group(self, problem_key: str, group: List[dict]) -> List[str]:
        """从同一问题的多个 rollout 中提取候选经验 (单题)"""
        if len(group) < 2:
            return []

        group_sorted = sorted(group, key=lambda r: r.get("total_score", -15), reverse=True)
        scores = [r.get("total_score", -15) for r in group_sorted]

        if max(scores) - min(scores) < 2.0:
            return []

        top_n = min(3, len(group_sorted))
        bottom_n = min(2, len(group_sorted) - top_n)
        selected = group_sorted[:top_n] + group_sorted[-bottom_n:] if bottom_n > 0 else group_sorted[:top_n]

        summaries = []
        for i, r in enumerate(selected):
            summary = self._summarize_single(r)
            summaries.append(f"## Rollout {i+1} (Score: {r['total_score']:.2f})\n{summary}")

        prompt = GROUP_ADVANTAGE_TEMPLATE.format(
            problem=problem_key[:1500],
            num_rollouts=len(group),
            rollout_summaries="\n---\n".join(summaries),
            learning_objective=(
                "Help the prompt generator (A model) create better system prompts "
                "for coding tasks. Focus on specific, actionable instructions."
            ),
            num_experiences=min(3, len(group)),
        )
        response = self._call_llm(prompt, max_tokens=800)

        experiences = []
        match = re.search(r'<Experiences>\s*(.*?)\s*</Experiences>', response, re.DOTALL)
        if match:
            for line in match.group(1).strip().split('\n'):
                line = line.strip()
                if re.match(r'\[G\d+\]\.', line):
                    experiences.append(line)

        return experiences

    # ============================================================
    # 跨题合并去重 (v3 新增 — 混合模式核心)
    # ============================================================
    def _cross_problem_merge(
        self,
        candidate_experiences: List[str],
        num_problems: int,
        max_experiences: int = 15,
    ) -> List[str]:
        """
        将所有单题提取的候选经验一起喂给 LLM，进行跨题泛化过滤。

        Args:
            candidate_experiences: 所有单题提取的候选经验列表
            num_problems: 来源问题数
            max_experiences: 最多保留的经验数

        Returns:
            经过泛化过滤和去重合并后的经验列表
        """
        if not candidate_experiences:
            return []

        # 如果候选太少，直接用本地过滤
        if len(candidate_experiences) <= 5:
            return self._local_generalization_filter(candidate_experiences)

        # 去重（本地 Jaccard + 保留长文本）
        deduped = self._local_dedup(candidate_experiences)
        logger.info(f"Cross-problem merge: {len(candidate_experiences)} candidates → "
                    f"{len(deduped)} after local dedup, from {num_problems} problems")

        # 格式化当前经验池
        current_pool_text = self.cache.format_for_injection(max_experiences=30)
        if not current_pool_text:
            current_pool_text = "(empty pool)"

        # 格式化候选经验
        candidates_text = "\n".join(
            f"[Candidate #{i+1}]. {exp}"
            for i, exp in enumerate(deduped)
        )

        prompt = CROSS_PROBLEM_MERGE_TEMPLATE.format(
            num_problems=num_problems,
            candidate_experiences=candidates_text,
            current_pool=current_pool_text,
            max_experiences=min(max_experiences, len(deduped)),
        )

        response = self._call_llm(prompt, max_tokens=1200)

        # 提取 <Experiences> 标签中的经验
        merged = []
        match = re.search(r'<Experiences>\s*(.*?)\s*</Experiences>', response, re.DOTALL)
        if match:
            for line in match.group(1).strip().split('\n'):
                line = line.strip()
                # 匹配 [G#]. <text> 格式（可能带 // General: ... 注释）
                m = re.match(r'\[G\d+\]\.\s*(.+?)(?:\s*//.*)?$', line)
                if m:
                    guideline = m.group(1).strip()
                    if len(guideline) > 15:  # 太短的不算
                        merged.append(f"[G#]. {guideline}")

        logger.info(f"Cross-problem merge: {len(deduped)} deduped → {len(merged)} validated")
        return merged

    def _local_dedup(self, experiences: List[str], threshold: float = 0.50) -> List[str]:
        """本地语义去重：用多级语义相似度，保留每个 cluster 中 score 最高的"""
        if len(experiences) <= 1:
            return experiences

        keep = []
        for exp in experiences:
            is_dup = False
            for i, kept in enumerate(keep):
                sim = self.cache.semantic_similarity(exp, kept)
                if sim > threshold:
                    # 保留更长的版本（更长通常意味着更具体）
                    if len(exp) > len(kept):
                        keep[i] = exp
                    is_dup = True
                    break
            if not is_dup:
                keep.append(exp)
        return keep

    def _local_generalization_filter(self, experiences: List[str]) -> List[str]:
        """
        本地泛化过滤：用启发式规则过滤明显特化的经验。
        过滤包含以下模式的经验：
        - 引用特定算法名 (merge sort, binary search, BFS, etc.)
        - 引用特定数据结构名 (linked list, heap, trie, etc.)
        - 引用具体数值/阈值
        """
        # 问题特定关键词（出现在经验中说明经验不够泛化）
        ALGORITHM_KEYWORDS = [
            r'\bmerge\s*sort\b', r'\bbinary\s*search\b', r'\bBFS\b', r'\bDFS\b',
            r'\bquick\s*sort\b', r'\bdynamic\s*programming\b', r'\bbacktracking\b',
            r'\bDijkstra\b', r'\bA\*\b', r'\btwo\s*pointer\b', r'\bsliding\s*window\b',
        ]
        DATA_STRUCTURE_KEYWORDS = [
            r'\blinked\s*list\b', r'\bheap\b', r'\b(?:min|max)\s*heap\b',
            r'\btrie\b', r'\b(?:binary|search)\s*tree\b', r'\bgraph\b',
            r'\bstack\b', r'\bqueue\b', r'\bhash\s*map\b', r'\bhashtable\b',
        ]
        SPECIFIC_PATTERNS = [
            r'\bO\([^)]+\)',  # 复杂度引用（可能是特化的）
            r'\binteger\s+array\b',  # 特定数据类型
            r'\bstring\s+input\b',
        ]

        import re as re_module
        filtered = []
        for exp in experiences:
            text = exp.lower()
            is_specific = False

            for pattern in ALGORITHM_KEYWORDS + DATA_STRUCTURE_KEYWORDS:
                if re_module.search(pattern, text):
                    is_specific = True
                    logger.debug(f"Filtered problem-specific experience: {exp[:80]}...")
                    break

            if not is_specific:
                filtered.append(exp)

        if len(filtered) < len(experiences):
            logger.info(f"Local generalization filter: {len(experiences)} → {len(filtered)} "
                        f"(removed {len(experiences) - len(filtered)} problem-specific)")

        return filtered if filtered else experiences  # 至少保留一些

    # ============================================================
    # LLM 调用
    # ============================================================
    def _call_llm(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        import requests
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.llm_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    logger.warning(f"LLM API status {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"LLM attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        return ""

    # ============================================================
    # 主流程 (v3: 混合模式)
    # ============================================================
    def run(
        self,
        trace_path: str = None,
        max_problems: int = 50,
        cross_problem_merge: bool = True,
    ):
        """
        从 trace 日志提取经验 (v3 混合模式)

        Args:
            trace_path: trace 日志路径 (None = 自动找最新)
            max_problems: 最多处理的问题数
            cross_problem_merge: 是否启用跨题合并去重 (默认开启)
        """
        if trace_path is None:
            trace_path = self.find_latest_trace()
        if trace_path is None:
            return []

        # 1. 解析
        logger.info("Parsing trace log...")
        rollouts = self.parse_trace(trace_path)
        if len(rollouts) < 10:
            logger.warning(f"Only {len(rollouts)} rollouts found, need more data")
            return []

        # 2. 分组
        logger.info("Grouping by problem...")
        groups = self.group_by_problem(rollouts)
        logger.info(f"Found {len(groups)} unique problems, "
                    f"avg {len(rollouts)/max(1,len(groups)):.1f} rollouts/problem")

        # 3. 筛选有对比信号的问题
        viable = []
        for key, group in groups.items():
            if len(group) < 2:
                continue
            scores = [r.get("total_score", -15) for r in group]
            if max(scores) - min(scores) >= 2.0 and max(scores) > 0:
                viable.append((key, group))

        viable.sort(key=lambda x: max(r["total_score"] for r in x[1]) - min(r["total_score"] for r in x[1]),
                    reverse=True)
        viable = viable[:max_problems]

        logger.info(f"Selected {len(viable)}/{len(groups)} problems with good contrast")

        # 4. 单题蒸馏 → 收集候选经验
        all_candidates = []
        batch_num = 0
        for key, group in viable:
            try:
                exps = self._extract_from_group(key, group)
                if exps:
                    all_candidates.extend(exps)
                    logger.info(f"  +{len(exps)} candidates from: {key[:80]}...")
            except Exception as e:
                logger.error(f"Failed on problem: {e}")

        logger.info(f"\nPhase 1 done: {len(all_candidates)} candidate experiences from {len(viable)} problems")

        if not all_candidates:
            logger.info("No candidate experiences extracted.")
            return []

        # 5. 跨题合并去重 (v3 新增)
        if cross_problem_merge and len(all_candidates) > 3:
            logger.info("\nPhase 2: Cross-problem merge & generalization filter...")
            validated = self._cross_problem_merge(
                all_candidates,
                num_problems=len(viable),
                max_experiences=15,
            )
        else:
            # 不做跨题合并时，用本地过滤兜底
            validated = self._local_generalization_filter(all_candidates)
            validated = self._local_dedup(validated)

        # 6. 写入经验池
        all_experiences = []
        for exp_text in validated:
            is_new = True
            for existing in self.cache.all:
                if self.cache.semantic_similarity(exp_text, existing.clean_text) > 0.55:
                    is_new = False
                    break
            if is_new:
                self.cache.add(exp_text, source_batch=batch_num)
                all_experiences.append(exp_text)

        logger.info(f"\n✅ Done! {len(all_experiences)} new experiences added to pool")
        logger.info(f"   Candidates: {len(all_candidates)} → Validated: {len(validated)} → New: {len(all_experiences)}")
        logger.info(f"Pool: {self.cache.count} total | {self.cache.stats()}")
        return all_experiences

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return 0.0
        return len(a_words & b_words) / len(a_words | b_words)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract prompt engineering experiences from GRPO trace logs (v3)")
    parser.add_argument("--trace_path", type=str, default=None,
                        help="Path to readable_trace log (auto-detect latest if not specified)")
    parser.add_argument("--max_problems", type=int, default=50,
                        help="Max problems to process")
    parser.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH,
                        help="Experience pool JSON path")
    parser.add_argument("--no_cross_merge", action="store_true",
                        help="Disable cross-problem merge (use local filter only)")
    args = parser.parse_args()

    extractor = TraceBasedExtractor(
        trace_dir=DEFAULT_TRACE_DIR,
        cache_path=args.cache_path,
    )
    experiences = extractor.run(
        trace_path=args.trace_path,
        max_problems=args.max_problems,
        cross_problem_merge=not args.no_cross_merge,
    )

    if experiences:
        print(f"\n📋 Extracted {len(experiences)} new experiences:")
        for e in experiences:
            print(f"  {e}")
    else:
        print("\n⚠️  No new experiences extracted (need more rollout data or stronger contrast)")
