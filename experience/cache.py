"""
经验缓存管理 — 基于 JSON 文件的简单 KV 存储 (v3: 语义去重 + 聚类)
支持 ADD / UPDATE / DELETE 操作，带元数据追踪
"""

import json
import os
import re
import time
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """单条经验"""
    id: str              # G0, G1, ...
    text: str            # 经验内容
    score_delta: float = 0.0     # 因果效应估计 (ATE)
    usage_count: int = 0         # 被注入次数
    source_batch: int = 0        # 来源批次
    created_at: float = 0.0      # 创建时间戳

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Experience":
        return cls(**d)

    @property
    def clean_text(self) -> str:
        """去掉 [G#]. 前缀的纯文本"""
        return re.sub(r'^\[G\d+\]\.\s*', '', self.text).strip()


class ExperienceCache:
    """经验池的 JSON 文件存储 (v3: 带语义聚类)"""

    def __init__(self, cache_path: str = "/data/zhuldz/lunwen/experience/experience_pool.json"):
        self.cache_path = cache_path
        self._experiences: Dict[str, Experience] = {}
        self._load()

    def _load(self):
        """从文件加载经验池"""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
            self._experiences = {
                k: Experience.from_dict(v) for k, v in data.items()
            }
            print(f"📂 Loaded {len(self._experiences)} experiences from {self.cache_path}")
        else:
            self._experiences = {}
            print(f"📂 No existing experience pool, starting fresh")

    def _save(self):
        """保存到文件"""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self._experiences.items()},
                      f, indent=2, ensure_ascii=False)

    @property
    def all(self) -> List[Experience]:
        """所有经验，按 ID 排序"""
        return sorted(self._experiences.values(), key=lambda e: e.id)

    @property
    def active(self) -> List[Experience]:
        """获取活跃经验 (score_delta >= 0)"""
        return [e for e in self.all if e.score_delta >= 0]

    @property
    def count(self) -> int:
        return len(self._experiences)

    def get(self, exp_id: str) -> Optional[Experience]:
        return self._experiences.get(exp_id)

    # ================================================================
    # CRUD 操作
    # ================================================================
    def add(self, text: str, source_batch: int = 0) -> str:
        """添加新经验，自动分配 ID"""
        next_id = f"G{len(self._experiences)}"
        exp = Experience(
            id=next_id,
            text=text,
            source_batch=source_batch,
            created_at=time.time()
        )
        self._experiences[next_id] = exp
        self._save()
        return next_id

    def update(self, exp_id: str, text: str, score_delta: Optional[float] = None):
        """更新已有经验"""
        if exp_id in self._experiences:
            self._experiences[exp_id].text = text
            self._experiences[exp_id].usage_count += 1
            if score_delta is not None:
                self._experiences[exp_id].score_delta = score_delta
            self._save()

    def update_score_ema(self, exp_id: str, score_delta: float, alpha: float = 0.1):
        """用 EMA 更新经验的因果效应估计"""
        if exp_id in self._experiences:
            exp = self._experiences[exp_id]
            old_delta = exp.score_delta
            exp.score_delta = (1 - alpha) * old_delta + alpha * score_delta
            exp.usage_count += 1
            self._save()
            return exp.score_delta
        return None

    def batch_update_scores(self, score_map: Dict[str, float], alpha: float = 0.1):
        """批量更新经验分数"""
        updated = 0
        for exp_id, delta in score_map.items():
            if exp_id in self._experiences:
                self.update_score_ema(exp_id, delta, alpha)
                updated += 1
        if updated > 0:
            self._save()
        return updated

    def delete(self, exp_id: str):
        """删除经验"""
        if exp_id in self._experiences:
            del self._experiences[exp_id]
            self._save()

    def add_or_update_batch(self, experiences: List[dict], source_batch: int = 0):
        """批量操作"""
        for exp in experiences:
            action = exp.get("action", "ADD")
            text = exp.get("text", "")
            target_id = exp.get("target_id")

            if action == "ADD" or target_id is None:
                self.add(text, source_batch)
            elif action == "UPDATE" and target_id in self._experiences:
                self.update(target_id, text)
            elif action == "DELETE":
                self.delete(target_id)

    # ================================================================
    # 语义相似度 (v3: 多级相似度 — unigram + bigram + keyword)
    # ================================================================
    @staticmethod
    def _tokenize(text: str, n: int = 1) -> Set[str]:
        """n-gram 分词 (去除标点，小写)"""
        words = re.findall(r'[a-zA-Z_]\w*', text.lower())
        if n == 1:
            return set(words)
        return set('_'.join(words[i:i+n]) for i in range(len(words) - n + 1))

    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        """提取核心关键词 (名词/动词，长度 > 3 的单词)"""
        words = re.findall(r'[a-zA-Z_]\w*', text.lower())
        # 停用词
        stopwords = {
            'the', 'and', 'for', 'that', 'with', 'this', 'from', 'have', 'which',
            'when', 'will', 'also', 'than', 'then', 'just', 'only', 'over', 'into',
            'such', 'more', 'each', 'both', 'what', 'they', 'been', 'being', 'does',
            'most', 'some', 'other', 'should', 'would', 'could', 'must', 'need',
            'ensure', 'avoid', 'make', 'like', 'well', 'good', 'instead',
        }
        return {w for w in words if len(w) > 3 and w not in stopwords}

    def semantic_similarity(self, a: str, b: str) -> float:
        """
        多级语义相似度: 组合 unigram + bigram + keyword 重叠

        三者的加权组合比单纯的 Jaccard 更能捕捉语义相似性：
        - unigram: 词级别重叠 (基础)
        - bigram: 短语级别重叠 (捕获 "output format" vs "format output")
        - keyword: 核心概念重叠 (排除停用词干扰)
        """
        a_uni = self._tokenize(a, 1)
        b_uni = self._tokenize(b, 1)
        a_bi = self._tokenize(a, 2)
        b_bi = self._tokenize(b, 2)
        a_kw = self._extract_keywords(a)
        b_kw = self._extract_keywords(b)

        def jaccard(x: set, y: set) -> float:
            if not x or not y:
                return 0.0
            return len(x & y) / len(x | y)

        uni_sim = jaccard(a_uni, b_uni)
        bi_sim = jaccard(a_bi, b_bi) if a_bi and b_bi else uni_sim
        kw_sim = jaccard(a_kw, b_kw) if a_kw and b_kw else uni_sim

        # 加权: unigram 30%, bigram 40%, keyword 30%
        return 0.30 * uni_sim + 0.40 * bi_sim + 0.30 * kw_sim

    def find_similar(self, text: str, threshold: float = 0.55) -> List[Tuple[Experience, float]]:
        """找到与给定文本语义相似的经验"""
        results = []
        for exp in self.all:
            sim = self.semantic_similarity(text, exp.clean_text)
            if sim > threshold:
                results.append((exp, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def is_duplicate(self, text: str, threshold: float = 0.60) -> bool:
        """检查是否与已有经验重复"""
        return len(self.find_similar(text, threshold)) > 0

    # ================================================================
    # 语义聚类 (v3 新增)
    # ================================================================
    def cluster_and_dedup(self, similarity_threshold: float = 0.55) -> int:
        """
        对经验池中的所有经验做语义聚类，合并高度相似的。

        使用贪心聚类 (single-pass clustering):
        1. 按 score_delta 降序排列
        2. 对每条经验，如果与已有 cluster 代表的相似度 > threshold，归入该 cluster
        3. 保留每个 cluster 中 score_delta 最高的经验，删除其余

        Returns:
            删除的经验数量
        """
        all_exps = sorted(self.all, key=lambda e: e.score_delta, reverse=True)
        if len(all_exps) < 2:
            return 0

        clusters = []  # List of (representative_id, List[exp_id])
        for exp in all_exps:
            assigned = False
            for rep_id, members in clusters:
                rep = self._experiences[rep_id]
                sim = self.semantic_similarity(exp.clean_text, rep.clean_text)
                if sim > similarity_threshold:
                    members.append(exp.id)
                    assigned = True
                    break
            if not assigned:
                clusters.append((exp.id, [exp.id]))

        # 对每个 cluster，保留 score_delta 最高的，删除其余
        deleted_count = 0
        merge_log = []
        for rep_id, members in clusters:
            if len(members) <= 1:
                continue

            # 找 cluster 内最佳经验
            best_id = max(members, key=lambda eid: self._experiences[eid].score_delta)
            for eid in members:
                if eid != best_id:
                    # 将被合并经验的 usage_count 加到保留经验上
                    self._experiences[best_id].usage_count += self._experiences[eid].usage_count
                    merge_log.append(f"  {eid} → {best_id}: '{self._experiences[eid].text[:60]}...'")
                    del self._experiences[eid]
                    deleted_count += 1

        if deleted_count > 0:
            self._save()
            logger.info(f"Semantic clustering: merged {deleted_count} redundant experiences")
            for log in merge_log:
                logger.info(log)

        return deleted_count

    # ================================================================
    # 注入格式化
    # ================================================================
    def format_for_injection(self, max_experiences: int = 15, max_tokens: int = 3000) -> str:
        """格式化为注入 prompt 的文本 (按综合分数排序, 带 token 预算)"""
        active = sorted(
            [e for e in self._experiences.values() if e.score_delta >= 0],
            key=lambda e: e.score_delta * (1 + e.usage_count * 0.1),
            reverse=True
        )
        # Token budget enforcement at injection time
        selected = []
        total_tokens = 0
        for exp in active:
            est = self._estimate_tokens(exp.text)
            if len(selected) < max_experiences and total_tokens + est <= max_tokens:
                selected.append(exp)
                total_tokens += est
        selected.sort(key=lambda e: e.id)

        lines = []
        for exp in selected:
            lines.append(f"[{exp.id}]. {exp.text}")
        return "\n".join(lines)

    def format_ucb_injection(self, max_experiences: int = 8, c: float = 2.0) -> Tuple[str, List[str]]:
        """UCB-based 经验选择 (多臂老虎机)"""
        import math
        total_usage = sum(e.usage_count for e in self._experiences.values()) + 1

        scored = []
        for exp in self._experiences.values():
            if exp.usage_count > 0:
                ucb = exp.score_delta + c * math.sqrt(math.log(total_usage) / (exp.usage_count + 1))
            else:
                ucb = float('inf')

            scored.append((exp, ucb))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in scored[:max_experiences]]
        selected.sort(key=lambda e: e.id)

        lines = []
        ids = []
        for exp in selected:
            lines.append(f"[{exp.id}]. {exp.text}")
            ids.append(exp.id)
        return "\n".join(lines), ids

    def format_randomized_injection(self, max_experiences: int = 8) -> Tuple[str, List[str]]:
        """随机采样经验注入 (构造自然实验用于因果效应估计)"""
        import random
        all_exps = list(self._experiences.values())
        n = min(max_experiences, len(all_exps))
        selected = random.sample(all_exps, n)
        selected.sort(key=lambda e: e.id)

        lines = []
        ids = []
        for exp in selected:
            lines.append(f"[{exp.id}]. {exp.text}")
            ids.append(exp.id)
        return "\n".join(lines), ids

    # ================================================================
    # 剪枝
    # ================================================================
    def prune_negative(self, min_usage: int = 3, threshold: float = -0.5) -> int:
        """剪枝: 删除因果效应为负且使用次数足够的经验"""
        to_delete = [
            eid for eid, exp in self._experiences.items()
            if exp.score_delta < threshold and exp.usage_count >= min_usage
        ]
        for eid in to_delete:
            del self._experiences[eid]
        if to_delete:
            self._save()
        return len(to_delete)

    def prune_unstable(self, min_cross_problems: int = 3, min_usage: int = 4) -> int:
        """
        剪枝不稳定经验 (v3 新增): 删除使用次数够多但 score_delta 接近 0 的经验
        (说明该经验效果不稳定，不是可靠的泛化规则)
        """
        to_delete = [
            eid for eid, exp in self._experiences.items()
            if (exp.usage_count >= min_usage and
                abs(exp.score_delta) < 0.15 and
                exp.usage_count >= min_cross_problems)
        ]
        for eid in to_delete:
            del self._experiences[eid]
        if to_delete:
            self._save()
            logger.info(f"Pruned {len(to_delete)} unstable (near-zero effect) experiences")
        return len(to_delete)

    # ================================================================
    # 统计
    # ================================================================
    # ================================================================
    # 注意力预算管理 (v3 新增)
    # ================================================================
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估计 token 数 (~4 chars per token for English)"""
        return max(1, len(text) // 4)

    def total_tokens(self) -> int:
        """经验池总 token 估计"""
        return sum(self._estimate_tokens(e.text) for e in self._experiences.values())

    def enforce_budget(
        self,
        max_experiences: int = 30,
        max_tokens: int = 4000,
        min_score: float = 0.0,
    ) -> int:
        """
        注意力预算管理: 硬性限制经验池大小和 token 总量。

        当超过上限时:
        1. 优先删除 score_delta 低的经验
        2. 在同样低分的情况下，优先删除新的（未验证的）经验
        3. 保证语义多样性: 高相似度聚类只保留最佳

        Returns:
            剪枝的数量
        """
        pruned = 0
        all_exps = list(self._experiences.values())

        # 按质量排序: score_delta 降序，同分时 usage_count 高的优先
        all_exps.sort(key=lambda e: (e.score_delta, e.usage_count), reverse=True)

        # Token budget
        kept = []
        total_tokens = 0
        for exp in all_exps:
            est_tokens = self._estimate_tokens(exp.text)
            if (len(kept) < max_experiences and
                total_tokens + est_tokens <= max_tokens and
                exp.score_delta >= min_score):
                kept.append(exp)
                total_tokens += est_tokens
            else:
                pruned += 1

        if pruned > 0:
            kept_ids = {e.id for e in kept}
            to_delete = [eid for eid in self._experiences if eid not in kept_ids]
            for eid in to_delete:
                del self._experiences[eid]
            self._save()
            logger.info(f"Attention budget enforcement: pruned {pruned} experiences "
                        f"(kept {len(kept)}, tokens: {total_tokens}/{max_tokens})")

        return pruned

    def stats(self) -> dict:
        active = [e for e in self._experiences.values() if e.score_delta >= 0]
        return {
            "total": self.count,
            "active": len(active),
            "avg_score_delta": sum(e.score_delta for e in self._experiences.values()) / max(1, self.count),
            "positive_effect": sum(1 for e in self._experiences.values() if e.score_delta > 0.1),
            "negative_effect": sum(1 for e in self._experiences.values() if e.score_delta < -0.1),
            "total_usage": sum(e.usage_count for e in self._experiences.values()),
        }
