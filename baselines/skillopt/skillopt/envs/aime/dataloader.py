"""AIME task dataloader (competition mathematics, numerical answers).

Loads AIME-style problem sets (JSONL, one item per line) from
``/data/zhuldz/lunwen/data/`` and normalises each item to the flat schema the
rollout/evaluator expect:

    {id, question, answer, year}

Schema is numerical-answer (integer 0-999) — no choices, no test code, no
reference solution. ``question`` uses the ``problem`` field. AIME has no
difficulty field, so unlike appsplus there is no difficulty filter; instead
``dataset_glob`` (default ``aime*.jsonl``) selects which year(s) to load.
"""
from __future__ import annotations

import json
import os
from typing import Any

from skillopt.datasets.base import SplitDataLoader


def _load_jsonl_or_json(path: str) -> list:
    """Load a .jsonl (one JSON object per line) or a .json (array) file."""
    if path.lower().endswith(".jsonl"):
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for key in ("data", "items", "test"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array or JSONL in {path}, got {type(raw).__name__}")
    return raw


def _normalize_item(item: dict, row_idx: int) -> dict | None:
    # AIME24 carries string id/year; AIME25 carries int id/year — coerce to str.
    item_id = str(item.get("id") if item.get("id") is not None else row_idx)
    problem = str(item.get("problem") or "").strip()
    answer = item.get("answer")
    if answer is None or not problem:
        return None
    year = item.get("year")
    year = str(year) if year is not None else ""
    return {
        "id": item_id,
        "question": problem,
        "answer": answer,  # kept raw (str or int); evaluator int-parses
        "year": year,
    }


def load_items(data_path: str, dataset_glob: str = "aime*.jsonl") -> list[dict]:
    """Load and normalise AIME items from a .jsonl file or directory.

    ``data_path`` may be a single file or a directory. When it is a directory we
    collect every file matching ``dataset_glob`` (default ``aime*.jsonl``) and
    concatenate — so pointing at the data dir yields AIME24+AIME25 together,
    while pointing at a single file yields one year.
    """
    paths: list[str] = []
    if os.path.isdir(data_path):
        prefix = dataset_glob.rstrip("*")
        candidates = sorted(
            p for p in os.listdir(data_path)
            if p.lower().endswith(".jsonl") and p.lower().startswith(prefix.lower())
        )
        if not candidates:
            raise ValueError(
                f"AIMEDataLoader: no {dataset_glob} found under {data_path}"
            )
        paths = [os.path.join(data_path, c) for c in candidates]
    else:
        paths = [data_path]

    items: list[dict] = []
    for path in paths:
        raw = _load_jsonl_or_json(path)
        for row_idx, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            norm = _normalize_item(item, row_idx=row_idx)
            if norm:
                items.append(norm)
    if not items:
        raise ValueError(f"No valid AIME items loaded from {paths}")
    return items


class AIMEDataLoader(SplitDataLoader):
    """AIME dataloader (ratio split over one or more JSONL files)."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "7:1:2",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        dataset_glob: str = "aime*.jsonl",
        **kwargs,
    ) -> None:
        super().__init__(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )
        self.dataset_glob = dataset_glob

    def load_raw_items(self, data_path: str) -> list[dict]:
        return load_items(data_path, dataset_glob=self.dataset_glob)

    def get_task_types(self) -> list[str]:
        return ["aime"]
