"""MBPP-Pro task dataloader.

Loads `data/CodeEval-Pro/dataset/mbpp_pro.json` (a JSON array of items with
fields id / new_problem / new_solution / test_code) and normalises each item to
the flat schema the rollout/evaluator expect:

    {id, question, test_code, raw_solution}

`question` uses `new_problem` (the CodeEval-Pro augmented prompt, which already
contains the function signature); `test_code` is the assert-based unit tests the
evaluator runs directly. No Docker, no stdin/stdout — pure exec'd asserts.
"""
from __future__ import annotations

import json
import os
from typing import Any

from skillopt.datasets.base import SplitDataLoader


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_item(item: dict, row_idx: int) -> dict | None:
    item_id = str(item.get("id") if item.get("id") is not None else row_idx)
    question = str(item.get("new_problem") or item.get("raw_problem") or "").strip()
    test_code = str(item.get("test_code") or "").strip()
    raw_solution = str(item.get("new_solution") or item.get("raw_solution") or "").strip()

    if not question or not test_code:
        return None

    return {
        "id": item_id,
        "question": question,
        "test_code": test_code,
        "raw_solution": raw_solution,
    }


def load_items(data_path: str, dataset_glob: str = "mbpp_pro*") -> list[dict]:
    """Load and normalise items from a JSON file or directory.

    The directory case accepts a folder containing exactly one file matching
    `dataset_glob` (default `mbpp_pro*`). Pass `humaneval_pro*` etc. to switch
    code benchmark without changing the env.
    """
    if os.path.isdir(data_path):
        prefix = dataset_glob.rstrip("*")
        candidates = sorted(
            p for p in os.listdir(data_path)
            if p.lower().startswith(prefix.lower()) and p.lower().endswith(".json")
        )
        if not candidates:
            raise ValueError(
                f"MBPPProDataLoader: no {dataset_glob}.json found in {data_path}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"MBPPProDataLoader: multiple {dataset_glob}.json in {data_path}: {candidates}"
            )
        data_path = os.path.join(data_path, candidates[0])

    raw = _load_json(data_path)
    if isinstance(raw, dict):
        # tolerate {"data": [...]} wrappers
        for key in ("data", "items", "test"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {data_path}, got {type(raw).__name__}")

    items: list[dict] = []
    for row_idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        norm = _normalize_item(item, row_idx=row_idx)
        if norm:
            items.append(norm)
    if not items:
        raise ValueError(f"No valid MBPP-Pro items loaded from {data_path}")
    return items


class MBPPProDataLoader(SplitDataLoader):
    """MBPP-Pro dataloader (ratio split over the single JSON array)."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        dataset_glob: str = "mbpp_pro*",
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
        return ["mbpp_pro"]
