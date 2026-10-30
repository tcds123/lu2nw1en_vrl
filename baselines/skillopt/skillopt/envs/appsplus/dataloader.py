"""APPS-Plus task dataloader (interview subset).

Loads `/data/zhuldz/lunwen/data/APPS_Plus/data/v1/data_with_id.json` (a JSON
array of competitive-programming items) and normalises each item to the flat
schema the rollout/evaluator expect:

    {id, question, test_code, raw_solution, difficulty}

APPS-Plus is stdin/stdout (not assert-based): each item carries parallel
`inputs` / `outputs` lists. We repackage them into a `test_code` string as a
JSON blob that the evaluator parses back into (stdin, expected_stdout) pairs —
keeping the `{test_code}` contract the SkillOpt loop already uses, without
inventing a new schema.

`question` uses `prompt` (the problem statement). We append `starter_code`
when present so the model sees the expected function scaffolding. Only the
`interview` difficulty subset is loaded by default (configurable via
`difficulty_filter`) — it is harder than mbpp/humaneval single-function tasks
(giving skill optimization real headroom) while still having non-zero baseline
pass rates (unlike BigCodeBench where the model could not even produce
runnable code).
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
    prompt = str(item.get("prompt") or "").strip()
    starter = str(item.get("starter_code") or "").strip()
    inputs = item.get("inputs") or []
    outputs = item.get("outputs") or []
    raw_solution = str(item.get("canonical_solution") or "").strip()
    difficulty = str(item.get("difficulty") or "").strip()

    if not prompt or not inputs or not outputs:
        return None
    if len(inputs) != len(outputs):
        # malformed paired tests — skip rather than silently mismatch
        return None

    # Build question: problem statement + starter code if provided.
    question = prompt
    if starter:
        question = f"{prompt}\n\n{starter}"

    # Repackage stdin/stdout pairs as a JSON blob in the test_code field.
    test_code = json.dumps(
        {"stdin": list(inputs), "stdout": list(outputs)},
        ensure_ascii=False,
    )

    return {
        "id": item_id,
        "question": question,
        "test_code": test_code,
        "raw_solution": raw_solution,
        "difficulty": difficulty,
    }


def load_items(
    data_path: str,
    dataset_glob: str = "data_with_id.json",
    difficulty_filter: str = "interview",
) -> list[dict]:
    """Load and normalise APPS-Plus items from a JSON file or directory.

    `data_path` may be a file or a directory. When it is a directory we look
    for `data_with_id.json` (or the first file matching `dataset_glob`).

    `difficulty_filter` keeps only items whose `difficulty` matches; pass "" or
    "all" to disable the filter.
    """
    if os.path.isdir(data_path):
        prefix = dataset_glob.rstrip("*")
        candidates = sorted(
            p for p in os.listdir(data_path)
            if p.lower().startswith(prefix.lower()) and p.lower().endswith(".json")
        )
        # Recurse one level (APPS_Plus data lives under data/v1/).
        if not candidates:
            for sub in sorted(os.listdir(data_path)):
                sub_path = os.path.join(data_path, sub)
                if os.path.isdir(sub_path):
                    candidates = sorted(
                        p for p in os.listdir(sub_path)
                        if p.lower().endswith(".json") and "with_id" in p.lower()
                    )
                    if candidates:
                        data_path = sub_path
                        break
        if not candidates:
            raise ValueError(
                f"APPSPlusDataLoader: no {dataset_glob} found under {data_path}"
            )
        data_path = os.path.join(data_path, candidates[0])

    raw = _load_json(data_path)
    if isinstance(raw, dict):
        for key in ("data", "items", "test"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {data_path}, got {type(raw).__name__}")

    diff_filter = (difficulty_filter or "").strip().lower()
    items: list[dict] = []
    for row_idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        if diff_filter and diff_filter != "all":
            if str(item.get("difficulty") or "").strip().lower() != diff_filter:
                continue
        norm = _normalize_item(item, row_idx=row_idx)
        if norm:
            items.append(norm)
    if not items:
        raise ValueError(
            f"No valid APPS-Plus items loaded from {data_path}"
            f" (difficulty_filter={difficulty_filter!r})"
        )
    return items


class APPSPlusDataLoader(SplitDataLoader):
    """APPS-Plus dataloader (ratio split over the single JSON array)."""

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
        dataset_glob: str = "data_with_id.json",
        difficulty_filter: str = "interview",
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
        self.difficulty_filter = difficulty_filter

    def load_raw_items(self, data_path: str) -> list[dict]:
        return load_items(
            data_path,
            dataset_glob=self.dataset_glob,
            difficulty_filter=self.difficulty_filter,
        )

    def get_task_types(self) -> list[str]:
        return ["apps_plus"]
