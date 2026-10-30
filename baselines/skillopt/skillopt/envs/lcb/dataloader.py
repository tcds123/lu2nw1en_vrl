"""LiveCodeBench (LCB) task dataloader with temporal contamination filtering.

Loads LCB code_generation_lite shards (test.jsonl .. test6.jsonl) from HuggingFace
`livecodebench/code_generation_lite`. Each item carries:
  - question_content (problem statement), contest_date (when the contest ran),
  - public_test_cases (plain JSON), private_test_cases (base64 + zlib + pickle
    of a JSON string of [{input, output, testtype}]).
  - platform, difficulty, question_id, starter_code.

We normalise each item to the flat schema the rollout/evaluator expect:
    {id, question, test_code, raw_solution, difficulty, contest_date}

test_code is repackaged as a JSON blob {stdin:[...], stdout:[...]} — exactly
the contract appsplus/evaluator.parse_test_pairs consumes, so we reuse that
evaluator unchanged (stdin/stdout subprocess comparison, no Docker).

**Contamination filtering** (the whole point of using LCB):
  - DeepSeek-V3.2-Exp (released 2025-09, built on V3 released 2024-12) almost
    certainly saw public datasets like mbpp/humaneval/APPS during pretraining.
    LCB problems each carry a `contest_date`; we keep only problems whose
    contest ran AFTER the model's training-data cutoff, so noskill baseline
    reflects genuine generalisation, not memorisation.
  - `contest_date_after` (YYYY-MM-DD): keep only items with contest_date >= this.
    Default "2025-01-01" is safely post-cutoff for V3.2.
  - `temporal_split` mode: instead of ratio split, split BY DATE — train/val
    from earlier post-cutoff window, test from the latest window. This avoids
    any same-contest leakage between train and test.
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import zlib
from typing import Any

from skillopt.datasets.base import SplitDataLoader


def _decode_test_cases(raw: Any) -> list[dict]:
    """Decode LCB test cases (public or private) into a list of {input,output,testtype}."""
    if not raw:
        return []
    # public_test_cases is a plain JSON string; private_test_cases is
    # base64(zlib(pickle(json_str))).
    if isinstance(raw, str):
        s = raw.strip()
        # try plain JSON first
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
        # try base64 -> zlib -> pickle -> json
        try:
            decoded = base64.b64decode(s)
            decompressed = zlib.decompress(decoded)
            unpickled = pickle.loads(decompressed)
            if isinstance(unpickled, str):
                obj = json.loads(unpickled)
            else:
                obj = unpickled
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    elif isinstance(raw, list):
        return raw
    return []


def _normalize_item(item: dict, row_idx: int) -> dict | None:
    item_id = str(item.get("question_id") or item.get("id") or row_idx)
    prompt = str(item.get("question_content") or item.get("content") or "").strip()
    starter = str(item.get("starter_code") or "").strip()
    difficulty = str(item.get("difficulty") or "").strip()
    contest_date = str(item.get("contest_date") or "").strip()

    if not prompt:
        return None

    pub = _decode_test_cases(item.get("public_test_cases"))
    priv = _decode_test_cases(item.get("private_test_cases"))
    cases = pub + priv
    # keep only stdin-type cases (LCB also has "functional" testtype for some)
    cases = [c for c in cases if c.get("testtype") == "stdin"]
    if not cases:
        return None

    stdins = [str(c.get("input", "")) for c in cases]
    stdouts = [str(c.get("output", "")) for c in cases]

    question = prompt
    if starter:
        question = f"{prompt}\n\n{starter}"

    test_code = json.dumps({"stdin": stdins, "stdout": stdouts}, ensure_ascii=False)

    return {
        "id": item_id,
        "question": question,
        "test_code": test_code,
        "raw_solution": "",
        "difficulty": difficulty,
        "contest_date": contest_date,
    }


def load_items(
    data_path: str,
    contest_date_after: str = "2025-01-01",
    contest_date_before: str = "",
    difficulty_filter: str = "",
) -> list[dict]:
    """Load and normalise LCB items from a directory of test*.jsonl shards.

    `data_path` is a directory containing test.jsonl .. test6.jsonl (or a single
    .jsonl file). Items with contest_date < `contest_date_after` are dropped
    (contamination filter). Items with contest_date >= `contest_date_before`
    are also dropped (upper bound, for temporal-stratified comparison windows).
    `difficulty_filter` ("easy"/"medium"/"hard" or "") further restricts.
    """
    files: list[str] = []
    if os.path.isdir(data_path):
        for name in sorted(os.listdir(data_path)):
            if name.endswith(".jsonl"):
                files.append(os.path.join(data_path, name))
    elif os.path.isfile(data_path) and data_path.endswith(".jsonl"):
        files.append(data_path)
    if not files:
        raise ValueError(f"LCBDataLoader: no .jsonl shards found under {data_path}")

    after = (contest_date_after or "").strip()
    before = (contest_date_before or "").strip()
    diff_filter = (difficulty_filter or "").strip().lower()
    items: list[dict] = []
    row_idx = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                if diff_filter and str(raw.get("difficulty") or "").strip().lower() != diff_filter:
                    continue
                cdate = str(raw.get("contest_date") or "").strip()
                if after and cdate < after:
                    continue
                if before and cdate >= before:
                    continue
                norm = _normalize_item(raw, row_idx=row_idx)
                if norm:
                    items.append(norm)
                row_idx += 1
    if not items:
        raise ValueError(
            f"No valid LCB items loaded from {data_path}"
            f" (contest_date_after={contest_date_after!r},"
            f" contest_date_before={contest_date_before!r},"
            f" difficulty_filter={difficulty_filter!r})"
        )
    return items


class LCBDataLoader(SplitDataLoader):
    """LCB dataloader: ratio split over the contamination-filtered item set.

    The ratio split is deterministic (seeded shuffle) over the post-cutoff
    items only — so train/val/test all come from problems the model has not
    seen. For a stricter temporal split, use split_mode=split_dir with a
    pre-built date-ordered split.
    """

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
        contest_date_after: str = "2025-01-01",
        contest_date_before: str = "",
        difficulty_filter: str = "",
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
        self.contest_date_after = contest_date_after
        self.contest_date_before = contest_date_before
        self.difficulty_filter = difficulty_filter

    def load_raw_items(self, data_path: str) -> list[dict]:
        return load_items(
            data_path,
            contest_date_after=self.contest_date_after,
            contest_date_before=self.contest_date_before,
            difficulty_filter=self.difficulty_filter,
        )

    def get_task_types(self) -> list[str]:
        return ["lcb"]
