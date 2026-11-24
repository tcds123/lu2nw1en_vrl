"""AIME evaluation helpers — numerical answer extraction and comparison.

AIME problems have a single integer answer (0-999). Unlike
LiveMathematicianBench (multiple-choice label match), here correctness is
exact integer equality after extracting ``\\boxed{N}`` / ``<answer>...</answer>``.

Reuses the extraction logic proven in ``eval_aime24_multiseed.py``.
"""
from __future__ import annotations

import re

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _to_int(s: str) -> int | None:
    s = s.strip().replace(",", "")
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def extract_int_answer(text: str) -> int | None:
    """Extract the final integer answer from a model response.

    Priority: ``\\boxed{...}`` → ``<answer>...</answer>`` → last integer in text.
    """
    m = _BOXED_RE.findall(text)
    if m:
        n = _to_int(m[-1])
        if n is not None:
            return n

    m = _ANS_RE.findall(text)
    if m:
        cand = m[-1].strip()
        inner = _BOXED_RE.findall(cand)
        if inner:
            cand = inner[-1].strip()
        n = _to_int(cand)
        if n is not None:
            return n

    nums = re.findall(r"-?\d+", text)
    if nums:
        return _to_int(nums[-1])
    return None


def evaluate(prediction_text: str, correct_answer) -> dict:
    """Numerical comparison. ``correct_answer`` is an integer or int-string."""
    predicted_answer = extract_int_answer(prediction_text)
    try:
        gt = int(str(correct_answer).strip())
    except (ValueError, TypeError):
        gt = None
    is_correct = predicted_answer is not None and gt is not None and predicted_answer == gt
    em = float(is_correct)
    return {
        "em": em,
        "f1": em,
        "sub_em": em,
        "predicted_answer": str(predicted_answer) if predicted_answer is not None else "",
        "predicted_int": predicted_answer,
        "correct_answer": str(gt) if gt is not None else str(correct_answer),
        "correct_int": gt,
    }
