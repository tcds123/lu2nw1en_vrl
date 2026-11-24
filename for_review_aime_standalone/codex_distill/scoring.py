"""Separate mathematical answer accuracy from the requested XML formatting."""
import re


SCORING_POLICY = {
    "name": "explicit_final_answer_v2",
    "math_answer": "Last explicit boxed answer or answer tag; integer 0..999; never the last arbitrary number in prose",
    "format": "Requested <answer>\\boxed{N}</answer> adherence reported separately",
    "budget": "Length-exhausted responses count as no valid answer within the fixed budget; no resampling",
}


def _integer(value):
    value = value.strip()
    if re.fullmatch(r"\+?[0-9]{1,3}", value):
        return int(value)
    wrapped = re.fullmatch(r"\\(?:text|mathrm|mathbf)\s*\{\s*(\+?[0-9]{1,3})\s*\}", value)
    return int(wrapped.group(1)) if wrapped else None


def extract_answer(text):
    boxes = list(re.finditer(r"\\boxed\s*\{", text))
    tags = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE))
    last_box = boxes[-1] if boxes else None
    last_tag = tags[-1] if tags else None
    if last_tag and (not last_box or last_tag.start() > last_box.start()):
        return _integer(last_tag.group(1))
    if not last_box:
        return None
    # Parse the final box, including unsupported/nested expressions. Never skip
    # an invalid final box in favor of a conveniently correct earlier one.
    start = last_box.end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return _integer(text[start:i])
    return None


def score_answer(text, gold, finish_reason=None):
    truncated = finish_reason == "length"
    pred = None if truncated else extract_answer(text)
    strict = re.findall(r"<answer>\s*\\boxed\{\s*(\d{1,3})\s*\}\s*</answer>", text)
    format_valid = bool(strict) and not truncated
    correct = pred == int(gold)
    result = {"predicted_answer": pred, "correct_answer": int(gold), "correct": correct,
              "answer_format_valid": format_valid, "strict_correct": correct and format_valid}
    if truncated:
        result.update(truncated=True, outcome="incomplete_within_token_budget")
    return result
