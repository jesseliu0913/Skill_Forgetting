from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

from skill_forgetting.memory.utils import compare_answers


def _strip_answer_tags(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _format_ok(text: str) -> float:
    has_answer = bool(re.search(r"<answer>.*?</answer>", text, flags=re.IGNORECASE | re.DOTALL))
    has_boxed = r"\boxed{" in text
    return 1.0 if has_answer or has_boxed else 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    del data_source, extra_info
    prediction = _strip_answer_tags(solution_str)
    comparison = compare_answers(prediction, ground_truth)
    exact = float(comparison.is_correct)
    format_ok = _format_ok(solution_str)
    score = exact if exact else 0.1 * format_ok
    return {
        "score": score,
        "exact": exact,
        "format_ok": format_ok,
        "contains": float(bool(comparison.gold.normalized_answer) and comparison.gold.normalized_answer in comparison.prediction.normalized_answer),
    }
