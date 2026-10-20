from __future__ import annotations

import re
import string
from typing import Any


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^<+", "", text)
    text = re.sub(r">+$", "", text)
    text = re.sub(r"(\])>+$", r"\1", text)
    text = re.sub(r"(\})>+$", r"\1", text)
    return text.strip(string.whitespace + string.punctuation)


def _tag_content(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return _normalize(match.group(1))
    return ""


def _environment(data_source: str, extra_info: dict[str, Any] | None) -> str:
    if extra_info and extra_info.get("environment"):
        return str(extra_info["environment"])
    if data_source.startswith("skillrl_"):
        return data_source.removeprefix("skillrl_")
    return ""


def _target(text: str, environment: str) -> str:
    if environment == "search":
        for tag in ("search", "answer"):
            content = _tag_content(text, tag)
            if content:
                return f"{tag}:{content}"
        return ""

    content = _tag_content(text, "action")
    if content:
        return content
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _normalize(lines[-1]) if lines else ""


def _target_tag(target: str) -> str:
    return target.split(":", 1)[0] if ":" in target else "action"


def _target_content(target: str) -> str:
    return target.split(":", 1)[1] if ":" in target else target


def _token_f1(prediction: str, answer: str) -> float:
    pred_tokens = re.findall(r"[a-z0-9]+", _target_content(prediction))
    answer_tokens = re.findall(r"[a-z0-9]+", _target_content(answer))
    if not pred_tokens or not answer_tokens:
        return 0.0

    overlap = 0
    remaining = list(answer_tokens)
    for token in pred_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def _format_ok(text: str, environment: str) -> float:
    has_think = bool(re.search(r"<think>.*?</think>", text, flags=re.IGNORECASE | re.DOTALL))
    if environment == "search":
        has_tool = bool(re.search(r"<(?:search|answer)>.*?</(?:search|answer)>", text, flags=re.IGNORECASE | re.DOTALL))
    else:
        has_tool = bool(re.search(r"<action>.*?</action>", text, flags=re.IGNORECASE | re.DOTALL))
    return float(has_think and has_tool)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    environment = _environment(data_source, extra_info)
    prediction = _target(solution_str, environment)
    answer = _target(ground_truth, environment)
    exact = float(prediction == answer)
    contains = float(bool(answer) and bool(prediction) and (answer in prediction or prediction in answer))
    format_ok = _format_ok(solution_str, environment)

    if environment == "search":
        tool_match = float(bool(prediction and answer and _target_tag(prediction) == _target_tag(answer)))
        query_f1 = _token_f1(prediction, answer) if tool_match else 0.0
        score = max(exact, 0.7 * query_f1, 0.2 * tool_match, 0.1 * format_ok)
        return {
            "score": score,
            "exact": exact,
            "contains": contains,
            "format_ok": format_ok,
            "tool_match": tool_match,
            "query_f1": query_f1,
        }

    score = exact if exact else max(0.5 * contains, 0.2 * format_ok)
    return {
        "score": score,
        "exact": exact,
        "contains": contains,
        "format_ok": format_ok,
        "tool_match": 0.0,
        "query_f1": 0.0,
    }
