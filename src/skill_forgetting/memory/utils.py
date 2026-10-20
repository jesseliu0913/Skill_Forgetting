from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


WORD_RE = re.compile(r"[A-Za-z0-9]+")
FINAL_ANSWER_LINE_RE = re.compile(r"Final Answer\s*:\s*(.+)", re.IGNORECASE)


def _extract_boxed_contents(text: str) -> list[str]:
    """Extract all \\boxed{...} contents, handling nested braces correctly."""
    results = []
    search_from = 0
    marker = r"\boxed{"
    while True:
        start = text.find(marker, search_from)
        if start == -1:
            break
        content_start = start + len(marker)
        depth = 1
        i = content_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[content_start : i - 1])
        search_from = start + 1
    return results


def _strip_outer_boxed(text: str) -> str:
    """If text is exactly \\boxed{...}, return the inner content; else return text."""
    stripped = text.strip()
    if not stripped.startswith(r"\boxed{"):
        return text
    contents = _extract_boxed_contents(stripped)
    if len(contents) == 1 and stripped == r"\boxed{" + contents[0] + "}":
        return contents[0]
    return text

MATH_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "find",
    "what",
    "when",
    "where",
    "which",
    "each",
    "have",
    "has",
    "into",
    "onto",
    "then",
    "than",
    "there",
    "their",
    "whose",
    "such",
    "point",
    "points",
    "line",
    "lines",
    "segment",
    "segments",
    "triangle",
    "triangles",
    "circle",
    "circles",
    "value",
    "values",
    "express",
    "answer",
    "common",
    "simplest",
}

FORMAT_CUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("fraction", re.compile(r"common fraction|fraction", re.IGNORECASE)),
    ("radical", re.compile(r"radical", re.IGNORECASE)),
    ("interval", re.compile(r"interval", re.IGNORECASE)),
    ("integer", re.compile(r"integer|integers|how many", re.IGNORECASE)),
    ("coordinate", re.compile(r"ordered pair|coordinate", re.IGNORECASE)),
    ("degrees", re.compile(r"degrees|degree", re.IGNORECASE)),
    ("pi_expression", re.compile(r"in terms of \\\\pi|in terms of pi", re.IGNORECASE)),
    ("remainder", re.compile(r"remainder|mod", re.IGNORECASE)),
]

STRUCTURAL_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("construct a number", re.compile(r"digit|digits|three-digit|four-digit|number formed", re.IGNORECASE)),
    ("sum of remainders", re.compile(r"sum of remainders|remainder", re.IGNORECASE)),
    ("finite integer enumeration", re.compile(r"how many integers|positive integers|nonnegative integers", re.IGNORECASE)),
    ("casework counting", re.compile(r"how many ways|how many|arrange|select|choose|probability", re.IGNORECASE)),
    ("divisibility constraint", re.compile(r"divisible|divisibility|multiple of|mod", re.IGNORECASE)),
    ("functional relation", re.compile(r"function|f\(|g\(|h\(|polynomial|sequence|recurrence", re.IGNORECASE)),
    ("analytic geometry", re.compile(r"coordinate|slope|distance|ellipse|parabola|circle|line", re.IGNORECASE)),
    ("geometry configuration", re.compile(r"triangle|circle|angle|tangent|chord|arc|perpendicular|parallel", re.IGNORECASE)),
    ("extremal objective", re.compile(r"maximum|minimum|greatest|least|largest|smallest", re.IGNORECASE)),
    ("inequality or interval", re.compile(r"nonnegative|positive|negative|interval|inequality", re.IGNORECASE)),
]

LATEX_WRAPPERS: tuple[tuple[str, str], ...] = (
    ("\\(", "\\)"),
    ("\\[", "\\]"),
    ("$$", "$$"),
    ("$", "$"),
)


@dataclass(frozen=True)
class AnswerAnalysis:
    extracted_answer: str
    normalized_answer: str
    source: str
    is_empty: bool


@dataclass(frozen=True)
class AnswerComparison:
    gold: AnswerAnalysis
    prediction: AnswerAnalysis
    is_correct: bool
    error_type: str


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def slugify(text: str) -> str:
    text = text.strip().lower()
    out: list[str] = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def tokenize(text: str) -> list[str]:
    tokens = [m.group(0).lower() for m in WORD_RE.finditer(text or "")]
    return [t for t in tokens if len(t) > 1]


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        norm = normalize_whitespace(str(value))
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def top_content_tokens(text: str, limit: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    for token in tokenize(text):
        if token in MATH_STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def infer_structural_cues(problem: str) -> list[str]:
    matches = [name for name, pattern in STRUCTURAL_RULES if pattern.search(problem or "")]
    return dedupe_preserve_order(matches)


def infer_output_format_cues(problem: str) -> list[str]:
    matches = [name for name, pattern in FORMAT_CUE_PATTERNS if pattern.search(problem or "")]
    return dedupe_preserve_order(matches)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_example_id(problem: str, problem_type: str, level: str) -> str:
    payload = json.dumps(
        {
            "problem": normalize_whitespace(problem),
            "type": normalize_whitespace(problem_type),
            "level": normalize_whitespace(level),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _has_outer_braces(text: str) -> bool:
    if len(text) < 2 or not (text.startswith("{") and text.endswith("}")):
        return False
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
        if depth < 0:
            return False
    return depth == 0


def canonicalize_answer(text: str) -> str:
    candidate = normalize_whitespace(text).strip(" .")
    if not candidate:
        return ""

    while True:
        updated = candidate
        stripped = _strip_outer_boxed(updated)
        if stripped != updated:
            updated = normalize_whitespace(stripped).strip(" .")
        for left, right in LATEX_WRAPPERS:
            if updated.startswith(left) and updated.endswith(right):
                updated = normalize_whitespace(updated[len(left) : len(updated) - len(right)]).strip(" .")
        if _has_outer_braces(updated):
            updated = normalize_whitespace(updated[1:-1]).strip(" .")
        if updated == candidate:
            return updated
        candidate = updated


def _answer_candidates(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        match = FINAL_ANSWER_LINE_RE.search(line.strip())
        if match:
            candidates.append(("final_answer_line", match.group(1)))
    for boxed in _extract_boxed_contents(text or ""):
        candidates.append(("boxed_expression", boxed))
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if lines:
        candidates.append(("terminal_line", lines[-1]))
    return candidates


def analyze_answer(text: str) -> AnswerAnalysis:
    normalized_text = normalize_whitespace(text)
    if not normalized_text:
        return AnswerAnalysis(extracted_answer="", normalized_answer="", source="empty", is_empty=True)

    candidates = _answer_candidates(text)
    if not candidates:
        return AnswerAnalysis(extracted_answer="", normalized_answer="", source="none", is_empty=False)

    last_candidate_by_source = {source: candidate_text for source, candidate_text in candidates}
    for source in ("final_answer_line", "boxed_expression", "terminal_line"):
        extracted = last_candidate_by_source.get(source, "")
        if extracted:
            break
    else:
        source, extracted = "none", ""
    extracted = normalize_whitespace(extracted).strip(" .")
    return AnswerAnalysis(
        extracted_answer=extracted,
        normalized_answer=canonicalize_answer(extracted),
        source=source,
        is_empty=False,
    )


def compare_answers(prediction: str, gold_solution: str) -> AnswerComparison:
    gold = analyze_answer(gold_solution)
    prediction_analysis = analyze_answer(prediction)

    if prediction_analysis.is_empty:
        error_type = "empty_prediction"
        is_correct = False
    elif not prediction_analysis.normalized_answer:
        error_type = "missing_prediction_answer"
        is_correct = False
    elif not gold.normalized_answer:
        error_type = "missing_gold_answer"
        is_correct = False
    else:
        is_correct = prediction_analysis.normalized_answer == gold.normalized_answer
        error_type = "correct" if is_correct else "wrong_final_answer"

    return AnswerComparison(
        gold=gold,
        prediction=prediction_analysis,
        is_correct=is_correct,
        error_type=error_type,
    )


def extract_final_answer(text: str) -> str:
    return analyze_answer(text).extracted_answer


def classify_error_type(prediction: str, gold_solution: str) -> str:
    return compare_answers(prediction, gold_solution).error_type


def stable_hash_embedding(text: str, dims: int = 256) -> list[float]:
    vector = [0.0] * dims
    tokens = tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dims
        sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(vec_a, vec_b))))


def overlap_ratio(query_items: Iterable[str], target_items: Iterable[str]) -> tuple[float, list[str]]:
    query_set = {normalize_whitespace(item).lower() for item in query_items if normalize_whitespace(item)}
    target_lookup = {normalize_whitespace(item).lower(): normalize_whitespace(item) for item in target_items}
    if not target_lookup or not query_set:
        return 0.0, []
    matched_keys = [key for key in query_set if key in target_lookup]
    matched = [target_lookup[key] for key in matched_keys]
    return len(matched_keys) / max(1, len(target_lookup)), dedupe_preserve_order(matched)
