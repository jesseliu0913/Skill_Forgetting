from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from skill_forgetting.agents.prompts import build_agent_query_row, build_agent_skill_prompt
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.retriever import HybridSkillRetriever
from skill_forgetting.memory.utils import write_json, write_jsonl

DEFAULT_SFT_INSTRUCTION = (
    "Follow the environment instruction, use retrieved skills only when relevant, "
    "reason concisely, and emit the action format expected by the environment."
)


def strip_retrieved_experience(instruction: str) -> str:
    """Remove pre-existing retrieval blocks before adding our retrieved skills."""
    text = instruction.strip()
    marker = "\n\n## Retrieved Relevant Experience\n"
    start = text.find(marker)
    if start < 0:
        return text

    after_marker = start + len(marker)
    next_heading = re.search(r"\n\n## [^\n]+\n", text[after_marker:])
    if next_heading:
        end = after_marker + next_heading.start()
    else:
        turn_marker = "\n\nNow it's your turn"
        end = text.find(turn_marker, after_marker)
        if end < 0:
            end = len(text)

    return f"{text[:start].rstrip()}\n\n{text[end:].lstrip()}".strip()


def extract_task_text(environment: str, instruction: str) -> str:
    text = instruction.strip()
    env = environment.lower()
    patterns = []
    if env in {"alfworld", "webshop"}:
        patterns.append(r"your task is to:\s*(.+)")
    if env == "search":
        patterns.append(r"your question:\s*(.+)")
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).split("\n## ", 1)[0].strip()
            if extracted:
                return extracted
    return text.split("\n## ", 1)[0].strip()


def infer_task_category(environment: str, instruction: str) -> str:
    text = extract_task_text(environment, instruction).lower()
    env = environment.lower()
    if env == "alfworld":
        if "clean" in text:
            return "clean"
        if "heat" in text or "hot" in text:
            return "heat"
        if "cool" in text or "cold" in text or "fridge" in text:
            return "cool"
        if "examine" in text or "look at" in text:
            return "examine"
        if "put" in text or "place" in text:
            return "pick_and_place"
        return "alfworld_general"
    if env == "webshop":
        if "compare" in text:
            return "compare"
        if "buy" in text or "purchase" in text:
            return "purchase"
        if "filter" in text or "sort" in text:
            return "filter"
        if "price" in text or "attribute" in text:
            return "attribute_matching"
        return "search_navigation"
    if env == "search":
        if "compare" in text or "difference" in text:
            return "comparison"
        if "why" in text or "how" in text:
            return "multi_hop_reasoning"
        if "verify" in text or "true" in text or "false" in text:
            return "verification"
        return "fact_lookup"
    return f"{env}_general"


def normalize_skillrl_row(row: dict[str, Any], environment: str) -> dict[str, Any]:
    instruction = strip_retrieved_experience(str(row.get("instruction", "")))
    output = str(row.get("output", "")).strip()
    return {
        "environment": environment,
        "instruction": instruction,
        "input": "",
        "output": output,
        "task_category": infer_task_category(environment, instruction),
        "problem": instruction,
        "type": infer_task_category(environment, instruction),
        "level": environment,
    }


def build_skill_augmented_sft_example(
    row: dict[str, Any],
    store: SkillMemoryStore,
    retrieved_skills: list[Any],
    *,
    instruction: str = DEFAULT_SFT_INSTRUCTION,
    prompt_style: str = "full",
) -> dict[str, str]:
    output = str(row.get("output", "")).strip()
    prompt = build_agent_skill_prompt(row, store, retrieved_skills, prompt_style=prompt_style)
    if not output or not prompt:
        return {}
    return {
        "instruction": instruction,
        "input": prompt,
        "output": output,
    }


def export_skill_augmented_dataset(
    *,
    input_rows: list[dict[str, Any]],
    store: SkillMemoryStore,
    output_jsonl: Path,
    top_k_skills: int | None = None,
    instruction: str = DEFAULT_SFT_INSTRUCTION,
    dataset_name: str | None = None,
    dataset_info_json: Path | None = None,
    retrieval_log_jsonl: Path | None = None,
    skill_prompt_style: str = "full",
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    retriever = HybridSkillRetriever(store)
    sft_rows: list[dict[str, str]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for row in input_rows:
        query_row = build_agent_query_row(row)
        retrieved_skills = retriever.retrieve(query_row, top_k=top_k_skills)
        sft_example = build_skill_augmented_sft_example(
            row,
            store,
            retrieved_skills,
            instruction=instruction,
            prompt_style=skill_prompt_style,
        )
        if not sft_example:
            continue
        retrieval_rows.append(
            {
                "environment": row.get("environment", ""),
                "task_category": row.get("task_category", ""),
                "instruction": row.get("instruction", ""),
                "retrieved_skills": [item.to_dict() for item in retrieved_skills],
                "skill_prompt_style": skill_prompt_style,
            }
        )
        sft_rows.append(sft_example)

    write_jsonl(output_jsonl, sft_rows)
    if retrieval_log_jsonl is not None:
        write_jsonl(retrieval_log_jsonl, retrieval_rows)
    if dataset_info_json is not None:
        if dataset_name is None:
            dataset_name = output_jsonl.stem
        write_json(
            dataset_info_json,
            {
                dataset_name: {
                    "file_name": output_jsonl.name,
                    "columns": {"prompt": "instruction", "query": "input", "response": "output"},
                }
            },
        )
    return sft_rows, retrieval_rows
