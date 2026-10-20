from __future__ import annotations

from pathlib import Path
from typing import Any

from skill_forgetting.memory.prompts import format_retrieved_skill_block
from skill_forgetting.memory.retriever import HybridSkillRetriever
from skill_forgetting.memory.schema import RetrievedSkill
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.utils import build_example_id, write_json, write_jsonl


DEFAULT_SFT_INSTRUCTION = (
    "Solve the following competition math problem. Use the retrieved skills when they fit the problem. "
    "Show concise reasoning and final answer."
)


def build_skill_augmented_input(
    row: dict[str, Any],
    store: SkillMemoryStore,
    retrieved_skills: list[RetrievedSkill],
    *,
    prompt_style: str = "full",
) -> str:
    skill_block = "\n".join(
        block
        for block in (
            format_retrieved_skill_block(store, item, prompt_style=prompt_style) for item in retrieved_skills
        )
        if block
    )
    if not skill_block:
        skill_block = "Relevant skills:\n- None retrieved.\n"
    return (
        f"{skill_block}\n"
        f"Type: {row.get('type', 'Unknown')}\n"
        f"Level: {row.get('level', 'Unknown')}\n\n"
        f"Problem:\n{row.get('problem', '')}"
    )


def build_skill_augmented_sft_example(
    row: dict[str, Any],
    store: SkillMemoryStore,
    retrieved_skills: list[RetrievedSkill],
    *,
    instruction: str = DEFAULT_SFT_INSTRUCTION,
    prompt_style: str = "full",
) -> dict[str, str]:
    solution = str(row.get("solution", row.get("gold_solution", ""))).strip()
    problem = str(row.get("problem", "")).strip()
    if not solution or not problem:
        return {}
    return {
        "instruction": instruction,
        "input": build_skill_augmented_input(row, store, retrieved_skills, prompt_style=prompt_style),
        "output": solution,
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
        retrieved_skills = retriever.retrieve(row, top_k=top_k_skills)
        sft_example = build_skill_augmented_sft_example(
            row,
            store,
            retrieved_skills,
            instruction=instruction,
            prompt_style=skill_prompt_style,
        )
        if not sft_example:
            continue
        example_id = build_example_id(
            str(row.get("problem", "")),
            str(row.get("type", "")),
            str(row.get("level", "")),
        )
        retrieval_rows.append(
            {
                "example_id": example_id,
                "problem": row.get("problem", ""),
                "type": row.get("type", ""),
                "level": row.get("level", ""),
                "retrieved_skill": retrieved_skills[0].skill_id if retrieved_skills else None,
                "retrieved_skill_family": retrieved_skills[0].skill_family if retrieved_skills else None,
                "retrieval_score": retrieved_skills[0].total_score if retrieved_skills else None,
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
                    "columns": {
                        "prompt": "instruction",
                        "query": "input",
                        "response": "output",
                    },
                }
            },
        )

    return sft_rows, retrieval_rows
