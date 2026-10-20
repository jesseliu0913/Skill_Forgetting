from __future__ import annotations

import json
from typing import Any

from skill_forgetting.memory.schema import RetrievedSkill
from skill_forgetting.memory.store import SkillMemoryStore


def format_retrieved_skill_block(
    store: SkillMemoryStore,
    retrieved: RetrievedSkill,
    *,
    prompt_style: str = "full",
) -> str:
    record = store.get_skill(retrieved.skill_id)
    if record is None:
        return ""
    skill = record.canonical_skill
    if prompt_style == "short":
        core_steps = ", ".join(skill.canonical_steps[:4]) or "solve directly from the core pattern"
        return (
            f"Relevant skill {retrieved.rank}:\n"
            f"- Skill ID: {skill.skill_id}\n"
            f"- Family: {skill.skill_family}\n"
            f"- When to use: {skill.applicability}\n"
            f"- Core steps: {core_steps}\n"
        )
    if prompt_style != "full":
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")
    return (
        f"Relevant skill {retrieved.rank}:\n"
        f"- Skill ID: {skill.skill_id}\n"
        f"- Family: {skill.skill_family}\n"
        f"- Goal: {skill.goal}\n"
        f"- When to use: {skill.applicability}\n"
        f"- Steps: {', '.join(skill.canonical_steps)}\n"
        f"- Common failure modes: {', '.join(skill.failure_modes)}\n"
    )


def build_solver_prompt(
    row: dict[str, Any],
    store: SkillMemoryStore,
    retrieved_skills: list[RetrievedSkill],
    *,
    prompt_style: str = "full",
) -> str:
    # Must match DEFAULT_SFT_INSTRUCTION in dataset_builder.py exactly
    instruction = (
        "Solve the following competition math problem. "
        "Use the retrieved skills when they fit the problem. "
        "Show concise reasoning and final answer."
    )
    skill_block = "\n".join(
        block
        for block in (
            format_retrieved_skill_block(store, item, prompt_style=prompt_style) for item in retrieved_skills
        )
        if block
    )
    if not skill_block:
        skill_block = "Relevant skills:\n- None retrieved. Solve directly."
    return (
        f"{instruction}\n"
        f"{skill_block}\n"
        f"Type: {row.get('type', 'Unknown')}\n"
        f"Level: {row.get('level', 'Unknown')}\n\n"
        f"Problem:\n{row.get('problem', '')}"
    )


def build_family_routing_prompt(row: dict[str, Any], family_descriptions: list[dict[str, Any]]) -> str:
    return (
        "Given the problem, identify the most relevant existing skill families.\n"
        "Return up to 2 families and the key cues.\n\n"
        f"Families:\n{json.dumps(family_descriptions, ensure_ascii=False, indent=2)}\n\n"
        f"Problem type: {row.get('type', '')}\n"
        f"Level: {row.get('level', '')}\n"
        f"Problem:\n{row.get('problem', '')}\n"
    )


def build_update_prompt(
    row: dict[str, Any],
    retrieved_skills: list[RetrievedSkill],
    store: SkillMemoryStore,
    prediction: str,
    families: list[dict[str, Any]],
) -> str:
    retrieved_payload = []
    for retrieved in retrieved_skills:
        record = store.get_skill(retrieved.skill_id)
        if record is None:
            continue
        retrieved_payload.append(
            {
                "skill_id": record.canonical_skill.skill_id,
                "skill_family": record.canonical_skill.skill_family,
                "goal": record.canonical_skill.goal,
                "applicability": record.canonical_skill.applicability,
                "keywords": record.canonical_skill.keywords,
                "structural_cues": record.canonical_skill.structural_cues,
                "canonical_steps": record.canonical_skill.canonical_steps,
                "failure_modes": record.canonical_skill.failure_modes,
                "score": retrieved.total_score,
            }
        )
    return (
        "You are maintaining a model-relative skill memory for competition math.\n"
        "Given:\n"
        "1. the current problem,\n"
        "2. the model's reasoning and final answer,\n"
        "3. the retrieved skills,\n"
        "decide whether to:\n"
        "- reuse an existing skill,\n"
        "- revise an existing skill,\n"
        "- create a new skill,\n"
        "- merge two near-duplicate skills,\n"
        "- or drop a harmful skill.\n\n"
        "Rules:\n"
        "- prefer reuse over create,\n"
        "- prefer revise over create if the pattern is close,\n"
        "- create a new skill only when no existing skill covers the reasoning pattern,\n"
        "- use a skill_family exactly from the provided family_id values,\n"
        "- reuse, revise, and drop require an exact target_skill_id for an existing skill,\n"
        "- merge requires an exact target_skill_id and an exact existing source skill id in proposed_skill.skill_id,\n"
        "- if you cannot name the required existing skill ids exactly, do not choose that action,\n"
        "- if no retrieved skill is the right target, do not emit reuse, revise, drop, or merge,\n"
        "- do not invent missing families, target ids, or merge sources,\n"
        "- return exactly one JSON object with no markdown fences and no extra text,\n"
        "- for reuse/drop set proposed_skill to null,\n"
        "- for revise/create/merge provide a fully specified proposed_skill.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "action": "reuse|revise|create|merge|drop",\n'
        '  "target_skill_id": "optional string",\n'
        '  "proposed_skill": {\n'
        '    "skill_id": "string",\n'
        '    "skill_family": "string",\n'
        '    "goal": "string",\n'
        '    "applicability": "string",\n'
        '    "keywords": ["..."],\n'
        '    "structural_cues": ["..."],\n'
        '    "output_format_cues": ["..."],\n'
        '    "canonical_steps": ["..."],\n'
        '    "failure_modes": ["..."],\n'
        '    "compose_with": ["..."]\n'
        "  },\n"
        '  "reason": "string",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        f"Families:\n{json.dumps(families, ensure_ascii=False, indent=2)}\n\n"
        f"Problem:\n{row.get('problem', '')}\n\n"
        f"Model output:\n{prediction}\n\n"
        f"Retrieved skills:\n{json.dumps(retrieved_payload, ensure_ascii=False, indent=2)}\n"
    )
