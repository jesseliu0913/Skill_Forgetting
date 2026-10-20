from __future__ import annotations

from typing import Any

from skill_forgetting.agents.constants import ENV_SPECS
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.schema import RetrievedSkill
from skill_forgetting.memory.prompts import format_retrieved_skill_block


def build_agent_query_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem": row.get("instruction", ""),
        "type": row.get("task_category", row.get("environment", "agent")),
        "level": row.get("environment", "agent"),
    }


def build_agent_skill_prompt(
    row: dict[str, Any],
    store: SkillMemoryStore,
    retrieved_skills: list[RetrievedSkill],
    *,
    prompt_style: str = "full",
) -> str:
    env_name = str(row.get("environment", "")).strip().lower()
    env_spec = ENV_SPECS.get(env_name, {})
    system_prompt = env_spec.get("system_prompt", "You are an expert interactive agent.")
    skill_block = "\n".join(
        block
        for block in (
            format_retrieved_skill_block(store, item, prompt_style=prompt_style) for item in retrieved_skills
        )
        if block
    )
    if not skill_block:
        skill_block = "Retrieved Relevant Experience:\n- None retrieved. Solve from the current observation only.\n"
    return (
        f"{system_prompt}\n\n"
        f"## Retrieved Relevant Experience\n"
        f"{skill_block}\n\n"
        f"## Current Task\n"
        f"Environment: {row.get('environment', 'unknown')}\n"
        f"Task Category: {row.get('task_category', 'unknown')}\n\n"
        f"{row.get('instruction', '')}"
    )
