#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


from skill_forgetting.agents.constants import ENVIRONMENT_ORDER, SKILL_RL_ROOT
from skill_forgetting.memory.schema import CanonicalSkill, SkillFamily
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.utils import read_json, read_jsonl


def export_paper_style_skillbank(store: SkillMemoryStore, output_path: Path) -> None:
    general_skills = []
    task_specific_skills: dict[str, list[dict]] = {}
    common_mistakes = []

    for record in store.active_skills():
        skill = record.canonical_skill
        payload = {
            "skill_id": skill.skill_id,
            "title": skill.goal,
            "principle": "; ".join(skill.canonical_steps) if skill.canonical_steps else skill.goal,
            "when_to_apply": skill.applicability,
        }
        if skill.skill_family.startswith("general_"):
            general_skills.append(payload)
        elif skill.skill_family.startswith("mistake_"):
            common_mistakes.append(
                {
                    "mistake_id": skill.skill_id,
                    "description": skill.goal,
                    "why_it_happens": "; ".join(skill.failure_modes) if skill.failure_modes else skill.applicability,
                    "how_to_avoid": "; ".join(skill.canonical_steps) if skill.canonical_steps else skill.applicability,
                }
            )
        else:
            task_specific_skills.setdefault(skill.skill_family, []).append(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "general_skills": general_skills,
                "task_specific_skills": task_specific_skills,
                "common_mistakes": common_mistakes,
                "environments": ENVIRONMENT_ORDER,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize SkillRL-style skill memory.")
    parser.add_argument("--run_root", type=Path, default=Path("runs/agents"))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    defaults_root = SKILL_RL_ROOT / "defaults"
    families = [SkillFamily.from_dict(row) for row in read_json(defaults_root / "families.json")["families"]]
    skills = [CanonicalSkill.from_dict(row) for row in read_jsonl(defaults_root / "skills.jsonl")]

    memory_root = args.run_root / "memory"
    store = SkillMemoryStore(memory_root)
    store.initialize(families=families, seed_skills=skills, reset=args.reset)
    export_paper_style_skillbank(store, memory_root / "skillbank.json")

    print(f"Initialized {len(store.active_skills())} active skills in {memory_root}")
    print(f"Exported paper-style skill bank to {memory_root / 'skillbank.json'}")
    print("\nStep 2 done.")


if __name__ == "__main__":
    main()
