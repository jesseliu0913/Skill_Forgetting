#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


from skill_forgetting.agents.constants import ENVIRONMENT_ORDER, SKILL_RL_ROOT
from skill_forgetting.memory.schema import CanonicalSkill, SkillFamily
from skill_forgetting.agents.seed_memory import export_paper_style_skillbank
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.llm import VLLMTextGenerator
from skill_forgetting.memory.pipeline import DynamicSkillMemoryPipeline
from skill_forgetting.memory.utils import read_json, read_jsonl, write_jsonl


def load_families(path: Path) -> list[SkillFamily]:
    payload = read_json(path)
    families = payload.get("families", payload)
    return [SkillFamily.from_dict(item) for item in families]


def load_skills(path: Path) -> list[CanonicalSkill]:
    return [CanonicalSkill.from_dict(item) for item in read_jsonl(path)]


def convert_groundtruth_rows(rows: list[dict], environment: str) -> list[dict]:
    converted: list[dict] = []
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        task_category = str(row.get("task_category", environment)).strip() or environment
        if not instruction or not output:
            continue
        converted.append(
            {
                **row,
                "environment": environment,
                "problem": instruction,
                "type": task_category,
                "level": environment,
                "solution": output,
                "gold_solution": output,
                "prediction": output,
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap SkillRL skill memory from ground-truth agent trajectories."
    )
    parser.add_argument("--run_root", type=Path, default=Path("runs/agents"))
    parser.add_argument("--extractor_model", type=str, required=True)
    parser.add_argument("--extractor_lora_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--environments",
        type=str,
        default=None,
        help="Comma-separated subset of environments to bootstrap. Defaults to all environments.",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    defaults_root = SKILL_RL_ROOT / "defaults"
    raw_root = args.run_root / "agent_sft_data"
    memory_root = args.run_root / "memory"
    bootstrap_root = args.run_root / "bootstrap_logs"
    bootstrap_input_root = bootstrap_root / "groundtruth_inputs"

    memory_root.mkdir(parents=True, exist_ok=True)
    bootstrap_root.mkdir(parents=True, exist_ok=True)
    bootstrap_input_root.mkdir(parents=True, exist_ok=True)

    store = SkillMemoryStore(memory_root)
    store.initialize(
        families=load_families(defaults_root / "families.json"),
        seed_skills=load_skills(defaults_root / "skills.jsonl"),
        reset=args.reset,
    )

    extractor_gen = VLLMTextGenerator(
        model=args.extractor_model,
        lora_path=args.extractor_lora_path,
    )
    pipeline = DynamicSkillMemoryPipeline.with_qwen_extractor(
        store,
        solver=None,
        extractor_generator=extractor_gen,
    )

    selected_environments = set(ENVIRONMENT_ORDER)
    if args.environments:
        selected_environments = {
            item.strip() for item in args.environments.split(",") if item.strip()
        }

    total_outputs = 0
    for environment in ENVIRONMENT_ORDER:
        if environment not in selected_environments:
            print(f"{environment:10s} skipped: filtered by --environments")
            continue
        train_path = raw_root / environment / "train_raw.jsonl"
        if not train_path.exists():
            print(f"{environment:10s} skipped: missing {train_path}")
            continue

        rows = convert_groundtruth_rows(read_jsonl(train_path), environment)
        if args.limit is not None:
            rows = rows[: args.limit]
        if not rows:
            print(f"{environment:10s} skipped: no usable rows")
            continue

        input_path = bootstrap_input_root / f"{environment}.groundtruth.jsonl"
        output_path = bootstrap_root / f"{environment}.bootstrap.jsonl"
        write_jsonl(input_path, rows)
        outputs = pipeline.bootstrap_from_gold(input_path, output_path, limit=None)
        total_outputs += len(outputs)
        print(f"{environment:10s} bootstrapped={len(outputs):>5d} -> {output_path}")

    print(f"\nActive skills after bootstrap: {len(store.active_skills())}")
    export_paper_style_skillbank(store, memory_root / "skillbank.json")
    print(f"Skill bank written to: {memory_root / 'skillbank.json'}")
    print(f"Bootstrap traces written: {total_outputs}")


if __name__ == "__main__":
    main()
