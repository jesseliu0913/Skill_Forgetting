#!/usr/bin/env python3
"""
Step 2: Initialize skill memory and bootstrap skills from training data.

This step:
  1. Initializes skill memory with 8 seed families + 8 seed skills
  2. For each math type (in sequential order), reads training data and uses
     gold solutions to bootstrap / revise skills in memory

Input:  outputs/<run>/per_type_data_with_raw/<type>/train_raw.jsonl
Output: outputs/<run>/memory/           (skill memory files)
        outputs/<run>/bootstrap_logs/   (per-type bootstrap logs)

Usage:
    python -m skill_forgetting.math.build_memory \\
        --run_root outputs/sequential_skill_by_type \\
        --extractor_model Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path(__file__).resolve().parent


from skill_forgetting.memory.llm import VLLMTextGenerator
from skill_forgetting.memory.pipeline import DynamicSkillMemoryPipeline
from skill_forgetting.memory.schema import CanonicalSkill, SkillFamily
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.utils import read_json, read_jsonl

DEFAULT_FAMILIES = SKILL / "defaults" / "families.json"
DEFAULT_SKILLS = SKILL / "defaults" / "skills.jsonl"

from skill_forgetting.math.constants import TRAIN_TYPES


def load_families(path: Path) -> list[SkillFamily]:
    payload = read_json(path)
    families = payload.get("families", payload)
    return [SkillFamily.from_dict(item) for item in families]


def load_skills(path: Path) -> list[CanonicalSkill]:
    return [CanonicalSkill.from_dict(item) for item in read_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Build skill memory from training data.")
    parser.add_argument(
        "--run_root", type=Path,
        default=Path("runs/math"),
    )
    parser.add_argument("--extractor_model", type=str, required=True)
    parser.add_argument("--extractor_lora_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max examples per type (for debugging).")
    parser.add_argument("--reset", action="store_true", help="Explicitly replace an existing skill memory.")
    args = parser.parse_args()

    memory_dir = args.run_root / "memory"
    bootstrap_dir = args.run_root / "bootstrap_logs"
    data_root = args.run_root / "per_type_data_with_raw"
    memory_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 2a: Initialize memory ---
    print("=" * 60)
    print("Step 2a: Initializing skill memory")
    print("=" * 60)
    store = SkillMemoryStore(memory_dir)
    store.initialize(
        families=load_families(DEFAULT_FAMILIES),
        seed_skills=load_skills(DEFAULT_SKILLS),
        reset=args.reset,
    )
    print(f"  Initialized {len(store.families)} families, {len(store.active_skills())} skills")
    print(f"  Memory dir: {memory_dir}\n")

    # --- Step 2b: Bootstrap skills per type ---
    print("=" * 60)
    print("Step 2b: Bootstrapping skills from gold solutions")
    print("=" * 60)

    extractor_gen = VLLMTextGenerator(
        model=args.extractor_model,
        lora_path=args.extractor_lora_path,
    )
    pipeline = DynamicSkillMemoryPipeline.with_qwen_extractor(
        store, solver=None, extractor_generator=extractor_gen,
    )

    total_types = len(TRAIN_TYPES)
    for i, type_name in enumerate(TRAIN_TYPES, 1):
        train_path = data_root / type_name / "train_raw.jsonl"
        if not train_path.exists():
            print(f"  [{i}/{total_types}] {type_name}: SKIPPED (no train_raw.jsonl)")
            continue

        output_path = bootstrap_dir / f"{type_name}.bootstrap.jsonl"
        print(f"  [{i}/{total_types}] {type_name}: bootstrapping from {train_path}")
        outputs = pipeline.bootstrap_from_gold(
            input_jsonl=train_path,
            output_jsonl=output_path,
            limit=args.limit,
        )
        print(f"         -> {len(outputs)} rows written to {output_path}")

    print(f"\nSkill memory after bootstrap: {len(store.active_skills())} active skills")
    print("Step 2 done.\n")


if __name__ == "__main__":
    main()
