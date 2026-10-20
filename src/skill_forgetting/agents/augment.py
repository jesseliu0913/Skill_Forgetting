#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys


from skill_forgetting.agents.constants import ENVIRONMENT_ORDER, SKILL_RL_ROOT
from skill_forgetting.agents.dataset_builder import DEFAULT_SFT_INSTRUCTION, export_skill_augmented_dataset
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.utils import read_jsonl


def write_dataset_info(path: Path, environment: str) -> None:
    payload = {
        f"skillrl_{environment}_train": {
            "file_name": "skillrl_train.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        f"skillrl_{environment}_eval": {
            "file_name": "skillrl_eval.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build skill-augmented SFT datasets for SkillRL environments.")
    parser.add_argument("--run_root", type=Path, default=Path("runs/agents"))
    parser.add_argument("--top_k_skills", type=int, default=4)
    parser.add_argument("--skill_prompt_style", choices=["full", "short"], default="full")
    args = parser.parse_args()
    if args.top_k_skills < 1:
        parser.error("--top_k_skills must be positive")

    raw_root = args.run_root / "agent_sft_data"
    sft_root = args.run_root / "agent_sft_skill_augmented"
    retrieval_dir = args.run_root / "retrieval_logs"
    memory_dir = args.run_root / "memory"

    store = SkillMemoryStore(memory_dir)
    if not store.active_skills():
        raise ValueError("Build a skill memory before augmenting the data.")
    sft_root.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    for environment in ENVIRONMENT_ORDER:
        env_raw_root = raw_root / environment
        env_sft_root = sft_root / environment
        env_sft_root.mkdir(parents=True, exist_ok=True)

        for filename in ("train_raw.jsonl", "eval_raw.jsonl"):
            if not (env_raw_root / filename).is_file():
                raise FileNotFoundError(env_raw_root / filename)
        train_rows = read_jsonl(env_raw_root / "train_raw.jsonl")
        eval_rows = read_jsonl(env_raw_root / "eval_raw.jsonl")
        if not train_rows or not eval_rows:
            raise ValueError(f"Empty train/eval split for {environment}")

        output_train = env_sft_root / "skillrl_train.jsonl"
        output_eval = env_sft_root / "skillrl_eval.jsonl"
        train_retrieval_log = retrieval_dir / f"{environment}.train.retrieval.jsonl"
        eval_retrieval_log = retrieval_dir / f"{environment}.eval.retrieval.jsonl"

        export_skill_augmented_dataset(
            input_rows=train_rows,
            store=store,
            output_jsonl=output_train,
            top_k_skills=args.top_k_skills,
            instruction=DEFAULT_SFT_INSTRUCTION,
            retrieval_log_jsonl=train_retrieval_log,
            skill_prompt_style=args.skill_prompt_style,
        )

        export_skill_augmented_dataset(
            input_rows=eval_rows,
            store=store,
            output_jsonl=output_eval,
            top_k_skills=args.top_k_skills,
            instruction=DEFAULT_SFT_INSTRUCTION,
            retrieval_log_jsonl=eval_retrieval_log,
            skill_prompt_style=args.skill_prompt_style,
        )

        skillbank_src = memory_dir / "skillbank.json"
        if skillbank_src.exists():
            shutil.copy2(skillbank_src, env_sft_root / "skillbank.json")
        write_dataset_info(env_sft_root / "dataset_info.json", environment)
        print(f"{environment:10s} train={len(train_rows):>5d} eval={len(eval_rows):>4d} -> {env_sft_root}")

    print("\nStep 3 done.")


if __name__ == "__main__":
    main()
