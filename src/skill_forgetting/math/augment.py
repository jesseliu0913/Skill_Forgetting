"""Build skill-augmented training prompts and retain held-out evaluation inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from skill_forgetting.math.constants import EVAL_TYPES, TRAIN_TYPES
from skill_forgetting.math.dataset_builder import export_skill_augmented_dataset
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.utils import read_jsonl, write_json, write_jsonl


def evaluation_example(row: dict) -> dict:
    return {
        "instruction": "Solve the following competition math problem. Show concise reasoning and final answer.",
        "input": (
            f"Type: {row.get('type', 'Unknown')}\n"
            f"Level: {row.get('level', 'Unknown')}\n\nProblem:\n{row['problem']}"
        ),
        "output": row["solution"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, default=Path("runs/math"))
    parser.add_argument("--top_k_skills", type=int, default=2)
    parser.add_argument("--skill_prompt_style", choices=["full", "short"], default="full")
    args = parser.parse_args()
    if args.top_k_skills < 1:
        parser.error("--top_k_skills must be positive")
    store = SkillMemoryStore(args.run_root / "memory")
    if not store.active_skills():
        raise ValueError("Build a skill memory before augmenting the data.")
    for task in EVAL_TYPES:
        source = args.run_root / "per_type_data_with_raw" / task
        target = args.run_root / "per_type_data_skill_augmented" / task
        test_path = source / "test_raw.jsonl"
        if not test_path.is_file():
            raise FileNotFoundError(test_path)
        test_rows = read_jsonl(test_path)
        write_jsonl(target / "test_raw.jsonl", test_rows)
        write_jsonl(target / "competition_math_test.jsonl", map(evaluation_example, test_rows))
        if task in TRAIN_TYPES:
            train_path = source / "train_raw.jsonl"
            if not train_path.is_file():
                raise FileNotFoundError(train_path)
            rows, _ = export_skill_augmented_dataset(
                input_rows=read_jsonl(train_path), store=store,
                output_jsonl=target / "competition_math_train.jsonl",
                top_k_skills=args.top_k_skills,
                retrieval_log_jsonl=args.run_root / "retrieval_logs" / f"{task}.jsonl",
                skill_prompt_style=args.skill_prompt_style,
            )
            if not rows:
                raise ValueError(f"No usable training examples for {task}")
        write_json(target / "dataset_info.json", {
            f"competition_math_{split}": {
                "file_name": f"competition_math_{split}.jsonl",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            }
            for split in (["train", "test"] if task in TRAIN_TYPES else ["test"])
        })
        print(f"{task}: {target}")


if __name__ == "__main__":
    main()
