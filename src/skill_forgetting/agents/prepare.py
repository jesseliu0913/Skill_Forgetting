#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys


from skill_forgetting.agents.constants import DEFAULT_SFT_DATASET, ENVIRONMENT_ORDER, SKILL_RL_ROOT
from skill_forgetting.agents.dataset_builder import normalize_skillrl_row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_source_rows(dataset_name: str, environment: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "step1_prepare_agent_data.py needs the `datasets` package to download SkillRL-SFT-Data."
        ) from exc
    ds = load_dataset(dataset_name, environment, split="train")
    return [normalize_skillrl_row(dict(row), environment) for row in ds]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_dataset_info(env_root: Path, environment: str) -> None:
    dataset_info = {
        f"skillrl_{environment}_train": {
            "file_name": "train_raw.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        f"skillrl_{environment}_eval": {
            "file_name": "eval_raw.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    with (env_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)


def repair_existing_env_split(run_root: Path, environment: str) -> None:
    env_root = run_root / "agent_sft_data" / environment
    train_path = env_root / "train_raw.jsonl"
    eval_path = env_root / "eval_raw.jsonl"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Cannot repair missing raw splits for {environment}: {train_path}, {eval_path}")

    train_rows = [normalize_skillrl_row(row, environment) for row in load_jsonl(train_path)]
    eval_rows = [normalize_skillrl_row(row, environment) for row in load_jsonl(eval_path)]
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_dataset_info(env_root, environment)
    print(f"{environment:10s} repaired train={len(train_rows):>5d} eval={len(eval_rows):>4d} -> {env_root}")


def prepare_env_split(run_root: Path, environment: str, test_ratio: float, seed: int, dataset_name: str) -> None:
    rows = load_source_rows(dataset_name, environment)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * test_ratio))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    env_root = run_root / "agent_sft_data" / environment
    write_jsonl(env_root / "train_raw.jsonl", train_rows)
    write_jsonl(env_root / "eval_raw.jsonl", eval_rows)
    write_dataset_info(env_root, environment)

    print(
        f"{environment:10s} train={len(train_rows):>5d} eval={len(eval_rows):>4d} -> {env_root}"
    )


def verify_prepared_data(run_root: Path) -> None:
    for environment in ENVIRONMENT_ORDER:
        env_root = run_root / "agent_sft_data" / environment
        train_n = count_lines(env_root / "train_raw.jsonl") if (env_root / "train_raw.jsonl").exists() else 0
        eval_n = count_lines(env_root / "eval_raw.jsonl") if (env_root / "eval_raw.jsonl").exists() else 0
        print(f"{environment:10s} train={train_n:>5d} eval={eval_n:>4d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SkillRL agent SFT data by environment.")
    parser.add_argument("--run_root", type=Path, default=Path("runs/agents"))
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_SFT_DATASET)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--repair_existing", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_prepared_data(args.run_root)
        return

    for environment in ENVIRONMENT_ORDER:
        if args.repair_existing:
            repair_existing_env_split(args.run_root, environment)
        else:
            prepare_env_split(args.run_root, environment, args.test_ratio, args.seed, args.dataset_name)

    print("\nStep 1 done.")


if __name__ == "__main__":
    main()
