#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ENVIRONMENTS = ["alfworld", "webshop", "search"]


def load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_prompt(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    query = str(row.get("input", "")).strip()
    return f"{instruction}\n\n{query}".strip() if query else instruction


def convert_rows(rows: list[dict[str, Any]], environment: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt = make_prompt(row)
        answer = str(row.get("output", "")).strip()
        if not prompt or not answer:
            continue
        converted.append(
            {
                "data_source": f"skillrl_{environment}",
                "prompt": [{"role": "user", "content": prompt}],
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": idx,
                    "environment": environment,
                    "task_category": str(row.get("task_category", "")),
                    "answer": answer,
                },
            }
        )
    return converted


def source_files(run_root: Path, environment: str) -> tuple[Path, Path]:
    root = run_root / "agent_sft_skill_augmented" / environment
    return root / "skillrl_train.jsonl", root / "skillrl_eval.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build upstream verl RL datasets from SkillRL JSONL.")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--environments", default=",".join(ENVIRONMENTS))
    parser.add_argument("--limit", type=int, default=-1, help="Optional per-split row cap for debug runs.")
    args = parser.parse_args()

    environments = [env.strip() for env in args.environments.split(",") if env.strip()]
    invalid = sorted(set(environments) - set(ENVIRONMENTS))
    if invalid:
        raise SystemExit(f"Invalid environments: {', '.join(invalid)}")

    output_root = args.run_root / "agent_verl_skill"
    mixed_train: list[dict[str, Any]] = []
    mixed_eval: list[dict[str, Any]] = []
    for environment in environments:
        train_src, eval_src = source_files(args.run_root, environment)
        if not train_src.exists() or not eval_src.exists():
            raise FileNotFoundError(f"Missing source files for {environment}: {train_src}, {eval_src}")
        train_rows = convert_rows(load_jsonl(train_src, args.limit), environment)
        eval_rows = convert_rows(load_jsonl(eval_src, args.limit), environment)
        write_jsonl(output_root / environment / "train.jsonl", train_rows)
        write_jsonl(output_root / environment / "eval.jsonl", eval_rows)
        mixed_train.extend(train_rows)
        mixed_eval.extend(eval_rows)
        print(
            f"{environment:10s} train={len(train_rows):>5d} eval={len(eval_rows):>4d} "
            f"-> {output_root / environment}"
        )

    write_jsonl(output_root / "mix_all" / "train.jsonl", mixed_train)
    write_jsonl(output_root / "mix_all" / "eval.jsonl", mixed_eval)
    print(f"mix_all    train={len(mixed_train):>5d} eval={len(mixed_eval):>4d} -> {output_root / 'mix_all'}")


if __name__ == "__main__":
    main()
