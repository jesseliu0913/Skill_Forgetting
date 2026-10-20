#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

from skill_forgetting.math.constants import EVAL_TYPES, TRAIN_TYPES
from skill_forgetting.memory.utils import analyze_answer


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
    query = str(row.get("input", row.get("problem", ""))).strip()
    return f"{instruction}\n\n{query}".strip() if query else instruction


def gold_answer(row: dict[str, Any]) -> str:
    output = str(row.get("output", row.get("solution", ""))).strip()
    return analyze_answer(output).extracted_answer or output


def convert_rows(rows: list[dict[str, Any]], math_type: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt = make_prompt(row)
        answer = gold_answer(row)
        if not prompt or not answer:
            continue
        converted.append(
            {
                "data_source": f"mathrl_{math_type}",
                "prompt": [{"role": "user", "content": prompt}],
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": idx,
                    "math_type": math_type,
                    "type": str(row.get("type", math_type)),
                    "level": str(row.get("level", "")),
                    "answer": answer,
                },
            }
        )
    return converted


def source_files(run_root: Path, math_type: str) -> tuple[Path, Path]:
    root = run_root / "per_type_data_skill_augmented" / math_type
    return root / "competition_math_train.jsonl", root / "competition_math_test.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verl RL datasets from math SFT JSONL.")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--types", default=",".join(TRAIN_TYPES))
    parser.add_argument("--eval_types", default=",".join(EVAL_TYPES))
    parser.add_argument("--limit", type=int, default=-1, help="Optional per-split row cap for debug runs.")
    args = parser.parse_args()

    train_types = [t.strip() for t in args.types.split(",") if t.strip()]
    eval_types = [t.strip() for t in args.eval_types.split(",") if t.strip()]
    known = set(EVAL_TYPES)
    invalid = sorted((set(train_types) | set(eval_types)) - known)
    if invalid:
        raise SystemExit(f"Invalid math types: {', '.join(invalid)}")

    output_root = args.run_root / "math_verl_skill"
    mixed_train: list[dict[str, Any]] = []
    mixed_eval: list[dict[str, Any]] = []

    for math_type in train_types:
        train_src, _ = source_files(args.run_root, math_type)
        if not train_src.exists():
            raise FileNotFoundError(f"Missing train source for {math_type}: {train_src}")
        train_rows = convert_rows(load_jsonl(train_src, args.limit), math_type)
        write_jsonl(output_root / math_type / "train.jsonl", train_rows)
        mixed_train.extend(train_rows)
        print(f"{math_type:30s} train={len(train_rows):>5d} -> {output_root / math_type / 'train.jsonl'}")

    for math_type in eval_types:
        _, eval_src = source_files(args.run_root, math_type)
        if not eval_src.exists():
            raise FileNotFoundError(f"Missing eval source for {math_type}: {eval_src}")
        eval_rows = convert_rows(load_jsonl(eval_src, args.limit), math_type)
        write_jsonl(output_root / math_type / "eval.jsonl", eval_rows)
        mixed_eval.extend(eval_rows)
        print(f"{math_type:30s} eval ={len(eval_rows):>5d} -> {output_root / math_type / 'eval.jsonl'}")

    write_jsonl(output_root / "mix_all" / "train.jsonl", mixed_train)
    write_jsonl(output_root / "mix_all" / "eval.jsonl", mixed_eval)
    print(f"{'mix_all':30s} train={len(mixed_train):>5d} eval={len(mixed_eval):>5d} -> {output_root / 'mix_all'}")


if __name__ == "__main__":
    main()
