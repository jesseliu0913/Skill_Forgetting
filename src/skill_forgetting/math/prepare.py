#!/usr/bin/env python3
"""
Step 1: Prepare per-type train/test data splits.

Takes the competition_math dataset (already split by math type) and creates
per-type directories with train_raw.jsonl / test_raw.jsonl for the experiment.

Input:  competition_math_splits/train/by_type/*.jsonl
        or hendrycks_math_splits/{train,test}/by_type/*.jsonl
Output: outputs/<run>/per_type_data_with_raw/<type>/train_raw.jsonl + test_raw.jsonl

Usage:
    python -m skill_forgetting.math.prepare --run_root outputs/sequential_skill_by_type
    python -m skill_forgetting.math.prepare --verify_only   # just check existing data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path(__file__).resolve().parent
BY_TYPE_DIR = Path("data/hendrycks_math_splits")

TYPES = [
    "algebra",
    "counting___probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def infer_split_root(requested: Path) -> Path | None:
    if (requested / "train" / "by_type").exists() and (requested / "test" / "by_type").exists():
        return requested
    if requested.name == "by_type" and requested.parent.name in {"train", "test"}:
        candidate = requested.parent.parent
        if (candidate / "train" / "by_type").exists() and (candidate / "test" / "by_type").exists():
            return candidate
    return None


def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def verify_source_data(source_dir: Path) -> bool:
    explicit_split_root = infer_split_root(source_dir)
    print("=" * 60)
    if explicit_split_root is not None:
        print(f"Verifying source data: {explicit_split_root}/{{train,test}}/by_type/")
    else:
        print("Verifying source data: competition_math_splits/train/by_type/")
    print("=" * 60)
    total = 0
    ok = True
    for t in TYPES:
        if explicit_split_root is not None:
            train_p = explicit_split_root / "train" / "by_type" / f"{t}.jsonl"
            test_p = explicit_split_root / "test" / "by_type" / f"{t}.jsonl"
            if train_p.exists() and test_p.exists():
                train_n = count_lines(train_p)
                test_n = count_lines(test_p)
                total += train_n + test_n
                print(f"  {t:30s} train={train_n:>5d}  test={test_n:>4d}")
            else:
                print(f"  {t:30s}  MISSING")
                ok = False
        else:
            p = source_dir / f"{t}.jsonl"
            if p.exists():
                n = count_lines(p)
                total += n
                print(f"  {t:30s} {n:>6d} examples")
            else:
                print(f"  {t:30s}  MISSING")
                ok = False
    print(f"  {'TOTAL':30s} {total:>6d}")
    return ok


def prepare_splits(run_root: Path, source_dir: Path, test_ratio: float, seed: int) -> None:
    """Split each type into train/test and write to run_root/per_type_data_with_raw/."""
    import random

    out_root = run_root / "per_type_data_with_raw"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nPreparing train/test splits (test_ratio={test_ratio}, seed={seed})")
    print(f"Output: {out_root}\n")

    rng = random.Random(seed)
    total_train, total_test = 0, 0
    explicit_split_root = infer_split_root(source_dir)

    for t in TYPES:
        if explicit_split_root is not None:
            train_src = explicit_split_root / "train" / "by_type" / f"{t}.jsonl"
            test_src = explicit_split_root / "test" / "by_type" / f"{t}.jsonl"
            with open(train_src) as f:
                train_rows = [json.loads(line) for line in f]
            with open(test_src) as f:
                test_rows = [json.loads(line) for line in f]
        else:
            src = source_dir / f"{t}.jsonl"
            with open(src) as f:
                rows = [json.loads(line) for line in f]
            rng.shuffle(rows)
            n_test = max(1, int(len(rows) * test_ratio))
            test_rows = rows[:n_test]
            train_rows = rows[n_test:]

        type_dir = out_root / t
        type_dir.mkdir(parents=True, exist_ok=True)

        train_path = type_dir / "train_raw.jsonl"
        test_path = type_dir / "test_raw.jsonl"
        for path, data in [(train_path, train_rows), (test_path, test_rows)]:
            with open(path, "w") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        total_train += len(train_rows)
        total_test += len(test_rows)
        print(f"  {t:30s} train={len(train_rows):>5d}  test={len(test_rows):>4d}")

    print(f"  {'TOTAL':30s} train={total_train:>5d}  test={total_test:>4d}")


def verify_prepared_data(run_root: Path) -> None:
    """Print summary of already-prepared data."""
    out_root = run_root / "per_type_data_with_raw"
    print(f"\nVerifying prepared data in: {out_root}\n")
    for t in TYPES:
        type_dir = out_root / t
        train_p = type_dir / "train_raw.jsonl"
        test_p = type_dir / "test_raw.jsonl"
        train_n = count_lines(train_p) if train_p.exists() else 0
        test_n = count_lines(test_p) if test_p.exists() else 0
        status = "OK" if train_n > 0 and test_n > 0 else "MISSING"
        print(f"  {t:30s} train={train_n:>5d}  test={test_n:>4d}  [{status}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Prepare per-type data splits.")
    parser.add_argument(
        "--run_root", type=Path,
        default=Path("runs/math"),
        help="Root dir for this experiment run.",
    )
    parser.add_argument(
        "--by_type_dir", type=Path,
        default=BY_TYPE_DIR,
        help="Directory containing <type>.jsonl, or a split root with train/by_type and test/by_type.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify_only", action="store_true", help="Only verify existing data, don't prepare.")
    args = parser.parse_args()

    if not verify_source_data(args.by_type_dir):
        print("\nSource data incomplete. Cannot proceed.")
        sys.exit(1)

    if args.verify_only:
        verify_prepared_data(args.run_root)
    else:
        prepare_splits(args.run_root, args.by_type_dir, args.test_ratio, args.seed)

    print("\nStep 1 done.")


if __name__ == "__main__":
    main()
