"""Sequential skill-augmented math SFT through an external LlamaFactory install."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

from skill_forgetting.math.constants import TRAIN_TYPES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_root", type=Path, default=Path("runs/math"))
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--finetuning_type", choices=["full", "lora"], default="full")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    model = args.model
    for index, task in enumerate(TRAIN_TYPES, 1):
        data = args.run_root.resolve() / "per_type_data_skill_augmented" / task
        destination = args.run_root.resolve() / "checkpoints" / f"step_{index:02d}_{task}"
        train_dir = destination.with_name(destination.name + "_adapter") if args.finetuning_type == "lora" else destination
        if not args.dry_run:
            for filename in ("competition_math_train.jsonl", "dataset_info.json"):
                if not (data / filename).is_file():
                    raise FileNotFoundError(data / filename)
            for path in {destination, train_dir}:
                if path.exists() and any(path.iterdir()):
                    raise FileExistsError(f"Use a fresh run directory; {path} already contains files.")
        command = [
            "llamafactory-cli", "train", "--stage", "sft", "--do_train", "true",
            "--model_name_or_path", model, "--template", args.template,
            "--finetuning_type", args.finetuning_type,
            "--dataset_dir", str(data), "--dataset", "competition_math_train",
            "--output_dir", str(train_dir), "--cutoff_len", str(args.cutoff_len),
            "--learning_rate", str(args.learning_rate), "--num_train_epochs", str(args.epochs),
            "--per_device_train_batch_size", str(args.batch_size),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--preprocessing_num_workers", "1", "--logging_steps", "10",
            "--save_strategy", "epoch", "--save_total_limit", "2",
            "--lr_scheduler_type", "cosine", "--warmup_ratio", "0.03",
            "--bf16", "true", "--report_to", "none",
        ]
        if args.finetuning_type == "lora":
            command += ["--lora_rank", "16", "--lora_alpha", "32", "--lora_dropout", "0.05", "--lora_target", "all"]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
        if args.finetuning_type == "lora":
            merge = [
                "llamafactory-cli", "export", "--model_name_or_path", model,
                "--adapter_name_or_path", str(train_dir), "--template", args.template,
                "--finetuning_type", "lora", "--export_dir", str(destination),
                "--export_size", "2", "--export_legacy_format", "false",
            ]
            print(shlex.join(merge), flush=True)
            if not args.dry_run:
                subprocess.run(merge, check=True)
        if not args.dry_run and not (
            list(destination.glob("*.safetensors")) or list(destination.glob("pytorch_model*.bin"))
        ):
            raise FileNotFoundError(f"Training did not produce model weights in {destination}")
        model = str(destination)


if __name__ == "__main__":
    main()
