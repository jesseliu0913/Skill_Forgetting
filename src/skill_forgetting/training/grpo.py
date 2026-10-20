"""Sequential skill-augmented GRPO with VERL, including checkpoint merging."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from skill_forgetting.agents.constants import ENVIRONMENT_ORDER
from skill_forgetting.math.constants import TRAIN_TYPES


# Settings retained from the experiment launchers; hardware settings are overridable.
DEFAULTS = {
    "algorithm.adv_estimator": "grpo",
    "critic.enable": False,
    "data.prompt_key": "prompt",
    "data.reward_fn_key": "data_source",
    "data.max_prompt_length": 2048,
    "data.max_response_length": 512,
    "data.dataloader_num_workers": 0,
    "data.train_batch_size": 64,
    "data.val_batch_size": 64,
    "actor_rollout_ref.rollout.name": "vllm",
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.load_format": "auto",
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.5,
    "actor_rollout_ref.rollout.enforce_eager": True,
    "actor_rollout_ref.rollout.free_cache_engine": True,
    "actor_rollout_ref.rollout.max_model_len": 4096,
    "actor_rollout_ref.rollout.max_num_seqs": 64,
    "actor_rollout_ref.rollout.max_num_batched_tokens": 4096,
    "actor_rollout_ref.rollout.n": 4,
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": 4096,
    "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 16384,
    "actor_rollout_ref.actor.optim.lr": 1e-6,
    "actor_rollout_ref.actor.fsdp_config.param_offload": True,
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
    "actor_rollout_ref.ref.fsdp_config.param_offload": True,
    "critic.ppo_micro_batch_size_per_gpu": 1,
    "trainer.logger": ["console"],
    "trainer.total_epochs": 1,
    "trainer.save_freq": 200,
    "trainer.test_freq": -1,
    "trainer.val_before_train": True,
    "reward.custom_reward_function.name": "compute_score",
    "+ray_kwargs.ray_init.include_dashboard": False,
}


def run(command: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def merged_checkpoint(stage: Path, *, dry_run: bool) -> Path:
    """Reload the exact checkpoint recorded by VERL, never guess a prior stage."""
    target = stage / "merged"
    if dry_run:
        actor = stage / "global_step_FINAL" / "actor"
    else:
        tracker = stage / "latest_checkpointed_iteration.txt"
        step = int(tracker.read_text().strip())
        actor = stage / f"global_step_{step}" / "actor"
        if not actor.is_dir():
            raise FileNotFoundError(actor)
        if list(actor.glob("*.safetensors")) or list(actor.glob("pytorch_model*.bin")):
            return actor
    run([
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp", "--local_dir", str(actor),
        "--target_dir", str(target), "--use_cpu_initialization",
    ], dry_run=dry_run)
    if not dry_run and not (list(target.glob("*.safetensors")) or list(target.glob("pytorch_model*.bin"))):
        raise FileNotFoundError(f"Checkpoint merge produced no model weights in {target}")
    return target


def train(args: argparse.Namespace, scratch: Path) -> None:
    tasks = TRAIN_TYPES if args.domain == "math" else ENVIRONMENT_ORDER
    data_name = "math_verl_skill" if args.domain == "math" else "agent_verl_skill"
    reward_module = importlib.import_module(f"skill_forgetting.{args.domain}.reward")
    model = args.model
    env = dict(os.environ)
    env.update(TMPDIR=str(scratch), RAY_TMPDIR=str(scratch / "ray"))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not args.dry_run:
        (scratch / "ray").mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(tasks, 1):
        data = args.run_root.resolve() / data_name / task
        stage = args.run_root.resolve() / "checkpoints" / f"step_{index:02d}_{task}"
        if not args.dry_run:
            for split in ("train", "eval"):
                path = data / f"{split}.jsonl"
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing or empty skill-augmented data: {path}")
            if stage.exists() and any(stage.iterdir()):
                raise FileExistsError(f"Use a fresh run directory; {stage} already contains files.")
        settings = {
            **DEFAULTS,
            "data.train_files": str(data / "train.jsonl"),
            "data.val_files": str(data / "eval.jsonl"),
            "actor_rollout_ref.model.path": model,
            "critic.model.path": model,
            "critic.model.tokenizer_path": model,
            "trainer.experiment_name": f"{args.domain}_{index:02d}_{task}",
            "trainer.default_local_dir": str(stage),
            "trainer.n_gpus_per_node": args.gpus,
            "actor_rollout_ref.rollout.agent.num_workers": args.gpus,
            "reward.num_workers": args.gpus,
            "reward.custom_reward_function.path": str(Path(reward_module.__file__).resolve()),
            "ray_kwargs.ray_init.num_cpus": args.cpus,
            "+ray_kwargs.ray_init._temp_dir": str(scratch / "ray"),
            "hydra.run.dir": str(args.run_root.resolve() / "hydra" / task),
            "+env.name": task,
        }
        command = [sys.executable, "-m", "verl.trainer.main_ppo"]
        command += [f"{key}={json.dumps(value)}" for key, value in settings.items()]
        command += args.override
        run(command, dry_run=args.dry_run, env=env)
        model = str(merged_checkpoint(stage, dry_run=args.dry_run))
    print(f"Final model: {model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["math", "agents"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Repeat to override a VERL setting, e.g. trainer.total_epochs=2.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.gpus < 1 or args.cpus < 3 * args.gpus + 1:
        parser.error("Use at least one GPU and 3 * GPUs + 1 CPUs.")
    if 16 % args.gpus:
        parser.error("The default PPO minibatch size (16) must be divisible by --gpus.")
    if args.dry_run:
        train(args, Path("/tmp/skill-forgetting-preview"))
    else:
        with tempfile.TemporaryDirectory(prefix="sf-") as scratch:
            train(args, Path(scratch))


if __name__ == "__main__":
    main()
