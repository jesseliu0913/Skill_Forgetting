#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


from skill_forgetting.agents.constants import ENVIRONMENT_ORDER, SKILL_RL_ROOT
from skill_forgetting.agents.seed_memory import export_paper_style_skillbank
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.agents.build_memory import (
    convert_groundtruth_rows,
    load_families,
    load_skills,
)
from skill_forgetting.memory.extractor import InvalidProposalError, QwenSkillExtractor
from skill_forgetting.memory.llm import VLLMTextGenerator, parse_json_object
from skill_forgetting.memory.prompts import build_solver_prompt, build_update_prompt
from skill_forgetting.memory.retriever import HybridSkillRetriever
from skill_forgetting.memory.schema import RetrievedSkill, ThresholdConfig, UpdateProposal
from skill_forgetting.memory.updater import UpdateManager
from skill_forgetting.memory.utils import build_example_id, compare_answers, read_jsonl, write_jsonl


def iter_chunks(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


class BatchedGoldBootstrapper:
    def __init__(
        self,
        store: SkillMemoryStore,
        generator: VLLMTextGenerator,
        *,
        batch_size: int,
        max_tokens: int,
    ) -> None:
        self.store = store
        self.generator = generator
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.thresholds = ThresholdConfig.from_dict(store.metadata.get("thresholds"))
        self.retriever = HybridSkillRetriever(store, thresholds=self.thresholds)
        self.updater = UpdateManager(store, thresholds=self.thresholds)
        self.extractor = QwenSkillExtractor(generator, store.families)

    def _row_example_id(self, row: dict[str, Any]) -> str:
        return build_example_id(str(row.get("problem", "")), str(row.get("type", "")), str(row.get("level", "")))

    def _apply_single(
        self,
        row: dict[str, Any],
        prediction: str,
        retrieved_skills: list[RetrievedSkill],
        raw_text: str,
    ) -> dict[str, Any]:
        comparison = compare_answers(prediction, str(row.get("solution", row.get("gold_solution", ""))))
        example_id = self._row_example_id(row)
        proposal_payload: dict[str, Any] | None = None
        proposal_error = ""
        update_action = "extractor_error"
        applied = False
        apply_reason = "extractor proposal unavailable"

        try:
            proposal_payload = parse_json_object(raw_text)
            proposal = UpdateProposal.from_dict(proposal_payload)
            self.extractor._validate_proposal(proposal, self.store)
            proposal.example_id = proposal.example_id or example_id
            update_action = proposal.action
            applied, apply_reason = self.updater.apply_proposal(
                proposal,
                row,
                prediction,
                correct=comparison.is_correct,
            )
        except Exception as exc:
            if isinstance(exc, InvalidProposalError):
                proposal_payload = exc.payload or proposal_payload
            proposal_error = f"Invalid extractor proposal: {exc}"
            apply_reason = f"extractor proposal rejected: {exc}"

        self.store.record_retrieval_event(
            {
                "example_id": example_id,
                "problem": row.get("problem", ""),
                "type": row.get("type", ""),
                "level": row.get("level", ""),
                "retrieved_skills": [item.to_dict() for item in retrieved_skills],
                "prediction": prediction,
                "correct": comparison.is_correct,
                "error_type": comparison.error_type,
                "update_action": update_action,
                "update_applied": applied,
                "update_apply_reason": apply_reason,
                "update_proposal_error": proposal_error,
                "source": "gold_bootstrap_batched",
            }
        )

        return {
            "example_id": example_id,
            "problem": row.get("problem", ""),
            "type": row.get("type", ""),
            "level": row.get("level", ""),
            "gold_solution": row.get("solution", row.get("gold_solution", "")),
            "gold_final": comparison.gold.extracted_answer,
            "prompt": build_solver_prompt(row, self.store, retrieved_skills),
            "prediction": prediction,
            "pred_final": comparison.prediction.extracted_answer,
            "correct": comparison.is_correct,
            "error_type": comparison.error_type,
            "retrieved_skills": [item.to_dict() for item in retrieved_skills],
            "update_proposal": proposal_payload,
            "update_proposal_raw_text": raw_text,
            "update_proposal_error": proposal_error,
            "update_applied": applied,
            "update_apply_reason": apply_reason,
        }

    def bootstrap(self, rows: list[dict[str, Any]], output_jsonl: Path) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        family_descriptions = [family.to_dict() for family in self.store.families.values()]

        for chunk_index, chunk in enumerate(iter_chunks(rows, self.batch_size), start=1):
            predictions = [str(row.get("solution", row.get("gold_solution", ""))) for row in chunk]
            retrieved_batches = [self.retriever.retrieve(row) for row in chunk]
            prompts = [
                build_update_prompt(
                    row,
                    retrieved_skills,
                    self.store,
                    prediction=prediction,
                    families=family_descriptions,
                )
                for row, retrieved_skills, prediction in zip(chunk, retrieved_batches, predictions)
            ]
            generations = self.generator.generate(prompts, max_tokens=self.max_tokens, temperature=0.0, top_p=1.0)
            for row, prediction, retrieved_skills, raw_text in zip(chunk, predictions, retrieved_batches, generations):
                if not prediction:
                    continue
                outputs.append(self._apply_single(row, prediction, retrieved_skills, raw_text))

            write_jsonl(output_jsonl, outputs)
            print(f"  batch={chunk_index:>4d} rows_done={len(outputs):>6d} active_skills={len(self.store.active_skills())}")

        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-vLLM bootstrap SkillRL skill memory from ground-truth trajectories."
    )
    parser.add_argument("--run_root", type=Path, required=True, help="Output root for extracted memory and logs.")
    parser.add_argument(
        "--raw_run_root",
        type=Path,
        default=Path("runs/agents"),
        help="Root containing agent_sft_data/{env}/train_raw.jsonl.",
    )
    parser.add_argument("--extractor_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--extractor_lora_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=700)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    defaults_root = SKILL_RL_ROOT / "defaults"
    raw_root = args.raw_run_root / "agent_sft_data"
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

    generator = VLLMTextGenerator(
        model=args.extractor_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lora_path=args.extractor_lora_path,
    )
    bootstrapper = BatchedGoldBootstrapper(
        store,
        generator,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    total_outputs = 0
    for environment in ENVIRONMENT_ORDER:
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
        print(f"{environment:10s} rows={len(rows):>6d} batch_size={args.batch_size}")
        outputs = bootstrapper.bootstrap(rows, output_path)
        total_outputs += len(outputs)
        print(f"{environment:10s} bootstrapped={len(outputs):>6d} -> {output_path}")

    export_paper_style_skillbank(store, memory_root / "skillbank.json")
    print(f"\nActive skills after bootstrap: {len(store.active_skills())}")
    print(f"Bootstrap traces written: {total_outputs}")
    print(f"Memory written to: {memory_root}")
    print(f"Skill bank written to: {memory_root / 'skillbank.json'}")


if __name__ == "__main__":
    main()
