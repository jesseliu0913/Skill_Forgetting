from __future__ import annotations

from pathlib import Path
from typing import Any

from skill_forgetting.memory.extractor import InvalidProposalError, QwenSkillExtractor, SkillExtractor
from skill_forgetting.memory.llm import TextGenerator
from skill_forgetting.memory.prompts import build_solver_prompt
from skill_forgetting.memory.retriever import HybridSkillRetriever
from skill_forgetting.memory.schema import RetrievedSkill, ThresholdConfig
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.updater import UpdateManager
from skill_forgetting.memory.utils import (
    AnswerComparison,
    compare_answers,
    build_example_id,
    read_jsonl,
    write_jsonl,
)


class DynamicSkillMemoryPipeline:
    def __init__(
        self,
        store: SkillMemoryStore,
        *,
        thresholds: ThresholdConfig | None = None,
        solver: TextGenerator | None = None,
        extractor: SkillExtractor | None = None,
    ) -> None:
        self.store = store
        self.thresholds = thresholds or ThresholdConfig.from_dict(store.metadata.get("thresholds"))
        self.retriever = HybridSkillRetriever(store, thresholds=self.thresholds)
        self.solver = solver
        self.extractor = extractor
        self.updater = UpdateManager(store, thresholds=self.thresholds)

    @classmethod
    def with_qwen_extractor(
        cls,
        store: SkillMemoryStore,
        *,
        solver: TextGenerator | None,
        extractor_generator: TextGenerator,
        thresholds: ThresholdConfig | None = None,
    ) -> "DynamicSkillMemoryPipeline":
        extractor = QwenSkillExtractor(extractor_generator, store.families)
        return cls(store, thresholds=thresholds, solver=solver, extractor=extractor)

    def _require_extractor(self) -> SkillExtractor:
        if self.extractor is None:
            raise ValueError("DynamicSkillMemoryPipeline requires an explicit extractor for update-bearing operations.")
        return self.extractor

    def _prediction_lookup(self, prediction_jsonl: Path | None) -> dict[str, dict[str, Any]]:
        if prediction_jsonl is None:
            return {}
        rows = read_jsonl(prediction_jsonl)
        lookup: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = build_example_id(str(row.get("problem", "")), str(row.get("type", "")), str(row.get("level", "")))
            lookup[key] = row
        return lookup

    def _evaluate_correctness(self, row: dict[str, Any], prediction: str) -> AnswerComparison:
        gold_solution = str(row.get("solution", row.get("gold_solution", "")))
        return compare_answers(prediction, gold_solution)

    def _row_example_id(self, row: dict[str, Any]) -> str:
        return build_example_id(str(row.get("problem", "")), str(row.get("type", "")), str(row.get("level", "")))

    def _attach_prediction_metadata(
        self,
        row: dict[str, Any],
        prediction_row: dict[str, Any],
    ) -> dict[str, Any]:
        if not prediction_row:
            return row
        merged = dict(row)
        if "baseline_correct" not in merged and isinstance(prediction_row.get("baseline_correct"), bool):
            merged["baseline_correct"] = prediction_row["baseline_correct"]
        return merged

    def _process_single(
        self,
        row: dict[str, Any],
        prediction: str,
        retrieved_skills: list[RetrievedSkill],
        *,
        source: str,
        track_retrieval_feedback: bool = True,
    ) -> dict[str, Any]:
        comparison = self._evaluate_correctness(row, prediction)
        correct = comparison.is_correct
        gold_final = comparison.gold.extracted_answer
        error_type = comparison.error_type
        example_id = self._row_example_id(row)
        baseline_correct = row.get("baseline_correct")
        if isinstance(baseline_correct, bool):
            baseline_value = baseline_correct
        else:
            baseline_value = None
        if track_retrieval_feedback:
            self.updater.record_retrieval_feedback(retrieved_skills, correct=correct, baseline_correct=baseline_value)

        proposal = None
        proposal_payload: dict[str, Any] | None = None
        proposal_error = ""
        update_action = "extractor_error"
        applied = False
        apply_reason = "extractor proposal unavailable"

        try:
            proposal = self._require_extractor().propose(row, retrieved_skills, self.store, prediction, correct=correct)
            proposal.example_id = proposal.example_id or example_id
            proposal_payload = proposal.to_dict()
            update_action = proposal.action
            applied, apply_reason = self.updater.apply_proposal(proposal, row, prediction, correct=correct)
        except InvalidProposalError as exc:
            proposal_error = str(exc)
            proposal_payload = exc.payload or None
            apply_reason = f"extractor proposal rejected: {exc}"

        retrieval_log_row = {
            "example_id": example_id,
            "problem": row.get("problem", ""),
            "type": row.get("type", ""),
            "level": row.get("level", ""),
            "retrieved_skills": [item.to_dict() for item in retrieved_skills],
            "prediction": prediction,
            "correct": correct,
            "error_type": error_type,
            "update_action": update_action,
            "update_applied": applied,
            "update_apply_reason": apply_reason,
            "update_proposal_error": proposal_error,
            "source": source,
        }
        self.store.record_retrieval_event(retrieval_log_row)

        prompt = build_solver_prompt(row, self.store, retrieved_skills)
        return {
            "example_id": example_id,
            "problem": row.get("problem", ""),
            "type": row.get("type", ""),
            "level": row.get("level", ""),
            "gold_solution": row.get("solution", row.get("gold_solution", "")),
            "gold_final": gold_final,
            "prompt": prompt,
            "prediction": prediction,
            "pred_final": comparison.prediction.extracted_answer,
            "correct": correct,
            "error_type": error_type,
            "retrieved_skills": [item.to_dict() for item in retrieved_skills],
            "update_proposal": proposal_payload,
            "update_proposal_error": proposal_error,
            "update_applied": applied,
            "update_apply_reason": apply_reason,
        }

    def warm_extract(
        self,
        input_jsonl: Path,
        output_jsonl: Path,
        *,
        prediction_jsonl: Path | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = read_jsonl(input_jsonl)
        prediction_lookup = self._prediction_lookup(prediction_jsonl)
        outputs: list[dict[str, Any]] = []
        for row in rows[:limit]:
            example_id = self._row_example_id(row)
            prediction_row = prediction_lookup.get(example_id, {})
            prediction = str(prediction_row.get("prediction", row.get("prediction", "")))
            if not prediction:
                continue
            row = self._attach_prediction_metadata(row, prediction_row)
            retrieved_skills = self.retriever.retrieve(row)
            outputs.append(self._process_single(row, prediction, retrieved_skills, source="warm_extract"))
        write_jsonl(output_jsonl, outputs)
        return outputs

    def bootstrap_from_gold(
        self,
        input_jsonl: Path,
        output_jsonl: Path,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = read_jsonl(input_jsonl)
        outputs: list[dict[str, Any]] = []
        for row in rows[:limit]:
            prediction = str(row.get("solution", row.get("gold_solution", "")))
            if not prediction:
                continue
            retrieved_skills = self.retriever.retrieve(row)
            outputs.append(
                self._process_single(
                    row,
                    prediction,
                    retrieved_skills,
                    source="gold_bootstrap",
                    track_retrieval_feedback=False,
                )
            )
        write_jsonl(output_jsonl, outputs)
        return outputs

    def run_batch(
        self,
        input_jsonl: Path,
        output_jsonl: Path,
        *,
        prediction_jsonl: Path | None = None,
        limit: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[dict[str, Any]]:
        rows = read_jsonl(input_jsonl)
        prediction_lookup = self._prediction_lookup(prediction_jsonl)
        outputs: list[dict[str, Any]] = []
        for row in rows[:limit]:
            example_id = self._row_example_id(row)
            prediction_row = prediction_lookup.get(example_id, {})
            row = self._attach_prediction_metadata(row, prediction_row)
            retrieved_skills = self.retriever.retrieve(row)
            prompt = build_solver_prompt(row, self.store, retrieved_skills)
            if prediction_row.get("prediction"):
                prediction = str(prediction_row["prediction"])
                source = "prediction_jsonl"
            elif row.get("prediction"):
                prediction = str(row["prediction"])
                source = "inline_prediction"
            elif self.solver is not None:
                prediction = self.solver.generate(
                    [prompt],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )[0]
                source = "solver"
            else:
                raise ValueError("run_batch needs either --prediction_jsonl, inline prediction, or a solver model.")
            outputs.append(self._process_single(row, prediction, retrieved_skills, source=source))
        write_jsonl(output_jsonl, outputs)
        return outputs
