from __future__ import annotations

from typing import Any

from skill_forgetting.memory.retriever import HybridSkillRetriever, encode_texts, skill_description
from skill_forgetting.memory.schema import CanonicalSkill, RetrievedSkill, SupportExample, ThresholdConfig, UpdateProposal
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.utils import build_example_id


class UpdateManager:
    def __init__(self, store: SkillMemoryStore, thresholds: ThresholdConfig | None = None) -> None:
        self.store = store
        self.thresholds = thresholds or ThresholdConfig.from_dict(store.metadata.get("thresholds"))

    def recompute_utility(self, skill_id: str) -> None:
        record = self.store.get_skill(skill_id)
        if record is None:
            return
        stats = record.stats
        stats.retrieval_precision = stats.times_helped / max(1, stats.times_retrieved)
        total_gain_trials = max(1, stats.times_improved + stats.times_regressed)
        stats.downstream_gain = max(0.0, (stats.times_improved - stats.times_regressed) / total_gain_trials)
        stats.interference_rate = stats.times_hurt / max(1, stats.times_retrieved)
        normalized_support = min(1.0, stats.support_count / 5.0)
        stats.utility_score = (
            self.thresholds.utility_source_weight * stats.source_correctness
            + self.thresholds.utility_support_weight * normalized_support
            + self.thresholds.utility_precision_weight * stats.retrieval_precision
            + self.thresholds.utility_gain_weight * stats.downstream_gain
            - self.thresholds.utility_interference_weight * stats.interference_rate
        )
        self.store.save()

    def record_retrieval_feedback(
        self,
        retrieved_skills: list[RetrievedSkill],
        *,
        correct: bool,
        baseline_correct: bool | None = None,
    ) -> None:
        for retrieved in retrieved_skills:
            record = self.store.get_skill(retrieved.skill_id)
            if record is None:
                continue
            record.stats.times_retrieved += 1
            if correct:
                record.stats.times_helped += 1
            else:
                record.stats.times_hurt += 1
            if baseline_correct is not None:
                if correct and not baseline_correct:
                    record.stats.times_improved += 1
                elif baseline_correct and not correct:
                    record.stats.times_regressed += 1
            self.recompute_utility(retrieved.skill_id)

    def record_support(self, skill_id: str, row: dict[str, Any], prediction: str, correct: bool, source: str) -> None:
        step = int(self.store.metadata.get("step", 0))
        support = SupportExample(
            skill_id=skill_id,
            example_id=build_example_id(str(row.get("problem", "")), str(row.get("type", "")), str(row.get("level", ""))),
            dataset=str(row.get("dataset", "")),
            problem=str(row.get("problem", "")),
            problem_type=str(row.get("type", "")),
            level=str(row.get("level", "")),
            correct=correct,
            difficulty=str(row.get("difficulty", row.get("level", ""))),
            gold_solution=str(row.get("solution", row.get("gold_solution", ""))),
            prediction=prediction,
            error_type=str(row.get("error_type", "")),
            source=source,
            step=step,
        )
        self.store.record_support(support)
        self.recompute_utility(skill_id)

    def apply_proposal(
        self,
        proposal: UpdateProposal,
        row: dict[str, Any],
        prediction: str,
        *,
        correct: bool,
    ) -> tuple[bool, str]:
        self.store.record_proposal(proposal)
        self.store.next_step()
        self.store.save()

        if proposal.action == "reuse" and proposal.target_skill_id:
            if correct:
                self.record_support(proposal.target_skill_id, row, prediction, correct=True, source="reuse")
            return True, "reused existing skill"

        if proposal.action == "revise" and proposal.target_skill_id and proposal.proposed_skill is not None:
            if proposal.confidence >= self.thresholds.revise_threshold:
                self.store.revise_skill(proposal.target_skill_id, proposal.proposed_skill, confidence=proposal.confidence)
                if correct:
                    self.record_support(proposal.target_skill_id, row, prediction, correct=True, source="revise")
                return True, "revised existing skill"
            return False, "revise proposal below threshold"

        if proposal.action == "create" and proposal.proposed_skill is not None:
            signature = str(proposal.metadata.get("signature", proposal.proposed_skill.skill_id))
            count = self.store.proposal_count(signature, "create")
            if correct and proposal.confidence >= self.thresholds.create_threshold and count >= self.thresholds.min_support_for_create:
                self.store.add_skill(proposal.proposed_skill, confidence=proposal.confidence)
                self.record_support(proposal.proposed_skill.skill_id, row, prediction, correct=True, source="create")
                return True, "created new skill"
            return False, "create proposal waiting for more support or confidence"

        if proposal.action == "drop" and proposal.target_skill_id:
            record = self.store.get_skill(proposal.target_skill_id)
            if record and record.stats.utility_score < self.thresholds.drop_threshold:
                self.store.mark_inactive(proposal.target_skill_id)
                return True, "dropped low-utility skill"
            return False, "drop proposal rejected by utility threshold"

        if proposal.action == "merge" and proposal.target_skill_id and proposal.proposed_skill is not None:
            self.store.merge_skills(proposal.target_skill_id, proposal.proposed_skill.skill_id)
            return True, "merged skills"

        return False, "proposal not applied"

    def scan_for_merge_candidates(self) -> list[tuple[str, str, float]]:
        records = self.store.active_skills()
        if len(records) < 2:
            return []
        texts = [skill_description(r) for r in records]
        embeddings = encode_texts(texts)
        sim_matrix = embeddings @ embeddings.T

        merge_candidates: list[tuple[str, str, float]] = []
        for i, left in enumerate(records):
            for j in range(i + 1, len(records)):
                right = records[j]
                if left.canonical_skill.skill_family != right.canonical_skill.skill_family:
                    continue
                similarity = float(sim_matrix[i, j])
                if similarity >= self.thresholds.merge_similarity_threshold:
                    merge_candidates.append((left.canonical_skill.skill_id, right.canonical_skill.skill_id, similarity))
        return merge_candidates

    def merge_top_duplicates(self) -> list[str]:
        actions: list[str] = []
        for left_id, right_id, similarity in self.scan_for_merge_candidates():
            left = self.store.get_skill(left_id)
            right = self.store.get_skill(right_id)
            if left is None or right is None:
                continue
            if left.stats.times_retrieved < self.thresholds.min_retrievals_for_merge:
                preferred, other = right_id, left_id
            else:
                preferred, other = left_id, right_id
            self.store.merge_skills(preferred, other)
            actions.append(f"merged {other} into {preferred} (sim={similarity:.3f})")
        return actions

    def drop_low_utility_skills(self) -> list[str]:
        actions: list[str] = []
        for record in self.store.active_skills():
            stats = record.stats
            if stats.times_retrieved < self.thresholds.min_retrievals_for_drop:
                continue
            self.recompute_utility(record.canonical_skill.skill_id)
            if stats.utility_score < self.thresholds.drop_threshold:
                self.store.mark_inactive(record.canonical_skill.skill_id)
                actions.append(f"dropped {record.canonical_skill.skill_id}")
        return actions
