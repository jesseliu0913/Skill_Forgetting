from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from skill_forgetting.memory.llm import TextGenerator, parse_json_object
from skill_forgetting.memory.prompts import build_update_prompt
from skill_forgetting.memory.schema import RetrievedSkill, SkillFamily, UpdateProposal
from skill_forgetting.memory.store import SkillMemoryStore
from skill_forgetting.memory.utils import build_example_id, slugify


ALLOWED_UPDATE_ACTIONS = {"reuse", "revise", "create", "merge", "drop"}


class InvalidProposalError(ValueError):
    def __init__(self, message: str, *, raw_text: str = "", payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.payload = dict(payload or {})


class SkillExtractor(ABC):
    @abstractmethod
    def propose(
        self,
        row: dict[str, Any],
        retrieved_skills: list[RetrievedSkill],
        store: SkillMemoryStore,
        prediction: str,
        *,
        correct: bool,
    ) -> UpdateProposal:
        raise NotImplementedError


class QwenSkillExtractor(SkillExtractor):
    def __init__(self, generator: TextGenerator, families: dict[str, SkillFamily]) -> None:
        if not families:
            raise ValueError("QwenSkillExtractor requires at least one skill family.")
        self.generator = generator
        self.families = families

    def _normalize_skill_id(self, raw_skill_id: str) -> str:
        skill_id = slugify(raw_skill_id)
        if not skill_id:
            raise ValueError("Model returned an empty skill_id.")
        return skill_id

    def _validate_proposal(self, proposal: UpdateProposal, store: SkillMemoryStore) -> None:
        proposal.action = proposal.action.strip().lower()
        if proposal.target_skill_id is not None:
            proposal.target_skill_id = str(proposal.target_skill_id).strip() or None

        if proposal.action not in ALLOWED_UPDATE_ACTIONS:
            raise ValueError(f"Unsupported update action: {proposal.action}")

        requires_target = {"reuse", "revise", "drop", "merge"}
        requires_skill = {"revise", "create", "merge"}

        if proposal.action in requires_target and not proposal.target_skill_id:
            raise ValueError(f"{proposal.action} proposals must include target_skill_id.")
        if proposal.target_skill_id and store.get_skill(proposal.target_skill_id) is None:
            raise ValueError(f"Unknown target skill: {proposal.target_skill_id}")

        if proposal.action in requires_skill:
            if proposal.proposed_skill is None:
                raise ValueError(f"{proposal.action} proposals must include proposed_skill.")
            proposal.proposed_skill.skill_id = self._normalize_skill_id(proposal.proposed_skill.skill_id)
            family_id = proposal.proposed_skill.skill_family.strip()
            if not family_id:
                raise ValueError("proposed_skill.skill_family must be provided.")
            proposal.proposed_skill.skill_family = family_id
            if family_id not in self.families:
                raise ValueError(f"Unknown skill family: {family_id}")
        elif proposal.proposed_skill is not None:
            raise ValueError(f"{proposal.action} proposals must not include proposed_skill.")

        if proposal.action == "create" and proposal.proposed_skill is not None:
            if store.get_skill(proposal.proposed_skill.skill_id) is not None:
                raise ValueError(f"create proposal reuses existing skill_id: {proposal.proposed_skill.skill_id}")

        if proposal.action == "merge" and proposal.proposed_skill is not None:
            source_skill_id = proposal.proposed_skill.skill_id
            if source_skill_id == proposal.target_skill_id:
                raise ValueError("merge proposal must reference two different skills.")
            if store.get_skill(source_skill_id) is None:
                raise ValueError(f"Unknown merge source skill: {source_skill_id}")

    def propose(
        self,
        row: dict[str, Any],
        retrieved_skills: list[RetrievedSkill],
        store: SkillMemoryStore,
        prediction: str,
        *,
        correct: bool,
    ) -> UpdateProposal:
        family_descriptions = [family.to_dict() for family in self.families.values()]
        prompt = build_update_prompt(row, retrieved_skills, store, prediction=prediction, families=family_descriptions)
        text = self.generator.generate([prompt], max_tokens=700, temperature=0.0, top_p=1.0)[0]
        payload: dict[str, Any] | None = None
        try:
            payload = parse_json_object(text)
            proposal = UpdateProposal.from_dict(payload)
            self._validate_proposal(proposal, store)
        except Exception as exc:
            raise InvalidProposalError(
                f"Invalid extractor proposal: {exc}",
                raw_text=text,
                payload=payload,
            ) from exc
        if not proposal.example_id:
            proposal.example_id = build_example_id(
                str(row.get("problem", "")),
                str(row.get("type", "")),
                str(row.get("level", "")),
            )
        return proposal
