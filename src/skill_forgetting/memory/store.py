from __future__ import annotations

from pathlib import Path
from typing import Any

from skill_forgetting.memory.schema import (
    CanonicalSkill,
    SkillFamily,
    SkillRecord,
    SupportExample,
    ThresholdConfig,
    UpdateProposal,
)
from skill_forgetting.memory.utils import (
    append_jsonl,
    dedupe_preserve_order,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _default_metadata() -> dict[str, Any]:
    return {
        "step": 0,
        "thresholds": ThresholdConfig().to_dict(),
        "skill_aliases": {},
    }


def _normalize_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(payload or {})
    normalized = {
        "step": int(raw.get("step", 0)),
        "thresholds": ThresholdConfig.from_dict(raw.get("thresholds")).to_dict(),
        "skill_aliases": {str(key): str(value) for key, value in dict(raw.get("skill_aliases", {})).items()},
    }
    for key, value in raw.items():
        if key not in normalized:
            normalized[key] = value
    return normalized


def _merge_unique_text(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        merged.extend(group)
    return dedupe_preserve_order(merged)


class SkillMemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.skills_path = root / "skills.jsonl"
        self.supports_path = root / "supports.jsonl"
        self.families_path = root / "families.json"
        self.proposals_path = root / "proposals.jsonl"
        self.retrieval_events_path = root / "retrieval_events.jsonl"
        self.metadata_path = root / "metadata.json"
        self.skills: dict[str, SkillRecord] = {}
        self.families: dict[str, SkillFamily] = {}
        self.supports: list[SupportExample] = []
        self.metadata: dict[str, Any] = _default_metadata()
        self.load()

    def load(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.skills = {}
        for row in read_jsonl(self.skills_path):
            record = SkillRecord.from_dict(row)
            self.skills[record.canonical_skill.skill_id] = record

        self.supports = [SupportExample.from_dict(row) for row in read_jsonl(self.supports_path)]
        if self.families_path.exists():
            payload = read_json(self.families_path)
            families = payload.get("families", payload)
            self.families = {family["family_id"]: SkillFamily.from_dict(family) for family in families}
        else:
            self.families = {}

        if self.metadata_path.exists():
            self.metadata = _normalize_metadata(read_json(self.metadata_path))
        else:
            self.metadata = _default_metadata()

        self._refresh_support_counts()

    def _refresh_support_counts(self) -> None:
        support_totals: dict[str, int] = {}
        support_correct: dict[str, int] = {}
        for support in self.supports:
            support_totals[support.skill_id] = support_totals.get(support.skill_id, 0) + 1
            if support.correct:
                support_correct[support.skill_id] = support_correct.get(support.skill_id, 0) + 1
        for skill_id, record in self.skills.items():
            record.stats.support_count = support_totals.get(skill_id, 0)
            if record.stats.support_count:
                record.stats.source_correctness = support_correct.get(skill_id, 0) / record.stats.support_count

    def save(self) -> None:
        write_jsonl(self.skills_path, (record.to_dict() for record in self.iter_skills(include_inactive=True)))
        write_json(self.families_path, {"families": [family.to_dict() for family in self.families.values()]})
        write_json(self.metadata_path, self.metadata)

    def iter_skills(self, include_inactive: bool = False) -> list[SkillRecord]:
        records = sorted(self.skills.values(), key=lambda record: record.canonical_skill.skill_id)
        if include_inactive:
            return records
        return [record for record in records if record.canonical_skill.active]

    def active_skills(self) -> list[SkillRecord]:
        return self.iter_skills(include_inactive=False)

    def get_skill(self, skill_id: str) -> SkillRecord | None:
        resolved = self.resolve_alias(skill_id)
        return self.skills.get(resolved)

    def resolve_alias(self, skill_id: str) -> str:
        aliases = self.metadata.get("skill_aliases", {})
        return str(aliases.get(skill_id, skill_id))

    def set_thresholds(self, thresholds: ThresholdConfig) -> None:
        self.metadata["thresholds"] = thresholds.to_dict()
        self.save()

    def next_step(self) -> int:
        step = int(self.metadata.get("step", 0)) + 1
        self.metadata["step"] = step
        return step

    def initialize(
        self,
        families: list[SkillFamily],
        seed_skills: list[CanonicalSkill] | None = None,
        reset: bool = False,
    ) -> None:
        if reset:
            self.skills = {}
            self.supports = []
            self.metadata = _default_metadata()
            write_jsonl(self.supports_path, [])
            write_jsonl(self.proposals_path, [])
            write_jsonl(self.retrieval_events_path, [])
        self.families = {family.family_id: family for family in families}
        for skill in seed_skills or []:
            if skill.skill_id not in self.skills:
                self.skills[skill.skill_id] = SkillRecord(canonical_skill=skill)
        self.save()

    def add_skill(self, skill: CanonicalSkill, confidence: float = 0.0) -> SkillRecord:
        record = SkillRecord(canonical_skill=skill)
        record.stats.confidence = confidence
        record.stats.last_updated_step = int(self.metadata.get("step", 0))
        self.skills[skill.skill_id] = record
        self.save()
        return record

    def revise_skill(self, skill_id: str, patch: CanonicalSkill, confidence: float = 0.0) -> SkillRecord | None:
        record = self.get_skill(skill_id)
        if record is None:
            return None
        skill = record.canonical_skill
        if patch.goal:
            skill.goal = patch.goal
        if patch.applicability:
            skill.applicability = patch.applicability
        skill.keywords = _merge_unique_text(skill.keywords, patch.keywords)
        skill.structural_cues = _merge_unique_text(skill.structural_cues, patch.structural_cues)
        skill.output_format_cues = _merge_unique_text(skill.output_format_cues, patch.output_format_cues)
        skill.canonical_steps = _merge_unique_text(skill.canonical_steps, patch.canonical_steps)
        skill.failure_modes = _merge_unique_text(skill.failure_modes, patch.failure_modes)
        skill.compose_with = _merge_unique_text(skill.compose_with, patch.compose_with)
        record.stats.confidence = max(record.stats.confidence, confidence)
        record.stats.last_updated_step = int(self.metadata.get("step", 0))
        self.save()
        return record

    def mark_inactive(self, skill_id: str) -> None:
        record = self.get_skill(skill_id)
        if record is None:
            return
        record.canonical_skill.active = False
        record.stats.last_updated_step = int(self.metadata.get("step", 0))
        self.save()

    def merge_skills(self, target_skill_id: str, source_skill_id: str) -> SkillRecord | None:
        target = self.get_skill(target_skill_id)
        source = self.get_skill(source_skill_id)
        if target is None or source is None or target_skill_id == source_skill_id:
            return target
        target.canonical_skill.keywords = _merge_unique_text(
            target.canonical_skill.keywords, source.canonical_skill.keywords
        )
        target.canonical_skill.structural_cues = _merge_unique_text(
            target.canonical_skill.structural_cues, source.canonical_skill.structural_cues
        )
        target.canonical_skill.output_format_cues = _merge_unique_text(
            target.canonical_skill.output_format_cues, source.canonical_skill.output_format_cues
        )
        target.canonical_skill.canonical_steps = _merge_unique_text(
            target.canonical_skill.canonical_steps, source.canonical_skill.canonical_steps
        )
        target.canonical_skill.failure_modes = _merge_unique_text(
            target.canonical_skill.failure_modes, source.canonical_skill.failure_modes
        )
        target.canonical_skill.compose_with = _merge_unique_text(
            target.canonical_skill.compose_with, source.canonical_skill.compose_with
        )
        target.stats.times_retrieved += source.stats.times_retrieved
        target.stats.times_helped += source.stats.times_helped
        target.stats.times_hurt += source.stats.times_hurt
        target.stats.times_improved += source.stats.times_improved
        target.stats.times_regressed += source.stats.times_regressed
        target.stats.support_count += source.stats.support_count
        target.stats.confidence = max(target.stats.confidence, source.stats.confidence)
        target.aliases = _merge_unique_text(target.aliases, [source_skill_id], source.aliases)
        self.metadata.setdefault("skill_aliases", {})[source_skill_id] = target_skill_id
        source.canonical_skill.active = False
        self.save()
        return target

    def record_support(self, support: SupportExample) -> None:
        append_jsonl(self.supports_path, support.to_dict())
        self.supports.append(support)
        record = self.get_skill(support.skill_id)
        if record is not None:
            record.stats.support_count += 1
            if support.correct:
                correct_total = record.stats.source_correctness * max(0, record.stats.support_count - 1) + 1
            else:
                correct_total = record.stats.source_correctness * max(0, record.stats.support_count - 1)
            record.stats.source_correctness = correct_total / max(1, record.stats.support_count)
            record.stats.last_updated_step = support.step
            self.save()

    def record_retrieval_event(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.retrieval_events_path, payload)

    def record_proposal(self, proposal: UpdateProposal) -> None:
        append_jsonl(self.proposals_path, proposal.to_dict())

    def proposal_count(self, signature: str, action: str) -> int:
        count = 0
        for row in read_jsonl(self.proposals_path):
            if str(row.get("action")) != action:
                continue
            if str(row.get("metadata", {}).get("signature", "")) == signature:
                count += 1
        return count

    def all_supports_for(self, skill_id: str) -> list[SupportExample]:
        resolved = self.resolve_alias(skill_id)
        return [support for support in self.supports if self.resolve_alias(support.skill_id) == resolved]
