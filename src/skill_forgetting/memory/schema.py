from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _copy_list(values: list[str] | None) -> list[str]:
    return list(values or [])


def _copy_dict(values: dict[str, float] | None) -> dict[str, float]:
    return dict(values or {})


@dataclass
class SkillFamily:
    family_id: str
    short_name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    structural_cues: list[str] = field(default_factory=list)
    output_format_cues: list[str] = field(default_factory=list)
    type_aliases: list[str] = field(default_factory=list)
    default_steps: list[str] = field(default_factory=list)
    default_failure_modes: list[str] = field(default_factory=list)
    priority: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillFamily":
        return cls(
            family_id=str(payload["family_id"]),
            short_name=str(payload.get("short_name", payload["family_id"])),
            description=str(payload.get("description", "")),
            keywords=_copy_list(payload.get("keywords")),
            structural_cues=_copy_list(payload.get("structural_cues")),
            output_format_cues=_copy_list(payload.get("output_format_cues")),
            type_aliases=_copy_list(payload.get("type_aliases")),
            default_steps=_copy_list(payload.get("default_steps")),
            default_failure_modes=_copy_list(payload.get("default_failure_modes")),
            priority=float(payload.get("priority", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalSkill:
    skill_id: str
    skill_family: str
    goal: str
    applicability: str
    keywords: list[str] = field(default_factory=list)
    structural_cues: list[str] = field(default_factory=list)
    output_format_cues: list[str] = field(default_factory=list)
    canonical_steps: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    compose_with: list[str] = field(default_factory=list)
    active: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalSkill":
        return cls(
            skill_id=str(payload["skill_id"]),
            skill_family=str(payload["skill_family"]),
            goal=str(payload.get("goal", "")),
            applicability=str(payload.get("applicability", "")),
            keywords=_copy_list(payload.get("keywords")),
            structural_cues=_copy_list(payload.get("structural_cues")),
            output_format_cues=_copy_list(payload.get("output_format_cues")),
            canonical_steps=_copy_list(payload.get("canonical_steps")),
            failure_modes=_copy_list(payload.get("failure_modes")),
            compose_with=_copy_list(payload.get("compose_with")),
            active=bool(payload.get("active", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillStats:
    times_retrieved: int = 0
    times_helped: int = 0
    times_hurt: int = 0
    times_improved: int = 0
    times_regressed: int = 0
    last_updated_step: int = 0
    confidence: float = 0.0
    source_correctness: float = 0.0
    support_count: int = 0
    retrieval_precision: float = 0.0
    downstream_gain: float = 0.0
    interference_rate: float = 0.0
    utility_score: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SkillStats":
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (payload or {}).items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillRecord:
    canonical_skill: CanonicalSkill
    stats: SkillStats = field(default_factory=SkillStats)
    aliases: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillRecord":
        skill_payload = payload.get("canonical_skill", payload)
        return cls(
            canonical_skill=CanonicalSkill.from_dict(skill_payload),
            stats=SkillStats.from_dict(payload.get("stats")),
            aliases=_copy_list(payload.get("aliases")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_skill": self.canonical_skill.to_dict(),
            "stats": self.stats.to_dict(),
            "aliases": list(self.aliases),
        }


@dataclass
class SupportExample:
    skill_id: str
    example_id: str
    dataset: str = ""
    problem: str = ""
    problem_type: str = ""
    level: str = ""
    correct: bool = False
    difficulty: str = ""
    gold_solution: str = ""
    prediction: str = ""
    error_type: str = ""
    source: str = "inference"
    step: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SupportExample":
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryBundle:
    semantic_query: str
    cue_tokens: list[str]
    structural_cues: list[str]
    format_cues: list[str]
    family_priors: dict[str, float]
    problem_type: str = ""
    level: str = ""


@dataclass
class RetrievedSkill:
    skill_id: str
    skill_family: str
    total_score: float
    rank: int
    semantic_score: float
    keyword_match: float = 0.0
    structural_match: float = 0.0
    cue_match: float = 0.0
    format_match: float = 0.0
    family_prior: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    matched_structural_cues: list[str] = field(default_factory=list)
    matched_format_cues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UpdateProposal:
    action: str
    target_skill_id: str | None
    proposed_skill: CanonicalSkill | None
    reason: str
    confidence: float
    example_id: str = ""
    observed_failure_modes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UpdateProposal":
        proposed = payload.get("proposed_skill")
        return cls(
            action=str(payload["action"]),
            target_skill_id=payload.get("target_skill_id"),
            proposed_skill=CanonicalSkill.from_dict(proposed) if isinstance(proposed, dict) else None,
            reason=str(payload.get("reason", "")),
            confidence=float(payload.get("confidence", 0.0)),
            example_id=str(payload.get("example_id", "")),
            observed_failure_modes=_copy_list(payload.get("observed_failure_modes")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_skill_id": self.target_skill_id,
            "proposed_skill": self.proposed_skill.to_dict() if self.proposed_skill else None,
            "reason": self.reason,
            "confidence": self.confidence,
            "example_id": self.example_id,
            "observed_failure_modes": list(self.observed_failure_modes),
            "metadata": dict(self.metadata),
        }


@dataclass
class ThresholdConfig:
    top_k: int = 2
    max_family_candidates: int = 2
    alpha: float = 0.45
    beta: float = 0.25
    gamma: float = 0.10
    delta: float = 0.20
    reuse_threshold: float = 0.62
    revise_threshold: float = 0.42
    create_threshold: float = 0.72
    drop_threshold: float = 0.10
    merge_similarity_threshold: float = 0.88
    min_support_for_create: int = 2
    min_retrievals_for_drop: int = 4
    min_retrievals_for_merge: int = 2
    revise_on_two_consecutive_declines: bool = True
    utility_source_weight: float = 0.30
    utility_support_weight: float = 0.20
    utility_precision_weight: float = 0.20
    utility_gain_weight: float = 0.20
    utility_interference_weight: float = 0.10

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ThresholdConfig":
        return cls(**dict(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
