"""
Skill retriever based on semantic embedding similarity.

Embedding model: BAAI/bge-small-en-v1.5
Strategy: encode question and each skill's task description,
          rank by cosine similarity, return top-k.

NOTE: All heavy imports (torch, sentence_transformers) are lazy
so that this module can be imported alongside vLLM without
triggering premature CUDA initialization.
"""
from __future__ import annotations

import os
from typing import Any

from skill_forgetting.memory.schema import RetrievedSkill, SkillRecord, ThresholdConfig
from skill_forgetting.memory.store import SkillMemoryStore

EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"

_model = None


def _load_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        model_name_or_path = os.environ.get("SKILL_EMBEDDING_MODEL", EMBEDDING_MODEL_ID)
        _model = SentenceTransformer(model_name_or_path)
    return _model


def encode_texts(texts: list[str]):
    import numpy as np
    model = _load_embedding_model()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def skill_description(record: SkillRecord) -> str:
    """Build a concise text representation of a skill for embedding."""
    s = record.canonical_skill
    parts = [s.goal, s.applicability]
    if s.canonical_steps:
        parts.append(" ".join(s.canonical_steps))
    return "\n".join(p for p in parts if p)


class HybridSkillRetriever:
    """Semantic embedding retriever (replaces the old weighted-score approach).

    Keeps the class name for backward compatibility with pipeline / updater / dataset_builder.
    """

    def __init__(self, store: SkillMemoryStore, thresholds: ThresholdConfig | None = None) -> None:
        self.store = store
        self.thresholds = thresholds or ThresholdConfig.from_dict(store.metadata.get("thresholds"))

    def _ensure_skill_embeddings(self):
        """Encode all active skills; returns (records, embeddings)."""
        import numpy as np
        records = self.store.active_skills()
        if not records:
            return [], np.empty((0, 0))
        texts = [skill_description(r) for r in records]
        embeddings = encode_texts(texts)
        return records, embeddings

    def skill_text(self, record: SkillRecord) -> str:
        """Public helper kept for updater.scan_for_merge_candidates."""
        return skill_description(record)

    def retrieve(self, row: dict[str, Any], top_k: int | None = None) -> list[RetrievedSkill]:
        import numpy as np

        top_k = self.thresholds.top_k if top_k is None else top_k
        if top_k <= 0:
            return []

        records, skill_embeddings = self._ensure_skill_embeddings()
        if len(records) == 0:
            return []

        question = str(row.get("problem", ""))
        if not question.strip():
            return []

        q_emb = encode_texts([question])                       # (1, dim)
        scores = (q_emb @ skill_embeddings.T).squeeze(0)       # (n_skills,)

        k = min(top_k, len(records))
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[RetrievedSkill] = []
        for rank, idx in enumerate(top_indices, start=1):
            rec = records[idx]
            s = rec.canonical_skill
            results.append(
                RetrievedSkill(
                    skill_id=s.skill_id,
                    skill_family=s.skill_family,
                    total_score=float(scores[idx]),
                    rank=rank,
                    semantic_score=float(scores[idx]),
                )
            )
        return results
