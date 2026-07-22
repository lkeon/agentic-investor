"""Calculate heuristic evidence confidence, investor importance, and base weights for canonical models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


EVIDENCE_DIRECTNESS_SCORE = {
    "directly_stated": 1.00,
    "strongly_implied": 0.70,
    "weakly_inferred": 0.35,
}


@dataclass(frozen=True)
class CanonicalScores:
    evidence_confidence: float
    investor_importance: float
    base_weight: float


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _saturating_count(count: int, scale: float) -> float:
    if count <= 0:
        return 0.0
    return _clamp(1.0 - math.exp(-count / scale))


def calculate_scores(
    fragments: Sequence[Any],
    *,
    cluster_coherence: float,
) -> CanonicalScores:
    if not fragments:
        raise ValueError("At least one supporting fragment is required.")

    directness = sum(
        EVIDENCE_DIRECTNESS_SCORE.get(
            _value(fragment.evidence_strength),
            0.50,
        )
        for fragment in fragments
    ) / len(fragments)

    fragment_count = len(fragments)
    document_count = len(
        {str(fragment.document_id) for fragment in fragments}
    )

    breadth = (
        0.70 * _saturating_count(document_count, 2.0)
        + 0.30 * _saturating_count(fragment_count, 3.0)
    )

    coherence = _clamp(cluster_coherence)

    evidence_confidence = _clamp(
        0.40 * directness
        + 0.30 * breadth
        + 0.30 * coherence
    )

    stages = {
        _value(stage)
        for fragment in fragments
        for stage in (fragment.decision_stages or [])
    }
    decision_stage_breadth = min(1.0, len(stages) / 4.0)

    investor_importance = _clamp(
        0.70 * breadth
        + 0.30 * decision_stage_breadth
    )

    base_weight = _clamp(
        0.70 * evidence_confidence
        + 0.30 * investor_importance
    )

    return CanonicalScores(
        evidence_confidence=round(evidence_confidence, 6),
        investor_importance=round(investor_importance, 6),
        base_weight=round(base_weight, 6),
    )
