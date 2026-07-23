"""Load safe MMC, MMF, and graph data from PostgreSQL."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select

from mental_model_pipeline.canonical.db_models import (
    CanonicalMentalModelDB,
    CanonicalModelEdgeDB,
)
from mental_model_pipeline.database.connection import SessionLocal
from mental_model_pipeline.fragments.db_models import MentalModelFragmentDB


def _canonical_summary(
    model: CanonicalMentalModelDB,
) -> dict[str, Any]:
    """Return safe generated fields for an MMC association."""

    return {
        "canonical_code": model.canonical_code,
        "investor_id": model.investor_id,
        "title": model.title,
        "proposition": model.proposition,
        "primary_domain": model.primary_domain,
        "concept_family": model.concept_family,
        "base_weight": float(model.base_weight),
    }


def load_graph_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return embedded canonical models and stored graph edges."""

    with SessionLocal() as session:
        models = list(
            session.scalars(
                select(CanonicalMentalModelDB)
                .where(CanonicalMentalModelDB.embedding.is_not(None))
                .order_by(
                    CanonicalMentalModelDB.investor_id,
                    CanonicalMentalModelDB.canonical_code,
                )
            )
        )

        edges = list(
            session.scalars(
                select(CanonicalModelEdgeDB).order_by(
                    CanonicalModelEdgeDB.relation_confidence.desc(),
                    CanonicalModelEdgeDB.relation_strength.desc(),
                )
            )
        )

    nodes = [
        {
            "canonical_id": str(model.canonical_id),
            "canonical_code": model.canonical_code,
            "investor_id": model.investor_id,
            "kind": model.kind,
            "title": model.title,
            "proposition": model.proposition,
            "mechanism": list(model.mechanism or []),
            "conditions": list(model.conditions or []),
            "failure_conditions": list(model.failure_conditions or []),
            "decision_implications": list(model.decision_implications or []),
            "decision_stages": list(model.decision_stages or []),
            "contextual_regimes": list(model.contextual_regimes or []),
            "supporting_fragment_codes": list(
                model.supporting_fragment_codes or []
            ),
            "evidence_confidence": float(model.evidence_confidence),
            "investor_importance": float(model.investor_importance),
            "base_weight": float(model.base_weight),
            "primary_domain": model.primary_domain,
            "secondary_domains": list(model.secondary_domains or []),
            "concept_family": model.concept_family,
            "embedding": [float(value) for value in model.embedding],
        }
        for model in models
    ]

    visible_ids = {node["canonical_id"] for node in nodes}

    graph_edges = [
        {
            "edge_id": str(edge.edge_id),
            "source_canonical_id": str(edge.source_canonical_id),
            "target_canonical_id": str(edge.target_canonical_id),
            "relation_type": edge.relation_type,
            "relation_strength": float(edge.relation_strength),
            "relation_confidence": float(edge.relation_confidence),
            "candidate_similarity": (
                float(edge.candidate_similarity)
                if edge.candidate_similarity is not None
                else None
            ),
            "scope": edge.scope,
        }
        for edge in edges
        if (
            str(edge.source_canonical_id) in visible_ids
            and str(edge.target_canonical_id) in visible_ids
        )
    ]

    return nodes, graph_edges


def load_supporting_fragments(
    fragment_codes: Sequence[str],
) -> list[dict[str, Any]]:
    """Load generated MMF fields without source quotations or references."""

    ordered_codes = [code for code in fragment_codes if code]

    if not ordered_codes:
        return []

    with SessionLocal() as session:
        statement = select(
            MentalModelFragmentDB.fragment_code,
            MentalModelFragmentDB.investor_id,
            MentalModelFragmentDB.kind,
            MentalModelFragmentDB.title,
            MentalModelFragmentDB.proposition,
            MentalModelFragmentDB.mechanism,
            MentalModelFragmentDB.conditions,
            MentalModelFragmentDB.failure_conditions,
            MentalModelFragmentDB.decision_implications,
            MentalModelFragmentDB.decision_stages,
            MentalModelFragmentDB.contextual_regimes,
            MentalModelFragmentDB.evidence_strength,
        ).where(
            MentalModelFragmentDB.fragment_code.in_(ordered_codes)
        )

        rows = session.execute(statement).all()

    fragments_by_code = {
        row.fragment_code: {
            "fragment_code": row.fragment_code,
            "investor_id": row.investor_id,
            "kind": row.kind,
            "title": row.title,
            "proposition": row.proposition,
            "mechanism": list(row.mechanism or []),
            "conditions": list(row.conditions or []),
            "failure_conditions": list(row.failure_conditions or []),
            "decision_implications": list(
                row.decision_implications or []
            ),
            "decision_stages": list(row.decision_stages or []),
            "contextual_regimes": list(row.contextual_regimes or []),
            "evidence_strength": row.evidence_strength,
        }
        for row in rows
    }

    return [
        fragments_by_code[code]
        for code in ordered_codes
        if code in fragments_by_code
    ]


def load_mmf_network_data() -> list[dict[str, Any]]:
    """Return embedded MMFs and their associated canonical models."""

    with SessionLocal() as session:
        fragments = list(
            session.scalars(
                select(MentalModelFragmentDB)
                .where(MentalModelFragmentDB.embedding.is_not(None))
                .order_by(
                    MentalModelFragmentDB.investor_id,
                    MentalModelFragmentDB.fragment_code,
                )
            )
        )

        canonical_models = list(
            session.scalars(
                select(CanonicalMentalModelDB).order_by(
                    CanonicalMentalModelDB.investor_id,
                    CanonicalMentalModelDB.canonical_code,
                )
            )
        )

    associated_mmcs: dict[str, list[dict[str, Any]]] = {}

    for model in canonical_models:
        summary = _canonical_summary(model)

        for fragment_code in model.supporting_fragment_codes or []:
            associated_mmcs.setdefault(
                fragment_code,
                [],
            ).append(summary)

    return [
        {
            "fragment_code": fragment.fragment_code,
            "investor_id": fragment.investor_id,
            "kind": fragment.kind,
            "title": fragment.title,
            "proposition": fragment.proposition,
            "mechanism": list(fragment.mechanism or []),
            "conditions": list(fragment.conditions or []),
            "failure_conditions": list(
                fragment.failure_conditions or []
            ),
            "decision_implications": list(
                fragment.decision_implications or []
            ),
            "decision_stages": list(fragment.decision_stages or []),
            "contextual_regimes": list(
                fragment.contextual_regimes or []
            ),
            "evidence_strength": fragment.evidence_strength,
            "associated_mmcs": associated_mmcs.get(
                fragment.fragment_code,
                [],
            ),
            "embedding": [
                float(value)
                for value in fragment.embedding
            ],
        }
        for fragment in fragments
    ]
