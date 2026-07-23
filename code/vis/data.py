"""Load canonical mental-model nodes and graph edges from PostgreSQL."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from mental_model_pipeline.canonical.db_models import (
    CanonicalMentalModelDB,
    CanonicalModelEdgeDB,
)
from mental_model_pipeline.database.connection import SessionLocal


def load_graph_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return embedded canonical models and edges as serialisable dictionaries."""

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
