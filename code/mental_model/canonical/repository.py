"""Perform database replacement, insertion, and update operations for canonical models and edges."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from mental_model.canonical.db_models import (
    CanonicalMentalModelDB,
    CanonicalModelEdgeDB,
)
from mental_model.canonical.schemas import CanonicalModelData


def investor_has_canonicals(session: Session, investor_id: str) -> bool:
    return (
        session.scalar(
            select(CanonicalMentalModelDB.canonical_id)
            .where(CanonicalMentalModelDB.investor_id == investor_id)
            .limit(1)
        )
        is not None
    )


def replace_investor_canonicals(
    session: Session,
    *,
    investor_id: str,
    models: list[CanonicalModelData],
    embeddings: list[list[float]],
    embedding_model: str,
    canonicalisation_model: str,
    prompt_version: str,
) -> None:
    if len(models) != len(embeddings):
        raise ValueError("Canonical model and embedding counts do not match.")

    # Edge rows are removed by the database ON DELETE CASCADE constraints.
    session.execute(
        delete(CanonicalMentalModelDB).where(
            CanonicalMentalModelDB.investor_id == investor_id
        )
    )

    for model, embedding in zip(models, embeddings, strict=True):
        session.add(
            CanonicalMentalModelDB(
                canonical_code=model.canonical_code,
                investor_id=model.investor_id,
                kind=model.kind.value,
                title=model.title,
                proposition=model.proposition,
                mechanism=model.mechanism,
                conditions=model.conditions,
                failure_conditions=model.failure_conditions,
                decision_implications=model.decision_implications,
                decision_stages=[
                    stage.value for stage in model.decision_stages
                ],
                contextual_regimes=model.contextual_regimes,
                supporting_fragment_codes=model.supporting_fragment_codes,
                evidence_confidence=model.evidence_confidence,
                investor_importance=model.investor_importance,
                base_weight=model.base_weight,
                primary_domain="unassigned",
                secondary_domains=[],
                concept_family=None,
                embedding=embedding,
                embedding_model=embedding_model,
                canonicalisation_model=canonicalisation_model,
                canonicalisation_prompt_version=prompt_version,
            )
        )

    session.flush()


def replace_hierarchy(
    session: Session,
    *,
    assignments: dict[str, Any],
    edges: list[dict[str, Any]],
) -> None:
    models = list(session.scalars(select(CanonicalMentalModelDB)))

    for model in models:
        assignment = assignments.get(model.canonical_code)

        if assignment is None:
            values = {
                "primary_domain": "unassigned",
                "secondary_domains": [],
                "concept_family": None,
            }
        else:
            values = {
                "primary_domain": assignment.primary_domain.value,
                "secondary_domains": [
                    domain.value for domain in assignment.secondary_domains
                ],
                "concept_family": assignment.concept_family,
            }

        session.execute(
            update(CanonicalMentalModelDB)
            .where(
                CanonicalMentalModelDB.canonical_id == model.canonical_id
            )
            .values(**values)
        )

    session.execute(delete(CanonicalModelEdgeDB))

    for edge in edges:
        session.add(CanonicalModelEdgeDB(**edge))

    session.flush()
