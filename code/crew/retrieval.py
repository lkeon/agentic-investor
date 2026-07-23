"""Retrieve canonical mental models separately for each investor."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import numpy as np
from sqlalchemy import select

from mental_model.canonical.db_models import (
    CanonicalMentalModelDB,
    CanonicalModelEdgeDB,
)
from mental_model.canonical.embeddings import create_embeddings
from crew.schemas import (
    InvestmentQuestion,
    RetrievedMentalModel,
)
from mental_model.database.connection import SessionLocal


@dataclass(frozen=True)
class EdgeLink:
    other_id: UUID
    relation_type: str
    score: float


def _cosine(query: np.ndarray, embedding: list[float]) -> float:
    vector = np.asarray(embedding, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return -1.0
    return float(np.dot(query, vector / norm))


def _as_retrieved(
    model: CanonicalMentalModelDB,
    *,
    score: float,
    origin: str,
    relation: str | None = None,
) -> RetrievedMentalModel:
    return RetrievedMentalModel(
        canonical_code=model.canonical_code,
        investor_id=model.investor_id,
        title=model.title,
        proposition=model.proposition,
        mechanism=list(model.mechanism),
        conditions=list(model.conditions),
        failure_conditions=list(model.failure_conditions),
        decision_implications=list(model.decision_implications),
        primary_domain=model.primary_domain,
        concept_family=model.concept_family,
        retrieval_score=max(-1.0, min(1.0, score)),
        retrieval_origin=origin,
        relation_to_source=relation,
    )


def retrieve_for_investors(
    question: InvestmentQuestion,
    *,
    investor_filter: set[str] | None = None,
    top_k: int = 5,
    neighbour_limit: int = 3,
) -> dict[str, list[RetrievedMentalModel]]:
    """Retrieve top models per investor and expand through same-investor edges."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if neighbour_limit < 0:
        raise ValueError("neighbour_limit cannot be negative.")

    query_embedding = create_embeddings([question.embedding_text()])[0]
    query = np.asarray(query_embedding, dtype=np.float64)
    query /= np.linalg.norm(query)

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
        edges = list(session.scalars(select(CanonicalModelEdgeDB)))

    if investor_filter:
        models = [
            model for model in models
            if model.investor_id in investor_filter
        ]

    model_by_id = {model.canonical_id: model for model in models}
    similarity_by_id = {
        model.canonical_id: _cosine(query, model.embedding)
        for model in models
    }

    adjacency: dict[UUID, list[EdgeLink]] = {}
    for edge in edges:
        source = model_by_id.get(edge.source_canonical_id)
        target = model_by_id.get(edge.target_canonical_id)
        if source is None or target is None:
            continue
        if source.investor_id != target.investor_id:
            continue

        edge_score = (
            0.5 * edge.relation_strength
            + 0.5 * edge.relation_confidence
        )
        adjacency.setdefault(source.canonical_id, []).append(
            EdgeLink(target.canonical_id, edge.relation_type, edge_score)
        )
        adjacency.setdefault(target.canonical_id, []).append(
            EdgeLink(source.canonical_id, edge.relation_type, edge_score)
        )

    by_investor: dict[str, list[CanonicalMentalModelDB]] = {}
    for model in models:
        by_investor.setdefault(model.investor_id, []).append(model)

    output: dict[str, list[RetrievedMentalModel]] = {}

    for investor_id, investor_models in sorted(by_investor.items()):
        ranked = sorted(
            investor_models,
            key=lambda model: (
                -(
                    0.85 * similarity_by_id[model.canonical_id]
                    + 0.15 * model.base_weight
                ),
                model.canonical_code,
            ),
        )
        direct = ranked[:top_k]
        selected_ids = {model.canonical_id for model in direct}

        retrieved = [
            _as_retrieved(
                model,
                score=(
                    0.85 * similarity_by_id[model.canonical_id]
                    + 0.15 * model.base_weight
                ),
                origin="direct",
            )
            for model in direct
        ]

        neighbour_candidates: dict[UUID, tuple[float, str]] = {}
        for source in direct:
            for link in adjacency.get(source.canonical_id, []):
                if link.other_id in selected_ids:
                    continue
                neighbour = model_by_id[link.other_id]
                score = (
                    0.55 * similarity_by_id[neighbour.canonical_id]
                    + 0.15 * neighbour.base_weight
                    + 0.30 * link.score
                )
                previous = neighbour_candidates.get(link.other_id)
                if previous is None or score > previous[0]:
                    neighbour_candidates[link.other_id] = (
                        score,
                        f"{link.relation_type} from {source.canonical_code}",
                    )

        for neighbour_id, (score, relation) in sorted(
            neighbour_candidates.items(),
            key=lambda item: (-item[1][0], model_by_id[item[0]].canonical_code),
        )[:neighbour_limit]:
            retrieved.append(
                _as_retrieved(
                    model_by_id[neighbour_id],
                    score=score,
                    origin="neighbour",
                    relation=relation,
                )
            )

        output[investor_id] = retrieved

    return output
