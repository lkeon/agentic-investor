"""Retrieve investor mental models for each evidence-backed bridge."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import numpy as np
from sqlalchemy import select

from crew.schemas import (
    MentalModelCandidate,
    MentalModelBridge,
)
from mental_model_pipeline.canonical.db_models import (
    CanonicalMentalModelDB,
    CanonicalModelEdgeDB,
)
from mental_model_pipeline.canonical.embeddings import (
    EmbeddingProvider,
    accepted_embedding_identities,
    create_embedding_provider,
)
from mental_model_pipeline.canonical.schemas import SYMMETRIC_RELATIONS
from mental_model_pipeline.database.connection import SessionLocal


@dataclass(frozen=True)
class EdgeLink:
    other_id: UUID
    relation_type: str
    score: float


def _normalise(vector: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("Embedding must be a finite one-dimensional vector.")
    norm = np.linalg.norm(array)
    if norm == 0.0:
        raise ValueError("Embedding cannot be a zero vector.")
    return array / norm


def _cosine(query: np.ndarray, embedding: list[float]) -> float:
    vector = _normalise(embedding)
    if query.shape != vector.shape:
        raise ValueError(
            "Query and canonical embedding dimensions do not match: "
            f"{query.shape} != {vector.shape}."
        )
    return float(np.dot(query, vector))


def _metadata_match(
    model: CanonicalMentalModelDB,
    bridge: MentalModelBridge,
) -> float:
    requested_domains = set(bridge.domains)
    model_domains = {
        model.primary_domain,
        *(model.secondary_domains or []),
    }
    domain_match = bool(requested_domains & model_domains)
    stage_match = bridge.decision_stage in (model.decision_stages or [])
    return 0.5 * float(domain_match) + 0.5 * float(stage_match)


def _as_candidate(
    model: CanonicalMentalModelDB,
    bridge: MentalModelBridge,
    *,
    score: float,
    origin: str,
    relation: str | None = None,
) -> MentalModelCandidate:
    return MentalModelCandidate(
        canonical_code=model.canonical_code,
        investor_id=model.investor_id,
        title=model.title,
        proposition=model.proposition,
        mechanism=list(model.mechanism),
        conditions=list(model.conditions),
        failure_conditions=list(model.failure_conditions),
        decision_implications=list(model.decision_implications),
        matched_bridge_ids=[bridge.bridge_id],
        relevant_claim_ids=[
            claim.claim_id for claim in bridge.retrieved_data
        ],
        # Retrieval cannot prove applicability. The investor must evaluate
        # every condition against the supplied evidence.
        unresolved_conditions=list(model.conditions),
        retrieval_score=max(-1.0, min(1.0, score)),
        retrieval_origin=origin,
        relation_to_source=relation,
    )


def _build_adjacency(
    edges: list[CanonicalModelEdgeDB],
    model_by_id: dict[UUID, CanonicalMentalModelDB],
) -> dict[UUID, list[EdgeLink]]:
    adjacency: dict[UUID, list[EdgeLink]] = {}
    symmetric_values = {relation.value for relation in SYMMETRIC_RELATIONS}

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

        # Directional relations retain their stored direction.
        if edge.relation_type in symmetric_values:
            adjacency.setdefault(target.canonical_id, []).append(
                EdgeLink(source.canonical_id, edge.relation_type, edge_score)
            )

    return adjacency


def _retrieve_one_bridge(
    *,
    bridge: MentalModelBridge,
    query: np.ndarray,
    investor_models: list[CanonicalMentalModelDB],
    model_by_id: dict[UUID, CanonicalMentalModelDB],
    adjacency: dict[UUID, list[EdgeLink]],
    top_k: int,
    neighbour_limit: int,
) -> list[MentalModelCandidate]:
    similarity_by_id = {
        model.canonical_id: _cosine(query, model.embedding)
        for model in investor_models
    }

    def direct_score(model: CanonicalMentalModelDB) -> float:
        return (
            0.75 * similarity_by_id[model.canonical_id]
            + 0.15 * model.base_weight
            + 0.10 * _metadata_match(model, bridge)
        )

    ranked = sorted(
        investor_models,
        key=lambda model: (-direct_score(model), model.canonical_code),
    )
    direct = ranked[:top_k]
    selected_ids = {model.canonical_id for model in direct}
    candidates = [
        _as_candidate(
            model,
            bridge,
            score=direct_score(model),
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
                0.50 * similarity_by_id[neighbour.canonical_id]
                + 0.15 * neighbour.base_weight
                + 0.10 * _metadata_match(neighbour, bridge)
                + 0.25 * link.score
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
        candidates.append(
            _as_candidate(
                model_by_id[neighbour_id],
                bridge,
                score=score,
                origin="neighbour",
                relation=relation,
            )
        )
    return candidates


def retrieve_for_bridges(
    bridges: list[MentalModelBridge],
    *,
    investor_filter: set[str] | None = None,
    top_k: int = 3,
    neighbour_limit: int = 1,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, list[MentalModelBridge]]:
    """Retrieve and attach mental models separately for each investor."""

    if not bridges:
        raise ValueError("At least one mental-model bridge is required.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if neighbour_limit < 0:
        raise ValueError("neighbour_limit cannot be negative.")

    provider = embedding_provider or create_embedding_provider()
    query_embeddings = provider.embed(
        [bridge.search_query for bridge in bridges]
    )
    if len(query_embeddings) != len(bridges):
        raise RuntimeError("Bridge embedding response count mismatch.")
    queries = [_normalise(embedding) for embedding in query_embeddings]

    accepted_identities = accepted_embedding_identities()
    with SessionLocal() as session:
        models = list(
            session.scalars(
                select(CanonicalMentalModelDB)
                .where(
                    CanonicalMentalModelDB.embedding.is_not(None),
                    CanonicalMentalModelDB.embedding_model.in_(
                        accepted_identities
                    ),
                )
                .order_by(
                    CanonicalMentalModelDB.investor_id,
                    CanonicalMentalModelDB.canonical_code,
                )
            )
        )
        edges = list(session.scalars(select(CanonicalModelEdgeDB)))

    available_investors = {model.investor_id for model in models}
    if investor_filter is not None:
        missing = investor_filter - available_investors
        if missing:
            raise ValueError(
                "Requested investors have no canonical models for embedding "
                f"{provider.identity}: {sorted(missing)}"
            )
        models = [
            model for model in models if model.investor_id in investor_filter
        ]

    if not models:
        raise RuntimeError(
            "No canonical models match the active embedding identity "
            f"{provider.identity!r}. Re-embed canonical models with the active "
            "configuration."
        )

    model_by_id = {model.canonical_id: model for model in models}
    adjacency = _build_adjacency(edges, model_by_id)
    by_investor: dict[str, list[CanonicalMentalModelDB]] = {}
    for model in models:
        by_investor.setdefault(model.investor_id, []).append(model)

    output: dict[str, list[MentalModelBridge]] = {}
    for investor_id, investor_models in sorted(by_investor.items()):
        investor_bridges: list[MentalModelBridge] = []
        for bridge, query in zip(bridges, queries, strict=True):
            candidates = _retrieve_one_bridge(
                bridge=bridge,
                query=query,
                investor_models=investor_models,
                model_by_id=model_by_id,
                adjacency=adjacency,
                top_k=top_k,
                neighbour_limit=neighbour_limit,
            )
            investor_bridges.append(
                bridge.model_copy(
                    update={
                        "investor_id": investor_id,
                        "mental_model_candidates": candidates,
                    }
                )
            )

        # Record every bridge that retrieved the same canonical model while
        # retaining candidates inside their self-contained bridge.
        matched_by_code: dict[str, set[str]] = {}
        for bridge in investor_bridges:
            for candidate in bridge.mental_model_candidates:
                matched_by_code.setdefault(
                    candidate.canonical_code,
                    set(),
                ).add(bridge.bridge_id)
        output[investor_id] = [
            bridge.model_copy(
                update={
                    "mental_model_candidates": [
                        candidate.model_copy(
                            update={
                                "matched_bridge_ids": sorted(
                                    matched_by_code[
                                        candidate.canonical_code
                                    ]
                                )
                            }
                        )
                        for candidate in bridge.mental_model_candidates
                    ]
                }
            )
            for bridge in investor_bridges
        ]

    return output
