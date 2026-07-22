"""Assign constitution domains and create weighted relationships between canonical models."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from mental_model_pipeline.canonical.db_models import CanonicalMentalModelDB
from mental_model_pipeline.canonical.export_jsonl import (
    DEFAULT_OUTPUT_PATH,
    export_canonical_jsonl,
)
from mental_model_pipeline.canonical.prompts import (
    CONSTITUTION_SYSTEM_PROMPT,
    RELATIONSHIP_SYSTEM_PROMPT,
)
from mental_model_pipeline.canonical.providers import OpenAIStructuredProvider
from mental_model_pipeline.canonical.repository import replace_hierarchy
from mental_model_pipeline.canonical.schemas import (
    ConstitutionBatchResult,
    HierarchyRelationshipBatchResult,
    RelationScope,
    RelationType,
    SYMMETRIC_RELATIONS,
)
from mental_model_pipeline.database.connection import SessionLocal


@dataclass(frozen=True)
class CandidatePair:
    pair_key: str
    left: CanonicalMentalModelDB
    right: CanonicalMentalModelDB
    similarity: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign the shared constitution and rebuild weighted canonical "
            "model edges."
        )
    )
    parser.add_argument(
        "--model",
        default=os.getenv("HIERARCHY_MODEL", "gpt-5.6-luna"),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("HIERARCHY_REASONING_EFFORT", "low"),
    )
    parser.add_argument(
        "--classification-batch-size",
        type=int,
        default=20,
    )
    parser.add_argument("--pairs-per-call", type=int, default=16)
    parser.add_argument("--top-k-neighbours", type=int, default=8)
    parser.add_argument(
        "--minimum-similarity",
        type=float,
        default=0.62,
    )
    parser.add_argument("--max-output-tokens", type=int, default=9000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--include-embeddings-in-export",
        action="store_true",
    )
    return parser.parse_args()


def _batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    if size < 1:
        raise ValueError("Batch size must be positive.")

    for start in range(0, len(values), size):
        yield values[start : start + size]


def _model_card(model: CanonicalMentalModelDB) -> dict[str, Any]:
    return {
        "canonical_code": model.canonical_code,
        "investor_id": model.investor_id,
        "kind": model.kind,
        "title": model.title,
        "proposition": model.proposition,
        "mechanism": list(model.mechanism),
        "conditions": list(model.conditions),
        "failure_conditions": list(model.failure_conditions),
        "decision_implications": list(model.decision_implications),
        "decision_stages": list(model.decision_stages),
        "contextual_regimes": list(model.contextual_regimes),
    }


def _load_models() -> list[CanonicalMentalModelDB]:
    with SessionLocal() as session:
        return list(
            session.scalars(
                select(CanonicalMentalModelDB)
                .where(CanonicalMentalModelDB.embedding.is_not(None))
                .order_by(
                    CanonicalMentalModelDB.investor_id,
                    CanonicalMentalModelDB.canonical_code,
                )
            )
        )


def _classify_constitution(
    *,
    models: list[CanonicalMentalModelDB],
    provider: OpenAIStructuredProvider,
    batch_size: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    assignments: dict[str, Any] = {}

    for model_batch in _batches(models, batch_size):
        result = provider.parse(
            schema=ConstitutionBatchResult,
            system_prompt=CONSTITUTION_SYSTEM_PROMPT,
            user_prompt=json.dumps(
                {
                    "canonical_models": [
                        _model_card(model) for model in model_batch
                    ]
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            max_output_tokens=max_output_tokens,
        )
        allowed = {model.canonical_code for model in model_batch}

        for assignment in result.assignments:
            if assignment.canonical_code in allowed:
                assignments[assignment.canonical_code] = assignment

    return assignments


def _normalised_matrix(models: list[CanonicalMentalModelDB]) -> np.ndarray:
    vectors = np.vstack(
        [np.asarray(model.embedding, dtype=np.float64) for model in models]
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    if np.any(norms == 0.0):
        raise ValueError("A canonical model has a zero embedding.")

    return vectors / norms


def _candidate_pairs(
    models: list[CanonicalMentalModelDB],
    *,
    assignments: dict[str, Any],
    top_k: int,
    minimum_similarity: float,
) -> list[CandidatePair]:
    if len(models) < 2:
        return []

    matrix = _normalised_matrix(models)
    similarity = matrix @ matrix.T
    chosen: dict[tuple[int, int], CandidatePair] = {}

    def add_pair(left_index: int, right_index: int, score: float) -> None:
        key = tuple(sorted((left_index, right_index)))
        if key in chosen:
            return

        left = models[key[0]]
        right = models[key[1]]
        chosen[key] = CandidatePair(
            pair_key=(
                f"pair_{left.canonical_code}__{right.canonical_code}"
            ),
            left=left,
            right=right,
            similarity=score,
        )

    for left_index, left in enumerate(models):
        same_investor: list[tuple[int, float]] = []
        cross_investor: list[tuple[int, float]] = []

        for right_index, right in enumerate(models):
            if left_index == right_index:
                continue

            score = float(similarity[left_index, right_index])
            if score < minimum_similarity:
                continue

            bucket = (
                same_investor
                if left.investor_id == right.investor_id
                else cross_investor
            )
            bucket.append((right_index, score))

        for bucket in (same_investor, cross_investor):
            bucket.sort(
                key=lambda item: (
                    -item[1],
                    models[item[0]].canonical_code,
                )
            )
            for right_index, score in bucket[:top_k]:
                add_pair(left_index, right_index, score)

    # A shared concept family is a cheap second candidate signal.
    family_to_indices: dict[str, list[int]] = {}

    for index, model in enumerate(models):
        assignment = assignments.get(model.canonical_code)
        if assignment is None:
            continue

        family = assignment.concept_family.strip().casefold()
        if family:
            family_to_indices.setdefault(family, []).append(index)

    for indices in family_to_indices.values():
        for left_index in indices:
            neighbours = sorted(
                (
                    (
                        right_index,
                        float(similarity[left_index, right_index]),
                    )
                    for right_index in indices
                    if right_index != left_index
                ),
                key=lambda item: (
                    -item[1],
                    models[item[0]].canonical_code,
                ),
            )
            for right_index, score in neighbours[:2]:
                if score >= max(0.0, minimum_similarity - 0.10):
                    add_pair(left_index, right_index, score)

    return sorted(chosen.values(), key=lambda pair: pair.pair_key)


def _relationship_prompt(pairs: list[CandidatePair]) -> str:
    return json.dumps(
        {
            "pairs": [
                {
                    "pair_key": pair.pair_key,
                    "candidate_similarity": pair.similarity,
                    "model_a": _model_card(pair.left),
                    "model_b": _model_card(pair.right),
                }
                for pair in pairs
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _edge_key(
    *,
    source_code: str,
    target_code: str,
    relation_type: RelationType,
) -> tuple[str, str, str]:
    if relation_type in SYMMETRIC_RELATIONS:
        source_code, target_code = sorted((source_code, target_code))
    return source_code, target_code, relation_type.value


def _build_edges(
    *,
    pairs: list[CandidatePair],
    provider: OpenAIStructuredProvider,
    pairs_per_call: int,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    model_by_code = {
        model.canonical_code: model
        for pair in pairs
        for model in (pair.left, pair.right)
    }
    pair_by_key = {pair.pair_key: pair for pair in pairs}
    edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    for pair_batch in _batches(pairs, pairs_per_call):
        result = provider.parse(
            schema=HierarchyRelationshipBatchResult,
            system_prompt=RELATIONSHIP_SYSTEM_PROMPT,
            user_prompt=_relationship_prompt(pair_batch),
            max_output_tokens=max_output_tokens,
        )

        for pair_result in result.pairs:
            pair = pair_by_key.get(pair_result.pair_key)
            if pair is None:
                continue

            allowed = {
                pair.left.canonical_code,
                pair.right.canonical_code,
            }

            for relation in pair_result.relationships:
                if {
                    relation.source_canonical_code,
                    relation.target_canonical_code,
                } != allowed:
                    continue

                key = _edge_key(
                    source_code=relation.source_canonical_code,
                    target_code=relation.target_canonical_code,
                    relation_type=relation.relation_type,
                )
                source_code, target_code, _ = key
                source = model_by_code[source_code]
                target = model_by_code[target_code]
                scope = (
                    RelationScope.WITHIN_INVESTOR
                    if source.investor_id == target.investor_id
                    else RelationScope.CROSS_INVESTOR
                )
                candidate = {
                    "source_canonical_id": source.canonical_id,
                    "target_canonical_id": target.canonical_id,
                    "relation_type": relation.relation_type.value,
                    "relation_strength": relation.relation_strength,
                    "relation_confidence": relation.relation_confidence,
                    "candidate_similarity": pair.similarity,
                    "scope": scope.value,
                }
                previous = edges_by_key.get(key)

                if (
                    previous is None
                    or candidate["relation_confidence"]
                    > previous["relation_confidence"]
                ):
                    edges_by_key[key] = candidate

    return list(edges_by_key.values())


def main() -> int:
    args = parse_arguments()
    models = _load_models()

    if not models:
        print("No embedded canonical models found.")
        return 0

    if args.dry_run:
        pairs = _candidate_pairs(
            models,
            assignments={},
            top_k=args.top_k_neighbours,
            minimum_similarity=args.minimum_similarity,
        )
        print(
            f"canonical_models={len(models)}, "
            f"embedding_candidate_pairs={len(pairs)}"
        )
        return 0

    provider = OpenAIStructuredProvider(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    assignments = _classify_constitution(
        models=models,
        provider=provider,
        batch_size=args.classification_batch_size,
        max_output_tokens=args.max_output_tokens,
    )
    pairs = _candidate_pairs(
        models,
        assignments=assignments,
        top_k=args.top_k_neighbours,
        minimum_similarity=args.minimum_similarity,
    )
    edges = _build_edges(
        pairs=pairs,
        provider=provider,
        pairs_per_call=args.pairs_per_call,
        max_output_tokens=args.max_output_tokens,
    )

    with SessionLocal() as session:
        try:
            replace_hierarchy(
                session,
                assignments=assignments,
                edges=edges,
            )
            session.commit()
        except Exception:
            session.rollback()
            raise

    exported = export_canonical_jsonl(
        output_path=args.output_path,
        include_embeddings=args.include_embeddings_in_export,
    )
    print(
        f"assigned={len(assignments)}/{len(models)}, "
        f"candidate_pairs={len(pairs)}, edges={len(edges)}, "
        f"exported={exported}, output={args.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
