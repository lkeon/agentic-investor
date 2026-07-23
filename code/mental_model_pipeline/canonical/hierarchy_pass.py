"""Assign domains, shared concept families, and canonical-model edges."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from mental_model_pipeline.canonical.clustering import (
    FragmentCluster,
    cluster_fragments,
)
from mental_model_pipeline.canonical.db_models import (
    CanonicalMentalModelDB,
)
from mental_model_pipeline.canonical.export_jsonl import (
    DEFAULT_OUTPUT_PATH,
    export_canonical_jsonl,
)
from mental_model_pipeline.canonical.prompts import (
    CONCEPT_FAMILY_SYSTEM_PROMPT,
    CONSTITUTION_SYSTEM_PROMPT,
    RELATIONSHIP_SYSTEM_PROMPT,
)
from mental_model_pipeline.canonical.providers import (
    OpenAIStructuredProvider,
)
from mental_model_pipeline.canonical.repository import (
    replace_hierarchy,
)
from mental_model_pipeline.canonical.schemas import (
    ConceptFamilyBatchResult,
    ConstitutionAssignment,
    ConstitutionBatchResult,
    ConstitutionDomain,
    HierarchyAssignment,
    HierarchyRelationshipBatchResult,
    RelationScope,
    RelationType,
    SYMMETRIC_RELATIONS,
)
from mental_model_pipeline.database.connection import SessionLocal


@dataclass(frozen=True)
class CandidatePair:
    """Two canonical models proposed for relationship classification."""

    pair_key: str
    left: CanonicalMentalModelDB
    right: CanonicalMentalModelDB
    similarity: float


@dataclass(frozen=True)
class FamilyClusterItem:
    """Adapter allowing canonical models to use the fragment clusterer."""

    fragment_code: str
    embedding: list[float]
    model: CanonicalMentalModelDB


def parse_arguments() -> argparse.Namespace:
    """Parse hierarchy command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Assign fixed constitution domains, derive shared concept "
            "families, and rebuild canonical-model edges."
        )
    )

    parser.add_argument(
        "--model",
        default=os.getenv(
            "HIERARCHY_MODEL",
            "gpt-5.6-luna",
        ),
    )

    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv(
            "HIERARCHY_REASONING_EFFORT",
            "low",
        ),
    )

    parser.add_argument(
        "--classification-batch-size",
        type=int,
        default=20,
        help=(
            "Number of canonical models included in each domain "
            "assignment call."
        ),
    )

    parser.add_argument(
        "--family-similarity-threshold",
        type=float,
        default=0.72,
        help=(
            "Minimum complete-link cosine similarity used to group "
            "canonical models into concept families."
        ),
    )

    parser.add_argument(
        "--max-family-size",
        type=int,
        default=20,
        help="Maximum number of canonical models in one family cluster.",
    )

    parser.add_argument(
        "--families-per-call",
        type=int,
        default=8,
        help=(
            "Maximum number of concept-family clusters included in "
            "one Luna call."
        ),
    )

    parser.add_argument(
        "--pairs-per-call",
        type=int,
        default=24,
        help=(
            "Maximum number of candidate relationship pairs included "
            "in one Luna call."
        ),
    )

    parser.add_argument(
        "--top-k-neighbours",
        type=int,
        default=2,
        help=(
            "Number of nearest within-investor and cross-investor "
            "neighbours selected for each canonical model."
        ),
    )

    parser.add_argument(
        "--minimum-similarity",
        type=float,
        default=0.72,
        help=(
            "Minimum cosine similarity required for a standard "
            "relationship candidate."
        ),
    )

    parser.add_argument(
        "--max-candidate-pairs",
        type=int,
        default=8000,
        help=(
            "Maximum number of relationship pairs sent to Luna. "
            "Highest-similarity pairs are retained. Use 0 for no cap."
        ),
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=9000,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

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


def _batches(
    values: list[Any],
    size: int,
) -> Iterable[list[Any]]:
    """Yield fixed-size batches."""

    if size < 1:
        raise ValueError(
            "Batch size must be positive."
        )

    for start in range(0, len(values), size):
        yield values[start : start + size]


def _planned_calls(
    item_count: int,
    batch_size: int,
) -> int:
    """Calculate the number of required API calls."""

    if item_count == 0:
        return 0

    return (
        item_count
        + batch_size
        - 1
    ) // batch_size


def _model_card(
    model: CanonicalMentalModelDB,
) -> dict[str, Any]:
    """Build a canonical-model representation for hierarchy prompts."""

    return {
        "canonical_code": model.canonical_code,
        "investor_id": model.investor_id,
        "kind": model.kind,
        "title": model.title,
        "proposition": model.proposition,
        "mechanism": list(model.mechanism),
        "conditions": list(model.conditions),
        "failure_conditions": list(
            model.failure_conditions
        ),
        "decision_implications": list(
            model.decision_implications
        ),
        "decision_stages": list(
            model.decision_stages
        ),
        "contextual_regimes": list(
            model.contextual_regimes
        ),
    }


def _family_model_card(
    model: CanonicalMentalModelDB,
) -> dict[str, Any]:
    """Build a compact representation used for concept-family naming."""

    return {
        "canonical_code": model.canonical_code,
        "investor_id": model.investor_id,
        "title": model.title,
        "proposition": model.proposition,
        "mechanism": list(model.mechanism),
        "conditions": list(model.conditions),
        "failure_conditions": list(
            model.failure_conditions
        ),
    }


def _load_models() -> list[CanonicalMentalModelDB]:
    """Load all canonical models that have embeddings."""

    with SessionLocal() as session:
        statement = (
            select(CanonicalMentalModelDB)
            .where(
                CanonicalMentalModelDB.embedding.is_not(None)
            )
            .order_by(
                CanonicalMentalModelDB.investor_id,
                CanonicalMentalModelDB.canonical_code,
            )
        )

        return list(
            session.scalars(statement)
        )


def _normalised_matrix(
    models: list[CanonicalMentalModelDB],
) -> np.ndarray:
    """Return an L2-normalised embedding matrix."""

    vectors = np.vstack(
        [
            np.asarray(
                model.embedding,
                dtype=np.float64,
            )
            for model in models
        ]
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0.0):
        raise ValueError(
            "A canonical model has a zero embedding."
        )

    return vectors / norms


def _classify_constitution(
    *,
    models: list[CanonicalMentalModelDB],
    provider: OpenAIStructuredProvider,
    batch_size: int,
    max_output_tokens: int,
) -> dict[str, ConstitutionAssignment]:
    """Assign fixed primary and secondary constitution domains."""

    assignments: dict[
        str,
        ConstitutionAssignment,
    ] = {}

    batches = list(
        _batches(
            models,
            batch_size,
        )
    )

    print(
        f"Domain assignment: "
        f"models={len(models)}, "
        f"planned_luna_calls={len(batches)}.",
        flush=True,
    )

    for call_number, model_batch in enumerate(
        batches,
        start=1,
    ):
        print(
            f"Domain Luna call "
            f"{call_number}/{len(batches)} starting — "
            f"models={len(model_batch)}...",
            flush=True,
        )

        started = time.perf_counter()

        result = provider.parse(
            schema=ConstitutionBatchResult,
            system_prompt=CONSTITUTION_SYSTEM_PROMPT,
            user_prompt=json.dumps(
                {
                    "canonical_models": [
                        _model_card(model)
                        for model in model_batch
                    ]
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            max_output_tokens=max_output_tokens,
        )

        allowed_codes = {
            model.canonical_code
            for model in model_batch
        }

        accepted = 0

        for assignment in result.assignments:
            if (
                assignment.canonical_code
                not in allowed_codes
            ):
                continue

            assignments[
                assignment.canonical_code
            ] = assignment

            accepted += 1

        print(
            f"Domain Luna call "
            f"{call_number}/{len(batches)} completed in "
            f"{time.perf_counter() - started:.1f}s — "
            f"returned={len(result.assignments)}, "
            f"accepted={accepted}.",
            flush=True,
        )

    return assignments


def _cluster_concept_families(
    *,
    models: list[CanonicalMentalModelDB],
    domain_assignments: dict[
        str,
        ConstitutionAssignment,
    ],
    similarity_threshold: float,
    max_family_size: int,
) -> list[FragmentCluster]:
    """Cluster canonical models within each assigned primary domain."""

    by_domain: dict[
        ConstitutionDomain,
        list[FamilyClusterItem],
    ] = {}

    for model in models:
        assignment = domain_assignments.get(
            model.canonical_code
        )

        if assignment is None:
            continue

        if (
            assignment.primary_domain
            == ConstitutionDomain.UNASSIGNED
        ):
            continue

        item = FamilyClusterItem(
            fragment_code=model.canonical_code,
            embedding=model.embedding,
            model=model,
        )

        by_domain.setdefault(
            assignment.primary_domain,
            [],
        ).append(item)

    all_clusters: list[FragmentCluster] = []

    for domain in sorted(
        by_domain,
        key=lambda value: value.value,
    ):
        domain_models = by_domain[domain]

        clusters = cluster_fragments(
            domain_models,
            similarity_threshold=similarity_threshold,
            max_cluster_size=max_family_size,
        )

        all_clusters.extend(clusters)

        singleton_count = sum(
            len(cluster.fragments) == 1
            for cluster in clusters
        )

        print(
            f"Family clustering: "
            f"domain={domain.value}, "
            f"models={len(domain_models)}, "
            f"families={len(clusters)}, "
            f"singletons={singleton_count}.",
            flush=True,
        )

    return all_clusters


def _concept_family_prompt(
    clusters: list[FragmentCluster],
) -> str:
    """Build a prompt for concept-family cluster naming."""

    return json.dumps(
        {
            "concept_family_clusters": [
                {
                    "cluster_key": cluster.cluster_key,
                    "canonical_models": [
                        _family_model_card(item.model)
                        for item in cluster.fragments
                    ],
                }
                for cluster in clusters
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _name_concept_families(
    *,
    clusters: list[FragmentCluster],
    provider: OpenAIStructuredProvider,
    families_per_call: int,
    max_output_tokens: int,
) -> tuple[dict[str, str], int]:
    """Assign one compact family value to every model in each cluster."""

    family_by_model_code: dict[str, str] = {}
    omitted_clusters = 0

    batches = list(
        _batches(
            clusters,
            families_per_call,
        )
    )

    print(
        f"Concept-family naming: "
        f"families={len(clusters)}, "
        f"planned_luna_calls={len(batches)}.",
        flush=True,
    )

    for call_number, cluster_batch in enumerate(
        batches,
        start=1,
    ):
        model_count = sum(
            len(cluster.fragments)
            for cluster in cluster_batch
        )

        print(
            f"Family Luna call "
            f"{call_number}/{len(batches)} starting — "
            f"families={len(cluster_batch)}, "
            f"models={model_count}...",
            flush=True,
        )

        started = time.perf_counter()

        result = provider.parse(
            schema=ConceptFamilyBatchResult,
            system_prompt=CONCEPT_FAMILY_SYSTEM_PROMPT,
            user_prompt=_concept_family_prompt(
                cluster_batch
            ),
            max_output_tokens=max_output_tokens,
        )

        result_by_key = {
            family.cluster_key: family
            for family in result.families
        }

        assigned_in_call = 0

        for cluster in cluster_batch:
            family = result_by_key.get(
                cluster.cluster_key
            )

            if family is None:
                omitted_clusters += 1
                continue

            for item in cluster.fragments:
                family_by_model_code[
                    item.model.canonical_code
                ] = family.concept_family

                assigned_in_call += 1

        print(
            f"Family Luna call "
            f"{call_number}/{len(batches)} completed in "
            f"{time.perf_counter() - started:.1f}s — "
            f"returned_families={len(result.families)}, "
            f"assigned_models={assigned_in_call}.",
            flush=True,
        )

    return (
        family_by_model_code,
        omitted_clusters,
    )


def _combine_assignments(
    *,
    models: list[CanonicalMentalModelDB],
    domain_assignments: dict[
        str,
        ConstitutionAssignment,
    ],
    family_by_model_code: dict[str, str],
) -> dict[str, HierarchyAssignment]:
    """Combine domain and concept-family results."""

    output: dict[
        str,
        HierarchyAssignment,
    ] = {}

    for model in models:
        domain = domain_assignments.get(
            model.canonical_code
        )

        if domain is None:
            continue

        output[
            model.canonical_code
        ] = HierarchyAssignment(
            canonical_code=model.canonical_code,
            primary_domain=domain.primary_domain,
            secondary_domains=domain.secondary_domains,
            concept_family=family_by_model_code.get(
                model.canonical_code
            ),
        )

    return output


def _family_key(
    value: str | None,
) -> str:
    """Extract the normalised family name before the colon."""

    if not value:
        return ""

    family_name, _, _ = value.partition(":")

    return family_name.strip().casefold()


def _candidate_pairs(
    models: list[CanonicalMentalModelDB],
    *,
    assignments: dict[
        str,
        HierarchyAssignment,
    ],
    top_k: int,
    minimum_similarity: float,
    max_candidate_pairs: int,
) -> tuple[list[CandidatePair], int]:
    """
    Build selective relationship candidates.

    Returns:
        selected candidates after the optional cap;
        raw unique candidate count before the cap.
    """

    if len(models) < 2:
        return [], 0

    if top_k < 0:
        raise ValueError(
            "top_k must be zero or positive."
        )

    matrix = _normalised_matrix(models)
    similarity = matrix @ matrix.T

    chosen: dict[
        tuple[int, int],
        CandidatePair,
    ] = {}

    def add_pair(
        left_index: int,
        right_index: int,
        score: float,
    ) -> None:
        key = tuple(
            sorted(
                (
                    left_index,
                    right_index,
                )
            )
        )

        existing = chosen.get(key)

        if (
            existing is not None
            and existing.similarity >= score
        ):
            return

        left = models[key[0]]
        right = models[key[1]]

        chosen[key] = CandidatePair(
            pair_key=(
                f"pair_{left.canonical_code}"
                f"__{right.canonical_code}"
            ),
            left=left,
            right=right,
            similarity=score,
        )

    if top_k > 0:
        for left_index, left in enumerate(models):
            same_investor: list[
                tuple[int, float]
            ] = []

            cross_investor: list[
                tuple[int, float]
            ] = []

            for right_index, right in enumerate(models):
                if left_index == right_index:
                    continue

                score = float(
                    similarity[
                        left_index,
                        right_index,
                    ]
                )

                if score < minimum_similarity:
                    continue

                if (
                    left.investor_id
                    == right.investor_id
                ):
                    same_investor.append(
                        (
                            right_index,
                            score,
                        )
                    )
                else:
                    cross_investor.append(
                        (
                            right_index,
                            score,
                        )
                    )

            for bucket in (
                same_investor,
                cross_investor,
            ):
                bucket.sort(
                    key=lambda item: (
                        -item[1],
                        models[
                            item[0]
                        ].canonical_code,
                    )
                )

                for right_index, score in bucket[:top_k]:
                    add_pair(
                        left_index,
                        right_index,
                        score,
                    )

    family_to_indices: dict[
        str,
        list[int],
    ] = {}

    for index, model in enumerate(models):
        assignment = assignments.get(
            model.canonical_code
        )

        if assignment is None:
            continue

        family = _family_key(
            assignment.concept_family
        )

        if not family:
            continue

        family_to_indices.setdefault(
            family,
            [],
        ).append(index)

    family_similarity_threshold = max(
        minimum_similarity - 0.05,
        0.0,
    )

    for indices in family_to_indices.values():
        if len(indices) < 2:
            continue

        for left_index in indices:
            neighbours = sorted(
                (
                    (
                        right_index,
                        float(
                            similarity[
                                left_index,
                                right_index,
                            ]
                        ),
                    )
                    for right_index in indices
                    if right_index != left_index
                ),
                key=lambda item: (
                    -item[1],
                    models[
                        item[0]
                    ].canonical_code,
                ),
            )

            for right_index, score in neighbours[:2]:
                if score < family_similarity_threshold:
                    continue

                add_pair(
                    left_index,
                    right_index,
                    score,
                )

    raw_count = len(chosen)

    ranked = sorted(
        chosen.values(),
        key=lambda pair: (
            -pair.similarity,
            pair.pair_key,
        ),
    )

    if (
        max_candidate_pairs > 0
        and len(ranked) > max_candidate_pairs
    ):
        ranked = ranked[:max_candidate_pairs]

    return ranked, raw_count


def _relationship_prompt(
    pairs: list[CandidatePair],
) -> str:
    """Build the relationship-classification prompt."""

    return json.dumps(
        {
            "pairs": [
                {
                    "pair_key": pair.pair_key,
                    "candidate_similarity": (
                        pair.similarity
                    ),
                    "model_a": _model_card(
                        pair.left
                    ),
                    "model_b": _model_card(
                        pair.right
                    ),
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
    """Create a deterministic key used to deduplicate edges."""

    if relation_type in SYMMETRIC_RELATIONS:
        source_code, target_code = sorted(
            (
                source_code,
                target_code,
            )
        )

    return (
        source_code,
        target_code,
        relation_type.value,
    )


def _build_edges(
    *,
    pairs: list[CandidatePair],
    provider: OpenAIStructuredProvider,
    pairs_per_call: int,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    """Ask Luna to classify selected relationship candidates."""

    if not pairs:
        print(
            "Relationship classification: no candidate pairs.",
            flush=True,
        )
        return []

    model_by_code = {
        model.canonical_code: model
        for pair in pairs
        for model in (
            pair.left,
            pair.right,
        )
    }

    pair_by_key = {
        pair.pair_key: pair
        for pair in pairs
    }

    edges_by_key: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    batches = list(
        _batches(
            pairs,
            pairs_per_call,
        )
    )

    print(
        f"Relationship classification: "
        f"candidate_pairs={len(pairs)}, "
        f"planned_luna_calls={len(batches)}.",
        flush=True,
    )

    for call_number, pair_batch in enumerate(
        batches,
        start=1,
    ):
        print(
            f"Relationship Luna call "
            f"{call_number}/{len(batches)} starting — "
            f"pairs={len(pair_batch)}...",
            flush=True,
        )

        started = time.perf_counter()

        result = provider.parse(
            schema=HierarchyRelationshipBatchResult,
            system_prompt=RELATIONSHIP_SYSTEM_PROMPT,
            user_prompt=_relationship_prompt(
                pair_batch
            ),
            max_output_tokens=max_output_tokens,
        )

        accepted_relationships = 0

        for pair_result in result.pairs:
            pair = pair_by_key.get(
                pair_result.pair_key
            )

            if pair is None:
                continue

            allowed_codes = {
                pair.left.canonical_code,
                pair.right.canonical_code,
            }

            for relationship in (
                pair_result.relationships
            ):
                returned_codes = {
                    relationship.source_canonical_code,
                    relationship.target_canonical_code,
                }

                if returned_codes != allowed_codes:
                    continue

                key = _edge_key(
                    source_code=(
                        relationship
                        .source_canonical_code
                    ),
                    target_code=(
                        relationship
                        .target_canonical_code
                    ),
                    relation_type=(
                        relationship.relation_type
                    ),
                )

                source_code, target_code, _ = key

                source = model_by_code[source_code]
                target = model_by_code[target_code]

                if (
                    source.investor_id
                    == target.investor_id
                ):
                    scope = (
                        RelationScope.WITHIN_INVESTOR
                    )
                else:
                    scope = (
                        RelationScope.CROSS_INVESTOR
                    )

                candidate = {
                    "source_canonical_id": (
                        source.canonical_id
                    ),
                    "target_canonical_id": (
                        target.canonical_id
                    ),
                    "relation_type": (
                        relationship
                        .relation_type
                        .value
                    ),
                    "relation_strength": (
                        relationship
                        .relation_strength
                    ),
                    "relation_confidence": (
                        relationship
                        .relation_confidence
                    ),
                    "candidate_similarity": (
                        pair.similarity
                    ),
                    "scope": scope.value,
                }

                previous = edges_by_key.get(key)

                if (
                    previous is None
                    or candidate[
                        "relation_confidence"
                    ]
                    > previous[
                        "relation_confidence"
                    ]
                ):
                    edges_by_key[key] = candidate

                accepted_relationships += 1

        print(
            f"Relationship Luna call "
            f"{call_number}/{len(batches)} completed in "
            f"{time.perf_counter() - started:.1f}s — "
            f"returned_pairs={len(result.pairs)}, "
            f"accepted_relationships="
            f"{accepted_relationships}, "
            f"unique_edges_so_far="
            f"{len(edges_by_key)}.",
            flush=True,
        )

    return list(
        edges_by_key.values()
    )


def _dry_run(
    *,
    models: list[CanonicalMentalModelDB],
    args: argparse.Namespace,
) -> int:
    """Run deterministic checks without API calls or database writes."""

    domain_calls = _planned_calls(
        len(models),
        args.classification_batch_size,
    )

    pairs, raw_pair_count = _candidate_pairs(
        models,
        assignments={},
        top_k=args.top_k_neighbours,
        minimum_similarity=(
            args.minimum_similarity
        ),
        max_candidate_pairs=(
            args.max_candidate_pairs
        ),
    )

    relationship_calls = _planned_calls(
        len(pairs),
        args.pairs_per_call,
    )

    print(
        f"canonical_models={len(models)}, "
        f"planned_domain_calls={domain_calls}, "
        f"embedding_candidate_pairs_raw="
        f"{raw_pair_count}, "
        f"embedding_candidate_pairs_selected="
        f"{len(pairs)}, "
        f"planned_relationship_calls="
        f"{relationship_calls}"
    )

    if (
        args.max_candidate_pairs > 0
        and raw_pair_count > args.max_candidate_pairs
    ):
        print(
            f"Candidate cap applied: retained the "
            f"{args.max_candidate_pairs} "
            "highest-similarity pairs."
        )

    print(
        "The family-call count is known after domains are "
        "assigned and models are clustered within each "
        "primary domain."
    )

    print(
        "Dry run made no Luna calls, no database changes, "
        "and no JSONL export."
    )

    return 0


def main() -> int:
    """Run the complete hierarchy pass."""

    args = parse_arguments()

    models = _load_models()

    if not models:
        print(
            "No embedded canonical models found."
        )
        return 0

    if args.dry_run:
        return _dry_run(
            models=models,
            args=args,
        )

    provider = OpenAIStructuredProvider(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )

    print(
        f"Hierarchy pass: "
        f"canonical_models={len(models)}, "
        f"model={provider.model}.",
        flush=True,
    )

    domain_assignments = _classify_constitution(
        models=models,
        provider=provider,
        batch_size=(
            args.classification_batch_size
        ),
        max_output_tokens=args.max_output_tokens,
    )

    print(
        f"Domains assigned="
        f"{len(domain_assignments)}/{len(models)}.",
        flush=True,
    )

    family_clusters = _cluster_concept_families(
        models=models,
        domain_assignments=domain_assignments,
        similarity_threshold=(
            args.family_similarity_threshold
        ),
        max_family_size=args.max_family_size,
    )

    (
        family_by_model_code,
        omitted_family_clusters,
    ) = _name_concept_families(
        clusters=family_clusters,
        provider=provider,
        families_per_call=(
            args.families_per_call
        ),
        max_output_tokens=args.max_output_tokens,
    )

    assignments = _combine_assignments(
        models=models,
        domain_assignments=domain_assignments,
        family_by_model_code=(
            family_by_model_code
        ),
    )

    pairs, raw_pair_count = _candidate_pairs(
        models,
        assignments=assignments,
        top_k=args.top_k_neighbours,
        minimum_similarity=(
            args.minimum_similarity
        ),
        max_candidate_pairs=(
            args.max_candidate_pairs
        ),
    )

    print(
        f"Relationship candidate selection: "
        f"raw={raw_pair_count}, "
        f"selected={len(pairs)}, "
        f"cap={args.max_candidate_pairs}, "
        f"planned_luna_calls="
        f"{_planned_calls(len(pairs), args.pairs_per_call)}.",
        flush=True,
    )

    edges = _build_edges(
        pairs=pairs,
        provider=provider,
        pairs_per_call=args.pairs_per_call,
        max_output_tokens=args.max_output_tokens,
    )

    print(
        "Writing hierarchy to PostgreSQL...",
        flush=True,
    )

    database_started = time.perf_counter()

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

    print(
        f"Database transaction committed in "
        f"{time.perf_counter() - database_started:.1f}s.",
        flush=True,
    )

    print(
        f"Exporting updated canonical models to "
        f"{args.output_path}...",
        flush=True,
    )

    export_started = time.perf_counter()

    exported = export_canonical_jsonl(
        output_path=args.output_path,
        include_embeddings=(
            args.include_embeddings_in_export
        ),
    )

    print(
        f"Export completed in "
        f"{time.perf_counter() - export_started:.1f}s.",
        flush=True,
    )

    print(
        f"assigned={len(assignments)}/{len(models)}, "
        f"family_clusters={len(family_clusters)}, "
        f"family_assigned_models="
        f"{len(family_by_model_code)}, "
        f"omitted_family_clusters="
        f"{omitted_family_clusters}, "
        f"candidate_pairs_raw={raw_pair_count}, "
        f"candidate_pairs_selected={len(pairs)}, "
        f"edges={len(edges)}, "
        f"exported={exported}, "
        f"output={args.output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())