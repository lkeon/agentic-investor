"""Build investor-specific canonical mental models from clustered fragments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from mental_model_pipeline.canonical.clustering import (
    FragmentCluster,
    cluster_coherence,
    cluster_fragments,
)
from mental_model_pipeline.canonical.embeddings import (
    CANONICAL_EMBEDDING_MODEL,
    create_canonical_embeddings,
)
from mental_model_pipeline.canonical.export_jsonl import (
    DEFAULT_OUTPUT_PATH,
    export_canonical_jsonl,
)
from mental_model_pipeline.canonical.prompts import (
    CANONICAL_PROMPT_VERSION,
    CANONICAL_SYSTEM_PROMPT,
)
from mental_model_pipeline.canonical.providers import (
    OpenAIStructuredProvider,
)
from mental_model_pipeline.canonical.repository import (
    investor_has_canonicals,
    replace_investor_canonicals,
)
from mental_model_pipeline.canonical.schemas import (
    CanonicalisationBatchResult,
    CanonicalModelData,
    CanonicalModelDraft,
)
from mental_model_pipeline.canonical.scoring import calculate_scores
from mental_model_pipeline.database.connection import SessionLocal
from mental_model_pipeline.fragments.db_models import (
    MentalModelFragmentDB,
)


@dataclass
class RepairCounters:
    """Track deterministic repairs made to provider output."""

    invalid_codes_removed: int = 0
    duplicate_assignments_removed: int = 0
    omitted_fragments_promoted: int = 0
    omitted_clusters_promoted: int = 0

    def add(self, other: RepairCounters) -> None:
        """Add another set of repair counters."""

        self.invalid_codes_removed += other.invalid_codes_removed
        self.duplicate_assignments_removed += (
            other.duplicate_assignments_removed
        )
        self.omitted_fragments_promoted += (
            other.omitted_fragments_promoted
        )
        self.omitted_clusters_promoted += (
            other.omitted_clusters_promoted
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Create investor-specific canonical mental models from embedded "
            "mental-model fragments."
        )
    )
    parser.add_argument("--investor-id")
    parser.add_argument(
        "--model",
        default=os.getenv("CANONICAL_MODEL", "gpt-5.6-luna"),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("CANONICAL_REASONING_EFFORT", "low"),
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.84,
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--clusters-per-call",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Transactionally replace existing canonical models.",
    )
    parser.add_argument(
        "--include-singletons-in-llm",
        action="store_true",
        help=(
            "By default singleton fragments are promoted without "
            "an API call."
        ),
    )
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


def _batches(
    values: list[Any],
    size: int,
) -> Iterable[list[Any]]:
    """Yield fixed-size batches."""

    if size < 1:
        raise ValueError("Batch size must be positive.")

    for start in range(0, len(values), size):
        yield values[start : start + size]


def _normalise_investor_code(investor_id: str) -> str:
    """Convert an investor ID into a canonical-code-safe value."""

    value = re.sub(r"[^a-z0-9]", "", investor_id.lower())

    if not value:
        raise ValueError("Invalid investor_id.")

    return value


def _canonical_code(
    *,
    investor_id: str,
    supporting_codes: list[str],
    proposition: str,
) -> str:
    """Create a deterministic canonical-model code."""

    material = (
        investor_id
        + "|"
        + "|".join(sorted(supporting_codes))
        + "|"
        + proposition.strip().casefold()
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()

    return (
        f"mmc_{_normalise_investor_code(investor_id)}_"
        f"{digest[:8]}"
    )


def _fragment_card(
    fragment: MentalModelFragmentDB,
) -> dict[str, Any]:
    """Build the compact fragment representation sent to the LLM."""

    return {
        "fragment_code": fragment.fragment_code,
        "kind": fragment.kind,
        "title": fragment.title,
        "proposition": fragment.proposition,
        "mechanism": list(fragment.mechanism),
        "conditions": list(fragment.conditions),
        "failure_conditions": list(fragment.failure_conditions),
        "decision_implications": list(
            fragment.decision_implications
        ),
        "decision_stages": list(fragment.decision_stages),
        "contextual_regimes": list(fragment.contextual_regimes),
    }


def _singleton_draft(
    fragment: MentalModelFragmentDB,
) -> CanonicalModelDraft:
    """Promote one fragment directly into a canonical-model draft."""

    return CanonicalModelDraft(
        kind=fragment.kind,
        title=fragment.title or fragment.proposition[:120],
        proposition=fragment.proposition,
        mechanism=list(fragment.mechanism),
        conditions=list(fragment.conditions),
        failure_conditions=list(fragment.failure_conditions),
        decision_implications=list(
            fragment.decision_implications
        ),
        decision_stages=list(fragment.decision_stages),
        contextual_regimes=list(fragment.contextual_regimes),
        supporting_fragment_codes=[fragment.fragment_code],
    )


def _provider_prompt(
    *,
    investor_id: str,
    clusters: list[FragmentCluster],
) -> str:
    """Build the compact JSON prompt for a batch of clusters."""

    return json.dumps(
        {
            "investor_id": investor_id,
            "clusters": [
                {
                    "cluster_key": cluster.cluster_key,
                    "fragments": [
                        _fragment_card(fragment)
                        for fragment in cluster.fragments
                    ],
                }
                for cluster in clusters
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _repair_cluster_result(
    *,
    cluster: FragmentCluster,
    returned_models: list[CanonicalModelDraft],
) -> tuple[list[CanonicalModelDraft], RepairCounters]:
    """Repair invalid, duplicate, or omitted fragment assignments."""

    counters = RepairCounters()

    allowed = {
        fragment.fragment_code
        for fragment in cluster.fragments
    }
    fragment_by_code = {
        fragment.fragment_code: fragment
        for fragment in cluster.fragments
    }

    assigned: set[str] = set()
    repaired: list[CanonicalModelDraft] = []

    for draft in returned_models:
        accepted: list[str] = []

        for code in draft.supporting_fragment_codes:
            if code not in allowed:
                counters.invalid_codes_removed += 1
                continue

            if code in assigned:
                counters.duplicate_assignments_removed += 1
                continue

            accepted.append(code)
            assigned.add(code)

        if accepted:
            repaired.append(
                draft.model_copy(
                    update={
                        "supporting_fragment_codes": accepted,
                    }
                )
            )

    for code in sorted(allowed - assigned):
        repaired.append(
            _singleton_draft(fragment_by_code[code])
        )
        counters.omitted_fragments_promoted += 1

    return repaired, counters


def _load_fragments_by_investor(
    investor_filter: str | None,
) -> dict[str, list[MentalModelFragmentDB]]:
    """Load embedded fragments grouped by investor."""

    with SessionLocal() as session:
        statement = (
            select(MentalModelFragmentDB)
            .where(
                MentalModelFragmentDB.embedding.is_not(None)
            )
            .order_by(
                MentalModelFragmentDB.investor_id,
                MentalModelFragmentDB.fragment_code,
            )
        )

        if investor_filter:
            statement = statement.where(
                MentalModelFragmentDB.investor_id
                == investor_filter
            )

        output: dict[
            str,
            list[MentalModelFragmentDB],
        ] = {}

        for fragment in session.scalars(statement):
            output.setdefault(
                fragment.investor_id,
                [],
            ).append(fragment)

        return output


def _build_models_for_investor(
    *,
    investor_id: str,
    fragments: list[MentalModelFragmentDB],
    provider: OpenAIStructuredProvider,
    similarity_threshold: float,
    max_cluster_size: int,
    clusters_per_call: int,
    max_output_tokens: int,
    include_singletons_in_llm: bool,
) -> tuple[list[CanonicalModelData], RepairCounters]:
    """Cluster fragments, call Luna, and construct canonical models."""

    print(
        f"{investor_id}: clustering {len(fragments)} fragments "
        f"(threshold={similarity_threshold:.2f}, "
        f"max_cluster_size={max_cluster_size})...",
        flush=True,
    )

    clustering_started = time.perf_counter()

    clusters = cluster_fragments(
        fragments,
        similarity_threshold=similarity_threshold,
        max_cluster_size=max_cluster_size,
    )

    clustering_seconds = (
        time.perf_counter() - clustering_started
    )

    direct_clusters: list[FragmentCluster] = []
    paid_clusters: list[FragmentCluster] = []

    for cluster in clusters:
        if (
            len(cluster.fragments) == 1
            and not include_singletons_in_llm
        ):
            direct_clusters.append(cluster)
        else:
            paid_clusters.append(cluster)

    luna_call_count = (
        len(paid_clusters) + clusters_per_call - 1
    ) // clusters_per_call

    print(
        f"{investor_id}: clustering complete in "
        f"{clustering_seconds:.1f}s — "
        f"clusters={len(clusters)}, "
        f"singletons_without_llm={len(direct_clusters)}, "
        f"clusters_for_luna={len(paid_clusters)}, "
        f"planned_luna_calls={luna_call_count}.",
        flush=True,
    )

    drafts_by_cluster: dict[
        str,
        list[CanonicalModelDraft],
    ] = {
        cluster.cluster_key: [
            _singleton_draft(cluster.fragments[0])
        ]
        for cluster in direct_clusters
    }

    counters = RepairCounters()

    if direct_clusters:
        print(
            f"{investor_id}: promoted "
            f"{len(direct_clusters)} singleton fragments "
            "directly without calling Luna.",
            flush=True,
        )

    paid_batches = list(
        _batches(
            paid_clusters,
            clusters_per_call,
        )
    )

    for call_number, cluster_batch in enumerate(
        paid_batches,
        start=1,
    ):
        batch_fragment_count = sum(
            len(cluster.fragments)
            for cluster in cluster_batch
        )

        print(
            f"{investor_id}: Luna call "
            f"{call_number}/{luna_call_count} starting — "
            f"clusters={len(cluster_batch)}, "
            f"fragments={batch_fragment_count}...",
            flush=True,
        )

        call_started = time.perf_counter()

        result = provider.parse(
            schema=CanonicalisationBatchResult,
            system_prompt=CANONICAL_SYSTEM_PROMPT,
            user_prompt=_provider_prompt(
                investor_id=investor_id,
                clusters=cluster_batch,
            ),
            max_output_tokens=max_output_tokens,
        )

        call_seconds = time.perf_counter() - call_started

        returned_model_count = sum(
            len(cluster_result.models)
            for cluster_result in result.clusters
        )

        print(
            f"{investor_id}: Luna call "
            f"{call_number}/{luna_call_count} completed "
            f"in {call_seconds:.1f}s — "
            f"returned_clusters={len(result.clusters)}, "
            f"returned_models={returned_model_count}.",
            flush=True,
        )

        result_by_key = {
            cluster_result.cluster_key: cluster_result
            for cluster_result in result.clusters
        }

        for cluster in cluster_batch:
            cluster_result = result_by_key.get(
                cluster.cluster_key
            )

            if cluster_result is None:
                drafts_by_cluster[cluster.cluster_key] = [
                    _singleton_draft(fragment)
                    for fragment in cluster.fragments
                ]

                counters.omitted_clusters_promoted += 1
                counters.omitted_fragments_promoted += len(
                    cluster.fragments
                )
                continue

            repaired, cluster_counters = (
                _repair_cluster_result(
                    cluster=cluster,
                    returned_models=cluster_result.models,
                )
            )

            drafts_by_cluster[
                cluster.cluster_key
            ] = repaired

            counters.add(cluster_counters)

    print(
        f"{investor_id}: assembling canonical models "
        "and calculating weights...",
        flush=True,
    )

    fragment_by_code = {
        fragment.fragment_code: fragment
        for fragment in fragments
    }

    output: list[CanonicalModelData] = []

    for cluster in clusters:
        for draft in drafts_by_cluster[
            cluster.cluster_key
        ]:
            supporting = [
                fragment_by_code[code]
                for code in draft.supporting_fragment_codes
            ]

            scores = calculate_scores(
                supporting,
                cluster_coherence=cluster_coherence(
                    supporting
                ),
            )

            output.append(
                CanonicalModelData(
                    canonical_code=_canonical_code(
                        investor_id=investor_id,
                        supporting_codes=(
                            draft.supporting_fragment_codes
                        ),
                        proposition=draft.proposition,
                    ),
                    investor_id=investor_id,
                    kind=draft.kind,
                    title=draft.title,
                    proposition=draft.proposition,
                    mechanism=draft.mechanism,
                    conditions=draft.conditions,
                    failure_conditions=(
                        draft.failure_conditions
                    ),
                    decision_implications=(
                        draft.decision_implications
                    ),
                    decision_stages=draft.decision_stages,
                    contextual_regimes=(
                        draft.contextual_regimes
                    ),
                    supporting_fragment_codes=(
                        draft.supporting_fragment_codes
                    ),
                    evidence_confidence=(
                        scores.evidence_confidence
                    ),
                    investor_importance=(
                        scores.investor_importance
                    ),
                    base_weight=scores.base_weight,
                )
            )

    print(
        f"{investor_id}: assembled {len(output)} "
        f"canonical models from {len(fragments)} fragments.",
        flush=True,
    )

    return output, counters


def main() -> int:
    """Run canonicalisation for one investor or all investors."""

    args = parse_arguments()

    fragments_by_investor = _load_fragments_by_investor(
        args.investor_id
    )

    if not fragments_by_investor:
        print("No embedded fragments were selected.")
        return 0

    if args.dry_run:
        for investor_id, fragments in (
            fragments_by_investor.items()
        ):
            clusters = cluster_fragments(
                fragments,
                similarity_threshold=(
                    args.similarity_threshold
                ),
                max_cluster_size=args.max_cluster_size,
            )

            singletons = sum(
                len(cluster.fragments) == 1
                for cluster in clusters
            )

            paid_clusters = (
                len(clusters)
                if args.include_singletons_in_llm
                else len(clusters) - singletons
            )

            estimated_calls = (
                paid_clusters
                + args.clusters_per_call
                - 1
            ) // args.clusters_per_call

            print(
                f"{investor_id}: "
                f"fragments={len(fragments)}, "
                f"clusters={len(clusters)}, "
                f"singletons={singletons}, "
                f"paid_clusters={paid_clusters}, "
                f"estimated_calls={estimated_calls}"
            )

        return 0

    provider = OpenAIStructuredProvider(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )

    for investor_id, fragments in (
        fragments_by_investor.items()
    ):
        with SessionLocal() as session:
            already_exists = investor_has_canonicals(
                session,
                investor_id,
            )

        if already_exists and not args.replace:
            print(
                f"{investor_id}: skipped; canonical models "
                "already exist. Use --replace to rebuild them."
            )
            continue

        print(
            f"{investor_id}: canonicalising "
            f"{len(fragments)} fragments "
            f"with {provider.model}...",
            flush=True,
        )

        models, counters = _build_models_for_investor(
            investor_id=investor_id,
            fragments=fragments,
            provider=provider,
            similarity_threshold=(
                args.similarity_threshold
            ),
            max_cluster_size=args.max_cluster_size,
            clusters_per_call=args.clusters_per_call,
            max_output_tokens=args.max_output_tokens,
            include_singletons_in_llm=(
                args.include_singletons_in_llm
            ),
        )

        print(
            f"{investor_id}: generating embeddings for "
            f"{len(models)} canonical models...",
            flush=True,
        )

        embedding_started = time.perf_counter()

        embeddings = create_canonical_embeddings(models)

        print(
            f"{investor_id}: embeddings completed in "
            f"{time.perf_counter() - embedding_started:.1f}s.",
            flush=True,
        )

        print(
            f"{investor_id}: writing canonical models "
            "to PostgreSQL in one transaction...",
            flush=True,
        )

        database_started = time.perf_counter()

        with SessionLocal() as session:
            try:
                replace_investor_canonicals(
                    session,
                    investor_id=investor_id,
                    models=models,
                    embeddings=embeddings,
                    embedding_model=(
                        CANONICAL_EMBEDDING_MODEL
                    ),
                    canonicalisation_model=provider.model,
                    prompt_version=(
                        CANONICAL_PROMPT_VERSION
                    ),
                )

                session.commit()

            except Exception:
                session.rollback()
                raise

        print(
            f"{investor_id}: database transaction "
            f"committed in "
            f"{time.perf_counter() - database_started:.1f}s.",
            flush=True,
        )

        print(
            f"{investor_id}: "
            f"stored={len(models)}, "
            f"invalid_codes_removed="
            f"{counters.invalid_codes_removed}, "
            f"duplicate_assignments_removed="
            f"{counters.duplicate_assignments_removed}, "
            f"omitted_fragments_promoted="
            f"{counters.omitted_fragments_promoted}, "
            f"omitted_clusters_promoted="
            f"{counters.omitted_clusters_promoted}"
        )

    print(
        f"Exporting canonical models to "
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
        f"Exported {exported} canonical models to "
        f"{args.output_path} in "
        f"{time.perf_counter() - export_started:.1f}s.",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())