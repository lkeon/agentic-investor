"""Export canonical mental models and their graph relationships from PostgreSQL to JSONL."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select

from mental_model_pipeline.canonical.db_models import (
    CanonicalMentalModelDB,
    CanonicalModelEdgeDB,
)
from mental_model_pipeline.database.connection import SessionLocal


DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "canonical"
    / "canonical_mental_models.jsonl"
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def build_export_records(
    *,
    include_embeddings: bool = False,
) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        models = list(
            session.scalars(
                select(CanonicalMentalModelDB).order_by(
                    CanonicalMentalModelDB.investor_id,
                    CanonicalMentalModelDB.canonical_code,
                )
            )
        )
        code_by_id = {
            model.canonical_id: model.canonical_code for model in models
        }
        outgoing: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        incoming: dict[Any, list[dict[str, Any]]] = defaultdict(list)

        if models:
            edges = list(
                session.scalars(
                    select(CanonicalModelEdgeDB).order_by(
                        CanonicalModelEdgeDB.source_canonical_id,
                        CanonicalModelEdgeDB.target_canonical_id,
                        CanonicalModelEdgeDB.relation_type,
                    )
                )
            )

            for edge in edges:
                common = {
                    "relation_type": edge.relation_type,
                    "relation_strength": edge.relation_strength,
                    "relation_confidence": edge.relation_confidence,
                    "candidate_similarity": edge.candidate_similarity,
                    "scope": edge.scope,
                }
                outgoing[edge.source_canonical_id].append(
                    {
                        **common,
                        "target_canonical_code": code_by_id[
                            edge.target_canonical_id
                        ],
                    }
                )
                incoming[edge.target_canonical_id].append(
                    {
                        **common,
                        "source_canonical_code": code_by_id[
                            edge.source_canonical_id
                        ],
                    }
                )

        output: list[dict[str, Any]] = []

        for model in models:
            embedding = None
            if include_embeddings and model.embedding is not None:
                embedding = [float(value) for value in model.embedding]

            output.append(
                {
                    "schema_version": "canonical_mental_model_mvp_v1",
                    "canonical_id": str(model.canonical_id),
                    "canonical_code": model.canonical_code,
                    "investor_id": model.investor_id,
                    "kind": model.kind,
                    "title": model.title,
                    "proposition": model.proposition,
                    "mechanism": list(model.mechanism),
                    "conditions": list(model.conditions),
                    "failure_conditions": list(model.failure_conditions),
                    "decision_implications": list(
                        model.decision_implications
                    ),
                    "decision_stages": list(model.decision_stages),
                    "contextual_regimes": list(model.contextual_regimes),
                    "supporting_fragment_codes": list(
                        model.supporting_fragment_codes
                    ),
                    "evidence_confidence": model.evidence_confidence,
                    "investor_importance": model.investor_importance,
                    "base_weight": model.base_weight,
                    "constitution": {
                        "primary_domain": model.primary_domain,
                        "secondary_domains": list(model.secondary_domains),
                        "concept_family": model.concept_family,
                    },
                    "hierarchy": {
                        "outgoing": outgoing.get(model.canonical_id, []),
                        "incoming": incoming.get(model.canonical_id, []),
                    },
                    "canonicalisation_model": (
                        model.canonicalisation_model
                    ),
                    "canonicalisation_prompt_version": (
                        model.canonicalisation_prompt_version
                    ),
                    "embedding_model": model.embedding_model,
                    "embedding_dimensions": (
                        len(model.embedding)
                        if model.embedding is not None
                        else None
                    ),
                    "embedding": embedding,
                    "created_at": _json_value(model.created_at),
                    "updated_at": _json_value(model.updated_at),
                }
            )

        return output


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")

            handle.flush()
            os.fsync(handle.fileno())

        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def export_canonical_jsonl(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    include_embeddings: bool = False,
) -> int:
    records = build_export_records(
        include_embeddings=include_embeddings
    )
    write_jsonl_atomic(output_path, records)
    return len(records)
