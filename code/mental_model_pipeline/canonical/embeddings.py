"""Create compact embedding text and vectors for canonical mental models."""

from __future__ import annotations

import os
from collections.abc import Sequence

from openai import OpenAI

from mental_model_pipeline.fragments.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
)


CANONICAL_EMBEDDING_MODEL = os.getenv(
    "CANONICAL_EMBEDDING_MODEL",
    EMBEDDING_MODEL,
)
CANONICAL_EMBEDDING_DIMENSIONS = EMBEDDING_DIMENSIONS

_client = OpenAI()


def build_canonical_embedding_text(model) -> str:
    parts = [
        f"Title: {model.title}",
        f"Kind: {model.kind.value}",
        f"Proposition: {model.proposition}",
    ]

    if model.mechanism:
        parts.append("Mechanism: " + " | ".join(model.mechanism))
    if model.conditions:
        parts.append("Conditions: " + " | ".join(model.conditions))
    if model.failure_conditions:
        parts.append(
            "Failure conditions: " + " | ".join(model.failure_conditions)
        )
    if model.decision_implications:
        parts.append(
            "Decision implications: "
            + " | ".join(model.decision_implications)
        )
    if model.decision_stages:
        parts.append(
            "Decision stages: "
            + " | ".join(stage.value for stage in model.decision_stages)
        )
    if model.contextual_regimes:
        parts.append("Context: " + " | ".join(model.contextual_regimes))

    return "\n".join(parts)


def create_embeddings(texts: Sequence[str]) -> list[list[float]]:
    cleaned = [text.strip() for text in texts]

    if not cleaned:
        return []
    if any(not text for text in cleaned):
        raise ValueError("Cannot embed empty text.")

    response = _client.embeddings.create(
        model=CANONICAL_EMBEDDING_MODEL,
        input=cleaned,
        dimensions=CANONICAL_EMBEDDING_DIMENSIONS,
        encoding_format="float",
    )
    items = sorted(response.data, key=lambda item: item.index)
    embeddings = [item.embedding for item in items]

    if len(embeddings) != len(cleaned):
        raise RuntimeError("Canonical embedding response count mismatch.")

    for embedding in embeddings:
        if len(embedding) != CANONICAL_EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Expected {CANONICAL_EMBEDDING_DIMENSIONS} dimensions, "
                f"received {len(embedding)}."
            )

    return embeddings


def create_canonical_embeddings(models: Sequence) -> list[list[float]]:
    return create_embeddings(
        [build_canonical_embedding_text(model) for model in models]
    )
