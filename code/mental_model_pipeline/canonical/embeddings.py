"""Create canonical-model embeddings through a replaceable provider."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

from mental_model_pipeline.fragments.constants import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)


CANONICAL_EMBEDDING_PROVIDER = os.getenv(
    "CANONICAL_EMBEDDING_PROVIDER",
    os.getenv("EMBEDDING_PROVIDER", "openai"),
)
CANONICAL_EMBEDDING_MODEL = os.getenv(
    "CANONICAL_EMBEDDING_MODEL",
    os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL),
)
CANONICAL_EMBEDDING_DIMENSIONS = int(
    os.getenv("CANONICAL_EMBEDDING_DIMENSIONS", str(EMBEDDING_DIMENSIONS))
)

if CANONICAL_EMBEDDING_DIMENSIONS != EMBEDDING_DIMENSIONS:
    raise RuntimeError(
        "The canonical pgvector column is fixed at "
        f"{EMBEDDING_DIMENSIONS} dimensions; requested "
        f"{CANONICAL_EMBEDDING_DIMENSIONS}. A database migration is required "
        "before changing dimensions."
    )


def make_embedding_identity(
    provider: str,
    model: str,
    dimensions: int,
) -> str:
    """Build the provenance value stored with every canonical vector."""

    return f"{provider}/{model}/{dimensions}"


CANONICAL_EMBEDDING_IDENTITY = make_embedding_identity(
    CANONICAL_EMBEDDING_PROVIDER,
    CANONICAL_EMBEDDING_MODEL,
    CANONICAL_EMBEDDING_DIMENSIONS,
)


class EmbeddingProvider(Protocol):
    """Small provider seam; only OpenAI is implemented for the MVP."""

    identity: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingProvider:
    """OpenAI implementation of the canonical embedding contract."""

    def __init__(self, *, model: str, dimensions: int) -> None:
        self.model = model
        self.dimensions = dimensions
        self.identity = make_embedding_identity(
            "openai",
            model,
            dimensions,
        )
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts]
        if not cleaned:
            return []
        if any(not text for text in cleaned):
            raise ValueError("Cannot embed empty text.")

        response = self.client.embeddings.create(
            model=self.model,
            input=cleaned,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        items = sorted(response.data, key=lambda item: item.index)
        embeddings = [item.embedding for item in items]

        if len(embeddings) != len(cleaned):
            raise RuntimeError("Canonical embedding response count mismatch.")
        for embedding in embeddings:
            if len(embedding) != self.dimensions:
                raise RuntimeError(
                    f"Expected {self.dimensions} dimensions, "
                    f"received {len(embedding)}."
                )
        return embeddings


def create_embedding_provider() -> EmbeddingProvider:
    """Create the active provider from global environment configuration."""

    if CANONICAL_EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddingProvider(
            model=CANONICAL_EMBEDDING_MODEL,
            dimensions=CANONICAL_EMBEDDING_DIMENSIONS,
        )
    raise ValueError(
        "Unsupported canonical embedding provider: "
        f"{CANONICAL_EMBEDDING_PROVIDER!r}. Implement an EmbeddingProvider "
        "adapter before selecting it."
    )


def accepted_embedding_identities() -> set[str]:
    """Include the legacy OpenAI model-only value during migration."""

    identities = {CANONICAL_EMBEDDING_IDENTITY}
    if CANONICAL_EMBEDDING_PROVIDER == "openai":
        identities.add(CANONICAL_EMBEDDING_MODEL)
    return identities


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
    """Generate embeddings using the active configured provider."""

    return create_embedding_provider().embed(texts)


def create_canonical_embeddings(models: Sequence) -> list[list[float]]:
    return create_embeddings(
        [build_canonical_embedding_text(model) for model in models]
    )
