"""Generate semantic embeddings for mental-model fragments."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from collections.abc import Sequence

from mental_model_pipeline.fragments.constants import (
    EMBEDDING_DIMENSIONS,
)
from mental_model_pipeline.fragments.schemas import MentalModelFragment


EMBEDDING_MODEL = "text-embedding-3-large"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found: {ENV_PATH}")

load_dotenv(dotenv_path=ENV_PATH)

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")

client = OpenAI()


def build_embedding_text(
    fragment: MentalModelFragment,
) -> str:
    """
    Combine the conceptually important fragment fields into one string.
    """

    parts: list[str] = []

    if fragment.title:
        parts.append(f"Title: {fragment.title}")

    parts.append(f"Kind: {fragment.kind.value}")
    parts.append(f"Proposition: {fragment.proposition}")

    if fragment.mechanism:
        parts.append(
            "Mechanism: " + " | ".join(fragment.mechanism)
        )

    if fragment.conditions:
        parts.append(
            "Conditions: " + " | ".join(fragment.conditions)
        )

    if fragment.failure_conditions:
        parts.append(
            "Failure conditions: "
            + " | ".join(fragment.failure_conditions)
        )

    if fragment.decision_implications:
        parts.append(
            "Decision implications: "
            + " | ".join(fragment.decision_implications)
        )

    if fragment.decision_stages:
        parts.append(
            "Decision stages: "
            + " | ".join(
                stage.value
                for stage in fragment.decision_stages
            )
        )

    if fragment.contextual_regimes:
        parts.append(
            "Context: "
            + " | ".join(fragment.contextual_regimes)
        )

    return "\n".join(parts)


def create_embeddings(
    texts: Sequence[str],
) -> list[list[float]]:
    """
    Generate embeddings for several strings in one API request.
    """

    cleaned_texts = [
        text.strip()
        for text in texts
    ]

    if not cleaned_texts:
        return []

    if any(not text for text in cleaned_texts):
        raise ValueError("Cannot embed empty text.")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=cleaned_texts,
        dimensions=EMBEDDING_DIMENSIONS,
        encoding_format="float",
    )

    response_items = sorted(
        response.data,
        key=lambda item: item.index,
    )

    embeddings = [
        item.embedding
        for item in response_items
    ]

    if len(embeddings) != len(cleaned_texts):
        raise RuntimeError(
            "Embedding response count does not match input count."
        )

    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                "Unexpected embedding dimensions: "
                f"expected {EMBEDDING_DIMENSIONS}, "
                f"received {len(embedding)}"
            )

    return embeddings


def create_embedding(
    text: str,
) -> list[float]:
    """
    Generate one embedding.
    """

    return create_embeddings([text])[0]


def create_fragment_embeddings(
    fragments: Sequence[MentalModelFragment],
) -> list[list[float]]:
    """
    Generate one conceptual embedding per fragment.
    """

    embedding_texts = [
        build_embedding_text(fragment)
        for fragment in fragments
    ]

    return create_embeddings(embedding_texts)
