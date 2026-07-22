"""Cluster semantically similar mental-model fragments using their embedding vectors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class FragmentCluster:
    cluster_key: str
    fragments: tuple[Any, ...]


def _normalised_matrix(fragments: Sequence[Any]) -> np.ndarray:
    vectors: list[np.ndarray] = []

    for fragment in fragments:
        if fragment.embedding is None:
            raise ValueError(
                f"Fragment {fragment.fragment_code} has no embedding."
            )

        vector = np.asarray(fragment.embedding, dtype=np.float64)
        norm = np.linalg.norm(vector)

        if norm == 0.0:
            raise ValueError(
                f"Fragment {fragment.fragment_code} has a zero embedding."
            )

        vectors.append(vector / norm)

    return np.vstack(vectors)


def _cluster_key(fragments: Sequence[Any]) -> str:
    codes = sorted(str(fragment.fragment_code) for fragment in fragments)
    digest = hashlib.sha256("|".join(codes).encode("utf-8")).hexdigest()
    return f"cluster_{digest[:12]}"


def cluster_coherence(fragments: Sequence[Any]) -> float:
    if not fragments:
        raise ValueError("Cannot score an empty cluster.")

    if len(fragments) == 1:
        return 1.0

    matrix = _normalised_matrix(fragments)
    similarity = matrix @ matrix.T
    values: list[float] = []

    for left in range(len(fragments)):
        for right in range(left + 1, len(fragments)):
            values.append(float(similarity[left, right]))

    return sum(values) / len(values)


def cluster_fragments(
    fragments: list[Any],
    *,
    similarity_threshold: float = 0.84,
    max_cluster_size: int = 10,
) -> list[FragmentCluster]:
    """Deterministic complete-link greedy clustering."""

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [0, 1].")

    if max_cluster_size < 1:
        raise ValueError("max_cluster_size must be positive.")

    if not fragments:
        return []

    ordered = sorted(fragments, key=lambda item: item.fragment_code)
    matrix = _normalised_matrix(ordered)
    similarity = matrix @ matrix.T
    unassigned: set[int] = set(range(len(ordered)))
    output: list[FragmentCluster] = []

    while unassigned:
        seed = min(unassigned)
        candidates = sorted(
            unassigned - {seed},
            key=lambda index: (
                -float(similarity[seed, index]),
                ordered[index].fragment_code,
            ),
        )
        members = [seed]

        for candidate in candidates:
            if len(members) >= max_cluster_size:
                break

            if all(
                float(similarity[candidate, member])
                >= similarity_threshold
                for member in members
            ):
                members.append(candidate)

        for member in members:
            unassigned.remove(member)

        member_fragments = tuple(ordered[index] for index in members)
        output.append(
            FragmentCluster(
                cluster_key=_cluster_key(member_fragments),
                fragments=member_fragments,
            )
        )

    return output
