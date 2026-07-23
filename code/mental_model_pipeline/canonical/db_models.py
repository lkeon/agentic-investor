"""Define SQLAlchemy tables for canonical mental models and weighted graph edges."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from mental_model_pipeline.database.base import Base
from mental_model_pipeline.fragments.constants import EMBEDDING_DIMENSIONS


class CanonicalMentalModelDB(Base):
    __tablename__ = "canonical_mental_models"

    canonical_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    canonical_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
    )
    investor_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    kind: Mapped[str] = mapped_column(String(60), nullable=False)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    proposition: Mapped[str] = mapped_column(Text, nullable=False)
    mechanism: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        default=list,
    )
    conditions: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        default=list,
    )
    failure_conditions: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        default=list,
    )
    decision_implications: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        default=list,
    )
    decision_stages: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        default=list,
    )
    contextual_regimes: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        default=list,
    )
    supporting_fragment_codes: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
    )

    evidence_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
    )
    investor_importance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
    )
    base_weight: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
    )

    primary_domain: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        default="unassigned",
        index=True,
    )
    secondary_domains: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        default=list,
    )
    concept_family: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=True,
    )
    embedding_model: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
    )

    canonicalisation_model: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
    )
    canonicalisation_prompt_version: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class CanonicalModelEdgeDB(Base):
    __tablename__ = "canonical_model_edges"
    __table_args__ = (
        UniqueConstraint(
            "source_canonical_id",
            "target_canonical_id",
            "relation_type",
            name="uq_canonical_model_edge",
        ),
    )

    edge_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source_canonical_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "canonical_mental_models.canonical_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )
    target_canonical_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "canonical_mental_models.canonical_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )
    relation_type: Mapped[str] = mapped_column(
        String(60),
        nullable=False,
        index=True,
    )
    relation_strength: Mapped[float] = mapped_column(Float, nullable=False)
    relation_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    candidate_similarity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    scope: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
