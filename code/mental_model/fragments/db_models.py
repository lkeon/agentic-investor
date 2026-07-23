"""SQLAlchemy models for documents, fragments, and related entities."""

from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from mental_model.database.base import Base


class DocumentDB(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(
        String(200),
        primary_key=True,
    )

    investor_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    file_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    content_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    markdown_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class MentalModelFragmentDB(Base):
    __tablename__ = "mental_model_fragments"

    fragment_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    fragment_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
    )

    document_id: Mapped[str] = mapped_column(
        ForeignKey(
            "documents.document_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    investor_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    kind: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    title: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    proposition: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    mechanism: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        default=list,
    )

    conditions: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        default=list,
    )

    failure_conditions: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        default=list,
    )

    decision_implications: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        default=list,
    )

    decision_stages: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)),
        nullable=False,
        default=list,
    )

    contextual_regimes: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        default=list,
    )

    source_quote: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    evidence_strength: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    attribution_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    attributed_to: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    requires_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    review_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1024),
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class RelatedEntityDB(Base):
    __tablename__ = "related_entities"

    entity_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    fragment_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "mental_model_fragments.fragment_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    entity_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
        index=True,
    )

    relation: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    period_text: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
    )

    explanation: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
