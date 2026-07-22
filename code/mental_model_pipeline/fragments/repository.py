"""Database operations for documents and mental-model fragments."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from mental_model_pipeline.fragments.db_models import (
    DocumentDB,
    MentalModelFragmentDB,
    RelatedEntityDB,
)
from mental_model_pipeline.fragments.schemas import MentalModelFragment


def add_document(
    session: Session,
    *,
    document_id: str,
    investor_id: str,
    file_path: str,
    content_sha256: str,
    markdown_text: str,
) -> DocumentDB:
    """
    Insert a document unless it already exists.
    """

    existing_document = session.get(
        DocumentDB,
        document_id,
    )

    if existing_document is not None:
        return existing_document

    document = DocumentDB(
        document_id=document_id,
        investor_id=investor_id,
        file_path=file_path,
        content_sha256=content_sha256,
        markdown_text=markdown_text,
    )

    session.add(document)
    session.flush()

    return document


def add_fragment(
    session: Session,
    *,
    fragment: MentalModelFragment,
    document_id: str,
    investor_id: str,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
) -> MentalModelFragmentDB:
    """
    Insert one validated mental model fragment and its related entities.
    """

    if fragment.fragment_code is None:
        raise ValueError(
            "fragment.fragment_code must be assigned before database insertion."
        )

    existing_fragment = get_fragment_by_code(
        session,
        fragment.fragment_code,
    )

    if existing_fragment is not None:
        return existing_fragment

    database_fragment = MentalModelFragmentDB(
        fragment_code=fragment.fragment_code,
        document_id=document_id,
        investor_id=investor_id,
        kind=fragment.kind.value,
        title=fragment.title,
        proposition=fragment.proposition,
        mechanism=fragment.mechanism,
        conditions=fragment.conditions,
        failure_conditions=fragment.failure_conditions,
        decision_implications=fragment.decision_implications,
        decision_stages=[
            stage.value
            for stage in fragment.decision_stages
        ],
        contextual_regimes=fragment.contextual_regimes,
        source_quote=fragment.source_quote,
        evidence_strength=fragment.evidence_strength.value,
        attribution_type=fragment.attribution_type.value,
        attributed_to=fragment.attributed_to,
        requires_review=fragment.requires_review,
        review_reason=fragment.review_reason,
        embedding=embedding,
        embedding_model=embedding_model,
    )

    session.add(database_fragment)

    # Executes the INSERT without committing so that fragment_id is created.
    session.flush()

    for entity in fragment.related_entities:
        database_entity = RelatedEntityDB(
            fragment_id=database_fragment.fragment_id,
            entity_type=entity.entity_type.value,
            name=entity.name,
            relation=entity.relation.value,
            period_text=entity.period_text,
            explanation=entity.explanation,
        )

        session.add(database_entity)

    session.flush()

    return database_fragment


def get_fragment_by_code(
    session: Session,
    fragment_code: str,
) -> MentalModelFragmentDB | None:
    """
    Return one fragment by its human-readable code.
    """

    statement = select(MentalModelFragmentDB).where(
        MentalModelFragmentDB.fragment_code == fragment_code
    )

    return session.execute(statement).scalar_one_or_none()


def list_fragments_by_investor(
    session: Session,
    investor_id: str,
) -> list[MentalModelFragmentDB]:
    """
    Return all fragments belonging to one investor.
    """

    statement = (
        select(MentalModelFragmentDB)
        .where(
            MentalModelFragmentDB.investor_id == investor_id
        )
        .order_by(
            MentalModelFragmentDB.fragment_code
        )
    )

    return list(
        session.execute(statement).scalars().all()
    )


def delete_fragment(
    session: Session,
    fragment_code: str,
) -> bool:
    """
    Delete a fragment and its related entities.
    """

    fragment = get_fragment_by_code(
        session,
        fragment_code,
    )

    if fragment is None:
        return False

    session.delete(fragment)
    session.flush()

    return True
