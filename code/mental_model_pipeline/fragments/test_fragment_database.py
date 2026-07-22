"""Integration-test fragment and related-entity persistence."""

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from mental_model_pipeline.database.connection import SessionLocal
from mental_model_pipeline.fragments.db_models import (
    DocumentDB,
    MentalModelFragmentDB,
    RelatedEntityDB,
)
from mental_model_pipeline.fragments.repository import (
    add_document,
    add_fragment,
)
from mental_model_pipeline.fragments.schemas import (
    AttributionType,
    DecisionStage,
    EntityRelation,
    EntityType,
    EvidenceStrength,
    FragmentKind,
    MentalModelFragment,
    RelatedEntity,
)


TEST_DOCUMENT_ID = "doc_fragment_test"
TEST_FRAGMENT_CODE = "mmf_test_001"
TEST_INVESTOR_ID = "test_investor"

TEST_EMBEDDING_MODEL = "synthetic-test-embedding"
TEST_EMBEDDING_DIMENSIONS = 1024


def build_test_fragment() -> MentalModelFragment:
    """Create one valid mental-model fragment for database testing."""

    return MentalModelFragment(
        fragment_code=TEST_FRAGMENT_CODE,
        kind=FragmentKind.PRINCIPLE,
        title="Durable pricing power",
        proposition=(
            "A business with durable pricing power can protect its "
            "economics during periods of inflation."
        ),
        mechanism=[
            "The company can increase prices without losing many customers.",
            "Revenue can grow without equivalent additional capital investment.",
        ],
        conditions=[
            "Strong customer loyalty",
            "Limited availability of substitutes",
        ],
        failure_conditions=[
            "Brand deterioration",
            "Government price controls",
        ],
        decision_implications=[
            "Increase the assessment of business quality.",
            "Test whether historical price increases affected customer retention.",
        ],
        decision_stages=[
            DecisionStage.BUSINESS_QUALITY,
            DecisionStage.VALUATION,
        ],
        contextual_regimes=[
            "Inflation",
        ],
        related_entities=[
            RelatedEntity(
                entity_type=EntityType.COMPANY,
                name="Example Consumer Company",
                relation=EntityRelation.POSITIVE_EXAMPLE,
                period_text="Test period",
                explanation=(
                    "Used only to verify insertion into the "
                    "related_entities table."
                ),
            )
        ],
        source_quote=(
            "A business that can increase prices without losing customers "
            "may possess durable pricing power."
        ),
        evidence_strength=EvidenceStrength.DIRECTLY_STATED,
        attribution_type=AttributionType.INVESTOR,
        requires_review=False,
    )


def build_test_embedding() -> list[float]:
    """
    Create a deterministic 1024-dimensional vector.

    This avoids making a paid external API call during the database test.
    """

    return [
        float(index % 100) / 100.0
        for index in range(TEST_EMBEDDING_DIMENSIONS)
    ]


def remove_existing_test_data(session: Session) -> None:
    """
    Delete a previous test document.

    Foreign-key cascade should also delete its fragments and related entities.
    """

    session.execute(
        delete(DocumentDB).where(
            DocumentDB.document_id == TEST_DOCUMENT_ID
        )
    )
    session.commit()


def run_test() -> None:
    session = SessionLocal()

    try:
        # Ensure the test can be run repeatedly.
        remove_existing_test_data(session)

        add_document(
            session,
            document_id=TEST_DOCUMENT_ID,
            investor_id=TEST_INVESTOR_ID,
            file_path="tests/dummy_fragment_document.md",
            content_sha256="test_sha256_fragment_database",
            markdown_text=(
                "# Test document\n\n"
                "This document exists only for database testing."
            ),
        )

        fragment = build_test_fragment()
        embedding = build_test_embedding()

        add_fragment(
            session,
            fragment=fragment,
            document_id=TEST_DOCUMENT_ID,
            investor_id=TEST_INVESTOR_ID,
            embedding=embedding,
            embedding_model=TEST_EMBEDDING_MODEL,
        )

        session.commit()

        stored_fragment = session.scalar(
            select(MentalModelFragmentDB).where(
                MentalModelFragmentDB.fragment_code
                == TEST_FRAGMENT_CODE
            )
        )

        assert stored_fragment is not None
        assert stored_fragment.fragment_code == TEST_FRAGMENT_CODE
        assert stored_fragment.document_id == TEST_DOCUMENT_ID
        assert stored_fragment.investor_id == TEST_INVESTOR_ID
        assert stored_fragment.kind == FragmentKind.PRINCIPLE.value
        assert stored_fragment.title == "Durable pricing power"
        assert stored_fragment.embedding_model == TEST_EMBEDDING_MODEL

        assert stored_fragment.embedding is not None
        assert (
            len(stored_fragment.embedding)
            == TEST_EMBEDDING_DIMENSIONS
        )

        stored_entities = session.scalars(
            select(RelatedEntityDB).where(
                RelatedEntityDB.fragment_id
                == stored_fragment.fragment_id
            )
        ).all()

        assert len(stored_entities) == 1
        assert stored_entities[0].name == "Example Consumer Company"
        assert (
            stored_entities[0].relation
            == EntityRelation.POSITIVE_EXAMPLE.value
        )

        print("Mental-model fragment database test passed.")
        print(f"Fragment code: {stored_fragment.fragment_code}")
        print(f"Fragment ID: {stored_fragment.fragment_id}")
        print(
            "Embedding dimensions:",
            len(stored_fragment.embedding),
        )
        print("Related entities:", len(stored_entities))

    except Exception:
        session.rollback()
        raise

    finally:
        try:
            remove_existing_test_data(session)
        finally:
            session.close()


if __name__ == "__main__":
    run_test()
