"""Populate the database with sample fragment data."""

from hashlib import sha256

from sqlalchemy import select

from mental_model.database.connection import SessionLocal
from mental_model.fragments.db_models import (
    DocumentDB,
    MentalModelFragmentDB,
    RelatedEntityDB,
)
from mental_model.fragments.repository import (
    add_document,
    add_fragment,
)
from mental_model.fragments.schemas import (
    AttributionType,
    DecisionStage,
    EntityRelation,
    EntityType,
    EvidenceStrength,
    FragmentKind,
    MentalModelFragment,
    RelatedEntity,
)


def create_dummy_fragment() -> MentalModelFragment:
    return MentalModelFragment(
        fragment_code="mmf_buffett_001",
        kind=FragmentKind.PRINCIPLE,
        title="Pricing power",
        proposition=(
            "A business with durable pricing power can protect "
            "its economics during inflation."
        ),
        mechanism=[
            "Prices can increase without a proportionate loss of customers.",
            "Revenue can rise without equivalent additional capital investment.",
        ],
        conditions=[
            "Strong customer loyalty",
            "Low substitution risk",
        ],
        failure_conditions=[
            "Government price controls",
            "Brand deterioration",
        ],
        decision_implications=[
            "Increase the assessment of business quality.",
            "Investigate whether pricing power is durable.",
        ],
        decision_stages=[
            DecisionStage.BUSINESS_QUALITY,
            DecisionStage.VALUATION,
        ],
        contextual_regimes=[
            "inflation",
        ],
        related_entities=[
            RelatedEntity(
                entity_type=EntityType.COMPANY,
                name="See's Candies",
                relation=EntityRelation.POSITIVE_EXAMPLE,
                period_text="1972 onward",
                explanation=(
                    "Illustrates pricing power and low incremental "
                    "capital requirements."
                ),
            )
        ],
        source_quote=(
            "A truly great business must have an enduring moat "
            "that protects excellent returns on invested capital."
        ),
        evidence_strength=EvidenceStrength.DIRECTLY_STATED,
        attribution_type=AttributionType.INVESTOR,
        attributed_to=None,
        requires_review=False,
        review_reason=None,
    )


def populate_dummy_data() -> None:
    markdown_text = """
# Test Berkshire Letter

A truly great business must have an enduring moat that protects
excellent returns on invested capital.
""".strip()

    document_hash = sha256(
        markdown_text.encode("utf-8")
    ).hexdigest()

    fragment = create_dummy_fragment()

    with SessionLocal() as session:
        try:
            add_document(
                session,
                document_id="doc_buffett_test",
                investor_id="buffett",
                file_path="data/test/buffett_test.md",
                content_sha256=document_hash,
                markdown_text=markdown_text,
            )

            add_fragment(
                session,
                fragment=fragment,
                document_id="doc_buffett_test",
                investor_id="buffett",
                embedding=None,
                embedding_model=None,
            )

            session.commit()

        except Exception:
            session.rollback()
            raise

    print("Dummy data inserted successfully.")


def print_dummy_data() -> None:
    with SessionLocal() as session:
        document = session.get(
            DocumentDB,
            "doc_buffett_test",
        )

        fragment = session.execute(
            select(MentalModelFragmentDB).where(
                MentalModelFragmentDB.fragment_code
                == "mmf_buffett_001"
            )
        ).scalar_one()

        entities = session.execute(
            select(RelatedEntityDB).where(
                RelatedEntityDB.fragment_id
                == fragment.fragment_id
            )
        ).scalars().all()

        print()
        print("Document")
        print("--------")
        print("ID:", document.document_id)
        print("Investor:", document.investor_id)

        print()
        print("Fragment")
        print("--------")
        print("Code:", fragment.fragment_code)
        print("Title:", fragment.title)
        print("Proposition:", fragment.proposition)
        print("Kind:", fragment.kind)

        print()
        print("Related entities")
        print("----------------")

        for entity in entities:
            print(
                f"{entity.name}: {entity.relation}"
            )


if __name__ == "__main__":
    populate_dummy_data()
    print_dummy_data()
