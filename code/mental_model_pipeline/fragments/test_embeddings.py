from mental_model_pipeline.fragments.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    build_embedding_text,
    create_fragment_embedding,
)
from mental_model_pipeline.fragments.schemas import (
    AttributionType,
    DecisionStage,
    EvidenceStrength,
    FragmentKind,
    MentalModelFragment,
)


def create_test_fragment() -> MentalModelFragment:
    return MentalModelFragment(
        fragment_code="mmf_test_001",
        kind=FragmentKind.PRINCIPLE,
        title="Pricing power",
        proposition=(
            "A business with durable pricing power can protect "
            "its economics during inflation."
        ),
        mechanism=[
            "Prices can increase without a proportionate loss of customers.",
            "Revenue can rise without equivalent capital investment.",
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
            "Test whether pricing power is durable.",
        ],
        decision_stages=[
            DecisionStage.BUSINESS_QUALITY,
            DecisionStage.VALUATION,
        ],
        contextual_regimes=[
            "inflation",
        ],
        related_entities=[],
        source_quote=(
            "A truly great business must have an enduring moat "
            "that protects excellent returns on invested capital."
        ),
        evidence_strength=EvidenceStrength.DIRECTLY_STATED,
        attribution_type=AttributionType.INVESTOR,
        requires_review=False,
    )


def main() -> None:
    fragment = create_test_fragment()

    embedding_text = build_embedding_text(fragment)

    print("Embedding input")
    print("---------------")
    print(embedding_text)

    embedding = create_fragment_embedding(fragment)

    print()
    print("Embedding model:", EMBEDDING_MODEL)
    print("Embedding dimensions:", len(embedding))
    print("Expected dimensions:", EMBEDDING_DIMENSIONS)
    print("First five values:", embedding[:5])


if __name__ == "__main__":
    main()
