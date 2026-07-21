from __future__ import annotations

from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class FragmentKind(str, Enum):
    PRINCIPLE = "principle"
    CAUSAL_CLAIM = "causal_claim"
    DECISION_RULE = "decision_rule"
    VALUATION_RULE = "valuation_rule"
    RISK_RULE = "risk_rule"
    PORTFOLIO_RULE = "portfolio_rule"
    BEHAVIOURAL_RULE = "behavioural_rule"
    CONDITION = "condition"
    EXCEPTION = "exception"
    OBSERVATION = "observation"


class EvidenceStrength(str, Enum):
    DIRECTLY_STATED = "directly_stated"
    STRONGLY_IMPLIED = "strongly_implied"
    WEAKLY_INFERRED = "weakly_inferred"


class AttributionType(str, Enum):
    INVESTOR = "investor"
    THIRD_PARTY_QUOTED_BY_INVESTOR = "third_party_quoted_by_investor"
    DOCUMENT_EDITOR = "document_editor"
    UNCLEAR = "unclear"


class DecisionStage(str, Enum):
    SCREENING = "screening"
    BUSINESS_QUALITY = "business_quality"
    MANAGEMENT = "management"
    VALUATION = "valuation"
    RISK = "risk"
    POSITION_SIZING = "position_sizing"
    PORTFOLIO_CONSTRUCTION = "portfolio_construction"
    MONITORING = "monitoring"
    SELL_DECISION = "sell_decision"
    MACRO_ASSESSMENT = "macro_assessment"


class EntityType(str, Enum):
    COMPANY = "company"
    PERSON = "person"
    INDUSTRY = "industry"
    ASSET = "asset"
    COUNTRY = "country"
    INSTITUTION = "institution"
    OTHER = "other"


class EntityRelation(str, Enum):
    POSITIVE_EXAMPLE = "positive_example"
    NEGATIVE_EXAMPLE = "negative_example"
    COUNTEREXAMPLE = "counterexample"
    BOUNDARY_CASE = "boundary_case"
    MODEL_ORIGIN = "model_origin"
    MODEL_APPLICATION = "model_application"
    MODEL_REVERSAL = "model_reversal"
    MENTIONED_ONLY = "mentioned_only"


class RelatedEntity(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    entity_type: EntityType = Field(
        description="Type of related entity.",
    )

    name: str = Field(
        min_length=1,
        description="Name of the entity.",
    )

    relation: EntityRelation = Field(
        description="Relationship to the fragment.",
    )

    period_text: str | None = Field(
        default=None,
        min_length=1,
        description="Relevant period, when stated.",
    )

    explanation: str | None = Field(
        default=None,
        min_length=1,
        description="Reason the entity is relevant.",
    )


class MentalModelFragment(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    fragment_code: str | None = Field(
        default=None,
        pattern=r"^mmf_[a-z0-9]+_[a-z0-9]{3}$",
        description="Application-generated fragment identifier.",
    )

    kind: FragmentKind = Field(
        description="Type of mental model fragment.",
    )

    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Short descriptive fragment title.",
    )

    proposition: str = Field(
        min_length=10,
        max_length=2000,
        description="Atomic investment claim supported by the source.",
    )

    mechanism: list[str] = Field(
        default_factory=list,
        description="Reasons the proposition is expected to hold.",
    )

    conditions: list[str] = Field(
        default_factory=list,
        description="Conditions required for the proposition to hold.",
    )

    failure_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions under which the proposition may fail.",
    )

    decision_implications: list[str] = Field(
        default_factory=list,
        description="Effects on an investment decision.",
    )

    decision_stages: list[DecisionStage] = Field(
        default_factory=list,
        description="Investment stages where the fragment applies.",
    )

    contextual_regimes: list[str] = Field(
        default_factory=list,
        description="Market or economic contexts where it applies.",
    )

    related_entities: list[RelatedEntity] = Field(
        default_factory=list,
        description="Entities illustrating or connected to the fragment.",
    )

    source_quote: str = Field(
        min_length=10,
        max_length=6000,
        description="Exact quotation supporting the fragment.",
    )

    evidence_strength: EvidenceStrength = Field(
        description="How directly the source supports the proposition.",
    )

    attribution_type: AttributionType = Field(
        description="Who originally expressed the proposition.",
    )

    attributed_to: str | None = Field(
        default=None,
        min_length=1,
        description="Named third party when applicable.",
    )

    requires_review: bool = Field(
        default=False,
        description="Whether the fragment requires manual review.",
    )

    review_reason: str | None = Field(
        default=None,
        min_length=1,
        description="Reason manual review is required.",
    )

    @field_validator(
        "mechanism",
        "conditions",
        "failure_conditions",
        "decision_implications",
        "contextual_regimes",
    )
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for value in values:
            value = value.strip()

            if not value:
                continue

            comparison_value = value.casefold()

            if comparison_value not in seen:
                cleaned.append(value)
                seen.add(comparison_value)

        return cleaned

    @model_validator(mode="after")
    def validate_review_and_attribution(self) -> MentalModelFragment:
        if self.requires_review and not self.review_reason:
            raise ValueError(
                "review_reason is required when requires_review is true"
            )

        if (
            self.attribution_type
            == AttributionType.THIRD_PARTY_QUOTED_BY_INVESTOR
            and not self.attributed_to
        ):
            raise ValueError(
                "attributed_to is required for third-party quotations"
            )

        return self


class MentalModelExtractionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    fragments: list[MentalModelFragment] = Field(
        default_factory=list,
        description="Mental model fragments extracted from the source text.",
    )
