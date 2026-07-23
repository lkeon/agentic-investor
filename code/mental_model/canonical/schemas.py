"""Define schemas for canonical models, hierarchy placement, and graph edges."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from mental_model.fragments.schemas import (
    DecisionStage,
    FragmentKind,
)


class ConstitutionDomain(str, Enum):
    UNASSIGNED = "unassigned"
    MANDATE_AND_OBJECTIVE = "mandate_and_objective"
    CIRCLE_OF_COMPETENCE = "circle_of_competence"
    BUSINESS_AND_ASSET_QUALITY = "business_and_asset_quality"
    MANAGEMENT_AND_GOVERNANCE = "management_and_governance"
    FINANCIAL_RESILIENCE = "financial_resilience"
    VALUATION_AND_EXPECTED_RETURN = (
        "valuation_and_expected_return"
    )
    RISK_AND_UNCERTAINTY = "risk_and_uncertainty"
    MARKET_AND_CYCLE_CONTEXT = "market_and_cycle_context"
    PORTFOLIO_CONSTRUCTION = "portfolio_construction"
    MONITORING_AND_EXIT = "monitoring_and_exit"


class RelationType(str, Enum):
    PARENT_OF = "parent_of"
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    OVERLAPS_WITH = "overlaps_with"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CAUSES = "causes"
    INCREASES = "increases"
    REDUCES = "reduces"
    PROTECTS_AGAINST = "protects_against"
    REQUIRES = "requires"
    APPLIES_WHEN = "applies_when"
    FAILS_WHEN = "fails_when"


class RelationScope(str, Enum):
    WITHIN_INVESTOR = "within_investor"
    CROSS_INVESTOR = "cross_investor"


SYMMETRIC_RELATIONS: frozenset[RelationType] = frozenset(
    {
        RelationType.RELATED_TO,
        RelationType.SIMILAR_TO,
        RelationType.OVERLAPS_WITH,
        RelationType.CONTRADICTS,
    }
)


class CanonicalModelDraft(BaseModel):
    """Minimal structured output for one canonical mental model."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    kind: FragmentKind
    title: str = Field(min_length=1, max_length=180)
    proposition: str = Field(min_length=10, max_length=1600)

    mechanism: list[str] = Field(
        default_factory=list,
        max_length=4,
    )
    conditions: list[str] = Field(
        default_factory=list,
        max_length=4,
    )
    failure_conditions: list[str] = Field(
        default_factory=list,
        max_length=4,
    )
    decision_implications: list[str] = Field(
        default_factory=list,
        max_length=4,
    )
    decision_stages: list[DecisionStage] = Field(
        default_factory=list,
    )
    contextual_regimes: list[str] = Field(
        default_factory=list,
        max_length=4,
    )

    supporting_fragment_codes: list[str] = Field(
        min_length=1,
    )

    @field_validator(
        "mechanism",
        "conditions",
        "failure_conditions",
        "decision_implications",
        "contextual_regimes",
    )
    @classmethod
    def clean_string_lists(
        cls,
        values: list[str],
    ) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for value in values:
            value = value.strip()

            if not value:
                continue

            key = value.casefold()

            if key not in seen:
                cleaned.append(value)
                seen.add(key)

        return cleaned

    @field_validator("supporting_fragment_codes")
    @classmethod
    def clean_fragment_codes(
        cls,
        values: list[str],
    ) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for value in values:
            value = value.strip()

            if value and value not in seen:
                cleaned.append(value)
                seen.add(value)

        if not cleaned:
            raise ValueError(
                "At least one supporting fragment is required."
            )

        return cleaned


class CanonicalClusterResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_key: str = Field(min_length=1)
    models: list[CanonicalModelDraft] = Field(
        min_length=1,
    )


class CanonicalisationBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clusters: list[CanonicalClusterResult] = Field(
        default_factory=list,
    )


class ConstitutionAssignment(BaseModel):
    """Fixed constitution-domain placement returned by Luna."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    canonical_code: str = Field(min_length=1)
    primary_domain: ConstitutionDomain

    secondary_domains: list[ConstitutionDomain] = Field(
        default_factory=list,
        max_length=3,
    )

    @model_validator(mode="after")
    def remove_primary_and_duplicates(
        self,
    ) -> ConstitutionAssignment:
        cleaned: list[ConstitutionDomain] = []

        for domain in self.secondary_domains:
            if (
                domain != self.primary_domain
                and domain not in cleaned
            ):
                cleaned.append(domain)

        self.secondary_domains = cleaned
        return self


class ConstitutionBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignments: list[ConstitutionAssignment] = Field(
        default_factory=list,
    )


class ConceptFamilyAssignment(BaseModel):
    """One compact name and description for a model cluster."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    cluster_key: str = Field(min_length=1)

    concept_family: str = Field(
        min_length=5,
        max_length=100,
    )

    @field_validator("concept_family")
    @classmethod
    def normalise_concept_family(
        cls,
        value: str,
    ) -> str:
        family, separator, description = value.partition(":")

        if not separator:
            raise ValueError(
                "Use 'family_name: very short description'."
            )

        family = re.sub(
            r"[^a-z0-9]+",
            "_",
            family.casefold(),
        ).strip("_")

        description = " ".join(
            description.split()
        ).strip(" .")

        if not family or not description:
            raise ValueError(
                "Both concept-family parts are required."
            )

        combined = f"{family}: {description}"

        if len(combined) > 100:
            raise ValueError(
                "concept_family must not exceed 100 characters."
            )

        return combined


class ConceptFamilyBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    families: list[ConceptFamilyAssignment] = Field(
        default_factory=list,
    )


class HierarchyAssignment(BaseModel):
    """Final domain and concept-family assignment stored in the DB."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    canonical_code: str = Field(min_length=1)
    primary_domain: ConstitutionDomain

    secondary_domains: list[ConstitutionDomain] = Field(
        default_factory=list,
        max_length=3,
    )

    concept_family: str | None = Field(
        default=None,
        max_length=100,
    )


class RelationshipDraft(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source_canonical_code: str = Field(min_length=1)
    target_canonical_code: str = Field(min_length=1)
    relation_type: RelationType

    relation_strength: float = Field(
        ge=0.0,
        le=1.0,
    )
    relation_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def prevent_self_edge(
        self,
    ) -> RelationshipDraft:
        if (
            self.source_canonical_code
            == self.target_canonical_code
        ):
            raise ValueError(
                "A model cannot connect to itself."
            )

        return self


class PairRelationshipResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_key: str = Field(min_length=1)

    relationships: list[RelationshipDraft] = Field(
        default_factory=list,
        max_length=3,
    )


class HierarchyRelationshipBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[PairRelationshipResult] = Field(
        default_factory=list,
    )


class CanonicalModelData(BaseModel):
    """Application representation before database insertion."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    canonical_code: str = Field(
        pattern=r"^mmc_[a-z0-9]+_[a-f0-9]{8}$"
    )
    investor_id: str = Field(min_length=1)

    kind: FragmentKind
    title: str
    proposition: str

    mechanism: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    failure_conditions: list[str] = Field(
        default_factory=list,
    )
    decision_implications: list[str] = Field(
        default_factory=list,
    )
    decision_stages: list[DecisionStage] = Field(
        default_factory=list,
    )
    contextual_regimes: list[str] = Field(
        default_factory=list,
    )

    supporting_fragment_codes: list[str] = Field(
        min_length=1,
    )

    evidence_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    investor_importance: float = Field(
        ge=0.0,
        le=1.0,
    )
    base_weight: float = Field(
        ge=0.0,
        le=1.0,
    )