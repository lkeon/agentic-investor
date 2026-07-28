"""Structured contracts for the value-investing committee MVP."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


Stance = Literal[
    "positive",
    "negative",
    "mixed",
    "insufficient_information",
]
HoldingPolicy = Literal["indefinite_while_thesis_valid"]
ClaimStatus = Literal[
    "reported_fact",
    "derived_metric",
    "management_claim",
    "analyst_estimate",
    "user_assumption",
    "unknown",
]


class InvestmentQuestion(BaseModel):
    """Normalized value-investing decision with a thesis-led holding policy."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    original_question: str = Field(min_length=3)
    normalised_question: str = Field(min_length=10, max_length=1600)
    subject: str | None = Field(default=None, max_length=200)
    security_or_asset: str | None = Field(default=None, max_length=200)
    decision_type: str = Field(min_length=2, max_length=100)
    as_of_date: date

    holding_policy: HoldingPolicy = "indefinite_while_thesis_valid"

    known_facts: list[str] = Field(default_factory=list, max_length=12)
    user_constraints: list[str] = Field(default_factory=list, max_length=10)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)

class ResearchSource(BaseModel):
    """One source made available to the structured researcher."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=500)
    publisher: str | None = Field(default=None, max_length=200)
    url: str | None = Field(default=None, max_length=2000)
    published_at: date | None = None
    source_type: str = Field(min_length=2, max_length=100)


class EvidenceClaim(BaseModel):
    """A sourced fact, estimate, assumption, or explicit unknown."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_id: str = Field(min_length=1, max_length=120)
    statement: str = Field(min_length=3, max_length=1600)
    value: str | float | int | bool | None = None
    unit: str | None = Field(default=None, max_length=80)
    period: str | None = Field(default=None, max_length=120)
    status: ClaimStatus
    source_ids: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)
    as_of_date: date | None = None


def _disambiguate_duplicate_claim_ids(
    value: object,
    claim_fields: tuple[str, ...],
) -> object:
    """Re-key repeated LLM-generated IDs before strict view validation.

    Every claim is retained. The first occurrence keeps its ID and later
    occurrences receive a deterministic numeric suffix. All original IDs are
    reserved up front so a repaired ID cannot collide with a valid later one.
    """

    if not isinstance(value, dict):
        return value

    def claim_id(claim: object) -> str | None:
        if isinstance(claim, EvidenceClaim):
            return claim.claim_id.strip()
        if isinstance(claim, dict):
            raw_id = claim.get("claim_id")
            if isinstance(raw_id, str):
                return raw_id.strip()
        return None

    reserved_ids: set[str] = set()
    for field_name in claim_fields:
        claims = value.get(field_name)
        if not isinstance(claims, list):
            continue
        reserved_ids.update(
            identifier
            for claim in claims
            if (identifier := claim_id(claim))
        )

    repaired_value = dict(value)
    seen_ids: set[str] = set()
    for field_name in claim_fields:
        claims = value.get(field_name)
        if not isinstance(claims, list):
            continue

        repaired_claims: list[object] = []
        for claim in claims:
            identifier = claim_id(claim)
            if not identifier or identifier not in seen_ids:
                if identifier:
                    seen_ids.add(identifier)
                repaired_claims.append(claim)
                continue

            suffix_number = 2
            while True:
                suffix = f"_{suffix_number}"
                candidate = f"{identifier[: 120 - len(suffix)]}{suffix}"
                if candidate not in reserved_ids:
                    break
                suffix_number += 1

            reserved_ids.add(candidate)
            seen_ids.add(candidate)
            if isinstance(claim, EvidenceClaim):
                repaired_claims.append(
                    claim.model_copy(update={"claim_id": candidate})
                )
            else:
                repaired_claim = dict(claim)
                repaired_claim["claim_id"] = candidate
                repaired_claims.append(repaired_claim)

        repaired_value[field_name] = repaired_claims

    return repaired_value


def _validate_evidence(
    *,
    claims: list[EvidenceClaim],
    sources: list[ResearchSource],
) -> None:
    claim_ids = [claim.claim_id for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Evidence claim IDs must be unique within a view.")

    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Research source IDs must be unique within a view.")

    known_sources = set(source_ids)
    for claim in claims:
        unknown = set(claim.source_ids) - known_sources
        if unknown:
            raise ValueError(
                f"Claim {claim.claim_id} references unknown sources: "
                f"{sorted(unknown)}"
            )
        if (
            claim.status
            not in {"user_assumption", "unknown"}
            and not claim.source_ids
        ):
            raise ValueError(
                f"Claim {claim.claim_id} requires at least one source."
            )


class MicroView(BaseModel):
    """Company and security evidence without an investment recommendation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    company_name: str = Field(min_length=1, max_length=300)
    ticker: str | None = Field(default=None, max_length=40)
    security_type: str | None = Field(default=None, max_length=100)
    as_of_date: date

    business: list[EvidenceClaim] = Field(default_factory=list, max_length=12)
    business_quality: list[EvidenceClaim] = Field(
        default_factory=list,
        max_length=12,
    )
    management_and_capital_allocation: list[EvidenceClaim] = Field(
        default_factory=list,
        max_length=12,
    )
    financial_performance: list[EvidenceClaim] = Field(
        default_factory=list,
        max_length=14,
    )
    balance_sheet_and_liquidity: list[EvidenceClaim] = Field(
        default_factory=list,
        max_length=14,
    )
    valuation: list[EvidenceClaim] = Field(default_factory=list, max_length=12)
    risks_and_catalysts: list[EvidenceClaim] = Field(
        default_factory=list,
        max_length=14,
    )

    unresolved_questions: list[str] = Field(default_factory=list, max_length=15)
    sources: list[ResearchSource] = Field(default_factory=list, max_length=30)

    @model_validator(mode="before")
    @classmethod
    def disambiguate_claim_ids(cls, value: object) -> object:
        return _disambiguate_duplicate_claim_ids(
            value,
            (
                "business",
                "business_quality",
                "management_and_capital_allocation",
                "financial_performance",
                "balance_sheet_and_liquidity",
                "valuation",
                "risks_and_catalysts",
            ),
        )

    def all_claims(self) -> list[EvidenceClaim]:
        return [
            *self.business,
            *self.business_quality,
            *self.management_and_capital_allocation,
            *self.financial_performance,
            *self.balance_sheet_and_liquidity,
            *self.valuation,
            *self.risks_and_catalysts,
        ]

    @model_validator(mode="after")
    def validate_claims(self) -> MicroView:
        _validate_evidence(claims=self.all_claims(), sources=self.sources)
        return self


class MacroView(BaseModel):
    """Only macro evidence with a plausible transmission to the company."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    as_of_date: date
    environment: list[EvidenceClaim] = Field(
        default_factory=list,
        max_length=12,
    )
    company_transmission_channels: list[EvidenceClaim] = Field(
        default_factory=list,
        max_length=14,
    )
    regime_risks: list[EvidenceClaim] = Field(
        default_factory=list,
        max_length=12,
    )
    unresolved_questions: list[str] = Field(default_factory=list, max_length=12)
    sources: list[ResearchSource] = Field(default_factory=list, max_length=25)

    @model_validator(mode="before")
    @classmethod
    def disambiguate_claim_ids(cls, value: object) -> object:
        return _disambiguate_duplicate_claim_ids(
            value,
            (
                "environment",
                "company_transmission_channels",
                "regime_risks",
            ),
        )

    def all_claims(self) -> list[EvidenceClaim]:
        return [
            *self.environment,
            *self.company_transmission_channels,
            *self.regime_risks,
        ]

    @model_validator(mode="after")
    def validate_claims(self) -> MacroView:
        _validate_evidence(claims=self.all_claims(), sources=self.sources)
        return self


class MentalModelCandidate(BaseModel):
    """One investor model retrieved for an analytical bridge."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    canonical_code: str
    investor_id: str
    title: str
    proposition: str
    mechanism: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    failure_conditions: list[str] = Field(default_factory=list)
    decision_implications: list[str] = Field(default_factory=list)

    matched_bridge_ids: list[str] = Field(min_length=1)
    relevant_claim_ids: list[str] = Field(default_factory=list)
    unresolved_conditions: list[str] = Field(default_factory=list)

    retrieval_score: float = Field(ge=-1.0, le=1.0)
    retrieval_origin: Literal["direct", "neighbour"]
    relation_to_source: str | None = None


class MentalModelBridge(BaseModel):
    """One evidence-backed analytical search and its retrieved models."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    bridge_id: str = Field(min_length=1, max_length=120)
    investor_id: str | None = Field(default=None, max_length=100)
    normalised_question: str = Field(min_length=10, max_length=1600)

    holding_policy: HoldingPolicy = "indefinite_while_thesis_valid"

    analytical_question: str = Field(min_length=10, max_length=1000)
    search_query: str = Field(min_length=10, max_length=1400)
    decision_stage: str = Field(min_length=2, max_length=100)
    domains: list[str] = Field(min_length=1, max_length=4)
    importance: float = Field(ge=0.0, le=1.0)

    retrieved_data: list[EvidenceClaim] = Field(min_length=1, max_length=12)
    missing_information: list[str] = Field(default_factory=list, max_length=8)
    mental_model_candidates: list[MentalModelCandidate] = Field(
        default_factory=list,
        max_length=15,
    )

    @model_validator(mode="after")
    def validate_bridge(self) -> MentalModelBridge:
        claim_ids = [claim.claim_id for claim in self.retrieved_data]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Bridge evidence claim IDs must be unique.")

        codes = [
            candidate.canonical_code
            for candidate in self.mental_model_candidates
        ]
        if len(codes) != len(set(codes)):
            raise ValueError(
                "Mental-model candidates must be unique within a bridge."
            )

        for candidate in self.mental_model_candidates:
            if self.investor_id != candidate.investor_id:
                raise ValueError(
                    "Bridge and candidate investor identities must match."
                )
            if self.bridge_id not in candidate.matched_bridge_ids:
                raise ValueError(
                    "A candidate must include its containing bridge ID."
                )
        return self


class MentalModelBridgeList(BaseModel):
    """Structured bridge collection compatible with OpenAI object schemas."""

    model_config = ConfigDict(extra="forbid")

    bridges: list[MentalModelBridge] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_bridges(self) -> MentalModelBridgeList:
        bridge_ids = [bridge.bridge_id for bridge in self.bridges]
        if len(bridge_ids) != len(set(bridge_ids)):
            raise ValueError("Mental-model bridge IDs must be unique.")
        return self


class OneInvestorReasoningInput(BaseModel):
    """Complete evidence and mental-model context for one investor."""

    model_config = ConfigDict(extra="forbid")

    investor_id: str
    micro_view: MicroView
    macro_view: MacroView
    mental_model_bridges: list[MentalModelBridge] = Field(
        min_length=1,
        max_length=8,
    )

    @model_validator(mode="after")
    def validate_bridges(self) -> OneInvestorReasoningInput:
        questions = {
            bridge.normalised_question
            for bridge in self.mental_model_bridges
        }
        policies = {
            bridge.holding_policy
            for bridge in self.mental_model_bridges
        }
        if len(questions) != 1 or len(policies) != 1:
            raise ValueError(
                "Every investor bridge must use the same question and holding policy."
            )
        for bridge in self.mental_model_bridges:
            if bridge.investor_id != self.investor_id:
                raise ValueError(
                    "Every bridge must belong to the reasoning investor."
                )
            if any(
                candidate.investor_id != self.investor_id
                for candidate in bridge.mental_model_candidates
            ):
                raise ValueError(
                    "Every mental-model candidate must belong to the investor."
                )
        return self


class MentalModelInference(BaseModel):
    """One auditable conclusion connecting evidence and mental models."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    conclusion: str = Field(min_length=3, max_length=1400)
    reasoning: str = Field(min_length=3, max_length=1800)
    mental_model_codes: list[str] = Field(default_factory=list, max_length=6)
    evidence_claim_ids: list[str] = Field(default_factory=list, max_length=10)
    counterevidence_claim_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
    )
    applicability: Literal[
        "applies",
        "partially_applies",
        "does_not_apply",
        "uncertain",
    ]
    missing_information: list[str] = Field(default_factory=list, max_length=6)


class InvestorReasoningOutput(BaseModel):
    """Independent reasoning from one investor perspective."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    investor_id: str
    stance: Stance
    thesis: str = Field(
        min_length=20,
        max_length=2600,
        description=(
            "Self-contained investment view of at least 80 words, targeting "
            "80–160 words, summarising the investor's stance, business "
            "quality, valuation, decisive risk, and conditions for continued "
            "ownership."
        ),
    )
    mental_model_inferences: list[MentalModelInference] = Field(
        min_length=3,
        max_length=6,
    )

    thesis_durability_assessment: str = Field(min_length=3, max_length=1200)
    suitable_while_thesis_valid: bool
    temporary_concerns: list[str] = Field(default_factory=list, max_length=6)
    permanent_loss_risks: list[str] = Field(default_factory=list, max_length=8)
    key_risks: list[str] = Field(default_factory=list, max_length=8)
    missing_information: list[str] = Field(default_factory=list, max_length=10)
    thesis_break_conditions: list[str] = Field(
        default_factory=list,
        max_length=8,
    )

    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("thesis")
    @classmethod
    def validate_investment_view_length(cls, value: str) -> str:
        """Keep the headline investor view useful when read in isolation."""

        if len(value.split()) < 80:
            raise ValueError(
                "Investment view must contain at least 80 words."
            )
        return value


class CIOReasoningInput(BaseModel):
    """Evidence and committee views supplied to the CIO."""

    model_config = ConfigDict(extra="forbid")

    question: InvestmentQuestion
    micro_view: MicroView
    macro_view: MacroView
    investor_outputs: dict[str, InvestorReasoningOutput]

    @model_validator(mode="after")
    def validate_investor_outputs(self) -> CIOReasoningInput:
        for investor_id, output in self.investor_outputs.items():
            if output.investor_id != investor_id:
                raise ValueError("CIO investor output map is inconsistent.")
        return self


class CIOReasoningOutput(BaseModel):
    """Compact, auditable CIO decision for the MVP."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision: Literal[
        "buy",
        "buy_below_price",
        "watchlist",
        "hold",
        "sell",
        "avoid",
        "insufficient_information",
    ]
    final_answer: str = Field(min_length=20, max_length=3000)
    committee_synthesis: str = Field(min_length=20, max_length=2400)

    holding_policy: HoldingPolicy

    decisive_evidence_claim_ids: list[str] = Field(
        default_factory=list,
        max_length=15,
    )
    decisive_mental_model_codes: list[str] = Field(
        default_factory=list,
        max_length=15,
    )
    key_risks: list[str] = Field(default_factory=list, max_length=8)
    decision_conditions: list[str] = Field(default_factory=list, max_length=10)
    missing_information: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)


class InvestmentCommitteeOutput(BaseModel):
    """Complete JSON artifact from research through final CIO decision."""

    model_config = ConfigDict(extra="forbid")

    question: InvestmentQuestion
    micro_view: MicroView
    macro_view: MacroView
    investor_reasoning_inputs: dict[str, OneInvestorReasoningInput]
    investor_outputs: dict[str, InvestorReasoningOutput]
    cio: CIOReasoningOutput
    model_configuration: dict[str, str]
    embedding_identity: str
