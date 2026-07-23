"""Pydantic schemas for the minimal investor-committee workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Stance = Literal[
    "positive",
    "negative",
    "mixed",
    "insufficient_information",
]


class InvestmentQuestion(BaseModel):
    """A compact, retrieval-oriented interpretation of the user's question."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    original_question: str = Field(min_length=3)
    retrieval_query: str = Field(min_length=10, max_length=1200)
    subject: str | None = Field(default=None, max_length=200)
    decision_type: str = Field(min_length=2, max_length=100)
    decision_stages: list[str] = Field(default_factory=list, max_length=8)
    known_facts: list[str] = Field(default_factory=list, max_length=10)
    uncertainties: list[str] = Field(default_factory=list, max_length=10)

    def embedding_text(self) -> str:
        parts = [self.retrieval_query]
        if self.decision_stages:
            parts.append("Decision stages: " + ", ".join(self.decision_stages))
        if self.uncertainties:
            parts.append("Uncertainties: " + " | ".join(self.uncertainties))
        return "\n".join(parts)


class RetrievedMentalModel(BaseModel):
    """One canonical model selected for an investor's guardrail packet."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    canonical_code: str
    investor_id: str
    title: str
    proposition: str
    mechanism: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    failure_conditions: list[str] = Field(default_factory=list)
    decision_implications: list[str] = Field(default_factory=list)
    primary_domain: str
    concept_family: str | None = None
    retrieval_score: float = Field(ge=-1.0, le=1.0)
    retrieval_origin: Literal["direct", "neighbour"]
    relation_to_source: str | None = None


class InvestorGuardrailPacket(BaseModel):
    """Question-specific context supplied to one investor agent."""

    model_config = ConfigDict(extra="forbid")

    investor_id: str
    question: InvestmentQuestion
    mental_models: list[RetrievedMentalModel] = Field(min_length=1)


class InvestorView(BaseModel):
    """Independent first-round analysis from one investor perspective."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    investor_id: str
    stance: Stance
    thesis: str = Field(min_length=20, max_length=2400)
    mental_models_used: list[str] = Field(default_factory=list)
    analysis_points: list[str] = Field(default_factory=list, max_length=8)
    key_risks: list[str] = Field(default_factory=list, max_length=6)
    disconfirming_evidence: list[str] = Field(default_factory=list, max_length=6)
    missing_information: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)


class PeerComment(BaseModel):
    """One investor's response to another investor's first-round view."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    peer_investor_id: str
    agreements: list[str] = Field(default_factory=list, max_length=4)
    disagreements: list[str] = Field(default_factory=list, max_length=4)
    missing_considerations: list[str] = Field(default_factory=list, max_length=4)


class InvestorRoundTwoView(BaseModel):
    """Peer review and revised view from one investor perspective."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    investor_id: str
    peer_comments: list[PeerComment] = Field(default_factory=list)
    updated_thesis: str = Field(min_length=20, max_length=2400)
    changed_view: bool
    reason_for_change: str | None = Field(default=None, max_length=1000)
    mental_models_used: list[str] = Field(default_factory=list)
    final_stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)


class InvestmentCommitteeResult(BaseModel):
    """MVP output without a CIO synthesis."""

    model_config = ConfigDict(extra="forbid")

    question: InvestmentQuestion
    retrieved_models: dict[str, list[RetrievedMentalModel]]
    round_one: dict[str, InvestorView]
    round_two: dict[str, InvestorRoundTwoView]
