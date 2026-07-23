"""UI-facing committee service.

The mock implementation lets the frontend run without CrewAI, a database,
or an LLM key. Replace `run_committee` with an adapter to your existing crew
when you are ready.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class InvestorPerspective:
    investor: str
    assessment: str


@dataclass(frozen=True)
class DebatePoint:
    speaker: str
    argument: str


@dataclass(frozen=True)
class CommitteeResult:
    mental_models: list[str]
    perspectives: list[InvestorPerspective]
    debate: list[DebatePoint]
    conclusion: str
    sources: list[str]


async def run_committee(question: str) -> CommitteeResult:
    """Return a deterministic mock result for local UI testing.

    CrewAI integration can later replace this function while preserving the
    `CommitteeResult` contract expected by `app.py`.
    """
    await asyncio.sleep(0.35)

    mental_models = [
        "Margin of safety",
        "Capital-cycle discipline",
        "Downside-first credit analysis",
        "Inflation-linked cash-flow durability",
        "Management capital-allocation record",
    ]

    perspectives = [
        InvestorPerspective(
            investor="Warren Buffett",
            assessment=(
                "The central question is whether the business has durable pricing "
                "power and whether leverage could force permanent capital loss. "
                "A predictable revenue contract is valuable only if maintenance "
                "capital expenditure and refinancing needs remain manageable."
            ),
        ),
        InvestorPerspective(
            investor="Howard Marks",
            assessment=(
                "The investment may be attractive if the market already discounts "
                "a difficult refinancing environment. The committee should compare "
                "the downside embedded in the price with the downside that is "
                "actually plausible under higher-for-longer rates."
            ),
        ),
        InvestorPerspective(
            investor="Bruce Flatt",
            assessment=(
                "Long-duration infrastructure can protect real cash flows, but the "
                "quality of the contractual framework matters more than the asset "
                "label. Focus on indexed revenues, operating resilience, funding "
                "structure, and opportunities to improve the asset operationally."
            ),
        ),
    ]

    debate = [
        DebatePoint(
            speaker="Marks",
            argument=(
                "Buffett's quality threshold may reject a security whose price "
                "already compensates investors for refinancing risk."
            ),
        ),
        DebatePoint(
            speaker="Buffett",
            argument=(
                "A low valuation is not protection when creditors control the "
                "outcome and equity lacks the ability to wait."
            ),
        ),
        DebatePoint(
            speaker="Flatt",
            argument=(
                "Both views depend on the debt ladder. The asset can be excellent "
                "while the security is poor, so financing must be analysed separately."
            ),
        ),
    ]

    conclusion = (
        f"For the question, **{question}**, the committee reaches a **conditional "
        "watch-list** conclusion. The business may have attractive real-asset "
        "characteristics, but investment should depend on a full debt-maturity "
        "schedule, interest-coverage stress test, maintenance-capex requirements, "
        "and evidence that revenue indexation survives adverse conditions. A price "
        "that provides a genuine margin of safety is essential."
    )

    sources = [
        "Mock source: Berkshire Hathaway shareholder-letter mental models",
        "Mock source: Oaktree memos on risk and market cycles",
        "Mock source: Brookfield shareholder letters on real assets",
    ]

    return CommitteeResult(
        mental_models=mental_models,
        perspectives=perspectives,
        debate=debate,
        conclusion=conclusion,
        sources=sources,
    )
