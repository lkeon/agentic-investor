"""Create minimal CrewAI agents and run the two committee rounds."""

from __future__ import annotations

import json

from crewai import Agent, Crew, Process, Task

from crew.schemas import (
    InvestorGuardrailPacket,
    InvestorRoundTwoView,
    InvestorView,
)


def _display_name(investor_id: str) -> str:
    return investor_id.replace("_", " ").replace("-", " ").title()


def _agent(investor_id: str, *, model: str, verbose: bool) -> Agent:
    name = _display_name(investor_id)
    return Agent(
        role=f"{name} mental-model investment analyst",
        goal=(
            f"Evaluate the investment question strictly through the "
            f"retrieved {name}-derived canonical mental models."
        ),
        backstory=(
            "You are not the historical investor and must not claim that the "
            "investor analysed this exact situation. You apply canonical "
            "models extracted from the investor's published material. Treat "
            "them as analytical guardrails, check their conditions, identify "
            "when they fail, and cite canonical codes for material claims."
        ),
        llm=f"openai/{model}",
        allow_delegation=False,
        reasoning=False,
        memory=False,
        max_iter=1,
        max_retry_limit=2,
        verbose=verbose,
    )


def _guardrail_json(packet: InvestorGuardrailPacket) -> str:
    return json.dumps(
        packet.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def run_round_one(
    packet: InvestorGuardrailPacket,
    *,
    model: str,
    verbose: bool = False,
) -> InvestorView:
    """Run one isolated first-round CrewAI task."""

    agent = _agent(packet.investor_id, model=model, verbose=verbose)
    task = Task(
        description=(
            "Produce an independent investment analysis using only the "
            "question and mental-model packet below. Do not infer missing "
            "company facts. Apply a model only when its conditions appear "
            "satisfied. Explicitly identify missing information and evidence "
            "that could disconfirm the thesis. Every entry in "
            "mental_models_used must be a supplied canonical_code.\n\n"
            f"PACKET:\n{_guardrail_json(packet)}"
        ),
        expected_output=(
            "A concise structured investor view with stance, thesis, applied "
            "canonical codes, analysis, risks, disconfirming evidence, missing "
            "information, and confidence."
        ),
        agent=agent,
        output_pydantic=InvestorView,
    )
    result = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        memory=False,
        verbose=verbose,
    ).kickoff()

    if result.pydantic is None:
        raise RuntimeError(
            f"Round 1 returned no Pydantic output for {packet.investor_id}."
        )
    return InvestorView.model_validate(result.pydantic)


def run_round_two(
    packet: InvestorGuardrailPacket,
    own_view: InvestorView,
    peer_views: dict[str, InvestorView],
    *,
    model: str,
    verbose: bool = False,
) -> InvestorRoundTwoView:
    """Review every peer view while retaining the investor's own guardrails."""

    agent = _agent(packet.investor_id, model=model, verbose=verbose)
    context = {
        "packet": packet.model_dump(mode="json"),
        "own_round_one_view": own_view.model_dump(mode="json"),
        "peer_round_one_views": {
            investor_id: view.model_dump(mode="json")
            for investor_id, view in peer_views.items()
        },
    }
    task = Task(
        description=(
            "Review every peer's first-round view from your own investor "
            "perspective. Comment on each peer separately. Continue using "
            "only your supplied mental models as guardrails; peer arguments "
            "are claims to assess, not new guardrails. State whether your "
            "view changed and why. Every mental_models_used entry must be a "
            "supplied canonical_code.\n\n"
            "CONTEXT:\n"
            + json.dumps(
                context,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
        expected_output=(
            "Structured peer comments plus an updated thesis, final stance, "
            "change explanation, applied canonical codes, and confidence."
        ),
        agent=agent,
        output_pydantic=InvestorRoundTwoView,
    )
    result = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        memory=False,
        verbose=verbose,
    ).kickoff()

    if result.pydantic is None:
        raise RuntimeError(
            f"Round 2 returned no Pydantic output for {packet.investor_id}."
        )
    return InvestorRoundTwoView.model_validate(result.pydantic)
