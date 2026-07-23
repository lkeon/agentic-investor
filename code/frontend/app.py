"""Minimal Chainlit frontend for the Investment Committee MVP."""

from __future__ import annotations

import chainlit as cl

from committee_service import CommitteeResult, run_committee


@cl.on_chat_start
async def on_chat_start() -> None:
    """Show a short product-style introduction when a session begins."""
    await cl.Message(
        content=(
            "# Investment Committee\n\n"
            "Submit one investment question. The committee will retrieve relevant "
            "mental models, form independent views, debate the case, and produce a "
            "final assessment.\n\n"
            "Try: `Should we invest in a highly leveraged infrastructure company "
            "with inflation-linked revenues?`"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Run one committee analysis for each user message."""
    question = message.content.strip()
    if not question:
        await cl.Message(content="Please enter an investment question.").send()
        return

    try:
        result = await run_committee(question)
        await render_committee_result(question, result)
    except Exception as exc:
        await cl.Message(
            content=(
                "The committee could not complete this analysis.\n\n"
                f"Technical detail: `{type(exc).__name__}: {exc}`"
            )
        ).send()


async def render_committee_result(
    question: str,
    result: CommitteeResult,
) -> None:
    """Render the structured committee result as mobile-friendly stacked steps."""
    async with cl.Step(name="Mental-model retrieval", type="retrieval") as step:
        step.input = question
        step.output = "\n".join(f"- {model}" for model in result.mental_models)

    for perspective in result.perspectives:
        async with cl.Step(name=perspective.investor, type="llm") as step:
            step.input = "Independent assessment"
            step.output = perspective.assessment

    async with cl.Step(name="Committee debate", type="llm") as step:
        step.input = "Challenge the initial assessments"
        step.output = "\n\n".join(
            f"**{item.speaker}:** {item.argument}" for item in result.debate
        )

    async with cl.Step(name="Committee synthesis", type="llm") as step:
        step.input = "Synthesize the committee view"
        step.output = result.conclusion

    sources = "\n".join(f"- {source}" for source in result.sources)
    final_content = (
        "# Committee conclusion\n\n"
        f"{result.conclusion}\n\n"
        "## Supporting references\n\n"
        f"{sources if sources else '- No references returned.'}"
    )
    await cl.Message(content=final_content).send()
