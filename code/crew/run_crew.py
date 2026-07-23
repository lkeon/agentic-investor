"""CLI entry point for the minimal two-round investor crew."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mental_model.canonical.providers import OpenAIStructuredProvider
from crew.agents import run_round_one, run_round_two
from crew.retrieval import retrieve_for_investors
from crew.schemas import (
    InvestmentCommitteeResult,
    InvestmentQuestion,
    InvestorGuardrailPacket,
)


QUESTION_SYSTEM_PROMPT = """
Rewrite the user's input into a compact investment-analysis retrieval query.
Do not add external facts. Separate facts explicitly supplied by the user from
uncertainties. Select only relevant decision stages. The retrieval_query should
be broad enough to retrieve business-quality, management, financial, valuation,
risk, market-cycle, portfolio, or monitoring mental models when relevant.
Return no commentary.
""".strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve investor-specific canonical models and run two CrewAI "
            "analysis rounds without a CIO synthesis."
        )
    )
    parser.add_argument("question", help="Investment question in quotes.")
    parser.add_argument(
        "--investor",
        action="append",
        help="Investor ID to include. Repeat for multiple investors.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--neighbours", type=int, default=3)
    parser.add_argument(
        "--normaliser-model",
        default=os.getenv("QUESTION_NORMALISER_MODEL", "gpt-5.6-luna"),
    )
    parser.add_argument(
        "--analysis-model",
        default=os.getenv("INVESTOR_ANALYSIS_MODEL", "gpt-5.6-luna"),
    )
    parser.add_argument(
        "--peer-review-model",
        default=os.getenv("PEER_REVIEW_MODEL", "gpt-5.6-luna"),
    )
    parser.add_argument("--skip-round-two", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Normalise and retrieve, but do not run CrewAI investor tasks.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/crew/committee_result.json"),
    )
    return parser.parse_args()


def _normalise_question(raw_question: str, model: str) -> InvestmentQuestion:
    provider = OpenAIStructuredProvider(
        model=model,
        reasoning_effort="low",
    )
    return provider.parse(
        schema=InvestmentQuestion,
        system_prompt=QUESTION_SYSTEM_PROMPT,
        user_prompt=json.dumps(
            {"original_question": raw_question},
            ensure_ascii=False,
        ),
        max_output_tokens=1600,
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_arguments()

    print(
        f"Normalising question with {args.normaliser_model}...",
        flush=True,
    )
    question = _normalise_question(args.question, args.normaliser_model)

    print("Embedding retrieval query and retrieving MMCs...", flush=True)
    retrieved = retrieve_for_investors(
        question,
        investor_filter=set(args.investor) if args.investor else None,
        top_k=args.top_k,
        neighbour_limit=args.neighbours,
    )

    if not retrieved:
        raise RuntimeError("No investors with embedded canonical models found.")

    for investor_id, models in retrieved.items():
        direct = sum(model.retrieval_origin == "direct" for model in models)
        neighbours = len(models) - direct
        print(
            f"{investor_id}: retrieved={len(models)} "
            f"(direct={direct}, neighbours={neighbours})",
            flush=True,
        )

    if args.dry_run:
        _write_json(
            args.output_path,
            {
                "question": question.model_dump(mode="json"),
                "retrieved_models": {
                    investor_id: [
                        model.model_dump(mode="json") for model in models
                    ]
                    for investor_id, models in retrieved.items()
                },
                "round_one": {},
                "round_two": {},
            },
        )
        print(
            f"Dry run complete; retrieval written to {args.output_path}.",
            flush=True,
        )
        return 0

    packets = {
        investor_id: InvestorGuardrailPacket(
            investor_id=investor_id,
            question=question,
            mental_models=models,
        )
        for investor_id, models in retrieved.items()
    }

    round_one = {}
    total = len(packets)
    for number, (investor_id, packet) in enumerate(packets.items(), start=1):
        print(
            f"Round 1 {number}/{total}: {investor_id}...",
            flush=True,
        )
        round_one[investor_id] = run_round_one(
            packet,
            model=args.analysis_model,
            verbose=args.verbose,
        )

    round_two = {}
    if not args.skip_round_two and len(round_one) > 1:
        for number, (investor_id, packet) in enumerate(packets.items(), start=1):
            print(
                f"Round 2 {number}/{total}: {investor_id}...",
                flush=True,
            )
            peers = {
                peer_id: view
                for peer_id, view in round_one.items()
                if peer_id != investor_id
            }
            round_two[investor_id] = run_round_two(
                packet,
                round_one[investor_id],
                peers,
                model=args.peer_review_model,
                verbose=args.verbose,
            )

    result = InvestmentCommitteeResult(
        question=question,
        retrieved_models=retrieved,
        round_one=round_one,
        round_two=round_two,
    )
    _write_json(args.output_path, result.model_dump(mode="json"))
    print(f"Crew result written to {args.output_path}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
