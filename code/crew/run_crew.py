"""CLI entry point for the value-investing committee MVP."""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from crew.agents import (
    build_mental_model_bridges,
    normalise_question,
    research_macro_view,
    research_micro_view,
    run_cio_synthesis,
    run_investor_reasoning,
)
from crew.config import reasoning_model
from crew.research.macro import get_or_build_daily_macro
from crew.research.micro import collect_micro_research
from crew.retrieval import retrieve_for_bridges
from crew.schemas import (
    CIOReasoningInput,
    ExternalResearchSettings,
    InvestmentCommitteeOutput,
    OneInvestorReasoningInput,
)
from mental_model_pipeline.canonical.embeddings import (
    create_embedding_provider,
)
from mental_model_pipeline.database.connection import engine


class _TerminalSpinner:
    """Small dependency-free progress indicator for interactive terminals."""

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = Event()
        self._enabled = sys.stderr.isatty() and os.environ.get("TERM") != "dumb"
        self._thread: Thread | None = None

    def __enter__(self) -> _TerminalSpinner:
        if not self._enabled:
            print(f"{self.message}...", flush=True)
            return self
        self._thread = Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join()
        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()

    def _animate(self) -> None:
        frames = ("|", "/", "-", "\\")
        index = 0
        while not self._stop.is_set():
            sys.stderr.write(f"\r{frames[index]} {self.message}")
            sys.stderr.flush()
            index = (index + 1) % len(frames)
            self._stop.wait(0.12)


def _database_is_ready() -> bool:
    """Return whether the database configured by DATABASE_URL accepts SQL."""

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return False
    return True


def _ensure_database_running() -> None:
    """Start the local PostgreSQL service only when its configured DB is down."""

    if _database_is_ready():
        return

    print("PostgreSQL is unavailable; starting postgresql service...", flush=True)
    # The MVP uses the host's systemd-managed PostgreSQL service. Capture its
    # output so an authentication or service-name failure is actionable.
    result = subprocess.run(
        ["systemctl", "start", "postgresql"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "The configured database is unavailable and `systemctl start "
            f"postgresql` failed: {detail or 'no diagnostic output'}"
        )

    # A successful service command does not mean PostgreSQL is already ready
    # to accept connections, so poll the configured DATABASE_URL for 10s.
    for _ in range(20):
        if _database_is_ready():
            print("PostgreSQL is ready.", flush=True)
            return
        time.sleep(0.5)
    raise RuntimeError(
        "PostgreSQL was started but the configured DATABASE_URL was not ready "
        "within 10 seconds. Check the service logs and DATABASE_URL."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research an investment question, retrieve investor-specific "
            "mental models, collect independent investor views, and "
            "synthesize a CIO decision."
        )
    )
    parser.add_argument("question", help="Investment question in quotes.")
    parser.add_argument(
        "--research-context",
        type=Path,
        help=(
            "Optional UTF-8 text or JSON evidence supplied to the researcher. "
            "Without it, unsupported current facts remain explicit unknowns."
        ),
    )
    parser.add_argument(
        "--investor",
        action="append",
        help="Investor ID to include. Repeat for multiple investors.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Direct mental models retrieved per bridge and investor.",
    )
    parser.add_argument(
        "--neighbours",
        type=int,
        default=1,
        help="Directional graph neighbours added per bridge and investor.",
    )
    parser.add_argument(
        "--external-research",
        action="store_true",
        help=(
            "Enable the bounded SEC, IR, market, rates, credit, and aggregate "
            "valuation research layer. Disabled by default."
        ),
    )
    parser.add_argument(
        "--sec-filings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SEC filings and XBRL when external research is enabled.",
    )
    parser.add_argument(
        "--investor-relations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bounded official IR document discovery.",
    )
    parser.add_argument(
        "--market-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable one Alpaca IEX market-price snapshot.",
    )
    parser.add_argument(
        "--rates-and-credit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the shared US rates and broad credit indicators.",
    )
    parser.add_argument(
        "--aggregate-valuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the Buffett proxy and Shiller CAPE.",
    )
    parser.add_argument(
        "--credit-rating",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimentally extract a rating only from bounded official IR "
            "documents; never infer a rating."
        ),
    )
    parser.add_argument(
        "--max-filings",
        type=int,
        choices=range(0, 3),
        default=2,
        help="Maximum SEC filing documents fetched (0-2).",
    )
    parser.add_argument(
        "--max-ir-documents",
        type=int,
        choices=range(0, 5),
        default=1,
        help="Maximum official IR source documents accepted from Exa (0-4).",
    )

    parser.add_argument(
        "--question-model",
        "--normaliser-model",
        dest="question_model",
    )
    parser.add_argument("--research-model")
    parser.add_argument("--bridge-model")
    parser.add_argument(
        "--investor-model",
        "--analysis-model",
        dest="investor_model",
    )
    parser.add_argument("--cio-model")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run normalization, structured research, bridge construction, and "
            "retrieval, but skip investor reasoning and CIO synthesis."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/crew/committee_result.json"),
    )
    args = parser.parse_args()
    return args


def _read_research_context(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Research context not found: {path}")
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _model_configuration(args: argparse.Namespace) -> dict[str, str]:
    return {
        "question": reasoning_model(
            "question",
            override=args.question_model,
        ),
        "research": reasoning_model(
            "research",
            override=args.research_model,
        ),
        "bridge": reasoning_model(
            "bridge",
            override=args.bridge_model,
        ),
        "investor": reasoning_model(
            "investor",
            override=args.investor_model,
        ),
        "cio": reasoning_model(
            "cio",
            override=args.cio_model,
        ),
    }


def _external_research_settings(
    args: argparse.Namespace,
) -> ExternalResearchSettings:
    return ExternalResearchSettings(
        enabled=args.external_research,
        sec_filings_enabled=args.sec_filings,
        investor_relations_enabled=args.investor_relations,
        market_data_enabled=args.market_data,
        rates_and_credit_enabled=args.rates_and_credit,
        aggregate_valuation_enabled=args.aggregate_valuation,
        credit_rating_enabled=args.credit_rating,
        max_filings=args.max_filings,
        max_ir_documents=args.max_ir_documents,
    )


def main() -> int:
    args = parse_arguments()
    models = _model_configuration(args)
    external_settings = _external_research_settings(args)
    research_context = _read_research_context(args.research_context)
    print(
        f"Normalising question with {models['question']}...",
        flush=True,
    )
    question = normalise_question(
        args.question,
        model=models["question"],
        verbose=args.verbose,
    )

    micro_research_data = None
    if external_settings.enabled:
        print("Collecting bounded MicroResearchData...", flush=True)
        micro_research_data = collect_micro_research(
            question,
            external_settings,
            progress=lambda message: print(message, flush=True),
        )

    print(f"Building MicroView with {models['research']}...", flush=True)
    micro_view = research_micro_view(
        question,
        research_context=research_context,
        external_research=micro_research_data,
        model=models["research"],
        verbose=args.verbose,
    )

    macro_research_data = None
    if external_settings.enabled:
        print("Loading shared daily MacroView...", flush=True)
        macro_research_data, macro_view = get_or_build_daily_macro(
            external_settings,
            applicable_date=date.today(),
            progress=lambda message: print(message, flush=True),
        )
    else:
        print(
            f"Building MacroView with {models['research']}...",
            flush=True,
        )
        macro_view = research_macro_view(
            question,
            micro_view,
            research_context=research_context,
            model=models["research"],
            verbose=args.verbose,
        )

    print(
        f"Building mental-model bridges with {models['bridge']}...",
        flush=True,
    )
    base_bridges = build_mental_model_bridges(
        question,
        micro_view,
        macro_view,
        model=models["bridge"],
        verbose=args.verbose,
    )

    # Retrieval reads canonical mental models directly from PostgreSQL.
    _ensure_database_running()
    embedding_provider = create_embedding_provider()
    embedding_identity = embedding_provider.identity
    print(
        f"Retrieving mental models with {embedding_identity}...",
        flush=True,
    )
    retrieved = retrieve_for_bridges(
        base_bridges,
        investor_filter=set(args.investor) if args.investor else None,
        top_k=args.top_k,
        neighbour_limit=args.neighbours,
        embedding_provider=embedding_provider,
    )

    investor_data = {
        investor_id: OneInvestorReasoningInput(
            investor_id=investor_id,
            micro_view=micro_view,
            macro_view=macro_view,
            mental_model_bridges=bridges,
        )
        for investor_id, bridges in retrieved.items()
    }
    investor_outputs = {}

    for investor_id, data in investor_data.items():
        unique_codes = {
            candidate.canonical_code
            for bridge in data.mental_model_bridges
            for candidate in bridge.mental_model_candidates
        }
        print(
            f"{investor_id}: bridges={len(data.mental_model_bridges)}, "
            f"unique_models={len(unique_codes)}",
            flush=True,
        )

    if args.dry_run:
        _write_json(
            args.output_path,
            {
                "question": question.model_dump(mode="json"),
                "external_research_settings": external_settings.model_dump(
                    mode="json"
                ),
                "micro_research_data": (
                    micro_research_data.model_dump(mode="json")
                    if micro_research_data
                    else None
                ),
                "macro_research_data": (
                    macro_research_data.model_dump(mode="json")
                    if macro_research_data
                    else None
                ),
                "micro_view": micro_view.model_dump(mode="json"),
                "macro_view": macro_view.model_dump(mode="json"),
                "investor_reasoning_inputs": {
                    investor_id: data.model_dump(mode="json")
                    for investor_id, data in investor_data.items()
                },
                "investor_outputs": {},
                "cio": None,
                "model_configuration": models,
                "embedding_identity": embedding_identity,
            },
        )
        print(
            f"Dry run written to {args.output_path}.",
            flush=True,
        )
        return 0

    total = len(investor_data)
    for number, (investor_id, data) in enumerate(
        investor_data.items(),
        start=1,
    ):
        print(f"Investor {number}/{total}: {investor_id}...", flush=True)
        print(
            f"Investor step {number}/{total} · {investor_id} · "
            "reviewing retrieved mental models...",
            flush=True,
        )
        print(
            f"Investor step {number}/{total} · {investor_id} · "
            "applying mental-model inference...",
            flush=True,
        )
        investor_outputs[investor_id] = run_investor_reasoning(
            data,
            model=models["investor"],
            verbose=args.verbose,
        )
    print(f"CIO synthesis with {models['cio']}...", flush=True)
    cio_data = CIOReasoningInput(
        question=question,
        micro_view=micro_view,
        macro_view=macro_view,
        investor_outputs=investor_outputs,
    )
    cio = run_cio_synthesis(
        cio_data,
        model=models["cio"],
        verbose=args.verbose,
    )

    result = InvestmentCommitteeOutput(
        question=question,
        external_research_settings=external_settings,
        micro_research_data=micro_research_data,
        macro_research_data=macro_research_data,
        micro_view=micro_view,
        macro_view=macro_view,
        investor_reasoning_inputs=investor_data,
        investor_outputs=investor_outputs,
        cio=cio,
        model_configuration=models,
        embedding_identity=embedding_identity,
    )
    _write_json(args.output_path, result.model_dump(mode="json"))
    print(f"Committee result written to {args.output_path}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
