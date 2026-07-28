"""CLI entry point for the value-investing committee MVP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
    run_round_one,
    run_round_two,
)
from crew.config import reasoning_model
from crew.retrieval import retrieve_for_bridges
from crew.schemas import (
    CIOReasoningInput,
    CIOReasoningOutput,
    InvestmentCommitteeOutput,
    InvestmentQuestion,
    InvestorPeerReviewOutput,
    InvestorReasoningOutput,
    MacroView,
    MicroView,
    OneInvestorReasoningInput,
)
from mental_model_pipeline.canonical.embeddings import (
    create_embedding_provider,
)
from mental_model_pipeline.database.connection import engine


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class _CheckpointState:
    question: InvestmentQuestion
    micro_view: MicroView
    macro_view: MacroView
    investor_data: dict[str, OneInvestorReasoningInput]
    round_one: dict[str, InvestorReasoningOutput]
    round_two: dict[str, InvestorPeerReviewOutput]
    cio: CIOReasoningOutput | None
    embedding_identity: str


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
            "mental models, run two committee rounds, and synthesize a CIO "
            "decision."
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
        "--question-model",
        "--normaliser-model",
        dest="question_model",
    )
    parser.add_argument("--research-model")
    parser.add_argument("--bridge-model")
    parser.add_argument(
        "--investor-round-one-model",
        "--analysis-model",
        dest="investor_round_one_model",
    )
    parser.add_argument(
        "--investor-round-two-model",
        "--peer-review-model",
        dest="investor_round_two_model",
    )
    parser.add_argument("--cio-model")

    parser.add_argument("--skip-round-two", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run normalization, structured research, bridge construction, and "
            "retrieval, but skip investor reasoning and CIO synthesis."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume reasoning from the matching checkpoint beside output-path. "
            "The question, research context, investors, retrieval settings, "
            "and model configuration must match the original run."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/crew/committee_result.json"),
    )
    args = parser.parse_args()
    if args.resume and args.dry_run:
        parser.error("--resume and --dry-run cannot be used together.")
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


def _checkpoint_path(output_path: Path) -> Path:
    """Keep partial state separate from the completed committee artifact."""

    return output_path.with_name(
        f"{output_path.stem}.checkpoint{output_path.suffix}"
    )


def _checkpoint_request(
    args: argparse.Namespace,
    models: dict[str, str],
    research_context: str | None,
) -> dict[str, object]:
    """Fingerprint every input that could alter a resumed committee run."""

    context_bytes = (research_context or "").encode("utf-8")
    return {
        "question": args.question,
        "research_context_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "investors": sorted(args.investor) if args.investor else None,
        "top_k": args.top_k,
        "neighbours": args.neighbours,
        "skip_round_two": args.skip_round_two,
        "model_configuration": models,
    }


def _write_checkpoint(
    *,
    output_path: Path,
    status: str,
    request: dict[str, object],
    question: InvestmentQuestion,
    micro_view: MicroView,
    macro_view: MacroView,
    investor_data: dict[str, OneInvestorReasoningInput],
    round_one: dict[str, InvestorReasoningOutput],
    round_two: dict[str, InvestorPeerReviewOutput],
    embedding_identity: str,
    cio: CIOReasoningOutput | None = None,
) -> None:
    """Atomically persist the last fully validated stage of a committee run."""

    _write_json(
        _checkpoint_path(output_path),
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "status": status,
            "request": request,
            "question": question.model_dump(mode="json"),
            "micro_view": micro_view.model_dump(mode="json"),
            "macro_view": macro_view.model_dump(mode="json"),
            "investor_reasoning_inputs": {
                investor_id: data.model_dump(mode="json")
                for investor_id, data in investor_data.items()
            },
            "round_one": {
                investor_id: output.model_dump(mode="json")
                for investor_id, output in round_one.items()
            },
            "round_two": {
                investor_id: output.model_dump(mode="json")
                for investor_id, output in round_two.items()
            },
            "cio": cio.model_dump(mode="json") if cio else None,
            "embedding_identity": embedding_identity,
        },
    )


def _load_checkpoint(
    *,
    output_path: Path,
    expected_request: dict[str, object],
) -> _CheckpointState:
    """Load a checkpoint only when it belongs to this exact invocation."""

    path = _checkpoint_path(output_path)
    if not path.is_file():
        raise FileNotFoundError(f"Committee checkpoint not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            "Checkpoint version is incompatible with this crew runtime."
        )
    if payload.get("request") != expected_request:
        raise ValueError(
            "Checkpoint inputs do not match this invocation. Re-run without "
            "--resume or restore the original options."
        )

    state = _CheckpointState(
        question=InvestmentQuestion.model_validate(payload["question"]),
        micro_view=MicroView.model_validate(payload["micro_view"]),
        macro_view=MacroView.model_validate(payload["macro_view"]),
        investor_data={
            investor_id: OneInvestorReasoningInput.model_validate(data)
            for investor_id, data in payload[
                "investor_reasoning_inputs"
            ].items()
        },
        round_one={
            investor_id: InvestorReasoningOutput.model_validate(output)
            for investor_id, output in payload.get("round_one", {}).items()
        },
        round_two={
            investor_id: InvestorPeerReviewOutput.model_validate(output)
            for investor_id, output in payload.get("round_two", {}).items()
        },
        cio=(
            CIOReasoningOutput.model_validate(payload["cio"])
            if payload.get("cio")
            else None
        ),
        embedding_identity=payload["embedding_identity"],
    )
    investor_ids = set(state.investor_data)
    if set(state.round_one) - investor_ids or set(state.round_two) - investor_ids:
        raise ValueError("Checkpoint contains outputs for unknown investors.")
    return state


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
        "investor_round_one": reasoning_model(
            "investor_round_one",
            override=args.investor_round_one_model,
        ),
        "investor_round_two": reasoning_model(
            "investor_round_two",
            override=args.investor_round_two_model,
        ),
        "cio": reasoning_model(
            "cio",
            override=args.cio_model,
        ),
    }


def main() -> int:
    args = parse_arguments()
    models = _model_configuration(args)
    research_context = _read_research_context(args.research_context)
    checkpoint_request = _checkpoint_request(args, models, research_context)

    if args.resume:
        state = _load_checkpoint(
            output_path=args.output_path,
            expected_request=checkpoint_request,
        )
        question = state.question
        micro_view = state.micro_view
        macro_view = state.macro_view
        investor_data = state.investor_data
        round_one = state.round_one
        round_two = state.round_two
        cio = state.cio
        embedding_identity = state.embedding_identity
        print(
            f"Resuming from {_checkpoint_path(args.output_path)}: "
            f"round_one={len(round_one)}, round_two={len(round_two)}.",
            flush=True,
        )
    else:
        print(
            f"Normalising question with {models['question']}...",
            flush=True,
        )
        question = normalise_question(
            args.question,
            model=models["question"],
            verbose=args.verbose,
        )

        print(
            f"Building MicroView with {models['research']}...",
            flush=True,
        )
        micro_view = research_micro_view(
            question,
            research_context=research_context,
            model=models["research"],
            verbose=args.verbose,
        )

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
            f"Building mental-model data bridges with {models['bridge']}...",
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
        round_one = {}
        round_two = {}
        cio = None

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
                "micro_view": micro_view.model_dump(mode="json"),
                "macro_view": macro_view.model_dump(mode="json"),
                "investor_reasoning_inputs": {
                    investor_id: data.model_dump(mode="json")
                    for investor_id, data in investor_data.items()
                },
                "round_one": {},
                "round_two": {},
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

    def save_checkpoint(status: str) -> None:
        _write_checkpoint(
            output_path=args.output_path,
            status=status,
            request=checkpoint_request,
            question=question,
            micro_view=micro_view,
            macro_view=macro_view,
            investor_data=investor_data,
            round_one=round_one,
            round_two=round_two,
            cio=cio,
            embedding_identity=embedding_identity,
        )

    if not args.resume:
        save_checkpoint("retrieval_complete")

    total = len(investor_data)
    for number, (investor_id, data) in enumerate(
        investor_data.items(),
        start=1,
    ):
        if investor_id in round_one:
            continue
        print(f"Round 1 {number}/{total}: {investor_id}...", flush=True)
        round_one[investor_id] = run_round_one(
            data,
            model=models["investor_round_one"],
            verbose=args.verbose,
        )
        save_checkpoint(f"round_one_{len(round_one)}_of_{total}")

    if not args.skip_round_two and len(round_one) > 1:
        for number, (investor_id, data) in enumerate(
            investor_data.items(),
            start=1,
        ):
            if investor_id in round_two:
                continue
            print(f"Round 2 {number}/{total}: {investor_id}...", flush=True)
            peers = {
                peer_id: view
                for peer_id, view in round_one.items()
                if peer_id != investor_id
            }
            round_two[investor_id] = run_round_two(
                data,
                round_one[investor_id],
                peers,
                model=models["investor_round_two"],
                verbose=args.verbose,
            )
            save_checkpoint(f"round_two_{len(round_two)}_of_{total}")

    if cio is None:
        print(f"CIO synthesis with {models['cio']}...", flush=True)
        cio_data = CIOReasoningInput(
            question=question,
            micro_view=micro_view,
            macro_view=macro_view,
            round_one_outputs=round_one,
            round_two_outputs=round_two,
        )
        allowed_model_codes = {
            candidate.canonical_code
            for data in investor_data.values()
            for bridge in data.mental_model_bridges
            for candidate in bridge.mental_model_candidates
        }
        cio = run_cio_synthesis(
            cio_data,
            allowed_model_codes=allowed_model_codes,
            model=models["cio"],
            verbose=args.verbose,
        )
        save_checkpoint("committee_complete")

    result = InvestmentCommitteeOutput(
        question=question,
        micro_view=micro_view,
        macro_view=macro_view,
        investor_reasoning_inputs=investor_data,
        round_one=round_one,
        round_two=round_two,
        cio=cio,
        model_configuration=models,
        embedding_identity=embedding_identity,
    )
    _write_json(args.output_path, result.model_dump(mode="json"))
    print(f"Committee result written to {args.output_path}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
