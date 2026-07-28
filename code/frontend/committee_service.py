"""Adapter between the Streamlit UI and the existing committee CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from tempfile import NamedTemporaryFile
from threading import Lock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "crew" / "committee_result.json"
)
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_PROCESS_LOCK = Lock()
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None
_STOPPED_PROCESS_IDS: set[int] = set()
MAX_TECHNICAL_LOG_LINES = 8000
STREAMLIT_CLOUD_MODE = "streamlit_cloud"


def _is_streamlit_cloud() -> bool:
    """Return whether shared cloud-session safeguards should be enabled."""

    return (
        os.getenv("DILIGENCE_DEPLOYMENT", "").strip().lower()
        == STREAMLIT_CLOUD_MODE
    )


@dataclass(frozen=True)
class ProgressUpdate:
    """One user-facing stage emitted while the CLI is running."""

    label: str
    progress: int
    detail: str


class CommitteeRunError(RuntimeError):
    """Raised when the committee CLI exits without a valid result."""

    def __init__(self, message: str, logs: list[str]) -> None:
        self.logs = logs
        super().__init__(message)


class CommitteeStopped(CommitteeRunError):
    """Raised when the user stops an active committee run."""


class InvestorDiscoveryError(RuntimeError):
    """Raised when a hosted app cannot discover usable database investors."""


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate the crew process group, escalating only when it does not exit."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=2)


def stop_active_committee() -> bool:
    """Stop the committee currently launched by this frontend process."""

    with _PROCESS_LOCK:
        process = _ACTIVE_PROCESS
        if process is None or process.poll() is not None:
            return False
        _STOPPED_PROCESS_IDS.add(process.pid)
    _terminate_process(process)
    return True


def _discover_investors_from_database() -> set[str]:
    """Return investors with retrievable canonical models in PostgreSQL."""

    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    try:
        from sqlalchemy import select
        from sqlalchemy.exc import SQLAlchemyError

        from mental_model_pipeline.canonical.db_models import (
            CanonicalMentalModelDB,
        )
        from mental_model_pipeline.canonical.embeddings import (
            accepted_embedding_identities,
        )
        from mental_model_pipeline.database.connection import SessionLocal
    except (ImportError, RuntimeError) as error:
        if _is_streamlit_cloud():
            raise InvestorDiscoveryError(
                "The hosted mental-model database could not be configured. "
                "Verify DATABASE_URL and the deployment dependencies."
            ) from error
        # Local rendering without configured infrastructure can still use the
        # canonical export or the minimal fallback below.
        return set()

    try:
        with SessionLocal() as session:
            return {
                investor_id.strip()
                for investor_id in session.scalars(
                    select(CanonicalMentalModelDB.investor_id)
                    .where(
                        CanonicalMentalModelDB.embedding.is_not(None),
                        CanonicalMentalModelDB.embedding_model.in_(
                            accepted_embedding_identities()
                        ),
                    )
                    .distinct()
                    .order_by(CanonicalMentalModelDB.investor_id)
                )
                if investor_id and investor_id.strip()
            }
    except SQLAlchemyError as error:
        if _is_streamlit_cloud():
            raise InvestorDiscoveryError(
                "The hosted mental-model database could not be queried. "
                "Verify DATABASE_URL, network access, TLS settings, and that "
                "the canonical tables exist."
            ) from error
        return set()


def discover_investors() -> list[str]:
    """Discover retrievable investor IDs, with local startup fallbacks."""

    investors = _discover_investors_from_database()
    if investors:
        return sorted(investors)

    if _is_streamlit_cloud():
        raise InvestorDiscoveryError(
            "The hosted database is reachable but contains no canonical "
            "mental models compatible with the configured embedding identity."
        )

    # The export is useful for local UI rendering before PostgreSQL starts,
    # but it is intentionally excluded from deployed source control.
    export_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "canonical"
        / "canonical_mental_models.jsonl"
    )
    if export_path.is_file():
        with export_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    investor_id = json.loads(line).get("investor_id")
                except json.JSONDecodeError:
                    continue
                if isinstance(investor_id, str) and investor_id.strip():
                    investors.add(investor_id.strip())
    return sorted(investors) or ["buffett", "flatt", "marks"]


def load_latest_result() -> dict[str, object] | None:
    """Load and validate the most recent completed committee artifact."""

    # A deployed Streamlit process is shared by multiple browser sessions.
    # Never expose a previous session's locally persisted committee result.
    if _is_streamlit_cloud():
        return None
    if not DEFAULT_OUTPUT_PATH.is_file():
        return None
    try:
        return _load_validated_result(DEFAULT_OUTPUT_PATH)
    except CommitteeRunError:
        # A dry run may have written a partial artifact. It is not a completed
        # result and should not break the UI.
        return None


def _stage_update(line: str) -> ProgressUpdate | None:
    """Translate stable CLI messages into customer-facing progress."""

    fixed_stages = (
        (
            "Normalising question",
            "Clarifying the investment decision",
            8,
            (
                "The question is being translated into a precise value-investing "
                "decision without adding unsupported facts."
            ),
        ),
        (
            "Building MicroView",
            "Structuring company evidence",
            20,
            (
                "The supplied company, financial, management, valuation, and "
                "risk evidence is being organised into an auditable MicroView."
            ),
        ),
        (
            "Building MacroView",
            "Mapping material macro influences",
            32,
            (
                "Only macro conditions with a direct, material connection to "
                "the company thesis are being identified."
            ),
        ),
        (
            "Building mental-model data bridges",
            "Defining analytical pathways",
            43,
            (
                "The evidence is being translated into focused questions for "
                "mental-model retrieval."
            ),
        ),
        (
            "Retrieving mental models",
            "Retrieving reasoning guardrails",
            54,
            (
                "The hierarchical model network is being searched separately "
                "for each selected investor perspective."
            ),
        ),
        (
            "CIO synthesis",
            "Synthesising the final decision",
            94,
            (
                "The CIO is weighing the independent perspectives, decisive "
                "evidence, risks, and unresolved information."
            ),
        ),
        (
            "Committee result written",
            "Diligence complete",
            100,
            (
                "The validated decision record is ready for review and "
                "download."
            ),
        ),
    )
    for marker, label, progress, detail in fixed_stages:
        if marker in line:
            return ProgressUpdate(label, progress, detail)

    investor_match = re.search(r"Investor (\d+)/(\d+): ([^\.]+)", line)
    if investor_match:
        current, total, investor_id = investor_match.groups()
        fraction = int(current) / max(1, int(total))
        progress = 55 + round(37 * fraction)
        investor_name = display_name(investor_id)
        return ProgressUpdate(
            f"Independent analysis · {investor_name}",
            progress,
            (
                f"The {investor_name} perspective is testing the supplied "
                "evidence against its retrieved mental-model set."
            ),
        )

    if "PostgreSQL is unavailable" in line:
        return ProgressUpdate(
            "Preparing the model library",
            46,
            (
                "The local mental-model database is being started before "
                "retrieval."
            ),
        )
    if "PostgreSQL is ready" in line:
        return ProgressUpdate(
            "Model library ready",
            49,
            "The canonical mental-model library is available for retrieval.",
        )
    return None


def display_name(identifier: str) -> str:
    """Convert a stable machine identifier into a UI label."""

    return identifier.replace("_", " ").replace("-", " ").title()


def run_committee(
    question: str,
    *,
    investors: list[str],
    research_context: str | None,
    top_k: int,
    neighbours: int,
    technical_debug: bool = False,
    on_progress: Callable[[ProgressUpdate, list[str]], None] | None = None,
) -> dict[str, object]:
    """Run the real committee CLI and return its validated JSON artifact."""

    global _ACTIVE_PROCESS

    if not question.strip():
        raise ValueError("An investment question is required.")
    if not investors:
        raise ValueError("Select at least one investor perspective.")

    output_path = DEFAULT_OUTPUT_PATH
    ephemeral_output = False
    if _is_streamlit_cloud():
        # Each hosted run gets an isolated result artifact. The validated
        # result is returned to that session and the temporary file is removed.
        with NamedTemporaryFile(
            suffix=".json",
            prefix="diligence-result-",
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)
        ephemeral_output = True

    command = [
        sys.executable,
        "-m",
        "crew.run_crew",
        question.strip(),
        "--top-k",
        str(top_k),
        "--neighbours",
        str(neighbours),
        "--output-path",
        str(output_path),
    ]
    for investor_id in investors:
        command.extend(["--investor", investor_id])
    if technical_debug:
        # CrewAI verbose output is useful for prompt, task, tool, validation,
        # timing, and model-quality diagnosis. It is shown only in the local
        # technical log selected by the user.
        command.append("--verbose")
    context_path: Path | None = None
    process: subprocess.Popen[str] | None = None
    try:
        if research_context and research_context.strip():
            # The CLI already owns research-file parsing. A temporary UTF-8
            # file keeps the UI adapter thin and avoids a second code path.
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".txt",
                prefix="committee-research-",
                delete=False,
            ) as context_file:
                context_file.write(research_context)
                context_path = Path(context_file.name)
            command.extend(["--research-context", str(context_path)])

        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(CODE_ROOT) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # A dedicated process group lets the Stop button terminate the
            # crew and any child processes it may have launched.
            start_new_session=True,
        )
        with _PROCESS_LOCK:
            _ACTIVE_PROCESS = process

        logs: list[str] = []
        current_update = ProgressUpdate(
            "Preparing the diligence workspace",
            3,
            (
                "The selected perspectives, research context, and local "
                "runtime are being prepared."
            ),
        )
        if on_progress:
            on_progress(current_update, logs)

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = ANSI_ESCAPE.sub("", raw_line.rstrip("\r\n"))
            # Preserve CrewAI panels and stderr diagnostics verbatim. Collapse
            # only repeated blank lines to keep long debug sessions readable.
            if not line and logs and not logs[-1]:
                continue
            logs.append(line)
            logs = logs[-MAX_TECHNICAL_LOG_LINES:]
            update = _stage_update(line)
            if update is not None:
                current_update = update
            if on_progress:
                on_progress(current_update, logs)

        return_code = process.wait()
        with _PROCESS_LOCK:
            was_stopped = process.pid in _STOPPED_PROCESS_IDS
            _STOPPED_PROCESS_IDS.discard(process.pid)
        if was_stopped:
            raise CommitteeStopped("Committee execution stopped.", logs)
        if return_code != 0:
            detail = next(
                (line for line in reversed(logs) if line.strip()),
                "No diagnostic output was returned.",
            )
            raise CommitteeRunError(
                f"Committee execution failed: {detail}",
                logs,
            )
        return _load_validated_result(output_path)
    finally:
        if process is not None:
            if process.poll() is None:
                _terminate_process(process)
            with _PROCESS_LOCK:
                if _ACTIVE_PROCESS is process:
                    _ACTIVE_PROCESS = None
                _STOPPED_PROCESS_IDS.discard(process.pid)
        if context_path is not None:
            context_path.unlink(missing_ok=True)
        if ephemeral_output:
            output_path.unlink(missing_ok=True)


def _load_validated_result(path: Path) -> dict[str, object]:
    """Validate UI input against the same Pydantic contract as the CLI."""

    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    from crew.schemas import InvestmentCommitteeOutput

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = InvestmentCommitteeOutput.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise CommitteeRunError(
            f"The committee result is missing or invalid: {error}",
            [],
        ) from error
    return result.model_dump(mode="json")
