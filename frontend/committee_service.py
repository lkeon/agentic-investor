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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "crew" / "committee_result.json"
)
CHECKPOINT_PATH = DEFAULT_OUTPUT_PATH.with_name(
    f"{DEFAULT_OUTPUT_PATH.stem}.checkpoint{DEFAULT_OUTPUT_PATH.suffix}"
)
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_PROCESS_LOCK = Lock()
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None
_STOPPED_PROCESS_IDS: set[int] = set()


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


def discover_investors() -> list[str]:
    """Discover investor IDs from the local canonical export."""

    export_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "canonical"
        / "canonical_mental_models.jsonl"
    )
    investors: set[str] = set()
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

    if not DEFAULT_OUTPUT_PATH.is_file():
        return None
    try:
        return _load_validated_result(DEFAULT_OUTPUT_PATH)
    except CommitteeRunError:
        # A dry run or interrupted legacy run may have written a partial
        # artifact. It is not a completed result and should not break the UI.
        return None


def _stage_update(line: str) -> ProgressUpdate | None:
    """Translate stable CLI messages into concise product progress."""

    fixed_stages = (
        ("Normalising question", "Clarifying the investment decision", 8),
        ("Building MicroView", "Structuring company evidence", 20),
        ("Building MacroView", "Mapping the macro context", 32),
        (
            "Building mental-model data bridges",
            "Forming analytical questions",
            43,
        ),
        ("Retrieving mental models", "Retrieving investor mental models", 54),
        ("CIO synthesis", "Synthesising the committee decision", 94),
        ("Committee result written", "Committee analysis complete", 100),
    )
    for marker, label, progress in fixed_stages:
        if marker in line:
            return ProgressUpdate(label, progress, line)

    round_match = re.search(r"Round ([12]) (\d+)/(\d+): ([^\.]+)", line)
    if round_match:
        round_number, current, total, investor_id = round_match.groups()
        fraction = int(current) / max(1, int(total))
        if round_number == "1":
            progress = 55 + round(19 * fraction)
            label = f"Independent analysis · {display_name(investor_id)}"
        else:
            progress = 75 + round(17 * fraction)
            label = f"Peer review · {display_name(investor_id)}"
        return ProgressUpdate(label, progress, line)

    if "Resuming from" in line:
        return ProgressUpdate("Restoring validated progress", 50, line)
    if "PostgreSQL is unavailable" in line:
        return ProgressUpdate("Starting the evidence database", 46, line)
    if "PostgreSQL is ready" in line:
        return ProgressUpdate("Evidence database ready", 49, line)
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
    resume: bool,
    on_progress: Callable[[ProgressUpdate, list[str]], None] | None = None,
) -> dict[str, object]:
    """Run the real committee CLI and return its validated JSON artifact."""

    global _ACTIVE_PROCESS

    if not question.strip():
        raise ValueError("An investment question is required.")
    if not investors:
        raise ValueError("Select at least one investor perspective.")

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
        str(DEFAULT_OUTPUT_PATH),
    ]
    for investor_id in investors:
        command.extend(["--investor", investor_id])
    if resume:
        command.append("--resume")

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
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = ANSI_ESCAPE.sub("", raw_line).strip()
            if not line:
                continue
            logs.append(line)
            logs = logs[-120:]
            update = _stage_update(line)
            if update and on_progress:
                on_progress(update, logs)

        return_code = process.wait()
        with _PROCESS_LOCK:
            was_stopped = process.pid in _STOPPED_PROCESS_IDS
            _STOPPED_PROCESS_IDS.discard(process.pid)
        if was_stopped:
            raise CommitteeStopped("Committee execution stopped.", logs)
        if return_code != 0:
            detail = logs[-1] if logs else "No diagnostic output was returned."
            raise CommitteeRunError(
                f"Committee execution failed: {detail}",
                logs,
            )
        return _load_validated_result(DEFAULT_OUTPUT_PATH)
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
