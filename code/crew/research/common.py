"""Shared bounded-research transport, errors, limits, and progress events."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from crew.schemas import ResearchGap


# common.py lives in code/crew/research, so the repository is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

MAX_DESCRIPTION_WORDS = 100
MAX_MICRO_EXTERNAL_REQUESTS = 16
ResearchProgress = Callable[[str], None]


class ExternalResearchError(RuntimeError):
    """A bounded source request or deterministic extraction failed."""


class ExternalResearchConfigurationError(ExternalResearchError):
    """A selected source is missing required configuration."""


def _normalise_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _progress_text(value: object, *, limit: int = 500) -> str:
    """Keep callback messages single-line and safe for terminal parsing."""

    return _normalise_text(str(value)).replace(" | ", " — ")[:limit]


def _report_progress(
    progress: ResearchProgress | None,
    provider: str,
    status: str,
    detail: str,
) -> None:
    if progress is not None:
        progress(
            "External research | "
            f"{_progress_text(provider, limit=100)} | "
            f"{_progress_text(status, limit=40)} | "
            f"{_progress_text(detail)}"
        )


@dataclass
class _RequestBudget:
    """Count actual HTTP attempts and reject work beyond the hard ceiling."""

    maximum: int
    used: int = 0

    def take(self) -> None:
        if self.used >= self.maximum:
            raise ExternalResearchError(
                f"External request budget of {self.maximum} is exhausted."
            )
        self.used += 1


def _request_bytes(
    url: str,
    *,
    budget: _RequestBudget,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    max_bytes: int,
    timeout: float = 20.0,
) -> bytes:
    """Fetch one bounded payload with one retry for transient network errors."""

    request_headers = {
        "Accept": "application/json,text/html,application/pdf,*/*",
        **(headers or {}),
    }
    last_error: Exception | None = None
    for attempt in range(2):
        budget.take()
        try:
            request = Request(
                url,
                data=data,
                headers=request_headers,
                method="POST" if data is not None else "GET",
            )
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise ExternalResearchError(
                    f"Source payload exceeded the {max_bytes:,}-byte limit."
                )
            return payload
        except HTTPError as error:
            last_error = error
            if error.code < 500 or attempt:
                break
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt:
                break
        time.sleep(0.25)
    raise ExternalResearchError(f"Request failed for {url}: {last_error}")


def _request_json(
    url: str,
    *,
    budget: _RequestBudget,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    max_bytes: int = 4_000_000,
) -> dict[str, Any]:
    payload = _request_bytes(
        url,
        budget=budget,
        headers=headers,
        data=data,
        max_bytes=max_bytes,
    )
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalResearchError(
            f"Source returned invalid JSON for {url}."
        ) from error
    if not isinstance(decoded, dict):
        raise ExternalResearchError(f"Source returned a non-object for {url}.")
    return decoded


def _gap(field: str, status: str, detail: str) -> ResearchGap:
    return ResearchGap(field=field, status=status, detail=detail)


def _word_limit(text: str, maximum: int) -> str:
    words = text.split()
    return " ".join(words[:maximum])
