"""Environment-backed configuration for the investor committee MVP."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)


DEFAULT_REASONING_MODEL = os.getenv(
    "DEFAULT_REASONING_MODEL",
    "openai/gpt-5-nano-2025-08-07",
)
DEFAULT_REASONING_PROVIDER = os.getenv(
    "DEFAULT_REASONING_PROVIDER",
    "openai",
)

STAGE_MODEL_VARIABLES = {
    "question": ("QUESTION_MODEL", "QUESTION_NORMALISER_MODEL"),
    "research": ("RESEARCH_MODEL",),
    "bridge": ("BRIDGE_MODEL",),
    "investor": ("INVESTOR_MODEL", "INVESTOR_ANALYSIS_MODEL"),
    "cio": ("CIO_MODEL",),
}


def qualify_reasoning_model(model: str) -> str:
    """Return a CrewAI provider-qualified model name."""

    cleaned = model.strip()
    if not cleaned:
        raise ValueError("Reasoning model cannot be empty.")
    if "/" in cleaned:
        return cleaned
    return f"{DEFAULT_REASONING_PROVIDER}/{cleaned}"


def reasoning_model(
    stage: str,
    *,
    override: str | None = None,
) -> str:
    """Resolve a stage override, environment value, or global default."""

    if stage not in STAGE_MODEL_VARIABLES:
        raise ValueError(f"Unknown reasoning stage: {stage}")

    environment_model = next(
        (
            value
            for variable in STAGE_MODEL_VARIABLES[stage]
            if (value := os.getenv(variable))
        ),
        None,
    )
    selected = override or environment_model or DEFAULT_REASONING_MODEL
    return qualify_reasoning_model(selected)
