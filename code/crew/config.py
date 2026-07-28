"""Environment-backed configuration for the investor committee MVP."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

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

OPENROUTER_MODEL_PREFIX = "openrouter/"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


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


def _openrouter_headers() -> dict[str, str]:
    """Return optional OpenRouter attribution headers."""

    headers: dict[str, str] = {}
    site_url = os.getenv("OR_SITE_URL", "").strip()
    app_name = os.getenv("OR_APP_NAME", "").strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-OpenRouter-Title"] = app_name
    return headers


def _openrouter_api_base() -> str:
    """Return and validate the OpenRouter-compatible API endpoint."""

    api_base = os.getenv(
        "OPENROUTER_API_BASE",
        DEFAULT_OPENROUTER_API_BASE,
    ).strip().rstrip("/")
    parsed = urlparse(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "OPENROUTER_API_BASE must be a valid HTTP(S) URL."
        )
    return api_base


def create_reasoning_llm(model: str) -> object:
    """Create an OpenRouter connection or retain a direct provider model."""

    qualified_model = qualify_reasoning_model(model)
    if not qualified_model.startswith(OPENROUTER_MODEL_PREFIX):
        return qualified_model

    openrouter_model = qualified_model.removeprefix(
        OPENROUTER_MODEL_PREFIX
    ).strip()
    if not openrouter_model:
        raise ValueError(
            "An OpenRouter model must use "
            "'openrouter/<provider>/<model>' syntax."
        )

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required when a reasoning model uses "
            "the 'openrouter/' prefix."
        )

    # CrewAI's explicit OpenAI provider preserves the complete OpenRouter
    # model slug while using its OpenAI-compatible chat-completions endpoint.
    # This avoids a mandatory LiteLLM dependency and does not affect the
    # separate OpenAI embedding provider.
    from crewai import LLM

    return LLM(
        model=openrouter_model,
        provider="openai",
        api_key=api_key,
        base_url=_openrouter_api_base(),
        default_headers=_openrouter_headers() or None,
        additional_params={
            "extra_body": {
                "provider": {
                    "require_parameters": True,
                }
            }
        },
    )
