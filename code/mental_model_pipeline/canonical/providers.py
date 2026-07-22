"""Provide a small structured-output interface for configurable language-model providers."""

from __future__ import annotations

from typing import Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredGenerationProvider(Protocol):
    model: str

    def parse(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> SchemaT:
        ...


class OpenAIStructuredProvider:
    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI()

    def parse(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> SchemaT:
        request: dict = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text_format": schema,
            "max_output_tokens": max_output_tokens,
        }

        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}

        response = self.client.responses.parse(**request)
        parsed = response.output_parsed

        if parsed is None:
            raise RuntimeError(
                "Structured response contained no parsed output. "
                f"status={getattr(response, 'status', None)!r}; "
                "incomplete_details="
                f"{getattr(response, 'incomplete_details', None)!r}"
            )

        return parsed
