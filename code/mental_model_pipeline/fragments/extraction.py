"""Extract and validate mental-model fragments from documents."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from mental_model_pipeline.fragments.schemas import (
    MentalModelExtractionResult,
    MentalModelFragment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

MAX_FRAGMENTS_PER_DOCUMENT = 10
MAX_OUTPUT_TOKENS = 32_000
MAX_OUTPUT_TOKEN_CAP = 128_000
MAX_EXTRACTION_ATTEMPTS = 3
EXTRACTION_MODEL = os.getenv(
    "FRAGMENT_EXTRACTION_MODEL",
    "gpt-5.6-terra",
)

client = OpenAI()


SYSTEM_PROMPT = """
    You extract mental model fragments from investor documents.

    A mental model fragment is an atomic, generalisable investment proposition,
    decision rule, causal mechanism, risk rule, valuation rule, condition,
    exception, or observation that could inform an investment decision.

    Requirements:

    1. Extract only ideas supported by the supplied text.
    2. Do not introduce external knowledge.
    3. Do not merely summarise the document.
    4. Each fragment must express one reasonably atomic idea.
    5. Generalise the idea enough to apply beyond the specific company discussed.
    6. Preserve limitations, conditions, and failure cases.
    7. source_quote must quote the supplied text directly.
    8. Set fragment_code to null. The application assigns it.
    9. Use related_entities only where an entity meaningfully illustrates the idea.
    10. If the text merely mentions an entity, use mentioned_only.
    11. Use directly_stated only when the proposition is explicit.
    12. Weak inference must set requires_review to true and provide review_reason.
    13. A quoted third-party idea must identify attributed_to.
    14. Return an empty fragments list when the text contains no useful investment
        mental model.
    15. Keep every field concise.
    16. Use the shortest exact source quote that adequately supports the fragment.
    17. Include no more than three items in each list unless additional items are
        essential to preserve the idea.
    18. Avoid repeating the same explanation across proposition, mechanism,
        conditions, and decision implications.
    19. If attribution_type is third_party_quoted_by_investor,
        attributed_to must contain the named person or organisation.
    20. If no third party is explicitly named, use investor or unclear instead;
        never return third_party_quoted_by_investor with attributed_to set to
        null.
    """.strip()


def normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def quote_appears_in_text(
    quote: str,
    chunk_text: str,
) -> bool:
    """
    Check the quote after normalising whitespace.

    Markdown line wrapping should not make an otherwise exact quote fail.
    """

    normalised_quote = normalise_whitespace(quote)
    normalised_chunk = normalise_whitespace(chunk_text)

    return normalised_quote in normalised_chunk


def extract_fragments_from_document(
    *,
    markdown_text: str,
    investor_name: str,
    document_id: str,
) -> MentalModelExtractionResult:
    """
    Extract up to ten mental-model fragments from one complete document.
    """

    if not markdown_text.strip():
        return MentalModelExtractionResult()

    user_prompt = f"""
Principal investor or author: {investor_name}
Document ID: {document_id}

Read the complete document and extract no more than
{MAX_FRAGMENTS_PER_DOCUMENT} mental-model fragments.

Select the most important, distinct and generally applicable investment
models in the document.

Do not simply select the first ideas encountered. Consider the complete
document before selecting the final fragments.

Avoid extracting several fragments that express essentially the same idea.

Set fragment_code to null. The application assigns it.

DOCUMENT START

{markdown_text}

DOCUMENT END
""".strip()

    last_validation_error: ValidationError | None = None

    for attempt in range(MAX_EXTRACTION_ATTEMPTS):
        output_tokens = min(
            MAX_OUTPUT_TOKENS * (2**attempt),
            MAX_OUTPUT_TOKEN_CAP,
        )
        request_kwargs: dict[str, object] = {
            "model": EXTRACTION_MODEL,
            "input": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "text_format": MentalModelExtractionResult,
            "max_output_tokens": output_tokens,
        }

        if attempt > 0:
            request_kwargs["reasoning"] = {"effort": "low"}

        try:
            response = client.responses.parse(**request_kwargs)
            break
        except ValidationError as exc:
            invalid_json = any(
                error["type"] == "json_invalid"
                for error in exc.errors()
            )

            if not invalid_json:
                raise

            last_validation_error = exc
            if attempt == MAX_EXTRACTION_ATTEMPTS - 1:
                raise RuntimeError(
                    "The extraction model returned incomplete or invalid JSON "
                    f"after {MAX_EXTRACTION_ATTEMPTS} attempts. The final "
                    f"output limit was {output_tokens} tokens. The document "
                    "was not stored."
                ) from exc
    else:  # pragma: no cover - every loop path either returns or raises
        raise RuntimeError(
            "Extraction attempts ended without a response."
        ) from last_validation_error

    parsed_result = response.output_parsed

    if parsed_result is None:
        raise RuntimeError(
            "The extraction model returned no parsed result."
        )

    validated_fragments: list[MentalModelFragment] = []

    for fragment in parsed_result.fragments[
        :MAX_FRAGMENTS_PER_DOCUMENT
    ]:
        updates: dict[str, object] = {
            "fragment_code": None,
        }

        if not quote_appears_in_text(
            fragment.source_quote,
            markdown_text,
        ):
            existing_reason = (
                f"{fragment.review_reason} "
                if fragment.review_reason
                else ""
            )

            updates["requires_review"] = True
            updates["review_reason"] = (
                existing_reason
                + "The source quote could not be matched "
                "against the complete source document."
            )

        validated_fragments.append(
            fragment.model_copy(update=updates)
        )

    return MentalModelExtractionResult(
        fragments=validated_fragments
    )
