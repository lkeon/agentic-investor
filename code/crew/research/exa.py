"""Structured Exa investor-relations research with official-domain grounding."""

from __future__ import annotations

from datetime import date
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from crew.research.common import (
    MAX_DESCRIPTION_WORDS,
    ExternalResearchConfigurationError,
    ExternalResearchError,
    _RequestBudget,
    _gap,
    _normalise_text,
    _request_json,
    _word_limit,
)
from crew.schemas import (
    EvidenceClaim,
    ExternalResearchSettings,
    ResearchGap,
    ResearchSource,
)


EXA_API_URL = "https://api.exa.ai/search"


class _ExaIRExtraction(BaseModel):
    """Minimal structured subset synthesized from official company sources."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    company_description: str = Field(default="", max_length=1200)
    qualitative_observations: list[str] = Field(
        default_factory=list,
        max_length=3,
    )
    credit_rating: str = Field(default="", max_length=80)


EXA_IR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "company_description",
        "qualitative_observations",
        "credit_rating",
    ],
    "properties": {
        "company_description": {
            "type": "string",
            "description": (
                "A factual description of the company's operations in no "
                "more than 100 words, or an empty string if unavailable."
            ),
        },
        "qualitative_observations": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string"},
            "description": (
                "Up to three distinct, decision-useful statements explicitly "
                "made by the company about capital allocation, liquidity, "
                "strategy, investment, or material business risks."
            ),
        },
        "credit_rating": {
            "type": "string",
            "description": (
                "An explicitly disclosed issuer credit rating, or an empty "
                "string when no official company source states one."
            ),
        },
    },
}


def _official_company_url(
    url: str,
    *,
    company_name: str,
) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    blocked = {
        "annualreports.com",
        "bloomberg.com",
        "marketscreener.com",
        "reuters.com",
        "sec.gov",
        "wikipedia.org",
        "yahoo.com",
    }
    if any(host == item or host.endswith("." + item) for item in blocked):
        return False
    company_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", company_name.lower())
        if len(token) >= 5
        and token not in {"company", "corporation", "limited", "holdings"}
    }
    return any(token in host.replace("-", "") for token in company_tokens)


def _exa_structured_ir_search(
    company_name: str,
    *,
    ticker: str | None,
    include_credit_rating: bool,
    maximum_documents: int,
    budget: _RequestBudget,
) -> tuple[_ExaIRExtraction, list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one Exa search and validate its grounded structured output."""

    api_key = os.getenv("EXA_API_KEY", "").strip()
    if not api_key:
        raise ExternalResearchConfigurationError(
            "EXA_API_KEY is required for structured official IR research."
        )
    requested_fields = (
        "a concise company description; up to three current, decision-useful "
        "company statements about capital allocation, liquidity, strategy, "
        "investment, or material business risks"
    )
    if include_credit_rating:
        requested_fields += "; and any explicitly disclosed issuer credit rating"
    else:
        requested_fields += "; set credit_rating to an empty string"
    body = json.dumps(
        {
            "query": (
                f"Using only official investor-relations materials published "
                f"on the company-owned website for {company_name} "
                f"({ticker or 'ticker unavailable'}), return {requested_fields}."
            ),
            "type": "auto",
            "numResults": min(6, max(3, maximum_documents * 3)),
            "systemPrompt": (
                "Use only the named company's official website. Do not use "
                "aggregators, news, analyst research, SEC pages, or model "
                "memory. Do not infer missing facts. Keep statements concise "
                "and preserve important qualifications. Return empty values "
                "when official evidence is unavailable."
            ),
            "outputSchema": EXA_IR_OUTPUT_SCHEMA,
        }
    ).encode("utf-8")
    payload = _request_json(
        EXA_API_URL,
        budget=budget,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
        data=body,
        max_bytes=1_000_000,
    )
    output = payload.get("output")
    if not isinstance(output, dict):
        raise ExternalResearchError(
            "Exa returned no structured investor-relations output."
        )
    content = output.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as error:
            raise ExternalResearchError(
                "Exa returned invalid structured investor-relations JSON."
            ) from error
    try:
        extraction = _ExaIRExtraction.model_validate(content)
    except ValidationError as error:
        raise ExternalResearchError(
            "Exa investor-relations output did not match the required schema."
        ) from error
    raw_grounding = output.get("grounding", [])
    if not isinstance(raw_grounding, list):
        raw_grounding = []
    grounding = [
        entry
        for entry in raw_grounding
        if isinstance(entry, dict)
    ]
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    results = [
        result
        for result in raw_results
        if isinstance(result, dict)
    ]
    return extraction, grounding, results


def _rating_from_text(text: str) -> str | None:
    rating_token = (
        r"AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|CCC[+-]?|"
        r"Aaa|Aa[1-3]|A[1-3]|Baa[1-3]|Ba[1-3]|B[1-3]"
    )
    direct = re.fullmatch(
        rf"\s*({rating_token})(?:\s+\([^)]{{1,40}}\))?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if direct:
        return direct.group(1)
    pattern = re.compile(
        r"(?:credit rating|rated|rating(?:s)?(?: of)?|S&P|Fitch|Moody'?s)"
        rf".{{0,90}}?\b({rating_token})\b",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1) if match else None


def _exa_field_name(value: object) -> str:
    field = str(value or "").strip()
    if field.startswith("content."):
        return field[len("content.") :]
    return field


def _exa_citations_for_field(
    grounding: list[dict[str, Any]],
    field_name: str,
    *,
    item_index: int | None = None,
) -> list[dict[str, Any]]:
    """Return citations attached to one Exa output field."""

    exact_fields = {field_name}
    if item_index is not None:
        exact_fields.add(f"{field_name}[{item_index}]")
    matched: list[dict[str, Any]] = []
    generic: list[dict[str, Any]] = []
    for entry in grounding:
        field = _exa_field_name(entry.get("field"))
        raw_citations = entry.get("citations", [])
        citations = [
            citation
            for citation in raw_citations
            if isinstance(citation, dict)
        ]
        if field in exact_fields:
            matched.extend(citations)
        elif item_index is None and field.startswith(field_name + "["):
            matched.extend(citations)
        elif field in {"", "content"}:
            generic.extend(citations)
    return matched or generic


def _exa_grounding_confidence(
    grounding: list[dict[str, Any]],
    field_name: str,
    *,
    item_index: int | None = None,
) -> float:
    levels: list[str] = []
    exact_fields = {field_name}
    if item_index is not None:
        exact_fields.add(f"{field_name}[{item_index}]")
    for entry in grounding:
        field = _exa_field_name(entry.get("field"))
        if field in exact_fields or (
            item_index is None and field.startswith(field_name + "[")
        ):
            levels.append(str(entry.get("confidence", "")).lower())
    if "high" in levels:
        return 0.9
    if "medium" in levels:
        return 0.8
    return 0.65


def _collect_ir(
    company_name: str,
    *,
    ticker: str | None,
    settings: ExternalResearchSettings,
    budget: _RequestBudget,
) -> dict[str, Any]:
    if settings.max_ir_documents == 0:
        return {
            "sources": [],
            "observations": [],
            "description": None,
            "rating": None,
            "searches": 0,
            "documents": 0,
            "gaps": [
                _gap(
                    "investor_relations",
                    "source_disabled",
                    "The IR document allowance is zero.",
                )
            ],
        }

    gaps: list[ResearchGap] = []
    try:
        extraction, grounding, results = _exa_structured_ir_search(
            company_name,
            ticker=ticker,
            include_credit_rating=settings.credit_rating_enabled,
            maximum_documents=settings.max_ir_documents,
            budget=budget,
        )
    except ExternalResearchConfigurationError:
        raise
    except ExternalResearchError as error:
        return {
            "sources": [],
            "observations": [],
            "description": None,
            "rating": None,
            "searches": 1,
            "documents": 0,
            "gaps": [_gap("IR research", "fetch_failed", str(error))],
        }

    result_by_url = {
        str(result.get("url", "")).strip(): result
        for result in results
        if str(result.get("url", "")).strip()
    }
    cited_fields = (
        ("company_description", None),
        *(
            ("qualitative_observations", index)
            for index in range(len(extraction.qualitative_observations))
        ),
        ("credit_rating", None),
    )
    official_citations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for field_name, item_index in cited_fields:
        for citation in _exa_citations_for_field(
            grounding,
            field_name,
            item_index=item_index,
        ):
            url = str(citation.get("url", "")).strip()
            if (
                url
                and url not in seen_urls
                and _official_company_url(url, company_name=company_name)
            ):
                official_citations.append(citation)
                seen_urls.add(url)
            if len(official_citations) >= settings.max_ir_documents:
                break
        if len(official_citations) >= settings.max_ir_documents:
            break

    sources: list[ResearchSource] = []
    source_id_by_url: dict[str, str] = {}
    for index, citation in enumerate(
        official_citations,
        start=1,
    ):
        url = str(citation["url"]).strip()
        source_id = f"ir_document_{index}"
        source_id_by_url[url] = source_id
        result = result_by_url.get(url, {})
        published_at = None
        raw_date = result.get("publishedDate")
        if isinstance(raw_date, str) and len(raw_date) >= 10:
            try:
                published_at = date.fromisoformat(raw_date[:10])
            except ValueError:
                published_at = None
        sources.append(
            ResearchSource(
                source_id=source_id,
                title=_normalise_text(
                    str(
                        citation.get("title")
                        or result.get("title")
                        or f"{company_name} IR document"
                    )
                )[:500],
                publisher=company_name,
                url=url,
                published_at=published_at,
                source_type="official investor relations",
            )
        )

    def source_ids_for(
        field_name: str,
        *,
        item_index: int | None = None,
    ) -> list[str]:
        source_ids: list[str] = []
        for citation in _exa_citations_for_field(
            grounding,
            field_name,
            item_index=item_index,
        ):
            source_id = source_id_by_url.get(str(citation.get("url", "")).strip())
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
        return source_ids

    description_sources = source_ids_for("company_description")
    description = (
        _word_limit(extraction.company_description, MAX_DESCRIPTION_WORDS)
        if extraction.company_description and description_sources
        else None
    )
    observations: list[EvidenceClaim] = []
    for index, statement in enumerate(extraction.qualitative_observations):
        source_ids = source_ids_for(
            "qualitative_observations",
            item_index=index,
        )
        if not statement or not source_ids:
            continue
        observations.append(
            EvidenceClaim(
                claim_id=f"micro_ext_ir_{index + 1}",
                statement=_normalise_text(statement)[:1600],
                status="management_claim",
                source_ids=source_ids,
                confidence=_exa_grounding_confidence(
                    grounding,
                    "qualitative_observations",
                    item_index=index,
                ),
            )
        )
    rating_sources = source_ids_for("credit_rating")
    rating = (
        _rating_from_text(extraction.credit_rating)
        if settings.credit_rating_enabled
        and extraction.credit_rating
        and rating_sources
        else None
    )

    if not sources:
        gaps.append(
            _gap(
                "investor_relations",
                "ambiguous_identity",
                "Exa returned no grounded citation whose domain could be "
                "validated as the company's official website.",
            )
        )
    if settings.credit_rating_enabled and rating is None:
        gaps.append(
            _gap(
                "credit_rating",
                "not_reported",
                "No reliable rating disclosure was found in the bounded "
                "official IR documents.",
            )
        )
    return {
        "sources": sources,
        "observations": observations,
        "description": description,
        "rating": rating,
        "searches": 1,
        "documents": len(sources),
        "gaps": gaps,
    }
