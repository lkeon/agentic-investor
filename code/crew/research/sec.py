"""SEC identity, XBRL normalization, and bounded filing retrieval."""

from __future__ import annotations

from datetime import date
from html.parser import HTMLParser
import os
import re
from typing import Any
from urllib.parse import urlparse

from crew.research.common import (
    MAX_DESCRIPTION_WORDS,
    ExternalResearchConfigurationError,
    ExternalResearchError,
    _RequestBudget,
    _gap,
    _normalise_text,
    _request_bytes,
    _request_json,
    _word_limit,
)
from crew.schemas import (
    EvidenceClaim,
    ExternalResearchSettings,
    FinancialPeriodData,
    InvestmentQuestion,
    ResearchGap,
    ResearchSource,
)


SEC_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
MAX_SEC_DOCUMENT_BYTES = 8_000_000


class _TextExtractor(HTMLParser):
    """Small standard-library HTML-to-text reducer."""

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif tag in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        return _normalise_text(" ".join(self.parts))


def _sec_headers() -> dict[str, str]:
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise ExternalResearchConfigurationError(
            "SEC_USER_AGENT is required and must identify the application "
            "and a contact email."
        )
    return {
        "User-Agent": user_agent,
        "Host": "data.sec.gov",
    }


def _sec_archive_headers() -> dict[str, str]:
    headers = _sec_headers()
    headers.pop("Host", None)
    return headers


def _identity_candidates(question: InvestmentQuestion) -> tuple[set[str], str]:
    raw = " ".join(
        part
        for part in (
            question.original_question,
            question.security_or_asset or "",
        )
        if part
    )
    tickers = {
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9.-]{0,7}\b", raw)
        if token not in {"I", "US", "USD"}
    }
    subject = _normalise_text(question.subject or "")
    return tickers, subject


def _resolve_sec_identity(
    question: InvestmentQuestion,
    *,
    budget: _RequestBudget,
) -> tuple[str, str, str]:
    payload = _request_json(
        SEC_TICKERS_URL,
        budget=budget,
        headers=_sec_archive_headers(),
        max_bytes=2_500_000,
    )
    requested_tickers, subject = _identity_candidates(question)
    subject_lower = subject.lower()
    ranked: list[tuple[int, str, str, str]] = []

    for raw_entry in payload.values():
        if not isinstance(raw_entry, dict):
            continue
        ticker = str(raw_entry.get("ticker", "")).strip().upper()
        title = _normalise_text(str(raw_entry.get("title", "")))
        raw_cik = raw_entry.get("cik_str")
        if not ticker or not title or raw_cik is None:
            continue
        score = 0
        if ticker in requested_tickers:
            score = 100
        title_lower = title.lower()
        if subject_lower:
            if subject_lower == title_lower:
                score = max(score, 95)
            elif subject_lower in title_lower:
                score = max(score, 85)
            elif title_lower in subject_lower:
                score = max(score, 75)
            else:
                subject_tokens = {
                    token
                    for token in re.findall(r"[a-z0-9]+", subject_lower)
                    if len(token) >= 5
                }
                title_tokens = set(re.findall(r"[a-z0-9]+", title_lower))
                overlap = len(subject_tokens & title_tokens)
                if overlap:
                    score = max(score, 40 + 5 * overlap)
        if score:
            ranked.append(
                (score, ticker, title, str(raw_cik).zfill(10))
            )

    if not ranked:
        raise ExternalResearchError(
            "The company could not be matched unambiguously to an SEC filer."
        )
    ranked.sort(reverse=True)
    best_score = ranked[0][0]
    best_ciks = {entry[3] for entry in ranked if entry[0] == best_score}
    if len(best_ciks) > 1:
        raise ExternalResearchError(
            "The company name matched multiple SEC filers; include a ticker "
            "in the investment question."
        )
    _, ticker, title, cik = ranked[0]
    return title, ticker, cik


def _fact_values(
    company_facts: dict[str, Any],
    concepts: tuple[str, ...],
    *,
    duration: bool,
) -> dict[int, tuple[float, date, str]]:
    """Return the strongest annual USD fact series among candidate concepts."""

    us_gaap = (
        company_facts.get("facts", {})
        .get("us-gaap", {})
    )
    best: dict[int, tuple[float, date, str]] = {}
    for concept in concepts:
        fact = us_gaap.get(concept)
        if not isinstance(fact, dict):
            continue
        units = fact.get("units", {})
        values = units.get("USD", [])
        if not isinstance(values, list):
            continue
        candidate: dict[int, tuple[float, date, str, str]] = {}
        for item in values:
            if not isinstance(item, dict):
                continue
            if item.get("form") != "10-K" or item.get("fp") != "FY":
                continue
            if duration and not item.get("start"):
                continue
            raw_value = item.get("val")
            raw_year = item.get("fy")
            raw_end = item.get("end")
            if not isinstance(raw_value, (int, float)) or not raw_year or not raw_end:
                continue
            try:
                period_end = date.fromisoformat(str(raw_end))
                fiscal_year = int(raw_year)
            except (TypeError, ValueError):
                continue
            filed = str(item.get("filed", ""))
            accession = str(item.get("accn", ""))
            existing = candidate.get(fiscal_year)
            if existing is None or (filed, accession) > (
                existing[3],
                existing[2],
            ):
                candidate[fiscal_year] = (
                    float(raw_value) / 1_000_000,
                    period_end,
                    accession,
                    filed,
                )
        reduced = {
            year: (value, end, accession)
            for year, (value, end, accession, _) in candidate.items()
        }
        if len(reduced) > len(best):
            best = reduced
    return best


def _debt_values(
    company_facts: dict[str, Any],
) -> dict[int, tuple[float, date, str]]:
    total = _fact_values(
        company_facts,
        (
            "LongTermDebtAndFinanceLeaseObligations",
            "LongTermDebtAndCapitalLeaseObligations",
            "LongTermDebt",
        ),
        duration=False,
    )
    if total:
        return total

    components = [
        _fact_values(
            company_facts,
            ("ShortTermBorrowings",),
            duration=False,
        ),
        _fact_values(
            company_facts,
            (
                "LongTermDebtCurrent",
                "LongTermDebtAndFinanceLeaseObligationsCurrent",
            ),
            duration=False,
        ),
        _fact_values(
            company_facts,
            (
                "LongTermDebtNoncurrent",
                "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
            ),
            duration=False,
        ),
    ]
    combined: dict[int, tuple[float, date, str]] = {}
    years = set().union(*(set(component) for component in components))
    for year in years:
        present = [component[year] for component in components if year in component]
        if not present:
            continue
        combined[year] = (
            sum(item[0] for item in present),
            max(item[1] for item in present),
            present[0][2],
        )
    return combined


def _latest_shares_outstanding(
    company_facts: dict[str, Any],
    *,
    as_of_date: date,
) -> tuple[float, date] | None:
    """Return the latest SEC-reported common shares outstanding."""

    candidates: list[tuple[date, str, float]] = []
    facts = company_facts.get("facts", {})
    if not isinstance(facts, dict):
        return None
    for taxonomy, concept in (
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ):
        taxonomy_facts = facts.get(taxonomy, {})
        if not isinstance(taxonomy_facts, dict):
            continue
        fact = taxonomy_facts.get(concept, {})
        if not isinstance(fact, dict):
            continue
        units = fact.get("units", {})
        values = units.get("shares", []) if isinstance(units, dict) else []
        for item in values:
            if not isinstance(item, dict):
                continue
            if item.get("form") not in {"10-K", "10-Q"}:
                continue
            raw_value = item.get("val")
            raw_end = item.get("end")
            if not isinstance(raw_value, (int, float)) or not raw_end:
                continue
            try:
                observation_date = date.fromisoformat(str(raw_end))
            except ValueError:
                continue
            if observation_date > as_of_date:
                continue
            candidates.append(
                (
                    observation_date,
                    str(item.get("filed", "")),
                    float(raw_value),
                )
            )
    if not candidates:
        return None
    observation_date, _, value = max(candidates)
    return value, observation_date


def _financial_periods(
    company_facts: dict[str, Any],
    *,
    as_of_date: date,
) -> list[FinancialPeriodData]:
    revenue = _fact_values(
        company_facts,
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "Revenues",
        ),
        duration=True,
    )
    reported_ebitda = _fact_values(
        company_facts,
        ("EarningsBeforeInterestTaxesDepreciationAndAmortization",),
        duration=True,
    )
    operating_income = _fact_values(
        company_facts,
        ("OperatingIncomeLoss",),
        duration=True,
    )
    depreciation = _fact_values(
        company_facts,
        (
            "DepreciationDepletionAndAmortization",
            "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
            "Depreciation",
        ),
        duration=True,
    )
    operating_cash_flow = _fact_values(
        company_facts,
        ("NetCashProvidedByUsedInOperatingActivities",),
        duration=True,
    )
    capex = _fact_values(
        company_facts,
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsForAdditionsToPropertyPlantAndEquipment",
        ),
        duration=True,
    )
    debt = _debt_values(company_facts)
    cash = _fact_values(
        company_facts,
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        duration=False,
    )

    years = sorted(
        {
            year
            for series in (
                revenue,
                operating_income,
                operating_cash_flow,
                debt,
            )
            for year, (_, period_end, _) in series.items()
            if period_end <= as_of_date
        },
        reverse=True,
    )[:4]
    periods: list[FinancialPeriodData] = []
    for year in sorted(years):
        period_end = max(
            (
                series[year][1]
                for series in (
                    revenue,
                    operating_income,
                    operating_cash_flow,
                    debt,
                    cash,
                )
                if year in series
            ),
            default=date(year, 12, 31),
        )
        revenue_value = revenue.get(year, (None, period_end, ""))[0]
        if year in reported_ebitda:
            ebitda = reported_ebitda[year][0]
            ebitda_basis = "reported"
        elif year in operating_income and year in depreciation:
            ebitda = operating_income[year][0] + depreciation[year][0]
            ebitda_basis = "operating_income_plus_da"
        elif year in operating_income:
            ebitda = operating_income[year][0]
            ebitda_basis = "operating_income_fallback"
        else:
            ebitda = None
            ebitda_basis = "unavailable"

        ocf = operating_cash_flow.get(year, (None, period_end, ""))[0]
        capex_value = capex.get(year, (None, period_end, ""))[0]
        fcf = (
            ocf - capex_value
            if ocf is not None and capex_value is not None
            else None
        )
        debt_value = debt.get(year, (None, period_end, ""))[0]
        cash_value = cash.get(year, (None, period_end, ""))[0]
        net_debt = (
            debt_value - cash_value
            if debt_value is not None and cash_value is not None
            else None
        )
        periods.append(
            FinancialPeriodData(
                fiscal_year=year,
                period_end=period_end,
                revenue=revenue_value,
                ebitda=ebitda,
                ebitda_basis=ebitda_basis,
                operating_cash_flow=ocf,
                capital_expenditure=capex_value,
                free_cash_flow=fcf,
                total_debt=debt_value,
                cash_and_equivalents=cash_value,
                net_debt=net_debt,
            )
        )
    return periods


def _decode_document(payload: bytes, url: str) -> str:
    if payload.startswith(b"%PDF") or urlparse(url).path.lower().endswith(".pdf"):
        try:
            import fitz

            with fitz.open(stream=payload, filetype="pdf") as document:
                text = " ".join(
                    document.load_page(page_number).get_text("text")
                    for page_number in range(min(document.page_count, 80))
                )
            return _normalise_text(text)
        except (ImportError, RuntimeError, ValueError) as error:
            raise ExternalResearchError(
                f"PDF extraction failed for {url}: {error}"
            ) from error
    try:
        decoded = payload.decode("utf-8", errors="replace")
    except UnicodeDecodeError as error:
        raise ExternalResearchError(
            f"Document text decoding failed for {url}."
        ) from error
    parser = _TextExtractor()
    parser.feed(decoded)
    return parser.text()


def _section(text: str, start_item: str, end_item: str) -> str:
    pattern = re.compile(
        rf"\bitem\s+{re.escape(start_item)}[.\s:-]+(.*?)"
        rf"(?=\bitem\s+{re.escape(end_item)}[.\s:-]+)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = [
        _normalise_text(match.group(1))
        for match in pattern.finditer(text)
    ]
    substantial = [match for match in matches if len(match) >= 250]
    return max(substantial, key=len, default="")[:20_000]


def _sentences(text: str) -> list[str]:
    return [
        _normalise_text(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if 40 <= len(_normalise_text(sentence)) <= 500
    ]


def _first_material_sentence(
    text: str,
    *,
    keywords: tuple[str, ...] = (),
) -> str | None:
    candidates = _sentences(text)
    if keywords:
        keyword_candidates = [
            sentence
            for sentence in candidates
            if any(keyword in sentence.lower() for keyword in keywords)
        ]
        if keyword_candidates:
            return keyword_candidates[0]
    return candidates[0] if candidates else None


def _latest_filings(
    submissions: dict[str, Any],
    maximum: int,
) -> list[dict[str, str]]:
    recent = submissions.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict):
        return []
    forms = recent.get("form", [])
    selected: list[dict[str, str]] = []
    seen_forms: set[str] = set()
    for index, form in enumerate(forms):
        if form not in {"10-K", "10-Q"} or form in seen_forms:
            continue
        try:
            filing = {
                "form": str(form),
                "accession": str(recent["accessionNumber"][index]),
                "primary_document": str(recent["primaryDocument"][index]),
                "filing_date": str(recent["filingDate"][index]),
            }
        except (IndexError, KeyError):
            continue
        selected.append(filing)
        seen_forms.add(form)
        if len(selected) >= maximum:
            break
    return selected


def _filing_url(cik: str, filing: dict[str, str]) -> str:
    accession = filing["accession"].replace("-", "")
    return (
        f"{SEC_ARCHIVES_URL}/{int(cik)}/{accession}/"
        f"{filing['primary_document']}"
    )


def _collect_sec(
    question: InvestmentQuestion,
    settings: ExternalResearchSettings,
    *,
    budget: _RequestBudget,
) -> dict[str, Any]:
    company_name, ticker, cik = _resolve_sec_identity(
        question,
        budget=budget,
    )
    headers = _sec_headers()
    company_facts = _request_json(
        f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json",
        budget=budget,
        headers=headers,
        max_bytes=12_000_000,
    )
    submissions = _request_json(
        f"{SEC_BASE_URL}/submissions/CIK{cik}.json",
        budget=budget,
        headers=headers,
        max_bytes=5_000_000,
    )
    company_name = _normalise_text(
        str(company_facts.get("entityName") or submissions.get("name") or company_name)
    )
    periods = _financial_periods(
        company_facts,
        as_of_date=question.as_of_date,
    )
    shares_outstanding = _latest_shares_outstanding(
        company_facts,
        as_of_date=question.as_of_date,
    )
    sources = [
        ResearchSource(
            source_id=f"sec_companyfacts_{cik}",
            title=f"{company_name} SEC Company Facts",
            publisher="U.S. Securities and Exchange Commission",
            url=f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json",
            source_type="SEC XBRL company facts",
        )
    ]
    gaps: list[ResearchGap] = []
    observations: list[EvidenceClaim] = []
    description: str | None = None
    filings_fetched = 0

    filings = _latest_filings(submissions, settings.max_filings)
    for filing in filings:
        url = _filing_url(cik, filing)
        source_id = "sec_" + filing["accession"].replace("-", "")
        try:
            payload = _request_bytes(
                url,
                budget=budget,
                headers=_sec_archive_headers(),
                max_bytes=MAX_SEC_DOCUMENT_BYTES,
            )
            text = _decode_document(payload, url)
            filing_date = date.fromisoformat(filing["filing_date"])
        except (ExternalResearchError, ValueError) as error:
            gaps.append(
                _gap(
                    f"{filing['form']} document",
                    "fetch_failed",
                    str(error),
                )
            )
            continue

        filings_fetched += 1
        sources.append(
            ResearchSource(
                source_id=source_id,
                title=f"{company_name} {filing['form']}",
                publisher=company_name,
                url=url,
                published_at=filing_date,
                source_type="SEC filing",
            )
        )
        if filing["form"] == "10-K":
            business = _section(text, "1", "1A")
            risks = _section(text, "1A", "1B") or _section(text, "1A", "2")
            management = _section(text, "7", "7A") or _section(text, "7", "8")
            if business and not description:
                description = _word_limit(business, MAX_DESCRIPTION_WORDS)
            candidates = (
                (
                    "business_development",
                    _first_material_sentence(
                        business,
                        keywords=(
                            "acquisition",
                            "expanded",
                            "growth",
                            "operates",
                            "business",
                        ),
                    ),
                ),
                (
                    "management_concern",
                    _first_material_sentence(
                        management,
                        keywords=(
                            "risk",
                            "challenge",
                            "uncertain",
                            "decline",
                            "cost",
                            "demand",
                        ),
                    ),
                ),
                (
                    "disclosed_risk",
                    _first_material_sentence(risks),
                ),
            )
            for label, statement in candidates:
                if not statement or len(observations) >= 3:
                    continue
                observations.append(
                    EvidenceClaim(
                        claim_id=f"micro_ext_filing_{label}",
                        statement=statement,
                        status="reported_fact",
                        source_ids=[source_id],
                        confidence=0.9,
                        as_of_date=filing_date,
                    )
                )

    if not periods:
        gaps.append(
            _gap(
                "financial_periods",
                "not_reported",
                "No complete annual USD XBRL series could be normalized.",
            )
        )
    if filings_fetched < len(filings):
        gaps.append(
            _gap(
                "filings",
                "fetch_failed",
                f"Fetched {filings_fetched} of {len(filings)} selected filings.",
            )
        )
    return {
        "company_name": company_name,
        "ticker": ticker,
        "cik": cik,
        "periods": periods,
        "shares_outstanding": (
            shares_outstanding[0] if shares_outstanding else None
        ),
        "shares_outstanding_date": (
            shares_outstanding[1] if shares_outstanding else None
        ),
        "sources": sources,
        "gaps": gaps,
        "observations": observations,
        "description": description,
        "filings_fetched": filings_fetched,
    }
