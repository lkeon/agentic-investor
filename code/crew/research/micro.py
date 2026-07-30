"""Company-level source orchestration and MicroView hydration."""

from __future__ import annotations

from crew.research.alpaca import _market_data
from crew.research.common import (
    MAX_DESCRIPTION_WORDS,
    MAX_MICRO_EXTERNAL_REQUESTS,
    ExternalResearchConfigurationError,
    ExternalResearchError,
    ResearchProgress,
    _RequestBudget,
    _gap,
    _report_progress,
    _word_limit,
)
from crew.research.exa import _collect_ir
from crew.research.sec import _collect_sec, _identity_candidates
from crew.schemas import (
    EvidenceClaim,
    ExternalResearchSettings,
    FinancialPeriodData,
    InvestmentQuestion,
    MicroResearchData,
    MicroView,
    ResearchGap,
    ResearchSource,
)


def _source_names(sources: list[ResearchSource]) -> str:
    """Compact source titles into one provider-level progress statement."""

    return "; ".join(source.title for source in sources) or "none accepted"


def _retrieved_summary(data: MicroResearchData) -> str:
    """Describe accepted evidence once for the customer-facing milestone."""

    source_types = {source.source_type for source in data.sources}
    retrieved: list[str] = []
    if "SEC XBRL company facts" in source_types:
        retrieved.append("SEC Company Facts")
    if data.filings_fetched:
        retrieved.append(f"{data.filings_fetched} SEC filing(s)")
    if data.share_price is not None:
        retrieved.append("an Alpaca IEX market price")
    if data.ir_documents_fetched:
        retrieved.append(
            f"{data.ir_documents_fetched} official IR document(s)"
        )
    return (
        "Retrieved " + "; ".join(retrieved) + "."
        if retrieved
        else "No external source was accepted for this company."
    )


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _period_calculations(
    periods: list[FinancialPeriodData],
) -> dict[str, float | None]:
    if not periods:
        return {
            "revenue_cagr": None,
            "latest_ebitda_margin": None,
            "latest_fcf_margin": None,
            "latest_net_debt_to_ebitda": None,
        }
    latest = periods[-1]
    first = periods[0]
    years = latest.fiscal_year - first.fiscal_year
    revenue_cagr = None
    if (
        years > 0
        and first.revenue is not None
        and latest.revenue is not None
        and first.revenue > 0
        and latest.revenue > 0
    ):
        revenue_cagr = (
            (latest.revenue / first.revenue) ** (1 / years) - 1
        ) * 100
    ebitda_margin = _safe_ratio(latest.ebitda, latest.revenue)
    fcf_margin = _safe_ratio(latest.free_cash_flow, latest.revenue)
    return {
        "revenue_cagr": revenue_cagr,
        "latest_ebitda_margin": (
            ebitda_margin * 100 if ebitda_margin is not None else None
        ),
        "latest_fcf_margin": fcf_margin * 100 if fcf_margin is not None else None,
        "latest_net_debt_to_ebitda": _safe_ratio(
            latest.net_debt,
            latest.ebitda,
        ),
    }


def collect_micro_research(
    question: InvestmentQuestion,
    settings: ExternalResearchSettings,
    *,
    progress: ResearchProgress | None = None,
) -> MicroResearchData:
    """Collect a compact company dataset within independent source budgets."""

    if not settings.enabled:
        raise ValueError("External research is not enabled.")

    budget = _RequestBudget(MAX_MICRO_EXTERNAL_REQUESTS)
    sources: list[ResearchSource] = []
    gaps: list[ResearchGap] = []
    observations: list[EvidenceClaim] = []
    periods: list[FinancialPeriodData] = []
    company_name = question.subject or question.security_or_asset or "Unknown company"
    requested_tickers, _ = _identity_candidates(question)
    ticker: str | None = (
        next(iter(requested_tickers))
        if len(requested_tickers) == 1
        else None
    )
    cik: str | None = None
    description: str | None = None
    filings_fetched = 0
    shares_outstanding = None
    shares_outstanding_date = None

    if settings.sec_filings_enabled:
        try:
            sec = _collect_sec(question, settings, budget=budget)
            company_name = sec["company_name"]
            ticker = sec["ticker"]
            cik = sec["cik"]
            periods = sec["periods"]
            shares_outstanding = sec["shares_outstanding"]
            shares_outstanding_date = sec["shares_outstanding_date"]
            sources.extend(sec["sources"])
            gaps.extend(sec["gaps"])
            observations.extend(sec["observations"])
            description = sec["description"]
            filings_fetched = sec["filings_fetched"]
            _report_progress(
                progress,
                "SEC filings",
                "complete",
                (
                    f"Resolved {company_name} ({ticker or 'ticker unavailable'}); "
                    f"retrieved {filings_fetched} filing document(s) and "
                    f"normalised {len(periods)} annual financial period(s). "
                    f"Sources: {_source_names(sec['sources'])}."
                ),
            )
        except ExternalResearchConfigurationError as error:
            gaps.append(_gap("SEC", "configuration_missing", str(error)))
            _report_progress(
                progress,
                "SEC filings",
                "unavailable",
                f"Configuration missing: {error}",
            )
        except ExternalResearchError as error:
            gaps.append(_gap("SEC identity or filings", "fetch_failed", str(error)))
            _report_progress(
                progress,
                "SEC filings",
                "unavailable",
                f"Retrieval failed within the bounded request budget: {error}",
            )
    else:
        gaps.append(
            _gap("SEC filings", "source_disabled", "SEC research is disabled.")
        )
        _report_progress(
            progress,
            "SEC filings",
            "skipped",
            "This source was disabled in Diligence Room Settings.",
        )

    share_price = None
    share_price_date = None
    if settings.market_data_enabled:
        if ticker:
            try:
                market = _market_data(ticker, budget=budget)
                share_price = market["share_price"]
                share_price_date = market["share_price_date"]
                sources.append(market["source"])
                _report_progress(
                    progress,
                    "Alpaca market data",
                    "complete",
                    (
                        f"Retrieved a USD {share_price:,.2f} reference price"
                        + (
                            f" dated {share_price_date.isoformat()}."
                            if share_price_date
                            else "; the provider date was unavailable."
                        )
                        + f" Source: {market['source'].title}."
                    ),
                )
            except ExternalResearchConfigurationError as error:
                gaps.append(
                    _gap("market_data", "configuration_missing", str(error))
                )
                _report_progress(
                    progress,
                    "Alpaca market data",
                    "unavailable",
                    f"Configuration missing: {error}",
                )
            except ExternalResearchError as error:
                gaps.append(_gap("market_data", "fetch_failed", str(error)))
                _report_progress(
                    progress,
                    "Alpaca market data",
                    "unavailable",
                    f"Snapshot retrieval failed: {error}",
                )
        else:
            gaps.append(
                _gap(
                    "market_data",
                    "ambiguous_identity",
                    "A ticker was not resolved, so market data was not queried.",
                )
            )
            _report_progress(
                progress,
                "Alpaca market data",
                "unavailable",
                "A unique ticker was not resolved, so no quote was requested.",
            )
    else:
        gaps.append(
            _gap(
                "market_data",
                "source_disabled",
                "Market reference data is disabled.",
            )
        )
        _report_progress(
            progress,
            "Alpaca market data",
            "skipped",
            "This source was disabled in Diligence Room Settings.",
        )

    ir_documents = 0
    discovery_searches = 0
    credit_rating = None
    if settings.investor_relations_enabled:
        try:
            ir = _collect_ir(
                company_name,
                ticker=ticker,
                settings=settings,
                budget=budget,
            )
            sources.extend(ir["sources"])
            gaps.extend(ir["gaps"])
            observations.extend(ir["observations"])
            description = description or ir["description"]
            credit_rating = ir["rating"]
            discovery_searches = ir["searches"]
            ir_documents = ir["documents"]
            _report_progress(
                progress,
                "Exa investor relations",
                "complete" if ir_documents else "unavailable",
                (
                    f"Used {discovery_searches} structured search(es), "
                    f"accepted {ir_documents} official document(s), and "
                    f"retained {len(ir['observations'])} grounded qualitative "
                    f"observation(s). Sources: {_source_names(ir['sources'])}."
                ),
            )
        except ExternalResearchConfigurationError as error:
            gaps.append(
                _gap(
                    "investor_relations",
                    "configuration_missing",
                    str(error),
                )
            )
            _report_progress(
                progress,
                "Exa investor relations",
                "unavailable",
                f"Configuration missing: {error}",
            )
        except ExternalResearchError as error:
            gaps.append(
                _gap("investor_relations", "fetch_failed", str(error))
            )
            _report_progress(
                progress,
                "Exa investor relations",
                "unavailable",
                f"Structured research failed: {error}",
            )
    else:
        gaps.append(
            _gap(
                "investor_relations",
                "source_disabled",
                "Official investor-relations discovery is disabled.",
            )
        )
        _report_progress(
            progress,
            "Exa investor relations",
            "skipped",
            "This source was disabled in Diligence Room Settings.",
        )

    calculations = _period_calculations(periods)
    latest = periods[-1] if periods else None
    market_cap = (
        share_price * shares_outstanding / 1_000_000
        if share_price is not None and shares_outstanding is not None
        else None
    )
    if share_price is not None and market_cap is None:
        gaps.append(
            _gap(
                "market_cap",
                "calculation_unavailable",
                "An Alpaca IEX reference price was available but SEC-reported "
                "shares outstanding were not.",
            )
        )
    enterprise_value = (
        market_cap + latest.net_debt
        if market_cap is not None
        and latest is not None
        and latest.net_debt is not None
        else None
    )
    ev_to_ebitda = (
        _safe_ratio(enterprise_value, latest.ebitda)
        if latest is not None
        else None
    )
    fcf_to_market_cap = _safe_ratio(
        latest.free_cash_flow if latest else None,
        market_cap,
    )
    fcf_yield = (
        fcf_to_market_cap * 100 if fcf_to_market_cap is not None else None
    )
    if market_cap is not None and enterprise_value is None:
        gaps.append(
            _gap(
                "enterprise_value",
                "calculation_unavailable",
                "Market capitalisation was available but net debt was not.",
            )
        )
    if description is None:
        gaps.append(
            _gap(
                "company_description",
                "not_reported",
                "No bounded official source produced a reliable description.",
            )
        )

    unique_sources = {
        source.source_id: source
        for source in sources
    }
    unique_observations = {
        claim.claim_id: claim
        for claim in observations
    }
    result = MicroResearchData(
        as_of_date=question.as_of_date,
        company_name=company_name,
        ticker=ticker,
        cik=cik,
        company_description=(
            _word_limit(description, MAX_DESCRIPTION_WORDS)
            if description
            else None
        ),
        financial_periods=periods,
        **calculations,
        share_price=share_price,
        share_price_date=share_price_date,
        shares_outstanding=shares_outstanding,
        shares_outstanding_date=shares_outstanding_date,
        market_cap=market_cap,
        enterprise_value=enterprise_value,
        ev_to_ebitda=ev_to_ebitda,
        fcf_yield=fcf_yield,
        credit_rating=credit_rating,
        qualitative_observations=list(unique_observations.values())[:3],
        sources=list(unique_sources.values()),
        gaps=gaps[:24],
        filings_fetched=filings_fetched,
        ir_documents_fetched=ir_documents,
        discovery_searches_used=discovery_searches,
        external_requests_used=budget.used,
    )
    _report_progress(
        progress,
        "Company evidence",
        "complete",
        _retrieved_summary(result),
    )
    return result


def _format_metric(value: float | None, suffix: str = "") -> str:
    return "unavailable" if value is None else f"{value:,.2f}{suffix}"


def _authoritative_micro_claims(
    data: MicroResearchData,
) -> dict[str, list[EvidenceClaim]]:
    claims: dict[str, list[EvidenceClaim]] = {
        "business": [],
        "business_quality": [],
        "management_and_capital_allocation": [],
        "financial_performance": [],
        "balance_sheet_and_liquidity": [],
        "valuation": [],
        "risks_and_catalysts": [],
    }
    facts_source = next(
        (
            source.source_id
            for source in data.sources
            if source.source_type == "SEC XBRL company facts"
        ),
        None,
    )
    market_source = next(
        (
            source.source_id
            for source in data.sources
            if source.source_type == "IEX market data via Alpaca"
        ),
        None,
    )
    if data.company_description:
        description_source = next(
            (
                source.source_id
                for source in data.sources
                if source.source_type in {
                    "SEC filing",
                    "official investor relations",
                }
            ),
            facts_source,
        )
        claims["business"].append(
            EvidenceClaim(
                claim_id="micro_ext_company_description",
                statement=data.company_description,
                status="reported_fact",
                source_ids=[description_source] if description_source else [],
                confidence=0.9 if description_source else 0.6,
                as_of_date=data.as_of_date,
            )
        )

    for period in data.financial_periods:
        financial_statement = (
            f"FY{period.fiscal_year}: revenue "
            f"{_format_metric(period.revenue)} USD millions; EBITDA or fallback "
            f"{_format_metric(period.ebitda)} USD millions "
            f"(basis: {period.ebitda_basis}); free cash flow "
            f"{_format_metric(period.free_cash_flow)} USD millions."
        )
        balance_statement = (
            f"FY{period.fiscal_year}: total debt "
            f"{_format_metric(period.total_debt)} USD millions; cash and cash "
            f"equivalents {_format_metric(period.cash_and_equivalents)} USD "
            f"millions; net debt {_format_metric(period.net_debt)} USD millions."
        )
        source_ids = [facts_source] if facts_source else []
        claims["financial_performance"].append(
            EvidenceClaim(
                claim_id=f"micro_ext_financials_fy{period.fiscal_year}",
                statement=financial_statement,
                status="derived_metric",
                source_ids=source_ids,
                confidence=0.95,
                period=f"FY{period.fiscal_year}",
                as_of_date=period.period_end,
            )
        )
        claims["balance_sheet_and_liquidity"].append(
            EvidenceClaim(
                claim_id=f"micro_ext_balance_fy{period.fiscal_year}",
                statement=balance_statement,
                status="derived_metric",
                source_ids=source_ids,
                confidence=0.95,
                period=f"FY{period.fiscal_year}",
                as_of_date=period.period_end,
            )
        )

    ratio_values = (
        (
            "revenue_cagr",
            data.revenue_cagr,
            "Revenue CAGR over the available completed fiscal years",
            "percent",
        ),
        (
            "ebitda_margin",
            data.latest_ebitda_margin,
            "Latest EBITDA or labelled fallback margin",
            "percent",
        ),
        (
            "fcf_margin",
            data.latest_fcf_margin,
            "Latest free-cash-flow margin",
            "percent",
        ),
        (
            "net_debt_to_ebitda",
            data.latest_net_debt_to_ebitda,
            "Latest net debt to EBITDA or labelled fallback",
            "times",
        ),
    )
    for code, value, label, unit in ratio_values:
        if value is None:
            continue
        claims["financial_performance"].append(
            EvidenceClaim(
                claim_id=f"micro_ext_{code}",
                statement=f"{label} is {value:,.2f} {unit}.",
                value=round(value, 4),
                unit=unit,
                status="derived_metric",
                source_ids=[facts_source] if facts_source else [],
                confidence=0.95,
                as_of_date=data.as_of_date,
            )
        )

    valuation_values = (
        (
            "share_price",
            data.share_price,
            "Alpaca IEX reference share price",
            "USD per share",
        ),
        ("market_cap", data.market_cap, "Market capitalisation", "USD millions"),
        (
            "enterprise_value",
            data.enterprise_value,
            "Enterprise value",
            "USD millions",
        ),
        ("ev_to_ebitda", data.ev_to_ebitda, "EV to EBITDA or fallback", "times"),
        ("fcf_yield", data.fcf_yield, "Free-cash-flow yield", "percent"),
    )
    for code, value, label, unit in valuation_values:
        if value is None:
            continue
        source_ids = (
            [market_source]
            if code == "share_price" and market_source
            else [
                source_id
                for source_id in (market_source, facts_source)
                if source_id
            ]
        )
        claims["valuation"].append(
            EvidenceClaim(
                claim_id=f"micro_ext_{code}",
                statement=f"{label} is {value:,.2f} {unit}.",
                value=round(value, 4),
                unit=unit,
                status=(
                    "reported_fact"
                    if code == "share_price"
                    else "derived_metric"
                ),
                source_ids=source_ids,
                confidence=0.95,
                as_of_date=data.share_price_date or data.as_of_date,
            )
        )

    if data.credit_rating:
        rating_sources = [
            source.source_id
            for source in data.sources
            if source.source_type == "official investor relations"
        ][:1]
        claims["balance_sheet_and_liquidity"].append(
            EvidenceClaim(
                claim_id="micro_ext_credit_rating",
                statement=(
                    f"An official company document reports a credit rating of "
                    f"{data.credit_rating}."
                ),
                value=data.credit_rating,
                status="reported_fact",
                source_ids=rating_sources,
                confidence=0.8,
                as_of_date=data.as_of_date,
            )
        )
    claims["risks_and_catalysts"].extend(data.qualitative_observations)
    return claims


def merge_micro_research(
    view: MicroView,
    data: MicroResearchData,
) -> MicroView:
    """Replace model paraphrases of external facts with collector-owned claims."""

    authoritative = _authoritative_micro_claims(data)
    authoritative_ids = {
        claim.claim_id
        for section in authoritative.values()
        for claim in section
    }
    external_source_ids = {source.source_id for source in data.sources}
    updates: dict[str, Any] = {
        "company_name": data.company_name,
        "ticker": data.ticker or view.ticker,
        "as_of_date": data.as_of_date,
    }
    for field_name, added in authoritative.items():
        retained = [
            claim
            for claim in getattr(view, field_name)
            if claim.claim_id not in authoritative_ids
            and not (set(claim.source_ids) & external_source_ids)
        ]
        maximum = {
            "business": 12,
            "business_quality": 12,
            "management_and_capital_allocation": 12,
            "financial_performance": 14,
            "balance_sheet_and_liquidity": 14,
            "valuation": 12,
            "risks_and_catalysts": 14,
        }[field_name]
        updates[field_name] = (retained + added)[:maximum]

    merged_sources = {
        source.source_id: source
        for source in view.sources
        if source.source_id not in external_source_ids
    }
    merged_sources.update(
        {source.source_id: source for source in data.sources}
    )
    updates["sources"] = list(merged_sources.values())[:30]
    gap_messages = [
        f"{gap.field}: {gap.detail}"
        for gap in data.gaps
        if gap.status
        not in {"source_disabled", "not_applicable"}
    ]
    updates["unresolved_questions"] = list(
        dict.fromkeys([*view.unresolved_questions, *gap_messages])
    )[:15]
    return MicroView.model_validate(
        view.model_copy(update=updates).model_dump(mode="json")
    )
