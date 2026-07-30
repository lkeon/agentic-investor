"""FRED/Yale macro collection, deterministic MacroView, and daily storage."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from threading import Lock
from typing import Any
from urllib.parse import urlencode

from crew.research.common import (
    PROJECT_ROOT,
    ExternalResearchConfigurationError,
    ExternalResearchError,
    ResearchProgress,
    _RequestBudget,
    _gap,
    _report_progress,
    _request_bytes,
    _request_json,
)
from crew.schemas import (
    EvidenceClaim,
    ExternalResearchSettings,
    MacroIndicatorData,
    MacroResearchData,
    MacroView,
    ResearchGap,
    ResearchSource,
)


DEFAULT_MACRO_STORE = (
    PROJECT_ROOT / "data" / "processed" / "crew" / "macro_views"
)
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
SHILLER_DATA_URL = "https://www.econ.yale.edu/~shiller/data/ie_data.xls"
MAX_MACRO_EXTERNAL_REQUESTS = 5
_DAILY_MACRO_LOCK = Lock()


def _fred_observations(
    series_id: str,
    *,
    start: date,
    budget: _RequestBudget,
) -> list[tuple[date, float]]:
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise ExternalResearchConfigurationError(
            "FRED_API_KEY is required for rates, credit, and the Buffett proxy."
        )
    query = urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "sort_order": "asc",
            "limit": 2000,
        }
    )
    payload = _request_json(
        f"{FRED_API_URL}?{query}",
        budget=budget,
        max_bytes=1_500_000,
    )
    observations: list[tuple[date, float]] = []
    for item in payload.get("observations", []):
        if not isinstance(item, dict) or item.get("value") in {None, "."}:
            continue
        try:
            observations.append(
                (
                    date.fromisoformat(str(item["date"])),
                    float(item["value"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not observations:
        raise ExternalResearchError(
            f"FRED returned no usable observations for {series_id}."
        )
    return observations


def _at_or_before(
    observations: list[tuple[date, float]],
    target: date,
) -> tuple[date, float] | None:
    eligible = [item for item in observations if item[0] <= target]
    return eligible[-1] if eligible else None


def _percentile(values: list[float], current: float) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return 100 * sum(value <= current for value in clean) / len(clean)


def _freshness(
    observation_date: date | None,
    applicable_date: date,
    *,
    expected_lag_days: int,
) -> str:
    if observation_date is None:
        return "unavailable"
    lag = (applicable_date - observation_date).days
    if lag <= expected_lag_days:
        return "current"
    if lag <= expected_lag_days * 2:
        return "delayed"
    return "stale"


def _series_indicator(
    *,
    indicator_id: str,
    label: str,
    observations: list[tuple[date, float]],
    applicable_date: date,
    unit: str,
    source_ids: list[str],
    methodology: str,
) -> MacroIndicatorData:
    latest = _at_or_before(observations, applicable_date)
    if latest is None:
        raise ExternalResearchError(f"No current observation for {label}.")
    one_year = _at_or_before(
        observations,
        applicable_date - timedelta(days=365),
    )
    comparison_start = applicable_date - timedelta(days=3 * 365)
    comparison = [
        value
        for observation_date, value in observations
        if comparison_start <= observation_date <= applicable_date
    ]
    return MacroIndicatorData(
        indicator_id=indicator_id,
        label=label,
        current=latest[1],
        one_year_ago=one_year[1] if one_year else None,
        comparison_average=mean(comparison) if comparison else None,
        historical_percentile=_percentile(
            [value for _, value in observations],
            latest[1],
        ),
        unit=unit,
        observation_date=latest[0],
        freshness=_freshness(
            latest[0],
            applicable_date,
            expected_lag_days=10,
        ),
        source_ids=source_ids,
        methodology=methodology,
    )


def _buffett_indicator(
    equities: list[tuple[date, float]],
    gdp: list[tuple[date, float]],
    applicable_date: date,
) -> MacroIndicatorData:
    ratios: list[tuple[date, float]] = []
    for equity_date, equity_millions in equities:
        matching_gdp = _at_or_before(gdp, equity_date)
        if matching_gdp is None:
            continue
        gdp_billions = matching_gdp[1]
        if gdp_billions:
            ratios.append(
                (
                    equity_date,
                    100 * equity_millions / (gdp_billions * 1000),
                )
            )
    latest = _at_or_before(ratios, applicable_date)
    if latest is None:
        raise ExternalResearchError(
            "The Buffett Indicator proxy could not be aligned."
        )
    one_year = _at_or_before(ratios, applicable_date - timedelta(days=365))
    twenty_years = applicable_date - timedelta(days=20 * 365)
    comparison = [
        value
        for observation_date, value in ratios
        if observation_date >= twenty_years
    ]
    return MacroIndicatorData(
        indicator_id="buffett_indicator_proxy",
        label="Buffett Indicator proxy",
        current=latest[1],
        one_year_ago=one_year[1] if one_year else None,
        comparison_average=mean(comparison) if comparison else None,
        historical_percentile=_percentile(
            [value for _, value in ratios],
            latest[1],
        ),
        unit="percent of nominal GDP",
        observation_date=latest[0],
        freshness=_freshness(
            latest[0],
            applicable_date,
            expected_lag_days=150,
        ),
        source_ids=["fred_ncbeilq027s", "fred_gdp"],
        methodology=(
            "Quarterly US nonfinancial corporate equity market value divided "
            "by nominal GDP. This is a deliberately labelled proxy and excludes "
            "financial-sector equity."
        ),
    )


def _shiller_cape(
    *,
    applicable_date: date,
    budget: _RequestBudget,
) -> MacroIndicatorData:
    payload = _request_bytes(
        SHILLER_DATA_URL,
        budget=budget,
        max_bytes=3_000_000,
    )
    try:
        import pandas as pd

        raw = pd.read_excel(BytesIO(payload), sheet_name="Data", header=None)
    except Exception as error:
        raise ExternalResearchError(
            f"Yale Shiller CAPE workbook could not be parsed: {error}"
        ) from error

    header_row = None
    cape_column = None
    for row_index in range(min(25, len(raw))):
        for column_index, value in enumerate(raw.iloc[row_index].tolist()):
            if str(value).strip().upper() == "CAPE":
                header_row = row_index
                cape_column = column_index
                break
        if header_row is not None:
            break
    if header_row is None or cape_column is None:
        raise ExternalResearchError("The Yale workbook did not contain CAPE.")

    observations: list[tuple[date, float]] = []
    for row_index in range(header_row + 1, len(raw)):
        raw_date = raw.iloc[row_index, 0]
        raw_value = raw.iloc[row_index, cape_column]
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if isinstance(raw_date, (datetime, date)):
            observation_date = (
                raw_date.date()
                if isinstance(raw_date, datetime)
                else raw_date
            )
        else:
            try:
                numeric_date = float(raw_date)
                year = int(numeric_date)
                month = int(round((numeric_date - year) * 100))
                observation_date = date(year, min(12, max(1, month)), 1)
            except (TypeError, ValueError):
                continue
        if observation_date <= applicable_date:
            observations.append((observation_date, value))
    observations.sort()
    latest = _at_or_before(observations, applicable_date)
    if latest is None:
        raise ExternalResearchError("The Yale workbook had no usable CAPE.")
    one_year = _at_or_before(
        observations,
        applicable_date - timedelta(days=365),
    )
    values = [value for _, value in observations]
    return MacroIndicatorData(
        indicator_id="shiller_cape",
        label="Shiller cyclically adjusted price-to-earnings ratio",
        current=latest[1],
        one_year_ago=one_year[1] if one_year else None,
        comparison_average=median(values),
        historical_percentile=_percentile(values, latest[1]),
        unit="times",
        observation_date=latest[0],
        freshness=_freshness(
            latest[0],
            applicable_date,
            expected_lag_days=60,
        ),
        source_ids=["yale_shiller_cape"],
        methodology=(
            "Latest official monthly Yale CAPE compared with the median and "
            "percentile of the available long-run series."
        ),
    )


def collect_macro_research(
    settings: ExternalResearchSettings,
    *,
    applicable_date: date,
    progress: ResearchProgress | None = None,
) -> MacroResearchData:
    """Collect at most five requests for one company-independent US view."""

    budget = _RequestBudget(MAX_MACRO_EXTERNAL_REQUESTS)
    indicators: list[MacroIndicatorData] = []
    sources: list[ResearchSource] = []
    gaps: list[ResearchGap] = []

    if settings.rates_and_credit_enabled:
        try:
            start = applicable_date - timedelta(days=3 * 365 + 45)
            treasury = _fred_observations(
                "DGS5",
                start=start,
                budget=budget,
            )
            indicators.append(
                _series_indicator(
                    indicator_id="treasury_5y",
                    label="5-year US Treasury yield",
                    observations=treasury,
                    applicable_date=applicable_date,
                    unit="percent",
                    source_ids=["fred_dgs5"],
                    methodology=(
                        "Latest observation, observation at or before one year "
                        "earlier, and arithmetic mean over the trailing three years."
                    ),
                )
            )
            sources.append(
                ResearchSource(
                    source_id="fred_dgs5",
                    title="Market Yield on U.S. Treasury Securities at 5-Year Constant Maturity",
                    publisher="Federal Reserve Bank of St. Louis",
                    url="https://fred.stlouisfed.org/series/DGS5",
                    source_type="FRED rates",
                )
            )
            _report_progress(
                progress,
                "FRED Treasury",
                "complete",
                (
                    f"Prepared the 5-year Treasury indicator through "
                    f"{indicators[-1].observation_date or 'an unavailable date'}. "
                    f"Source: {sources[-1].title}."
                ),
            )
        except ExternalResearchConfigurationError as error:
            gaps.append(_gap("treasury_5y", "configuration_missing", str(error)))
            _report_progress(
                progress,
                "FRED Treasury",
                "unavailable",
                f"Configuration missing: {error}",
            )
        except ExternalResearchError as error:
            gaps.append(_gap("treasury_5y", "fetch_failed", str(error)))
            _report_progress(
                progress,
                "FRED Treasury",
                "unavailable",
                f"Series retrieval failed: {error}",
            )

        try:
            start = applicable_date - timedelta(days=3 * 365 + 45)
            credit = _fred_observations(
                "BAMLC0A0CM",
                start=start,
                budget=budget,
            )
            indicators.append(
                _series_indicator(
                    indicator_id="investment_grade_credit_spread",
                    label="Broad US investment-grade credit spread",
                    observations=credit,
                    applicable_date=applicable_date,
                    unit="percentage points",
                    source_ids=["fred_bamlc0a0cm"],
                    methodology=(
                        "ICE BofA US Corporate Master option-adjusted spread "
                        "used as a broad proxy; it is not a company-specific spread."
                    ),
                )
            )
            sources.append(
                ResearchSource(
                    source_id="fred_bamlc0a0cm",
                    title="ICE BofA US Corporate Master Option-Adjusted Spread",
                    publisher="Federal Reserve Bank of St. Louis",
                    url="https://fred.stlouisfed.org/series/BAMLC0A0CM",
                    source_type="FRED credit",
                )
            )
            _report_progress(
                progress,
                "FRED credit spreads",
                "complete",
                (
                    "Prepared the broad investment-grade spread as a market "
                    "backdrop, not a company-specific borrowing cost. "
                    f"Source: {sources[-1].title}."
                ),
            )
        except ExternalResearchConfigurationError as error:
            if not any(gap.field == "treasury_5y" for gap in gaps):
                gaps.append(_gap("credit_spread", "configuration_missing", str(error)))
            _report_progress(
                progress,
                "FRED credit spreads",
                "unavailable",
                f"Configuration missing: {error}",
            )
        except ExternalResearchError as error:
            gaps.append(_gap("credit_spread", "fetch_failed", str(error)))
            _report_progress(
                progress,
                "FRED credit spreads",
                "unavailable",
                f"Series retrieval failed: {error}",
            )
    else:
        gaps.append(
            _gap(
                "rates_and_credit",
                "source_disabled",
                "Rates and credit collection is disabled.",
            )
        )
        _report_progress(
            progress,
            "FRED rates and credit",
            "skipped",
            "These sources were disabled in Diligence Room Settings.",
        )

    if settings.aggregate_valuation_enabled:
        try:
            start = applicable_date - timedelta(days=30 * 365)
            equities = _fred_observations(
                "NCBEILQ027S",
                start=start,
                budget=budget,
            )
            gdp = _fred_observations(
                "GDP",
                start=start,
                budget=budget,
            )
            indicators.append(
                _buffett_indicator(equities, gdp, applicable_date)
            )
            sources.extend(
                [
                    ResearchSource(
                        source_id="fred_ncbeilq027s",
                        title="Nonfinancial Corporate Business; Corporate Equities; Liability, Level",
                        publisher="Federal Reserve Bank of St. Louis",
                        url="https://fred.stlouisfed.org/series/NCBEILQ027S",
                        source_type="FRED aggregate valuation",
                    ),
                    ResearchSource(
                        source_id="fred_gdp",
                        title="Gross Domestic Product",
                        publisher="Federal Reserve Bank of St. Louis",
                        url="https://fred.stlouisfed.org/series/GDP",
                        source_type="FRED aggregate valuation",
                    ),
                ]
            )
            _report_progress(
                progress,
                "FRED Buffett proxy",
                "complete",
                (
                    "Calculated the nonfinancial corporate-equities-to-GDP "
                    "proxy with its methodology kept explicit. Sources: "
                    f"{sources[-2].title}; {sources[-1].title}."
                ),
            )
        except ExternalResearchConfigurationError as error:
            gaps.append(
                _gap(
                    "buffett_indicator_proxy",
                    "configuration_missing",
                    str(error),
                )
            )
            _report_progress(
                progress,
                "FRED Buffett proxy",
                "unavailable",
                f"Configuration missing: {error}",
            )
        except ExternalResearchError as error:
            gaps.append(
                _gap("buffett_indicator_proxy", "fetch_failed", str(error))
            )
            _report_progress(
                progress,
                "FRED Buffett proxy",
                "unavailable",
                f"Proxy construction failed: {error}",
            )
        try:
            indicators.append(
                _shiller_cape(
                    applicable_date=applicable_date,
                    budget=budget,
                )
            )
            sources.append(
                ResearchSource(
                    source_id="yale_shiller_cape",
                    title="U.S. Stock Markets 1871-Present and CAPE Ratio",
                    publisher="Robert J. Shiller, Yale University",
                    url="https://www.econ.yale.edu/~shiller/data.htm",
                    source_type="official Shiller CAPE",
                )
            )
            _report_progress(
                progress,
                "Yale Shiller CAPE",
                "complete",
                (
                    "Prepared the latest CAPE observation and long-run "
                    f"comparison. Source: {sources[-1].title}."
                ),
            )
        except ExternalResearchError as error:
            gaps.append(_gap("shiller_cape", "fetch_failed", str(error)))
            _report_progress(
                progress,
                "Yale Shiller CAPE",
                "unavailable",
                f"Workbook retrieval or parsing failed: {error}",
            )
    else:
        gaps.append(
            _gap(
                "aggregate_valuation",
                "source_disabled",
                "Buffett and Shiller valuation indicators are disabled.",
            )
        )
        _report_progress(
            progress,
            "Aggregate valuation",
            "skipped",
            "The Buffett proxy and Shiller CAPE were disabled in settings.",
        )

    by_id = {indicator.indicator_id: indicator for indicator in indicators}
    treasury = by_id.get("treasury_5y")
    credit = by_id.get("investment_grade_credit_spread")
    indicative_borrowing_yield = (
        treasury.current + credit.current
        if treasury
        and credit
        and treasury.current is not None
        and credit.current is not None
        else None
    )
    result = MacroResearchData(
        applicable_date=applicable_date,
        generated_at=datetime.now(timezone.utc),
        indicators=indicators,
        indicative_borrowing_yield=indicative_borrowing_yield,
        sources=list({source.source_id: source for source in sources}.values()),
        gaps=gaps,
        external_requests_used=budget.used,
    )
    _report_progress(
        progress,
        "Macro evidence",
        "complete",
        (
            f"Prepared MacroResearchData with {len(result.sources)} source(s), "
            f"{len(result.indicators)} indicator(s), {len(result.gaps)} "
            f"explicit gap(s), and {result.external_requests_used} HTTP "
            "attempt(s)."
        ),
    )
    return result


def macro_research_to_view(data: MacroResearchData) -> MacroView:
    """Convert compressed numerical data into an identical shared MacroView."""

    environment: list[EvidenceClaim] = []
    for indicator in data.indicators:
        comparisons: list[str] = []
        if indicator.one_year_ago is not None:
            comparisons.append(
                f"one year earlier {indicator.one_year_ago:,.2f}"
            )
        if indicator.comparison_average is not None:
            comparison_label = (
                "long-run median"
                if indicator.indicator_id == "shiller_cape"
                else "comparison average"
            )
            comparisons.append(
                f"{comparison_label} {indicator.comparison_average:,.2f}"
            )
        if indicator.historical_percentile is not None:
            comparisons.append(
                f"historical percentile {indicator.historical_percentile:,.1f}"
            )
        comparison_text = "; ".join(comparisons)
        statement = (
            f"{indicator.label} is {indicator.current:,.2f} {indicator.unit}"
            if indicator.current is not None
            else f"{indicator.label} is unavailable"
        )
        if comparison_text:
            statement += f" ({comparison_text})."
        else:
            statement += "."
        environment.append(
            EvidenceClaim(
                claim_id=f"macro_{indicator.indicator_id}",
                statement=statement,
                value=(
                    round(indicator.current, 4)
                    if indicator.current is not None
                    else None
                ),
                unit=indicator.unit,
                period=indicator.observation_date.isoformat()
                if indicator.observation_date
                else None,
                status=(
                    "derived_metric"
                    if indicator.indicator_id == "buffett_indicator_proxy"
                    else "reported_fact"
                ),
                source_ids=indicator.source_ids,
                confidence=0.95 if indicator.freshness != "stale" else 0.7,
                as_of_date=indicator.observation_date,
            )
        )
    if data.indicative_borrowing_yield is not None:
        environment.append(
            EvidenceClaim(
                claim_id="macro_indicative_borrowing_yield",
                statement=(
                    "The broad indicative borrowing yield proxy is "
                    f"{data.indicative_borrowing_yield:,.2f} percent, calculated "
                    "as the 5-year Treasury yield plus the broad investment-grade "
                    "credit spread."
                ),
                value=round(data.indicative_borrowing_yield, 4),
                unit="percent",
                status="derived_metric",
                source_ids=["fred_dgs5", "fred_bamlc0a0cm"],
                confidence=0.9,
                as_of_date=data.applicable_date,
            )
        )

    by_id = {indicator.indicator_id: indicator for indicator in data.indicators}
    regime_risks: list[EvidenceClaim] = []
    financing = [
        by_id.get("treasury_5y"),
        by_id.get("investment_grade_credit_spread"),
    ]
    elevated_financing = [
        indicator
        for indicator in financing
        if indicator
        and indicator.current is not None
        and indicator.comparison_average is not None
        and indicator.current > indicator.comparison_average
    ]
    if elevated_financing:
        source_ids = list(
            dict.fromkeys(
                source_id
                for indicator in elevated_financing
                for source_id in indicator.source_ids
            )
        )
        regime_risks.append(
            EvidenceClaim(
                claim_id="macro_regime_financing",
                statement=(
                    "At least one broad financing measure is above its "
                    "three-year average; company relevance must be tested "
                    "against refinancing needs and balance-sheet resilience."
                ),
                status="derived_metric",
                source_ids=source_ids,
                confidence=0.9,
                as_of_date=data.applicable_date,
            )
        )
    elevated_valuation = [
        indicator
        for indicator_id in ("buffett_indicator_proxy", "shiller_cape")
        if (indicator := by_id.get(indicator_id))
        and indicator.historical_percentile is not None
        and indicator.historical_percentile >= 75
    ]
    if elevated_valuation:
        source_ids = list(
            dict.fromkeys(
                source_id
                for indicator in elevated_valuation
                for source_id in indicator.source_ids
            )
        )
        regime_risks.append(
            EvidenceClaim(
                claim_id="macro_regime_aggregate_valuation",
                statement=(
                    "At least one broad US market valuation measure is at or "
                    "above its 75th historical percentile. This is a valuation "
                    "guardrail, not a company-level investment conclusion."
                ),
                status="derived_metric",
                source_ids=source_ids,
                confidence=0.85,
                as_of_date=data.applicable_date,
            )
        )

    unresolved = [
        f"{gap.field}: {gap.detail}"
        for gap in data.gaps
        if gap.status not in {"source_disabled", "not_applicable"}
    ]
    return MacroView(
        as_of_date=data.applicable_date,
        environment=environment[:8],
        company_transmission_channels=[],
        regime_risks=regime_risks[:2],
        unresolved_questions=unresolved[:12],
        sources=data.sources,
    )


def _macro_store_path(
    settings: ExternalResearchSettings,
    applicable_date: date,
) -> Path:
    root = Path(
        os.getenv("MACRO_VIEW_STORE_PATH", str(DEFAULT_MACRO_STORE))
    ).expanduser()
    configuration = {
        "rates_and_credit_enabled": settings.rates_and_credit_enabled,
        "aggregate_valuation_enabled": settings.aggregate_valuation_enabled,
        "schema_version": 1,
    }
    fingerprint = sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return root / f"US_{applicable_date.isoformat()}_{fingerprint}.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def get_or_build_daily_macro(
    settings: ExternalResearchSettings,
    *,
    applicable_date: date | None = None,
    force_refresh: bool = False,
    progress: ResearchProgress | None = None,
) -> tuple[MacroResearchData, MacroView]:
    """Load today's validated daily macro file or construct it in five requests."""

    selected_date = applicable_date or date.today()
    path = _macro_store_path(settings, selected_date)
    with _DAILY_MACRO_LOCK:
        if path.is_file() and not force_refresh:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                research = MacroResearchData.model_validate(
                    payload["macro_research_data"]
                )
                view = MacroView.model_validate(payload["macro_view"])
                if (
                    research.applicable_date == selected_date
                    and view.as_of_date == selected_date
                ):
                    _report_progress(
                        progress,
                        "Daily MacroView",
                        "reused",
                        (
                            f"Loaded the validated {selected_date.isoformat()} "
                            f"daily macro file with {len(research.indicators)} "
                            "indicator(s); no macro HTTP requests were needed. "
                            "Sources: "
                            + (
                                "; ".join(
                                    source.title
                                    for source in research.sources
                                )
                                or "none recorded"
                            )
                            + "."
                        ),
                    )
                    return research, view
            except (OSError, KeyError, json.JSONDecodeError, ValueError):
                # An invalid daily macro file is never trusted; the bounded
                # source collector reconstructs it below.
                pass

        research = collect_macro_research(
            settings,
            applicable_date=selected_date,
            progress=progress,
        )
        view = macro_research_to_view(research)
        _write_json_atomic(
            path,
            {
                "macro_research_data": research.model_dump(mode="json"),
                "macro_view": view.model_dump(mode="json"),
            },
        )
        return research, view
