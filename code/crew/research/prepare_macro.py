"""Build the shared daily MacroView file for a selected day."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json

from crew.research.macro import get_or_build_daily_macro
from crew.schemas import ExternalResearchSettings


def _next_weekday(value: date) -> date:
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the bounded shared US daily macro file for a future "
            "applicable date. Without --applicable-date, use the next weekday."
        )
    )
    parser.add_argument(
        "--applicable-date",
        type=date.fromisoformat,
        default=None,
        help="Date in YYYY-MM-DD form. Defaults to the next weekday.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild an existing valid daily macro file.",
    )
    parser.add_argument(
        "--no-rates-and-credit",
        action="store_true",
    )
    parser.add_argument(
        "--no-aggregate-valuation",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    applicable_date = args.applicable_date or _next_weekday(date.today())
    settings = ExternalResearchSettings(
        enabled=True,
        rates_and_credit_enabled=not args.no_rates_and_credit,
        aggregate_valuation_enabled=not args.no_aggregate_valuation,
    )
    research, view = get_or_build_daily_macro(
        settings,
        applicable_date=applicable_date,
        force_refresh=args.force,
        progress=lambda message: print(message, flush=True),
    )
    print(
        json.dumps(
            {
                "applicable_date": applicable_date.isoformat(),
                "generated_at": research.generated_at.isoformat(),
                "external_requests_used": research.external_requests_used,
                "indicator_count": len(research.indicators),
                "claim_count": len(view.all_claims()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
