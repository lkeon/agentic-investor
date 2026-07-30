"""Focused tests for the Streamlit committee adapter."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from sqlalchemy.exc import SQLAlchemyError

from committee_service import (
    InvestorDiscoveryError,
    _build_committee_command,
    _database_query_with_retries,
    _external_stage_update,
    _stage_update,
    available_investors_for_session,
    discover_investors,
)


class InvestorDiscoveryTests(unittest.TestCase):
    @patch("committee_service.time.sleep")
    def test_database_discovery_retries_transient_failure(
        self,
        sleep: object,
    ) -> None:
        operation = Mock(
            side_effect=[
                SQLAlchemyError("database is waking"),
                {"buffett", "marks"},
            ]
        )

        result = _database_query_with_retries(operation)

        self.assertEqual(result, {"buffett", "marks"})
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(0.5)

    @patch(
        "committee_service._discover_investors_from_database",
        return_value={"marks", "buffett", "pabrai"},
    )
    def test_database_investors_take_precedence(self, _: object) -> None:
        self.assertEqual(
            discover_investors(),
            ["buffett", "marks", "pabrai"],
        )

    @patch.dict(
        "os.environ",
        {"DILIGENCE_DEPLOYMENT": "streamlit_cloud"},
    )
    @patch(
        "committee_service._discover_investors_from_database",
        return_value=set(),
    )
    def test_cloud_does_not_mask_empty_database_with_fallback(
        self,
        _: object,
    ) -> None:
        with self.assertRaisesRegex(
            InvestorDiscoveryError,
            "contains no canonical mental models",
        ):
            discover_investors()

    @patch(
        "committee_service.discover_investors",
        return_value=["buffett", "flatt", "marks", "munger"],
    )
    def test_investors_are_discovered_once_per_browser_session(
        self,
        discover: object,
    ) -> None:
        state: dict[str, object] = {}

        first = available_investors_for_session(state)
        second = available_investors_for_session(state)

        self.assertEqual(first, second)
        self.assertEqual(discover.call_count, 1)


class CommitteeCommandTests(unittest.TestCase):
    def test_external_research_controls_are_forwarded_to_cli(self) -> None:
        command = _build_committee_command(
            "Should I buy Example?",
            investors=["buffett"],
            top_k=3,
            neighbours=1,
            external_research_settings={
                "enabled": True,
                "sec_filings_enabled": True,
                "investor_relations_enabled": False,
                "market_data_enabled": True,
                "rates_and_credit_enabled": True,
                "aggregate_valuation_enabled": False,
                "credit_rating_enabled": False,
                "max_filings": 1,
                "max_ir_documents": 0,
            },
        )

        self.assertIn("--external-research", command)
        self.assertIn("--sec-filings", command)
        self.assertIn("--no-investor-relations", command)
        self.assertIn("--no-aggregate-valuation", command)
        self.assertEqual(command[command.index("--max-filings") + 1], "1")
        self.assertNotIn("--max-discovery-searches", command)


class ExternalProgressTests(unittest.TestCase):
    def test_provider_messages_remain_technical_only(self) -> None:
        update = _external_stage_update(
            "External research | Alpaca market data | complete | "
            "Retrieved a USD 42.50 reference price dated 2026-07-24."
        )

        self.assertIsNone(update)

    def test_company_research_summary_becomes_customer_milestone(self) -> None:
        update = _external_stage_update(
            "External research | Company evidence | complete | "
            "Retrieved SEC Company Facts; 1 SEC filing(s); an Alpaca IEX "
            "market price."
        )

        self.assertIsNotNone(update)
        assert update is not None
        self.assertEqual(update.label, "Bounded research complete")
        self.assertEqual(update.progress, 26)
        self.assertIn("Alpaca IEX", update.detail)

    def test_investor_step_becomes_customer_facing_detail(self) -> None:
        update = _stage_update(
            "Investor step 1/2 · buffett · applying mental-model inference..."
        )

        self.assertIsNotNone(update)
        assert update is not None
        self.assertEqual(update.label, "Independent analysis · Buffett")
        self.assertIn("applying mental-model inference", update.detail)


if __name__ == "__main__":
    unittest.main()
