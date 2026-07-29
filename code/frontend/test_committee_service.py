"""Focused tests for the Streamlit committee adapter."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from sqlalchemy.exc import SQLAlchemyError

from committee_service import (
    InvestorDiscoveryError,
    _database_query_with_retries,
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


if __name__ == "__main__":
    unittest.main()
