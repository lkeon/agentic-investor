"""Focused tests for the Streamlit committee adapter."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from committee_service import discover_investors


class InvestorDiscoveryTests(unittest.TestCase):
    @patch(
        "committee_service._discover_investors_from_database",
        return_value={"marks", "buffett", "pabrai"},
    )
    def test_database_investors_take_precedence(self, _: object) -> None:
        self.assertEqual(
            discover_investors(),
            ["buffett", "marks", "pabrai"],
        )


if __name__ == "__main__":
    unittest.main()
