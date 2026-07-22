"""Test fragment extraction without making OpenAI API requests."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

from mental_model_pipeline.fragments import extraction
from mental_model_pipeline.fragments.schemas import (
    MentalModelExtractionResult,
)


class FragmentExtractionTests(unittest.TestCase):
    def test_extraction_sets_output_token_limit(self) -> None:
        parsed = MentalModelExtractionResult()
        parse = Mock(return_value=SimpleNamespace(output_parsed=parsed))

        with patch.object(extraction.client.responses, "parse", parse):
            result = extraction.extract_fragments_from_document(
                markdown_text="A sufficiently long investment observation.",
                investor_name="Test Investor",
                document_id="doc_test_001",
            )

        self.assertEqual(result, parsed)
        self.assertEqual(
            parse.call_args.kwargs["max_output_tokens"],
            extraction.MAX_OUTPUT_TOKENS,
        )

    def test_empty_markdown_makes_no_api_request(self) -> None:
        parse = Mock(side_effect=AssertionError("API should not be called"))

        with patch.object(extraction.client.responses, "parse", parse):
            result = extraction.extract_fragments_from_document(
                markdown_text="  \n",
                investor_name="Test Investor",
                document_id="doc_test_001",
            )

        self.assertEqual(result, MentalModelExtractionResult())
        parse.assert_not_called()

    def test_truncated_json_has_clear_non_retryable_error(self) -> None:
        try:
            MentalModelExtractionResult.model_validate_json(
                '{"fragments":[{"proposition":"unfinished'
            )
        except ValidationError as validation_error:
            parse = Mock(side_effect=validation_error)
        else:  # pragma: no cover - guards the test fixture itself
            self.fail("The truncated JSON fixture unexpectedly validated")

        with (
            patch.object(extraction.client.responses, "parse", parse),
            self.assertRaisesRegex(
                RuntimeError,
                "incomplete or invalid JSON",
            ) as raised,
        ):
            extraction.extract_fragments_from_document(
                markdown_text="A sufficiently long investment observation.",
                investor_name="Test Investor",
                document_id="doc_test_001",
            )

        self.assertIn(
            f"{extraction.MAX_OUTPUT_TOKENS}-token output limit",
            str(raised.exception),
        )
        self.assertIsInstance(raised.exception.__cause__, ValidationError)
        parse.assert_called_once()


if __name__ == "__main__":
    unittest.main()
