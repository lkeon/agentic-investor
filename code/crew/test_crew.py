"""Focused unit tests for the investor committee MVP contracts."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from pydantic import ValidationError

from crew.agents import (
    CitationValidationError,
    _claim_map,
    _disambiguate_macro_claim_ids,
    _validate_investor_output,
    build_mental_model_bridges,
    research_macro_view,
    run_cio_synthesis,
    run_investor_reasoning,
)
from crew.config import create_reasoning_llm, qualify_reasoning_model
from crew.research.alpaca import _market_data
from crew.research.common import (
    PROJECT_ROOT,
    ExternalResearchError,
    _RequestBudget,
)
from crew.research.exa import EXA_IR_OUTPUT_SCHEMA, _collect_ir
from crew.research.macro import (
    DEFAULT_MACRO_STORE,
    get_or_build_daily_macro,
    macro_research_to_view,
)
from crew.research.micro import merge_micro_research
from crew.retrieval import _build_adjacency, _normalise
from crew.run_crew import _ensure_database_running
from crew.research.sec import _collect_sec, _latest_shares_outstanding
from crew.schemas import (
    CIOReasoningInput,
    CIOReasoningOutput,
    EvidenceClaim,
    ExternalResearchSettings,
    FinancialPeriodData,
    InvestmentQuestion,
    OneInvestorReasoningInput,
    InvestorReasoningOutput,
    MentalModelInference,
    MacroView,
    MacroIndicatorData,
    MacroResearchData,
    MentalModelCandidate,
    MentalModelBridge,
    MentalModelBridgeDraft,
    MentalModelBridgeDraftList,
    MentalModelBridgeList,
    MicroView,
    MicroResearchData,
    ResearchSource,
)


TODAY = date(2026, 7, 24)


def _question() -> InvestmentQuestion:
    return InvestmentQuestion(
        original_question="Should I buy Example Company?",
        normalised_question=(
            "Should a long-term value investor buy Example Company?"
        ),
        subject="Example Company",
        decision_type="buy",
        as_of_date=TODAY,
    )


def _evidence() -> EvidenceClaim:
    return EvidenceClaim(
        claim_id="micro_valuation_1",
        statement="The supplied context reports a price-to-earnings ratio.",
        value=12.0,
        unit="times",
        period="FY2025",
        status="reported_fact",
        source_ids=["source_1"],
        confidence=0.9,
        as_of_date=TODAY,
    )


def _micro() -> MicroView:
    return MicroView(
        company_name="Example Company",
        ticker="EXM",
        security_type="common_equity",
        as_of_date=TODAY,
        valuation=[_evidence()],
        sources=[
            ResearchSource(
                source_id="source_1",
                title="Example filing",
                publisher="Example Company",
                source_type="filing",
                published_at=TODAY,
            )
        ],
    )


def _candidate() -> MentalModelCandidate:
    return MentalModelCandidate(
        canonical_code="mmc_example_12345678",
        investor_id="buffett",
        title="Margin of safety",
        proposition="Require a discount to conservatively estimated value.",
        conditions=["Value can be estimated conservatively."],
        matched_bridge_ids=["valuation"],
        relevant_claim_ids=["micro_valuation_1"],
        unresolved_conditions=["Value can be estimated conservatively."],
        retrieval_score=0.8,
        retrieval_origin="direct",
    )


def _bridge(*, candidate: bool = True) -> MentalModelBridge:
    question = _question()
    return MentalModelBridge(
        bridge_id="valuation",
        investor_id="buffett" if candidate else None,
        normalised_question=question.normalised_question,
        holding_policy=question.holding_policy,
        analytical_question=(
            "Does the current valuation provide a margin of safety?"
        ),
        search_query=(
            "Conservative value, normalized earnings, and margin of safety"
        ),
        decision_stage="valuation",
        domains=["valuation_and_expected_return"],
        importance=0.9,
        retrieved_data=[_evidence()],
        mental_model_candidates=[_candidate()] if candidate else [],
    )


def _reasoning_data() -> OneInvestorReasoningInput:
    return OneInvestorReasoningInput(
        investor_id="buffett",
        micro_view=_micro(),
        macro_view=MacroView(as_of_date=TODAY),
        mental_model_bridges=[_bridge()],
    )


def _output(
    *,
    code: str = "mmc_example_12345678",
    evidence_id: str = "micro_valuation_1",
) -> InvestorReasoningOutput:
    return InvestorReasoningOutput(
        investor_id="buffett",
        stance="mixed",
        thesis=(
            "The supplied evidence does not yet support an investment "
            "decision. The reported valuation multiple alone does not "
            "establish normalized owner earnings, intrinsic value, or an "
            "adequate margin of safety. Business durability, competitive "
            "economics, balance-sheet resilience, and management's capital-"
            "allocation record also remain insufficiently evidenced. From a "
            "Buffett perspective, uncertainty about these core microeconomic "
            "factors outweighs any provisional attraction in the headline "
            "valuation. The appropriate stance is therefore mixed and patient "
            "rather than affirmative. The company could become suitable for "
            "indefinite ownership only after durable economics, conservative "
            "financing, trustworthy stewardship, and a meaningful discount "
            "to conservatively estimated value are demonstrated."
        ),
        mental_model_inferences=[
            MentalModelInference(
                conclusion="A margin of safety is not yet established.",
                reasoning="One multiple is insufficient to estimate value.",
                mental_model_codes=[code],
                evidence_claim_ids=[evidence_id],
                applicability="uncertain",
            ),
            MentalModelInference(
                conclusion="Business durability remains unverified.",
                reasoning=(
                    "The supplied evidence does not establish durable "
                    "competitive economics."
                ),
                mental_model_codes=[code],
                evidence_claim_ids=[evidence_id],
                applicability="uncertain",
            ),
            MentalModelInference(
                conclusion="Permanent-loss exposure cannot yet be bounded.",
                reasoning=(
                    "One valuation observation does not establish financial "
                    "resilience or downside protection."
                ),
                mental_model_codes=[code],
                evidence_claim_ids=[evidence_id],
                applicability="uncertain",
            ),
        ],
        thesis_durability_assessment=(
            "The evidence is insufficient to support indefinite ownership."
        ),
        suitable_while_thesis_valid=False,
        missing_information=["Normalized owner earnings"],
        confidence=0.3,
    )
class SchemaTests(unittest.TestCase):
    def test_question_defaults_to_indefinite_thesis_policy(self) -> None:
        self.assertEqual(
            _question().holding_policy,
            "indefinite_while_thesis_valid",
        )

    def test_sourced_claim_requires_known_source(self) -> None:
        with self.assertRaises(ValidationError):
            MicroView(
                company_name="Example",
                as_of_date=TODAY,
                valuation=[_evidence()],
            )

    def test_micro_view_rekeys_duplicate_claim_ids_without_data_loss(
        self,
    ) -> None:
        first = _evidence().model_copy(
            update={"status": "unknown", "source_ids": []}
        )
        duplicate = first.model_copy(
            update={"statement": "A second, distinct valuation observation."}
        )
        reserved = first.model_copy(
            update={
                "claim_id": "micro_valuation_1_2",
                "statement": "An observation whose original ID is reserved.",
            }
        )

        # CrewAI hands Pydantic decoded JSON dictionaries, so exercise that
        # exact validation boundary rather than only model-instance inputs.
        view = MicroView.model_validate(
            {
                "company_name": "Example",
                "as_of_date": TODAY.isoformat(),
                "business": [first.model_dump(mode="json")],
                "business_quality": [duplicate.model_dump(mode="json")],
                "valuation": [reserved.model_dump(mode="json")],
            }
        )

        self.assertEqual(
            [claim.claim_id for claim in view.all_claims()],
            [
                "micro_valuation_1",
                "micro_valuation_1_3",
                "micro_valuation_1_2",
            ],
        )
        self.assertEqual(
            [claim.statement for claim in view.all_claims()],
            [first.statement, duplicate.statement, reserved.statement],
        )

    def test_macro_view_rekeys_duplicate_claim_ids(self) -> None:
        first = _evidence().model_copy(
            update={
                "claim_id": "macro_rates_1",
                "status": "unknown",
                "source_ids": [],
            }
        )
        duplicate = first.model_copy(
            update={"statement": "A distinct rates transmission observation."}
        )

        view = MacroView(
            as_of_date=TODAY,
            environment=[first],
            regime_risks=[duplicate],
        )

        self.assertEqual(
            [claim.claim_id for claim in view.all_claims()],
            ["macro_rates_1", "macro_rates_1_2"],
        )

    def test_reasoning_model_accepts_provider_qualified_name(self) -> None:
        self.assertEqual(
            qualify_reasoning_model("anthropic/example"),
            "anthropic/example",
        )

    def test_reasoning_model_qualifies_bare_name(self) -> None:
        self.assertEqual(
            qualify_reasoning_model("example"),
            "openai/example",
        )

    def test_direct_reasoning_model_remains_a_provider_string(self) -> None:
        self.assertEqual(
            create_reasoning_llm("openai/example"),
            "openai/example",
        )

    def test_openrouter_model_builds_openai_compatible_connection(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENROUTER_API_KEY": "test-openrouter-key",
                    "OPENROUTER_API_BASE": "https://router.example/v1/",
                    "OR_SITE_URL": "https://diligence.example",
                    "OR_APP_NAME": "The Diligence Room",
                },
            ),
            patch("crewai.LLM") as llm,
        ):
            configured = create_reasoning_llm(
                "openrouter/anthropic/example"
            )

        self.assertIs(configured, llm.return_value)
        llm.assert_called_once_with(
            model="anthropic/example",
            provider="openai",
            api_key="test-openrouter-key",
            base_url="https://router.example/v1",
            default_headers={
                "HTTP-Referer": "https://diligence.example",
                "X-OpenRouter-Title": "The Diligence Room",
            },
            additional_params={
                "extra_body": {
                    "provider": {
                        "require_parameters": True,
                    }
                }
            },
        )

    def test_openrouter_model_requires_its_own_api_key(self) -> None:
        with patch.dict(
            "os.environ",
            {"OPENROUTER_API_KEY": ""},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "OPENROUTER_API_KEY is required",
            ):
                create_reasoning_llm("openrouter/openai/example")

    def test_bridge_list_schema_has_an_object_root(self) -> None:
        schema = MentalModelBridgeList.model_json_schema()
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["bridges"]["type"], "array")

    def test_investor_schema_contains_only_independent_output_fields(self) -> None:
        schema = InvestorReasoningOutput.model_json_schema()
        self.assertNotIn("round_number", schema["properties"])
        self.assertNotIn("changed_view", schema["properties"])

    def test_investor_schema_rejects_a_change_field(self) -> None:
        payload = _output().model_dump()
        payload["changed_view"] = False
        with self.assertRaises(ValidationError):
            InvestorReasoningOutput.model_validate(payload)

    def test_investor_schema_requires_three_reasoning_inferences(self) -> None:
        payload = _output().model_dump()
        payload["mental_model_inferences"] = payload[
            "mental_model_inferences"
        ][:2]
        with self.assertRaises(ValidationError):
            InvestorReasoningOutput.model_validate(payload)

    def test_investor_schema_requires_substantive_investment_view(self) -> None:
        payload = _output().model_dump()
        payload["thesis"] = (
            "The valuation evidence remains incomplete for a decision."
        )
        with self.assertRaisesRegex(
            ValidationError,
            "Investment view must contain at least 80 words",
        ):
            InvestorReasoningOutput.model_validate(payload)

    def test_external_research_limits_are_hard_schema_limits(self) -> None:
        with self.assertRaises(ValidationError):
            ExternalResearchSettings(max_ir_documents=5)
        with self.assertRaises(ValidationError):
            ExternalResearchSettings(max_filings=3)


class ExternalResearchTests(unittest.TestCase):
    def test_research_paths_resolve_to_the_repository(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        self.assertEqual(PROJECT_ROOT, repository)
        self.assertEqual(
            DEFAULT_MACRO_STORE,
            repository / "data" / "processed" / "crew" / "macro_views",
        )

    def _macro_data(self) -> MacroResearchData:
        return MacroResearchData(
            applicable_date=TODAY,
            generated_at=datetime.now(timezone.utc),
            indicators=[
                MacroIndicatorData(
                    indicator_id="treasury_5y",
                    label="5-year US Treasury yield",
                    current=4.0,
                    one_year_ago=3.5,
                    comparison_average=3.75,
                    historical_percentile=70,
                    unit="percent",
                    observation_date=TODAY,
                    freshness="current",
                    source_ids=["fred_dgs5"],
                    methodology="Test methodology.",
                )
            ],
            sources=[
                ResearchSource(
                    source_id="fred_dgs5",
                    title="5-year Treasury",
                    publisher="FRED",
                    source_type="FRED rates",
                )
            ],
            external_requests_used=1,
        )

    def test_request_budget_counts_attempts_and_stops(self) -> None:
        budget = _RequestBudget(maximum=2)
        budget.take()
        budget.take()
        with self.assertRaises(ExternalResearchError):
            budget.take()

    def test_latest_sec_shares_outstanding_are_selected(self) -> None:
        facts = {
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {
                                    "form": "10-Q",
                                    "end": "2025-09-30",
                                    "filed": "2025-11-01",
                                    "val": 100,
                                },
                                {
                                    "form": "10-K",
                                    "end": "2025-12-31",
                                    "filed": "2026-02-01",
                                    "val": 110,
                                },
                            ]
                        }
                    }
                }
            }
        }

        self.assertEqual(
            _latest_shares_outstanding(facts, as_of_date=TODAY),
            (110.0, date(2025, 12, 31)),
        )

    def test_oversized_sec_filing_becomes_a_gap(self) -> None:
        filing = {
            "form": "10-K",
            "accession": "0000000000-26-000001",
            "primary_document": "annual-report.htm",
            "filing_date": TODAY.isoformat(),
        }
        with (
            patch(
                "crew.research.sec._resolve_sec_identity",
                return_value=("Example Company", "EXM", "0000000001"),
            ),
            patch("crew.research.sec._sec_headers", return_value={}),
            patch(
                "crew.research.sec._request_json",
                side_effect=[
                    {"entityName": "Example Company", "facts": {}},
                    {},
                ],
            ),
            patch("crew.research.sec._latest_filings", return_value=[filing]),
            patch(
                "crew.research.sec._request_bytes",
                side_effect=ExternalResearchError("Source payload exceeded limit."),
            ),
        ):
            result = _collect_sec(
                _question(),
                ExternalResearchSettings(max_filings=1),
                budget=_RequestBudget(maximum=8),
            )

        self.assertEqual(result["filings_fetched"], 0)
        self.assertTrue(
            any(gap.field == "10-K document" for gap in result["gaps"])
        )

    @patch.dict(
        "os.environ",
        {
            "ALPACA_API_KEY_ID": "test-key",
            "ALPACA_API_SECRET_KEY": "test-secret",
            "ALPACA_DATA_API_BASE": "https://data.example",
        },
    )
    @patch(
        "crew.research.alpaca._request_json",
        return_value={
            "latestTrade": {
                "p": 42.5,
                "t": "2026-07-24T19:59:59Z",
            }
        },
    )
    def test_alpaca_snapshot_uses_one_market_request(
        self,
        request_json: object,
    ) -> None:
        result = _market_data(
            "EXM",
            budget=_RequestBudget(maximum=2),
        )

        self.assertEqual(result["share_price"], 42.5)
        self.assertEqual(result["share_price_date"], TODAY)
        self.assertEqual(result["source"].publisher, "Alpaca")
        request_json.assert_called_once()
        self.assertIn(
            "/v2/stocks/EXM/snapshot?feed=iex",
            request_json.call_args.args[0],
        )
        self.assertEqual(
            request_json.call_args.kwargs["headers"]["APCA-API-KEY-ID"],
            "test-key",
        )
        self.assertEqual(
            request_json.call_args.kwargs["headers"]["APCA-API-SECRET-KEY"],
            "test-secret",
        )
        self.assertNotIn("test-key", request_json.call_args.args[0])

    @patch.dict("os.environ", {"EXA_API_KEY": "test-exa-key"})
    @patch(
        "crew.research.exa._request_json",
        return_value={
            "results": [
                {
                    "title": "Example Company annual report",
                    "url": "https://www.example.com/investors/annual-report",
                    "publishedDate": "2026-06-30T00:00:00.000Z",
                }
            ],
            "output": {
                "content": {
                    "company_description": (
                        "Example Company operates a focused industrial business."
                    ),
                    "qualitative_observations": [
                        "Management prioritises reinvestment at attractive returns.",
                        "The company maintains substantial available liquidity.",
                    ],
                    "credit_rating": "AA-",
                },
                "grounding": [
                    {
                        "field": "company_description",
                        "confidence": "high",
                        "citations": [
                            {
                                "title": "Example Company annual report",
                                "url": (
                                    "https://www.example.com/investors/"
                                    "annual-report"
                                ),
                            }
                        ],
                    },
                    {
                        "field": "qualitative_observations",
                        "confidence": "medium",
                        "citations": [
                            {
                                "title": "Example Company annual report",
                                "url": (
                                    "https://www.example.com/investors/"
                                    "annual-report"
                                ),
                            }
                        ],
                    },
                    {
                        "field": "credit_rating",
                        "confidence": "high",
                        "citations": [
                            {
                                "title": "Example Company annual report",
                                "url": (
                                    "https://www.example.com/investors/"
                                    "annual-report"
                                ),
                            }
                        ],
                    },
                ],
            },
        },
    )
    def test_exa_ir_search_uses_schema_and_grounded_official_sources(
        self,
        request_json: object,
    ) -> None:
        result = _collect_ir(
            "Example Company",
            ticker="EXM",
            settings=ExternalResearchSettings(
                enabled=True,
                credit_rating_enabled=True,
                max_ir_documents=1,
            ),
            budget=_RequestBudget(maximum=2),
        )

        self.assertEqual(result["searches"], 1)
        self.assertEqual(result["documents"], 1)
        self.assertEqual(result["rating"], "AA-")
        self.assertEqual(len(result["observations"]), 2)
        self.assertEqual(
            result["description"],
            "Example Company operates a focused industrial business.",
        )
        request_json.assert_called_once()
        request = request_json.call_args
        self.assertEqual(request.args[0], "https://api.exa.ai/search")
        self.assertEqual(request.kwargs["headers"]["x-api-key"], "test-exa-key")
        body = json.loads(request.kwargs["data"])
        self.assertEqual(body["outputSchema"], EXA_IR_OUTPUT_SCHEMA)
        self.assertEqual(body["type"], "auto")
        self.assertNotIn("api_key", body)

    @patch.dict("os.environ", {"EXA_API_KEY": "test-exa-key"})
    @patch(
        "crew.research.exa._request_json",
        return_value={
            "results": [],
            "output": {
                "content": {
                    "company_description": "An unsupported description.",
                    "qualitative_observations": [
                        "An unsupported management claim."
                    ],
                    "credit_rating": "AAA",
                },
                "grounding": [
                    {
                        "field": "content",
                        "confidence": "high",
                        "citations": [
                            {
                                "title": "Third-party article",
                                "url": "https://www.reuters.com/example",
                            }
                        ],
                    }
                ],
            },
        },
    )
    def test_exa_ir_discards_non_official_grounding(
        self,
        _: object,
    ) -> None:
        result = _collect_ir(
            "Example Company",
            ticker="EXM",
            settings=ExternalResearchSettings(
                enabled=True,
                credit_rating_enabled=True,
                max_ir_documents=1,
            ),
            budget=_RequestBudget(maximum=2),
        )

        self.assertEqual(result["sources"], [])
        self.assertEqual(result["observations"], [])
        self.assertIsNone(result["description"])
        self.assertIsNone(result["rating"])
        self.assertEqual(result["gaps"][0].status, "ambiguous_identity")

    def test_shared_macro_view_has_no_company_transmission(self) -> None:
        view = macro_research_to_view(self._macro_data())

        self.assertEqual(view.company_transmission_channels, [])
        self.assertEqual(view.environment[0].value, 4.0)
        self.assertEqual(view.environment[0].source_ids, ["fred_dgs5"])

    def test_daily_macro_artifact_is_reused_without_recollection(self) -> None:
        research = self._macro_data()
        view = macro_research_to_view(research)
        settings = ExternalResearchSettings(enabled=True)
        progress_messages: list[str] = []
        with TemporaryDirectory() as directory:
            with (
                patch.dict(
                    "os.environ",
                    {"MACRO_VIEW_STORE_PATH": directory},
                ),
                patch(
                    "crew.research.macro.collect_macro_research",
                    return_value=research,
                ) as collector,
            ):
                first = get_or_build_daily_macro(
                    settings,
                    applicable_date=TODAY,
                    progress=progress_messages.append,
                )
                second = get_or_build_daily_macro(
                    settings,
                    applicable_date=TODAY,
                    progress=progress_messages.append,
                )

            self.assertEqual(first, (research, view))
            self.assertEqual(second, (research, view))
            collector.assert_called_once()
            artifacts = list(Path(directory).glob("*.json"))
            self.assertEqual(len(artifacts), 1)
            payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
            self.assertIn("macro_view", payload)
            self.assertTrue(
                any(
                    "Daily MacroView | reused" in message
                    for message in progress_messages
                )
            )
            self.assertTrue(
                any(
                    "Daily MacroView | reused" in message
                    and "Sources: 5-year Treasury" in message
                    for message in progress_messages
                )
            )

    def test_micro_merge_hydrates_exact_collector_values(self) -> None:
        source = ResearchSource(
            source_id="sec_companyfacts_1",
            title="Company Facts",
            publisher="SEC",
            source_type="SEC XBRL company facts",
        )
        data = MicroResearchData(
            as_of_date=TODAY,
            company_name="Example Company",
            ticker="EXM",
            cik="0000000001",
            financial_periods=[
                FinancialPeriodData(
                    fiscal_year=2025,
                    period_end=TODAY,
                    revenue=100,
                    ebitda=20,
                    ebitda_basis="operating_income_plus_da",
                    free_cash_flow=10,
                    total_debt=30,
                    cash_and_equivalents=5,
                    net_debt=25,
                )
            ],
            sources=[source],
        )
        model_claim = EvidenceClaim(
            claim_id="micro_model_paraphrase",
            statement="The model changed the collected revenue to 999.",
            status="reported_fact",
            source_ids=[source.source_id],
            confidence=0.5,
        )
        view = MicroView(
            company_name="Wrong Name",
            as_of_date=TODAY,
            financial_performance=[model_claim],
            sources=[source],
        )

        merged = merge_micro_research(view, data)

        statements = [
            claim.statement
            for claim in merged.financial_performance
        ]
        self.assertFalse(any("999" in statement for statement in statements))
        self.assertTrue(any("100.00" in statement for statement in statements))
        self.assertEqual(merged.company_name, "Example Company")


class DatabaseStartupTests(unittest.TestCase):
    @patch("crew.run_crew._database_is_ready", side_effect=[False, True])
    @patch("crew.run_crew.subprocess.run")
    def test_starts_postgres_when_database_is_unavailable(
        self,
        run: object,
        _: object,
    ) -> None:
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        )

        _ensure_database_running()

        run.assert_called_once_with(
            ["systemctl", "start", "postgresql"],
            check=False,
            capture_output=True,
            text=True,
        )


class BridgeTests(unittest.TestCase):
    def test_macro_research_rekeys_cross_view_claim_collision(self) -> None:
        colliding_claim = _evidence().model_copy(
            update={"status": "unknown", "source_ids": []}
        )
        reserved_macro_claim = colliding_claim.model_copy(
            update={"claim_id": "macro_micro_valuation_1"}
        )
        returned = MacroView(
            as_of_date=TODAY,
            environment=[colliding_claim, reserved_macro_claim],
        )

        with patch(
            "crew.agents.run_structured_reasoning",
            return_value=returned,
        ):
            result = research_macro_view(
                _question(),
                _micro(),
                research_context=None,
                model="openai/example",
            )

        self.assertEqual(
            result.environment[0].claim_id,
            "macro_2_micro_valuation_1",
        )
        self.assertEqual(
            result.environment[1].claim_id,
            "macro_micro_valuation_1",
        )
        self.assertEqual(len(_claim_map(_micro(), result)), 3)

    def test_macro_rekey_leaves_unique_claim_ids_unchanged(self) -> None:
        claim = _evidence().model_copy(
            update={
                "claim_id": "macro_rates_1",
                "status": "unknown",
                "source_ids": [],
            }
        )
        macro_view = MacroView(as_of_date=TODAY, environment=[claim])

        result = _disambiguate_macro_claim_ids(_micro(), macro_view)

        self.assertEqual(result.environment[0].claim_id, "macro_rates_1")

    def test_bridge_builder_rehydrates_authoritative_evidence(self) -> None:
        bridge = _bridge(candidate=False)
        returned = MentalModelBridgeDraftList(
            bridges=[
                MentalModelBridgeDraft(
                    bridge_id=bridge.bridge_id,
                    analytical_question=bridge.analytical_question,
                    search_query=bridge.search_query,
                    decision_stage=bridge.decision_stage,
                    domains=bridge.domains,
                    importance=bridge.importance,
                    retrieved_claim_ids=["micro_valuation_1"],
                )
            ]
        )

        with patch(
            "crew.agents.run_structured_reasoning",
            return_value=returned,
        ):
            result = build_mental_model_bridges(
                _question(),
                _micro(),
                MacroView(as_of_date=TODAY),
                model="openai/example",
            )

        self.assertEqual(
            result[0].retrieved_data[0].statement,
            _evidence().statement,
        )

    def test_bridge_builder_rehydrates_authoritative_question(self) -> None:
        bridge = _bridge(candidate=False)
        returned = MentalModelBridgeDraftList(
            bridges=[
                MentalModelBridgeDraft(
                    bridge_id=bridge.bridge_id,
                    analytical_question=bridge.analytical_question,
                    search_query=bridge.search_query,
                    decision_stage=bridge.decision_stage,
                    domains=bridge.domains,
                    importance=bridge.importance,
                    retrieved_claim_ids=["micro_valuation_1"],
                )
            ]
        )

        with patch(
            "crew.agents.run_structured_reasoning",
            return_value=returned,
        ):
            result = build_mental_model_bridges(
                _question(),
                _micro(),
                MacroView(as_of_date=TODAY),
                model="openai/example",
            )

        self.assertEqual(
            result[0].normalised_question,
            _question().normalised_question,
        )

    def test_directional_edges_are_not_reversed(self) -> None:
        source_id = uuid4()
        target_id = uuid4()
        models = {
            source_id: SimpleNamespace(
                canonical_id=source_id,
                investor_id="buffett",
            ),
            target_id: SimpleNamespace(
                canonical_id=target_id,
                investor_id="buffett",
            ),
        }
        edge = SimpleNamespace(
            source_canonical_id=source_id,
            target_canonical_id=target_id,
            relation_type="parent_of",
            relation_strength=1.0,
            relation_confidence=1.0,
        )

        adjacency = _build_adjacency([edge], models)

        self.assertEqual(adjacency[source_id][0].other_id, target_id)
        self.assertNotIn(target_id, adjacency)

    def test_symmetric_edges_are_reversed(self) -> None:
        source_id = uuid4()
        target_id = uuid4()
        models = {
            source_id: SimpleNamespace(
                canonical_id=source_id,
                investor_id="buffett",
            ),
            target_id: SimpleNamespace(
                canonical_id=target_id,
                investor_id="buffett",
            ),
        }
        edge = SimpleNamespace(
            source_canonical_id=source_id,
            target_canonical_id=target_id,
            relation_type="similar_to",
            relation_strength=1.0,
            relation_confidence=1.0,
        )

        adjacency = _build_adjacency([edge], models)

        self.assertEqual(adjacency[target_id][0].other_id, source_id)

    def test_zero_embedding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _normalise(np.zeros(3))


class OutputGuardrailTests(unittest.TestCase):
    def test_unknown_evidence_id_is_rejected(self) -> None:
        with self.assertRaises(CitationValidationError):
            _validate_investor_output(
                _reasoning_data(),
                _output(evidence_id="REG-REG-04"),
            )

    def test_unknown_mental_model_code_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _validate_investor_output(
                _reasoning_data(),
                _output(code="invented_code"),
            )

    def test_valid_investor_output_passes(self) -> None:
        output = _output()
        self.assertIs(
            _validate_investor_output(
                _reasoning_data(),
                output,
            ),
            output,
        )

    def test_investor_reasoning_repairs_invalid_citation_once(self) -> None:
        invalid = _output(evidence_id="REG-REG-04")
        valid = _output()
        with patch(
            "crew.agents.run_structured_reasoning",
            side_effect=[invalid, valid],
        ) as reasoning:
            result = run_investor_reasoning(
                _reasoning_data(),
                model="openai/example",
            )

        self.assertIs(result, valid)
        self.assertEqual(reasoning.call_count, 2)
        first_context = reasoning.call_args_list[0].kwargs["context"]
        self.assertIn("citation_contract", first_context)
        repair_context = reasoning.call_args_list[1].kwargs["context"]
        self.assertEqual(
            repair_context["validation_error"],
            "unknown evidence IDs=['REG-REG-04']",
        )

    def test_investor_reasoning_stops_after_one_failed_repair(self) -> None:
        invalid = _output(evidence_id="REG-REG-04")
        with patch(
            "crew.agents.run_structured_reasoning",
            side_effect=[invalid, invalid],
        ) as reasoning:
            with self.assertRaises(CitationValidationError):
                run_investor_reasoning(
                    _reasoning_data(),
                    model="openai/example",
                )

        self.assertEqual(reasoning.call_count, 2)

    def test_cio_rejects_a_model_not_applied_by_an_investor(self) -> None:
        cio_input = CIOReasoningInput(
            question=_question(),
            micro_view=_micro(),
            macro_view=MacroView(as_of_date=TODAY),
            investor_outputs={"buffett": _output()},
        )
        cio_output = CIOReasoningOutput(
            decision="insufficient_information",
            final_answer=(
                "The supplied information is insufficient to establish a "
                "margin of safety."
            ),
            committee_synthesis=(
                "The independent view requires normalized owner earnings "
                "before valuation can be assessed."
            ),
            holding_policy="indefinite_while_thesis_valid",
            decisive_mental_model_codes=["mmc_not_applied_12345678"],
            confidence=0.2,
        )
        with patch(
            "crew.agents.run_structured_reasoning",
            return_value=cio_output,
        ):
            with self.assertRaises(ValueError):
                run_cio_synthesis(
                    cio_input,
                    model="openai/example",
                )


if __name__ == "__main__":
    unittest.main()
