"""Focused unit tests for the investor committee MVP contracts."""

from __future__ import annotations

import unittest
from datetime import date
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
from crew.config import qualify_reasoning_model
from crew.retrieval import _build_adjacency, _normalise
from crew.run_crew import _ensure_database_running
from crew.schemas import (
    CIOReasoningInput,
    CIOReasoningOutput,
    EvidenceClaim,
    InvestmentQuestion,
    OneInvestorReasoningInput,
    InvestorReasoningOutput,
    MentalModelInference,
    MacroView,
    MentalModelCandidate,
    MentalModelBridge,
    MentalModelBridgeList,
    MicroView,
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
        altered_claim = bridge.retrieved_data[0].model_copy(
            update={"statement": "Altered by the bridge builder."}
        )
        returned = MentalModelBridgeList(
            bridges=[
                bridge.model_copy(
                    update={"retrieved_data": [altered_claim]}
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
