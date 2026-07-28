"""Human-readable Markdown export for the committee MVP."""

from __future__ import annotations


def _title(identifier: object) -> str:
    return str(identifier or "").replace("_", " ").replace("-", " ").title()


def _append_list(lines: list[str], heading: str, items: list[str]) -> None:
    if not items:
        return
    lines.extend([f"### {heading}", ""])
    lines.extend(f"- {item}" for item in items)
    lines.append("")


def _literal(value: object) -> str:
    """Keep currency markers literal in Markdown renderers with math support."""

    return str(value or "").replace("$", r"\$")


def _append_evidence(
    lines: list[str],
    *,
    heading: str,
    evidence_ids: list[str],
    claims: dict[str, dict[str, object]],
    sources: dict[str, dict[str, object]],
) -> None:
    if not evidence_ids:
        return
    lines.extend([f"##### {heading}", ""])
    for evidence_id in dict.fromkeys(evidence_ids):
        claim = claims.get(evidence_id)
        if claim is None:
            lines.append(f"- `{evidence_id}` — evidence record unavailable")
            continue
        source_titles = [
            str(sources[source_id]["title"])
            for source_id in claim.get("source_ids", [])
            if source_id in sources
        ]
        source_text = (
            " Sources: " + "; ".join(source_titles) + "."
            if source_titles
            else ""
        )
        lines.append(
            f"- {_literal(claim['statement'])} (`{evidence_id}`)."
            f"{source_text}"
        )
    lines.append("")


def committee_markdown(result: dict[str, object]) -> str:
    """Render the structured committee artifact in reasoning order."""

    question = result["question"]
    lines = [
        "# The Diligence Room",
        "",
        "## Investment question",
        "",
        str(question["original_question"]),
        "",
        "## 1. Investor perspectives and mental models",
        "",
    ]

    claims = {
        claim["claim_id"]: claim
        for view_name, field_names in (
            (
                "micro_view",
                (
                    "business",
                    "business_quality",
                    "management_and_capital_allocation",
                    "financial_performance",
                    "balance_sheet_and_liquidity",
                    "valuation",
                    "risks_and_catalysts",
                ),
            ),
            (
                "macro_view",
                (
                    "environment",
                    "company_transmission_channels",
                    "regime_risks",
                ),
            ),
        )
        for field_name in field_names
        for claim in result.get(view_name, {}).get(field_name, [])
    }
    sources = {
        source["source_id"]: source
        for view_name in ("micro_view", "macro_view")
        for source in result.get(view_name, {}).get("sources", [])
    }
    investor_inputs = result.get("investor_reasoning_inputs", {})
    for investor_id, output in result.get("investor_outputs", {}).items():
        confidence = round(float(output["confidence"]) * 100)
        lines.extend(
            [
                f"### {_title(investor_id)}",
                "",
                f"**Stance:** {_title(output['stance'])}  ",
                f"**Confidence:** {confidence}%",
                "",
                _literal(output["thesis"]),
                "",
            ]
        )
        models = {
            candidate["canonical_code"]: candidate["title"]
            for bridge in investor_inputs.get(investor_id, {}).get(
                "mental_model_bridges", []
            )
            for candidate in bridge.get("mental_model_candidates", [])
        }
        if output.get("mental_model_inferences"):
            lines.extend(["#### Reasoning", ""])
            for number, inference in enumerate(
                output["mental_model_inferences"],
                start=1,
            ):
                lines.extend(
                    [
                        f"##### {number}. {_literal(inference['conclusion'])}",
                        "",
                        _literal(inference["reasoning"]),
                        "",
                    ]
                )
                model_codes = inference.get("mental_model_codes", [])
                if model_codes:
                    lines.extend(["**Applicable mental models**", ""])
                    lines.extend(
                        f"- {models.get(code, code)} (`{code}`)"
                        for code in model_codes
                    )
                    lines.append("")
                _append_evidence(
                    lines,
                    heading="Supporting evidence",
                    evidence_ids=inference.get("evidence_claim_ids", []),
                    claims=claims,
                    sources=sources,
                )
                _append_evidence(
                    lines,
                    heading="Counterevidence",
                    evidence_ids=inference.get(
                        "counterevidence_claim_ids",
                        [],
                    ),
                    claims=claims,
                    sources=sources,
                )
            lines.append("")
        lines.extend(
            [
                "#### Continued ownership view",
                "",
                _literal(output["thesis_durability_assessment"]),
                "",
            ]
        )
        _append_list(lines, "Permanent-loss risks", output.get("permanent_loss_risks", []))
        _append_list(lines, "Thesis-break conditions", output.get("thesis_break_conditions", []))

    cio = result["cio"]
    confidence = round(float(cio["confidence"]) * 100)
    lines.extend(
        [
            "## 2. CIO decision",
            "",
            f"**Decision:** {_title(cio['decision'])}  ",
            f"**Confidence:** {confidence}%",
            "",
            _literal(cio["final_answer"]),
            "",
            "### Committee synthesis",
            "",
            _literal(cio["committee_synthesis"]),
            "",
        ]
    )
    applicable_codes = cio.get("decisive_mental_model_codes", [])
    if applicable_codes:
        all_models = {
            candidate["canonical_code"]: (
                candidate["title"],
                _title(candidate.get("investor_id")),
            )
            for investor_input in investor_inputs.values()
            for bridge in investor_input.get("mental_model_bridges", [])
            for candidate in bridge.get("mental_model_candidates", [])
        }
        lines.extend(["### Applicable investor mental models", ""])
        for code in applicable_codes:
            title, investor = all_models.get(code, (code, "Investor"))
            lines.append(f"- {title} — {investor} (`{code}`)")
        lines.append("")
    _append_evidence(
        lines,
        heading="Evidence informing the CIO decision",
        evidence_ids=cio.get("decisive_evidence_claim_ids", []),
        claims=claims,
        sources=sources,
    )
    _append_list(lines, "Decision conditions", cio.get("decision_conditions", []))
    _append_list(lines, "Key risks", cio.get("key_risks", []))
    _append_list(lines, "Missing information", cio.get("missing_information", []))

    lines.extend(
        [
            "---",
            "",
            "Decision-support research, not personalised financial advice.",
            "",
        ]
    )
    return "\n".join(lines)
