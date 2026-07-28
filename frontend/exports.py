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


def committee_markdown(result: dict[str, object]) -> str:
    """Render the structured committee artifact in reasoning order."""

    question = result["question"]
    lines = [
        "# Investment Committee",
        "",
        "## Investment question",
        "",
        str(question["original_question"]),
        "",
        "## 1. Investor perspectives and mental models",
        "",
    ]

    investor_inputs = result.get("investor_reasoning_inputs", {})
    for investor_id, output in result.get("round_one", {}).items():
        confidence = round(float(output["confidence"]) * 100)
        lines.extend(
            [
                f"### {_title(investor_id)}",
                "",
                f"**Stance:** {_title(output['stance'])}  ",
                f"**Confidence:** {confidence}%",
                "",
                str(output["thesis"]),
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
        used_codes = list(
            dict.fromkeys(
                code
                for inference in output.get("mental_model_inferences", [])
                for code in inference.get("mental_model_codes", [])
            )
        )
        if used_codes:
            lines.extend(["#### Mental models applied", ""])
            lines.extend(
                f"- {models.get(code, code)} (`{code}`)" for code in used_codes
            )
            lines.append("")
        if output.get("mental_model_inferences"):
            lines.extend(["#### Reasoning", ""])
            for inference in output["mental_model_inferences"]:
                lines.extend(
                    [
                        f"- **{inference['conclusion']}** — {inference['reasoning']}",
                        "  Evidence: "
                        + ", ".join(inference.get("evidence_claim_ids", [])),
                    ]
                )
            lines.append("")
        _append_list(lines, "Permanent-loss risks", output.get("permanent_loss_risks", []))
        _append_list(lines, "Thesis-break conditions", output.get("thesis_break_conditions", []))

    lines.extend(["## 2. Peer review", ""])
    round_two = result.get("round_two", {})
    if not round_two:
        lines.extend(["Peer review was not run.", ""])
    for investor_id, output in round_two.items():
        changed = "Changed" if output.get("changed_view") else "Unchanged"
        lines.extend(
            [
                f"### {_title(investor_id)} — {changed}",
                "",
                str(
                    output.get("reason_for_change")
                    or "The peer-review round did not change the conclusion."
                ),
                "",
            ]
        )
        for review in output.get("peer_reviews", []):
            lines.extend(
                [
                    f"#### Review of {_title(review['peer_investor_id'])}",
                    "",
                ]
            )
            _append_list(lines, "Agreements", review.get("agreements", []))
            _append_list(lines, "Disagreements", review.get("disagreements", []))

    cio = result["cio"]
    confidence = round(float(cio["confidence"]) * 100)
    lines.extend(
        [
            "## 3. CIO decision",
            "",
            f"**Decision:** {_title(cio['decision'])}  ",
            f"**Confidence:** {confidence}%",
            "",
            str(cio["final_answer"]),
            "",
            "### Committee synthesis",
            "",
            str(cio["committee_synthesis"]),
            "",
        ]
    )
    _append_list(lines, "Decision conditions", cio.get("decision_conditions", []))
    _append_list(lines, "Key risks", cio.get("key_risks", []))
    _append_list(lines, "Missing information", cio.get("missing_information", []))

    sources = {
        source["source_id"]: source
        for view_name in ("micro_view", "macro_view")
        for source in result.get(view_name, {}).get("sources", [])
    }
    lines.extend(["## Sources", ""])
    if not sources:
        lines.extend(["No external sources were supplied.", ""])
    else:
        for source in sources.values():
            publisher = f" — {source['publisher']}" if source.get("publisher") else ""
            url = f" ({source['url']})" if source.get("url") else ""
            lines.append(f"- {source['title']}{publisher}{url}")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "Decision-support research, not personalised financial advice.",
            "",
        ]
    )
    return "\n".join(lines)
