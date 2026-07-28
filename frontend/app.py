"""Professional Streamlit interface for the investment committee MVP."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from committee_service import (
    CHECKPOINT_PATH,
    CommitteeRunError,
    CommitteeStopped,
    ProgressUpdate,
    discover_investors,
    display_name,
    load_latest_result,
    run_committee,
    stop_active_committee,
)
from exports import committee_markdown


EXAMPLE_QUESTION = "Should I invest in Brookfield at the current price?"
DECISION_LABELS = {
    "buy": "Buy",
    "buy_below_price": "Buy below price",
    "watchlist": "Watchlist",
    "hold": "Hold",
    "sell": "Sell",
    "avoid": "Avoid",
    "insufficient_information": "Insufficient information",
}


@st.cache_data(show_spinner=False)
def _available_investors() -> list[str]:
    return discover_investors()


def _inject_styles() -> None:
    styles = (APP_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)


def _safe(value: object) -> str:
    return escape(str(value or ""))


def _chips(labels: list[str], *, tone: str = "neutral") -> str:
    unique = list(dict.fromkeys(label for label in labels if label))
    return "".join(
        f'<span class="chip chip-{tone}">{_safe(label)}</span>'
        for label in unique
    )


def _bullet_html(items: list[str], empty: str) -> str:
    if not items:
        return f'<p class="muted">{_safe(empty)}</p>'
    return "<ul>" + "".join(f"<li>{_safe(item)}</li>" for item in items) + "</ul>"


def _holding_policy_label(policy: str) -> str:
    labels = {
        "fixed_horizon": "Fixed horizon",
        "long_term": "Long term",
        "indefinite_while_thesis_valid": "Indefinite while thesis remains valid",
    }
    return labels.get(policy, display_name(policy))


def _horizon_label(cio: dict[str, object]) -> str:
    minimum = cio.get("investment_horizon_min_years")
    maximum = cio.get("investment_horizon_max_years")
    if maximum is None:
        return f"{minimum:g}+ years" if isinstance(minimum, (int, float)) else "Long term"
    if minimum == maximum:
        return f"{minimum:g} years"
    return f"{minimum:g}–{maximum:g} years"


def _claim_catalogue(result: dict[str, object]) -> dict[str, dict[str, object]]:
    claims: dict[str, dict[str, object]] = {}
    micro = result.get("micro_view", {})
    macro = result.get("macro_view", {})
    for key in (
        "business",
        "business_quality",
        "management_and_capital_allocation",
        "financial_performance",
        "balance_sheet_and_liquidity",
        "valuation",
        "risks_and_catalysts",
    ):
        for claim in micro.get(key, []):
            claims[claim["claim_id"]] = {**claim, "section": display_name(key)}
    for key in ("environment", "company_transmission_channels", "regime_risks"):
        for claim in macro.get(key, []):
            claims[claim["claim_id"]] = {**claim, "section": display_name(key)}
    return claims


def _source_catalogue(result: dict[str, object]) -> dict[str, dict[str, object]]:
    sources: dict[str, dict[str, object]] = {}
    for view_name in ("micro_view", "macro_view"):
        for source in result.get(view_name, {}).get("sources", []):
            sources[source["source_id"]] = source
    return sources


def _model_catalogue(
    result: dict[str, object],
    investor_id: str,
) -> dict[str, dict[str, object]]:
    catalogue: dict[str, dict[str, object]] = {}
    investor_input = result.get("investor_reasoning_inputs", {}).get(
        investor_id,
        {},
    )
    for bridge in investor_input.get("mental_model_bridges", []):
        for candidate in bridge.get("mental_model_candidates", []):
            catalogue[candidate["canonical_code"]] = candidate
    return catalogue


def _committee_model_catalogue(result: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return every retrieved model that may be cited in the CIO synthesis."""

    catalogue: dict[str, dict[str, object]] = {}
    for investor_id in result.get("investor_reasoning_inputs", {}):
        catalogue.update(_model_catalogue(result, investor_id))
    return catalogue


def _render_header() -> None:
    st.markdown(
        """
        <div class="product-header">
          <div class="brand-lockup">
            <div class="brand-mark">IC</div>
            <div>
              <div class="brand-name">Investment Committee</div>
              <div class="brand-subtitle">Mental-model decision support</div>
            </div>
          </div>
          <div class="product-pill">VALUE INVESTING · 2+ YEAR HORIZON</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_input() -> None:
    st.markdown(
        """
        <div class="intro-block">
          <div class="eyebrow">CONVENE THE COMMITTEE</div>
          <h1>Turn an investment question into a disciplined decision.</h1>
          <p>Company evidence is separated from mental models, investors reason
          independently, peers challenge the conclusions, and the CIO makes the
          final call.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "question_input" not in st.session_state:
        st.session_state.question_input = EXAMPLE_QUESTION

    with st.form("committee_form", border=False):
        question = st.text_area(
            "Investment question",
            key="question_input",
            height=105,
            placeholder="Should I invest in … at the current price?",
            help=(
                "Include the security and decision you face. The committee "
                "defaults to a minimum two-year horizon when none is supplied."
            ),
        )

        research_context = st.text_area(
            "Research context (optional)",
            height=115,
            placeholder=(
                "Paste current valuation, filings extracts, balance-sheet data, "
                "or other evidence. Unsupported facts remain explicit unknowns."
            ),
        )
        uploaded = st.file_uploader(
            "Or attach UTF-8 research",
            type=["txt", "md", "json", "csv"],
            help="The MVP structures supplied research; it does not browse the web.",
        )

        with st.expander("Committee settings", expanded=False):
            available = _available_investors()
            preferred = [
                investor
                for investor in ("buffett", "marks", "flatt")
                if investor in available
            ]
            investors = st.pills(
                "Investor perspectives",
                options=available,
                selection_mode="multi",
                default=preferred or available[:3],
                format_func=display_name,
                width="stretch",
                help=(
                    "Each perspective receives only its retrieved canonical "
                    "mental models and the same evidence record."
                ),
            )
            st.caption(
                f"All {len(available)} available investor perspectives are "
                "shown. Select one or more."
            )
            settings_left, settings_right = st.columns(2)
            with settings_left:
                top_k = st.slider(
                    "Direct models per question",
                    min_value=1,
                    max_value=6,
                    value=3,
                )
            with settings_right:
                neighbours = st.slider(
                    "Related models per question",
                    min_value=0,
                    max_value=3,
                    value=1,
                )
            resume = st.checkbox(
                "Resume the matching checkpoint",
                value=False,
                disabled=not CHECKPOINT_PATH.is_file(),
                help=(
                    "Skips completed stages only when every question and run "
                    "setting matches the saved checkpoint."
                ),
            )
            show_logs = st.checkbox(
                "Show technical execution log",
                value=False,
            )

        actions_col, _ = st.columns([1.7, 2.3])
        with actions_col:
            run_col, stop_col = st.columns([1.45, 0.55])
            with run_col:
                submitted = st.form_submit_button(
                    "Run investment committee",
                    type="primary",
                    use_container_width=True,
                )
            with stop_col:
                stop_requested = st.form_submit_button(
                    "Stop",
                    use_container_width=True,
                )

    st.caption(
        "Decision-support research, not personalised financial advice. "
        "Verify current facts before committing capital."
    )

    if stop_requested:
        if stop_active_committee():
            st.warning("Stopping the active committee run…")
        else:
            st.info("No committee run is currently active.")
        return

    if not submitted:
        return
    if not question.strip():
        st.error("Enter an investment question before convening the committee.")
        return
    if not investors:
        st.error("Select at least one investor perspective.")
        return

    context_parts = [research_context.strip()] if research_context.strip() else []
    if uploaded is not None:
        try:
            context_parts.append(uploaded.getvalue().decode("utf-8"))
        except UnicodeDecodeError:
            st.error("The attached research file must be UTF-8 encoded.")
            return
    combined_context = "\n\n".join(context_parts) or None

    with st.status("Committee convened", expanded=True) as run_status:
        progress_bar = st.progress(2)
        stage_text = st.empty()
        log_view = st.empty()

        def on_progress(update: ProgressUpdate, logs: list[str]) -> None:
            progress_bar.progress(update.progress)
            stage_text.markdown(f"**{update.label}**")
            if show_logs:
                # Streamlit's HTML sanitisation can collapse literal newlines
                # in a Markdown block. Explicit breaks preserve one terminal
                # event per visible line.
                rendered_logs = "<br>".join(
                    _safe(line).replace("\n", "<br>") for line in logs[-120:]
                )
                log_view.markdown(
                    '<div class="execution-log"><pre>'
                    f"{rendered_logs}"
                    "</pre></div>",
                    unsafe_allow_html=True,
                )

        try:
            result = run_committee(
                question,
                investors=investors,
                research_context=combined_context,
                top_k=top_k,
                neighbours=neighbours,
                resume=resume,
                on_progress=on_progress,
            )
        except CommitteeStopped:
            run_status.update(
                label="Committee analysis stopped",
                state="error",
                expanded=False,
            )
            st.warning("Committee execution was stopped. The previous completed result is unchanged.")
            return
        except CommitteeRunError as error:
            run_status.update(
                label="Committee analysis could not be completed",
                state="error",
                expanded=True,
            )
            st.error(str(error))
            if error.logs and not show_logs:
                with st.expander("Technical details"):
                    st.code("\n".join(error.logs[-30:]), language=None)
            return
        except Exception as error:
            run_status.update(
                label="Committee analysis could not be completed",
                state="error",
                expanded=True,
            )
            st.error(f"{type(error).__name__}: {error}")
            return

        progress_bar.progress(100)
        stage_text.markdown("**Committee analysis completed**")
        run_status.update(
            label="Committee analysis completed",
            state="complete",
            expanded=False,
        )
        st.session_state.committee_result = result


def _render_decision(result: dict[str, object]) -> None:
    cio = result["cio"]
    decision = str(cio["decision"])
    confidence = round(float(cio["confidence"]) * 100)
    decision_label = DECISION_LABELS.get(decision, display_name(decision))
    conditions = cio.get("decision_conditions", [])

    st.markdown(
        f"""
        <div class="decision-card decision-{_safe(decision)}">
          <div class="decision-topline">
            <div>
              <div class="card-label">CIO DECISION</div>
              <div class="decision-name">{_safe(decision_label)}</div>
            </div>
            <div class="confidence-block">
              <div class="confidence-value">{confidence}%</div>
              <div class="confidence-label">confidence</div>
            </div>
          </div>
          <div class="confidence-track"><span style="width:{confidence}%"></span></div>
          <p class="final-answer">{_safe(cio['final_answer'])}</p>
          <div class="condition-panel">
            <div class="card-label">KEY CONDITIONS</div>
            {_bullet_html(conditions, 'No explicit decision conditions returned.')}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Investment horizon", _horizon_label(cio))
    metric_two.metric("Holding policy", _holding_policy_label(cio["holding_policy"]))
    metric_three.metric(
        "Decisive evidence",
        str(len(cio.get("decisive_evidence_claim_ids", []))),
        help="Number of supplied evidence claims cited in the final decision.",
    )

    st.markdown("#### Committee synthesis")
    st.write(cio["committee_synthesis"])

    risk_col, missing_col = st.columns(2)
    with risk_col:
        st.markdown("#### Key risks")
        st.markdown(_bullet_html(cio.get("key_risks", []), "No risks returned."), unsafe_allow_html=True)
    with missing_col:
        st.markdown("#### Missing information")
        st.markdown(
            _bullet_html(
                cio.get("missing_information", []),
                "No decision-critical gaps returned.",
            ),
            unsafe_allow_html=True,
        )

    decisive_models = cio.get("decisive_mental_model_codes", [])
    if decisive_models:
        model_catalogue = _committee_model_catalogue(result)
        model_titles = [
            str(model_catalogue.get(code, {}).get("title", code))
            for code in decisive_models
        ]
        st.markdown("#### CIO-selected mental models")
        st.markdown(_chips(model_titles, tone="model"), unsafe_allow_html=True)


def _render_investor(
    result: dict[str, object],
    investor_id: str,
) -> None:
    round_one = result.get("round_one", {}).get(investor_id)
    round_two = result.get("round_two", {}).get(investor_id)
    # The first section intentionally shows the independent round-one view.
    # Changes made after peer review belong to the next visible stage.
    output = round_one or round_two
    if not output:
        st.info("No completed reasoning output is available for this investor.")
        return

    round_label = "Independent view" if round_one else "Peer-reviewed view"
    confidence = round(float(output["confidence"]) * 100)
    st.markdown(
        f"""
        <div class="perspective-header">
          <div>
            <div class="card-label">{_safe(round_label.upper())}</div>
            <h3>{_safe(display_name(investor_id))} perspective</h3>
          </div>
          <div class="stance-block">
            <span class="stance stance-{_safe(output['stance'])}">{_safe(display_name(output['stance']))}</span>
            <span class="confidence-inline">{confidence}% confidence</span>
          </div>
        </div>
        <p class="thesis">{_safe(output['thesis'])}</p>
        """,
        unsafe_allow_html=True,
    )

    catalogue = _model_catalogue(result, investor_id)
    used_codes = [
        code
        for inference in output.get("mental_model_inferences", [])
        for code in inference.get("mental_model_codes", [])
    ]
    used_titles = [
        catalogue.get(code, {}).get("title", code)
        for code in dict.fromkeys(used_codes)
    ]
    st.markdown("#### Mental models applied")
    if used_titles:
        st.markdown(
            f'<div class="chip-row">{_chips(used_titles, tone="model")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No mental model was marked applicable to the available evidence.")

    with st.expander(
        f"Retrieved mental-model set · {len(catalogue)} models",
        expanded=False,
    ):
        if not catalogue:
            st.caption("No mental-model candidates were retrieved.")
        for code, model in catalogue.items():
            score = round(float(model.get("retrieval_score", 0)) * 100)
            origin = display_name(model.get("retrieval_origin", "direct"))
            st.markdown(
                f"""
                <div class="model-row">
                  <div class="model-row-top">
                    <strong>{_safe(model['title'])}</strong>
                    <span>{score}% · {_safe(origin)}</span>
                  </div>
                  <p>{_safe(model['proposition'])}</p>
                  <code>{_safe(code)}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### Reasoning")
    for number, inference in enumerate(output.get("mental_model_inferences", []), start=1):
        applicability = display_name(inference.get("applicability", "uncertain"))
        with st.expander(f"{number}. {inference['conclusion']}"):
            st.caption(f"Applicability · {applicability}")
            st.write(inference["reasoning"])
            model_names = [
                catalogue.get(code, {}).get("title", code)
                for code in inference.get("mental_model_codes", [])
            ]
            if model_names:
                st.markdown("**Models:** " + " · ".join(model_names))
            evidence_ids = inference.get("evidence_claim_ids", [])
            counter_ids = inference.get("counterevidence_claim_ids", [])
            if evidence_ids:
                st.markdown("**Evidence:** `" + "`, `".join(evidence_ids) + "`")
            if counter_ids:
                st.markdown("**Counterevidence:** `" + "`, `".join(counter_ids) + "`")
            if inference.get("missing_information"):
                st.markdown(
                    _bullet_html(
                        inference["missing_information"],
                        "",
                    ),
                    unsafe_allow_html=True,
                )

    left, middle, right = st.columns(3)
    with left:
        st.markdown("##### Temporary concerns")
        st.markdown(
            _bullet_html(output.get("temporary_concerns", []), "None identified."),
            unsafe_allow_html=True,
        )
    with middle:
        st.markdown("##### Permanent-loss risks")
        st.markdown(
            _bullet_html(output.get("permanent_loss_risks", []), "None identified."),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("##### Thesis breaks")
        st.markdown(
            _bullet_html(output.get("thesis_break_conditions", []), "None returned."),
            unsafe_allow_html=True,
        )

    cited_ids = list(
        dict.fromkeys(
            evidence_id
            for inference in output.get("mental_model_inferences", [])
            for evidence_id in (
                inference.get("evidence_claim_ids", [])
                + inference.get("counterevidence_claim_ids", [])
            )
        )
    )
    claims = _claim_catalogue(result)
    with st.expander("Evidence and references", expanded=False):
        if not cited_ids:
            st.caption("This perspective did not cite a supplied evidence claim.")
        for evidence_id in cited_ids:
            claim = claims.get(evidence_id)
            if not claim:
                st.warning(f"Unknown evidence reference: {evidence_id}")
                continue
            st.markdown(
                f"**`{evidence_id}` · {claim['section']}**  \n{claim['statement']}"
            )
            st.caption(
                f"Status: {display_name(claim['status'])} · "
                f"Confidence: {round(float(claim['confidence']) * 100)}%"
            )


def _render_investors(result: dict[str, object]) -> None:
    investor_ids = list(result.get("investor_reasoning_inputs", {}))
    if not investor_ids:
        st.info("No investor perspectives were returned.")
        return
    selected_investor = st.pills(
        "Choose an investor perspective",
        options=investor_ids,
        selection_mode="single",
        default=investor_ids[0],
        format_func=lambda investor_id: f"👤\n{display_name(investor_id)}",
        width="stretch",
        key="result_investor_selector",
    )
    if selected_investor:
        _render_investor(result, selected_investor)


def _render_peer_review(result: dict[str, object]) -> None:
    round_two = result.get("round_two", {})
    if not round_two:
        st.info(
            "Peer review was not run. Select at least two investors and keep "
            "round two enabled in the CLI workflow."
        )
        return

    agreements: list[str] = []
    disagreements: list[str] = []
    for output in round_two.values():
        for review in output.get("peer_reviews", []):
            agreements.extend(review.get("agreements", []))
            disagreements.extend(review.get("disagreements", []))

    agreement_col, disagreement_col = st.columns(2)
    with agreement_col:
        st.markdown("#### Areas of agreement")
        st.markdown(
            _bullet_html(
                list(dict.fromkeys(agreements)),
                "No explicit agreements returned.",
            ),
            unsafe_allow_html=True,
        )
    with disagreement_col:
        st.markdown("#### Material disagreements")
        st.markdown(
            _bullet_html(
                list(dict.fromkeys(disagreements)),
                "No explicit disagreements returned.",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("#### View changes")
    for investor_id, output in round_two.items():
        changed = bool(output.get("changed_view"))
        label = "Changed" if changed else "Unchanged"
        reason = output.get("reason_for_change") or (
            "The peer-review round did not change the conclusion."
        )
        st.markdown(
            f"""
            <div class="change-row">
              <div><strong>{_safe(display_name(investor_id))}</strong><br><span>{_safe(reason)}</span></div>
              <span class="change-badge {'changed' if changed else 'unchanged'}">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Detailed peer assessments"):
        for reviewer_id, output in round_two.items():
            st.markdown(f"### {display_name(reviewer_id)}")
            for review in output.get("peer_reviews", []):
                st.markdown(f"**Review of {display_name(review['peer_investor_id'])}**")
                detail_columns = (
                    ("Factual", review.get("factual_disagreements", [])),
                    (
                        "Model applicability",
                        review.get("model_applicability_disagreements", []),
                    ),
                    ("Weighting", review.get("weighting_disagreements", [])),
                )
                for label, items in detail_columns:
                    if items:
                        st.markdown(f"**{label}:** " + " · ".join(items))


def _valid_external_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _render_evidence_and_sources(result: dict[str, object]) -> None:
    evidence_tab, sources_tab = st.tabs(["Evidence ledger", "Sources"])
    with evidence_tab:
        claims = _claim_catalogue(result)
        if not claims:
            st.info("No evidence claims were returned.")
        for claim_id, claim in claims.items():
            with st.expander(f"{claim_id} · {claim['section']}"):
                st.write(claim["statement"])
                value = claim.get("value")
                if value is not None:
                    unit = f" {claim.get('unit')}" if claim.get("unit") else ""
                    st.markdown(f"**Value:** {value}{unit}")
                st.caption(
                    f"{display_name(claim['status'])} · "
                    f"confidence {round(float(claim['confidence']) * 100)}%"
                )

    with sources_tab:
        sources = _source_catalogue(result)
        if not sources:
            st.info(
                "No external sources were supplied. Current unsupported facts "
                "should therefore appear as assumptions or unknowns."
            )
        for source_id, source in sources.items():
            source_col, link_col = st.columns([5, 1])
            with source_col:
                st.markdown(f"**{source['title']}**")
                metadata = " · ".join(
                    str(value)
                    for value in (
                        source.get("publisher"),
                        source.get("published_at"),
                        display_name(source.get("source_type", "")),
                    )
                    if value
                )
                st.caption(f"{source_id} · {metadata}")
            with link_col:
                url = _valid_external_url(source.get("url"))
                if url:
                    st.link_button("Open", url, use_container_width=True)
            st.divider()

def _render_result(result: dict[str, object]) -> None:
    st.divider()
    st.markdown(
        """
        <div class="section-kicker success-dot">INVESTMENT COMMITTEE · ANALYSIS COMPLETED</div>
        """,
        unsafe_allow_html=True,
    )
    export_col, _ = st.columns([1, 3])
    with export_col:
        st.download_button(
            "Download Markdown report",
            data=committee_markdown(result),
            file_name="investment_committee_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with st.container(border=True):
        st.markdown(
            """
            <div class="stage-heading">
              <span>01</span><div><div class="card-label">INDEPENDENT ROUND</div>
              <h2>Investor perspectives & mental models</h2>
              <p>Each investor reasons from the same evidence through a separately retrieved mental-model set.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_investors(result)

    st.markdown('<div class="stage-divider"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            """
            <div class="stage-heading">
              <span>02</span><div><div class="card-label">SECOND ROUND</div>
              <h2>Peer review</h2>
              <p>Investors challenge factual interpretation, model applicability, and weighting before the final synthesis.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_peer_review(result)

    st.markdown('<div class="stage-divider"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            """
            <div class="stage-heading">
              <span>03</span><div><div class="card-label">FINAL SYNTHESIS</div>
              <h2>CIO decision</h2>
              <p>The CIO weighs the independent views and peer challenges against the stated investment horizon.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_decision(result)

    st.markdown('<div class="stage-divider compact"></div>', unsafe_allow_html=True)
    st.markdown("## Supporting evidence & sources")
    _render_evidence_and_sources(result)

    with st.expander("Run details and structured output"):
        config = result.get("model_configuration", {})
        st.caption(
            "Embedding identity: " + str(result.get("embedding_identity", "Unknown"))
        )
        st.json(config)
        st.download_button(
            "Download committee JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="investment_committee_result.json",
            mime="application/json",
        )


def main() -> None:
    st.set_page_config(
        page_title="Investment Committee",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()
    _render_header()
    _render_input()
    if "committee_result" not in st.session_state:
        latest_result = load_latest_result()
        if latest_result is not None:
            st.session_state.committee_result = latest_result
    result = st.session_state.get("committee_result")
    if result:
        _render_result(result)


if __name__ == "__main__":
    main()
