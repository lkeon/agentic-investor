"""Professional Streamlit interface for the investment committee MVP."""

from __future__ import annotations

import base64
from html import escape
import json
from pathlib import Path
import sys
import traceback
from urllib.parse import urlparse

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
BRAND_ICON_PATH = APP_DIR / "assets" / "the-diligence-room-icon.png"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from committee_service import (
    CommitteeRunError,
    CommitteeStopped,
    InvestorDiscoveryError,
    ProgressUpdate,
    discover_investors,
    display_name,
    load_latest_result,
    run_committee,
    stop_active_committee,
)
from exports import committee_markdown


QUESTION_PLACEHOLDER = (
    "Is Berkshire Hathaway still a forever compounder at today’s price—or am "
    "I paying tomorrow’s value upfront?"
)
DECISION_LABELS = {
    "buy": "Buy",
    "buy_below_price": "Buy below price",
    "watchlist": "Watchlist",
    "hold": "Hold",
    "sell": "Sell",
    "avoid": "Avoid",
    "insufficient_information": "Insufficient information",
}


@st.cache_data(show_spinner=False, ttl=60)
def _available_investors() -> list[str]:
    return discover_investors()


@st.cache_data(show_spinner=False)
def _brand_icon_data_uri() -> str:
    """Return the local brand icon as an embeddable image source."""

    encoded = base64.b64encode(BRAND_ICON_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _inject_styles() -> None:
    styles = (APP_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)


def _safe(value: object) -> str:
    # Streamlit still processes Markdown maths inside unsafe HTML blocks.
    # Encode currency markers as HTML entities so financial prose such as
    # "US$176B" remains literal text rather than being rendered as LaTex.
    return escape(str(value or "")).replace("$", "&#36;")


def _literal_markdown_label(value: object) -> str:
    """Escape a Streamlit Markdown label without allowing inline maths."""

    return escape(str(value or "")).replace("$", r"\$")


def _render_customer_progress(
    target: object,
    updates: list[ProgressUpdate],
) -> None:
    """Render a persistent, plain-language timeline for end customers."""

    steps: list[str] = []
    for index, update in enumerate(updates, start=1):
        is_active = index == len(updates) and update.progress < 100
        state = "active" if is_active else "complete"
        marker = f"{index:02d}" if is_active else "✓"
        steps.append(
            f'<div class="customer-progress-step {state}">'
            f'<div class="customer-progress-marker">{marker}</div>'
            "<div>"
            f"<strong>{_safe(update.label)}</strong>"
            f"<p>{_safe(update.detail)}</p>"
            "</div>"
            "</div>"
        )
    target.markdown(
        '<div class="customer-progress">'
        '<div class="customer-progress-heading">DILIGENCE PROGRESS</div>'
        f"{''.join(steps)}"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_technical_log(
    target: object,
    logs: list[str],
    *,
    error: bool = False,
) -> None:
    """Render raw diagnostics in a compact, collapsible terminal view."""

    content = "\n".join(logs) if logs else "Waiting for technical output…"
    # Terminal progress output may use carriage returns to redraw a line.
    # Convert those control characters so the browser displays each update on
    # its own line instead of collapsing the log into an illegible row.
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    line_count = len(content.splitlines())
    label = f"Technical execution log · {line_count:,} lines retained"
    log_panel = target.expander(label, expanded=False)
    caption = (
        "Run failed · raw CrewAI, model, validation, database, and Python "
        "diagnostics."
        if error
        else (
            "Raw CrewAI, model, validation, database, and Python diagnostics."
        )
    )
    log_panel.caption(caption)
    # Streamlit's native code viewer preserves literal newlines and monospace
    # terminal spacing more reliably than Markdown containing a raw <pre>.
    log_panel.code(
        content,
        language=None,
        wrap_lines=True,
        height=448,
    )


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


def _plain_prose_html(value: object, *, class_name: str = "plain-prose") -> str:
    """Render model prose literally so currency markers cannot become LaTeX."""

    content = _safe(value).replace("\n", "<br>")
    return f'<div class="{class_name}">{content}</div>'


def _evidence_cards_html(
    result: dict[str, object],
    evidence_ids: list[str],
    *,
    heading: str | None,
    tone: str = "supporting",
) -> str:
    """Join cited claim and source details directly to one reasoning point."""

    unique_ids = list(dict.fromkeys(evidence_ids))
    if not unique_ids:
        return ""
    claims = _claim_catalogue(result)
    sources = _source_catalogue(result)
    cards: list[str] = []

    for evidence_id in unique_ids:
        claim = claims.get(evidence_id)
        if claim is None:
            cards.append(
                '<div class="inline-evidence-card unavailable">'
                f"<strong>{_safe(evidence_id)}</strong>"
                "<p>Referenced evidence is unavailable in the validated record.</p>"
                "</div>"
            )
            continue

        metadata = [
            display_name(str(claim.get("section", ""))),
            display_name(str(claim.get("status", ""))),
            f"{round(float(claim.get('confidence', 0)) * 100)}% confidence",
        ]
        value = claim.get("value")
        if value is not None:
            unit = f" {claim.get('unit')}" if claim.get("unit") else ""
            metadata.append(f"{value}{unit}")
        if claim.get("period"):
            metadata.append(str(claim["period"]))

        source_links: list[str] = []
        for source_id in claim.get("source_ids", []):
            source = sources.get(source_id)
            if source is None:
                source_links.append(_safe(source_id))
                continue
            title = _safe(source.get("title", source_id))
            url = _valid_external_url(source.get("url"))
            if url:
                source_links.append(
                    f'<a href="{_safe(url)}" target="_blank" '
                    f'rel="noopener noreferrer">{title}</a>'
                )
            else:
                source_links.append(title)

        source_line = (
            '<div class="inline-evidence-sources"><strong>Sources:</strong> '
            + " · ".join(source_links)
            + "</div>"
            if source_links
            else ""
        )
        cards.append(
            '<div class="inline-evidence-card">'
            '<div class="inline-evidence-topline">'
            f"<code>{_safe(evidence_id)}</code>"
            f"<span>{_safe(' · '.join(metadata))}</span>"
            "</div>"
            f"<p>{_safe(claim.get('statement', ''))}</p>"
            f"{source_line}"
            "</div>"
        )

    heading_html = (
        f'<div class="inline-evidence-heading">{_safe(heading)}</div>'
        if heading
        else ""
    )
    return (
        f'<div class="inline-evidence-group {tone}">'
        f"{heading_html}"
        f"{''.join(cards)}"
        "</div>"
    )


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
    icon_source = _brand_icon_data_uri()
    st.markdown(
        f"""
        <div class="product-header">
          <div class="brand-lockup">
            <div class="brand-mark">
              <img src="{icon_source}" alt="The Diligence Room icon">
            </div>
            <div class="brand-name">The Diligence Room</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_input() -> object:
    st.markdown(
        """
        <div class="intro-block">
          <h1>Test investment theses through the established mental models of renowned investors</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    try:
        available = _available_investors()
    except InvestorDiscoveryError as error:
        st.error("The mental-model library is unavailable.")
        st.caption(str(error))
        st.stop()

    with st.form("committee_form", border=False):
        st.markdown(
            '<div class="form-heading">Convene the Diligence Room</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="input-heading"><span>01</span><div>'
            "<strong>Investment question</strong>"
            "<p>Frame the company, decision, and valuation issue you want to test.</p>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        question = st.text_area(
            "Investment question",
            key="question_input",
            height=105,
            placeholder=QUESTION_PLACEHOLDER,
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="input-heading"><span>02</span><div>'
            '<strong>Research context <em>Optional</em></strong>'
            "<p>Paste the evidence and investment brief you want the Room to examine.</p>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        research_context = st.text_area(
            "Research context (optional)",
            height=175,
            placeholder=(
                "Business quality, financial performance, valuation, risks, "
                "material macro factors, and source notes…"
            ),
            label_visibility="collapsed",
        )
        with st.expander(
            "What should I include in the research context?",
            expanded=False,
        ):
            st.markdown(
                """
                **MicroView — company and security evidence**

                - Business model and competitive position
                - Management and capital allocation
                - Financial performance, balance sheet, and liquidity
                - Current valuation, material risks, and potential catalysts

                **MacroView — only direct company transmission**

                - Interest rates, regulation, currencies, industry cycles, or
                  financing conditions that could materially change the thesis

                Include source titles, publishers, dates, and URLs where
                possible. Distinguish reported facts, estimates, assumptions,
                and unknowns. Plain text is sufficient; the Room converts it
                into validated MicroView and MacroView Pydantic records.
                """
            )

        st.markdown(
            '<div class="input-heading attachment-heading"><span>03</span><div>'
            '<strong>Attach text files <em>Optional</em></strong>'
            "<p>Add supporting TXT, Markdown, JSON, or CSV research files.</p>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Attach text files",
            type=["txt", "md", "json", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        with st.container(key="diligence_info"):
            with st.expander("How the Diligence Room works", expanded=False):
                st.markdown(
                    """
                    **Bring your investment idea and supporting research.**

                    The Diligence Room provides structured mental-model
                    reasoning around the thesis you supply. The mental-model
                    library is derived from renowned investors' public writings
                    and engagements and connected in a hierarchical network.

                    Selected investor perspectives use the relevant models as
                    reasoning guardrails—not as substitutes for evidence—and
                    reason independently before the CIO produces the final
                    synthesis. Missing current facts remain explicit
                    uncertainties.
                    """
                )
        with st.expander("Diligence Room Settings", expanded=False):
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
            )
            st.caption(
                f"All {len(available)} available investor perspectives are "
                "shown. Select one or more."
            )
            settings_left, settings_right = st.columns(2)
            with settings_left:
                top_k = st.slider(
                    "Direct canonical models per analytical bridge",
                    min_value=1,
                    max_value=6,
                    value=3,
                )
            with settings_right:
                neighbours = st.slider(
                    "Related network models per analytical bridge",
                    min_value=0,
                    max_value=3,
                    value=1,
                )
            show_logs = st.checkbox(
                "Show technical execution log",
                value=False,
            )

        _, actions_col, _ = st.columns([1, 2, 1])
        with actions_col:
            run_col, stop_col = st.columns(2)
            with run_col:
                submitted = st.form_submit_button(
                    "Begin Diligence",
                    type="primary",
                    use_container_width=True,
                )
            with stop_col:
                stop_requested = st.form_submit_button(
                    "Stop Diligence",
                    use_container_width=True,
                )

    # This placeholder is declared immediately after the action buttons so
    # both customer progress and optional technical output stay anchored there.
    feedback_slot = st.empty()
    st.caption(
        "Decision-support research, not personalised financial advice. "
        "Verify current facts before committing capital."
    )
    # Every committee result is rendered through this stable placeholder. It
    # lets a new run remove the previous result immediately instead of leaving
    # Streamlit's greyed-out stale elements visible during model execution.
    result_slot = st.empty()

    if stop_requested:
        if stop_active_committee():
            st.warning("Stopping the active diligence run…")
        else:
            st.info("No diligence run is currently active.")
        return result_slot

    if not submitted:
        saved_updates = st.session_state.get("latest_customer_progress", [])
        saved_logs = st.session_state.get("latest_technical_logs", [])
        if saved_updates or (show_logs and saved_logs):
            with feedback_slot.container():
                with st.container(border=True, key="execution_feedback"):
                    if saved_updates:
                        _render_customer_progress(st, saved_updates)
                    if show_logs:
                        _render_technical_log(st, saved_logs)
        return result_slot
    if not question.strip():
        st.error("Enter an investment question before beginning diligence.")
        return result_slot
    if not investors:
        st.error("Select at least one investor perspective.")
        return result_slot

    context_parts = [research_context.strip()] if research_context.strip() else []
    for uploaded in uploaded_files:
        try:
            context_parts.append(uploaded.getvalue().decode("utf-8"))
        except UnicodeDecodeError:
            st.error(f"{uploaded.name} must be UTF-8 encoded.")
            return result_slot
    combined_context = "\n\n".join(context_parts) or None
    st.session_state.latest_research_context = combined_context
    st.session_state.latest_run_options = {
        "investors": list(investors),
        "direct_models_per_question": top_k,
        "related_models_per_question": neighbours,
        "technical_debug": show_logs,
    }

    stage_history: list[ProgressUpdate] = []
    latest_logs: list[str] = []
    st.session_state.latest_customer_progress = []
    st.session_state.latest_technical_logs = []
    st.session_state.pop("committee_result", None)
    result_slot.empty()

    with feedback_slot.container():
        with st.container(border=True, key="execution_feedback"):
            progress_bar = st.progress(2)
            customer_view = st.empty()
            technical_view = st.empty()
            feedback_message = st.empty()

        def remember_progress(update: ProgressUpdate) -> None:
            if stage_history and stage_history[-1].label == update.label:
                stage_history[-1] = update
            else:
                stage_history.append(update)

        def on_progress(update: ProgressUpdate, logs: list[str]) -> None:
            progress_bar.progress(update.progress)
            remember_progress(update)
            latest_logs[:] = logs
            _render_customer_progress(customer_view, stage_history)
            if show_logs:
                _render_technical_log(technical_view, logs)

        try:
            result = run_committee(
                question,
                investors=investors,
                research_context=combined_context,
                top_k=top_k,
                neighbours=neighbours,
                technical_debug=show_logs,
                on_progress=on_progress,
            )
        except CommitteeStopped as error:
            latest_logs[:] = error.logs
            st.session_state.latest_customer_progress = stage_history.copy()
            st.session_state.latest_technical_logs = latest_logs.copy()
            _render_technical_log(
                technical_view,
                latest_logs,
                error=True,
            )
            feedback_message.warning(
                "Diligence was stopped. No partial committee result is displayed."
            )
            return result_slot
        except CommitteeRunError as error:
            latest_logs[:] = error.logs
            st.session_state.latest_customer_progress = stage_history.copy()
            st.session_state.latest_technical_logs = latest_logs.copy()
            _render_technical_log(
                technical_view,
                latest_logs,
                error=True,
            )
            feedback_message.error(str(error))
            return result_slot
        except Exception as error:
            latest_logs.extend(
                [
                    "",
                    "FRONTEND EXCEPTION",
                    *traceback.format_exc().splitlines(),
                ]
            )
            st.session_state.latest_customer_progress = stage_history.copy()
            st.session_state.latest_technical_logs = latest_logs.copy()
            _render_technical_log(
                technical_view,
                latest_logs,
                error=True,
            )
            feedback_message.error(f"{type(error).__name__}: {error}")
            return result_slot

        if not stage_history or stage_history[-1].progress < 100:
            remember_progress(
                ProgressUpdate(
                    "Diligence complete",
                    100,
                    "The validated decision record is ready for review.",
                )
            )
        progress_bar.progress(100)
        _render_customer_progress(customer_view, stage_history)
        feedback_message.success("Diligence completed successfully.")
        st.session_state.latest_customer_progress = stage_history.copy()
        st.session_state.latest_technical_logs = latest_logs.copy()
        st.session_state.committee_result = result
    return result_slot


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
            <div class="card-label">CONDITIONS FOR CONTINUED OWNERSHIP</div>
            {_bullet_html(conditions, 'No explicit decision conditions returned.')}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Committee synthesis")
    st.markdown(
        _plain_prose_html(cio["committee_synthesis"]),
        unsafe_allow_html=True,
    )

    applicable_models = cio.get("decisive_mental_model_codes", [])
    if applicable_models:
        model_catalogue = _committee_model_catalogue(result)
        model_rows: list[str] = []
        for code in applicable_models:
            model = model_catalogue.get(code, {})
            investor_name = display_name(str(model.get("investor_id", "")))
            attribution = (
                f"{investor_name} perspective" if investor_name else "Investor view"
            )
            model_rows.append(
                '<div class="applicable-model-row">'
                "<div>"
                f"<strong>{_safe(model.get('title', code))}</strong>"
                f"<span>{_safe(attribution)}</span>"
                "</div>"
                f"<p>{_safe(model.get('proposition', ''))}</p>"
                f"<code>{_safe(code)}</code>"
                "</div>"
            )
        st.markdown("#### Applicable investor mental models")
        st.markdown(
            '<div class="applicable-models">'
            + "".join(model_rows)
            + "</div>",
            unsafe_allow_html=True,
        )

    decisive_evidence = cio.get("decisive_evidence_claim_ids", [])
    if decisive_evidence:
        with st.expander(
            "Evidence informing the CIO decision",
            expanded=False,
        ):
            st.markdown(
                _evidence_cards_html(
                    result,
                    decisive_evidence,
                    heading=None,
                ),
                unsafe_allow_html=True,
            )

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


def _render_investor(
    result: dict[str, object],
    investor_id: str,
) -> None:
    output = result.get("investor_outputs", {}).get(investor_id)
    if not output:
        st.info("No completed reasoning output is available for this investor.")
        return

    confidence = round(float(output["confidence"]) * 100)
    st.markdown(
        f"""
        <div class="perspective-header">
          <div>
            <div class="card-label">INDEPENDENT VIEW</div>
            <h3>{_safe(display_name(investor_id))} perspective</h3>
          </div>
          <div class="stance-block">
            <span class="stance stance-{_safe(output['stance'])}">{_safe(display_name(output['stance']))}</span>
            <span class="confidence-inline">{confidence}% confidence</span>
          </div>
        </div>
        <div class="investor-section-label">INVESTMENT VIEW</div>
        <p class="thesis">{_safe(output['thesis'])}</p>
        """,
        unsafe_allow_html=True,
    )

    catalogue = _model_catalogue(result, investor_id)
    st.markdown("#### Reasoning")
    for number, inference in enumerate(output.get("mental_model_inferences", []), start=1):
        applicability = display_name(inference.get("applicability", "uncertain"))
        with st.expander(
            f"{number}. {_literal_markdown_label(inference['conclusion'])}"
        ):
            st.caption(f"Applicability · {applicability}")
            model_names = [
                catalogue.get(code, {}).get("title", code)
                for code in inference.get("mental_model_codes", [])
            ]
            if model_names:
                st.markdown(
                    '<div class="reasoning-models">'
                    '<div class="inline-evidence-heading">Applicable mental models</div>'
                    f'<div class="chip-row">{_chips(model_names, tone="model")}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                _plain_prose_html(inference["reasoning"]),
                unsafe_allow_html=True,
            )
            evidence_ids = inference.get("evidence_claim_ids", [])
            counter_ids = inference.get("counterevidence_claim_ids", [])
            if evidence_ids:
                st.markdown(
                    _evidence_cards_html(
                        result,
                        evidence_ids,
                        heading="Supporting evidence",
                    ),
                    unsafe_allow_html=True,
                )
            if counter_ids:
                st.markdown(
                    _evidence_cards_html(
                        result,
                        counter_ids,
                        heading="Counterevidence",
                        tone="counter",
                    ),
                    unsafe_allow_html=True,
                )
            if inference.get("missing_information"):
                st.markdown("**Missing information for this conclusion**")
                st.markdown(
                    _bullet_html(
                        inference["missing_information"],
                        "",
                    ),
                    unsafe_allow_html=True,
                )

    st.markdown("#### Continued ownership view")
    st.markdown(
        _plain_prose_html(output["thesis_durability_assessment"]),
        unsafe_allow_html=True,
    )
    if output.get("suitable_while_thesis_valid"):
        st.caption(
            "This perspective supports continued ownership while the core "
            "business thesis remains valid."
        )
    else:
        st.caption(
            "This perspective does not yet support continued ownership; the "
            "identified evidence gaps or conditions must be resolved first."
        )

    st.markdown("#### Risk and monitoring")
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

    with st.expander(
        f"Advanced · Retrieved mental-model set ({len(catalogue)})",
        expanded=False,
    ):
        if not catalogue:
            st.caption("No mental-model candidates were retrieved.")
        for code, model in catalogue.items():
            score = round(float(model.get("retrieval_score", 0)) * 100)
            origin = display_name(model.get("retrieval_origin", "direct"))
            st.markdown(
                '<div class="model-row">'
                '<div class="model-row-top">'
                f"<strong>{_safe(model['title'])}</strong>"
                f"<span>{score}% · {_safe(origin)}</span>"
                "</div>"
                f"<p>{_safe(model['proposition'])}</p>"
                f"<code>{_safe(code)}</code>"
                "</div>",
                unsafe_allow_html=True,
            )


def _render_investors(result: dict[str, object]) -> None:
    investor_ids = list(result.get("investor_reasoning_inputs", {}))
    if not investor_ids:
        st.info("No investor perspectives were returned.")
        return
    with st.container(border=True, key="investor_selector_banner"):
        st.markdown(
            """
            <div class="investor-selector-heading">
              <span>PERSPECTIVE SELECTOR</span>
              <div class="investor-selector-copy">Choose an investor perspective. Compare how each investor applies a distinct mental-model set to the same evidence.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_investor = st.pills(
            "Choose an investor perspective",
            options=investor_ids,
            selection_mode="single",
            default=investor_ids[0],
            required=True,
            format_func=lambda investor_id: f"👤\n{display_name(investor_id)}",
            width="stretch",
            key="result_investor_selector",
            label_visibility="collapsed",
        )
    if selected_investor:
        with st.container(border=True, key="selected_investor_perspective"):
            _render_investor(result, selected_investor)


def _valid_external_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _render_structured_debug(result: dict[str, object]) -> None:
    """Expose retained stage data through one compact debug selector."""

    question = result.get("question", {})
    micro_view = result.get("micro_view", {})
    macro_view = result.get("macro_view", {})
    investor_inputs = result.get("investor_reasoning_inputs", {})
    investor_outputs = result.get("investor_outputs", {})
    research_context = st.session_state.get("latest_research_context")

    st.caption(
        "Validated JSON retained by the MVP, grouped by its corresponding "
        "Pydantic reasoning boundary. Raw provider and CrewAI messages remain "
        "available in the technical execution log."
    )

    retrieved_bridges = {
        investor_id: data.get("mental_model_bridges", [])
        for investor_id, data in investor_inputs.items()
    }
    stages: list[dict[str, object]] = [
        {
            "title": "01 · Question normalisation",
            "input_name": "QuestionNormalizerInput",
            "input": {
                "original_question": question.get("original_question"),
                "current_date": question.get("as_of_date"),
            },
            "output_name": "InvestmentQuestion",
            "output": question,
        },
        {
            "title": "02 · Company evidence research",
            "input_name": "MicroResearchInput",
            "input": {
                "question": question,
                "research_context": (
                    research_context
                    if research_context is not None
                    else "Not retained when loading an earlier artifact."
                ),
            },
            "output_name": "MicroView",
            "output": micro_view,
        },
        {
            "title": "03 · Material macro research",
            "input_name": "MacroResearchInput",
            "input": {
                "question": question,
                "micro_view": micro_view,
                "research_context": (
                    research_context
                    if research_context is not None
                    else "Not retained when loading an earlier artifact."
                ),
            },
            "output_name": "MacroView",
            "output": macro_view,
        },
        {
            "title": "04 · Mental-model bridges and retrieval",
            "input_name": "BridgeAndRetrievalInput",
            "input": {
                "question": question,
                "micro_view": micro_view,
                "macro_view": macro_view,
                "retrieval_options": st.session_state.get(
                    "latest_run_options",
                    "Not retained when loading an earlier artifact.",
                ),
            },
            "output_name": "RetrievedMentalModelBridgesByInvestor",
            "output": retrieved_bridges,
        },
    ]
    for investor_id, investor_input in investor_inputs.items():
        stages.append(
            {
                "title": (
                    f"05 · {display_name(investor_id)} investor reasoning"
                ),
                "input_name": "OneInvestorReasoningInput",
                "input": investor_input,
                "output_name": "InvestorReasoningOutput",
                "output": investor_outputs.get(investor_id, {}),
            }
        )
    stages.extend(
        [
            {
                "title": "06 · CIO synthesis",
                "input_name": "CIOReasoningInput",
                "input": {
                    "question": question,
                    "micro_view": micro_view,
                    "macro_view": macro_view,
                    "investor_outputs": investor_outputs,
                },
                "output_name": "CIOReasoningOutput",
                "output": result.get("cio", {}),
            },
            {
                "title": "Complete · InvestmentCommitteeOutput",
                "input_name": "Validated stage outputs",
                "input": {
                    "model_configuration": result.get(
                        "model_configuration",
                        {},
                    ),
                    "embedding_identity": result.get("embedding_identity"),
                },
                "output_name": "InvestmentCommitteeOutput",
                "output": result,
            },
        ]
    )

    selected_stage = st.selectbox(
        "Inspect a structured stage",
        options=range(len(stages)),
        format_func=lambda index: str(stages[index]["title"]),
        key="structured_debug_stage",
    )
    stage = stages[selected_stage]
    st.markdown(f"**Input · `{stage['input_name']}`**")
    st.json(stage["input"], expanded=2)
    st.markdown(f"**Output · `{stage['output_name']}`**")
    st.json(stage["output"], expanded=2)


def _render_reasoning_chain(result: dict[str, object]) -> None:
    """Show how evidence and canonical mental models become a CIO decision."""

    investor_inputs = result.get("investor_reasoning_inputs", {})
    investor_outputs = result.get("investor_outputs", {})
    bridge_ids = {
        bridge["bridge_id"]
        for investor_input in investor_inputs.values()
        for bridge in investor_input.get("mental_model_bridges", [])
    }
    retrieved_codes = {
        candidate["canonical_code"]
        for investor_input in investor_inputs.values()
        for bridge in investor_input.get("mental_model_bridges", [])
        for candidate in bridge.get("mental_model_candidates", [])
    }
    applied_codes = {
        code
        for output in investor_outputs.values()
        for inference in output.get("mental_model_inferences", [])
        for code in inference.get("mental_model_codes", [])
    }
    inference_count = sum(
        len(output.get("mental_model_inferences", []))
        for output in investor_outputs.values()
    )
    steps = (
        ("1", len(_claim_catalogue(result)), "Evidence claims"),
        ("2", len(bridge_ids), "Analytical bridges"),
        ("3", len(retrieved_codes), "Retrieved mental models"),
        ("4", len(applied_codes), "Applied mental models"),
        ("5", inference_count, "Investor inferences"),
        ("6", 1 if result.get("cio") else 0, "CIO decision"),
    )
    step_html = "".join(
        '<div class="reasoning-flow-step">'
        f'<span class="reasoning-flow-number">{number}</span>'
        f"<strong>{count}</strong>"
        f"<p>{_safe(label)}</p>"
        "</div>"
        for number, count, label in steps
    )

    with st.expander(
        "Reasoning Chain · How evidence becomes a decision",
        expanded=False,
    ):
        st.markdown(
            '<div class="reasoning-provenance">'
            "<strong>PUBLIC INVESTOR KNOWLEDGE</strong>"
            "<p>Public engagements → Mental-model fragments → Canonical mental "
            "models → Hierarchical mental-model network</p>"
            "</div>"
            f'<div class="reasoning-flow">{step_html}</div>'
            '<div class="reasoning-flow-note">Analytical bridges connect the '
            "supplied evidence to relevant canonical mental models. Investor "
            "perspectives apply those mental models as reasoning guardrails "
            "before the CIO produces the final synthesis.</div>",
            unsafe_allow_html=True,
        )


def _render_result(result: dict[str, object]) -> None:
    st.divider()
    st.markdown(
        """
        <div class="section-kicker success-dot">INVESTMENT COMMITTEE · ANALYSIS COMPLETED</div>
        """,
        unsafe_allow_html=True,
    )

    _render_reasoning_chain(result)

    with st.container(border=True):
        st.markdown(
            """
            <div class="stage-heading">
              <span>01</span><div><div class="card-label">INDEPENDENT ANALYSIS</div>
              <h2>Investor Perspectives &amp; Mental Models</h2>
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
              <span>02</span><div><div class="card-label">FINAL SYNTHESIS</div>
              <h2>CIO Decision</h2>
              <p>The CIO compares the independent views and decides whether the thesis supports continued ownership, with macro treated as a secondary condition.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_decision(result)

    st.markdown('<div class="stage-divider compact"></div>', unsafe_allow_html=True)
    with st.expander("Run configuration and downloads"):
        config = result.get("model_configuration", {})
        st.caption(
            "Embedding identity: " + str(result.get("embedding_identity", "Unknown"))
        )
        st.json(config)
        markdown_download, json_download = st.columns(2)
        with markdown_download:
            st.download_button(
                "Download Markdown report",
                data=committee_markdown(result),
                file_name="investment_committee_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with json_download:
            st.download_button(
                "Download committee JSON",
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name="investment_committee_result.json",
                mime="application/json",
                use_container_width=True,
            )

    with st.expander("Structured inputs & outputs · Debugging"):
        _render_structured_debug(result)


def main() -> None:
    st.set_page_config(
        page_title="The Diligence Room",
        page_icon=BRAND_ICON_PATH,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()
    _render_header()
    result_slot = _render_input()
    if "latest_result_checked" not in st.session_state:
        st.session_state.latest_result_checked = True
        latest_result = load_latest_result()
        if (
            "committee_result" not in st.session_state
            and latest_result is not None
        ):
            st.session_state.committee_result = latest_result
    result = st.session_state.get("committee_result")
    if result:
        with result_slot.container():
            _render_result(result)


if __name__ == "__main__":
    main()
