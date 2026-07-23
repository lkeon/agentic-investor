"""Interactive 3D visualisation of canonical mental models and graph links."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from vis.data import load_graph_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CREW_RESULT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "crew"
    / "committee_result.json"
)


@st.cache_data(show_spinner="Loading MMC graph from PostgreSQL...")
def _load_and_project() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load graph data and project all embeddings into a stable 3D PCA space."""

    nodes, edges = load_graph_data()

    if len(nodes) < 3:
        raise RuntimeError(
            "At least three embedded canonical models are required."
        )

    embeddings = np.asarray(
        [node.pop("embedding") for node in nodes],
        dtype=np.float64,
    )

    coordinates = PCA(
        n_components=3,
        svd_solver="randomized",
        random_state=42,
    ).fit_transform(embeddings)

    node_frame = pd.DataFrame(nodes)
    node_frame["x"] = coordinates[:, 0]
    node_frame["y"] = coordinates[:, 1]
    node_frame["z"] = coordinates[:, 2]
    node_frame["node_size"] = 5.0 + 14.0 * node_frame["base_weight"]

    edge_frame = pd.DataFrame(edges)

    return node_frame, edge_frame


def _filter_nodes(
    nodes: pd.DataFrame,
    *,
    investors: list[str],
    domains: list[str],
    minimum_weight: float,
    maximum_nodes: int,
) -> pd.DataFrame:
    selected = nodes[
        nodes["investor_id"].isin(investors)
        & nodes["primary_domain"].isin(domains)
        & (nodes["base_weight"] >= minimum_weight)
    ].copy()

    selected = selected.sort_values(
        ["base_weight", "evidence_confidence"],
        ascending=False,
    )

    return selected.head(maximum_nodes)


def _filter_edges(
    edges: pd.DataFrame,
    selected_nodes: pd.DataFrame,
    *,
    relation_types: list[str],
    minimum_strength: float,
    minimum_confidence: float,
    maximum_edges: int,
) -> pd.DataFrame:
    if edges.empty or selected_nodes.empty or not relation_types:
        return edges.iloc[0:0].copy()

    selected_ids = set(selected_nodes["canonical_id"])

    selected = edges[
        edges["source_canonical_id"].isin(selected_ids)
        & edges["target_canonical_id"].isin(selected_ids)
        & edges["relation_type"].isin(relation_types)
        & (edges["relation_strength"] >= minimum_strength)
        & (edges["relation_confidence"] >= minimum_confidence)
    ].copy()

    similarity = selected["candidate_similarity"].fillna(0.0)
    selected["edge_rank"] = (
        0.45 * selected["relation_confidence"]
        + 0.35 * selected["relation_strength"]
        + 0.20 * similarity
    )

    return selected.sort_values(
        "edge_rank",
        ascending=False,
    ).head(maximum_edges)


def _edge_traces(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> list[go.Scatter3d]:
    """Create one 3D line trace per relationship type."""

    if edges.empty:
        return []

    coordinates = nodes.set_index("canonical_id")[["x", "y", "z"]]
    palette = px.colors.qualitative.Plotly
    traces: list[go.Scatter3d] = []

    for number, (relation_type, group) in enumerate(
        edges.groupby("relation_type", sort=True)
    ):
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        z_values: list[float | None] = []

        for edge in group.itertuples(index=False):
            source = coordinates.loc[edge.source_canonical_id]
            target = coordinates.loc[edge.target_canonical_id]

            x_values.extend([source.x, target.x, None])
            y_values.extend([source.y, target.y, None])
            z_values.extend([source.z, target.z, None])

        traces.append(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines",
                name=relation_type,
                legendgroup=f"edge-{relation_type}",
                line={
                    "width": 2,
                    "color": palette[number % len(palette)],
                },
                opacity=0.35,
                hoverinfo="skip",
            )
        )

    return traces


def _graph_figure(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    colour_by: str,
) -> go.Figure:
    """Create the interactive Plotly 3D network figure."""

    figure = go.Figure()

    for trace in _edge_traces(nodes, edges):
        figure.add_trace(trace)

    node_figure = px.scatter_3d(
        nodes,
        x="x",
        y="y",
        z="z",
        color=colour_by,
        size="node_size",
        size_max=18,
        custom_data=[
            "canonical_code",
            "investor_id",
            "primary_domain",
            "concept_family",
            "base_weight",
        ],
        hover_name="title",
        hover_data={
            "x": False,
            "y": False,
            "z": False,
            "node_size": False,
            "proposition": True,
            "base_weight": ":.3f",
            "evidence_confidence": ":.3f",
            "investor_importance": ":.3f",
        },
        opacity=0.88,
    )

    for trace in node_figure.data:
        trace.marker.line = {"width": 0.4, "color": "rgba(30,30,30,0.45)"}
        figure.add_trace(trace)

    figure.update_layout(
        height=760,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
        },
        scene={
            "xaxis_title": "PCA 1",
            "yaxis_title": "PCA 2",
            "zaxis_title": "PCA 3",
            "aspectmode": "data",
        },
        uirevision="mmc-graph",
    )

    return figure


def _write_list(label: str, values: list[str]) -> None:
    st.markdown(f"**{label}**")
    if values:
        for value in values:
            st.write(f"- {value}")
    else:
        st.caption("None")


def _show_model_details(nodes: pd.DataFrame) -> None:
    st.subheader("MMC details")

    labels = {
        row.canonical_code: f"{row.investor_id} — {row.title}"
        for row in nodes.itertuples(index=False)
    }

    selected_code = st.selectbox(
        "Select a canonical model",
        options=list(labels),
        format_func=lambda code: labels[code],
    )

    record = nodes.loc[
        nodes["canonical_code"] == selected_code
    ].iloc[0]

    st.code(record["canonical_code"])
    st.markdown(f"### {record['title']}")
    st.write(record["proposition"])

    left, right = st.columns(2)

    with left:
        st.write(f"Investor: **{record['investor_id']}**")
        st.write(f"Kind: **{record['kind']}**")
        st.write(f"Primary domain: **{record['primary_domain']}**")
        st.write(
            "Concept family: "
            f"**{record['concept_family'] or 'unassigned'}**"
        )

    with right:
        st.metric("Base weight", f"{record['base_weight']:.3f}")
        st.metric(
            "Evidence confidence",
            f"{record['evidence_confidence']:.3f}",
        )
        st.metric(
            "Investor importance",
            f"{record['investor_importance']:.3f}",
        )

    _write_list("Mechanism", record["mechanism"])
    _write_list("Conditions", record["conditions"])
    _write_list("Failure conditions", record["failure_conditions"])
    _write_list(
        "Decision implications",
        record["decision_implications"],
    )
    _write_list("Decision stages", record["decision_stages"])

    st.caption(
        "Supporting fragments: "
        + ", ".join(record["supporting_fragment_codes"])
    )


def _crew_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def _show_crew_tab() -> None:
    st.subheader("Crew result")

    path_text = st.text_input(
        "Committee result JSON",
        value=str(DEFAULT_CREW_RESULT),
    )
    path = Path(path_text).expanduser()

    result = _crew_result(path)

    if result is None:
        st.info(
            "No crew result found at this path. Run the crew first or "
            "select another JSON file."
        )
        return

    question = result.get("question", {})
    st.markdown("### Question")
    st.write(question.get("original_question", ""))
    st.caption(question.get("retrieval_query", ""))

    retrieved = result.get("retrieved_models", {})
    counts = pd.DataFrame(
        [
            {
                "investor": investor_id,
                "retrieved_mmc": len(models),
            }
            for investor_id, models in retrieved.items()
        ]
    )
    if not counts.empty:
        st.dataframe(counts, hide_index=True, width="stretch")

    st.markdown("### Round 1")
    for investor_id, view in result.get("round_one", {}).items():
        with st.expander(
            f"{investor_id}: {view.get('stance', 'unknown')}",
            expanded=False,
        ):
            st.write(view.get("thesis", ""))
            st.caption(
                f"Confidence: {view.get('confidence', 0.0):.2f}"
            )
            _write_list(
                "Mental models used",
                view.get("mental_models_used", []),
            )
            _write_list(
                "Key risks",
                view.get("key_risks", []),
            )

    st.markdown("### Round 2")
    for investor_id, view in result.get("round_two", {}).items():
        with st.expander(
            f"{investor_id}: {view.get('final_stance', 'unknown')}",
            expanded=False,
        ):
            st.write(view.get("updated_thesis", ""))
            st.caption(
                f"Changed view: {view.get('changed_view', False)} | "
                f"Confidence: {view.get('confidence', 0.0):.2f}"
            )

            for comment in view.get("peer_comments", []):
                st.markdown(
                    f"**Comment on "
                    f"{comment.get('peer_investor_id', 'peer')}**"
                )
                _write_list(
                    "Agreements",
                    comment.get("agreements", []),
                )
                _write_list(
                    "Disagreements",
                    comment.get("disagreements", []),
                )


def main() -> None:
    st.set_page_config(
        page_title="MMC Visualisation",
        layout="wide",
    )

    st.title("Canonical mental-model visualisation")
    st.caption(
        "PCA projects semantic embeddings into three dimensions. "
        "Lines are stored canonical-model relationships."
    )

    try:
        all_nodes, all_edges = _load_and_project()
    except Exception as error:
        st.error(f"Could not load the MMC graph: {error}")
        st.stop()

    graph_tab, crew_tab = st.tabs(
        ["MMC network", "Crew result"]
    )

    with graph_tab:
        if st.sidebar.button("Reload database"):
            st.cache_data.clear()
            st.rerun()

        investors = sorted(
            all_nodes["investor_id"].dropna().unique()
        )
        domains = sorted(
            all_nodes["primary_domain"].dropna().unique()
        )
        relation_types = (
            sorted(all_edges["relation_type"].dropna().unique())
            if not all_edges.empty
            else []
        )

        st.sidebar.header("MMC filters")

        selected_investors = st.sidebar.multiselect(
            "Investors",
            options=investors,
            default=investors,
        )
        selected_domains = st.sidebar.multiselect(
            "Primary domains",
            options=domains,
            default=domains,
        )
        colour_by = st.sidebar.radio(
            "Colour nodes by",
            options=["investor_id", "primary_domain"],
            horizontal=True,
        )
        minimum_weight = st.sidebar.slider(
            "Minimum base weight",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )
        maximum_nodes = st.sidebar.slider(
            "Maximum displayed nodes",
            min_value=min(100, len(all_nodes)),
            max_value=len(all_nodes),
            value=min(1200, len(all_nodes)),
            step=max(1, len(all_nodes) // 100),
        )

        st.sidebar.header("Link filters")
        show_edges = st.sidebar.checkbox(
            "Show graph links",
            value=True,
        )
        selected_relation_types = st.sidebar.multiselect(
            "Relationship types",
            options=relation_types,
            default=relation_types,
        )
        minimum_strength = st.sidebar.slider(
            "Minimum relationship strength",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
        )
        minimum_confidence = st.sidebar.slider(
            "Minimum relationship confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
        )
        maximum_edges = st.sidebar.slider(
            "Maximum displayed links",
            min_value=0,
            max_value=max(1, len(all_edges)),
            value=min(2500, len(all_edges)),
            step=max(1, len(all_edges) // 100),
        )

        nodes = _filter_nodes(
            all_nodes,
            investors=selected_investors,
            domains=selected_domains,
            minimum_weight=minimum_weight,
            maximum_nodes=maximum_nodes,
        )

        if show_edges:
            edges = _filter_edges(
                all_edges,
                nodes,
                relation_types=selected_relation_types,
                minimum_strength=minimum_strength,
                minimum_confidence=minimum_confidence,
                maximum_edges=maximum_edges,
            )
        else:
            edges = all_edges.iloc[0:0].copy()

        first, second, third, fourth = st.columns(4)
        first.metric("Loaded MMCs", f"{len(all_nodes):,}")
        second.metric("Displayed MMCs", f"{len(nodes):,}")
        third.metric("Stored links", f"{len(all_edges):,}")
        fourth.metric("Displayed links", f"{len(edges):,}")

        if nodes.empty:
            st.warning("The current filters select no canonical models.")
        else:
            figure = _graph_figure(
                nodes,
                edges,
                colour_by=colour_by,
            )
            st.plotly_chart(
                figure,
                width="stretch",
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )
            _show_model_details(nodes)

    with crew_tab:
        _show_crew_tab()


if __name__ == "__main__":
    main()
