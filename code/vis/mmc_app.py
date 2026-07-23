"""Interactive 3D visualisation of canonical models and source fragments."""

from __future__ import annotations

import html
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from vis.data import (
    load_graph_data,
    load_mmf_network_data,
    load_supporting_fragments,
)

try:
    from streamlit_plotly_events2 import plotly_events
except ImportError:
    try:
        from streamlit_plotly_events import plotly_events
    except ImportError:
        plotly_events = None


DETAIL_WIDTH = 900
SECONDARY_SELECTOR_WIDTH = 840
GRAPH_HEIGHT = 760
PLOTLY_GL_PIXEL_RATIO = 1

INVESTOR_COLOURS = {
    "buffett": "#3B82F6",
    "flatt": "#14B8A6",
    "marks": "#F59E0B",
    "munger": "#8B5CF6",
    "pabrai": "#EF4444",
    "spier": "#EC4899",
}

DOMAIN_COLOURS = {
    "mandate_and_objective": "#2563EB",
    "circle_of_competence": "#7C3AED",
    "business_and_asset_quality": "#059669",
    "management_and_governance": "#D97706",
    "financial_resilience": "#0891B2",
    "valuation_and_expected_return": "#DC2626",
    "risk_and_uncertainty": "#9333EA",
    "market_and_cycle_context": "#0D9488",
    "portfolio_construction": "#EA580C",
    "monitoring_and_exit": "#4F46E5",
}

MMF_KIND_COLOURS = {
    "behavioural_rule": "#2563EB",
    "causal_claim": "#0D9488",
    "condition": "#D97706",
    "decision_rule": "#7C3AED",
    "exception": "#DC2626",
    "observation": "#0891B2",
    "portfolio_rule": "#EA580C",
    "principle": "#059669",
    "risk_rule": "#9333EA",
    "valuation_rule": "#DB2777",
}

EVIDENCE_COLOURS = {
    "directly_stated": "#16A34A",
    "strongly_implied": "#2563EB",
    "weakly_inferred": "#D97706",
}

FALLBACK_COLOURS = [
    "#2563EB",
    "#E11D48",
    "#059669",
    "#7C3AED",
    "#D97706",
    "#0891B2",
    "#DB2777",
    "#4F46E5",
    "#0D9488",
    "#EA580C",
    "#9333EA",
    "#65A30D",
    "#C2410C",
    "#0284C7",
    "#A21CAF",
    "#CA8A04",
]


@st.cache_data(show_spinner="Loading MMC graph from PostgreSQL...")
def _load_mmc_projection() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MMCs and project their embeddings into three dimensions."""

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
    node_frame["node_size"] = (
        5.0 + 14.0 * node_frame["base_weight"]
    )

    return node_frame, pd.DataFrame(edges)


@st.cache_data(show_spinner="Loading MMF network from PostgreSQL...")
def _load_mmf_projection() -> pd.DataFrame:
    """Load MMFs and project their embeddings into three dimensions."""

    fragments = load_mmf_network_data()

    if len(fragments) < 3:
        raise RuntimeError(
            "At least three embedded mental-model fragments are required."
        )

    embeddings = np.asarray(
        [fragment.pop("embedding") for fragment in fragments],
        dtype=np.float64,
    )
    coordinates = PCA(
        n_components=3,
        svd_solver="randomized",
        random_state=42,
    ).fit_transform(embeddings)

    frame = pd.DataFrame(fragments)
    frame["x"] = coordinates[:, 0]
    frame["y"] = coordinates[:, 1]
    frame["z"] = coordinates[:, 2]
    frame["point_size"] = 7.0

    return frame


@st.cache_data(show_spinner=False)
def _load_supporting_fragments(
    fragment_codes: tuple[str, ...],
) -> list[dict[str, Any]]:
    return load_supporting_fragments(fragment_codes)



def _categorical_colour_map(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, str]:
    """Return stable, explicit colours for a categorical plot field."""

    categories = sorted(
        str(value)
        for value in frame[column].dropna().unique()
    )

    preferred_maps = {
        "investor_id": INVESTOR_COLOURS,
        "primary_domain": DOMAIN_COLOURS,
        "kind": MMF_KIND_COLOURS,
        "evidence_strength": EVIDENCE_COLOURS,
    }
    preferred = preferred_maps.get(column, {})

    result: dict[str, str] = {}

    for position, category in enumerate(categories):
        result[category] = preferred.get(
            category,
            FALLBACK_COLOURS[
                position % len(FALLBACK_COLOURS)
            ],
        )

    return result



def _wrap_hover_text(
    value: Any,
    *,
    width: int,
    maximum_lines: int | None = None,
) -> str:
    """Escape and wrap text for compact Plotly hover cards."""

    plain_text = " ".join(str(value or "").split())

    if not plain_text:
        return "—"

    lines = textwrap.wrap(
        plain_text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    if maximum_lines and len(lines) > maximum_lines:
        lines = lines[:maximum_lines]
        final_line = lines[-1].rstrip(" .,:;")
        lines[-1] = f"{final_line}…"

    return "<br>".join(
        html.escape(line)
        for line in lines
    )


def _symmetric_range(values: pd.Series) -> list[float]:
    maximum = max(float(values.abs().max()), 0.01)
    return [-maximum, maximum]


def _base_scene(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a scene centred on the PCA origin."""

    return {
        "xaxis": {
            "title": "PCA 1",
            "range": _symmetric_range(frame["x"]),
            "autorange": False,
        },
        "yaxis": {
            "title": "PCA 2",
            "range": _symmetric_range(frame["y"]),
            "autorange": False,
        },
        "zaxis": {
            "title": "PCA 3",
            "range": _symmetric_range(frame["z"]),
            "autorange": False,
        },
        "aspectmode": "cube",
        "dragmode": "orbit",
    }


def _write_list(label: str, values: list[str]) -> None:
    st.markdown(f"**{label}**")

    if values:
        for value in values:
            st.write(f"- {value}")
    else:
        st.caption("None")


def _render_clickable_plot(
    figure: go.Figure,
    *,
    key: str,
) -> list[dict[str, Any]]:
    """
    Render Plotly with true click events.

    Streamlit's built-in Plotly hook only exposes selection events. A
    Scatter3d point click emits a Plotly click event instead, so this app uses
    streamlit-plotly-events2 for reliable 3D clicks.
    """

    config = {
        "displayModeBar": False,
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
        # Plotly defaults to 2, which renders a 4K canvas internally at
        # approximately 8K. Native resolution keeps 3D interaction responsive.
        "plotGlPixelRatio": PLOTLY_GL_PIXEL_RATIO,
    }

    if plotly_events is None:
        st.warning(
            "Point-click support is unavailable. Install "
            "`streamlit-plotly-events2==0.0.7` in the same virtual "
            "environment used to run Streamlit, then restart the app. "
            "The dropdown selectors remain available."
        )
        st.plotly_chart(
            figure,
            width="stretch",
            height="content",
            key=f"{key}_fallback",
            config=config,
        )
        return []

    arguments = {
        "plot_fig": figure,
        "click_event": True,
        "select_event": False,
        "hover_event": False,
        "override_height": GRAPH_HEIGHT,
        "override_width": "100%",
        "key": key,
    }

    try:
        return plotly_events(
            **arguments,
            config=config,
        )
    except TypeError:
        # Compatibility with the older component signature.
        return plotly_events(**arguments)


def _clicked_code(
    events: list[dict[str, Any]],
    figure: go.Figure,
    frame: pd.DataFrame,
    *,
    code_column: str,
) -> str | None:
    """
    Resolve a Plotly click to an MMC or MMF code.

    Prefer Plotly's trace and point indices so the code is read directly from
    the clicked trace's customdata. Coordinate matching remains as a fallback
    for component versions that omit those indices.
    """

    if not events or frame.empty:
        return None

    event = events[-1]
    visible_codes = {
        str(value)
        for value in frame[code_column].dropna()
    }

    # Primary path: identify the exact trace and point that Plotly reported.
    try:
        curve_number = int(event["curveNumber"])
        raw_point_number = event.get("pointNumber")

        if raw_point_number is None:
            raw_point_number = event.get("pointIndex")

        if raw_point_number is None:
            raise KeyError("Point index missing from Plotly click event")

        point_number = int(raw_point_number)
        trace = figure.data[curve_number]
        customdata = trace.customdata

        if customdata is not None:
            point_customdata = customdata[point_number]

            if isinstance(
                point_customdata,
                (list, tuple, np.ndarray),
            ):
                code = str(point_customdata[0])
            else:
                code = str(point_customdata)

            if code in visible_codes:
                return code
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ):
        pass

    # Fallback: match the exact 3D click coordinates to a displayed point.
    try:
        clicked = np.asarray(
            [
                float(event["x"]),
                float(event["y"]),
                float(event["z"]),
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None

    coordinates = frame[["x", "y", "z"]].to_numpy(
        dtype=np.float64
    )
    distances = np.linalg.norm(
        coordinates - clicked,
        axis=1,
    )
    nearest_position = int(np.argmin(distances))

    full_scale = max(
        float(np.ptp(coordinates[:, 0])),
        float(np.ptp(coordinates[:, 1])),
        float(np.ptp(coordinates[:, 2])),
        0.01,
    )

    if float(distances[nearest_position]) > full_scale * 1e-5:
        return None

    return str(
        frame.iloc[nearest_position][code_column]
    )


def _edge_traces(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> list[go.Scatter3d]:
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


def _stratified_random_sample(
    frame: pd.DataFrame,
    *,
    maximum_rows: int,
    group_columns: list[str],
    random_state: int,
) -> pd.DataFrame:
    """
    Return a stable random sample with broad group representation.

    When space permits, at least one randomly chosen record is retained from
    every investor/domain or investor/kind group. Remaining places are sampled
    randomly from the remaining population, so larger groups remain more
    likely to contribute additional points. A fixed seed prevents the graph
    from reshuffling on every Streamlit rerun.
    """

    if maximum_rows <= 0 or frame.empty:
        return frame.iloc[0:0].copy()

    if len(frame) <= maximum_rows:
        return frame.copy()

    rng = np.random.default_rng(random_state)
    groups = [
        group
        for _, group in frame.groupby(
            group_columns,
            dropna=False,
            sort=True,
        )
    ]

    chosen_indices: list[Any] = []

    if maximum_rows >= len(groups):
        for group in groups:
            chosen_indices.append(
                rng.choice(group.index.to_numpy()).item()
            )
    else:
        group_weights = np.asarray(
            [len(group) for group in groups],
            dtype=np.float64,
        )
        group_weights /= group_weights.sum()

        selected_group_positions = rng.choice(
            len(groups),
            size=maximum_rows,
            replace=False,
            p=group_weights,
        )

        for position in selected_group_positions:
            group = groups[int(position)]
            chosen_indices.append(
                rng.choice(group.index.to_numpy()).item()
            )

        rng.shuffle(chosen_indices)
        return frame.loc[chosen_indices].copy()

    remaining_slots = maximum_rows - len(chosen_indices)

    if remaining_slots > 0:
        remaining_indices = frame.index[
            ~frame.index.isin(chosen_indices)
        ].to_numpy()
        extra_indices = rng.choice(
            remaining_indices,
            size=remaining_slots,
            replace=False,
        )
        chosen_indices.extend(extra_indices.tolist())

    rng.shuffle(chosen_indices)
    return frame.loc[chosen_indices].copy()


def _filter_mmc_nodes(
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

    return _stratified_random_sample(
        selected,
        maximum_rows=maximum_nodes,
        group_columns=[
            "investor_id",
            "primary_domain",
        ],
        random_state=20260723,
    )


def _filter_mmf_nodes(
    fragments: pd.DataFrame,
    *,
    investors: list[str],
    kinds: list[str],
    maximum_fragments: int,
) -> pd.DataFrame:
    selected = fragments[
        fragments["investor_id"].isin(investors)
        & fragments["kind"].isin(kinds)
    ].copy()

    return _stratified_random_sample(
        selected,
        maximum_rows=maximum_fragments,
        group_columns=[
            "investor_id",
            "kind",
        ],
        random_state=20260724,
    )


def _filter_edges(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    relation_types: list[str],
    minimum_strength: float,
    minimum_confidence: float,
    maximum_edges: int,
    selected_canonical_id: str | None,
) -> pd.DataFrame:
    if (
        edges.empty
        or nodes.empty
        or not relation_types
        or maximum_edges == 0
    ):
        return edges.iloc[0:0].copy()

    visible_ids = set(nodes["canonical_id"])
    selected = edges[
        edges["source_canonical_id"].isin(visible_ids)
        & edges["target_canonical_id"].isin(visible_ids)
        & edges["relation_type"].isin(relation_types)
        & (edges["relation_strength"] >= minimum_strength)
        & (edges["relation_confidence"] >= minimum_confidence)
    ].copy()

    selected["edge_rank"] = (
        0.45 * selected["relation_confidence"]
        + 0.35 * selected["relation_strength"]
        + 0.20
        * selected["candidate_similarity"].fillna(0.0)
    )
    selected = selected.sort_values(
        "edge_rank",
        ascending=False,
    )

    if not selected_canonical_id:
        return selected.head(maximum_edges)

    boundary_mask = (
        (selected["source_canonical_id"] == selected_canonical_id)
        | (selected["target_canonical_id"] == selected_canonical_id)
    )
    boundary_edges = selected.loc[boundary_mask]
    other_edges = selected.loc[~boundary_mask]

    return pd.concat(
        [
            boundary_edges,
            other_edges.head(
                max(0, maximum_edges - len(boundary_edges))
            ),
        ],
        ignore_index=True,
    ).head(maximum_edges)


def _mmc_figure(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    colour_by: str,
    selected_code: str | None,
    mark_selection: bool,
    view_revision: int,
) -> go.Figure:
    figure = go.Figure()

    selected_id: str | None = None
    boundary_ids: set[str] = set()
    boundary_edges = edges.iloc[0:0].copy()

    if selected_code and mark_selection:
        selected_rows = nodes.loc[
            nodes["canonical_code"] == selected_code
        ]

        if not selected_rows.empty:
            selected_id = str(
                selected_rows.iloc[0]["canonical_id"]
            )

    if selected_id and not edges.empty:
        boundary_mask = (
            (edges["source_canonical_id"] == selected_id)
            | (edges["target_canonical_id"] == selected_id)
        )
        boundary_edges = edges.loc[boundary_mask].copy()

        for edge in boundary_edges.itertuples(index=False):
            if edge.source_canonical_id != selected_id:
                boundary_ids.add(
                    str(edge.source_canonical_id)
                )
            if edge.target_canonical_id != selected_id:
                boundary_ids.add(
                    str(edge.target_canonical_id)
                )

    regular_edges = edges

    if not boundary_edges.empty:
        regular_edges = edges.loc[
            ~edges["edge_id"].isin(
                set(boundary_edges["edge_id"])
            )
        ]

    for trace in _edge_traces(nodes, regular_edges):
        figure.add_trace(trace)

    for trace in _edge_traces(nodes, boundary_edges):
        trace.name = f"Selected: {trace.name}"
        trace.legendgroup = f"selected-{trace.legendgroup}"
        trace.line.width = 3.5
        trace.opacity = 0.78
        figure.add_trace(trace)

    plot_nodes = nodes.copy()
    plot_nodes["_hover_title"] = plot_nodes["title"].map(
        lambda value: _wrap_hover_text(
            value,
            width=42,
            maximum_lines=3,
        )
    )
    plot_nodes["_hover_proposition"] = plot_nodes[
        "proposition"
    ].map(
        lambda value: _wrap_hover_text(
            value,
            width=54,
            maximum_lines=6,
        )
    )
    plot_nodes["_hover_concept_family"] = plot_nodes[
        "concept_family"
    ].map(
        lambda value: _wrap_hover_text(
            value,
            width=48,
            maximum_lines=3,
        )
    )

    node_figure = px.scatter_3d(
        plot_nodes,
        x="x",
        y="y",
        z="z",
        color=colour_by,
        color_discrete_map=_categorical_colour_map(
            plot_nodes,
            colour_by,
        ),
        size="node_size",
        size_max=18,
        custom_data=[
            "canonical_code",
            "investor_id",
            "primary_domain",
            "_hover_concept_family",
            "base_weight",
            "_hover_title",
            "_hover_proposition",
            "evidence_confidence",
            "investor_importance",
        ],
        opacity=0.88,
    )

    for trace in node_figure.data:
        trace.marker.line = {
            "width": 0.8,
            "color": trace.marker.color,
        }
        trace.hovertemplate = (
            "<b>%{customdata[5]}</b><br>"
            "%{customdata[6]}<br><br>"
            "Code: %{customdata[0]}<br>"
            "Investor: %{customdata[1]}<br>"
            "Domain: %{customdata[2]}<br>"
            "Concept family: %{customdata[3]}<br>"
            "Base weight: %{customdata[4]:.3f}<br>"
            "Evidence confidence: %{customdata[7]:.3f}<br>"
            "Investor importance: %{customdata[8]:.3f}"
            "<extra></extra>"
        )
        figure.add_trace(trace)

    if boundary_ids:
        boundary_nodes = nodes.loc[
            nodes["canonical_id"].isin(boundary_ids)
        ]

        figure.add_trace(
            go.Scatter3d(
                x=boundary_nodes["x"],
                y=boundary_nodes["y"],
                z=boundary_nodes["z"],
                mode="markers",
                name="Boundary MMCs",
                customdata=boundary_nodes[
                    [
                        "canonical_code",
                        "investor_id",
                        "primary_domain",
                        "concept_family",
                        "base_weight",
                    ]
                ].to_numpy(),
                text=boundary_nodes["title"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Boundary MMC<br>"
                    "Code: %{customdata[0]}<br>"
                    "Investor: %{customdata[1]}<br>"
                    "Domain: %{customdata[2]}"
                    "<extra></extra>"
                ),
                marker={
                    "size": boundary_nodes["node_size"] + 1.5,
                    "symbol": "circle-open",
                    "color": "rgba(0,0,0,0.01)",
                    "line": {
                        "width": 1.7,
                        "color": "#24324A",
                    },
                },
                opacity=1.0,
                legendgroup="selection",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if selected_id:
        selected_node = nodes.loc[
            nodes["canonical_id"] == selected_id
        ].iloc[0]

        figure.add_trace(
            go.Scatter3d(
                x=[selected_node["x"]],
                y=[selected_node["y"]],
                z=[selected_node["z"]],
                mode="markers",
                name="Selected MMC",
                customdata=[[
                    selected_node["canonical_code"],
                    selected_node["investor_id"],
                    selected_node["primary_domain"],
                    selected_node["concept_family"],
                    selected_node["base_weight"],
                ]],
                text=[selected_node["title"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Selected MMC<br>"
                    "Code: %{customdata[0]}<br>"
                    "Investor: %{customdata[1]}<br>"
                    "Domain: %{customdata[2]}"
                    "<extra></extra>"
                ),
                marker={
                    "size": min(
                        float(selected_node["node_size"]) + 5.0,
                        21.0,
                    ),
                    "symbol": "circle-open",
                    "color": "rgba(0,0,0,0.01)",
                    "line": {
                        "width": 3.2,
                        "color": "#24324A",
                    },
                },
                opacity=1.0,
                legendgroup="selection",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        height=GRAPH_HEIGHT,
        autosize=True,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
        },
        scene=_base_scene(nodes),
        hovermode="closest",
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#CBD5E1",
            "align": "left",
            "namelength": -1,
            "font": {
                "color": "#111827",
                "size": 13,
            },
        },
        clickmode="event",
        uirevision=f"mmc-network-{view_revision}",
    )

    return figure


def _mmf_figure(
    fragments: pd.DataFrame,
    *,
    colour_by: str,
    selected_code: str | None,
    mark_selection: bool,
    view_revision: int,
) -> go.Figure:
    plot_fragments = fragments.copy()
    plot_fragments["_hover_title"] = plot_fragments["title"].map(
        lambda value: _wrap_hover_text(
            value,
            width=42,
            maximum_lines=3,
        )
    )
    plot_fragments["_hover_proposition"] = plot_fragments[
        "proposition"
    ].map(
        lambda value: _wrap_hover_text(
            value,
            width=54,
            maximum_lines=6,
        )
    )

    figure = px.scatter_3d(
        plot_fragments,
        x="x",
        y="y",
        z="z",
        color=colour_by,
        color_discrete_map=_categorical_colour_map(
            plot_fragments,
            colour_by,
        ),
        size="point_size",
        size_max=10,
        custom_data=[
            "fragment_code",
            "investor_id",
            "kind",
            "evidence_strength",
            "_hover_title",
            "_hover_proposition",
        ],
        opacity=0.86,
    )

    for trace in figure.data:
        trace.marker.line = {
            "width": 0.8,
            "color": trace.marker.color,
        }
        trace.hovertemplate = (
            "<b>%{customdata[4]}</b><br>"
            "%{customdata[5]}<br><br>"
            "Code: %{customdata[0]}<br>"
            "Investor: %{customdata[1]}<br>"
            "Kind: %{customdata[2]}<br>"
            "Evidence strength: %{customdata[3]}"
            "<extra></extra>"
        )

    if selected_code and mark_selection:
        selected_rows = fragments.loc[
            fragments["fragment_code"] == selected_code
        ]

        if not selected_rows.empty:
            selected = selected_rows.iloc[0]

            figure.add_trace(
                go.Scatter3d(
                    x=[selected["x"]],
                    y=[selected["y"]],
                    z=[selected["z"]],
                    mode="markers",
                    name="Selected MMF",
                    customdata=[[
                        selected["fragment_code"],
                        selected["investor_id"],
                        selected["kind"],
                        selected["evidence_strength"],
                    ]],
                    text=[selected["title"]],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Selected MMF<br>"
                        "Code: %{customdata[0]}<br>"
                        "Investor: %{customdata[1]}<br>"
                        "Kind: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                    marker={
                        "size": 15,
                        "symbol": "circle-open",
                        "color": "rgba(0,0,0,0.01)",
                        "line": {
                            "width": 3.0,
                            "color": "#24324A",
                        },
                    },
                    opacity=1.0,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    figure.update_layout(
        height=GRAPH_HEIGHT,
        autosize=True,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
        },
        scene=_base_scene(fragments),
        hovermode="closest",
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#CBD5E1",
            "align": "left",
            "namelength": -1,
            "font": {
                "color": "#111827",
                "size": 13,
            },
        },
        clickmode="event",
        uirevision=f"mmf-network-{view_revision}",
    )

    return figure


def _fragment_label(fragment: dict[str, Any]) -> str:
    title = fragment.get("title") or fragment.get(
        "proposition",
        "",
    )
    return f"{fragment['fragment_code']} — {title}"


def _show_supporting_fragments(
    fragment_codes: list[str],
) -> None:
    with st.expander(
        f"Supporting mental-model fragments ({len(fragment_codes)})",
        expanded=False,
        width=DETAIL_WIDTH,
    ):
        st.caption(
            "Generated analytical fields only. Source quotations, "
            "document references, paths, and citations are excluded."
        )

        fragments = _load_supporting_fragments(
            tuple(fragment_codes)
        )

        if not fragments:
            st.info("No supporting fragment records were found.")
            return

        by_code = {
            fragment["fragment_code"]: fragment
            for fragment in fragments
        }
        selected_code = st.selectbox(
            "Select a supporting fragment",
            options=list(by_code),
            format_func=lambda code: _fragment_label(
                by_code[code]
            ),
            key="supporting_fragment_selector",
            width=SECONDARY_SELECTOR_WIDTH,
        )
        fragment = by_code[selected_code]

        st.code(
            fragment["fragment_code"],
            width=SECONDARY_SELECTOR_WIDTH,
        )
        st.markdown(
            f"#### {fragment.get('title') or 'Untitled fragment'}"
        )
        st.write(fragment["proposition"])
        st.write(f"Investor: **{fragment['investor_id']}**")
        st.write(f"Kind: **{fragment['kind']}**")
        st.write(
            "Evidence strength: "
            f"**{fragment.get('evidence_strength') or 'unassigned'}**"
        )
        _write_list("Mechanism", fragment["mechanism"])
        _write_list("Conditions", fragment["conditions"])
        _write_list(
            "Failure conditions",
            fragment["failure_conditions"],
        )
        _write_list(
            "Decision implications",
            fragment["decision_implications"],
        )
        _write_list(
            "Decision stages",
            fragment["decision_stages"],
        )
        _write_list(
            "Contextual regimes",
            fragment["contextual_regimes"],
        )


def _boundary_records(
    *,
    selected_code: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> list[dict[str, Any]]:
    selected_rows = nodes.loc[
        nodes["canonical_code"] == selected_code
    ]

    if selected_rows.empty or edges.empty:
        return []

    selected_id = str(
        selected_rows.iloc[0]["canonical_id"]
    )
    node_by_id = {
        str(row.canonical_id): row
        for row in nodes.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []

    connected = edges.loc[
        (edges["source_canonical_id"] == selected_id)
        | (edges["target_canonical_id"] == selected_id)
    ].sort_values(
        ["relation_confidence", "relation_strength"],
        ascending=False,
    )

    for edge in connected.itertuples(index=False):
        if edge.source_canonical_id == selected_id:
            boundary_id = str(edge.target_canonical_id)
            direction = "outgoing"
        else:
            boundary_id = str(edge.source_canonical_id)
            direction = "incoming"

        boundary = node_by_id.get(boundary_id)

        if boundary is None:
            continue

        records.append(
            {
                "edge_id": str(edge.edge_id),
                "canonical_code": boundary.canonical_code,
                "investor_id": boundary.investor_id,
                "title": boundary.title,
                "proposition": boundary.proposition,
                "primary_domain": boundary.primary_domain,
                "concept_family": boundary.concept_family,
                "relation_type": edge.relation_type,
                "relation_strength": float(
                    edge.relation_strength
                ),
                "relation_confidence": float(
                    edge.relation_confidence
                ),
                "candidate_similarity": (
                    float(edge.candidate_similarity)
                    if pd.notna(edge.candidate_similarity)
                    else None
                ),
                "scope": edge.scope,
                "direction": direction,
            }
        )

    return records


def _show_boundary_models(
    *,
    selected_code: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    records = _boundary_records(
        selected_code=selected_code,
        nodes=nodes,
        edges=edges,
    )

    with st.expander(
        f"Connected boundary MMCs ({len(records)})",
        expanded=False,
        width=DETAIL_WIDTH,
    ):
        st.caption(
            "Directly connected MMCs visible under the current "
            "node and relationship filters."
        )

        if not records:
            st.info("No connected boundary MMCs are currently visible.")
            return

        by_edge = {
            record["edge_id"]: record
            for record in records
        }
        selected_edge = st.selectbox(
            "Select a connected boundary MMC",
            options=list(by_edge),
            format_func=lambda edge_id: (
                f"{by_edge[edge_id]['investor_id']} — "
                f"{by_edge[edge_id]['title']} "
                f"[{by_edge[edge_id]['relation_type']}]"
            ),
            key=f"boundary_selector_{selected_code}",
            width=SECONDARY_SELECTOR_WIDTH,
        )
        boundary = by_edge[selected_edge]

        st.code(
            boundary["canonical_code"],
            width=SECONDARY_SELECTOR_WIDTH,
        )
        st.markdown(f"#### {boundary['title']}")
        st.write(boundary["proposition"])
        st.write(f"Investor: **{boundary['investor_id']}**")
        st.write(
            f"Primary domain: **{boundary['primary_domain']}**"
        )
        st.write(
            "Concept family: "
            f"**{boundary['concept_family'] or 'unassigned'}**"
        )
        st.write(
            f"Relationship: **{boundary['relation_type']}** "
            f"({boundary['direction']}, {boundary['scope']})"
        )

        first, second, third = st.columns(3)
        first.metric(
            "Relation strength",
            f"{boundary['relation_strength']:.3f}",
        )
        second.metric(
            "Relation confidence",
            f"{boundary['relation_confidence']:.3f}",
        )
        third.metric(
            "Candidate similarity",
            (
                f"{boundary['candidate_similarity']:.3f}"
                if boundary["candidate_similarity"] is not None
                else "n/a"
            ),
        )


def _show_mmc_details(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    st.subheader("MMC details")

    labels = {
        row.canonical_code: (
            f"{row.investor_id} — {row.title}"
        )
        for row in nodes.itertuples(index=False)
    }

    if not labels:
        return

    if (
        st.session_state.get("selected_mmc_code")
        not in labels
    ):
        st.session_state["selected_mmc_code"] = next(
            iter(labels)
        )

    selected_code = st.selectbox(
        "Select a canonical model",
        options=list(labels),
        format_func=lambda code: labels[code],
        width=DETAIL_WIDTH,
        key="selected_mmc_code",
    )
    record = nodes.loc[
        nodes["canonical_code"] == selected_code
    ].iloc[0]

    st.code(record["canonical_code"], width=DETAIL_WIDTH)
    st.markdown(f"### {record['title']}")
    st.write(record["proposition"])

    left, right = st.columns(2)

    with left:
        st.write(f"Investor: **{record['investor_id']}**")
        st.write(f"Kind: **{record['kind']}**")
        st.write(
            f"Primary domain: **{record['primary_domain']}**"
        )
        st.write(
            "Concept family: "
            f"**{record['concept_family'] or 'unassigned'}**"
        )

    with right:
        st.metric(
            "Base weight",
            f"{record['base_weight']:.3f}",
        )
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
    _write_list(
        "Failure conditions",
        record["failure_conditions"],
    )
    _write_list(
        "Decision implications",
        record["decision_implications"],
    )
    _write_list(
        "Decision stages",
        record["decision_stages"],
    )

    _show_supporting_fragments(
        list(record["supporting_fragment_codes"])
    )
    _show_boundary_models(
        selected_code=selected_code,
        nodes=nodes,
        edges=edges,
    )


def _show_associated_mmcs(
    associations: list[dict[str, Any]],
    *,
    fragment_code: str,
) -> None:
    with st.expander(
        f"Associated canonical mental models ({len(associations)})",
        expanded=False,
        width=DETAIL_WIDTH,
    ):
        if not associations:
            st.info(
                "This fragment is not currently assigned to an MMC."
            )
            return

        by_code = {
            association["canonical_code"]: association
            for association in associations
        }
        selected_code = st.selectbox(
            "Select an associated MMC",
            options=list(by_code),
            format_func=lambda code: (
                f"{by_code[code]['investor_id']} — "
                f"{by_code[code]['title']}"
            ),
            key=f"mmf_associated_mmc_{fragment_code}",
            width=SECONDARY_SELECTOR_WIDTH,
        )
        association = by_code[selected_code]

        st.code(
            association["canonical_code"],
            width=SECONDARY_SELECTOR_WIDTH,
        )
        st.markdown(f"#### {association['title']}")
        st.write(association["proposition"])
        st.write(
            f"Primary domain: "
            f"**{association['primary_domain']}**"
        )
        st.write(
            "Concept family: "
            f"**{association['concept_family'] or 'unassigned'}**"
        )
        st.write(
            f"Base weight: **{association['base_weight']:.3f}**"
        )


def _show_mmf_details(
    fragments: pd.DataFrame,
) -> None:
    st.subheader("MMF details")

    labels = {
        row.fragment_code: (
            f"{row.investor_id} — {row.title}"
        )
        for row in fragments.itertuples(index=False)
    }

    if not labels:
        return

    if (
        st.session_state.get("selected_mmf_code")
        not in labels
    ):
        st.session_state["selected_mmf_code"] = next(
            iter(labels)
        )

    selected_code = st.selectbox(
        "Select a mental-model fragment",
        options=list(labels),
        format_func=lambda code: labels[code],
        width=DETAIL_WIDTH,
        key="selected_mmf_code",
    )
    fragment = fragments.loc[
        fragments["fragment_code"] == selected_code
    ].iloc[0]

    st.code(fragment["fragment_code"], width=DETAIL_WIDTH)
    st.markdown(
        f"### {fragment.get('title') or 'Untitled fragment'}"
    )
    st.write(fragment["proposition"])
    st.write(f"Investor: **{fragment['investor_id']}**")
    st.write(f"Kind: **{fragment['kind']}**")
    st.write(
        "Evidence strength: "
        f"**{fragment.get('evidence_strength') or 'unassigned'}**"
    )

    _write_list("Mechanism", fragment["mechanism"])
    _write_list("Conditions", fragment["conditions"])
    _write_list(
        "Failure conditions",
        fragment["failure_conditions"],
    )
    _write_list(
        "Decision implications",
        fragment["decision_implications"],
    )
    _write_list(
        "Decision stages",
        fragment["decision_stages"],
    )
    _write_list(
        "Contextual regimes",
        fragment["contextual_regimes"],
    )
    _show_associated_mmcs(
        list(fragment["associated_mmcs"]),
        fragment_code=selected_code,
    )


def _reset_mmc_filters(
    *,
    investors: list[str],
    domains: list[str],
    relation_types: list[str],
    node_count: int,
    edge_count: int,
) -> None:
    defaults = {
        "mmc_investors": investors,
        "mmc_domains": domains,
        "mmc_colour_label": "Value investor",
        "minimum_weight": 0.0,
        "maximum_nodes": min(1200, node_count),
        "show_edges": True,
        "relation_types": relation_types,
        "minimum_strength": 0.55,
        "minimum_confidence": 0.65,
        "maximum_edges": min(2500, edge_count),
    }

    for key, value in defaults.items():
        st.session_state[key] = value

    st.rerun()


def _reset_mmf_filters(
    *,
    investors: list[str],
    kinds: list[str],
    fragment_count: int,
) -> None:
    defaults = {
        "mmf_investors": investors,
        "mmf_kinds": kinds,
        "mmf_colour_label": "Value investor",
        "maximum_mmfs": min(1800, fragment_count),
    }

    for key, value in defaults.items():
        st.session_state[key] = value

    st.rerun()


def _plotting_controls(
    *,
    network_key: str,
    reset_filters: Any,
) -> tuple[bool, int]:
    """Render plot controls for the currently active network."""

    st.sidebar.header("Plotting")

    mark_selection_key = f"{network_key}_mark_selected_point"
    view_revision_key = f"{network_key}_plot_view_revision"

    mark_selection = st.sidebar.checkbox(
        "Mark selected point and neighbours",
        value=True,
        key=mark_selection_key,
    )

    if view_revision_key not in st.session_state:
        st.session_state[view_revision_key] = 0

    with st.sidebar:
        (
            reset_filters_column,
            reset_view_column,
        ) = st.columns(
            [1, 1],
            gap="small",
        )

        if reset_filters_column.button(
            "Reset filters",
            key=f"{network_key}_reset_filters_button",
            width="stretch",
        ):
            reset_filters()

        if reset_view_column.button(
            "Reset 3D view",
            key=f"{network_key}_reset_3d_view",
            width="stretch",
        ):
            st.session_state[view_revision_key] += 1
            st.rerun()

    return (
        mark_selection,
        int(st.session_state[view_revision_key]),
    )


def _navigation_help() -> None:
    with st.sidebar.expander(
        "How to navigate the 3D graphs",
        expanded=False,
    ):
        st.markdown(
            """
- **Left-click and drag** to rotate.
- **Mouse wheel** to zoom.
- **Middle-click and drag** to move the plot.
- **Click an MMC point** in the MMC Network to select it. The selected MMC
  menu and its detailed description below the graph update automatically.
- **Click an MMF point** in the MMF Network to show that fragment's details.
            """
        )

    with st.sidebar.expander(
        "How to interpret the visualisation",
        expanded=False,
    ):
        st.markdown(
            """
**MMC — Canonical Mental Model**

A consolidated investment principle created from one or more related mental-
model fragments. MMCs preserve the investor-specific proposition, mechanism,
conditions, failure conditions, and decision implications.

**MMF — Mental-Model Fragment**

A structured investment insight extracted from one source document. Related
MMFs can be consolidated into a broader MMC.

**Position in the 3D plot**

Each point begins as a high-dimensional semantic embedding. PCA projects it
onto the first three principal components. Nearby points are generally more
similar in meaning, but the three-dimensional view is only an approximation.

**Connections**

Lines in the MMC Network are stored relationships between canonical models,
such as `similar_to`, `supports`, `contradicts`, `requires`, or `parent_of`.
The MMF Network does not infer links; it only shows each fragment's associated
MMC.

**Base weight**

A combined relevance score derived from evidence confidence and investor
importance. It affects MMC point size.

**Evidence confidence (EC)**

How strongly the available fragments support the canonical model, considering
evidence directness, breadth, and cluster coherence.

**Investor importance**

How central the model appears to that investor, considering how broadly and
repeatedly it appears across the material and investment decision stages.

**Relationship strength**

How strong or material the connection between two MMCs appears to be.

**Relationship confidence**

How confident the classifier is that the assigned relationship type and
direction are correct.

**Candidate similarity**

The embedding cosine similarity that caused a pair of MMCs to be considered
for relationship classification.

**Primary knowledge domain**

The broad investment-constitution category assigned to the MMC, such as
business quality, valuation, risk, financial resilience, or portfolio
construction.

**Concept family**

A narrower recurring idea within a domain, such as pricing power, margin of
safety, or capital allocation.
            """
        )


def main() -> None:
    st.set_page_config(
        page_title="Mental Model Networks",
        layout="wide",
    )

    st.title("Visualisation of Canonical Mental Models (MMC)")
    st.caption(
        "This project reconstructs and organises the investment mental "
        "models used by leading value investors. The Canonical Mental "
        "Model (MMC) network shows consolidated investment principles, "
        "while the Mental-Model Fragment (MMF) network shows the "
        "individual structured insights from which those models are "
        "formed. Both are displayed as semantic maps for exploring "
        "recurring ideas, investor-specific perspectives, and "
        "relationships between concepts."
    )

    try:
        all_mmcs, all_edges = _load_mmc_projection()
        all_mmfs = _load_mmf_projection()
    except Exception as error:
        st.error(f"Could not load the mental-model data: {error}")
        st.stop()

    _, network_selector_column, _ = st.columns(
        [1, 2, 1],
        gap="small",
    )

    with network_selector_column:
        active_network = st.segmented_control(
            "Network view",
            options=["MMC Network", "MMF Network"],
            default="MMC Network",
            selection_mode="single",
            key="active_network",
            label_visibility="collapsed",
            width="stretch",
        )

    if active_network == "MMC Network":
        investors = sorted(
            all_mmcs["investor_id"].dropna().unique()
        )
        domains = sorted(
            all_mmcs["primary_domain"].dropna().unique()
        )
        relation_types = (
            sorted(
                all_edges["relation_type"].dropna().unique()
            )
            if not all_edges.empty
            else []
        )

        st.sidebar.header("MMC filters")

        selected_investors = st.sidebar.multiselect(
            "Select value investors to display",
            options=investors,
            default=investors,
            key="mmc_investors",
        )
        selected_domains = st.sidebar.multiselect(
            "Filter by primary knowledge domain",
            options=domains,
            default=domains,
            key="mmc_domains",
        )
        colour_label = st.sidebar.radio(
            "Colour-code MMCs by",
            options=[
                "Value investor",
                "Primary knowledge domain",
            ],
            key="mmc_colour_label",
        )
        colour_by = {
            "Value investor": "investor_id",
            "Primary knowledge domain": "primary_domain",
        }[colour_label]
        minimum_weight = st.sidebar.slider(
            "Minimum base weight",
            0.0,
            1.0,
            0.0,
            0.01,
            key="minimum_weight",
        )
        maximum_nodes = st.sidebar.slider(
            "Maximum displayed MMCs",
            min_value=min(100, len(all_mmcs)),
            max_value=len(all_mmcs),
            value=min(1200, len(all_mmcs)),
            step=max(1, len(all_mmcs) // 100),
            key="maximum_nodes",
        )

        st.sidebar.header("MMC link filters")

        show_edges = st.sidebar.checkbox(
            "Show graph links",
            value=True,
            key="show_edges",
        )
        selected_relation_types = st.sidebar.multiselect(
            "Relationship types",
            options=relation_types,
            default=relation_types,
            key="relation_types",
        )
        minimum_strength = st.sidebar.slider(
            "Minimum relationship strength",
            0.0,
            1.0,
            0.55,
            0.05,
            key="minimum_strength",
        )
        minimum_confidence = st.sidebar.slider(
            "Minimum relationship confidence",
            0.0,
            1.0,
            0.65,
            0.05,
            key="minimum_confidence",
        )
        maximum_edges = st.sidebar.slider(
            "Maximum displayed links",
            min_value=0,
            max_value=max(1, len(all_edges)),
            value=min(2500, len(all_edges)),
            step=max(1, len(all_edges) // 100),
            key="maximum_edges",
        )

        mark_selection, view_revision = _plotting_controls(
            network_key="mmc",
            reset_filters=lambda: _reset_mmc_filters(
                investors=investors,
                domains=domains,
                relation_types=relation_types,
                node_count=len(all_mmcs),
                edge_count=len(all_edges),
            ),
        )
        _navigation_help()

        mmcs = _filter_mmc_nodes(
            all_mmcs,
            investors=selected_investors,
            domains=selected_domains,
            minimum_weight=minimum_weight,
            maximum_nodes=maximum_nodes,
        )

        if mmcs.empty:
            st.warning("The current filters select no MMCs.")
            return

        visible_codes = set(mmcs["canonical_code"])

        if (
            st.session_state.get("selected_mmc_code")
            not in visible_codes
        ):
            st.session_state["selected_mmc_code"] = str(
                mmcs.iloc[0]["canonical_code"]
            )

        selected_code = st.session_state[
            "selected_mmc_code"
        ]
        selected_id = str(
            mmcs.loc[
                mmcs["canonical_code"] == selected_code,
                "canonical_id",
            ].iloc[0]
        )

        edges = (
            _filter_edges(
                all_edges,
                mmcs,
                relation_types=selected_relation_types,
                minimum_strength=minimum_strength,
                minimum_confidence=minimum_confidence,
                maximum_edges=maximum_edges,
                selected_canonical_id=selected_id,
            )
            if show_edges
            else all_edges.iloc[0:0].copy()
        )

        first, second, third, fourth = st.columns(4)
        first.metric("Loaded MMCs", f"{len(all_mmcs):,}")
        second.metric("Displayed MMCs", f"{len(mmcs):,}")
        third.metric("Stored links", f"{len(all_edges):,}")
        fourth.metric("Displayed links", f"{len(edges):,}")

        figure = _mmc_figure(
            mmcs,
            edges,
            colour_by=colour_by,
            selected_code=selected_code,
            mark_selection=mark_selection,
            view_revision=view_revision,
        )
        events = _render_clickable_plot(
            figure,
            key=f"mmc_click_graph_{view_revision}",
        )
        clicked = _clicked_code(
            events,
            figure,
            mmcs,
            code_column="canonical_code",
        )

        if (
            clicked
            and clicked
            != st.session_state.get("selected_mmc_code")
        ):
            st.session_state["selected_mmc_code"] = clicked
            st.rerun()

        _show_mmc_details(mmcs, edges)
        return

    mmf_investors = sorted(
        all_mmfs["investor_id"].dropna().unique()
    )
    mmf_kinds = sorted(
        all_mmfs["kind"].dropna().unique()
    )

    st.sidebar.header("MMF filters")

    selected_mmf_investors = st.sidebar.multiselect(
        "Value investors",
        options=mmf_investors,
        default=mmf_investors,
        key="mmf_investors",
    )
    selected_mmf_kinds = st.sidebar.multiselect(
        "MMF kinds",
        options=mmf_kinds,
        default=mmf_kinds,
        key="mmf_kinds",
    )
    mmf_colour_label = st.sidebar.selectbox(
        "Colour-code MMFs by",
        options=[
            "Value investor",
            "MMF kind",
            "Evidence strength",
        ],
        key="mmf_colour_label",
    )
    maximum_mmfs = st.sidebar.slider(
        "Maximum displayed MMFs",
        min_value=min(100, len(all_mmfs)),
        max_value=len(all_mmfs),
        value=min(1800, len(all_mmfs)),
        step=max(1, len(all_mmfs) // 100),
        key="maximum_mmfs",
    )

    mark_selection, view_revision = _plotting_controls(
        network_key="mmf",
        reset_filters=lambda: _reset_mmf_filters(
            investors=mmf_investors,
            kinds=mmf_kinds,
            fragment_count=len(all_mmfs),
        ),
    )
    _navigation_help()

    mmfs = _filter_mmf_nodes(
        all_mmfs,
        investors=selected_mmf_investors,
        kinds=selected_mmf_kinds,
        maximum_fragments=maximum_mmfs,
    )

    if mmfs.empty:
        st.warning("The current filters select no MMFs.")
        return

    visible_codes = set(mmfs["fragment_code"])

    if (
        st.session_state.get("selected_mmf_code")
        not in visible_codes
    ):
        st.session_state["selected_mmf_code"] = str(
            mmfs.iloc[0]["fragment_code"]
        )

    mmf_colour_by = {
        "Value investor": "investor_id",
        "MMF kind": "kind",
        "Evidence strength": "evidence_strength",
    }[mmf_colour_label]

    first, second = st.columns(2)
    first.metric("Loaded MMFs", f"{len(all_mmfs):,}")
    second.metric("Displayed MMFs", f"{len(mmfs):,}")

    figure = _mmf_figure(
        mmfs,
        colour_by=mmf_colour_by,
        selected_code=st.session_state[
            "selected_mmf_code"
        ],
        mark_selection=mark_selection,
        view_revision=view_revision,
    )
    events = _render_clickable_plot(
        figure,
        key=f"mmf_click_graph_{view_revision}",
    )
    clicked = _clicked_code(
        events,
        figure,
        mmfs,
        code_column="fragment_code",
    )

    if (
        clicked
        and clicked
        != st.session_state.get("selected_mmf_code")
    ):
        st.session_state["selected_mmf_code"] = clicked
        st.rerun()

    _show_mmf_details(mmfs)


if __name__ == "__main__":
    main()
