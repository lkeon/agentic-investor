"""Tests for visualisation click handling."""

import pandas as pd
import plotly.graph_objects as go

import vis.mmc_app as mmc_app


def test_clicked_code_accepts_scatter3d_point_number_payload() -> None:
    """Resolve the payload emitted for a Plotly Scatter3d point click."""

    frame = pd.DataFrame(
        {
            "canonical_code": ["MMC-A", "MMC-B"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 1.0],
        }
    )
    figure = go.Figure(
        go.Scatter3d(
            x=frame["x"],
            y=frame["y"],
            z=frame["z"],
            customdata=frame[["canonical_code"]].to_numpy(),
        )
    )
    events = [
        {
            "x": 1.0,
            "y": 1.0,
            "curveNumber": 0,
            "pointNumber": 1,
        }
    ]

    assert (
        mmc_app._clicked_code(
            events,
            figure,
            frame,
            code_column="canonical_code",
        )
        == "MMC-B"
    )


def test_clicked_code_resolves_a_graph_link_endpoint() -> None:
    """An edge click selects the MMC located at that edge endpoint."""

    frame = pd.DataFrame(
        {
            "canonical_code": ["MMC-A", "MMC-B"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 1.0],
        }
    )
    figure = go.Figure(
        go.Scatter3d(
            x=[0.0, 1.0, None],
            y=[0.0, 1.0, None],
            z=[0.0, 1.0, None],
            mode="lines",
        )
    )
    events = [
        {
            "curveNumber": 0,
            "pointNumber": 1,
        }
    ]

    assert (
        mmc_app._clicked_code(
            events,
            figure,
            frame,
            code_column="canonical_code",
        )
        == "MMC-B"
    )


def test_replayed_click_is_not_resolved_again() -> None:
    """Ignore a retained component event after trace ordering changes."""

    frame = pd.DataFrame(
        {
            "canonical_code": ["MMC-A", "MMC-B"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 1.0],
        }
    )
    first_figure = go.Figure(
        go.Scatter3d(
            x=frame["x"],
            y=frame["y"],
            z=frame["z"],
            customdata=[["MMC-A"], ["MMC-B"]],
        )
    )
    reordered_figure = go.Figure(
        go.Scatter3d(
            x=frame["x"],
            y=frame["y"],
            z=frame["z"],
            customdata=[["MMC-B"], ["MMC-A"]],
        )
    )
    events = [
        {
            "x": 1.0,
            "y": 1.0,
            "curveNumber": 0,
            "pointNumber": 1,
        }
    ]
    state: dict[str, str] = {}

    first_events = mmc_app._consume_click_events(
        events,
        state=state,
        state_key="last_click",
    )
    assert (
        mmc_app._clicked_code(
            first_events,
            first_figure,
            frame,
            code_column="canonical_code",
        )
        == "MMC-B"
    )

    replayed_events = mmc_app._consume_click_events(
        events,
        state=state,
        state_key="last_click",
    )
    assert (
        mmc_app._clicked_code(
            replayed_events,
            reordered_figure,
            frame,
            code_column="canonical_code",
        )
        is None
    )


def test_plot_uses_native_webgl_resolution() -> None:
    """Avoid Plotly's expensive two-times WebGL framebuffer."""

    assert mmc_app.PLOTLY_GL_PIXEL_RATIO == 1
