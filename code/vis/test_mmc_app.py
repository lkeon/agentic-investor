"""Tests for visualisation click handling."""

import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

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


def test_plot_uses_native_webgl_resolution() -> None:
    """Avoid Plotly's expensive two-times WebGL framebuffer."""

    assert mmc_app.PLOTLY_GL_PIXEL_RATIO == 1


def test_deployment_entrypoint_configures_source_imports() -> None:
    """Run the app entry point without relying on PYTHONPATH."""

    entrypoint = Path(__file__).with_name("streamlit_app.py")
    namespace = runpy.run_path(
        str(entrypoint),
        run_name="deployment_entrypoint_test",
    )
    rendered: list[bool] = []
    fake_app = ModuleType("vis.mmc_app")
    fake_app.main = lambda: rendered.append(True)

    with patch.dict(sys.modules, {"vis.mmc_app": fake_app}):
        namespace["main"]()

    assert str(namespace["CODE_ROOT"]) in sys.path
    assert rendered == [True]
