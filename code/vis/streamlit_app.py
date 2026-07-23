"""Deployment entry point for the Streamlit visualisation."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def main() -> None:
    """Import and render the app after making the source tree importable."""

    from vis.mmc_app import main as render_app

    render_app()


if __name__ == "__main__":
    main()
