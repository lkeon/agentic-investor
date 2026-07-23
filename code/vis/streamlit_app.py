"""Streamlit entry point for local and Community Cloud deployment."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Start the visualisation with the project's source directory available."""

    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))

    from vis.mmc_app import main as render_app

    render_app()


if __name__ == "__main__":
    main()
