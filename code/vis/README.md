# MMC visualisation MVP

This package provides one Streamlit application with two tabs:

1. **MMC network**
   - reads canonical models and graph edges directly from PostgreSQL;
   - reduces stored embeddings to three dimensions using PCA;
   - displays canonical models as Plotly 3D nodes;
   - displays stored `canonical_model_edges` as links;
   - filters by investor, primary domain, weight, relationship type,
     relationship strength and relationship confidence;
   - shows the complete content of a selected canonical model.

2. **Crew result**
   - optionally reads `data/processed/crew/committee_result.json`;
   - shows the question, retrieved-MMC counts, Round 1 analyses and
     Round 2 peer reviews.

No database writes are performed.

## Install

From the active project virtual environment:

```bash
pip install "streamlit>=1.53,<2" "plotly>=6,<7" \
  "scikit-learn>=1.5,<2" "pandas>=2,<3"
```

## Run

From the repository root:

```bash
PYTHONPATH=code python -m streamlit run \
  code/vis/mmc_app.py
```

Streamlit prints the local browser address, normally:

```text
http://localhost:8501
```

## Notes

- PCA is deterministic and recalculated from all embedded MMCs when the
  database cache is refreshed.
- The initial view limits the number of nodes and edges so the browser remains
  responsive. The sidebar can increase both limits.
- Graph positions are semantic projections, not investment scores.
- The app is read-only.
