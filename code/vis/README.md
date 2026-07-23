# Mental Model Visualisation

This project reconstructs and organises the investment mental models used by leading value investors.

The read-only Streamlit app contains two interactive PCA views:

- **MMC Network** — canonical mental models and stored graph links;
- **MMF Network** — mental-model fragments without graph analysis.

Both networks project stored embeddings onto the first three principal
components. Clicking a point updates its generated analytical details below.
The MMF view also lists the canonical models associated with the selected
fragment.

Source quotations, paths, citations, and source references are excluded.

## Install

Install the visualisation dependencies through the Python interpreter belonging
to the active project virtual environment:

```bash
python -m pip install -r code/vis/requirements.txt
```

Verify that the click component is importable in that same environment:

```bash
python -c "from streamlit_plotly_events2 import plotly_events; print('click support ready')"
```

`streamlit-plotly-events2` installs the Python module
`streamlit_plotly_events2`. It is required because Streamlit's built-in Plotly
selection hook does not reliably return ordinary clicks from 3D scatter plots.

PostgreSQL must be running and `.env` must contain a valid `DATABASE_URL`.

## Run

```bash
PYTHONPATH=code python -m streamlit run code/vis/mmc_app.py
```

Open the address printed by Streamlit, normally
`http://localhost:8501`.

## Deploy

On Streamlit Community Cloud, select:

- branch: `streamline-deploy`;
- entrypoint: `code/vis/streamlit_app.py`.

The adjacent `code/vis/requirements.txt` contains the app dependencies. In the
app's **Secrets** settings, add the pooled Neon connection string as a
root-level secret:

```toml
DATABASE_URL = "postgresql+psycopg://user:password@endpoint-pooler.region.aws.neon.tech/database_name?sslmode=require"
```

Copy the pooled URL from Neon's **Connect** dialog and do not commit it. The
application accepts Neon's standard `postgresql://` URL too and selects the
installed Psycopg 3 driver automatically.

## Neon database reminder

Use the Neon database branch `production` and the `agentic_investor` database.
The connection type depends on the task:

- **Direct URL** — no `-pooler` in the hostname; use only for `psql`,
  `pg_dump`, and `pg_restore`.
- **Pooled URL** — `-pooler` in the hostname; use as `DATABASE_URL` in
  Streamlit Community Cloud Secrets.

To migrate the local database manually, first load the local URL from `.env`
and convert its SQLAlchemy-specific scheme for PostgreSQL command-line tools:

```bash
set -a
source .env
set +a
SOURCE_DATABASE_URL="${DATABASE_URL/postgresql+psycopg:/postgresql:}"

read -r -s -p "Neon direct URL: " NEON_DIRECT_DATABASE_URL
echo
```

Create the vector extension, dump the local database, and restore it once into
an empty Neon database:

```bash
psql "$NEON_DIRECT_DATABASE_URL" \
  -c 'CREATE EXTENSION IF NOT EXISTS vector;'

pg_dump --format=custom --verbose \
  --file=/tmp/agentic_investor.dump \
  "$SOURCE_DATABASE_URL"

pg_restore --verbose --no-owner --no-acl \
  --dbname="$NEON_DIRECT_DATABASE_URL" \
  /tmp/agentic_investor.dump
```

Do not rerun `pg_restore` against a database that already contains the tables.
Verify the migration by comparing this query on the local and Neon URLs:

```bash
psql "$NEON_DIRECT_DATABASE_URL" -c \
  "SELECT
     (SELECT count(*) FROM documents) AS documents,
     (SELECT count(*) FROM mental_model_fragments) AS fragments,
     (SELECT count(*) FROM canonical_mental_models) AS mmcs,
     (SELECT count(*) FROM canonical_model_edges) AS edges;"
```

Use left-drag to rotate, the mouse wheel to zoom, and middle-drag to move the
plot. PCA positions represent semantic similarity only and are not investment
scores.

## Plot controls

The sidebar **Plotting** section can:

- enable or disable highlighting of the selected point and its neighbours;
- reset all MMC filters;
- reset the 3D camera to its initial centred view.

The two reset buttons are displayed next to each other. Selection and boundary
markers are visual annotations and are omitted from the graph legend.

There is no database reload button. Restarting or redeploying the app refreshes
the cached database data.


## Help sections

The bottom of the sidebar contains:

- **How to navigate the 3D graphs** — rotate, zoom, move, and select points;
- **How to interpret the visualisation** — definitions of MMCs, MMFs, PCA
  position, graph connections, base weight, evidence confidence, investor
  importance, relationship strength, relationship confidence, candidate
  similarity, knowledge domains, and concept families.


## Network navigation and sampling

The MMC and MMF views use a segmented network selector rather than
`st.tabs`. This lets the app render only the active view and show only that
view's controls in the sidebar.

Display limits use stable stratified random sampling instead of taking the
first or highest-ranked rows:

- MMC samples are stratified by investor and primary knowledge domain;
- MMF samples are stratified by investor and fragment kind;
- when the limit is large enough, every surviving group contributes at least
  one randomly selected point;
- remaining spaces are sampled from the remaining population, so larger groups
  can contribute proportionally more points;
- fixed seeds keep the sample stable when clicking points or rerunning the app.


## Network selector and colours

The tab-style segmented network selector is left-aligned and constrained to a
compact column rather than spanning the full page.

Plot colours are assigned explicitly rather than relying on Plotly defaults:

- each value investor has one stable colour used in both networks;
- primary domains, MMF kinds, and evidence-strength categories use their own
  stable palettes;
- black and white are not used as categorical colours;
- ordinary marker outlines reuse the marker's fill colour.


## Display controls

The compact MMC/MMF network selector is centred above the visualisation.

The sidebar **Plotting** section contains two aligned controls:

- **Reset filters**
- **Reset 3D view**

Plotly hover popups use a white background, light border, and dark text in both
network views.

The subtitle now introduces both terms in full:

- **Canonical Mental Model (MMC)** — a consolidated investment principle;
- **Mental-Model Fragment (MMF)** — an individual structured insight used to
  form or support a canonical model.


## Hover-card behaviour

Hover cards are white, left-aligned, and anchored using Plotly's
closest-point hover mode. Long titles, propositions, and concept-family text
are wrapped into multiple lines to keep each card narrow.

The transparent selected-point and boundary-point outline traces do not
participate in hover detection. This prevents an invisible overlay from
placing a popup directly under the mouse; the underlying coloured node
provides the hover card instead.
