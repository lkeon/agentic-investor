# Agentic Investor

Agentic Investor is an experimental research pipeline for turning investor
documents into structured, searchable mental models.

The current corpus focuses on Warren Buffett, Charlie Munger, Howard Marks,
Bruce Flatt, Mohnish Pabrai, and Guy Spier. Source material is validated,
converted to canonical Markdown, analysed with an OpenAI model, validated with
Pydantic, embedded, and stored in PostgreSQL. The same fragment data is also
exported as JSONL for inspection and downstream processing.

The longer-term goal is an agentic investment committee: investor-specific
agents retrieve relevant mental models, debate an investment case, and
contribute to a consolidated research memo.

## Pipeline

```text
investor source manifests
        ↓
validated raw corpus manifest
        ↓
TXT/PDF documents → canonical Markdown
        ↓
processed Markdown manifest
        ↓
Pydantic-validated mental-model fragments
        ↓
OpenAI embeddings
        ↓
PostgreSQL + complete JSONL export
```

The extraction stage currently reads each document as a whole and selects up
to ten of its most important, distinct, and generally applicable investment
models. Documents are not chunked during the MVP stage.

## Generated Artifacts

| Artifact | Location |
| --- | --- |
| Validated raw corpus manifest | `data/raw/corpus_manifest.jsonl` |
| Canonical Markdown | `data/processed/markdown/investors/...` |
| Processed Markdown manifest | `data/processed/markdown_manifest.jsonl` |
| Exported mental-model fragments | `data/processed/fragments/mental_model_fragments.jsonl` |

The fragment export contains document provenance, fragment fields, related
entities, database identifiers, embedding metadata, and complete
1,024-dimensional embedding vectors. Files under `data/` are generated or
local corpus artifacts and are excluded from Git.

## Setup

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venvinvest
source .venvinvest/bin/activate
python -m pip install -r requirements.txt
```

Browser binaries are only needed for browser-based acquisition scripts:

```bash
python -m playwright install chromium
```

Create a `.env` file in the project root containing:

```dotenv
OPENAI_API_KEY=your-api-key
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/agentic_investor

# Optional; defaults to gpt-5.6-terra.
FRAGMENT_EXTRACTION_MODEL=gpt-5.6-terra
```

The database must be PostgreSQL with the `pgvector` extension enabled. Once the
database exists, create the application tables:

```bash
PYTHONPATH=code python -m mental_model_pipeline.database.setup_database
```

The setup module creates missing tables from the SQLAlchemy models; it is not a
general schema-migration system.

## Run the Pipeline

Corpus acquisition, PDF splitting, manifest building, and Markdown conversion
scripts are kept together in [code/data_ingestion](code/data_ingestion/README.md).

### 1. Validate and build the raw corpus manifest

Validate investor-level manifests without writing the combined manifest:

```bash
python code/data_ingestion/build_raw_manifest.py --check
```

Build `data/raw/corpus_manifest.jsonl` after validation succeeds:

```bash
python code/data_ingestion/build_raw_manifest.py
```

The builder treats investor-level manifests as authoritative whitelists. It
checks document identifiers, resolves source paths, validates date metadata,
calculates SHA-256 hashes, detects duplicate identifiers and content, and
writes the combined manifest atomically.

Use `--strict` with either command when warnings should produce a non-zero exit
status.

### 2. Convert source documents to Markdown

```bash
python code/data_ingestion/convert_corpus_to_markdown.py
```

TXT sources use deterministic text cleaning; PDF sources use Docling. The
investor directory structure and document-level metadata are preserved in the
generated Markdown. Successful unchanged documents are skipped on subsequent
runs.

To regenerate every Markdown document:

```bash
python code/data_ingestion/convert_corpus_to_markdown.py --force
```

### 3. Validate extraction inputs safely

`--dry-run` validates manifest records, paths, file types, SHA-256 hashes,
front matter, character counts, and investor identifiers. It deliberately does
not load the database runtime or call the OpenAI API.

Validate the first ten successful manifest entries:

```bash
PYTHONPATH=code python -m mental_model_pipeline.fragments.ingest_markdown_all \
  --dry-run --process-num 10
```

Validate one manifest-listed Markdown file:

```bash
PYTHONPATH=code python -m mental_model_pipeline.fragments.ingest_markdown_all \
  --dry-run \
  --single-run data/processed/markdown/investors/flatt/shareholder_letters/shareholder_letter_2012_q1.md
```

### 4. Extract, embed, and store fragments

Process one document:

```bash
PYTHONPATH=code python -m mental_model_pipeline.fragments.ingest_markdown_all \
  --single-run data/processed/markdown/investors/flatt/shareholder_letters/shareholder_letter_2012_q1.md
```

Process only the first ten selected documents:

```bash
PYTHONPATH=code python -m mental_model_pipeline.fragments.ingest_markdown_all \
  --process-num 10
```

Process every successful manifest entry:

```bash
PYTHONPATH=code python -m mental_model_pipeline.fragments.ingest_markdown_all
```

These commands make paid OpenAI extraction and embedding requests for new
documents. Before making a request, the ingester checks the database and skips
an identical document that has already been stored. A conflicting document ID
or duplicate content is reported rather than silently overwritten.

Each document is committed in its own transaction. Transient OpenAI failures
are retried with capped exponential backoff, and processing continues after an
individual document failure unless `--fail-fast` is supplied. Use
`--retry-attempts NUMBER` to change the default of five attempts.

Extraction starts with a 32,000-token output limit. If structured JSON is
truncated, it retries twice with 64,000 and then 128,000 output tokens while
using low reasoning effort. Other validation errors are reported without this
fallback, and no database record is written for that document.

After processing, committed manifest-backed fragments are read from the
database in deterministic order and written atomically to
`data/processed/fragments/mental_model_fragments.jsonl`.

## Fragment Model

Each extracted mental model records:

- its kind, title, and core proposition;
- mechanism, conditions, and failure conditions;
- decision implications and applicable investment stages;
- contextual regimes and related entities;
- an exact supporting source quotation;
- evidence strength and attribution;
- review status and reason;
- a conceptual embedding and embedding-model metadata.

The Pydantic schemas reject unexpected fields, normalise list values, enforce
attribution and review requirements, and constrain identifiers and field
lengths. Source quotations that cannot be matched against the document are
retained but marked for manual review.

## MMC Visualisation

The Streamlit visualisation in [`code/vis`](code/vis/README.md) reads canonical
mental models and their stored relationships directly from PostgreSQL. MMC
embeddings are projected onto the first three principal components and shown
as an interactive 3D network.

The interface supports investor and domain filtering, alternative colour
coding, relationship filters, MMC inspection, and expandable supporting MMFs.
Only generated analytical MMF fields are shown; source quotations and source
references are excluded.

![MMC detail explorer](docs/images/mmc_visualisation.png)

Run it from the repository root:

```bash
PYTHONPATH=code python -m streamlit run code/vis/mmc_app.py
```

The visualisation's dependencies are listed separately in
[`code/vis/requirements.txt`](code/vis/requirements.txt).

## Tests

Run the manifest-ingestion unit tests with:

```bash
PYTHONPATH=code python -m unittest \
  mental_model_pipeline.fragments.test_extraction \
  mental_model_pipeline.fragments.test_ingest_markdown_all -v
```

These tests use temporary files and mocked database/API boundaries. They cover
the extraction output limit and malformed responses as well as dry-run
isolation, single-file selection, document limits, hash validation, idempotent
skips, retry behaviour, orchestration, and complete embedding exports without
making paid requests.

The other scripts named `test_*.py` under `fragments/` are manual database or
embedding smoke tests and may require PostgreSQL or make an OpenAI API request.

## Current Boundaries

- Extraction operates on complete documents and currently returns at most ten
  fragments per document.
- Fragment codes use compact application-generated identifiers.
- Database tables are created directly from SQLAlchemy metadata; migrations are
  not yet managed by a migration framework.
- Retrieval, investor-specific agents, debate orchestration, and consolidated
  memo generation are not implemented yet.

## Core Technologies

- Pydantic for validated structured extraction
- OpenAI Responses API and embeddings
- SQLAlchemy, PostgreSQL, and pgvector
- Docling and PyMuPDF for document processing
- Playwright for browser-based acquisition tools
- Streamlit, Plotly, and scikit-learn for MMC visualisation
