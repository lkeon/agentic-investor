# Agentic Investor

Agentic Investor turns investor documents into structured, searchable mental
models. It is the data and retrieval layer for an investment committee in which
investor-specific agents analyse a question, challenge one another, and produce
a consolidated research memo.

The current corpus focuses on Warren Buffett, Charlie Munger, Howard Marks,
Bruce Flatt, Mohnish Pabrai, and Guy Spier.

## Current functionality

- Manifest-based document acquisition and validation
- TXT and PDF conversion to canonical Markdown
- Structured mental-model extraction validated with Pydantic
- Source quotations, attribution and review flags for each fragment
- OpenAI embeddings stored in PostgreSQL with `pgvector`
- Idempotent ingestion, retries and per-document transactions
- Deterministic JSONL export for inspection and downstream use
- Unit tests with mocked API and database boundaries
- Minimal Chainlit interface for testing the committee flow locally

## Pipeline

```text
source manifests
      ↓
validated corpus manifest
      ↓
TXT / PDF → canonical Markdown
      ↓
mental-model fragments
      ↓
embeddings
      ↓
PostgreSQL + JSONL export
      ↓
retrieval and investment committee
```

For the MVP, each document is processed as a whole and returns up to ten
important, distinct, and generally applicable mental models.

## Repository layout

```text
code/
├── data_ingestion/          # acquisition, manifests and Markdown conversion
├── mental_model/   # extraction, schemas, embeddings and database code
├── crew/                    # investment-committee workflow
├── frontend/                # local Chainlit interface
└── vis/                     # mental-model visualisation

data/
├── raw/
└── processed/
```

Files under `data/` are local or generated artifacts and are excluded from Git.

## Setup

```bash
python -m venv .venvinvest
source .venvinvest/bin/activate
python -m pip install -r requirements.txt
```

Browser-based acquisition scripts also require:

```bash
python -m playwright install chromium
```

Create `.env` in the repository root:

```dotenv
OPENAI_API_KEY=your-api-key
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/agentic_investor
FRAGMENT_EXTRACTION_MODEL=gpt-5.6-terra
```

`FRAGMENT_EXTRACTION_MODEL` is optional. PostgreSQL must have the `pgvector`
extension enabled.

Create the application tables:

```bash
PYTHONPATH=code python -m mental_model.database.setup_database
```

## Run the pipeline

### 1. Build the raw corpus manifest

```bash
python code/data_ingestion/build_raw_manifest.py --check
python code/data_ingestion/build_raw_manifest.py
```

The builder validates IDs, paths and dates, calculates SHA-256 hashes, detects
duplicate IDs or content, and writes the combined manifest atomically. Add
`--strict` when warnings should fail the command.

### 2. Convert source documents to Markdown

```bash
python code/data_ingestion/convert_corpus_to_markdown.py
```

TXT files use deterministic cleaning and PDFs use Docling. Unchanged documents
are skipped; use `--force` to rebuild everything.

### 3. Validate extraction inputs without API calls

```bash
PYTHONPATH=code python -m mental_model.fragments.ingest_markdown_all \
  --dry-run --process-num 10
```

For one file:

```bash
PYTHONPATH=code python -m mental_model.fragments.ingest_markdown_all \
  --dry-run --single-run path/to/document.md
```

### 4. Extract, embed and store fragments

```bash
# One document
PYTHONPATH=code python -m mental_model.fragments.ingest_markdown_all \
  --single-run path/to/document.md

# Limited batch
PYTHONPATH=code python -m mental_model.fragments.ingest_markdown_all \
  --process-num 10

# Complete manifest
PYTHONPATH=code python -m mental_model.fragments.ingest_markdown_all
```

New documents make paid extraction and embedding requests. Existing identical
documents are skipped. Conflicting IDs and duplicate content are reported rather
than overwritten. Use `--fail-fast` or `--retry-attempts NUMBER` to adjust error
handling.

## Local frontend

```bash
cd code/frontend
python -m pip install -r requirements.txt
chainlit run app.py -w
```

Open `http://localhost:8000`. The current frontend can run with mock investor
responses and provides the integration point for the committee runner.

## Generated artifacts

| Artifact | Location |
| --- | --- |
| Raw corpus manifest | `data/raw/corpus_manifest.jsonl` |
| Canonical Markdown | `data/processed/markdown/investors/...` |
| Markdown manifest | `data/processed/markdown_manifest.jsonl` |
| Mental-model export | `data/processed/fragments/mental_model_fragments.jsonl` |

The JSONL export includes provenance, structured fragment fields, related
entities, database IDs, embedding metadata and full embedding vectors.

## Tests

```bash
PYTHONPATH=code python -m unittest \
  mental_model.fragments.test_extraction \
  mental_model.fragments.test_ingest_markdown_all -v
```

Other `test_*.py` files under `fragments/` are manual smoke tests and may require
PostgreSQL or make an OpenAI request.

## MVP boundaries

- Documents are not chunked during extraction.
- Extraction is limited to ten fragments per document.
- Tables are created directly from SQLAlchemy metadata; migrations are not yet
  managed by a migration framework.
- Retrieval, debate orchestration and final memo generation are still being
  connected to the frontend.

## Stack

Python, Pydantic, OpenAI Responses API and embeddings, SQLAlchemy, PostgreSQL,
`pgvector`, Docling, PyMuPDF, Playwright, CrewAI and Chainlit.