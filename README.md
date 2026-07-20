# Agentic Investor

Agentic Investor is an experimental investment-research system designed to extract and structure the reasoning frameworks of investors such as Warren Buffett, Charlie Munger, Howard Marks, Bruce Flatt, Mohnish Pabrai, and Guy Spier.

The long-term goal is to build an agentic investment committee in which investor-specific agents retrieve relevant mental models, debate an investment case, and contribute to a consolidated investment memo.

## Current Scope

The project currently covers the ingestion pipeline:

```text
raw documents
→ validated corpus manifest
→ TXT/PDF conversion
→ canonical Markdown
→ processed manifest
```

Documents remain intact at this stage. Semantic splitting, mental-model extraction, embeddings, retrieval, and agent workflows will be added later.

## Installation

```bash
python -m venv .venvinvest
source .venvinvest/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Build the Raw Manifest

```bash
python code/build_raw_manifest.py --check
python code/build_raw_manifest.py
```

This validates the investor manifests, verifies listed files, calculates SHA-256 hashes, detects duplicates, and generates the top-level corpus manifest.

## Convert to Markdown

```bash
python code/convert_corpus_to_markdown.py
```

TXT files are processed deterministically, while PDFs are converted using Docling. The script preserves document structure, skips unchanged files on reruns, and maintains a processed Markdown manifest.

To reconvert all documents:

```bash
python code/convert_corpus_to_markdown.py --force
```

## Dependencies

```text
docling
PyMuPDF
playwright
```
