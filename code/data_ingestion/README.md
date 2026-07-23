# Data Ingestion Scripts

This flat directory contains scripts that build and prepare the source corpus.

| Script | Purpose |
| --- | --- |
| `build_raw_manifest.py` | Validate investor manifests and build `data/raw/corpus_manifest.jsonl`. |
| `convert_corpus_to_markdown.py` | Convert manifest-listed TXT and PDF sources into canonical Markdown. |
| `split_marks_pdf.py` | Split the combined Howard Marks memo PDF into individual PDFs and a manifest. |
| `download_flatt.py` | Download manifest-listed Brookfield shareholder letters. |
| `flatt_download_manifest.json` | Input and updated-source metadata for `download_flatt.py`. |

Run the first two scripts from the project root:

```bash
python code/data_ingestion/build_raw_manifest.py --check
python code/data_ingestion/convert_corpus_to_markdown.py
```

The Howard Marks splitter takes an input PDF and output directory explicitly:

```bash
python code/data_ingestion/split_marks_pdf.py \
  data/raw/investors/marks/memos/marks_complete_collection.pdf \
  data/raw/investors/marks/memos_split \
  --dry-run
```

The Brookfield downloader makes network requests and updates
`flatt_download_manifest.json`; it writes PDFs under `data/raw/`:

```bash
python code/data_ingestion/download_flatt.py
```
