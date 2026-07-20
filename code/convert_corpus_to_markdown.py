#!/usr/bin/env python3
"""
Convert the raw corpus into one Markdown file per manifest-listed document.

Input:
    data/raw/corpus_manifest.jsonl

Outputs:
    data/processed/markdown/investors/...
    data/processed/markdown_manifest.jsonl

Rules:
- TXT files use deterministic text cleaning.
- PDF files use Docling.
- The investor folder structure is preserved.
- Documents are not split.
- Unchanged successful documents are skipped on reruns.
- Change CONVERSION_VERSION when conversion logic changes and all documents
  should be regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


CONVERSION_VERSION = "source_to_markdown_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert every document.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")

            records.append(record)

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")

    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


def yaml_value(value: Any) -> str:
    # JSON scalar and flow syntax are valid YAML.
    return json.dumps(value, ensure_ascii=False)


def front_matter(record: dict[str, Any]) -> str:
    fields = {
        "document_id": record.get("document_id"),
        "author": record.get("author"),
        "title": record.get("title"),
        "document_type": record.get("document_type"),
        "publication_date": record.get("publication_date"),
        "period": record.get("period"),
        "source_local_path": record.get("local_path"),
        "source_sha256": record.get("sha256"),
        "conversion_version": CONVERSION_VERSION,
    }

    lines = ["---"]
    lines.extend(f"{key}: {yaml_value(value)}" for key, value in fields.items())
    lines.extend(["---", ""])
    return "\n".join(lines)


def txt_to_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def pdf_to_markdown(path: Path, converter: Any) -> str:
    result = converter.convert(path)
    return result.document.export_to_markdown().strip() + "\n"


def output_path_for(
    source_path: Path,
    raw_root: Path,
    markdown_root: Path,
) -> Path:
    relative = source_path.relative_to(raw_root)
    return (markdown_root / relative).with_suffix(".md")


def successful_and_current(
    old: dict[str, Any] | None,
    *,
    source_sha256: str,
    output_path: Path,
) -> bool:
    return bool(
        old
        and old.get("status") == "success"
        and old.get("source_sha256") == source_sha256
        and old.get("conversion_version") == CONVERSION_VERSION
        and output_path.is_file()
    )


def processed_record(
    source_record: dict[str, Any],
    *,
    markdown_path: Path,
    project_root: Path,
    markdown_sha256: str | None,
    method: str,
    status: str,
    character_count: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    record = dict(source_record)
    record["source_local_path"] = record.pop("local_path")
    record["source_sha256"] = record.pop("sha256")
    record["markdown_local_path"] = markdown_path.relative_to(project_root).as_posix()
    record["markdown_sha256"] = markdown_sha256
    record["conversion_method"] = method
    record["conversion_version"] = CONVERSION_VERSION
    record["status"] = status
    record["character_count"] = character_count
    record["error"] = error
    return record


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()

    raw_root = project_root / "data" / "raw"
    input_manifest = raw_root / "corpus_manifest.jsonl"
    markdown_root = project_root / "data" / "processed" / "markdown"
    output_manifest = project_root / "data" / "processed" / "markdown_manifest.jsonl"

    if not input_manifest.is_file():
        print(f"ERROR: manifest not found: {input_manifest}", file=sys.stderr)
        return 2

    source_records = read_jsonl(input_manifest)
    old_records = read_jsonl(output_manifest) if output_manifest.is_file() else []
    old_by_id = {
        record["document_id"]: record
        for record in old_records
        if record.get("document_id")
    }

    converter = None
    output_records: list[dict[str, Any]] = []
    converted = 0
    skipped = 0
    failed = 0

    for source_record in source_records:
        document_id = source_record.get("document_id")
        local_path = source_record.get("local_path")
        source_sha256 = source_record.get("sha256")

        if not document_id or not local_path or not source_sha256:
            print(f"FAILED: incomplete manifest record: {document_id!r}")
            failed += 1
            continue

        source_path = (project_root / local_path).resolve()
        markdown_path = output_path_for(source_path, raw_root, markdown_root)
        suffix = source_path.suffix.lower()

        if suffix == ".txt":
            method = "deterministic_txt"
        elif suffix == ".pdf":
            method = "docling"
        else:
            print(f"FAILED {document_id}: unsupported format {suffix}")
            failed += 1
            continue

        old = old_by_id.get(document_id)

        if (
            not args.force
            and successful_and_current(
                old,
                source_sha256=source_sha256,
                output_path=markdown_path,
            )
        ):
            output_records.append(old)
            skipped += 1
            continue

        try:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)

            if suffix == ".txt":
                body = txt_to_markdown(source_path)
            else:
                if converter is None:
                    try:
                        from docling.document_converter import DocumentConverter
                    except ImportError as exc:
                        raise RuntimeError(
                            "Docling is not installed. Run: pip install docling"
                        ) from exc

                    converter = DocumentConverter()

                body = pdf_to_markdown(source_path, converter)

            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(
                front_matter(source_record) + body,
                encoding="utf-8",
                newline="\n",
            )

            output_records.append(
                processed_record(
                    source_record,
                    markdown_path=markdown_path,
                    project_root=project_root,
                    markdown_sha256=sha256_file(markdown_path),
                    method=method,
                    status="success",
                    character_count=len(body),
                )
            )
            converted += 1
            print(
                f"OK      {document_id}: "
                f"{markdown_path.relative_to(project_root)}"
            )

        except Exception as exc:
            output_records.append(
                processed_record(
                    source_record,
                    markdown_path=markdown_path,
                    project_root=project_root,
                    markdown_sha256=None,
                    method=method,
                    status="failed",
                    error=str(exc),
                )
            )
            failed += 1
            print(f"FAILED  {document_id}: {exc}")

    write_jsonl(output_manifest, output_records)

    print(
        f"\nConverted: {converted} | Skipped: {skipped} | "
        f"Failed: {failed} | Total: {len(source_records)}"
    )
    print(f"Manifest: {output_manifest}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
