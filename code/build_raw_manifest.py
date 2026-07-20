#!/usr/bin/env python3
"""
Build the top-level raw corpus manifest from investor-level manifests.

Local investor manifests are authoritative whitelists:
- Only files listed in a local manifest are included.
- Unlisted files are ignored.
- Investor folders and names are discovered dynamically.
- Local manifests are never modified.
- The top-level manifest is rebuilt on every successful run.
- SHA-256 hashes are calculated for listed files.

Expected placement:
    project_root/code/build_raw_manifest_v2.py

Expected manifests:
    project_root/data/raw/investors/**/manifest.jsonl

Local manifests must use strict JSONL:
    one complete JSON object per physical line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TOP_MANIFEST_NAME = "corpus_manifest.jsonl"
LOCAL_MANIFEST_NAMES = {"manifest.jsonl", "manifest.json"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/raw/corpus_manifest.jsonl from local investor manifests."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to the parent of the script's code directory.",
    )
    parser.add_argument(
        "--investors-dir",
        type=Path,
        default=None,
        help="Defaults to <project-root>/data/raw/investors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <project-root>/data/raw/corpus_manifest.jsonl.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate without writing the top-level manifest.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 if warnings are present.",
    )
    return parser.parse_args()


def infer_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def discover_manifests(investors_dir: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in investors_dir.rglob("*")
        if path.is_file() and path.name.lower() in LOCAL_MANIFEST_NAMES
    )


def load_jsonl(path: Path, report: Report) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        report.error(f"{path}: cannot open manifest: {exc}")
        return records

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    f"line {line_number}, column {exc.colno}: {exc.msg}"
                )
                continue

            if not isinstance(value, dict):
                failures.append(
                    f"line {line_number}: expected an object, "
                    f"got {type(value).__name__}"
                )
                continue

            value["_manifest_path"] = str(path)
            value["_manifest_line"] = line_number
            records.append(value)

    if failures:
        preview = "; ".join(failures[:5])
        extra = len(failures) - min(5, len(failures))
        if extra:
            preview += f"; plus {extra} more"
        report.error(
            f"{path}: invalid JSONL: {preview}. "
            "Each record must occupy one complete line."
        )

    return records


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalise_path_text(value: str) -> str:
    text = value.strip().replace("\\", "/")
    text = re.sub(r"^\./+", "", text)
    return text


def part_below_data_raw(value: str) -> str | None:
    """
    Extract the portion below data/raw from either:
      data/raw/investors/...
      /data/raw/investors/...
      /some/project/data/raw/investors/...
    """
    match = re.search(r"(?:^|/)data/raw/(.+)$", value)
    return match.group(1) if match else None


def candidate_paths(
    raw_value: str,
    *,
    manifest_path: Path,
    project_root: Path,
    raw_dir: Path,
) -> list[Path]:
    """
    Produce all reasonable interpretations of a manifest local_path.

    Supports:
    - data/raw/investors/...
    - /data/raw/investors/...
    - investors/...
    - investor-folder-relative paths
    - project-relative paths
    - genuine absolute paths
    """
    text = normalise_path_text(raw_value)
    candidate = Path(text).expanduser()
    candidates: list[Path] = []

    below_raw = part_below_data_raw(text)
    if below_raw is not None:
        candidates.append(raw_dir / below_raw)
        candidates.append(project_root / "data" / "raw" / below_raw)

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        if text.startswith("investors/"):
            candidates.append(raw_dir / candidate)

        candidates.append(project_root / candidate)
        candidates.append(manifest_path.parent / candidate)
        candidates.append(raw_dir / candidate)

    unique: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()

        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)

    return unique


def find_unique_basename_match(
    raw_value: str,
    *,
    manifest_path: Path,
    investors_dir: Path,
) -> Path | None:
    """
    Diagnostic fallback only.

    Search within the investor folder containing the manifest. If exactly one
    file has the requested basename, use it and emit a warning. This catches
    stale folder prefixes without silently choosing between duplicates.
    """
    basename = Path(normalise_path_text(raw_value)).name
    if not basename:
        return None

    matches = [
        path.resolve()
        for path in manifest_path.parent.rglob(basename)
        if path.is_file() and path.name not in LOCAL_MANIFEST_NAMES
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def resolve_local_path(
    raw_value: Any,
    *,
    manifest_path: Path,
    project_root: Path,
    raw_dir: Path,
    investors_dir: Path,
    report: Report,
    context: str,
) -> tuple[Path | None, list[Path]]:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None, []

    candidates = candidate_paths(
        raw_value,
        manifest_path=manifest_path,
        project_root=project_root,
        raw_dir=raw_dir,
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate, candidates

    basename_match = find_unique_basename_match(
        raw_value,
        manifest_path=manifest_path,
        investors_dir=investors_dir,
    )

    if basename_match is not None:
        report.warning(
            f"{context}: manifest path {raw_value!r} did not resolve exactly; "
            f"using the unique matching file {basename_match}. "
            "Update local_path in the local manifest."
        )
        return basename_match, candidates

    return None, candidates


def canonical_project_path(path: Path, project_root: Path) -> str | None:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return None


def validate_date(
    value: Any,
    *,
    field_name: str,
    context: str,
    report: Report,
) -> bool:
    if value is None:
        return True

    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        report.warning(
            f"{context}: {field_name} should be null or YYYY-MM-DD; "
            f"got {value!r}"
        )
        return False

    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        report.warning(
            f"{context}: {field_name} is not a valid calendar date: {value!r}"
        )
        return False

    return True


def validate_dates(record: dict[str, Any], context: str, report: Report) -> None:
    validate_date(
        record.get("publication_date"),
        field_name="publication_date",
        context=context,
        report=report,
    )

    period = record.get("period")
    if period is None:
        return

    if not isinstance(period, dict):
        report.warning(f"{context}: period should be an object or null")
        return

    start = period.get("start_date")
    end = period.get("end_date")

    start_valid = validate_date(
        start,
        field_name="period.start_date",
        context=context,
        report=report,
    )
    end_valid = validate_date(
        end,
        field_name="period.end_date",
        context=context,
        report=report,
    )

    if (start is None) != (end is None):
        report.warning(
            f"{context}: period.start_date and period.end_date should both be "
            "populated or both be null"
        )

    if (
        start_valid
        and end_valid
        and isinstance(start, str)
        and isinstance(end, str)
        and start > end
    ):
        report.warning(
            f"{context}: period.start_date is later than period.end_date"
        )


def clean_record(
    source: dict[str, Any],
    *,
    local_path: str,
    sha256: str,
) -> dict[str, Any]:
    record = {
        key: value
        for key, value in source.items()
        if not key.startswith("_")
    }
    record["local_path"] = local_path
    record["sha256"] = sha256

    preferred = [
        "document_id",
        "author",
        "title",
        "document_type",
        "memo_to",
        "publication_date",
        "period",
        "source_page_start",
        "source_page_end",
        "source_url",
        "local_path",
        "parent_local_path",
        "sha256",
        "scraper_version",
    ]

    ordered: dict[str, Any] = {}
    for key in preferred:
        if key in record:
            ordered[key] = record.pop(key)

    for key in sorted(record):
        ordered[key] = record[key]

    return ordered


def build_records(
    *,
    manifests: list[Path],
    project_root: Path,
    raw_dir: Path,
    investors_dir: Path,
    report: Report,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    seen_ids: dict[str, str] = {}
    seen_paths: dict[str, str] = {}
    hashes: defaultdict[str, list[str]] = defaultdict(list)

    for manifest in manifests:
        for record in load_jsonl(manifest, report):
            line_number = record["_manifest_line"]
            context = f"{manifest}: line {line_number}"

            document_id = record.get("document_id")
            if not isinstance(document_id, str) or not document_id.strip():
                report.error(f"{context}: missing non-empty document_id")
                continue
            document_id = document_id.strip()

            resolved, attempted = resolve_local_path(
                record.get("local_path"),
                manifest_path=manifest,
                project_root=project_root,
                raw_dir=raw_dir,
                investors_dir=investors_dir,
                report=report,
                context=context,
            )

            if resolved is None:
                attempted_text = "\n      ".join(str(path) for path in attempted)
                report.error(
                    f"{context}: listed file was not found.\n"
                    f"    manifest local_path: {record.get('local_path')!r}\n"
                    f"    paths tested:\n      {attempted_text or '(none)'}"
                )
                continue

            canonical = canonical_project_path(resolved, project_root)
            if canonical is None:
                report.error(
                    f"{context}: resolved file is outside the project root: "
                    f"{resolved}"
                )
                continue

            try:
                resolved.relative_to(raw_dir)
            except ValueError:
                report.error(
                    f"{context}: resolved file is outside data/raw: {resolved}"
                )
                continue

            if resolved.stat().st_size == 0:
                report.error(f"{context}: listed file is empty: {canonical}")
                continue

            if document_id in seen_ids:
                report.error(
                    f"{context}: duplicate document_id {document_id!r}; "
                    f"already used by {seen_ids[document_id]}"
                )
                continue
            seen_ids[document_id] = canonical

            if canonical in seen_paths:
                report.error(
                    f"{context}: duplicate local_path {canonical!r}; "
                    f"already listed by {seen_paths[canonical]}"
                )
                continue
            seen_paths[canonical] = str(manifest)

            validate_dates(record, context, report)

            try:
                file_hash = sha256_file(resolved)
            except OSError as exc:
                report.error(
                    f"{context}: could not hash {canonical}: {exc}"
                )
                continue

            hashes[file_hash].append(canonical)
            output.append(
                clean_record(
                    record,
                    local_path=canonical,
                    sha256=file_hash,
                )
            )

    for file_hash, paths in sorted(hashes.items()):
        if len(paths) > 1:
            report.warning(
                f"Duplicate file content, sha256={file_hash}: "
                + ", ".join(sorted(paths))
            )

    output.sort(
        key=lambda record: (
            str(record.get("author") or ""),
            str(record.get("document_id") or ""),
        )
    )
    return output


def serialise_jsonl(records: Iterable[dict[str, Any]]) -> bytes:
    text = "\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    )
    return (text + "\n").encode("utf-8") if text else b""


def write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()

    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root
        else infer_project_root()
    )
    raw_dir = (project_root / "data" / "raw").resolve()
    investors_dir = (
        args.investors_dir.expanduser().resolve()
        if args.investors_dir
        else (raw_dir / "investors").resolve()
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else (raw_dir / TOP_MANIFEST_NAME).resolve()
    )

    print(f"Project root:  {project_root}")
    print(f"Investors dir: {investors_dir}")
    print(f"Output:        {output_path}")

    if not investors_dir.is_dir():
        print(
            f"ERROR: investors directory does not exist: {investors_dir}",
            file=sys.stderr,
        )
        return 2

    manifests = discover_manifests(investors_dir)
    print(f"Manifests:     {len(manifests)}")

    if not manifests:
        print(
            f"ERROR: no local manifests found below {investors_dir}",
            file=sys.stderr,
        )
        return 2

    report = Report()
    records = build_records(
        manifests=manifests,
        project_root=project_root,
        raw_dir=raw_dir,
        investors_dir=investors_dir,
        report=report,
    )

    generated = serialise_jsonl(records)
    existing = output_path.read_bytes() if output_path.exists() else None
    changed = generated != existing

    if not args.check and not report.errors and changed:
        write_atomic(output_path, generated)

    print(f"\nValid records: {len(records)}")
    print(
        "Status:        "
        + (
            "would update output"
            if args.check and changed
            else "output updated"
            if changed and not report.errors
            else "already up to date"
            if not changed
            else "not written because of errors"
        )
    )
    print(
        f"Validation:    {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s)"
    )

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")

    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"  - {error}")
        print("\nThe top-level manifest was not modified.")
        return 1

    if args.strict and report.warnings:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
