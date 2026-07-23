#!/usr/bin/env python3
"""
Split a combined Howard Marks memo PDF into individual PDF files and create a JSONL manifest.

Boundary detection deliberately does NOT require "Memo to:".

A page is treated as the start of a memo when, near the top of the page, it has:
    1. "From:" identifying Howard Marks
    2. "Re:" followed by a memo title
    3. A horizontal divider below the header

The divider may be:
    - text made from hyphens/dashes/underscores, or
    - an actual horizontal vector line drawn in the PDF.

Outputs:
    output_dir/
        memo_000001_<title-slug>.pdf
        memo_000002_<title-slug>.pdf
        ...
        manifest.jsonl

Dependency:
    pip install pymupdf

Example:
    python code/data_ingestion/split_marks_pdf.py \
        /data/raw/investors/marks/memos/marks_complete_collection.pdf \
        /data/raw/investors/marks/memos/individual \
        --first-file-number 1 \
        --source-url "https://www.oaktreecapital.com/insights" \
        --dry-run

Remove --dry-run after reviewing the detected start pages.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import fitz  # PyMuPDF


TOP_REGION_RATIO = 0.45
MIN_RULE_WIDTH_RATIO = 0.35
MAX_HORIZONTAL_RULE_THICKNESS_PT = 4.0
MAX_RULE_DISTANCE_BELOW_RE_PT = 180.0

FROM_PATTERN = re.compile(r"^\s*From\s*:\s*(?P<value>.*)$", re.IGNORECASE)
RE_PATTERN = re.compile(r"^\s*Re\s*:\s*(?P<value>.*)$", re.IGNORECASE)
DATE_PATTERN = re.compile(r"^\s*Date\s*:\s*(?P<value>.*)$", re.IGNORECASE)
MEMO_TO_PATTERN = re.compile(r"^\s*Memo\s+to\s*:\s*(?P<value>.*)$", re.IGNORECASE)

# A divider represented as extracted text.
TEXT_RULE_PATTERN = re.compile(r"^\s*[-‐‑‒–—_.]{15,}\s*$")

# Used only for optional publication-date extraction.
DATE_FORMATS = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%m/%d/%Y",
    "%Y-%m-%d",
)


@dataclass(frozen=True)
class VisualLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class MemoStart:
    page_index: int          # zero-based PDF page index
    title: str
    publication_date: Optional[str]
    memo_to: Optional[str]
    header_text: str
    separator_y: float


@dataclass(frozen=True)
class MemoRange:
    sequence: int            # one-based memo sequence
    start_page_index: int
    end_page_index: int
    title: str
    publication_date: Optional[str]
    memo_to: Optional[str]
    header_text: str


def normalise_text(value: str) -> str:
    """Normalise Unicode and collapse internal whitespace."""
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str, max_length: int = 90) -> str:
    """Create a conservative ASCII filename slug."""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return (value[:max_length].rstrip("_") or "untitled")


def parse_date(value: str) -> Optional[str]:
    """
    Parse an exact date into ISO YYYY-MM-DD.

    Month-only or year-only values deliberately return None because the
    manifest publication_date field is intended to hold an exact date.
    """
    cleaned = normalise_text(value)
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", cleaned, flags=re.IGNORECASE)

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue

    return None


def extract_visual_lines(page: fitz.Page) -> list[VisualLine]:
    """Extract visually ordered text lines from the top part of a page."""
    clip = fitz.Rect(
        page.rect.x0,
        page.rect.y0,
        page.rect.x1,
        page.rect.y0 + page.rect.height * TOP_REGION_RATIO,
    )

    page_dict = page.get_text("dict", clip=clip, sort=True)
    lines: list[VisualLine] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            text = normalise_text(" ".join(span.get("text", "") for span in spans))
            if not text:
                continue

            x0 = min(float(span["bbox"][0]) for span in spans)
            y0 = min(float(span["bbox"][1]) for span in spans)
            x1 = max(float(span["bbox"][2]) for span in spans)
            y1 = max(float(span["bbox"][3]) for span in spans)

            lines.append(VisualLine(text=text, x0=x0, y0=y0, x1=x1, y1=y1))

    lines.sort(key=lambda line: (round(line.y0, 1), line.x0))
    return lines


def find_vector_horizontal_rules(page: fitz.Page) -> list[float]:
    """
    Return y coordinates for long horizontal vector lines near the page top.
    """
    rules: list[float] = []
    max_y = page.rect.y0 + page.rect.height * TOP_REGION_RATIO
    min_width = page.rect.width * MIN_RULE_WIDTH_RATIO

    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            kind = item[0]

            if kind == "l":
                p1, p2 = item[1], item[2]
                width = abs(float(p2.x) - float(p1.x))
                height = abs(float(p2.y) - float(p1.y))

                if (
                    width >= min_width
                    and height <= MAX_HORIZONTAL_RULE_THICKNESS_PT
                    and max(float(p1.y), float(p2.y)) <= max_y
                ):
                    rules.append((float(p1.y) + float(p2.y)) / 2.0)

            elif kind == "re":
                rect = item[1]
                if (
                    float(rect.width) >= min_width
                    and float(rect.height) <= MAX_HORIZONTAL_RULE_THICKNESS_PT
                    and float(rect.y1) <= max_y
                ):
                    rules.append((float(rect.y0) + float(rect.y1)) / 2.0)

    return sorted(set(round(y, 2) for y in rules))


def find_text_horizontal_rules(lines: Iterable[VisualLine]) -> list[float]:
    """Return y coordinates for divider lines extracted as text."""
    return [
        (line.y0 + line.y1) / 2.0
        for line in lines
        if TEXT_RULE_PATTERN.match(line.text)
    ]


def value_from_label_line(
    lines: list[VisualLine],
    index: int,
    pattern: re.Pattern[str],
    stop_patterns: tuple[re.Pattern[str], ...],
    max_y: Optional[float] = None,
) -> str:
    """
    Read a header value after a label.

    A continuation line is accepted only when it remains above ``max_y``.
    For memo titles, ``max_y`` is the detected horizontal divider, which
    prevents the first body sentence below the divider from being appended.
    """
    match = pattern.match(lines[index].text)
    if match is None:
        return ""

    parts: list[str] = []
    inline_value = normalise_text(match.group("value"))
    if inline_value:
        parts.append(inline_value)

    for next_index in range(index + 1, min(index + 4, len(lines))):
        next_line = lines[next_index]
        candidate = next_line.text

        if max_y is not None and next_line.y0 >= max_y:
            break
        if any(stop.match(candidate) for stop in stop_patterns):
            break
        if TEXT_RULE_PATTERN.match(candidate):
            break

        previous = lines[next_index - 1]
        normal_height = max(previous.y1 - previous.y0, 1.0)

        # A large vertical gap indicates a new section rather than a wrapped
        # header value.
        if next_line.y0 - previous.y1 > normal_height * 1.8:
            break

        parts.append(candidate)

        # One continuation line is sufficient for wrapped header values.
        if len(parts) >= 2:
            break

    return normalise_text(" ".join(parts))


def detect_memo_start(page: fitz.Page) -> Optional[MemoStart]:
    """
    Detect a memo header without requiring "Memo to:".

    Required:
        - From: Howard Marks
        - Re: <non-empty title>
        - horizontal divider below the Re line
    """
    lines = extract_visual_lines(page)
    if not lines:
        return None

    from_index: Optional[int] = None
    re_index: Optional[int] = None

    for index, line in enumerate(lines):
        if from_index is None and FROM_PATTERN.match(line.text):
            author = value_from_label_line(
                lines,
                index,
                FROM_PATTERN,
                stop_patterns=(RE_PATTERN, DATE_PATTERN, MEMO_TO_PATTERN),
            )
            if re.search(r"\bHoward\s+Marks\b", author, re.IGNORECASE):
                from_index = index
                continue

        if from_index is not None and index > from_index and RE_PATTERN.match(line.text):
            re_index = index
            break

    if from_index is None or re_index is None:
        return None

    separator_candidates = (
        find_text_horizontal_rules(lines)
        + find_vector_horizontal_rules(page)
    )

    re_bottom = lines[re_index].y1
    valid_separators = [
        y
        for y in separator_candidates
        if re_bottom < y <= re_bottom + MAX_RULE_DISTANCE_BELOW_RE_PT
    ]
    if not valid_separators:
        return None

    separator_y = min(valid_separators)

    # Extract the title only from text above the divider. This allows one
    # wrapped title line while excluding the memo body below the divider.
    title = value_from_label_line(
        lines,
        re_index,
        RE_PATTERN,
        stop_patterns=(FROM_PATTERN, DATE_PATTERN, MEMO_TO_PATTERN),
        max_y=separator_y,
    )
    if not title:
        return None

    publication_date: Optional[str] = None
    memo_to: Optional[str] = None

    for index, line in enumerate(lines):
        if line.y0 >= separator_y:
            break

        date_match = DATE_PATTERN.match(line.text)
        if date_match:
            raw_date = value_from_label_line(
                lines,
                index,
                DATE_PATTERN,
                stop_patterns=(FROM_PATTERN, RE_PATTERN, MEMO_TO_PATTERN),
                max_y=separator_y,
            )
            publication_date = parse_date(raw_date)

        memo_to_match = MEMO_TO_PATTERN.match(line.text)
        if memo_to_match:
            memo_to = value_from_label_line(
                lines,
                index,
                MEMO_TO_PATTERN,
                stop_patterns=(FROM_PATTERN, RE_PATTERN, DATE_PATTERN),
                max_y=separator_y,
            ) or None

    header_text = "\n".join(
        line.text for line in lines if line.y0 < separator_y
    )

    return MemoStart(
        page_index=page.number,
        title=title,
        publication_date=publication_date,
        memo_to=memo_to,
        header_text=header_text,
        separator_y=separator_y,
    )


def find_all_memo_starts(document: fitz.Document) -> list[MemoStart]:
    starts: list[MemoStart] = []

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        detected = detect_memo_start(page)
        if detected is not None:
            starts.append(detected)

    return starts


def build_ranges(starts: list[MemoStart], page_count: int) -> list[MemoRange]:
    ranges: list[MemoRange] = []

    for index, start in enumerate(starts):
        end_page_index = (
            starts[index + 1].page_index - 1
            if index + 1 < len(starts)
            else page_count - 1
        )

        ranges.append(
            MemoRange(
                sequence=index + 1,
                start_page_index=start.page_index,
                end_page_index=end_page_index,
                title=start.title,
                publication_date=start.publication_date,
                memo_to=start.memo_to,
                header_text=start.header_text,
            )
        )

    return ranges



def write_memo_pdf(
    source_document: fitz.Document,
    start_page_index: int,
    end_page_index: int,
    output_path: Path,
) -> None:
    output_document = fitz.open()

    try:
        output_document.insert_pdf(
            source_document,
            from_page=start_page_index,
            to_page=end_page_index,
            links=True,
            annots=True,
        )
        output_document.save(
            output_path,
            garbage=4,
            deflate=True,
            clean=True,
        )
    finally:
        output_document.close()


def ensure_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite to replace generated memo files."
            )

        for item in output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)


def print_detection_report(
    starts: list[MemoStart],
    page_count: int,
) -> None:
    print(f"PDF pages: {page_count}")
    print(f"Detected memo starts: {len(starts)}")

    if not starts:
        return

    first_start = starts[0].page_index
    if first_start > 0:
        print(
            f"Warning: pages 1-{first_start} occur before the first detected "
            "memo and will be ignored."
        )

    print()
    for sequence, start in enumerate(starts, start=1):
        date_display = start.publication_date or "date not parsed"
        print(
            f"{sequence:04d} | page {start.page_index + 1:4d} | "
            f"{date_display} | {start.title}"
        )




def create_manifest_record(
    memo: MemoRange,
    file_number: int,
    source_pdf: Path,
    source_url: Optional[str],
    pdf_path: Path,
) -> dict:
    """
    Create one raw-document manifest record for an individual split memo.

    The combined collection is retained as provenance through
    ``parent_local_path`` and source-page fields, but it does not consume a
    document number. With the default --first-file-number 1, the first split
    memo is doc_marks_000001.
    """
    return {
        "document_id": f"doc_marks_{file_number:06d}",
        "author": "Howard Marks",
        "title": memo.title,
        "document_type": "investment_memo",
        "memo_to": memo.memo_to,
        "publication_date": memo.publication_date,
        "period": {
            "start_date": None,
            "end_date": None,
        },
        "source_page_start": memo.start_page_index + 1,
        "source_page_end": memo.end_page_index + 1,
        "source_url": source_url,
        "local_path": str(pdf_path),
        "parent_local_path": str(source_pdf),
        "sha256": None,
        "scraper_version": "pdf_header_split_v2",
    }

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a combined Howard Marks memo PDF into individual PDFs "
            "and create a JSONL manifest using From/Re headers and "
            "horizontal divider lines."
        )
    )
    parser.add_argument(
        "input_pdf",
        type=Path,
        help="Path to the combined Howard Marks PDF.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for split memo PDFs and manifest.jsonl.",
    )
    parser.add_argument(
        "--first-file-number",
        type=int,
        default=1,
        help=(
            "Number assigned to the first output PDF and document ID. "
            "Default: 1, producing memo_000001_...pdf and "
            "doc_marks_000001."
        ),
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional source URL stored in every manifest record.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.jsonl",
        help="Manifest filename written inside output_dir. Default: manifest.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print detected memo starts; do not write files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear a non-empty output directory before writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.first_file_number < 1:
        print("--first-file-number must be at least 1.", file=sys.stderr)
        return 2

    if not args.input_pdf.is_file():
        print(f"Input PDF not found: {args.input_pdf}", file=sys.stderr)
        return 2

    document = fitz.open(args.input_pdf)

    try:
        starts = find_all_memo_starts(document)
        print_detection_report(starts, document.page_count)

        if not starts:
            print(
                "\nNo memo boundaries were detected. Review the header "
                "format or adjust the constants near the top of the script.",
                file=sys.stderr,
            )
            return 1

        if args.dry_run:
            return 0

        ensure_output_directory(args.output_dir, args.overwrite)
        memo_ranges = build_ranges(starts, document.page_count)
        manifest_path = args.output_dir / args.manifest_name

        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            for offset, memo in enumerate(memo_ranges):
                file_number = args.first_file_number + offset
                stem = f"memo_{file_number:06d}_{slugify(memo.title)}"
                pdf_path = args.output_dir / f"{stem}.pdf"

                write_memo_pdf(
                    source_document=document,
                    start_page_index=memo.start_page_index,
                    end_page_index=memo.end_page_index,
                    output_path=pdf_path,
                )

                record = create_manifest_record(
                    memo=memo,
                    file_number=file_number,
                    source_pdf=args.input_pdf,
                    source_url=args.source_url,
                    pdf_path=pdf_path,
                )
                manifest_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        first_number = args.first_file_number
        last_number = args.first_file_number + len(memo_ranges) - 1

        print()
        print(f"Wrote {len(memo_ranges)} memo PDFs.")
        print(f"File numbers: {first_number} to {last_number}")
        print(
            f"Document IDs: doc_marks_{first_number:06d} "
            f"to doc_marks_{last_number:06d}"
        )
        print(f"Manifest: {manifest_path}")
        return 0

    finally:
        document.close()


if __name__ == "__main__":
    raise SystemExit(main())
