"""Ingest mental-model fragments for Markdown files listed in a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "markdown_manifest.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fragments"
    / "mental_model_fragments.jsonl"
)

EXPORT_SCHEMA_VERSION = "mental_model_fragment_v1"
EXTRACTION_PIPELINE_VERSION = "mental_model_extraction_v1"


@dataclass(frozen=True)
class ManifestRecord:
    """Validated fields needed from one successful manifest record."""

    document_id: str
    author: str
    markdown_local_path: str
    markdown_sha256: str
    character_count: int | None


@dataclass(frozen=True)
class PreparedDocument:
    """A manifest document whose local Markdown has been validated."""

    record: ManifestRecord
    markdown_path: Path
    markdown_text: str
    content_sha256: str
    investor_id: str


@dataclass(frozen=True)
class ProcessResult:
    """Outcome of processing one prepared document."""

    status: str
    fragment_count: int


@dataclass(frozen=True)
class Runtime:
    """Lazily loaded database and paid-API dependencies."""

    session_factory: Callable[[], Any]
    document_model: Any
    fragment_model: Any
    related_entity_model: Any
    add_document: Callable[..., Any]
    add_fragment: Callable[..., Any]
    generate_fragment_code: Callable[..., str]
    extract_fragments: Callable[..., Any]
    create_fragment_embeddings: Callable[..., list[list[float]]]
    extraction_model: str
    embedding_model: str
    embedding_dimensions: int
    retryable_exceptions: tuple[type[BaseException], ...]


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")

    return parsed


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract, embed, store, and export mental-model fragments for "
            "Markdown documents listed in a JSONL manifest."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Input Markdown manifest JSONL.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Consolidated fragment JSONL export.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and list selected Markdown files without loading the "
            "database or making any OpenAI API requests."
        ),
    )
    parser.add_argument(
        "--single-run",
        type=Path,
        metavar="PATH_TO_MD",
        help="Process only the successful manifest entry for this Markdown file.",
    )
    parser.add_argument(
        "--process-num",
        type=positive_integer,
        metavar="NUMBER",
        help="Process only the first NUMBER selected Markdown files.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=positive_integer,
        default=5,
        help="Maximum attempts for transient OpenAI API failures (default: 5).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first document failure instead of continuing.",
    )
    return parser.parse_args(argv)


def iter_jsonl_objects(
    path: Path,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield line numbers and JSON objects from a JSONL file."""

    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}: line {line_number} must contain a JSON object"
                )

            yield line_number, value


def require_non_empty_string(
    record: dict[str, Any],
    field: str,
    *,
    source: str,
) -> str:
    """Return a required non-empty string field."""

    value = record.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: missing or invalid {field!r}")

    return value.strip()


def load_successful_manifest(path: Path) -> list[ManifestRecord]:
    """Load and validate successful document records from a manifest."""

    successful: list[ManifestRecord] = []
    seen_document_ids: set[str] = set()

    for line_number, raw in iter_jsonl_objects(path):
        if raw.get("status") != "success":
            continue

        source = f"{path}: line {line_number}"
        document_id = require_non_empty_string(
            raw,
            "document_id",
            source=source,
        )

        if document_id in seen_document_ids:
            raise ValueError(f"{source}: duplicate document_id {document_id!r}")

        seen_document_ids.add(document_id)
        character_count = raw.get("character_count")

        if character_count is not None and (
            not isinstance(character_count, int) or character_count < 0
        ):
            raise ValueError(f"{source}: invalid 'character_count'")

        successful.append(
            ManifestRecord(
                document_id=document_id,
                author=require_non_empty_string(
                    raw,
                    "author",
                    source=source,
                ),
                markdown_local_path=require_non_empty_string(
                    raw,
                    "markdown_local_path",
                    source=source,
                ),
                markdown_sha256=require_non_empty_string(
                    raw,
                    "markdown_sha256",
                    source=source,
                ),
                character_count=character_count,
            )
        )

    return successful


def project_path(path: Path, *, project_root: Path) -> Path:
    """Resolve an absolute path or a path relative to the project root."""

    if path.is_absolute():
        return path.expanduser().resolve()

    return (project_root / path).resolve()


def selected_single_path(path: Path, *, project_root: Path) -> Path:
    """Resolve a user-supplied single-run path from the CWD or project root."""

    expanded = path.expanduser()

    if expanded.is_absolute():
        return expanded.resolve()

    cwd_candidate = (Path.cwd() / expanded).resolve()

    if cwd_candidate.exists():
        return cwd_candidate

    return (project_root / expanded).resolve()


def select_records(
    records: list[ManifestRecord],
    *,
    project_root: Path,
    single_run: Path | None,
    process_num: int | None,
) -> list[ManifestRecord]:
    """Apply optional single-file and count limits in manifest order."""

    selected = records

    if single_run is not None:
        requested_path = selected_single_path(
            single_run,
            project_root=project_root,
        )
        selected = [
            record
            for record in records
            if project_path(
                Path(record.markdown_local_path),
                project_root=project_root,
            )
            == requested_path
        ]

        if not selected:
            raise ValueError(
                "--single-run must match a successful markdown_local_path "
                f"in the manifest: {requested_path}"
            )

    if process_num is not None:
        selected = selected[:process_num]

    return selected


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest without loading it all at once."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


def calculate_sha256(text: str) -> str:
    """Calculate the UTF-8 SHA-256 digest of text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def remove_front_matter(markdown_text: str) -> str:
    """Remove a leading YAML front-matter block when present."""

    lines = markdown_text.splitlines()

    if not lines or lines[0].strip() != "---":
        return markdown_text

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :]).lstrip()

    raise ValueError("Front matter has no closing '---'.")


def investor_id_from_path(markdown_local_path: str) -> str:
    """Read the investor identifier following the path's investors segment."""

    parts = PurePosixPath(markdown_local_path).parts

    try:
        investor_index = parts.index("investors")
        investor_id = parts[investor_index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            "markdown_local_path must contain an 'investors/<investor_id>' "
            f"segment: {markdown_local_path}"
        ) from exc

    if not investor_id:
        raise ValueError(f"Invalid investor identifier in {markdown_local_path}")

    return investor_id


def prepare_document(
    record: ManifestRecord,
    *,
    project_root: Path,
) -> PreparedDocument:
    """Validate the source file and prepare its Markdown body for extraction."""

    markdown_path = project_path(
        Path(record.markdown_local_path),
        project_root=project_root,
    )

    try:
        markdown_path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Markdown path is outside the project root: {markdown_path}"
        ) from exc

    if not markdown_path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    if markdown_path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError(f"Source file must be Markdown: {markdown_path}")

    actual_sha256 = sha256_file(markdown_path)

    if actual_sha256 != record.markdown_sha256:
        raise ValueError(
            f"Markdown SHA-256 mismatch for {record.document_id}: "
            f"manifest={record.markdown_sha256}, actual={actual_sha256}"
        )

    complete_text = markdown_path.read_text(encoding="utf-8-sig")

    has_front_matter = complete_text.startswith("---\n") or complete_text.startswith(
        "---\r\n"
    )
    markdown_text = remove_front_matter(complete_text)
    body_character_count = len(markdown_text) + int(
        has_front_matter and complete_text.endswith("\n")
    )

    if (
        record.character_count is not None
        and body_character_count != record.character_count
    ):
        raise ValueError(
            f"Character-count mismatch for {record.document_id}: "
            f"manifest={record.character_count}, actual={body_character_count}"
        )

    if not markdown_text.strip():
        raise ValueError(f"Markdown document is empty: {markdown_path}")

    return PreparedDocument(
        record=record,
        markdown_path=markdown_path,
        markdown_text=markdown_text,
        content_sha256=calculate_sha256(markdown_text),
        investor_id=investor_id_from_path(record.markdown_local_path),
    )


def load_runtime() -> Runtime:
    """Load dependencies that require database and OpenAI configuration."""

    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    from mental_model_pipeline.database.connection import SessionLocal
    from mental_model_pipeline.fragments.db_models import (
        DocumentDB,
        MentalModelFragmentDB,
        RelatedEntityDB,
    )
    from mental_model_pipeline.fragments.embeddings import (
        EMBEDDING_DIMENSIONS,
        EMBEDDING_MODEL,
        create_fragment_embeddings,
    )
    from mental_model_pipeline.fragments.extraction import (
        EXTRACTION_MODEL,
        extract_fragments_from_document,
    )
    from mental_model_pipeline.fragments.ingest_markdown import (
        generate_fragment_code,
    )
    from mental_model_pipeline.fragments.repository import (
        add_document,
        add_fragment,
    )

    return Runtime(
        session_factory=SessionLocal,
        document_model=DocumentDB,
        fragment_model=MentalModelFragmentDB,
        related_entity_model=RelatedEntityDB,
        add_document=add_document,
        add_fragment=add_fragment,
        generate_fragment_code=generate_fragment_code,
        extract_fragments=extract_fragments_from_document,
        create_fragment_embeddings=create_fragment_embeddings,
        extraction_model=EXTRACTION_MODEL,
        embedding_model=EMBEDDING_MODEL,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        retryable_exceptions=(
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        ),
    )


def call_with_retries(
    operation: Callable[[], Any],
    *,
    retryable_exceptions: tuple[type[BaseException], ...],
    attempts: int,
    label: str,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
) -> Any:
    """Retry transient failures with capped exponential backoff and jitter."""

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except retryable_exceptions as exc:
            if attempt == attempts:
                raise

            delay = min(30.0, 2 ** (attempt - 1)) + random_value()
            print(
                f"Retrying {label} after {type(exc).__name__} "
                f"({attempt}/{attempts}) in {delay:.1f}s...",
                file=sys.stderr,
            )
            sleep(delay)

    raise AssertionError("retry loop exited unexpectedly")


def document_disposition(
    session: Any,
    runtime: Runtime,
    prepared: PreparedDocument,
) -> str:
    """Return new/existing or raise for conflicting database content."""

    document_id = prepared.record.document_id
    existing = session.get(runtime.document_model, document_id)

    if existing is not None:
        if existing.content_sha256 == prepared.content_sha256:
            return "existing"

        raise ValueError(
            f"Document {document_id} already exists with a different content hash."
        )

    from sqlalchemy import select

    duplicate_id = session.scalar(
        select(runtime.document_model.document_id).where(
            runtime.document_model.content_sha256 == prepared.content_sha256
        )
    )

    if duplicate_id is not None:
        raise ValueError(
            f"Document content for {document_id} is already stored as {duplicate_id}."
        )

    return "new"


def validate_embeddings(
    embeddings: list[list[float]],
    *,
    fragment_count: int,
    dimensions: int,
) -> None:
    """Validate embedding count and dimensions before database insertion."""

    if len(embeddings) != fragment_count:
        raise RuntimeError(
            "Embedding count does not match fragment count: "
            f"{len(embeddings)} != {fragment_count}"
        )

    for index, embedding in enumerate(embeddings):
        if len(embedding) != dimensions:
            raise RuntimeError(
                f"Embedding {index} has {len(embedding)} dimensions; "
                f"expected {dimensions}."
            )


def process_document(
    prepared: PreparedDocument,
    *,
    runtime: Runtime,
    retry_attempts: int,
) -> ProcessResult:
    """Extract, embed, and transactionally insert one prepared document."""

    with runtime.session_factory() as session:
        if document_disposition(session, runtime, prepared) == "existing":
            return ProcessResult(status="skipped", fragment_count=0)

    extraction_result = call_with_retries(
        lambda: runtime.extract_fragments(
            markdown_text=prepared.markdown_text,
            investor_name=prepared.record.author,
            document_id=prepared.record.document_id,
        ),
        retryable_exceptions=runtime.retryable_exceptions,
        attempts=retry_attempts,
        label=f"fragment extraction for {prepared.record.document_id}",
    )
    fragments = extraction_result.fragments

    if fragments:
        embeddings = call_with_retries(
            lambda: runtime.create_fragment_embeddings(fragments),
            retryable_exceptions=runtime.retryable_exceptions,
            attempts=retry_attempts,
            label=f"embedding generation for {prepared.record.document_id}",
        )
    else:
        embeddings = []

    validate_embeddings(
        embeddings,
        fragment_count=len(fragments),
        dimensions=runtime.embedding_dimensions,
    )

    with runtime.session_factory() as session:
        try:
            # Recheck after the API calls in case another process inserted it.
            if document_disposition(session, runtime, prepared) == "existing":
                session.rollback()
                return ProcessResult(status="skipped", fragment_count=0)

            runtime.add_document(
                session,
                document_id=prepared.record.document_id,
                investor_id=prepared.investor_id,
                file_path=prepared.record.markdown_local_path,
                content_sha256=prepared.content_sha256,
                markdown_text=prepared.markdown_text,
            )

            for fragment, embedding in zip(
                fragments,
                embeddings,
                strict=True,
            ):
                fragment_code = runtime.generate_fragment_code(
                    session,
                    investor_id=prepared.investor_id,
                )
                coded_fragment = fragment.model_copy(
                    update={"fragment_code": fragment_code}
                )
                runtime.add_fragment(
                    session,
                    fragment=coded_fragment,
                    document_id=prepared.record.document_id,
                    investor_id=prepared.investor_id,
                    embedding=embedding,
                    embedding_model=runtime.embedding_model,
                )

            session.commit()
        except Exception:
            session.rollback()
            raise

    return ProcessResult(status="inserted", fragment_count=len(fragments))


def existing_extraction_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Preserve extraction provenance already present in an export."""

    if not path.is_file():
        return {}

    metadata: dict[str, dict[str, Any]] = {}

    for line_number, raw in iter_jsonl_objects(path):
        document_id = raw.get("document_id")

        if not isinstance(document_id, str) or not document_id:
            raise ValueError(
                f"{path}: line {line_number} has no valid document_id"
            )

        metadata.setdefault(
            document_id,
            {
                "extraction_model": raw.get("extraction_model"),
                "extraction_version": raw.get("extraction_version"),
            },
        )

    return metadata


def json_value(value: Any) -> Any:
    """Convert common database scalar types into JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def fragment_export_record(
    *,
    document: Any,
    fragment: Any,
    entities: list[Any],
    source_metadata: ManifestRecord | None,
    extraction_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one complete, JSON-compatible fragment export record."""

    embedding = (
        [float(value) for value in fragment.embedding]
        if fragment.embedding is not None
        else None
    )
    extraction_metadata = extraction_metadata or {}

    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "document_id": document.document_id,
        "investor_id": document.investor_id,
        "author": source_metadata.author if source_metadata else None,
        "markdown_local_path": (
            source_metadata.markdown_local_path
            if source_metadata
            else document.file_path
        ),
        "markdown_sha256": (
            source_metadata.markdown_sha256 if source_metadata else None
        ),
        "content_sha256": document.content_sha256,
        "extraction_model": extraction_metadata.get("extraction_model"),
        "extraction_version": extraction_metadata.get("extraction_version"),
        "fragment_id": str(fragment.fragment_id),
        "fragment_code": fragment.fragment_code,
        "kind": fragment.kind,
        "title": fragment.title,
        "proposition": fragment.proposition,
        "mechanism": list(fragment.mechanism),
        "conditions": list(fragment.conditions),
        "failure_conditions": list(fragment.failure_conditions),
        "decision_implications": list(fragment.decision_implications),
        "decision_stages": list(fragment.decision_stages),
        "contextual_regimes": list(fragment.contextual_regimes),
        "related_entities": [
            {
                "entity_id": str(entity.entity_id),
                "entity_type": entity.entity_type,
                "name": entity.name,
                "relation": entity.relation,
                "period_text": entity.period_text,
                "explanation": entity.explanation,
                "created_at": json_value(entity.created_at),
            }
            for entity in entities
        ],
        "source_quote": fragment.source_quote,
        "evidence_strength": fragment.evidence_strength,
        "attribution_type": fragment.attribution_type,
        "attributed_to": fragment.attributed_to,
        "requires_review": fragment.requires_review,
        "review_reason": fragment.review_reason,
        "embedding": embedding,
        "embedding_model": fragment.embedding_model,
        "embedding_dimensions": len(embedding) if embedding is not None else None,
        "document_created_at": json_value(document.created_at),
        "fragment_created_at": json_value(fragment.created_at),
    }


def database_export_records(
    runtime: Runtime,
    *,
    source_by_document: dict[str, ManifestRecord],
    extraction_by_document: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read all committed fragments and related entities in stable order."""

    from sqlalchemy import select

    with runtime.session_factory() as session:
        rows = session.execute(
            select(runtime.document_model, runtime.fragment_model)
            .join(
                runtime.fragment_model,
                runtime.fragment_model.document_id
                == runtime.document_model.document_id,
            )
            .where(
                runtime.document_model.document_id.in_(source_by_document)
            )
            .order_by(
                runtime.document_model.document_id,
                runtime.fragment_model.fragment_code,
            )
        ).all()

        fragment_ids = [fragment.fragment_id for _, fragment in rows]
        entities_by_fragment: dict[Any, list[Any]] = {}

        if fragment_ids:
            entities = session.scalars(
                select(runtime.related_entity_model)
                .where(
                    runtime.related_entity_model.fragment_id.in_(fragment_ids)
                )
                .order_by(
                    runtime.related_entity_model.fragment_id,
                    runtime.related_entity_model.entity_id,
                )
            ).all()

            for entity in entities:
                entities_by_fragment.setdefault(entity.fragment_id, []).append(entity)

        return [
            fragment_export_record(
                document=document,
                fragment=fragment,
                entities=entities_by_fragment.get(fragment.fragment_id, []),
                source_metadata=source_by_document.get(document.document_id),
                extraction_metadata=extraction_by_document.get(
                    document.document_id
                ),
            )
            for document, fragment in rows
        ]


def write_jsonl_atomic(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> int:
    """Write compact JSONL through a temporary file and replace atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    written = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
                written += 1

            handle.flush()
            os.fsync(handle.fileno())

        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return written


def print_dry_run(prepared: PreparedDocument) -> None:
    """Print one validated dry-run selection."""

    print(
        "VALID "
        f"{prepared.record.document_id} "
        f"investor={prepared.investor_id} "
        f"characters={len(prepared.markdown_text):,} "
        f"path={prepared.record.markdown_local_path}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run manifest-driven fragment ingestion."""

    args = parse_arguments(argv)
    project_root = PROJECT_ROOT.resolve()
    manifest_path = project_path(
        args.manifest_path,
        project_root=project_root,
    )
    output_path = project_path(
        args.output_path,
        project_root=project_root,
    )
    manifest_records = load_successful_manifest(manifest_path)
    selected = select_records(
        manifest_records,
        project_root=project_root,
        single_run=args.single_run,
        process_num=args.process_num,
    )

    print(f"Manifest: {manifest_path}")
    print(f"Successful manifest entries: {len(manifest_records)}")
    print(f"Selected entries: {len(selected)}")

    if not selected:
        print("No successful manifest entries were selected.")
        return 0

    if args.dry_run:
        failures = 0
        validated = 0

        for record in selected:
            try:
                print_dry_run(
                    prepare_document(
                        record,
                        project_root=project_root,
                    )
                )
                validated += 1
            except Exception as exc:
                failures += 1
                print(f"FAILED {record.document_id}: {exc}", file=sys.stderr)

                if args.fail_fast:
                    break

        print(
            "Dry run complete: "
            f"validated={validated}, failed={failures}. "
            "No database or OpenAI API calls were made."
        )
        return 1 if failures else 0

    runtime = load_runtime()
    extraction_by_document = existing_extraction_metadata(output_path)
    source_by_document = {
        record.document_id: record for record in manifest_records
    }
    inserted = 0
    skipped = 0
    failed = 0
    fragments_inserted = 0

    print(f"Extraction model: {runtime.extraction_model}")
    print(f"Embedding model: {runtime.embedding_model}")

    for index, record in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {record.document_id}")

        try:
            prepared = prepare_document(
                record,
                project_root=project_root,
            )
            result = process_document(
                prepared,
                runtime=runtime,
                retry_attempts=args.retry_attempts,
            )

            if result.status == "skipped":
                skipped += 1
                print("  skipped: identical document is already stored")
            else:
                inserted += 1
                fragments_inserted += result.fragment_count
                extraction_by_document[record.document_id] = {
                    "extraction_model": runtime.extraction_model,
                    "extraction_version": EXTRACTION_PIPELINE_VERSION,
                }
                print(f"  inserted fragments: {result.fragment_count}")
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}", file=sys.stderr)

            if args.fail_fast:
                break

    export_records = database_export_records(
        runtime,
        source_by_document=source_by_document,
        extraction_by_document=extraction_by_document,
    )
    exported = write_jsonl_atomic(output_path, export_records)

    print()
    print(f"Fragment export: {output_path}")
    print(f"Exported fragment records: {exported}")
    print(
        "Ingestion summary: "
        f"inserted_documents={inserted}, "
        f"inserted_fragments={fragments_inserted}, "
        f"skipped_documents={skipped}, "
        f"failed_documents={failed}"
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
