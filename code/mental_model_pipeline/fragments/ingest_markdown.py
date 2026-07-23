"""Extract, embed, and store mental-model fragments from Markdown."""

from __future__ import annotations

import argparse
import hashlib
import re
import secrets
import string
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from mental_model_pipeline.database.connection import SessionLocal
from mental_model_pipeline.fragments.db_models import (
    DocumentDB,
    MentalModelFragmentDB,
)
from mental_model_pipeline.fragments.embeddings import (
    EMBEDDING_MODEL,
    create_fragment_embeddings,
)
from mental_model_pipeline.fragments.extraction import (
    EXTRACTION_MODEL,
    extract_fragments_from_document,
)
from mental_model_pipeline.fragments.repository import (
    add_document,
    add_fragment,
)
from mental_model_pipeline.fragments.schemas import (
    MentalModelFragment,
)


FRAGMENT_CODE_ALPHABET = (
    string.ascii_lowercase
    + string.digits
)


def calculate_sha256(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def normalise_investor_code(
    investor_id: str,
) -> str:
    investor_code = re.sub(
        r"[^a-z0-9]",
        "",
        investor_id.lower(),
    )

    if not investor_code:
        raise ValueError(
            "investor_id must contain letters or numbers."
        )

    return investor_code


def generate_fragment_code(
    session: Session,
    *,
    investor_id: str,
) -> str:
    investor_code = normalise_investor_code(
        investor_id
    )

    for _ in range(200):
        suffix = "".join(
            secrets.choice(FRAGMENT_CODE_ALPHABET)
            for _ in range(3)
        )

        fragment_code = (
            f"mmf_{investor_code}_{suffix}"
        )

        existing_id = session.scalar(
            select(
                MentalModelFragmentDB.fragment_id
            ).where(
                MentalModelFragmentDB.fragment_code
                == fragment_code
            )
        )

        if existing_id is None:
            return fragment_code

    raise RuntimeError(
        "Could not generate a unique fragment code."
    )


def ensure_document_is_new(
    session: Session,
    *,
    document_id: str,
    content_sha256: str,
) -> None:
    existing_document = session.scalar(
        select(DocumentDB.document_id).where(
            DocumentDB.document_id
            == document_id
        )
    )

    if existing_document is not None:
        raise ValueError(
            f"Document already exists: {document_id}"
        )

    duplicate_content = session.scalar(
        select(DocumentDB.document_id).where(
            DocumentDB.content_sha256
            == content_sha256
        )
    )

    if duplicate_content is not None:
        raise ValueError(
            "The same document content is already stored as "
            f"{duplicate_content}."
        )


def insert_document_and_fragments(
    *,
    markdown_path: Path,
    markdown_text: str,
    document_id: str,
    investor_id: str,
    fragments: list[MentalModelFragment],
) -> int:
    content_sha256 = calculate_sha256(
        markdown_text
    )

    print(
        f"Generating {len(fragments)} embeddings..."
    )

    embeddings = create_fragment_embeddings(
        fragments
    )

    with SessionLocal() as session:
        try:
            ensure_document_is_new(
                session,
                document_id=document_id,
                content_sha256=content_sha256,
            )

            add_document(
                session,
                document_id=document_id,
                investor_id=investor_id,
                file_path=str(markdown_path),
                content_sha256=content_sha256,
                markdown_text=markdown_text,
            )

            for fragment, embedding in zip(
                fragments,
                embeddings,
                strict=True,
            ):
                fragment_code = generate_fragment_code(
                    session,
                    investor_id=investor_id,
                )

                coded_fragment = fragment.model_copy(
                    update={
                        "fragment_code": fragment_code,
                    }
                )

                add_fragment(
                    session,
                    fragment=coded_fragment,
                    document_id=document_id,
                    investor_id=investor_id,
                    embedding=embedding,
                    embedding_model=EMBEDDING_MODEL,
                )

            session.commit()

            return len(fragments)

        except Exception:
            session.rollback()
            raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract up to ten mental-model fragments "
            "from one Markdown document."
        )
    )

    parser.add_argument(
        "markdown_path",
        type=Path,
        help="Path to the Markdown document.",
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help="Stable document identifier.",
    )

    parser.add_argument(
        "--investor-id",
        required=True,
        help="Stable investor identifier.",
    )

    parser.add_argument(
        "--investor-name",
        help=(
            "Investor display name. Defaults to investor-id."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Extract fragments but do not generate embeddings "
            "or insert records."
        ),
    )

    return parser.parse_args()


def remove_front_matter(markdown_text: str) -> str:
    lines = markdown_text.splitlines()

    if not lines or lines[0].strip() != "---":
        return markdown_text

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1:]).lstrip()

    raise ValueError("Front matter has no closing '---'.")


def main() -> None:
    args = parse_arguments()

    markdown_path = args.markdown_path.resolve()

    if not markdown_path.exists():
        raise FileNotFoundError(
            f"Markdown file not found: {markdown_path}"
        )

    if markdown_path.suffix.lower() not in {
        ".md",
        ".markdown",
    }:
        raise ValueError(
            "The source file must be Markdown."
        )

    complete_markdown_text = markdown_path.read_text(
    encoding="utf-8"
    )

    markdown_text = remove_front_matter(
        complete_markdown_text
    )

    if not markdown_text.strip():
        raise ValueError(
            "The Markdown document is empty."
        )

    content_sha256 = calculate_sha256(
        markdown_text
    )

    # Check duplicates before making a paid API call.
    with SessionLocal() as session:
        ensure_document_is_new(
            session,
            document_id=args.document_id,
            content_sha256=content_sha256,
        )

    investor_name = (
        args.investor_name
        or args.investor_id
    )

    print(f"Document: {args.document_id}")
    print(f"Investor: {investor_name}")
    print(f"Characters: {len(markdown_text):,}")
    print(f"Extraction model: {EXTRACTION_MODEL}")

    extraction_result = (
        extract_fragments_from_document(
            markdown_text=markdown_text,
            investor_name=investor_name,
            document_id=args.document_id,
        )
    )

    print(
        "Fragments extracted:",
        len(extraction_result.fragments),
    )

    if args.dry_run:
        print(
            extraction_result.model_dump_json(
                indent=2
            )
        )

        print(
            "Dry run complete. Nothing was inserted."
        )
        return

    inserted_count = insert_document_and_fragments(
        markdown_path=markdown_path,
        markdown_text=markdown_text,
        document_id=args.document_id,
        investor_id=args.investor_id,
        fragments=extraction_result.fragments,
    )

    print()
    print("Ingestion completed successfully.")
    print(f"Document ID: {args.document_id}")
    print(f"Fragments inserted: {inserted_count}")
    print(f"Embedding model: {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
