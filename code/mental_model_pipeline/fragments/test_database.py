"""Smoke-test document insertion and retrieval."""

from sqlalchemy import select

from mental_model_pipeline.database.connection import SessionLocal
from mental_model_pipeline.fragments.db_models import DocumentDB


def test_document_insert() -> None:
    with SessionLocal() as session:
        existing_document = session.get(
            DocumentDB,
            "doc_test_001",
        )

        if existing_document is None:
            document = DocumentDB(
                document_id="doc_test_001",
                investor_id="buffett",
                file_path="data/test_document.md",
                content_sha256="0" * 64,
                markdown_text="# Test document\n\nTest content.",
            )

            session.add(document)
            session.commit()

        result = session.execute(
            select(DocumentDB).where(
                DocumentDB.document_id == "doc_test_001"
            )
        ).scalar_one()

        print("Document ID:", result.document_id)
        print("Investor:", result.investor_id)
        print("Text:", result.markdown_text)


if __name__ == "__main__":
    test_document_insert()
