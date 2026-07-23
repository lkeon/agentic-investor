"""Test manifest-driven fragment ingestion without external services."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID

from mental_model.fragments import ingest_markdown_all as batch_ingest


class ManifestFixture:
    """Create a small valid Markdown manifest under a temporary project root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: list[dict[str, object]] = []

    def add_document(
        self,
        *,
        document_id: str,
        investor_id: str,
        body: str,
    ) -> Path:
        relative_path = Path(
            "data",
            "processed",
            "markdown",
            "investors",
            investor_id,
            f"{document_id}.md",
        )
        absolute_path = self.root / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text(body, encoding="utf-8")
        self.records.append(
            {
                "document_id": document_id,
                "author": investor_id.title(),
                "markdown_local_path": relative_path.as_posix(),
                "markdown_sha256": hashlib.sha256(
                    body.encode("utf-8")
                ).hexdigest(),
                "status": "success",
                "character_count": len(body),
            }
        )
        return absolute_path

    def write_manifest(self) -> Path:
        path = self.root / "data" / "processed" / "markdown_manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(record, separators=(",", ":")) + "\n"
                for record in self.records
            ),
            encoding="utf-8",
        )
        return path


class IngestMarkdownAllTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        self.fixture = ManifestFixture(self.project_root)
        self.first_path = self.fixture.add_document(
            document_id="doc_test_001",
            investor_id="buffett",
            body="A durable advantage can protect long-term returns.",
        )
        self.second_path = self.fixture.add_document(
            document_id="doc_test_002",
            investor_id="marks",
            body="Risk depends on the price paid as well as asset quality.",
        )
        self.manifest_path = self.fixture.write_manifest()
        self.output_path = (
            self.project_root
            / "data"
            / "processed"
            / "fragments"
            / "mental_model_fragments.jsonl"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_dry(self, *extra_arguments: str) -> tuple[int, str]:
        arguments = [
            "--manifest-path",
            str(self.manifest_path),
            "--output-path",
            str(self.output_path),
            "--dry-run",
            *extra_arguments,
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch.object(batch_ingest, "PROJECT_ROOT", self.project_root),
            patch.object(
                batch_ingest,
                "load_runtime",
                side_effect=AssertionError("dry-run loaded paid runtime"),
            ),
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            result = batch_ingest.main(arguments)

        return result, stdout.getvalue()

    def test_dry_run_makes_no_api_or_database_calls(self) -> None:
        result, output = self.run_dry()

        self.assertEqual(result, 0)
        self.assertIn("validated=2, failed=0", output)
        self.assertIn("No database or OpenAI API calls were made", output)
        self.assertFalse(self.output_path.exists())

    def test_single_run_selects_only_the_requested_manifest_file(self) -> None:
        result, output = self.run_dry("--single-run", str(self.second_path))

        self.assertEqual(result, 0)
        self.assertIn("Selected entries: 1", output)
        self.assertIn("doc_test_002", output)
        self.assertNotIn("doc_test_001 investor=", output)

    def test_process_num_limits_documents_in_manifest_order(self) -> None:
        result, output = self.run_dry("--process-num", "1")

        self.assertEqual(result, 0)
        self.assertIn("Selected entries: 1", output)
        self.assertIn("doc_test_001", output)
        self.assertNotIn("doc_test_002 investor=", output)

    def test_live_orchestration_processes_limit_and_exports_database(self) -> None:
        runtime = SimpleNamespace(
            extraction_model="test-extraction",
            embedding_model="test-embedding",
        )
        process = Mock(
            return_value=batch_ingest.ProcessResult(
                status="inserted",
                fragment_count=2,
            )
        )
        export = Mock(return_value=[{"fragment_code": "mmf_test_001"}])
        write = Mock(return_value=1)
        stdout = io.StringIO()

        with (
            patch.object(batch_ingest, "PROJECT_ROOT", self.project_root),
            patch.object(batch_ingest, "load_runtime", return_value=runtime),
            patch.object(batch_ingest, "process_document", process),
            patch.object(batch_ingest, "database_export_records", export),
            patch.object(batch_ingest, "write_jsonl_atomic", write),
            redirect_stdout(stdout),
        ):
            result = batch_ingest.main(
                [
                    "--manifest-path",
                    str(self.manifest_path),
                    "--output-path",
                    str(self.output_path),
                    "--process-num",
                    "1",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(process.call_count, 1)
        self.assertEqual(
            process.call_args.args[0].record.document_id,
            "doc_test_001",
        )
        export.assert_called_once()
        write.assert_called_once_with(
            self.output_path,
            [{"fragment_code": "mmf_test_001"}],
        )
        self.assertIn("inserted_documents=1", stdout.getvalue())

    def test_hash_mismatch_fails_before_runtime_loading(self) -> None:
        self.first_path.write_text("Changed text", encoding="utf-8")

        result, output = self.run_dry("--process-num", "1")

        self.assertEqual(result, 1)
        self.assertIn("validated=0, failed=1", output)
        self.assertFalse(self.output_path.exists())

    def test_retry_uses_exponential_delays_for_transient_errors(self) -> None:
        class TransientError(Exception):
            pass

        operation = Mock(
            side_effect=[TransientError("one"), TransientError("two"), "ok"]
        )
        delays: list[float] = []

        with redirect_stderr(io.StringIO()):
            result = batch_ingest.call_with_retries(
                operation,
                retryable_exceptions=(TransientError,),
                attempts=3,
                label="test operation",
                sleep=delays.append,
                random_value=lambda: 0.0,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(delays, [1.0, 2.0])
        self.assertEqual(operation.call_count, 3)

    def test_existing_document_is_skipped_before_paid_calls(self) -> None:
        record = batch_ingest.load_successful_manifest(self.manifest_path)[0]
        prepared = batch_ingest.prepare_document(
            record,
            project_root=self.project_root,
        )

        class ExistingSession:
            def __enter__(self) -> ExistingSession:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def get(self, _model: object, _key: str) -> object:
                return SimpleNamespace(content_sha256=prepared.content_sha256)

        extract = Mock(side_effect=AssertionError("extraction should be skipped"))
        embed = Mock(side_effect=AssertionError("embedding should be skipped"))
        runtime = batch_ingest.Runtime(
            session_factory=ExistingSession,
            document_model=object(),
            fragment_model=object(),
            related_entity_model=object(),
            add_document=Mock(),
            add_fragment=Mock(),
            generate_fragment_code=Mock(),
            extract_fragments=extract,
            create_fragment_embeddings=embed,
            extraction_model="test-extraction",
            embedding_model="test-embedding",
            embedding_dimensions=1024,
            retryable_exceptions=(RuntimeError,),
        )

        result = batch_ingest.process_document(
            prepared,
            runtime=runtime,
            retry_attempts=1,
        )

        self.assertEqual(result.status, "skipped")
        extract.assert_not_called()
        embed.assert_not_called()

    def test_atomic_jsonl_export_includes_full_embedding(self) -> None:
        record = {
            "document_id": "doc_test_001",
            "fragment_code": "mmf_test_001",
            "embedding": [0.25, -0.5, 0.75],
            "embedding_dimensions": 3,
        }

        count = batch_ingest.write_jsonl_atomic(self.output_path, [record])
        stored = json.loads(self.output_path.read_text(encoding="utf-8"))

        self.assertEqual(count, 1)
        self.assertEqual(stored, record)
        self.assertFalse(
            self.output_path.with_suffix(".jsonl.tmp").exists()
        )

    def test_fragment_export_record_contains_database_and_provenance_data(
        self,
    ) -> None:
        document = SimpleNamespace(
            document_id="doc_test_001",
            investor_id="buffett",
            file_path="fallback.md",
            content_sha256="body-hash",
            created_at=None,
        )
        fragment = SimpleNamespace(
            fragment_id=UUID("00000000-0000-0000-0000-000000000001"),
            fragment_code="mmf_buffett_001",
            kind="principle",
            title="Durability",
            proposition="Durable advantages can protect long-term returns.",
            mechanism=[],
            conditions=[],
            failure_conditions=[],
            decision_implications=[],
            decision_stages=[],
            contextual_regimes=[],
            source_quote="Durable advantages can protect long-term returns.",
            evidence_strength="directly_stated",
            attribution_type="investor",
            attributed_to=None,
            requires_review=False,
            review_reason=None,
            embedding=[0.1] * 1024,
            embedding_model="test-embedding",
            created_at=None,
        )
        source = batch_ingest.load_successful_manifest(self.manifest_path)[0]

        exported = batch_ingest.fragment_export_record(
            document=document,
            fragment=fragment,
            entities=[],
            source_metadata=source,
            extraction_metadata={
                "extraction_model": "test-extraction",
                "extraction_version": "test-v1",
            },
        )

        self.assertEqual(exported["markdown_sha256"], source.markdown_sha256)
        self.assertEqual(exported["embedding_dimensions"], 1024)
        self.assertEqual(exported["extraction_model"], "test-extraction")
        self.assertEqual(len(exported["embedding"]), 1024)


if __name__ == "__main__":
    unittest.main()
