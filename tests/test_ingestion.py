import pytest

from src.hospital_rag import ingestion
from src.hospital_rag.fakes import FakeEmbedder, InMemoryVectorStore
from src.hospital_rag.ingestion import IngestionService
from src.hospital_rag.parsing import ParsedPage

SOP_TEXT = "SOP-TST-001: Test\nDepartment: Lab\n1. Purpose\n" + "Purpose text. " * 20


class HalfWrittenStore(InMemoryVectorStore):
    def add(self, chunks, vectors):
        super().add(chunks, vectors)
        raise RuntimeError("boom")


@pytest.fixture
def pdf(tmp_path, monkeypatch):
    path = tmp_path / "sops.pdf"
    path.write_bytes(b"fake pdf bytes")
    monkeypatch.setattr(
        ingestion, "parse_pdf", lambda _path: [ParsedPage(page_number=1, text=SOP_TEXT)]
    )
    return path


def make_service(repo, store=None):
    return IngestionService(
        repo=repo,
        embedder=FakeEmbedder(),
        store=store or InMemoryVectorStore(),
        min_chars=50,
        max_chars=800,
    )


def count_chunks(store) -> int:
    return len(store.search(FakeEmbedder().embed(["x"])[0], top_k=100))


def test_successful_ingest_marks_ready_and_stores_chunks(repo, pdf):
    store = InMemoryVectorStore()

    result = make_service(repo, store).ingest(pdf)

    assert result.document.status == "ready"
    assert result.document.chunk_count == count_chunks(store) >= 1
    assert not result.duplicate


def test_same_file_twice_is_skipped(repo, pdf):
    store = InMemoryVectorStore()
    service = make_service(repo, store)
    first = service.ingest(pdf)
    chunks_after_first = count_chunks(store)

    second = service.ingest(pdf)

    assert second.duplicate
    assert second.document.id == first.document.id
    assert count_chunks(store) == chunks_after_first


def test_failure_marks_failed_and_removes_partial_chunks(repo, pdf):
    store = HalfWrittenStore()

    result = make_service(repo, store).ingest(pdf)

    assert result.document.status == "failed"
    assert "boom" in result.document.error
    assert count_chunks(store) == 0


def test_failed_document_can_be_retried(repo, pdf):
    failed = make_service(repo, HalfWrittenStore()).ingest(pdf)
    store = InMemoryVectorStore()

    retry = make_service(repo, store).ingest(pdf)

    assert retry.document.id == failed.document.id
    assert retry.document.status == "ready"
    assert count_chunks(store) >= 1


def test_pdf_without_sops_is_marked_failed(repo, pdf, monkeypatch):
    monkeypatch.setattr(
        ingestion, "parse_pdf", lambda _path: [ParsedPage(page_number=1, text="Just prose.")]
    )

    result = make_service(repo).ingest(pdf)

    assert result.document.status == "failed"
    assert "no SOP sections" in result.document.error