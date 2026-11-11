import uuid

import psycopg
import pytest

from src.hospital_rag.config import get_settings
from src.hospital_rag.documents import DocumentRepository


@pytest.fixture
def repo():
    settings = get_settings()
    if not settings.postgres_password:
        pytest.skip("RAG_POSTGRES_PASSWORD is not set in .env")
    repository = DocumentRepository(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        schema=f"test_{uuid.uuid4().hex[:8]}",
    )
    try:
        repository.ensure_schema()
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres is not available: {exc}")
    yield repository
    repository.drop_schema()


def test_new_document_starts_as_processing(repo):
    doc = repo.create("sops.pdf", "hash-1")

    assert doc.status == "processing"
    assert doc.chunk_count == 0
    assert doc.error is None
    assert doc.filename == "sops.pdf"


def test_duplicate_hash_is_rejected_by_database(repo):
    repo.create("a.pdf", "same-hash")

    with pytest.raises(psycopg.errors.UniqueViolation):
        repo.create("b.pdf", "same-hash")


def test_get_by_hash_finds_existing_document(repo):
    created = repo.create("a.pdf", "hash-1")

    found = repo.get_by_hash("hash-1")

    assert found is not None
    assert found.id == created.id


def test_get_by_hash_returns_none_when_missing(repo):
    assert repo.get_by_hash("nope") is None


def test_mark_ready_sets_status_and_chunk_count(repo):
    doc = repo.create("a.pdf", "hash-1")

    updated = repo.mark_ready(doc.id, chunk_count=30)

    assert updated.status == "ready"
    assert updated.chunk_count == 30
    assert updated.updated_at >= doc.updated_at


def test_mark_failed_stores_error_message(repo):
    doc = repo.create("a.pdf", "hash-1")

    updated = repo.mark_failed(doc.id, "could not parse PDF")

    assert updated.status == "failed"
    assert updated.error == "could not parse PDF"