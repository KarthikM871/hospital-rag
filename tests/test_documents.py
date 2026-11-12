import psycopg
import pytest


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


def test_list_documents_returns_newest_first(repo):
    first = repo.create("a.pdf", "hash-1")
    second = repo.create("b.pdf", "hash-2")

    assert [d.id for d in repo.list_documents()] == [second.id, first.id]


def test_ready_ids_contains_only_ready_documents(repo):
    ready = repo.create("a.pdf", "hash-1")
    repo.mark_ready(ready.id, 3)
    repo.create("b.pdf", "hash-2")
    failed = repo.create("c.pdf", "hash-3")
    repo.mark_failed(failed.id, "boom")

    assert repo.ready_ids() == {ready.id}