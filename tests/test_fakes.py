import pytest

from src.hospital_rag.fakes import FakeEmbedder, InMemoryVectorStore
from src.hospital_rag.interfaces import Chunk


def make_chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, document_id=document_id, text=text)


def test_embedder_is_deterministic():
    embedder = FakeEmbedder(dimension=8)

    first = embedder.embed(["hello"])[0]
    second = embedder.embed(["hello"])[0]

    assert first == second
    assert len(first) == 8


def test_different_texts_give_different_vectors():
    embedder = FakeEmbedder(dimension=8)

    assert embedder.embed(["hello"])[0] != embedder.embed(["world"])[0]


def test_search_returns_best_match_first():
    embedder = FakeEmbedder(dimension=16)
    store = InMemoryVectorStore()
    texts = ["admission procedure", "discharge procedure", "fire safety"]
    chunks = [make_chunk(str(i), "doc1", t) for i, t in enumerate(texts)]
    store.add(chunks, embedder.embed(texts))

    query = embedder.embed(["discharge procedure"])[0]
    hits = store.search(query, top_k=2)

    assert len(hits) == 2
    assert hits[0].chunk.text == "discharge procedure"
    assert hits[0].score == pytest.approx(1.0)


def test_delete_document_removes_only_that_document():
    embedder = FakeEmbedder(dimension=16)
    store = InMemoryVectorStore()
    chunks = [
        make_chunk("1", "doc1", "alpha"),
        make_chunk("2", "doc2", "beta"),
    ]
    store.add(chunks, embedder.embed(["alpha", "beta"]))

    store.delete_document("doc1")
    hits = store.search(embedder.embed(["alpha"])[0], top_k=5)

    assert [h.chunk.document_id for h in hits] == ["doc2"]


def test_add_rejects_mismatched_lengths():
    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.add([make_chunk("1", "doc1", "alpha")], [])