import uuid

import pytest
from pymilvus import MilvusClient

from src.hospital_rag.fakes import FakeEmbedder, InMemoryVectorStore
from src.hospital_rag.interfaces import Chunk

DIM = 16
URI = "http://localhost:19530"


@pytest.fixture(params=["memory", "milvus"])
def store(request):
    if request.param == "memory":
        yield InMemoryVectorStore()
        return

    try:
        MilvusClient(uri=URI).list_collections()
    except Exception as exc:
        pytest.skip(f"Milvus is not running: {exc}")

    from src.hospital_rag.milvus_store import MilvusVectorStore

    milvus = MilvusVectorStore(
        dimension=DIM, collection_name=f"test_{uuid.uuid4().hex[:8]}", uri=URI
    )
    yield milvus
    milvus.drop()


def make_chunk(chunk_id: str, document_id: str, text: str, **metadata) -> Chunk:
    return Chunk(id=chunk_id, document_id=document_id, text=text, metadata=metadata)


def test_search_returns_best_match_first(store):
    embedder = FakeEmbedder(dimension=DIM)
    texts = ["admission procedure", "discharge procedure", "fire safety"]
    chunks = [make_chunk(f"c{i}", "doc1", t) for i, t in enumerate(texts)]
    store.add(chunks, embedder.embed(texts))

    hits = store.search(embedder.embed(["discharge procedure"])[0], top_k=2)

    assert len(hits) == 2
    assert hits[0].chunk.text == "discharge procedure"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_search_on_empty_store_returns_nothing(store):
    embedder = FakeEmbedder(dimension=DIM)

    assert store.search(embedder.embed(["anything"])[0]) == []


def test_delete_document_removes_only_that_document(store):
    embedder = FakeEmbedder(dimension=DIM)
    chunks = [make_chunk("c1", "doc1", "alpha"), make_chunk("c2", "doc2", "beta")]
    store.add(chunks, embedder.embed(["alpha", "beta"]))

    store.delete_document("doc1")
    hits = store.search(embedder.embed(["alpha"])[0], top_k=5)

    assert [h.chunk.document_id for h in hits] == ["doc2"]


def test_add_rejects_mismatched_lengths(store):
    with pytest.raises(ValueError):
        store.add([make_chunk("c1", "doc1", "alpha")], [])


def test_metadata_round_trips(store):
    embedder = FakeEmbedder(dimension=DIM)
    chunk = make_chunk("c1", "doc1", "alpha", sop_id="SOP-X", page=3)
    store.add([chunk], embedder.embed(["alpha"]))

    hit = store.search(embedder.embed(["alpha"])[0], top_k=1)[0]

    assert hit.chunk.metadata == {"sop_id": "SOP-X", "page": 3}