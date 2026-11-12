from src.hospital_rag.documents import DocumentRepository
from src.hospital_rag.interfaces import Chunk, SearchHit, VectorStore


class ReadyOnlyStore:
    def __init__(
        self, store: VectorStore, repo: DocumentRepository, overfetch: int = 3
    ) -> None:
        self._store = store
        self._repo = repo
        self._overfetch = overfetch

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._store.add(chunks, vectors)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchHit]:
        hits = self._store.search(query_vector, top_k=top_k * self._overfetch)
        ready = self._repo.ready_ids()
        return [h for h in hits if h.chunk.document_id in ready][:top_k]

    def delete_document(self, document_id: str) -> None:
        self._store.delete_document(document_id)