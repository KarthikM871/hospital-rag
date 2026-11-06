import hashlib

import numpy as np

from src.hospital_rag.interfaces import Chunk, SearchHit


class FakeEmbedder:
    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.normal(size=self.dimension).tolist()


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchHit]:
        if not self._chunks:
            return []
        matrix = np.array(self._vectors)
        query = np.array(query_vector)
        scores = (matrix @ query) / (
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        )
        best = np.argsort(scores)[::-1][:top_k]
        return [
            SearchHit(chunk=self._chunks[i], score=float(scores[i])) for i in best
        ]

    def delete_document(self, document_id: str) -> None:
        keep = [i for i, c in enumerate(self._chunks) if c.document_id != document_id]
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = [self._vectors[i] for i in keep]