from typing import Protocol

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class SearchHit(BaseModel):
    chunk: Chunk
    score: float


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchHit]: ...

    def delete_document(self, document_id: str) -> None: ...