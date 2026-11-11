import hashlib
from pathlib import Path

from pydantic import BaseModel

from src.hospital_rag.chunking import chunk_sops
from src.hospital_rag.documents import Document, DocumentRepository
from src.hospital_rag.interfaces import Embedder, VectorStore
from src.hospital_rag.parsing import parse_pdf
from src.hospital_rag.refine import refine_chunks


class IngestionError(Exception):
    pass


class IngestionResult(BaseModel):
    document: Document
    duplicate: bool = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionService:
    def __init__(
        self,
        repo: DocumentRepository,
        embedder: Embedder,
        store: VectorStore,
        min_chars: int,
        max_chars: int,
    ) -> None:
        self._repo = repo
        self._embedder = embedder
        self._store = store
        self._min_chars = min_chars
        self._max_chars = max_chars

    def ingest(self, path: Path) -> IngestionResult:
        file_hash = file_sha256(path)
        existing = self._repo.get_by_hash(file_hash)

        if existing is not None and existing.status == "ready":
            return IngestionResult(document=existing, duplicate=True)
        if existing is not None and existing.status == "processing":
            raise IngestionError("this file is already being processed")

        if existing is None:
            document = self._repo.create(path.name, file_hash)
        else:
            document = self._repo.mark_processing(existing.id)

        try:
            pages = parse_pdf(path)
            chunks = refine_chunks(
                chunk_sops(pages, document.id), self._min_chars, self._max_chars
            )
            if not chunks:
                raise ValueError("no SOP sections found in this PDF")
            vectors = self._embedder.embed([c.text for c in chunks])
            self._store.add(chunks, vectors)
        except Exception as exc:
            try:
                self._store.delete_document(document.id)
            finally:
                failed = self._repo.mark_failed(document.id, str(exc))
            return IngestionResult(document=failed)

        return IngestionResult(document=self._repo.mark_ready(document.id, len(chunks)))