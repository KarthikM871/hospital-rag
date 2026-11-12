import secrets
import uuid
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from src.hospital_rag.config import Settings, get_settings
from src.hospital_rag.documents import Document, DocumentRepository
from src.hospital_rag.ingestion import IngestionError, IngestionResult, IngestionService
from src.hospital_rag.interfaces import Embedder, LLMClient, VectorStore
from src.hospital_rag.rag import answer_question
from src.hospital_rag.ready_store import ReadyOnlyStore

READ_BLOCK = 1024 * 1024

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_repo(settings: SettingsDep) -> DocumentRepository:
    return DocumentRepository(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


@lru_cache
def get_embedder() -> Embedder:
    from src.hospital_rag.embedder import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder()


@lru_cache
def get_store() -> VectorStore:
    from src.hospital_rag.milvus_store import MilvusVectorStore

    settings = get_settings()
    return MilvusVectorStore(
        dimension=get_embedder().dimension,
        collection_name=settings.milvus_collection,
        uri=settings.milvus_uri,
    )


@lru_cache
def get_llm() -> LLMClient:
    from src.hospital_rag.llm import OllamaLLMClient

    settings = get_settings()
    return OllamaLLMClient(model=settings.ollama_model, base_url=settings.ollama_url)


RepoDep = Annotated[DocumentRepository, Depends(get_repo)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
StoreDep = Annotated[VectorStore, Depends(get_store)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]


def get_ingestion_service(
    repo: RepoDep, embedder: EmbedderDep, store: StoreDep, settings: SettingsDep
) -> IngestionService:
    return IngestionService(
        repo, embedder, store, settings.min_chunk_size, settings.chunk_size
    )


ServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]


def require_admin(
    settings: SettingsDep, x_api_key: Annotated[str | None, Header()] = None
) -> None:
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="admin API key is not configured")
    supplied = (x_api_key or "").encode()
    if not secrets.compare_digest(supplied, settings.admin_api_key.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class Health(BaseModel):
    status: str
    app_name: str
    environment: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class SourceOut(BaseModel):
    number: int
    document_id: str
    sop_id: str
    section: str
    page: int | None
    version: str | None
    score: float


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceOut]


def _save_upload(file: UploadFile, dest: Path, limit: int) -> None:
    size = 0
    with dest.open("wb") as out:
        while block := file.file.read(READ_BLOCK):
            if size == 0 and not block.startswith(b"%PDF-"):
                raise HTTPException(status_code=415, detail="file is not a PDF")
            size += len(block)
            if size > limit:
                raise HTTPException(status_code=413, detail="file is too large")
            out.write(block)
    if size == 0:
        raise HTTPException(status_code=400, detail="file is empty")


def create_app() -> FastAPI:
    app = FastAPI(title="Hospital RAG")

    @app.get("/health")
    def health(settings: SettingsDep) -> Health:
        return Health(
            status="ok",
            app_name=settings.app_name,
            environment=settings.environment,
        )

    @app.post(
        "/documents", status_code=201, dependencies=[Depends(require_admin)]
    )
    def upload_document(
        file: UploadFile,
        response: Response,
        service: ServiceDep,
        settings: SettingsDep,
    ) -> IngestionResult:
        name = PureWindowsPath(file.filename or "").name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="only .pdf files are accepted")
        name = name[:255]

        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        tmp = upload_dir / f"tmp-{uuid.uuid4().hex}.pdf"
        try:
            _save_upload(file, tmp, settings.max_upload_mb * 1024 * 1024)
            result = service.ingest(tmp, filename=name)
            doc = result.document
            if result.duplicate:
                response.status_code = 200
            elif doc.status == "failed":
                raise HTTPException(
                    status_code=422, detail=f"ingestion failed: {doc.error}"
                )
            else:
                tmp.replace(upload_dir / f"{doc.id}.pdf")
            return result
        except IngestionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except psycopg.errors.UniqueViolation as exc:
            raise HTTPException(
                status_code=409, detail="this file is being processed by another request"
            ) from exc
        finally:
            tmp.unlink(missing_ok=True)

    @app.get("/documents", dependencies=[Depends(require_admin)])
    def list_documents(repo: RepoDep) -> list[Document]:
        return repo.list_documents()

    @app.post("/ask")
    def ask(
        body: AskRequest,
        embedder: EmbedderDep,
        store: StoreDep,
        llm: LLMDep,
        repo: RepoDep,
        settings: SettingsDep,
    ) -> AskResponse:
        result = answer_question(
            body.question,
            embedder,
            ReadyOnlyStore(store, repo),
            llm,
            top_k=settings.top_k,
            min_score=settings.min_score,
        )
        sources = [
            SourceOut(
                number=number,
                document_id=hit.chunk.document_id,
                sop_id=hit.chunk.metadata.get("sop_id", ""),
                section=hit.chunk.metadata.get("section", ""),
                page=hit.chunk.metadata.get("page"),
                version=hit.chunk.metadata.get("version"),
                score=hit.score,
            )
            for number, hit in enumerate(result.sources, start=1)
        ]
        return AskResponse(answer=result.text, refused=result.refused, sources=sources)

    return app


app = create_app()