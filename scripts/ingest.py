import sys
from pathlib import Path

from src.hospital_rag.config import get_settings
from src.hospital_rag.documents import DocumentRepository
from src.hospital_rag.embedder import SentenceTransformerEmbedder
from src.hospital_rag.ingestion import IngestionService
from src.hospital_rag.milvus_store import MilvusVectorStore

settings = get_settings()
args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = Path(args[0]) if args else Path("data/raw/sample_sops.pdf")

repo = DocumentRepository(
    host=settings.postgres_host,
    port=settings.postgres_port,
    dbname=settings.postgres_db,
    user=settings.postgres_user,
    password=settings.postgres_password,
)
repo.ensure_schema()

embedder = SentenceTransformerEmbedder()
store = MilvusVectorStore(
    dimension=embedder.dimension,
    collection_name=settings.milvus_collection,
    uri=settings.milvus_uri,
)

if "--clear-legacy" in sys.argv:
    store.delete_document("sops")
    print("removed legacy chunks with document_id 'sops'")

service = IngestionService(
    repo, embedder, store, settings.min_chunk_size, settings.chunk_size
)
result = service.ingest(path)
doc = result.document
print(f"file:      {doc.filename}")
print(f"status:    {doc.status}")
print(f"chunks:    {doc.chunk_count}")
print(f"duplicate: {result.duplicate}")
print(f"id:        {doc.id}")
if doc.error:
    print(f"error:     {doc.error}")