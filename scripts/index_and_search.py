import sys
from pathlib import Path

from src.hospital_rag.chunking import chunk_sops
from src.hospital_rag.config import get_settings
from src.hospital_rag.embedder import SentenceTransformerEmbedder
from src.hospital_rag.milvus_store import MilvusVectorStore
from src.hospital_rag.parsing import parse_pdf
from src.hospital_rag.refine import refine_chunks

DOCUMENT_ID = "sops"
settings = get_settings()
embedder = SentenceTransformerEmbedder()
store = MilvusVectorStore(
    dimension=embedder.dimension,
    collection_name=settings.milvus_collection,
    uri=settings.milvus_uri,
)

if "--search-only" not in sys.argv:
    pages = parse_pdf(Path("data/raw/sample_sops.pdf"))
    chunks = refine_chunks(
        chunk_sops(pages, DOCUMENT_ID), settings.min_chunk_size, settings.chunk_size
    )
    store.delete_document(DOCUMENT_ID)
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    print(f"indexed {len(chunks)} chunks\n")

questions = [
    "What are the five moments for hand hygiene?",
    "Who must double-check high-alert medicines?",
    "What is the target time for discharge after the order?",
    "What do I do after a needle stick injury?",
    "How fast should chest compressions be?",
    "What is the visiting policy for the cafeteria?",
]
for q in questions:
    print("Q:", q)
    for hit in store.search(embedder.embed([q])[0], top_k=3):
        m = hit.chunk.metadata
        print(f"   {hit.score:.3f}  {m['sop_id']}  {m['section']}")
    print()