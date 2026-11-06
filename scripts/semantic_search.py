from pathlib import Path

from src.hospital_rag.chunking import chunk_sops
from src.hospital_rag.config import get_settings
from src.hospital_rag.embedder import SentenceTransformerEmbedder
from src.hospital_rag.fakes import InMemoryVectorStore
from src.hospital_rag.parsing import parse_pdf
from src.hospital_rag.refine import refine_chunks

settings = get_settings()
pages = parse_pdf(Path("data/raw/sample_sops.pdf"))
chunks = refine_chunks(
    chunk_sops(pages, "sops"), settings.min_chunk_size, settings.chunk_size
)

embedder = SentenceTransformerEmbedder()
store = InMemoryVectorStore()
store.add(chunks, embedder.embed([c.text for c in chunks]))
print(f"{len(chunks)} chunks embedded, dimension {embedder.dimension}\n")

questions = [
    "What are the five moments for hand hygiene?",
    "Who must double-check high-alert medicines?",
    "What is the target time for discharge after the order?",
    "What do I do after a needle stick injury?",
    "How fast should chest compressions be?",
]
for q in questions:
    print("Q:", q)
    for hit in store.search(embedder.embed([q])[0], top_k=3):
        m = hit.chunk.metadata
        print(f"   {hit.score:.3f}  {m['sop_id']}  {m['section']}")
    print()