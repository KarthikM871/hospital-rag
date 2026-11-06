import sys
import time

from src.hospital_rag.config import get_settings
from src.hospital_rag.embedder import SentenceTransformerEmbedder
from src.hospital_rag.llm import OllamaLLMClient
from src.hospital_rag.milvus_store import MilvusVectorStore
from src.hospital_rag.rag import answer_question

settings = get_settings()
embedder = SentenceTransformerEmbedder()
store = MilvusVectorStore(
    dimension=embedder.dimension,
    collection_name=settings.milvus_collection,
    uri=settings.milvus_uri,
)
llm = OllamaLLMClient(model=settings.ollama_model, base_url=settings.ollama_url)

questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
    "How long should alcohol hand rub be applied?",
    "How fast should chest compressions be?",
    "What are the ICU visiting hours?",
]
for question in questions:
    start = time.time()
    result = answer_question(
        question, embedder, store, llm,
        top_k=settings.top_k, min_score=settings.min_score,
    )
    print(f"Q: {question}")
    print(f"A: {result.text}")
    print(f"refused={result.refused}  time={time.time() - start:.1f}s")
    for number, hit in enumerate(result.sources, start=1):
        m = hit.chunk.metadata
        print(
            f"   [{number}] {hit.score:.3f} {m['sop_id']} | {m['section']} | "
            f"page {m['page']} | version {m['version']}"
        )
    print()