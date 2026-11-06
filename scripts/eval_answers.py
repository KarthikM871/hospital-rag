from src.hospital_rag.config import get_settings
from src.hospital_rag.embedder import SentenceTransformerEmbedder
from src.hospital_rag.llm import OllamaLLMClient
from src.hospital_rag.milvus_store import MilvusVectorStore
from src.hospital_rag.rag import answer_question

ANSWERABLE = [
    "What if a patient wants to leave against medical advice?",
    "What is checked on the crash cart?",
    "How long should alcohol hand rub be applied?",
]
UNANSWERABLE = [
    "What is the visiting policy for the cafeteria?",
    "How do I apply for annual leave?",
    "What is the parking fee for visitors?",
    "Who won the cricket match yesterday?",
    "What are the ICU visiting hours?",
]

settings = get_settings()
embedder = SentenceTransformerEmbedder()
store = MilvusVectorStore(
    dimension=embedder.dimension,
    collection_name=settings.milvus_collection,
    uri=settings.milvus_uri,
)
llm = OllamaLLMClient(model=settings.ollama_model, base_url=settings.ollama_url)


def ask(question: str):
    return answer_question(
        question, embedder, store, llm,
        top_k=settings.top_k, min_score=settings.min_score,
    )


false_refusals = 0
print("ANSWERABLE (should NOT be refused)")
for question in ANSWERABLE:
    result = ask(question)
    false_refusals += result.refused
    print(f"refused={result.refused}  {question}")
    print(f"   A: {result.text[:200]}\n")

unsafe = 0
print("UNANSWERABLE (should be refused)")
for question in UNANSWERABLE:
    result = ask(question)
    unsafe += not result.refused
    print(f"refused={result.refused}  {question}")
    print(f"   A: {result.text[:200]}\n")

print(f"false refusals: {false_refusals}/{len(ANSWERABLE)}")
print(f"unsafe answers: {unsafe}/{len(UNANSWERABLE)}")